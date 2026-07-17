#!/usr/bin/env python3
"""
train.py -- contrastive fine-tuning of the embedder (step 3 of 3).

    generate_queries.py  ->  build_dataset.py  ->  train.py

Fine-tunes Qwen3-Embedding-0.6B on the mined (query, positive, hard-negatives) triples
with MultipleNegativesRankingLoss (MNRL, aka InfoNCE): the query is pulled toward its
gold document and pushed away from the hard negatives -- the documents the BASE model
wrongly ranked above the gold. That is the mechanism that closes the register gap.

ASYMMETRY (do not break this): at inference `encode_queries` prepends
"Instruct: {task}\\nQuery: " while documents get no prefix. Training replicates that
exactly -- anchors carry the instruction prefix, positives/negatives are raw document
text. Train/inference mismatch here silently destroys retrieval quality.

MNRL uses BOTH the explicit hard negatives AND in-batch negatives, so effective
negatives per anchor = (batch_size * (1 + n_neg)) - 1. Bigger batches => stronger
signal; that is why this wants an A100, not a 12 GB card.

  TARGET: Anvil A100 (bf16, gradient checkpointing). ~66k triples, 1 epoch.
          Drew's 3060s can run it only with tiny batches / few negatives.

AFTER TRAINING -- the model swap is one env var, because config.py reads
EMBED_MODEL from DH_EMBED_MODEL:

    export DH_EMBED_MODEL=/path/to/models/qwen3-dh-ft
    python 04_generate_embeddings.py --devices cuda:0   # re-embed (vectors invalidated)
    python 05_build_index.py                            # rebuild FAISS + BM25
    python 10_eval_retrieval.py --qrels .../synthetic_queries.jsonl.eval.jsonl

Fine-tuning invalidates every existing vector: the old index CANNOT be reused.

Usage:
  python train.py --dry-run                       # build data, print stats, no GPU
  python train.py --triples data/finetune/train_triples.jsonl \\
                  --out models/qwen3-dh-ft --epochs 1 --batch-size 8 --n-negatives 4
  python train.py --resume                        # continue from last checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config
from discovery_hub.embedding import query_prompt_prefix


def load_doc_texts(docs_path: Path) -> dict[str, str]:
    """doc_id -> embedding_text (the exact string stage 04 embeds)."""
    out: dict[str, str] = {}
    with docs_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("embedding_text") or \
                f"{d.get('title', '')} {d.get('abstract', '')}".strip()
            if d.get("doc_id") and t:
                out[d["doc_id"]] = t
    return out


def build_records(triples_path: Path, doc_texts: dict[str, str], n_neg_use: int,
                  query_prefix: str, max_train: int = 0) -> tuple[dict, dict]:
    """
    Build MNRL columns: anchor, positive, negative_1..negative_n.

    ST's MultipleNegativesRankingLoss treats column 0 as the anchor, column 1 as the
    positive, and EVERY remaining column as a negative. Anchors get the instruction
    prefix (matching encode_queries); documents do not (matching encode_documents).

    Rows are skipped when the positive text is missing, or when fewer than n_neg_use
    negatives resolve to text -- ragged rows would misalign the columns.
    """
    cols: dict[str, list[str]] = {"anchor": [], "positive": []}
    for i in range(n_neg_use):
        cols[f"negative_{i + 1}"] = []
    stats = {"read": 0, "kept": 0, "no_positive_text": 0, "too_few_negatives": 0}

    with triples_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            r = json.loads(line)
            pos_text = doc_texts.get(r["positive"])
            if not pos_text:
                stats["no_positive_text"] += 1
                continue
            neg_texts = [doc_texts[n] for n in r["negatives"] if n in doc_texts]
            # keep the hardest ones (build_dataset writes them hardest-first)
            neg_texts = neg_texts[:n_neg_use]
            if len(neg_texts) < n_neg_use:
                stats["too_few_negatives"] += 1
                continue

            cols["anchor"].append(query_prefix + r["query"])   # instructed query
            cols["positive"].append(pos_text)                  # raw document text
            for i, nt in enumerate(neg_texts):
                cols[f"negative_{i + 1}"].append(nt)
            stats["kept"] += 1
            if max_train and stats["kept"] >= max_train:
                break
    return cols, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triples", default=None,
                    help="train_triples.jsonl from build_dataset.py")
    ap.add_argument("--docs", default=None, help="normalized docs.jsonl (for text)")
    ap.add_argument("--out", default=None, help="output model dir")
    ap.add_argument("--base-model", default=None,
                    help="base encoder (default: config.EMBED_MODEL). NOTE: if "
                         "DH_EMBED_MODEL already points at a fine-tuned dir you would "
                         "be training on top of it -- pass this explicitly to be sure.")
    ap.add_argument("--n-negatives", type=int, default=4,
                    help="explicit hard negatives per anchor USED IN TRAINING "
                         "(<= what build_dataset mined). Each one multiplies memory.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="anchors per device; each carries 1 positive + n negatives")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="truncate long patent abstracts (bounds GPU memory)")
    ap.add_argument("--max-train", type=int, default=0, help="0 = all triples")
    ap.add_argument("--scale", type=float, default=20.0,
                    help="MNRL temperature scale (1/T); 20 is the ST default")
    ap.add_argument("--loss", choices=["mnrl", "cached-mnrl"], default="mnrl",
                    help="mnrl = standard (negatives limited by GPU memory); "
                         "cached-mnrl = GradCache, lets --batch-size (and thus "
                         "in-batch negatives) scale far beyond memory")
    ap.add_argument("--mini-batch-size", type=int, default=32,
                    help="cached-mnrl only: per-forward chunk that sets VRAM "
                         "(smaller = less memory, more passes). Ignored for mnrl.")
    ap.add_argument("--use-lora", action="store_true",
                    help="LoRA fine-tuning (trains ~0.1%% of params). REQUIRED for 4B/8B "
                         "on a single 80GB GPU — full fine-tuning won't fit. Adapter is "
                         "MERGED into the base weights on save, so the output loads as a "
                         "normal SentenceTransformer (stages 04/05/07 need no change).")
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank (higher = more capacity, more memory)")
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling; commonly 2x rank)")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA: load the base in 4-bit (bitsandbytes) to cut VRAM further. "
                         "Use for 8B if 80GB is tight, or to train 8B on a 40GB card.")
    ap.add_argument("--no-bf16", action="store_true", help="disable bf16 (use fp32)")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in --out")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the dataset, print stats, and exit (no torch/GPU)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    root = config.DATA_ROOT
    triples_path = Path(args.triples) if args.triples else \
        root / "finetune" / "train_triples.jsonl"
    docs_path = Path(args.docs) if args.docs else config.NORM_DIR / "docs.jsonl"
    out_dir = Path(args.out) if args.out else root / "models" / "qwen3-dh-ft"
    if not triples_path.exists():
        print(f"ERROR: {triples_path} not found -- run build_dataset.py first.",
              file=sys.stderr)
        return 2
    if not docs_path.exists():
        print(f"ERROR: {docs_path} not found -- run 02_parse_normalize.py first.",
              file=sys.stderr)
        return 2

    prefix = query_prompt_prefix(config.QUERY_INSTRUCTION)
    print(f"train: triples={triples_path}")
    print(f"  query prefix (must match encode_queries): {prefix!r}")
    print("  loading document texts ...")
    doc_texts = load_doc_texts(docs_path)
    print(f"  {len(doc_texts)} docs")

    cols, stats = build_records(triples_path, doc_texts, args.n_negatives,
                                prefix, args.max_train)
    print(f"  triples read={stats['read']} kept={stats['kept']} "
          f"(dropped: no_positive_text={stats['no_positive_text']}, "
          f"too_few_negatives={stats['too_few_negatives']})")
    if not cols["anchor"]:
        print("ERROR: no usable training rows.", file=sys.stderr)
        return 1
    print(f"  columns: {list(cols)}")
    print(f"  example anchor  : {cols['anchor'][0][:110]!r}")
    print(f"  example positive: {cols['positive'][0][:110]!r}")
    eff = args.batch_size * (1 + args.n_negatives) - 1
    print(f"  effective negatives per anchor (in-batch + explicit): ~{eff}")

    if args.dry_run:
        print("\n--dry-run: dataset built, not training.")
        return 0

    # ---- heavy deps imported only for the real run ----
    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformer,
                                           SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments)
        from sentence_transformers.losses import (
            MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss)
    except ImportError as e:
        print(f"ERROR: missing training deps ({e}).\n"
              "  pip install 'sentence-transformers>=3.0' datasets accelerate",
              file=sys.stderr)
        return 2

    base = args.base_model or config.EMBED_MODEL
    print(f"\n  base model: {base}")
    if not args.base_model and base != "Qwen/Qwen3-Embedding-0.6B":
        print("  [warn] DH_EMBED_MODEL is overridden; you may be fine-tuning a model "
              "that was already fine-tuned. Pass --base-model to be explicit.")

    # ---- optional 4-bit base (QLoRA) — must be set at model load ----
    st_model_kwargs = {}
    if args.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("ERROR: --load-in-4bit needs bitsandbytes: pip install bitsandbytes",
                  file=sys.stderr)
            return 2
        from transformers import BitsAndBytesConfig
        st_model_kwargs["model_kwargs"] = {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            "device_map": "auto",
        }
        print("  QLoRA: base loaded in 4-bit (nf4)")

    model = SentenceTransformer(base, **st_model_kwargs)
    model.max_seq_length = args.max_seq_length

    # ---- LoRA: train tiny adapters instead of full weights (needed for 4B/8B) ----
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            print("ERROR: --use-lora needs peft: pip install peft", file=sys.stderr)
            return 2
        transformer = model[0].auto_model            # the underlying HF model inside ST
        if args.load_in_4bit:
            transformer = prepare_model_for_kbit_training(transformer)
        lora = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="FEATURE_EXTRACTION",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        peft_model = get_peft_model(transformer, lora)   # keep an explicit handle for the merge
        model[0].auto_model = peft_model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA: r={args.lora_r} alpha={args.lora_alpha} -> "
              f"training {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")

    dataset = Dataset.from_dict(cols)

    # CachedMNRL (GradCache) decouples the number of in-batch negatives from GPU
    # memory: it does a two-pass forward, caching embeddings in mini-batches, so a
    # huge --batch-size (=> ~batch*(1+n_neg) negatives per anchor) fits at roughly
    # constant VRAM. `mini_batch_size` is the real per-forward chunk that sets memory.
    if args.loss == "cached-mnrl":
        loss = CachedMultipleNegativesRankingLoss(
            model, scale=args.scale, mini_batch_size=args.mini_batch_size)
        print(f"  loss: CachedMNRL (GradCache) mini_batch_size={args.mini_batch_size}")
    else:
        loss = MultipleNegativesRankingLoss(model, scale=args.scale)
        print("  loss: MultipleNegativesRankingLoss")

    bf16 = (not args.no_bf16) and torch.cuda.is_available() and \
        torch.cuda.is_bf16_supported()
    out_dir.mkdir(parents=True, exist_ok=True)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=not args.no_grad_checkpointing,
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],                       # no wandb/tensorboard side effects
        dataloader_drop_last=True,          # MNRL wants uniform batches
    )
    print(f"  bf16={bf16}  grad_checkpointing={not args.no_grad_checkpointing}  "
          f"batch={args.batch_size}x{args.grad_accum} accum  lr={args.lr}")

    trainer = SentenceTransformerTrainer(model=model, args=targs,
                                         train_dataset=dataset, loss=loss)
    trainer.train(resume_from_checkpoint=args.resume or None)

    final = out_dir / "final"
    # LoRA: merge the adapter back into the base weights so the saved model is a
    # standard SentenceTransformer — stages 04/05/07 load it with no LoRA/PEFT code.
    # (4-bit QLoRA can't merge in-place; save the adapter instead and note it.)
    if args.use_lora:
        if args.load_in_4bit:
            print("  [note] 4-bit base can't merge; saving LoRA ADAPTER only. To serve, "
                  "load base + adapter, or re-merge in fp16. See dh_train_meta.json.")
            model[0].auto_model.save_pretrained(str(final / "lora_adapter"))
        else:
            print("  merging LoRA adapter into base weights ...")
            # The trainer/accelerate can reassign or wrap model[0].auto_model during
            # training, leaving it as the bare Qwen3Model (no .merge_and_unload) -- that
            # was the crash. Prefer the live attribute if it still exposes merge; else
            # fall back to the handle we kept at setup.
            to_merge = model[0].auto_model
            if not hasattr(to_merge, "merge_and_unload"):
                to_merge = peft_model
            model[0].auto_model = to_merge.merge_and_unload()
    model.save(str(final))
    meta = {"base_model": base, "triples": str(triples_path),
            "loss": args.loss, "mini_batch_size": args.mini_batch_size,
            "use_lora": args.use_lora, "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha, "load_in_4bit": args.load_in_4bit,
            "train_rows": stats["kept"], "n_negatives": args.n_negatives,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "lr": args.lr, "scale": args.scale,
            "max_seq_length": args.max_seq_length, "seed": args.seed,
            "query_prefix": prefix}
    (final / "dh_train_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nDONE. fine-tuned model -> {final}")
    print("Next (embeddings are now INVALID -- full re-embed + rebuild required):")
    print(f"  export DH_EMBED_MODEL={final}")
    print("  python 04_generate_embeddings.py --devices cuda:0,cuda:1")
    print("  python 05_build_index.py")
    print("  python 10_eval_retrieval.py --qrels "
          f"{root}/finetune/synthetic_queries.jsonl.eval.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
