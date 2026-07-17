#!/usr/bin/env python3
"""

! mock run:
python3 04_generate_embeddings.py --mock --devices a,b

! actual run:
nohup python3 04_generate_embeddings.py --devices cuda:0,cuda:1 --batch-size 64 > embed.log 2>&1 &
tail -f embed.log


04_generate_embeddings.py  --  LAYER 2, step 4 of 6.  [dual-GPU + checkpointed]

Embeds every DiscoveryDoc's embedding_text into a dense vector matrix. Drop-in
replacement for the single-pass version: SAME outputs (doc_vectors.npy + aligned
doc_ids.json), plus two robustness features for long runs on modest GPUs:

  * CHECKPOINTED -- the corpus is split into shards; each shard is embedded and
    written atomically (temp + rename). If the run dies (OOM, power, SSH drop),
    just re-run the same command: finished shards are skipped and it resumes.
  * MULTI-GPU -- one worker process per device in --devices, each pinned to its
    own GPU, draining a shared queue of pending shards. Two RTX 3060s ~halve the
    wall-clock vs one. Works for 1 device too (still checkpointed).

Outputs (unchanged, so stage 05 consumes them as-is):
  data/embeddings/doc_vectors.npy   float32 [N, dim], L2-normalized, doc-order
  data/embeddings/doc_ids.json      ordered doc_id list aligned to the rows

Usage:
  python 04_generate_embeddings.py --mock                         # plumbing test, no GPU
  python 04_generate_embeddings.py --devices cuda:0,cuda:1        # both GPUs, real model
  python 04_generate_embeddings.py --devices cuda:0 --batch-size 32   # one GPU, smaller batch
  # crashed mid-run? re-run the exact command -- it resumes from the last shard.

Note: the final concat loads all shards into RAM (~2.5 GB for 600k x 1024 f32).
At Anvil/40M scale, memmap the output instead; fine for the Drew MVP.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reduce fragmentation OOMs on 12 GB cards (the "reserved but unallocated" case).
# Set before torch/CUDA initializes; inherited by spawned workers.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder
from discovery_hub.schema import read_docs


def _shard_path(shards_dir: Path, i: int) -> Path:
    return shards_dir / f"shard_{i:05d}.npy"


def _shard_manifest_path(shards_dir: Path, i: int) -> Path:
    return shards_dir / f"shard_{i:05d}.manifest.json"


def _shard_indices(n: int, shard_size: int) -> list[int]:
    return list(range((n + shard_size - 1) // shard_size))


# --------------------------------------------------------------------------- #
# Shard provenance -- the fix for a silent, shipped corruption.
#
# WHAT HAPPENED. This stage used to resume with:
#
#     pending = [i for i in shards if not _shard_path(shards_dir, i).exists()]
#
# A shard was skipped because its FILE EXISTED. Nothing recorded which model, which text,
# or which config wrote it. Re-running with a different checkpoint therefore reused every
# shard left on disk from the previous run and vstacked old and new into one matrix. The
# row count still equalled `n`, so the only integrity check below (`vectors.shape[0] != n`)
# passed and the artifact looked complete.
#
# It shipped: 60,000 rows (shards 0-2 = 9.9% of the corpus, 40.7% of all clinical trials)
# of data/embeddings/doc_vectors.npy were embeddings of documents OTHER than the ones they
# were indexed under. Serving is dense-only, so those records were unreachable for ANY
# query. Confirmed by re-embedding every row on an H200 and comparing: cos ~0.00 for rows
# 0-59,999, ~1.00 for the rest.
#
# WHY THE EXISTING GUARDS MISSED IT, precisely. dh2.manifest.doc_order_checksum fingerprints
# row ORDER and dh2.validate.assert_serving_compat checks counts and dims. This bug leaves
# order, count AND dim perfectly intact -- it swaps only row CONTENT. `compatible()` returns
# True; `assert_serving_compat` passes. Those guards are right about what they guard; they
# simply never bound a shard to the MODEL that wrote it. That binding is what follows, and
# it is built from those same primitives rather than a competing scheme.
#
# The safe default is re-embedding: a shard with no manifest, an unreadable manifest, or a
# manifest that disagrees with this run is re-embedded, never reused. Existence is not
# evidence.
# --------------------------------------------------------------------------- #
def _shard_stamp(model: str, max_seq_length: int, doc_ids: list[str]) -> dict:
    """The identity of a shard's work: which model, which config, which documents.

    Deliberately NOT embedding_dim. The dim is a function of the model, and only a worker
    that has loaded the model knows it -- the parent process would have to guess from
    config.EMBED_DIM, which on this box still says 1024 while the deployed 4B emits 2560.
    Comparing a guessed dim would force a full re-embed on every resume. The manifest still
    RECORDS the real dim (the worker knows it); the reuse decision just doesn't rest on it.
    """
    from dh2.manifest import doc_order_checksum
    return {"base_model": model, "max_sequence_length": max_seq_length,
            "document_order_checksum": doc_order_checksum(doc_ids),
            "n_rows": len(doc_ids)}


def _write_shard_manifest(shards_dir: Path, i: int, model: str, max_seq_length: int,
                          dim: int, doc_ids: list[str], batch_size: int) -> None:
    """Bind this shard to the model and documents that produced it (dh2.manifest)."""
    from dh2.manifest import write_manifest
    want = _shard_stamp(model, max_seq_length, doc_ids)
    write_manifest(_shard_manifest_path(shards_dir, i), artifact_kind="embedding_shard",
                   base_model=model, max_seq_length=max_seq_length, embedding_dim=dim,
                   doc_ids=doc_ids, document_prefix="",
                   query_prefix="(document side: no instruction)",
                   extra={"shard": i, "batch_size": batch_size, **want})
    # write_manifest nests our keys under "extra"; hoist the ones the resume check reads so
    # the comparison is against top-level fields it also writes itself.
    p = _shard_manifest_path(shards_dir, i)
    m = json.loads(p.read_text())
    m.update(want)
    p.write_text(json.dumps(m, indent=2))


def _shard_is_reusable(shards_dir: Path, i: int, want: dict) -> tuple[bool, str]:
    """A shard may be reused ONLY if it is provably this run's work."""
    if not _shard_path(shards_dir, i).exists():
        return False, "no shard file"
    mpath = _shard_manifest_path(shards_dir, i)
    if not mpath.exists():
        return False, ("no manifest -- shard predates provenance stamping and cannot be "
                       "attributed to any model; re-embedding")
    try:
        got = json.loads(mpath.read_text())
    except Exception as e:                                  # noqa: BLE001
        return False, f"unreadable manifest ({e})"
    for k, v in want.items():
        if got.get(k) != v:
            return False, f"manifest mismatch on {k}: shard={got.get(k)!r} run={v!r}"
    return True, "manifest matches this run"


def _load_texts(docs_path: Path) -> list[str]:
    return [d.embedding_text for d in read_docs(docs_path)]


def _load_ids_and_texts(docs_path: Path) -> tuple[list[str], list[str]]:
    """One pass for both -- the corpus is 1.5 GB and each worker would otherwise read it
    twice just to learn which doc_ids its rows belong to."""
    ids, texts = [], []
    for d in read_docs(docs_path):
        ids.append(d.doc_id)
        texts.append(d.embedding_text)
    return ids, texts


def _worker(device: str, mock: bool, batch_size: int, docs_path: str,
            shards_dir: Path, shard_size: int, task_q: "mp.Queue", seed: int,
            max_seq_length: int) -> None:
    """Pull shard indices off task_q, embed each, write its shard atomically."""
    set_global_determinism(seed)
    embedder = (get_embedder(mock=True) if mock
                else get_embedder(mock=False, device=device, batch_size=batch_size))
    if not mock:
        # Cap input length so per-batch memory is bounded regardless of a stray
        # very-long abstract (Qwen3 pads each batch to its longest input).
        embedder.model.max_seq_length = max_seq_length
    ids, texts = _load_ids_and_texts(Path(docs_path))
    n = len(texts)
    model_name = "mock" if mock else config.EMBED_MODEL
    dim = getattr(embedder, "dim", 0)
    while True:
        i = task_q.get()
        if i is None:                       # sentinel: no more work
            break
        s, e = i * shard_size, min(i * shard_size + shard_size, n)
        t0 = time.time()
        vecs = embedder.encode_documents(texts[s:e], batch_size=batch_size).astype(np.float32)
        tmp = shards_dir / f"shard_{i:05d}.tmp"
        with open(tmp, "wb") as fh:         # file object -> np.save won't munge the name
            np.save(fh, vecs)
        os.replace(tmp, _shard_path(shards_dir, i))   # atomic: shard appears only when complete
        # Write the manifest AFTER the shard, and only then: a manifest that exists without
        # its shard would licence reusing a file that isn't there. Order matters.
        _write_shard_manifest(shards_dir, i, model_name, max_seq_length, dim, ids[s:e],
                              batch_size)
        print(f"  [{device}] shard {i:>4} ({e - s} docs) {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="deterministic mock embedder (no GPU)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="truncate inputs to this many tokens (bounds GPU memory; "
                         "512 covers almost all abstracts)")
    ap.add_argument("--devices", default="cuda:0,cuda:1",
                    help="comma-separated devices; one worker per device")
    ap.add_argument("--shard-size", type=int, default=20000,
                    help="docs per checkpoint shard (crash loses at most this many)")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep shard files after combining (default: delete)")
    args = ap.parse_args()

    config.ensure_dirs()
    docs_path = config.NORM_DIR / "docs.jsonl"
    if not docs_path.exists():
        print(f"ERROR: {docs_path} not found -- run stage 02 first.", file=sys.stderr)
        return 2
    shards_dir = config.EMB_DIR / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    ids = [d.doc_id for d in read_docs(docs_path)]      # doc order == row order
    n = len(ids)
    shards = _shard_indices(n, args.shard_size)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    print(f"Embedding {n} docs (mock={args.mock}, "
          f"model={'mock' if args.mock else config.EMBED_MODEL})")
    print(f"  {len(shards)} shard(s) x {args.shard_size}, devices={devices}")

    # Resume on PROVENANCE, not on file existence. See the _shard_stamp block above: the
    # old `not _shard_path(...).exists()` shipped 60,000 rows of another model's vectors.
    model_name = "mock" if args.mock else config.EMBED_MODEL
    pending, reused = [], []
    for i in shards:
        s, e = i * args.shard_size, min(i * args.shard_size + args.shard_size, n)
        want = _shard_stamp(model_name, args.max_seq_length, ids[s:e])
        ok, why = _shard_is_reusable(shards_dir, i, want)
        (reused if ok else pending).append(i)
        if not ok and _shard_path(shards_dir, i).exists():
            print(f"  shard {i:>4}: RE-EMBEDDING -- {why}")
    print(f"  {len(reused)} reusable (manifest verified), {len(pending)} pending")

    if pending:
        mp.set_start_method("spawn", force=True)       # required for CUDA in children
        task_q: mp.Queue = mp.Queue()
        for i in pending:
            task_q.put(i)
        for _ in devices:                              # one sentinel per worker
            task_q.put(None)
        procs = []
        for dev in devices:
            p = mp.Process(target=_worker,
                           args=(dev, args.mock, args.batch_size, str(docs_path),
                                 shards_dir, args.shard_size, task_q, config.SEED,
                                 args.max_seq_length))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        missing = [i for i in shards if not _shard_path(shards_dir, i).exists()]
        if missing:
            print(f"ERROR: {len(missing)} shard(s) did not complete (e.g. {missing[:8]}). "
                  f"A worker likely crashed -- if it was CUDA OOM, lower --batch-size and "
                  f"re-run (it resumes).", file=sys.stderr)
            return 1

    # --- combine shards in order -> single aligned matrix ---
    print("combining shards ...")
    mats = [np.load(_shard_path(shards_dir, i)) for i in shards]
    vectors = np.vstack(mats) if mats else np.zeros((0, config.EMBED_DIM), np.float32)
    if vectors.shape[0] != n:
        print(f"ERROR: row count {vectors.shape[0]} != doc count {n}; not writing.",
              file=sys.stderr)
        return 1

    np.save(config.EMB_DIR / "doc_vectors.npy", vectors.astype(np.float32))
    with (config.EMB_DIR / "doc_ids.json").open("w") as fh:
        json.dump(ids, fh)
    print(f"Wrote vectors {vectors.shape} -> {config.EMB_DIR / 'doc_vectors.npy'}")

    if not args.keep_shards:
        for i in shards:
            _shard_path(shards_dir, i).unlink()
        try:
            shards_dir.rmdir()
        except OSError:
            pass
        print("  removed shard dir (pass --keep-shards to retain for resume)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
