#!/usr/bin/env bash
# run_bigmodel_h100.sh  --  RUN ON A RENTED VAST H100 (inside /workspace).
# Fine-tunes a LARGER base embedder (4B or 8B) with LoRA, then re-embeds, indexes,
# and evals on the SAME held-out cross-register queries as every other arm — so the
# number is directly comparable to the 0.6B primary (0.406) and the cached-MNRL arm.
#
# Pass the model as arg 1:  bash run_bigmodel_h100.sh 4B    (or 8B)
# Self-contained: reads/writes only /workspace/dh/data. Copy results back with
# pull_bigmodel_h100.sh (run on Drew) when done, then DESTROY the instance.
set -euo pipefail

SIZE="${1:?usage: run_bigmodel_h100.sh <4B|8B>}"
case "$SIZE" in
  4B) BASE="Qwen/Qwen3-Embedding-4B"; EMB_BATCH=384 ;;
  8B) BASE="Qwen/Qwen3-Embedding-8B"; EMB_BATCH=192 ;;
  *)  echo "SIZE must be 4B or 8B"; exit 2 ;;
esac
TAG="qwen3-dh-ft-${SIZE,,}"          # e.g. qwen3-dh-ft-4b
ROOT=/workspace
export DH_DATA_ROOT=$ROOT/dh/data
export HF_HOME=$ROOT/hf
PYBIN="$(command -v python || echo /venv/main/bin/python)"
cd "$ROOT/dh_pipeline"

# Load an HF token if one was staged (removes download throttling; needed if the
# base model is gated). Create on Drew: echo 'export HF_TOKEN=hf_...' > hf_token.env
[ -f "$ROOT/hf_token.env" ] && { source "$ROOT/hf_token.env"; echo "  HF_TOKEN loaded from hf_token.env"; }

echo "==> model=$BASE  tag=$TAG  python=$PYBIN"
"$PYBIN" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())"
"$PYBIN" - <<'PY'
import sys
try:
    import sentence_transformers, datasets, accelerate, peft  # noqa
    print("deps OK (peft importable for LoRA)")
except Exception as e:
    print("MISSING DEPS:", e); sys.exit(1)
PY

MODEL_OUT=$ROOT/dh/models/$TAG

echo; echo "==> [1/4] dry-run (build dataset, no GPU)"
"$PYBIN" train.py --dry-run --base-model "$BASE" --use-lora --batch-size 8 --n-negatives 4

echo; echo "==> [2/4] TRAIN  $SIZE + LoRA  (merges adapter on save -> plain SentenceTransformer)"
# LoRA is mandatory at 4B/8B on one 80GB card. If OOM, add: --load-in-4bit
nohup "$PYBIN" -u train.py \
  --base-model "$BASE" --use-lora --lora-r 16 --lora-alpha 32 \
  --out "$MODEL_OUT" \
  --batch-size 8 --n-negatives 4 --epochs 1 \
  > "train_${SIZE}.log" 2>&1 &
TP=$!
echo "   training PID $TP -> train_${SIZE}.log   (watch: tail -f $ROOT/dh_pipeline/train_${SIZE}.log)"
wait $TP
test -f "$MODEL_OUT/final/model.safetensors" || { echo "TRAIN FAILED — see train_${SIZE}.log (try --load-in-4bit if OOM)"; exit 1; }

echo; echo "==> [3/4] RE-EMBED all docs with the $SIZE model, then rebuild index"
export DH_EMBED_MODEL=$MODEL_OUT/final
"$PYBIN" -u 04_generate_embeddings.py --devices cuda:0 --batch-size "$EMB_BATCH" --max-seq-length 512
"$PYBIN" -u 05_build_index.py

echo; echo "==> [4/4] EVAL on held-out cross-register qrels"
HF_HUB_OFFLINE=0 "$PYBIN" -u 10_eval_retrieval.py \
  --qrels "$DH_DATA_ROOT/finetune/synthetic_queries.jsonl.eval.jsonl" \
  2>&1 | tee "eval_${SIZE}.log"

echo; echo "==> DONE ($SIZE). Report: $DH_DATA_ROOT/reports/eval_report.md"
echo "   Pull to Drew (pull_bigmodel_h100.sh <alias> $SIZE), then DESTROY this instance."
