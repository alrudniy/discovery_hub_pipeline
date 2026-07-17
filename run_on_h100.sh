#!/usr/bin/env bash
# run_on_h100.sh  --  RUN ON THE RENTED VAST H100 (inside /workspace).
# The cheap, informative ablation: SAME 0.6B base, SAME triples, SAME eval as Drew's
# run -- the only change is CachedMNRL with a huge batch (~1279 negatives vs Drew's 79).
# If Recall@10 jumps here, in-batch negatives were the bottleneck. If flat, capacity
# (a bigger base model) is the lever -- that's the next rung, not this one.
#
# Self-contained: reads/writes only /workspace/dh/data (its own DH_DATA_ROOT), so it
# cannot touch Drew. Copy results back with pull_from_h100.sh (run on Drew) when done.
set -euo pipefail

ROOT=/workspace
export DH_DATA_ROOT=$ROOT/dh/data
export HF_HOME=$ROOT/hf
export HF_HUB_OFFLINE=1                      # use the rsynced base-model cache; no downloads
PYBIN="$(command -v python || echo /venv/main/bin/python)"
cd "$ROOT/dh_pipeline"

echo "==> python: $PYBIN"; "$PYBIN" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())"
"$PYBIN" - <<'PY'
import sys
try:
    import sentence_transformers, datasets, accelerate  # noqa
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss  # noqa
    print("deps OK (CachedMNRL importable)")
except Exception as e:
    print("MISSING DEPS:", e); sys.exit(1)
PY

MODEL_OUT=$ROOT/dh/models/qwen3-dh-ft-cachedmnrl

echo; echo "==> [1/4] dry-run (build dataset, no GPU) -- expect kept~66214, prefix on anchor only"
"$PYBIN" train.py --dry-run --loss cached-mnrl --batch-size 256 --mini-batch-size 32 --n-negatives 8

echo; echo "==> [2/4] TRAIN  cached-MNRL, batch 256, 8 negs  (~1279 effective negatives/anchor)"
# grad checkpointing ON (proven necessary); mini-batch 32 sets VRAM. If OOM: mini-batch 16.
nohup "$PYBIN" -u train.py \
  --out "$MODEL_OUT" \
  --loss cached-mnrl --batch-size 256 --mini-batch-size 32 \
  --n-negatives 8 --epochs 1 \
  > train_cachedmnrl.log 2>&1 &
TRAIN_PID=$!
echo "   training PID $TRAIN_PID -> train_cachedmnrl.log"
echo "   watch:   tail -f $ROOT/dh_pipeline/train_cachedmnrl.log"
echo "   loss should fall from ~2-3 toward <1 (random baseline ln(2304)=7.7 here)."
wait $TRAIN_PID
test -f "$MODEL_OUT/final/model.safetensors" || { echo "TRAIN FAILED -- see log"; exit 1; }

echo; echo "==> [3/4] RE-EMBED all docs with the new model, then rebuild index"
export DH_EMBED_MODEL=$MODEL_OUT/final
"$PYBIN" -u 04_generate_embeddings.py --devices cuda:0 --batch-size 512 --max-seq-length 512
"$PYBIN" -u 05_build_index.py

echo; echo "==> [4/4] EVAL on the held-out cross-register qrels (rerank loads from HF cache)"
# reranker (BGE) may need HF; if it errors on download, unset HF_HUB_OFFLINE for this step.
HF_HUB_OFFLINE=0 "$PYBIN" -u 10_eval_retrieval.py \
  --qrels "$DH_DATA_ROOT/finetune/synthetic_queries.jsonl.eval.jsonl" \
  2>&1 | tee eval_cachedmnrl.log

echo; echo "==> DONE. Report: $DH_DATA_ROOT/reports/eval_report.md"
echo "   Compare dense Recall@10 / MRR@10 to Drew baseline (0.287 / 0.143)."
echo "   Pull back to Drew (run pull_from_h100.sh on Drew), then DESTROY this instance."
