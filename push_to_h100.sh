#!/usr/bin/env bash
# push_to_h100.sh  --  RUN ON DREW.
# Stages everything the parallel cached-MNRL arm needs onto a rented Vast H100.
# The arm is self-contained: it only needs docs.jsonl + train_triples.jsonl +
# the pipeline code + the base model cache. Nothing on Drew is modified.
#
# Usage:  ./push_to_h100.sh <ssh-alias-or-host>
#   e.g.  ./push_to_h100.sh vasth100      (if you added an ~/.ssh/config stanza)
#   or edit VAST below and run with no args.
set -euo pipefail

VAST="${1:-vasth100}"                       # ssh alias OR user@host (config handles -p/-i)
DREW_DATA="/home/alex/discovery_hub/data"
DREW_CODE="/home/alex/discovery_hub_pipeline"
DREW_HFCACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B"
REMOTE_ROOT="/workspace"

echo "==> target: $VAST"
ssh "$VAST" "mkdir -p $REMOTE_ROOT/dh_pipeline \
                     $REMOTE_ROOT/dh/data/normalized \
                     $REMOTE_ROOT/dh/data/finetune \
                     $REMOTE_ROOT/hf/hub"

echo "==> pipeline code (excludes venv/data/caches)"
rsync -avz --exclude venv --exclude data --exclude '__pycache__' --exclude '*.pyc' \
  "$DREW_CODE/" "$VAST:$REMOTE_ROOT/dh_pipeline/"

echo "==> the two data files the arm needs (docs + triples)"
rsync -avz --progress "$DREW_DATA/normalized/docs.jsonl" \
  "$VAST:$REMOTE_ROOT/dh/data/normalized/"
rsync -avz --progress "$DREW_DATA/finetune/train_triples.jsonl" \
  "$VAST:$REMOTE_ROOT/dh/data/finetune/"

echo "==> held-out eval qrels (so the H100 can score without Drew)"
rsync -avz --progress \
  "$DREW_DATA/finetune/synthetic_queries.jsonl.eval.jsonl" \
  "$VAST:$REMOTE_ROOT/dh/data/finetune/"

echo "==> base model cache (rsync -L dereferences HF symlinks -> real files)"
if [ -d "$DREW_HFCACHE" ]; then
  rsync -avzL --progress "$DREW_HFCACHE" "$VAST:$REMOTE_ROOT/hf/hub/"
else
  echo "   [warn] $DREW_HFCACHE not found; the H100 will download it from HF instead."
fi

cat <<EOF

==> done staging. Next:
   1) copy run_on_h100.sh up:   rsync -avz run_on_h100.sh $VAST:$REMOTE_ROOT/
   2) ssh in:                   ssh $VAST
   3) run the arm:              cd $REMOTE_ROOT && bash run_on_h100.sh
EOF
