#!/usr/bin/env bash
# pull_bigmodel_h100.sh  --  RUN ON DREW after a 4B/8B arm finishes.
# Pulls the model + vectors + eval report into SIZE-tagged dirs so nothing collides
# with the 0.6B primary or the other big-model arm.
#
# Usage:  ./pull_bigmodel_h100.sh <ssh-alias> <4B|8B>
#   e.g.  ./pull_bigmodel_h100.sh vast4b 4B
set -euo pipefail

VAST="${1:?usage: pull_bigmodel_h100.sh <alias> <4B|8B>}"
SIZE="${2:?usage: pull_bigmodel_h100.sh <alias> <4B|8B>}"
TAG="qwen3-dh-ft-${SIZE,,}"
REMOTE=/workspace
DREW=/home/alex/discovery_hub

echo "==> [$SIZE] model -> $DREW/models/$TAG/"
mkdir -p "$DREW/models/$TAG"
rsync -avz --progress "$VAST:$REMOTE/dh/models/$TAG/final/" "$DREW/models/$TAG/"

echo "==> [$SIZE] vectors -> $DREW/data/embeddings_${SIZE,,}/  (NOT overwriting primary)"
mkdir -p "$DREW/data/embeddings_${SIZE,,}"
rsync -avz --progress \
  "$VAST:$REMOTE/dh/data/embeddings/doc_vectors.npy" \
  "$VAST:$REMOTE/dh/data/embeddings/doc_ids.json" \
  "$DREW/data/embeddings_${SIZE,,}/"

echo "==> [$SIZE] report -> $DREW/data/reports/eval_report_${SIZE}.md"
rsync -avz --progress "$VAST:$REMOTE/dh/data/reports/eval_report.md" \
  "$DREW/data/reports/eval_report_${SIZE}.md"
rsync -avz --progress "$VAST:$REMOTE/dh/data/reports/eval_report.json" \
  "$DREW/data/reports/eval_report_${SIZE}.json" || true

echo; echo "==> pulled $SIZE. Compare the capacity curve:"
echo "grep -E 'recall@10|recall@1 |mrr@10' \\"
echo "  $DREW/data/reports/eval_report_FT_primary.md \\"
echo "  $DREW/data/reports/eval_report_4B.md \\"
echo "  $DREW/data/reports/eval_report_8B.md 2>/dev/null"
echo "Then DESTROY the $SIZE instance."
