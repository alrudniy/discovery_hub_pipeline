#!/usr/bin/env bash
# pull_from_h100.sh  --  RUN ON DREW, after the H100 arm finishes.
# Copies the cached-MNRL model + its vectors + its eval report into SEPARATE dirs on
# Drew (suffix _cachedmnrl) so nothing overwrites your primary fine-tuned model,
# doc_vectors.npy, or eval_report.md. Then you can diff the two reports at leisure.
#
# Usage:  ./pull_from_h100.sh <ssh-alias-or-host>
set -euo pipefail

VAST="${1:-vasth100}"
REMOTE_ROOT="/workspace"
DREW="/home/alex/discovery_hub"

echo "==> model -> $DREW/models/qwen3-dh-ft-cachedmnrl/"
mkdir -p "$DREW/models/qwen3-dh-ft-cachedmnrl"
rsync -avz --progress \
  "$VAST:$REMOTE_ROOT/dh/models/qwen3-dh-ft-cachedmnrl/final/" \
  "$DREW/models/qwen3-dh-ft-cachedmnrl/"

echo "==> vectors -> $DREW/data/embeddings_cachedmnrl/  (NOT overwriting primary)"
mkdir -p "$DREW/data/embeddings_cachedmnrl"
rsync -avz --progress \
  "$VAST:$REMOTE_ROOT/dh/data/embeddings/doc_vectors.npy" \
  "$VAST:$REMOTE_ROOT/dh/data/embeddings/doc_ids.json" \
  "$DREW/data/embeddings_cachedmnrl/"

echo "==> eval report -> $DREW/data/reports/eval_report_CACHEDMNRL.{md,json}"
rsync -avz --progress \
  "$VAST:$REMOTE_ROOT/dh/data/reports/eval_report.md" \
  "$DREW/data/reports/eval_report_CACHEDMNRL.md"
rsync -avz --progress \
  "$VAST:$REMOTE_ROOT/dh/data/reports/eval_report.json" \
  "$DREW/data/reports/eval_report_CACHEDMNRL.json" || true

cat <<EOF

==> pulled. Compare the two arms:
   grep -E 'recall@10|mrr@10|ndcg@10' \\
     $DREW/data/reports/eval_report_BASELINE_crossregister.md \\
     $DREW/data/reports/eval_report_CACHEDMNRL.md

   Then DESTROY the Vast instance (meter runs until destroyed).
EOF
