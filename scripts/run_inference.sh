#!/bin/bash
# ==============================================================================
# ICRL-Sum: Inference & Evaluation
# Purpose: Generate final metrics (ROUGE, BERTScore, Grounding) using
#          the optimized policies or final outputs.
# ==============================================================================

set -e
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

# --- Paths ---
PRED_DIR="/data/project/output/summaries/train_trajectories"
GT_DIR="/data/project/data/annotations/gold_standard"
REPORT_DIR="/data/project/output/reports"

# --- 1. Aggregation (Optional) ---
# If prediction files are scattered, we might want to merge them for bulk eval
# Here we assume evaluator handles batch loading or we iterate.

echo ">>> [Inference] Starting Evaluation Protocol..."

# Timestamp for the report
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$REPORT_DIR/eval_metrics_$TIMESTAMP.json"

# Check dependencies
if [ -z "$(ls -A $PRED_DIR)" ]; then
   echo "[Error] No predictions found in $PRED_DIR. Run run_icrl_train.sh first."
   exit 1
fi

# Run Evaluator
# Note: The Python evaluator script expects a specific format. 
# We assume here that we run evaluation on all matching pairs.

echo "  > Computing Metrics (NLI Grounding, tIoU, BERTScore)..."

python summary/evaluator.py \
    --pred_path "$PRED_DIR" \
    --gt_path "$GT_DIR" \
    --out_report "$REPORT_FILE"

# --- Output Summary ---
echo "----------------------------------------------------------------"
echo "Evaluation Complete."
echo "Metrics saved to: $REPORT_FILE"
if [ -f "$REPORT_FILE" ]; then
    cat "$REPORT_FILE"
fi
echo "----------------------------------------------------------------"