#!/usr/bin/env bash
set -euo pipefail

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yoloe-env
cd /workspace/YOLOE-RCAP

export PYTHONPATH="/workspace/YOLOE-RCAP:/workspace/YOLOE-RCAP/baseline_logs:/workspace/YOLOE-RCAP/third_party/lvis-api:${PYTHONPATH:-}"

REPORT="/workspace/YOLOE-RCAP/baseline_logs/baseline_v8_ml_result.txt"
ANN="/workspace/datasets/lvis/annotations/lvis_v1_minival.json"

{
  echo
  echo "============================================================"
  echo "FULL YOLOE-v8-M/L BASELINE RUN"
  echo "Started: $(date -Is)"
  echo "============================================================"

  echo
  echo "=== YOLOE-v8-M FULL INFERENCE ==="
} | tee -a "$REPORT"

python -u baseline_logs/v8_ml_inference.py --scale m \
  2>&1 | tee -a "$REPORT"

{
  echo
  echo "=== YOLOE-v8-M FIXED AP ==="
} | tee -a "$REPORT"

PYTHONPATH="/workspace/YOLOE-RCAP/third_party/lvis-api:/workspace/YOLOE-RCAP:/workspace/YOLOE-RCAP/baseline_logs:${PYTHONPATH:-}" \
python -u tools/eval_fixed_ap.py \
  "$ANN" \
  "/workspace/YOLOE-RCAP/baseline_logs/v8m_fixed_ap/predictions.json" \
  --type bbox \
  2>&1 | tee -a "$REPORT"

{
  echo
  echo "=== YOLOE-v8-L FULL INFERENCE ==="
} | tee -a "$REPORT"

python -u baseline_logs/v8_ml_inference.py --scale l \
  2>&1 | tee -a "$REPORT"

{
  echo
  echo "=== YOLOE-v8-L FIXED AP ==="
} | tee -a "$REPORT"

PYTHONPATH="/workspace/YOLOE-RCAP/third_party/lvis-api:/workspace/YOLOE-RCAP:/workspace/YOLOE-RCAP/baseline_logs:${PYTHONPATH:-}" \
python -u tools/eval_fixed_ap.py \
  "$ANN" \
  "/workspace/YOLOE-RCAP/baseline_logs/v8l_fixed_ap/predictions.json" \
  --type bbox \
  2>&1 | tee -a "$REPORT"

{
  echo
  echo "============================================================"
  echo "M/L FULL BASELINE PIPELINE FINISHED"
  echo "Finished: $(date -Is)"
  echo "============================================================"
} | tee -a "$REPORT"
