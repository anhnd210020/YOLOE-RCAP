#!/usr/bin/env bash
set -Eeo pipefail
BASELINE_LOGS=/workspace/YOLOE-RCAP/baseline_logs
exec 9> "$BASELINE_LOGS/bbox_runner.lock"
flock -n 9 || { echo 'Bbox baseline runner is already active.'; exit 1; }
if [[ -e "$BASELINE_LOGS/bbox_job_started.lock" || -e "$BASELINE_LOGS/bbox_inference_started.lock" ]]; then
    echo 'Approved bbox job was already launched; no automatic retry is permitted.'
    exit 1
fi
(set -o noclobber; : > "$BASELINE_LOGS/bbox_job_started.lock")
exec >> "$BASELINE_LOGS/baseline_result.txt" 2>&1
finish() {
    local code=$?
    trap - EXIT
    /workspace/yoloe-env/bin/python -B -u "$BASELINE_LOGS/finalize_bbox_baseline.py" "$code"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$BASELINE_LOGS/ultralytics"
export MPLCONFIGDIR="$BASELINE_LOGS/matplotlib"
export TMPDIR="$BASELINE_LOGS/tmp"
export XDG_CACHE_HOME="$BASELINE_LOGS/xdg_cache"
export TORCH_HOME="$BASELINE_LOGS/torch_cache"
export CUDA_CACHE_PATH="$BASELINE_LOGS/cuda_cache"
export PYTHONPATH=/workspace/YOLOE-RCAP
export TQDM_MININTERVAL=5
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yoloe-env
cd /workspace/YOLOE-RCAP
set -u
python -u "$BASELINE_LOGS/bbox_job.py"
