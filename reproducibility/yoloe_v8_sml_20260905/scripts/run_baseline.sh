#!/usr/bin/env bash
set -Eeo pipefail
BASELINE_LOGS=/workspace/YOLOE-RCAP/baseline_logs
mkdir -p "$BASELINE_LOGS"
exec 9> "$BASELINE_LOGS/runner.lock"
flock -n 9 || { echo 'Baseline runner already active; no second run started.'; exit 1; }
if [[ -e "$BASELINE_LOGS/job_launched.lock" || -e "$BASELINE_LOGS/inference_started.lock" ]]; then
    echo 'Baseline was already launched; refusing a second inference.'
    exit 1
fi
(set -o noclobber; : > "$BASELINE_LOGS/job_launched.lock")
exec >> "$BASELINE_LOGS/baseline_result.txt" 2>&1
finish() {
    local code=$?
    trap - EXIT
    /workspace/yoloe-env/bin/python -B -u "$BASELINE_LOGS/finalize_baseline.py" "$code"
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$BASELINE_LOGS/ultralytics"
export MPLCONFIGDIR="$BASELINE_LOGS/matplotlib"
export TMPDIR="$BASELINE_LOGS/tmp"
export XDG_CACHE_HOME="$BASELINE_LOGS/xdg_cache"
export TORCH_HOME="$BASELINE_LOGS/torch_cache"
export CUDA_CACHE_PATH="$BASELINE_LOGS/cuda_cache"
export TQDM_MININTERVAL=5
export PYTHONPATH=/workspace/YOLOE-RCAP
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH"
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/yoloe-env
cd /workspace/YOLOE-RCAP
set -u
printf '\nRunner: bash /workspace/YOLOE-RCAP/baseline_logs/run_baseline.sh\n'
python -u "$BASELINE_LOGS/baseline_driver.py"
