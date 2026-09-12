"""Later on the GPU server, try 32, then 16, then 8 using train_pilot --smoke.

Pass all train_pilot data/lock arguments after --. No GPU work occurs without
--execute. Each trial runs in a fresh process so an OOM does not poison the next.
"""
import argparse
import subprocess
import sys
from pathlib import Path

CUDA_OOM_EXIT_CODE = 75  # Must match train_pilot.py; only this code is retryable.


def should_retry(exit_code):
    return exit_code == CUDA_OOM_EXIT_CODE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually run GPU smoke trials")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if "--physical-batch" in forwarded or "--smoke" in forwarded:
        parser.error("Batch and smoke mode are controlled by this tool")
    for batch in (32, 16, 8):
        command = [sys.executable, str(Path(__file__).with_name("train_pilot.py")),
                   "--smoke", "--physical-batch", str(batch), *forwarded]
        print(" ".join(command), flush=True)
        if not args.execute:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"Smoke passed at physical batch {batch}; select this batch explicitly for the main run")
            return
        if not should_retry(result.returncode):
            raise SystemExit(f"Smoke batch {batch} failed with non-OOM exit {result.returncode}; stopping without retry")
        print(f"Smoke batch {batch} hit CUDA OOM (exit {CUDA_OOM_EXIT_CODE}); trying smaller batch", flush=True)
    if args.execute:
        raise SystemExit("All 32/16/8 smoke trials hit CUDA OOM; do not start main training")


if __name__ == "__main__":
    main()
