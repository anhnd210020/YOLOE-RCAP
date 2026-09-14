"""Prepare or launch a locked A1/A3 YOLOE robust-vocabulary training run.

--dry-run performs CPU/file-only inspection and creates no output. Without it,
this command starts a full GPU training run; use only after separate approval.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT.parent / "YOLOE-RCAP"
PILOT = BASELINE / "pilot/v8s_10pct_v1"
RUNS = ROOT / "robust_vocab/runs"
PARENT_SHA = "9a7065c3e8c75b358e023d84473cbd78de24e56f"
sys.path.insert(0, str(ROOT))
# Reuse the locked pilot's input guards. No bytecode is written to its worktree.
sys.dont_write_bytecode = True
sys.path.insert(1, str(PILOT / "scripts"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def validate_baseline():
    def baseline_git(*args):
        return subprocess.check_output(["git", "-C", str(BASELINE), *args], text=True).strip()
    if baseline_git("branch", "--show-current") != "pilot/yoloe-v8s-10pct-1gpu":
        raise RuntimeError("Baseline worktree is on the wrong branch")
    if baseline_git("rev-parse", "HEAD") != PARENT_SHA:
        raise RuntimeError("Baseline worktree is not at the locked parent commit")
    if baseline_git("status", "--porcelain=v1", "-uall"):
        raise RuntimeError("Baseline worktree has local changes")


def tau_for(experiment, raw):
    if experiment == "A1":
        if raw is not None:
            raise ValueError("A1 fixes tau at 0.0; omit --tau")
        return 0.0, "0.0"
    if raw is None:
        raise ValueError("A3 requires an explicit --tau > 0")
    try:
        decimal = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError("--tau must be a finite positive decimal") from exc
    value = float(decimal)
    if not decimal.is_finite() or not math.isfinite(value) or value <= 0:
        raise ValueError("A3 requires a finite --tau > 0")
    # One canonical identity per parsed floating-point value, without unsafe path characters.
    return value, repr(value)


def run_name(experiment, tau_text, seed):
    return f"{experiment}_tau_{tau_text.replace('.', 'p').replace('-', 'm').replace('+', '')}_seed_{seed}"


def read_lock():
    path = PILOT / "PILOT_DATASET_LOCK.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    from train_pilot import require_lock_sections
    return path, require_lock_sections(lock)


def paths_from_lock(lock):
    return dict(
        lock=str(PILOT / "PILOT_DATASET_LOCK.json"),
        objects_json=lock["Objects365v1"]["subset_json"],
        objects_manifest=str(PILOT / "manifests/objects365_pilot10_v1.json"),
        objects_image_root=lock["Objects365v1"]["image_root"],
        objects_yaml=lock["Objects365v1"]["yaml_path"],
        gqa_json=lock["GQA"]["subset_json"],
        gqa_manifest=str(PILOT / "manifests/gqa_pilot10_v1.json"),
        gqa_image_root=lock["GQA"]["image_root"],
        flickr_json=lock["Flickr30k"]["subset_json"],
        flickr_manifest=str(PILOT / "manifests/flickr_pilot10_v1.json"),
        flickr_image_root=lock["Flickr30k"]["image_root"],
        mobileclip_weights=lock["text_artifacts"]["mobileclip_checkpoint"],
    )


def settings(tau, physical_batch, lock):
    from ultralytics.nn.tasks import guess_model_scale
    from ultralytics.utils import yaml_load
    model_path = "yoloe-v8s-seg.yaml"
    scale = guess_model_scale(model_path)
    cfg_dir = ROOT / "ultralytics/cfg"
    defaults = yaml_load(cfg_dir / "default.yaml")
    extends = yaml_load(cfg_dir / f"{scale}_train.yaml")
    if not all(key in defaults for key in extends):
        raise ValueError("Official scale overrides contain unknown keys")
    data = {"train": {"yolo_data": [lock["Objects365v1"]["yaml_path"]],
                      "grounding_data": [
                          {"source_name": "Flickr30k", "img_path": lock["Flickr30k"]["image_root"],
                           "json_file": lock["Flickr30k"]["subset_json"]},
                          {"source_name": "GQA", "img_path": lock["GQA"]["image_root"],
                           "json_file": lock["GQA"]["subset_json"]}]},
            "val": {"yolo_data": ["lvis.yaml"]}}
    kwargs = dict(extends)
    kwargs.update(dict(data=data, task="segment", imgsz=640, batch=physical_batch,
                       epochs=30, nbs=128, close_mosaic=2, optimizer="AdamW",
                       lr0=0.002, warmup_bias_lr=0.0, weight_decay=0.05,
                       momentum=0.9, workers=4, seed=0, deterministic=True,
                       device=0, text_model="mobileclip:blt", robust_vocab_tau=tau,
                       fraction=1.0, val=True, save=True, plots=False,
                       resume=False, pretrained=True, amp=defaults["amp"],
                       cos_lr=defaults["cos_lr"], lrf=defaults["lrf"],
                       warmup_epochs=defaults["warmup_epochs"],
                       warmup_momentum=defaults["warmup_momentum"],
                       save_period=defaults["save_period"], exist_ok=True))
    return model_path, kwargs, {"defaults": defaults, "scale_overrides": extends}


def manifest(experiment, tau, tau_text, name, lock_path, lock, model_path, kwargs, cfg):
    status = git("status", "--porcelain=v1", "-uall")
    run_dir = RUNS / name
    source_rows = {}
    for key in ("Objects365v1", "GQA", "Flickr30k"):
        source_rows[key] = {k: lock[key][k] for k in ("subset_json", "subset_sha256", "manifest_sha256")}
        source_rows[key]["cache_sha256"] = lock[key].get("cache_sha256")
    return {
        "experiment_id": name, "arm": experiment, "robust_vocab_tau": tau,
        "tau_repr": tau_text, "git_branch": git("branch", "--show-current"),
        "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(status),
        "git_status_porcelain": status.splitlines(), "baseline_parent_commit": PARENT_SHA,
        "dataset_lock_path": str(lock_path), "dataset_lock_sha256": sha256_file(lock_path),
        "subsets": source_rows,
        "model_config": {"path": model_path, "sha256": sha256_file(ROOT / "ultralytics/cfg/models/v8/yoloe-v8-seg.yaml"),
                         "scale_overrides": cfg["scale_overrides"]},
        "initialization_policy": "Fresh YOLOE-v8-S YAML detector; trainer seeds before get_model; no detector checkpoint; MobileCLIP BLT from locked weights; no A1/baseline continuation",
        "initialization_fingerprint": None,
        "mobileclip_path": lock["text_artifacts"]["mobileclip_checkpoint"],
        "mobileclip_sha256": lock["text_artifacts"]["mobileclip_checkpoint_sha256"],
        "text_artifacts": {k: lock["text_artifacts"][k] for k in (
            "train_label_embeddings_sha256", "global_negative_categories_sha256", "global_negative_embeddings_sha256")},
        "seed": kwargs["seed"], "deterministic": kwargs["deterministic"],
        "epochs": kwargs["epochs"], "imgsz": kwargs["imgsz"],
        "physical_batch": kwargs["batch"], "nbs": kwargs["nbs"],
        "initial_accumulation": round(kwargs["nbs"] / kwargs["batch"]),
        "optimizer": kwargs["optimizer"], "lr0": kwargs["lr0"],
        "weight_decay_argument": kwargs["weight_decay"], "momentum": kwargs["momentum"],
        "scheduler": {k: kwargs[k] for k in ("cos_lr", "lrf", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "close_mosaic")},
        "augmentation": {k: cfg["defaults"][k] for k in (
            "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "copy_paste_mode")},
        "amp": kwargs["amp"], "ema": True, "workers": kwargs["workers"],
        "train_only_validation_bypass": True, "save_period": kwargs["save_period"],
        "run_directory": str(run_dir), "start_timestamp": None, "end_timestamp": None,
        "exit_status": "not_started", "last_pt_path": str(run_dir / "weights/last.pt"),
        "last_pt_sha256": None, "peak_vram_bytes": None, "training_duration_seconds": None,
        "final_metrics": None,
        "final_evaluator": "full LVIS v1 minival, text prompt, 640, conf .001, IoU .7, max_det 1000, bbox Fixed AP, epoch-30 last.pt",
    }


def write_manifest(path, payload, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def model_fingerprint(model):
    """Hash named initial state tensors, including buffers, before the first update."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("A1", "A3"))
    parser.add_argument("--tau", help="Required explicitly for A3; forbidden for A1")
    parser.add_argument("--physical-batch", type=int, default=32, choices=(32,),
                        help="Locked pilot physical batch (32)")
    parser.add_argument("--dry-run", action="store_true", help="Print pending manifest; no training or output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tau, tau_text = tau_for(args.experiment, args.tau)
    validate_baseline()
    if git("branch", "--show-current") != "research/robust-vocab-v1":
        raise RuntimeError("Wrong robust-vocabulary worktree branch")
    if git("merge-base", "HEAD", PARENT_SHA) != PARENT_SHA:
        raise RuntimeError("Expected baseline parent is not an ancestor")
    lock_path, lock = read_lock()
    model_path, kwargs, cfg = settings(tau, args.physical_batch, lock)
    name = run_name(args.experiment, tau_text, kwargs["seed"])
    output = RUNS / name
    if output.exists():
        raise FileExistsError(f"Refusing existing experiment output: {output}")
    record = manifest(args.experiment, tau, tau_text, name, lock_path, lock, model_path, kwargs, cfg)
    if args.dry_run:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    # The baseline preflight validates the immutable dataset, text and LVIS inputs.
    from types import SimpleNamespace
    from train_pilot import preflight
    # Baseline preflight's last check reserves its own main_batch_<physical> path.
    # Its other checks do not use the batch value; supply a distinct sentinel so
    # the completed baseline run cannot block read-only validation of its inputs.
    # The actual train kwargs and manifest retain the locked physical batch 32.
    preflight(SimpleNamespace(**paths_from_lock(lock), physical_batch=1_000_000_000, smoke=False))
    from pilot_trainer import PilotTrainOnlyTrainer, PilotYOLOESegTrainerFromScratch
    from pilot_text import install_pilot_text
    from ultralytics import YOLOE
    install_pilot_text(lock["text_artifacts"])
    PilotYOLOESegTrainerFromScratch.lock_path = str(lock_path)
    manifest_path = output / "experiment_manifest.json"

    class ExperimentTrainer(PilotTrainOnlyTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise ValueError("A1/A3 must start from fresh detector initialization")
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            record["initialization_fingerprint"] = model_fingerprint(model)
            write_manifest(manifest_path, record)
            return model

    RUNS.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # Atomic reservation; Ultralytics may use this exact directory only.
    record["start_timestamp"] = utc_now()
    record["exit_status"] = "running"
    write_manifest(manifest_path, record, exclusive=True)
    kwargs.update(trainer=ExperimentTrainer, project=str(RUNS), name=name)
    started = time.monotonic()
    old_cwd = Path.cwd()
    try:
        os.chdir(Path(lock["text_artifacts"]["mobileclip_checkpoint"]).parent)
        YOLOE(model_path).train(**kwargs)
    except BaseException as exc:
        record["exit_status"] = f"failed: {type(exc).__name__}: {exc}"
        raise
    else:
        record["exit_status"] = "completed"
    finally:
        os.chdir(old_cwd)
        record["end_timestamp"] = utc_now()
        record["training_duration_seconds"] = time.monotonic() - started
        if Path(record["last_pt_path"]).is_file():
            record["last_pt_sha256"] = sha256_file(record["last_pt_path"])
        try:
            import torch
            if torch.cuda.is_initialized():
                record["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        except ImportError:
            pass
        write_manifest(manifest_path, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
