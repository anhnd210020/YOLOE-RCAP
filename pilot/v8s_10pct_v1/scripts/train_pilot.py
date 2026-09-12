"""Guarded single-GPU YOLOE-v8-S pilot entry point for later server use.

No action is taken merely by importing this module. A physical batch must be
explicitly chosen after smoke testing. Main training uses nbs=128; optimizer
updates approximate 128 images but are not mathematically identical to 8-GPU DDP.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from make_subset import inside_pilot, sha256_file
from verify_subset import verify
from pilot_grounding import load_lock_entry, read_lock, selected_image_files, verify_cache_file

CUDA_OOM_EXIT_CODE = 75  # Retryable smoke-only CUDA out-of-memory failure.


def is_cuda_oom(exc):
    """Classify the exception itself, never subprocess stdout or generic errors."""
    if not isinstance(exc, RuntimeError):
        return False
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def run_smoke_trainer(model, kwargs):
    """Mirror the pinned YOLOE.train setup without its mandatory checkpoint reload."""
    from pilot_trainer import PilotSmokeTrainer
    overrides = dict(kwargs)
    overrides.pop("trainer", None)
    overrides.update(model=model.overrides["model"], task=model.task, mode="train")
    trainer = PilotSmokeTrainer(overrides=overrides, _callbacks=model.callbacks)
    trainer.model = trainer.get_model(weights=model.model if model.ckpt else None, cfg=model.model.yaml)
    trainer.train()


def require_lock_sections(lock):
    required = {
        "Objects365v1": {"subset_json", "subset_sha256", "manifest_sha256", "expected_image_count",
                         "expected_annotation_count", "class_coverage_status", "class_coverage", "yaml_path", "yaml_sha256",
                         "image_root", "label_root", "label_count", "label_tree_sha256"},
        "GQA": {"subset_json", "subset_sha256", "manifest_sha256", "expected_image_count",
                "expected_instance_count", "cache_sha256", "image_root"},
        "Flickr30k": {"subset_json", "subset_sha256", "manifest_sha256", "expected_image_count",
                      "expected_instance_count", "cache_sha256", "image_root"},
        "text_artifacts": {"text_model", "mobileclip_checkpoint", "mobileclip_checkpoint_sha256",
                           "train_label_embeddings", "train_label_embeddings_sha256",
                           "global_negative_categories", "global_negative_categories_sha256",
                           "global_negative_embeddings", "global_negative_embeddings_sha256",
                           "global_negative_threshold", "global_negative_count"},
        "evaluation": {"name", "expected_image_count", "minival_list", "minival_list_sha256",
                       "minival_annotation", "minival_annotation_sha256"},
    }
    if lock.get("version") != "v1" or lock.get("pilot_id") != "yoloe-v8s-10pct-v1":
        raise ValueError("Wrong unified pilot dataset lock version or ID")
    for section, keys in required.items():
        if section not in lock or not isinstance(lock[section], dict):
            raise ValueError(f"Unified dataset lock missing section: {section}")
        missing = keys - set(lock[section])
        if missing:
            raise ValueError(f"Unified dataset lock {section} missing fields: {sorted(missing)}")
    return lock


def preflight(args):
    if args.physical_batch is None or args.physical_batch <= 0:
        raise ValueError("Select --physical-batch intentionally after GPU smoke testing")
    lock_path = inside_pilot(args.lock)
    lock = require_lock_sections(read_lock(lock_path))
    from prepare_text_embeddings import verify_evaluation
    from prepare_objects365 import label_tree_sha256
    from pilot_text import verify_text_artifacts
    verify_evaluation(lock["evaluation"])
    verify_text_artifacts(lock["text_artifacts"], args.mobileclip_weights)
    local_lvis = Path(__file__).resolve().parents[4] / "datasets" / "lvis"
    if Path(lock["evaluation"]["minival_list"]).resolve() != (local_lvis / "minival.txt").resolve():
        raise ValueError("Locked LVIS list path differs from the official relative dataset layout")
    if Path(lock["evaluation"]["minival_annotation"]).resolve() != (local_lvis / "annotations/lvis_v1_minival.json").resolve():
        raise ValueError("Locked LVIS annotation path differs from the official relative dataset layout")
    objects = verify(SimpleNamespace(subset_json=args.objects_json, manifest=args.objects_manifest,
                                     source_json=None, image_root=args.objects_image_root,
                                     flat_images=True, cache=None, lock_out=None))
    if objects["source_name"] != "Objects365v1":
        raise ValueError("Objects365 subset source mismatch")
    object_entry = lock["Objects365v1"]
    if Path(object_entry["subset_json"]).resolve() != Path(args.objects_json).resolve():
        raise ValueError("Objects365 locked subset path mismatch")
    if object_entry["subset_sha256"] != objects["subset_sha256"] or object_entry["manifest_sha256"] != sha256_file(args.objects_manifest):
        raise ValueError("Objects365 subset or manifest SHA256 mismatch")
    if object_entry["expected_image_count"] != objects["images"] or object_entry["expected_annotation_count"] != objects["annotations"]:
        raise ValueError("Objects365 locked image or annotation count mismatch")
    if object_entry["class_coverage_status"] != "PASS" or object_entry["class_coverage"] != objects["class_coverage"]:
        raise ValueError("Objects365 class coverage lock mismatch")
    if Path(object_entry["image_root"]).resolve() != Path(args.objects_image_root).resolve():
        raise ValueError("Objects365 image root changed since preparation")
    if Path(object_entry["label_root"]).resolve() != Path(args.objects_image_root).resolve().parents[1] / "labels" / "train":
        raise ValueError("Objects365 label root no longer matches official YOLO image/label layout")
    label_count, label_hash = label_tree_sha256(Path(object_entry["label_root"]))
    if label_count != object_entry["label_count"] or label_hash != object_entry["label_tree_sha256"]:
        raise ValueError("Objects365 prepared labels differ from lock")
    if label_count != objects["images"]:
        raise ValueError("Objects365 prepared label count differs from selected image count")
    for source, subset, manifest, image_root in (
        ("GQA", args.gqa_json, args.gqa_manifest, args.gqa_image_root),
        ("Flickr30k", args.flickr_json, args.flickr_manifest, args.flickr_image_root),
    ):
        if Path(subset).name != ("gqa_pilot10_v1.json" if source == "GQA" else "flickr_pilot10_v1.json"):
            raise ValueError(f"Wrong pilot JSON filename for {source}")
        result = verify(SimpleNamespace(subset_json=subset, manifest=manifest, source_json=None,
                                        image_root=image_root, flat_images=False, cache=None, lock_out=None))
        if result["source_name"] != source:
            raise ValueError(f"Source mismatch for {source}")
        entry = load_lock_entry(lock_path, source, subset)
        if Path(entry["image_root"]).resolve() != Path(image_root).resolve():
            raise ValueError(f"{source} image root changed since cache preparation")
        if entry["manifest_sha256"] != sha256_file(manifest):
            raise ValueError(f"Manifest SHA256 mismatch for {source}")
        verify_cache_file(Path(subset).with_suffix(".cache"), entry["expected_image_count"],
                          entry["expected_instance_count"], True, entry["image_root"],
                          selected_image_files(subset, entry["image_root"]))
    objects_yaml = inside_pilot(args.objects_yaml)
    if not objects_yaml.is_file() or "names: {}" in objects_yaml.read_text(encoding="utf-8"):
        raise ValueError("Rendered Objects365 pilot YAML required")
    if objects_yaml != Path(object_entry["yaml_path"]).resolve() or sha256_file(objects_yaml) != object_entry["yaml_sha256"]:
        raise ValueError("Objects365 rendered YAML path or SHA256 mismatch")
    runs_root = inside_pilot(Path(__file__).resolve().parents[1] / "runs")
    run_name = f"{'smoke' if args.smoke else 'main'}_batch_{args.physical_batch}"
    if (runs_root / run_name).exists():
        raise FileExistsError(f"Refusing to reuse pilot training output directory: {runs_root / run_name}")
    return lock_path, objects_yaml, lock, runs_root, run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-batch", type=int, help="Required: selected after 32/16/8 GPU smoke")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--objects-json", required=True)
    parser.add_argument("--objects-manifest", required=True)
    parser.add_argument("--objects-image-root", required=True)
    parser.add_argument("--objects-yaml", required=True)
    parser.add_argument("--gqa-json", required=True)
    parser.add_argument("--gqa-manifest", required=True)
    parser.add_argument("--gqa-image-root", required=True)
    parser.add_argument("--flickr-json", required=True)
    parser.add_argument("--flickr-manifest", required=True)
    parser.add_argument("--flickr-image-root", required=True)
    parser.add_argument("--mobileclip-weights", required=True)
    parser.add_argument("--smoke", action="store_true", help="Later GPU trial only: one epoch on 0.1%% of pilot data")
    args = parser.parse_args()
    lock_path, objects_yaml, lock, runs_root, run_name = preflight(args)
    from ultralytics import YOLOE
    from ultralytics.nn.tasks import guess_model_scale
    from ultralytics.utils import yaml_load
    from pilot_trainer import PilotYOLOESegTrainerFromScratch
    from pilot_text import install_pilot_text
    install_pilot_text(lock["text_artifacts"])
    model_path = "yoloe-v8s-seg.yaml"
    scale = guess_model_scale(model_path)
    cfg_dir = Path(__file__).resolve().parents[3] / "ultralytics" / "cfg"
    defaults = yaml_load(cfg_dir / "default.yaml")
    extends = yaml_load(cfg_dir / f"{scale}_train.yaml")
    if not all(key in defaults for key in extends):
        raise ValueError("Official scale overrides contain unknown keys")
    PilotYOLOESegTrainerFromScratch.lock_path = str(lock_path)
    data = {"train": {"yolo_data": [str(objects_yaml)], "grounding_data": [
        {"source_name": "Flickr30k", "img_path": str(Path(args.flickr_image_root).resolve()), "json_file": str(Path(args.flickr_json).resolve())},
        {"source_name": "GQA", "img_path": str(Path(args.gqa_image_root).resolve()), "json_file": str(Path(args.gqa_json).resolve())}]},
        "val": {"yolo_data": ["lvis.yaml"]}}
    kwargs = dict(extends)
    kwargs.update(dict(data=data, trainer=PilotYOLOESegTrainerFromScratch, task="segment", imgsz=640,
                       batch=args.physical_batch, epochs=1 if args.smoke else 30,
                       nbs=128, close_mosaic=2, optimizer="AdamW", lr0=0.002,
                       warmup_bias_lr=0.0, weight_decay=0.05, momentum=0.9,
                       workers=4, seed=0, deterministic=True, device=0,
                       text_model="mobileclip:blt",
                       fraction=0.001 if args.smoke else 1.0,
                       val=not args.smoke, save=not args.smoke, plots=False,
                       project=str(runs_root), name=run_name, exist_ok=False))
    # The pinned MobileCLIP constructor resolves mobileclip_blt.pt from cwd.
    os.chdir(Path(lock["text_artifacts"]["mobileclip_checkpoint"]).parent)
    if args.smoke:
        try:
            run_smoke_trainer(YOLOE(model_path), kwargs)
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            traceback.print_exc()
            print(f"Smoke CUDA OOM; dedicated retry exit code {CUDA_OOM_EXIT_CODE}", file=sys.stderr)
            return CUDA_OOM_EXIT_CODE
        return 0
    YOLOE(model_path).train(**kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
