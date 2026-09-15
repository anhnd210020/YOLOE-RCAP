"""Later-server full LVIS minival Fixed AP for a completed YOLOE-v8-S pilot checkpoint.

Import the validated baseline bbox-only validator unchanged. This module performs
no inference, CUDA work, or output creation on import or during --help.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import types

from make_subset import inside_pilot, sha256_file
from pilot_grounding import read_lock
from prepare_text_embeddings import verify_evaluation
from pilot_text import verify_text_artifacts

PILOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
LVIS = ROOT.parent / "datasets" / "lvis"
BASELINE_VALIDATOR = ROOT / "reproducibility/yoloe_v8_sml_20260905/scripts/bbox_inference.py"
BASELINE_VALIDATOR_SHA256 = "325ffb30892f732a9b906c18cd8d7ce922fd9f248a1c46c584460dba724dd15e"
FIXED_AP = ROOT / "tools/eval_fixed_ap.py"
FIXED_AP_MODULE = "tools.eval_fixed_ap"
EXPECTED = 4809
AP_METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf")
AR_METRICS = {"all": "AR", "s": "ARs", "m": "ARm", "l": "ARl"}
PRIMARY_AP_METRICS = ("AP", "APr", "APc", "APf")


def preflight(args):
    """Verify immutable inputs and the output reservation before loading a model."""
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError("--checkpoint must be a file")
    checkpoint_sha256 = sha256_file(checkpoint)
    lock = read_lock(inside_pilot(args.lock))
    if lock.get("version") != "v1" or lock.get("pilot_id") != "yoloe-v8s-10pct-v1":
        raise ValueError("Wrong pilot dataset lock")
    evaluation = verify_evaluation(lock["evaluation"])
    verify_text_artifacts(lock["text_artifacts"])
    if Path(evaluation["minival_list"]).resolve() != (LVIS / "minival.txt").resolve():
        raise ValueError("Locked minival list does not match official lvis.yaml layout")
    annotation = (LVIS / "annotations/lvis_v1_minival.json").resolve()
    if Path(evaluation["minival_annotation"]).resolve() != annotation:
        raise ValueError("Locked minival annotation does not match official lvis.yaml layout")
    paths = sorted(str((LVIS / row.strip()).resolve()) for row in
                   Path(evaluation["minival_list"]).read_text(encoding="utf-8").splitlines() if row.strip())
    if len(paths) != EXPECTED or len(set(paths)) != EXPECTED or any(not Path(p).is_file() for p in paths):
        raise ValueError("Full minival list must resolve to 4809 distinct existing images")
    ids = [int(Path(p).stem) for p in paths]
    if len(set(ids)) != EXPECTED:
        raise ValueError("Minival image IDs are not unique")
    gt = json.loads(annotation.read_text(encoding="utf-8"))
    if {int(image["id"]) for image in gt["images"]} != set(ids) or len(gt["categories"]) != 1203:
        raise ValueError("Minival annotation image IDs or categories differ from expected full LVIS")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        raise ValueError("--name must be a simple directory name")
    output = inside_pilot(PILOT / "runs" / "eval" / args.name)
    if output.exists():
        raise FileExistsError(f"Refusing existing evaluation output directory: {output}")
    if not BASELINE_VALIDATOR.is_file():
        raise FileNotFoundError(BASELINE_VALIDATOR)
    # The archived Linux manifest hashes LF; Windows checkout stores the same
    # Python source with CRLF. Verify the manifest's canonical LF bytes.
    canonical = BASELINE_VALIDATOR.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(canonical).hexdigest() != BASELINE_VALIDATOR_SHA256:
        raise ValueError("Validated baseline bbox evaluator source differs from its manifest")
    if not FIXED_AP.is_file():
        raise FileNotFoundError(FIXED_AP)
    return checkpoint, checkpoint_sha256, lock, annotation, paths, ids, output


def import_validated_validator():
    """Import the exact class, isolating baseline-only import side effects."""
    from ultralytics.data import dataset as dataset_module
    old_load, old_save = dataset_module.load_dataset_cache_file, dataset_module.save_dataset_cache_file
    prior_common = sys.modules.get("bbox_common")
    shim = types.ModuleType("bbox_common")
    shim.protect_inputs = lambda: None  # Baseline audit hook is tied to /workspace/baseline_logs.
    shim.LOGS = PILOT / "runs" / "eval"
    prior_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules["bbox_common"] = shim
    try:
        spec = importlib.util.spec_from_file_location("pilot_validated_bbox_inference", BASELINE_VALIDATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BBoxFixedAPValidator
    finally:
        dataset_module.load_dataset_cache_file = old_load
        dataset_module.save_dataset_cache_file = old_save
        if prior_common is None:
            sys.modules.pop("bbox_common", None)
        else:
            sys.modules["bbox_common"] = prior_common
        sys.dont_write_bytecode = prior_bytecode


@contextmanager
def pilot_cache_routing(output):
    """Keep generated dataset caches under the pilot evaluation run."""
    from ultralytics.data import dataset as dataset_module
    old_load, old_save = dataset_module.load_dataset_cache_file, dataset_module.save_dataset_cache_file

    def load_cache(path):
        existing = output / "label_cache" / Path(path).name
        return old_load(existing if existing.is_file() else path)

    def save_cache(prefix, path, data, version):
        target = output / "bbox_label_cache" / Path(path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        return old_save(prefix, target, data, version)

    dataset_module.load_dataset_cache_file = load_cache
    dataset_module.save_dataset_cache_file = save_cache
    try:
        yield
    finally:
        dataset_module.load_dataset_cache_file = old_load
        dataset_module.save_dataset_cache_file = old_save


def check_model_identity(model):
    import yaml
    from ultralytics.nn.tasks import YOLOESegModel
    from ultralytics.nn.modules.head import YOLOESegment
    network = model.model
    official = yaml.safe_load((ROOT / "ultralytics/cfg/models/v8/yoloe-v8-seg.yaml").read_text(encoding="utf-8"))
    architecture = network.yaml
    if not (model.task == "segment" and isinstance(network, YOLOESegModel)
            and isinstance(network.model[-1], YOLOESegment) and architecture.get("scale") == "s"
            and network.args.get("text_model") == "mobileclip:blt"
            and not hasattr(network, "pe") and network.stride.tolist() == [8.0, 16.0, 32.0]
            and architecture.get("scales", {}).get("s") == official["scales"]["s"]
            and architecture["backbone"] == official["backbone"] and architecture["head"] == official["head"]
            and network.model[0].conv.out_channels == 32 and len(network.model[2].m) == 1):
        raise ValueError("Checkpoint is not the official-architecture YOLOE-v8-S segmentation model")
    return {"task": model.task, "scale": "s", "text_model": network.args["text_model"],
            "parameters": sum(p.numel() for p in network.parameters())}


def fixed_ap_command(annotation, prediction):
    """Build an import-safe invocation which cannot let tools/lvis shadow lvis-api."""
    return [sys.executable, "-m", FIXED_AP_MODULE, str(annotation), str(prediction), "--type", "bbox"]


def evaluator_provenance():
    """Identify both the evaluator bytes and the checkout in which they ran."""
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                             capture_output=True, check=True).stdout.strip()
    git_branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True,
                                capture_output=True, check=True).stdout.strip()
    return {"eval_pilot_sha256": sha256_file(Path(__file__).resolve()),
            "evaluation_git_sha": git_sha, "evaluation_git_branch": git_branch or "DETACHED"}


def parse_fixed_ap(output):
    """Parse the official AP-point block and recall fractions into AP-point units."""
    clean_lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in output.splitlines()]
    copy_lines = [line.split("copypaste:", 1)[1].strip()
                  for line in clean_lines if "copypaste:" in line]
    ap_metrics = None
    for header, values in zip(copy_lines, copy_lines[1:]):
        keys = [key.strip() for key in header.split(",")]
        if "AP" not in keys:
            continue
        if len(keys) != len(set(keys)) or any(key not in AP_METRICS for key in keys):
            raise ValueError("Malformed Fixed AP copypaste metric header")
        missing = [key for key in AP_METRICS if key not in keys]
        if missing:
            raise ValueError(f"Fixed AP copypaste output is missing metrics: {', '.join(missing)}")
        value_tokens = [value.strip() for value in values.split(",")]
        if len(value_tokens) != len(keys):
            raise ValueError("Fixed AP copypaste metric/value counts differ")
        try:
            parsed = dict(zip(keys, (float(value) for value in value_tokens)))
        except ValueError as error:
            raise ValueError("Fixed AP copypaste output contains a non-numeric value") from error
        ap_metrics = {key: parsed[key] for key in AP_METRICS}
        break
    if ap_metrics is None:
        raise ValueError("Official Fixed AP tool did not emit its complete AP copypaste values")

    recall_pattern = re.compile(
        r"Average Recall\s+\(AR\)\s+@\[[^]]*\|\s*area=\s*(all|s|m|l)\s*\|[^]]*\]\s*=\s*(\S+)\s*$"
    )
    recall_metrics = {}
    for line in clean_lines:
        match = recall_pattern.search(line)
        if not match:
            continue
        key = AR_METRICS[match.group(1)]
        if key in recall_metrics:
            raise ValueError(f"Fixed AP output contains duplicate {key} values")
        try:
            recall_metrics[key] = round(float(match.group(2)) * 100, 10)
        except ValueError as error:
            raise ValueError(f"Fixed AP output contains a non-numeric {key} value") from error
    missing_recall = [key for key in AR_METRICS.values() if key not in recall_metrics]
    if missing_recall:
        raise ValueError(f"Fixed AP output is missing recall metrics: {', '.join(missing_recall)}")

    metrics = {**ap_metrics, **recall_metrics}
    invalid = [key for key, value in metrics.items()
               if not math.isfinite(value) or not 0 <= value <= 100]
    if invalid:
        raise ValueError(f"Fixed AP values are not finite percentages in [0, 100]: {', '.join(invalid)}")
    if any(key not in metrics for key in PRIMARY_AP_METRICS):
        raise ValueError("Fixed AP output omitted legacy AP/APr/APc/APf metrics")
    return metrics


def evaluate(args):
    checkpoint, checkpoint_hash, lock, annotation, paths, ids, output = preflight(args)
    provenance = evaluator_provenance()
    from ultralytics import YOLOE
    validator_class = import_validated_validator()
    model = YOLOE(str(checkpoint))
    identity = check_model_identity(model)
    prediction = output / "predictions.json"
    active = {}

    def on_start(validator):
        actual = [str(Path(path).resolve()) for path in validator.dataloader.dataset.im_files]
        if actual != paths or len(validator.data["names"]) != 1203:
            raise ValueError("Validator dataset differs from locked full LVIS minival")
        config = dict(imgsz=640, batch=1, split="minival", rect=False, conf=0.001,
                      iou=0.7, max_det=1000, half=False, save_json=True, load_vp=False, plots=False)
        if validator.args.task != "segment" or validator.device.type != "cuda":
            raise ValueError("Expected YOLOE segmentation on one CUDA GPU")
        for key, value in config.items():
            if getattr(validator.args, key) != value:
                raise ValueError(f"Fixed AP configuration changed: {key}")
        if (validator.save_dir / "predictions.json").resolve() != prediction:
            raise ValueError("Validator output path differs from reserved pilot run")
        active["validator"] = validator

    def on_end(validator):
        if validator.seen != EXPECTED or validator.visited_image_ids != ids:
            raise ValueError("Full minival image count or visit order changed")

    model.add_callback("on_val_start", on_start)
    model.add_callback("on_val_end", on_end)
    config = dict(data="ultralytics/cfg/datasets/lvis.yaml", imgsz=640, batch=1,
                  split="minival", rect=False, conf=0.001, iou=0.7, max_det=1000,
                  half=False, save_json=True, load_vp=False, device=0,
                  project=str(output.parent), name=output.name, exist_ok=False, plots=False)
    old_cwd = Path.cwd()
    try:
        # The pinned text encoder resolves mobileclip_blt.pt relative to cwd.
        os.chdir(Path(lock["text_artifacts"]["mobileclip_checkpoint"]).parent)
        with pilot_cache_routing(output):
            model.val(validator=validator_class, **config)
    finally:
        os.chdir(old_cwd)
    if "validator" not in active or not prediction.is_file() or not prediction.stat().st_size:
        raise ValueError("Full inference did not produce a bbox predictions file")
    rows = json.loads(prediction.read_text(encoding="utf-8"))
    allowed_ids = set(ids)
    if not isinstance(rows, list) or not rows or any(row.get("image_id") not in allowed_ids for row in rows):
        raise ValueError("BBox predictions are empty or contain foreign image IDs")
    command = fixed_ap_command(annotation, prediction)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    metrics = parse_fixed_ap(result.stdout + "\n" + result.stderr)
    payload = {"protocol": "full LVIS v1 minival bbox Fixed AP; validated segmentation bbox-only serializer",
               "images_evaluated": EXPECTED, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
               "model": identity, "validator_source": str(BASELINE_VALIDATOR),
               "validator_manifest_lf_sha256": BASELINE_VALIDATOR_SHA256,
               "validator_checkout_sha256": sha256_file(BASELINE_VALIDATOR), "fixed_ap_tool": str(FIXED_AP),
               "fixed_ap_tool_sha256": sha256_file(FIXED_AP),
               **provenance,
               "minival_list_sha256": lock["evaluation"]["minival_list_sha256"],
               "minival_annotation_sha256": lock["evaluation"]["minival_annotation_sha256"],
               "predictions": str(prediction), "predictions_sha256": sha256_file(prediction),
               "prediction_records": len(rows), "configuration": config, "metrics_ap_points": metrics}
    with (output / "fixed_ap_result.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Completed pilot YOLOE-v8-S segmentation checkpoint")
    parser.add_argument("--lock", required=True, help="Pilot unified dataset lock")
    parser.add_argument("--name", default="full_fixed_ap", help="New run name under pilot/runs/eval")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
