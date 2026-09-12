"""Pilot-only GroundingDataset verification; official cache/transform logic is inherited."""
import json
import math
from pathlib import Path


def verify_cache_labels(labels, expected_images, expected_instances, strict_files, expected_root=None, expected_files=None):
    if len(labels) != expected_images:
        raise ValueError(f"Grounding cache images: expected {expected_images}, found {len(labels)}")
    seen = set()
    resolved_files = set()
    instance_count = 0
    required = {"im_file", "shape", "cls", "bboxes", "segments", "normalized", "bbox_format", "texts"}
    for index, label in enumerate(labels):
        missing_keys = required - set(label)
        if missing_keys:
            raise ValueError(f"Grounding label {index} missing keys: {sorted(missing_keys)}")
        im_file = str(label["im_file"])
        if im_file in seen:
            raise ValueError(f"Duplicate grounding im_file: {im_file}")
        seen.add(im_file)
        resolved_files.add(str(Path(im_file).resolve()))
        if strict_files and not Path(im_file).is_file():
            raise ValueError(f"Missing grounding image: {im_file}")
        if expected_root is not None:
            try:
                Path(im_file).resolve().relative_to(Path(expected_root).resolve())
            except ValueError as exc:
                raise ValueError(f"Grounding cache path is outside locked image root: {im_file}") from exc
        cls, boxes, segments = label["cls"], label["bboxes"], label["segments"]
        if len(getattr(cls, "shape", ())) != 2 or cls.shape[1] != 1:
            raise ValueError(f"Label {index} cls must be Nx1")
        if len(getattr(boxes, "shape", ())) != 2 or boxes.shape != (cls.shape[0], 4):
            raise ValueError(f"Label {index} bboxes must be Nx4 and match cls")
        if not all(math.isfinite(float(x)) for row in boxes for x in row):
            raise ValueError(f"Label {index} contains non-finite bbox")
        if label["normalized"] is not True or label["bbox_format"] != "xywh":
            raise ValueError(f"Label {index} must use normalized xywh")
        if not isinstance(segments, list) or len(segments) != cls.shape[0]:
            raise ValueError(f"Label {index} segment/label count mismatch")
        if segments and any(len(getattr(seg, "shape", ())) != 2 or seg.shape[1] != 2 for seg in segments):
            raise ValueError(f"Label {index} segments must be Nx2")
        instance_count += cls.shape[0]
    if expected_instances is not None and instance_count != expected_instances:
        raise ValueError(f"Grounding cache instances: expected {expected_instances}, found {instance_count}")
    if expected_files is not None and resolved_files != expected_files:
        raise ValueError("Grounding cache im_file set differs from selected subset image paths")
    return {"images": len(labels), "instances": instance_count}


def verify_cache_file(path, expected_images, expected_instances, strict_files=True, expected_root=None, expected_files=None):
    import numpy as np
    labels = np.load(str(path), allow_pickle=True)
    return verify_cache_labels(labels, expected_images, expected_instances, strict_files, expected_root, expected_files)


def selected_image_files(subset_json, image_root):
    from make_subset import items
    root = Path(image_root).resolve(strict=True)
    paths = set()
    for image in items(subset_json, "images.item"):
        path = (root / image["file_name"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Selected image path escapes grounding root: {image['file_name']}") from exc
        paths.add(str(path))
    return paths


def load_then_reduce_for_smoke(load_full, dataset, fraction):
    """Call the full verifying loader before any runtime-only smoke reduction."""
    labels = load_full()
    if fraction < 1.0:
        if not 0 < fraction < 1:
            raise ValueError("Smoke fraction must be in (0, 1)")
        count = max(1, round(len(labels) * fraction)) if len(labels) else 0
        labels = labels[:count]
        dataset.im_files = dataset.im_files[:count]
    return labels


def load_lock_entry(lock_path, source_name, json_file):
    lock = read_lock(lock_path)
    entry = lock[source_name]
    if Path(entry["subset_json"]).resolve() != Path(json_file).resolve():
        raise ValueError(f"Lock subset path mismatch for {source_name}")
    from make_subset import sha256_file
    if sha256_file(json_file) != entry["subset_sha256"]:
        raise ValueError(f"Lock subset SHA256 mismatch for {source_name}")
    cache_path = Path(json_file).with_suffix(".cache")
    if sha256_file(cache_path) != entry["cache_sha256"]:
        raise ValueError(f"Lock cache SHA256 mismatch for {source_name}")
    return entry


def read_lock(lock_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("version") != "v1" or lock.get("pilot_id") != "yoloe-v8s-10pct-v1":
        raise ValueError("Wrong pilot dataset lock version or ID")
    return lock


def write_lock_section(lock_path, section, entry):
    write_lock_sections(lock_path, {section: entry})


def write_lock_sections(lock_path, sections):
    from make_subset import inside_pilot
    path = inside_pilot(lock_path)
    allowed = ("Objects365v1", "GQA", "Flickr30k", "text_artifacts", "evaluation")
    lock = read_lock(path) if path.exists() else {"version": "v1", "pilot_id": "yoloe-v8s-10pct-v1"}
    for section, entry in sections.items():
        if section not in allowed:
            raise ValueError(f"Unknown lock section: {section}")
        if section in lock:
            raise FileExistsError(f"Lock section already exists: {section}")
        lock[section] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


try:
    from ultralytics.data.dataset import GroundingDataset
except ImportError:
    GroundingDataset = object  # Pure verification helpers remain CPU-only/importable.


class PilotGroundingDataset(GroundingDataset):
    def __init__(self, *args, lock_path, source_name, strict_files=True, **kwargs):
        self.pilot_entry = load_lock_entry(lock_path, source_name, kwargs["json_file"])
        self.pilot_strict_files = strict_files
        super().__init__(*args, **kwargs)

    def verify_labels(self, labels):
        entry = self.pilot_entry
        verify_cache_labels(labels, entry["expected_image_count"], entry["expected_instance_count"],
                            self.pilot_strict_files, entry["image_root"])

    def get_labels(self):
        # Official get_labels loads and verifies the entire pilot cache first.
        return load_then_reduce_for_smoke(super().get_labels, self, self.fraction)


def build_cache_and_lock(args):
    """Strictly verify images first, then invoke the unchanged official cache generator."""
    from make_subset import inside_pilot, sha256_file
    from verify_subset import verify
    subset = inside_pilot(args.subset_json)
    cache_path = subset.with_suffix(".cache")
    if cache_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {cache_path}")
    from types import SimpleNamespace
    preflight = verify(SimpleNamespace(subset_json=args.subset_json, manifest=args.manifest,
                                       source_json=None, image_root=args.image_root,
                                       flat_images=False, cache=None, lock_out=None))
    if preflight["images"] <= 0:
        raise ValueError("Empty subset")
    if preflight["source_name"] not in ("GQA", "Flickr30k"):
        raise ValueError("Cache wrapper accepts only GQA or Flickr30k")
    lock_path = inside_pilot(args.lock_out)
    if lock_path.exists() and preflight["source_name"] in read_lock(lock_path):
        raise FileExistsError(f"Grounding source already locked: {preflight['source_name']}")
    # Load the unchanged official script by path, independent of the caller's cwd.
    import importlib.util
    official = Path(__file__).resolve().parents[3] / "tools" / "generate_grounding_cache.py"
    spec = importlib.util.spec_from_file_location("official_generate_grounding_cache", official)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generate_cache = module.generate_cache
    generate_cache(str(subset), str(Path(args.image_root).resolve(strict=True)))
    image_root = str(Path(args.image_root).resolve(strict=True))
    measured = verify_cache_file(cache_path, preflight["images"], None, True, image_root,
                                 selected_image_files(subset, image_root))
    source_name = preflight["source_name"]
    entry = {
        "subset_json": str(subset), "subset_sha256": preflight["subset_sha256"],
        "manifest_sha256": sha256_file(args.manifest),
        "expected_image_count": measured["images"],
        "expected_instance_count": measured["instances"],
        "cache_sha256": sha256_file(cache_path),
        "image_root": image_root,
    }
    write_lock_section(args.lock_out, source_name, entry)
    print(json.dumps({"source": source_name, **measured, "lock": str(inside_pilot(args.lock_out))}, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--lock-out", required=True)
    args = parser.parse_args()
    args.source_json = None
    args.cache = None
    build_cache_and_lock(args)


if __name__ == "__main__":
    main()
