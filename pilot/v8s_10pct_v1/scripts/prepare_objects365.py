"""Convert a verified Objects365 pilot subset to YOLO segmentation labels.

Requires numpy and the repository's existing ultralytics package at execution.
No source data is deleted, copied, or overwritten. Optional images are hardlinked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from make_subset import PILOT_ROOT, inside_pilot, items, sha256_file


def label_tree_sha256(directory):
    digest = hashlib.sha256()
    files = sorted(directory.glob("*.txt"))
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def preflight_hardlink(source_file):
    """Check same filesystem and pilot hardlink support using only tiny own files."""
    source = Path(source_file).resolve(strict=True)
    if source.stat().st_dev != PILOT_ROOT.stat().st_dev:
        raise RuntimeError("Objects365 source and pilot destination are on different filesystems; copying would be required")
    try:
        with tempfile.TemporaryDirectory(prefix="hardlink_probe_", dir=PILOT_ROOT) as directory:
            original = Path(directory) / "probe.txt"
            linked = Path(directory) / "probe_link.txt"
            original.write_bytes(b"x")
            os.link(original, linked)
            if linked.read_bytes() != b"x":
                raise RuntimeError("Pilot hardlink capability check produced unexpected content")
    except OSError as exc:
        raise RuntimeError("Pilot destination does not support hardlinks; copying would be required") from exc


def prepare(args):
    from types import SimpleNamespace
    from verify_subset import verify
    from pilot_grounding import read_lock
    root = inside_pilot(args.pilot_data_root)
    lock_path = inside_pilot(args.lock_out)
    if lock_path.exists() and "Objects365v1" in read_lock(lock_path):
        raise FileExistsError("Objects365v1 lock section already exists")
    subset = Path(args.subset_json).resolve(strict=True)
    verified = verify(SimpleNamespace(subset_json=subset, manifest=args.manifest,
                                      source_json=None, image_root=args.source_image_root,
                                      flat_images=False, cache=None, lock_out=None))
    if verified["source_name"] != "Objects365v1":
        raise ValueError("Objects365 subset required")
    labels_dir = root / "objects365" / "labels" / "train"
    images_dir = root / "objects365" / "images" / "train"
    yaml_path = root / "Objects365v1-pilot10-v1.yaml"
    if labels_dir.exists() or yaml_path.exists() or (args.materialize and images_dir.exists()):
        raise FileExistsError("Pilot labels, YAML, or image destination already exists; refusing overwrite")
    categories = sorted(items(subset, "categories.item"), key=lambda x: x["id"])
    ids = [cat["id"] for cat in categories]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("Official category_id - 1 mapping requires contiguous IDs beginning at 1")
    names = [cat["name"].strip().lower() for cat in categories]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate category names would change official converter semantics")
    image_map = {}
    basenames = set()
    stems = set()
    for im in items(subset, "images.item"):
        name = Path(im["file_name"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe image path: {name}")
        basename = name.name
        if basename in basenames:
            raise ValueError(f"Duplicate image basename: {basename}")
        basenames.add(basename)
        if name.stem in stems:
            raise ValueError(f"Duplicate image stem would overwrite label: {name.stem}")
        stems.add(name.stem)
        if im["id"] in image_map:
            raise ValueError(f"Duplicate image ID: {im['id']}")
        if im["width"] <= 0 or im["height"] <= 0:
            raise ValueError(f"Invalid image dimensions: {im['id']}")
        image_map[im["id"]] = im
    if not image_map:
        raise ValueError("No selected images")
    if args.materialize:
        if not args.source_image_root:
            raise ValueError("--materialize requires --source-image-root")
        source_root = Path(args.source_image_root).resolve(strict=True)
        missing = [im["file_name"] for im in image_map.values() if not (source_root / im["file_name"]).is_file()]
        if missing:
            raise ValueError(f"Missing source images ({len(missing)}), first: {missing[:5]}")
        preflight_hardlink(source_root / next(iter(image_map.values()))["file_name"])
    elif args.source_image_root:
        source_root = Path(args.source_image_root).resolve(strict=True)
        missing = [im["file_name"] for im in image_map.values() if not (source_root / im["file_name"]).is_file()]
        if missing:
            raise ValueError(f"Missing source images ({len(missing)}), first: {missing[:5]}")
    from ultralytics.data.converter import merge_multi_segment
    import numpy as np
    labels = {}
    represented = set()
    duplicate_count = 0
    for ann in items(subset, "annotations.item"):
        im = image_map.get(ann["image_id"])
        if im is None:
            raise ValueError(f"Orphan annotation: {ann.get('id')}")
        if ann.get("iscrowd"):
            continue
        category_id = ann["category_id"]
        if category_id not in ids:
            raise ValueError(f"Unknown category ID: {category_id}")
        polygons = ann.get("segmentation")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"Segmentation polygons required for annotation {ann.get('id')}")
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError(f"Invalid polygon in annotation {ann.get('id')}")
        if len(polygons) > 1:
            merged = merge_multi_segment(polygons)
            coords = np.concatenate(merged, axis=0)
        else:
            coords = np.array(polygons[0], dtype=np.float64).reshape(-1, 2)
        coords = (coords / np.array([im["width"], im["height"]])).reshape(-1).tolist()
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"Non-finite polygon: {ann.get('id')}")
        label = tuple([category_id - 1] + coords)
        bucket = labels.setdefault(im["id"], set())
        if label in bucket:
            duplicate_count += 1
        else:
            bucket.add(label)
            represented.add(category_id)
    missing_classes = sorted(set(ids) - represented)
    print(json.dumps({"images": len(image_map), "represented_classes": len(represented),
                      "total_classes": len(ids), "missing_category_ids": missing_classes,
                      "duplicate_labels": duplicate_count}, indent=2))
    if missing_classes:
        print(
            f"WARNING: Objects365 deterministic subset is missing categories "
            f"{missing_classes}; selected images remain unchanged."
        )
    # Create outputs after validating subset structure and source files.
    labels_dir.mkdir(parents=True, exist_ok=False)
    for image_id, im in image_map.items():
        label_path = labels_dir / (Path(im["file_name"]).stem + ".txt")
        with open(label_path, "x", encoding="utf-8", newline="\n") as stream:
            for label in sorted(labels.get(image_id, ())):
                stream.write(("%g " * len(label)).rstrip() % label + "\n")
    if args.materialize:
        images_dir.mkdir(parents=True, exist_ok=False)
        for im in image_map.values():
            src = source_root / im["file_name"]
            dst = images_dir / Path(im["file_name"]).name
            try:
                os.link(src, dst)
            except OSError as exc:
                raise RuntimeError(f"Hardlink failed for {src}; copying would be required. No copy was attempted") from exc
    missing_prepared = [im["file_name"] for im in image_map.values()
                        if not (images_dir / Path(im["file_name"]).name).is_file()]
    if missing_prepared:
        raise ValueError(f"Prepared Objects365 images missing ({len(missing_prepared)}), first: {missing_prepared[:5]}")
    root.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"path: {json.dumps(str((root / 'objects365').resolve()))}\n")
        stream.write("train: images/train\nval: null\ntest: null\nnames:\n")
        for index, name in enumerate(names):
            stream.write(f"  {index}: {json.dumps(name, ensure_ascii=False)}\n")
    label_count, label_hash = label_tree_sha256(labels_dir)
    if label_count != verified["images"]:
        raise ValueError("Prepared Objects365 label count differs from selected image count")
    from pilot_grounding import write_lock_section
    write_lock_section(args.lock_out, "Objects365v1", {
        "subset_json": str(subset), "subset_sha256": verified["subset_sha256"],
        "manifest_sha256": sha256_file(args.manifest),
        "expected_image_count": verified["images"],
        "expected_annotation_count": verified["annotations"],
        "class_coverage_status": (
            "WARN" if verified["class_coverage"]["missing_category_ids"] else "PASS"
        ), "class_coverage": verified["class_coverage"],
        "yaml_path": str(yaml_path.resolve()), "yaml_sha256": sha256_file(yaml_path),
        "image_root": str(images_dir.resolve()), "label_root": str(labels_dir.resolve()),
        "label_count": label_count, "label_tree_sha256": label_hash,
    })
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lock-out", required=True)
    parser.add_argument("--pilot-data-root", required=True)
    parser.add_argument("--source-image-root")
    parser.add_argument("--materialize", action="store_true", help="Hardlink selected images; never copy")
    args = parser.parse_args()
    print(prepare(args))


if __name__ == "__main__":
    main()
