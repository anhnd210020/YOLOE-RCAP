"""Verify a real pilot subset and its selection manifest; optionally lock a cache."""
import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from make_subset import canonical_id, items, manifest_digest, selection_row, sha256_file


def verify(args):
    subset = Path(args.subset_json).resolve(strict=True)
    manifest_path = Path(args.manifest).resolve(strict=True)
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("pilot_id") != "yoloe-v8s-10pct-v1":
        raise ValueError("Wrong pilot_id in manifest")
    if report.get("fraction") != 0.1:
        raise ValueError("Pilot fraction must be 0.1")
    if type(report.get("seed")) is not int or report["seed"] != 0:
        raise ValueError("Pilot seed must be 0")
    if report.get("source_name") not in ("Objects365v1", "GQA", "Flickr30k"):
        raise ValueError("Unknown pilot source_name")
    source_count = report.get("source_image_count")
    selected_count = report.get("selected_image_count")
    if type(source_count) is not int or source_count < 0 or type(selected_count) is not int:
        raise ValueError("Manifest image counts must be integers")
    if selected_count != int(Decimal("0.1") * source_count):
        raise ValueError("Selected image count is not floor(0.1 * source image count)")
    if report["subset_sha256"] != sha256_file(subset):
        raise ValueError("Subset SHA256 does not match manifest")
    if args.source_json and report["source_sha256"] != sha256_file(args.source_json):
        raise ValueError("Source SHA256 does not match manifest")
    rows = []
    ids = set()
    filenames = set()
    image_count = 0
    missing = []
    for image in items(subset, "images.item"):
        image_id = canonical_id(image["id"])
        if image_id in ids:
            raise ValueError(f"Duplicate image ID: {image_id}")
        ids.add(image_id)
        image_count += 1
        rows.append(selection_row(image, report["pilot_id"], report["seed"], report["source_name"]))
        name = image["file_name"]
        if report["source_name"] == "Objects365v1":
            if name in filenames:
                raise ValueError(f"Duplicate file_name: {name}")
            filenames.add(name)
        relative = Path(name).name if getattr(args, "flat_images", False) else name
        if args.image_root and not (Path(args.image_root) / relative).is_file():
            missing.append(name)
    if missing:
        raise ValueError(f"Missing selected images ({len(missing)}), first: {missing[:5]}")
    rows.sort()
    expected = [(r["sha256_key"], r["image_id"], r["file_name"]) for r in report["selected_images"]]
    if rows != expected or manifest_digest(rows) != report["selection_manifest_sha256"]:
        raise ValueError("Selected image rows or deterministic manifest hash mismatch")
    if image_count != report["selected_image_count"]:
        raise ValueError("Selected image count mismatch")
    annotation_count = 0
    represented = set()
    for ann in items(subset, "annotations.item"):
        if canonical_id(ann["image_id"]) not in ids:
            raise ValueError(f"Orphan annotation: {ann.get('id')}")
        annotation_count += 1
        if report["source_name"] == "Objects365v1" and not ann.get("iscrowd") and ann.get("segmentation"):
            represented.add(ann["category_id"])
    if annotation_count != report["selected_annotation_count"]:
        raise ValueError("Selected annotation count mismatch")
    categories = list(items(subset, "categories.item"))
    if report["source_name"] == "Objects365v1":
        if not categories:
            raise ValueError("Objects365 subset has no categories")
        expected_categories = {cat["id"] for cat in categories}
        missing_categories = sorted(expected_categories - represented)
        coverage = {"represented": len(represented), "total": len(expected_categories), "missing_category_ids": missing_categories}
        if missing_categories:
            print(
                f"WARNING: Objects365 deterministic subset is missing categories "
                f"{missing_categories}; retained unchanged without repair."
            )
    else:
        coverage = None
    result = {"source_name": report["source_name"], "images": image_count,
              "annotations": annotation_count, "subset_sha256": report["subset_sha256"],
              "selection_manifest_sha256": report["selection_manifest_sha256"], "class_coverage": coverage}
    if args.cache:
        if not args.image_root:
            raise ValueError("--cache requires --image-root for strict file verification")
        from pilot_grounding import selected_image_file_counts, verify_cache_file
        cache = verify_cache_file(args.cache, image_count, None, True, args.image_root,
                                  expected_file_counts=selected_image_file_counts(subset, args.image_root))
        result["cache_images"] = cache["images"]
        result["cache_instances"] = cache["instances"]
    if args.lock_out:
        raise ValueError("Dataset locks are written by pilot_grounding.py, prepare_objects365.py, and prepare_text_embeddings.py")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-json")
    parser.add_argument("--image-root")
    parser.add_argument("--flat-images", action="store_true", help="Objects365 hardlinked images are flat")
    parser.add_argument("--cache")
    parser.add_argument("--lock-out")
    args = parser.parse_args()
    print(json.dumps(verify(args), indent=2))


if __name__ == "__main__":
    main()
