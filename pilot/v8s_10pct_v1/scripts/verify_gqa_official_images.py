"""Validate selected GQA JPEGs while preserving released YOLOE metadata."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from make_subset import canonical_id, inside_pilot, items, sha256_file
from verify_subset import verify


# (file_name, official processed JSON width/height, official Stanford JPEG width/height).
# These are warnings only when the complete selected record matches the hashed source.
KNOWN_UPSTREAM_DIMENSIONS = {
    "597913": ("285761.jpg", (1024, 768), (612, 612)),
    "687015": ("61530.jpg", (1024, 768), (1000, 586)),
    "687017": ("61530.jpg", (1024, 768), (1000, 586)),
    "769131": ("498098.jpg", (1024, 768), (1024, 683)),
    "769134": ("498098.jpg", (1024, 768), (1024, 683)),
    "770291": ("61564.jpg", (1024, 768), (1024, 1022)),
    "814367": ("285743.jpg", (1024, 768), (1024, 683)),
    "827542": ("286093.jpg", (1024, 768), (1024, 685)),
    "869127": ("150417.jpg", (1024, 768), (1024, 683)),
    "869131": ("150417.jpg", (1024, 768), (1024, 683)),
}


def classify_dimensions(image, actual):
    """Return a known official warning, or fail for any unapproved mismatch."""
    image_id = canonical_id(image["id"])
    expected = (int(image["width"]), int(image["height"]))
    if actual == expected:
        if image_id in KNOWN_UPSTREAM_DIMENSIONS:
            raise ValueError(f"Known GQA mismatch unexpectedly disappeared: {image_id}")
        return None
    observed = (image["file_name"], expected, actual)
    if KNOWN_UPSTREAM_DIMENSIONS.get(image_id) != observed:
        raise ValueError(f"Unknown or changed GQA dimensions: image_id={image_id}, {observed}")
    return {
        "image_id": image["id"],
        "file_name": image["file_name"],
        "json_width": expected[0], "json_height": expected[1],
        "jpeg_width": actual[0], "jpeg_height": actual[1],
        "status": "WARN_UPSTREAM_OFFICIAL",
    }


def audit(args):
    subset = inside_pilot(args.subset_json)
    manifest = inside_pilot(args.manifest)
    source = Path(args.source_json).resolve(strict=True)
    root = Path(args.image_root).resolve(strict=True)
    report_path = inside_pilot(args.report_out)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite GQA audit: {report_path}")

    verified = verify(SimpleNamespace(subset_json=subset, manifest=manifest,
                                      source_json=source, image_root=root,
                                      flat_images=False, cache=None, lock_out=None))
    if verified["source_name"] != "GQA":
        raise ValueError("This audit accepts only the deterministic GQA subset")

    selected = {}
    files = Counter()
    for image in items(subset, "images.item"):
        image_id = canonical_id(image["id"])
        if image_id in selected:
            raise ValueError(f"Duplicate GQA image_id: {image_id}")
        selected[image_id] = image
        files[image["file_name"]] += 1
    source_seen = set()
    for image in items(source, "images.item"):
        image_id = canonical_id(image["id"])
        if image_id in selected:
            if image_id in source_seen or image != selected[image_id]:
                raise ValueError(f"Selected GQA record differs from official processed source: {image_id}")
            source_seen.add(image_id)
    if source_seen != selected.keys():
        raise ValueError(f"Selected image_id missing from official source: {len(selected.keys() - source_seen)}")

    dimensions = {}
    missing = []
    decode_failures = []
    for index, name in enumerate(sorted(files), 1):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"GQA path escapes selected image root: {name}")
        if not path.is_file():
            missing.append(name)
            continue
        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise ValueError(f"Expected JPEG, got {image.format}")
                image.verify()
            with Image.open(path) as image:
                image.load()
                dimensions[name] = image.size
        except Exception as exc:
            decode_failures.append({"file_name": name, "error": str(exc)})
        if index % 5000 == 0:
            print(f"GQA_JPEGS_CHECKED {index}/{len(files)}", flush=True)
    if missing or decode_failures:
        raise ValueError(f"GQA selected JPEG failure: missing={missing[:5]} ({len(missing)}), "
                         f"decode={decode_failures[:5]} ({len(decode_failures)})")

    warnings = []
    for image in selected.values():
        warning = classify_dimensions(image, dimensions[image["file_name"]])
        if warning is not None:
            warnings.append(warning)
    warnings.sort(key=lambda row: row["image_id"])
    if {canonical_id(row["image_id"]) for row in warnings} != KNOWN_UPSTREAM_DIMENSIONS.keys():
        raise ValueError("The approved set of GQA upstream dimension warnings changed")
    report = {
        "source_name": "GQA", "status": "WARN_UPSTREAM_OFFICIAL",
        "subset_json": str(subset), "subset_sha256": sha256_file(subset),
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "official_processed_source": str(source), "official_processed_source_sha256": sha256_file(source),
        "image_root": str(root), "selected_records": len(selected),
        "selected_annotations": verified["annotations"], "unique_physical_jpegs": len(files),
        "repeated_file_name_references": len(selected) - len(files),
        "missing": 0, "decode_failures": 0, "duplicate_image_ids": 0,
        "dimension_mismatch_records": len(warnings),
        "dimension_mismatch_jpegs": len({row["file_name"] for row in warnings}),
        "metadata_repaired": False, "records_dropped": 0,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--report-out", required=True)
    report = audit(parser.parse_args())
    print(json.dumps({key: value for key, value in report.items() if key != "warnings"}, indent=2))


if __name__ == "__main__":
    main()
