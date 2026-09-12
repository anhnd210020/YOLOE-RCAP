"""Stream COCO-like JSON into a deterministic, image-level pilot subset.

Requires the small pure-Python package ijson on the preparation host. It is
deliberately not installed here. All output paths are confined to the pilot tree.
"""
import argparse
import hashlib
import heapq
import json
from decimal import Decimal
from pathlib import Path

PILOT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = ("final_mixed_train_no_coco", "final_flickr_separateGT_train")


def inside_pilot(path):
    path = Path(path).resolve()
    if path != PILOT_ROOT and PILOT_ROOT not in path.parents:
        raise ValueError(f"Output must be inside {PILOT_ROOT}: {path}")
    return path


def items(path, prefix):
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("Install the small pure-Python dependency ijson on the preparation host") from exc
    with open(path, "rb") as stream:
        yield from ijson.items(stream, prefix, use_float=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(value):
    return str(value)


def selection_row(image, pilot_id, seed, source_name):
    image_id = canonical_id(image["id"])
    file_name = image["file_name"]
    if "|" in file_name or "|" in image_id:
        raise ValueError("'|' in image_id or file_name makes the selection key ambiguous")
    raw = f"{pilot_id}|{seed}|{source_name}|{image_id}|{file_name}"
    return (hashlib.sha256(raw.encode("utf-8")).hexdigest(), image_id, file_name)


class _MaxRow:
    __slots__ = ("row",)

    def __init__(self, row):
        self.row = row

    def __lt__(self, other):
        return self.row > other.row


def selected_rows(images, count, fraction, pilot_id, seed, source_name):
    # A max-heap of size K avoids holding all image metadata in memory.
    k = int(Decimal(str(fraction)) * count)
    heap = []
    for image in images:
        row = selection_row(image, pilot_id, seed, source_name)
        if len(heap) < k:
            heapq.heappush(heap, _MaxRow(row))
        elif k and row < heap[0].row:
            heapq.heapreplace(heap, _MaxRow(row))
    return sorted(entry.row for entry in heap)


def manifest_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(list(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def write_array(stream, records):
    first = True
    count = 0
    for record in records:
        if not first:
            stream.write(",\n")
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        first = False
        count += 1
    return count


def build(args):
    source = Path(args.input_json).resolve(strict=True)
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if args.source_name in ("GQA", "Flickr30k"):
        expected = "gqa_pilot10_v1.json" if args.source_name == "GQA" else "flickr_pilot10_v1.json"
        if not args.dry_run and Path(args.output_json).name != expected:
            raise ValueError(f"Grounding subset must be named {expected}")
    count = 0
    seen = set()
    for image in items(source, "images.item"):
        image_id = canonical_id(image["id"])
        if image_id in seen:
            raise ValueError(f"Duplicate source image ID: {image_id}")
        seen.add(image_id)
        count += 1
    del seen
    rows = selected_rows(items(source, "images.item"), count, args.fraction, args.pilot_id, args.seed, args.source_name)
    selected = {row[1] for row in rows}
    report = {"version": "v1", "pilot_id": args.pilot_id, "source_name": args.source_name,
              "fraction": args.fraction, "seed": args.seed, "source_image_count": count,
              "selected_image_count": len(rows), "selection_manifest_sha256": manifest_digest(rows),
              "source_sha256": sha256_file(source),
              "selected_images": [{"sha256_key": h, "image_id": i, "file_name": f} for h, i, f in rows]}
    if args.dry_run:
        print(json.dumps({key: value for key, value in report.items() if key != "selected_images"}, indent=2))
        return report
    output = inside_pilot(args.output_json)
    manifest = inside_pilot(args.manifest_out)
    for path in (output, manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if output == manifest or source == output or source == manifest:
        raise ValueError("Input, output, and manifest paths must differ")
    metadata = {}
    for key in ("info", "licenses", "categories", "type"):
        values = list(items(source, key))
        if values:
            metadata[key] = values[0]
    represented = set()

    def selected_annotations():
        for ann in items(source, "annotations.item"):
            if canonical_id(ann["image_id"]) in selected:
                if args.source_name == "Objects365v1" and not ann.get("iscrowd") and ann.get("segmentation"):
                    represented.add(ann["category_id"])
                yield ann

    with open(output, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("{\n")
        for key, value in metadata.items():
            stream.write(json.dumps(key) + ":" + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + ",\n")
        stream.write('"images":[\n')
        image_count = write_array(stream, (im for im in items(source, "images.item") if canonical_id(im["id"]) in selected))
        stream.write('],\n"annotations":[\n')
        annotation_count = write_array(stream, selected_annotations())
        stream.write("]\n}\n")
    if image_count != len(rows):
        raise RuntimeError("Selected image count changed during streaming")
    report["selected_annotation_count"] = annotation_count
    report["subset_sha256"] = sha256_file(output)
    if args.source_name == "Objects365v1":
        categories = metadata.get("categories", [])
        if not categories:
            raise ValueError("Objects365 source has no categories")
        missing_classes = sorted({cat["id"] for cat in categories} - represented)
        report["class_coverage"] = {"represented": len(represented), "total": len(categories),
                                    "missing_category_ids": missing_classes,
                                    "status": "FAIL" if missing_classes else "PASS"}
    with open(manifest, "x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "selected_images"}, indent=2))
    if args.source_name == "Objects365v1" and report["class_coverage"]["missing_category_ids"]:
        raise ValueError("Objects365 class coverage failure; deterministic subset was not repaired")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-name", required=True, choices=("Objects365v1", "GQA", "Flickr30k"))
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--manifest-out")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot-id", default="yoloe-v8s-10pct-v1")
    parser.add_argument("--dry-run", "--scan", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and (not args.output_json or not args.manifest_out):
        parser.error("--output-json and --manifest-out are required unless --dry-run")
    build(args)


if __name__ == "__main__":
    main()
