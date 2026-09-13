"""Account for GQA annotations filtered by the released YOLOE cache generator."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ultralytics.utils.ops import xyxy2xywhn

from make_subset import inside_pilot, items, sha256_file


def audit(args):
    subset = inside_pilot(args.subset_json)
    cache = subset.with_suffix(".cache")
    image_report_path = inside_pilot(args.image_report)
    output = inside_pilot(args.report_out)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite GQA cache audit: {output}")
    image_report = json.loads(image_report_path.read_text(encoding="utf-8"))
    if (image_report["source_name"] != "GQA"
            or image_report["status"] != "WARN_UPSTREAM_OFFICIAL"
            or image_report["subset_sha256"] != sha256_file(subset)
            or image_report["metadata_repaired"] is not False
            or image_report["records_dropped"] != 0):
        raise ValueError("GQA source/JPEG audit or subset identity changed")

    labels = np.load(str(cache), allow_pickle=True)
    images = list(items(subset, "images.item"))
    if len(images) != len(labels) or len(images) != image_report["selected_records"]:
        raise ValueError("Official GQA cache dropped image records")
    annotation_counts = Counter(ann["image_id"] for ann in items(subset, "annotations.item"))
    affected = [
        {"image_id": image["id"], "file_name": image["file_name"],
         "subset_annotations": annotation_counts[image["id"]], "cache_instances": len(label["cls"])}
        for image, label in zip(images, labels)
        if annotation_counts[image["id"]] != len(label["cls"])
    ]
    affected_ids = {row["image_id"] for row in affected}
    annotations = defaultdict(list)
    for ann in items(subset, "annotations.item"):
        if ann["image_id"] in affected_ids:
            annotations[ann["image_id"]].append(ann)
    image_by_id = {image["id"]: image for image in images}
    reasons = Counter()
    for row in affected:
        image = image_by_id[row["image_id"]]
        width, height = int(image["width"]), int(image["height"])
        category_ids = {}
        boxes = []
        filtered = []
        for ann in annotations[row["image_id"]]:
            if ann["iscrowd"]:
                reason = "iscrowd"
            else:
                box = np.array(ann["bbox"], dtype=np.float64)
                box[2] += box[0]
                box[3] += box[1]
                box = xyxy2xywhn(box, w=float(width), h=float(height), clip=True)
                if box[2] <= 0 or box[3] <= 0:
                    reason = "zero_box_after_json_clipping"
                else:
                    caption = " ".join(image["caption"][start:end]
                                       for start, end in ann["tokens_positive"]).lower().strip()
                    if not caption:
                        reason = "empty_caption_span"
                    else:
                        if caption not in category_ids:
                            category_ids[caption] = len(category_ids)
                        full_box = [category_ids[caption]] + box.tolist()
                        if full_box in boxes:
                            reason = "duplicate_class_and_normalized_box"
                        else:
                            boxes.append(full_box)
                            continue
            filtered.append({"annotation_id": ann["id"], "reason": reason})
            reasons[reason] += 1
        if len(boxes) != row["cache_instances"] or len(filtered) != row["subset_annotations"] - row["cache_instances"]:
            raise ValueError(f"Unexplained GQA cache instance count for image_id {row['image_id']}")
        row["filtered_by_official_generator"] = filtered

    selected_annotations = sum(annotation_counts.values())
    cache_instances = sum(len(label["cls"]) for label in labels)
    if (selected_annotations != image_report["selected_annotations"]
            or selected_annotations - cache_instances != sum(reasons.values())):
        raise ValueError("GQA cache instance accounting differs from subset annotations")
    result = {
        "status": "PASS_OFFICIAL_CACHE_SEMANTICS",
        "subset_sha256": image_report["subset_sha256"],
        "cache_path": str(cache), "cache_sha256": sha256_file(cache),
        "source_image_report": str(image_report_path),
        "source_image_report_sha256": sha256_file(image_report_path),
        "selected_image_records": len(images), "image_records_dropped": 0,
        "subset_annotation_count": selected_annotations,
        "official_cache_instance_count": cache_instances,
        "official_annotation_filter_count": sum(reasons.values()),
        "filter_reasons": dict(reasons), "affected_image_records": affected,
        "metadata_repaired": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--image-report", required=True)
    parser.add_argument("--report-out", required=True)
    report = audit(parser.parse_args())
    print(json.dumps({key: value for key, value in report.items() if key != "affected_image_records"}, indent=2))


if __name__ == "__main__":
    main()
