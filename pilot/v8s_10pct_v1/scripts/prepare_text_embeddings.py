"""Build pilot-only MobileCLIP text artifacts from the three locked pilot sources.

The official scripts are read as reference but never executed. This command is
for the later preparation server; importing it performs no encoding or GPU work.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from make_subset import inside_pilot, sha256_file
from pilot_grounding import load_lock_entry, read_lock, selected_image_file_counts, selected_image_files, verify_cache_file, write_lock_sections
from prepare_objects365 import label_tree_sha256

PILOT_THRESHOLD = max(1, round(100 * 0.10))
LVIS_LIST_SHA256 = "1b7c59904861001d0c0d71a024cee80617a4a74f565b38e9ab6c60d63123c4df"
LVIS_JSON_SHA256 = "02301f6ccd89d1ee3d35112cb57d000c3396f34e4073066c90b2c1fbf47b55ce"


def verify_evaluation(entry):
    if entry.get("name") != "LVIS v1 full minival" or entry.get("expected_image_count") != 4809:
        raise ValueError("Evaluation lock must identify full LVIS v1 minival (4809 images)")
    for path_key, hash_key, known in (("minival_list", "minival_list_sha256", LVIS_LIST_SHA256),
                                      ("minival_annotation", "minival_annotation_sha256", LVIS_JSON_SHA256)):
        path = Path(entry[path_key])
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(f"Locked LVIS input missing or not absolute: {path}")
        actual = sha256_file(path)
        if actual != known or entry[hash_key] != known:
            raise ValueError(f"LVIS baseline SHA256 mismatch: {path}")
    with open(entry["minival_list"], encoding="utf-8") as stream:
        count = sum(bool(line.strip()) for line in stream)
    if count != 4809:
        raise ValueError(f"Full LVIS minival must contain 4809 entries, found {count}")
    return entry


def collect_names(yaml_path, cache_paths):
    import numpy as np
    from ultralytics.utils import yaml_load
    data = yaml_load(str(yaml_path), append_filename=True)
    names = set()
    for name in data["names"].values():
        for alias in name.split("/"):
            alias = alias.strip()
            if not alias:
                raise ValueError("Blank Objects365 class alias")
            names.add(alias)
    frequency = Counter()
    for cache_path in cache_paths:
        labels = np.load(str(cache_path), allow_pickle=True)
        for label in labels:
            for prompts in label["texts"]:
                for prompt in prompts:
                    prompt = prompt.strip()
                    if not prompt:
                        raise ValueError("Blank grounding text")
                    names.add(prompt)
                    frequency[prompt] += 1
    return sorted(names), sorted(name for name, count in frequency.items() if count >= PILOT_THRESHOLD)


def encode_names(names, checkpoint, device, batch):
    import torch
    from ultralytics.nn.text_model import build_text_model
    if not names:
        raise ValueError("Cannot encode an empty vocabulary")
    old_cwd = Path.cwd()
    try:
        # Pinned MobileCLIP passes pretrained='mobileclip_blt.pt' relative to cwd.
        os.chdir(checkpoint.parent)
        model = build_text_model("mobileclip:blt", device=device)
    finally:
        os.chdir(old_cwd)
    if model.training:
        raise RuntimeError("Official MobileCLIP text encoder must be in eval mode")
    with torch.inference_mode():
        tokens = model.tokenize(names)
        features = torch.cat([model.encode_text(chunk).cpu() for chunk in tokens.split(batch)], dim=0)
    return features


def prepare(args):
    import torch
    output = inside_pilot(args.output_root)
    lock_path = inside_pilot(args.lock)
    lock = read_lock(lock_path)
    if "text_artifacts" in lock or "evaluation" in lock:
        raise FileExistsError("Text or evaluation lock section already exists")
    if not all(source in lock for source in ("Objects365v1", "GQA", "Flickr30k")):
        raise ValueError("Objects365v1, GQA, and Flickr30k must be locked first")
    objects = lock["Objects365v1"]
    if objects.get("class_coverage_status") not in {"PASS", "WARN"}:
        raise ValueError("Objects365 class coverage status must be PASS or documented WARN before text preparation")
    for path_key, hash_key in (("subset_json", "subset_sha256"), ("manifest", "manifest_sha256")):
        path = Path(objects[path_key]) if path_key != "manifest" else Path(args.objects_manifest)
        if not path.is_file() or sha256_file(path) != objects[hash_key]:
            raise ValueError(f"Locked Objects365 {path_key} missing or changed")
    yaml_path = Path(objects["yaml_path"])
    if not yaml_path.is_file() or sha256_file(yaml_path) != objects["yaml_sha256"]:
        raise ValueError("Locked Objects365 YAML missing or changed")
    label_count, label_hash = label_tree_sha256(Path(objects["label_root"]))
    if label_count != objects["label_count"] or label_hash != objects["label_tree_sha256"]:
        raise ValueError("Locked Objects365 prepared labels changed")
    cache_paths = []
    for source, subset in (("GQA", args.gqa_json), ("Flickr30k", args.flickr_json)):
        entry = load_lock_entry(lock_path, source, subset)
        cache_path = Path(subset).with_suffix(".cache")
        verify_cache_file(cache_path, entry["expected_image_count"], entry["expected_instance_count"], True,
                          entry["image_root"], selected_image_files(subset, entry["image_root"]),
                          selected_image_file_counts(subset, entry["image_root"]))
        cache_paths.append(cache_path)
    checkpoint = Path(args.mobileclip_weights).resolve(strict=True)
    if checkpoint.name != "mobileclip_blt.pt":
        raise ValueError("Official MobileCLIP requires a file named mobileclip_blt.pt")
    evaluation = {"name": "LVIS v1 full minival", "expected_image_count": 4809,
                  "minival_list": str(Path(args.lvis_minival_list).resolve(strict=True)),
                  "minival_list_sha256": LVIS_LIST_SHA256,
                  "minival_annotation": str(Path(args.lvis_annotation).resolve(strict=True)),
                  "minival_annotation_sha256": LVIS_JSON_SHA256}
    verify_evaluation(evaluation)
    categories_path = output / "global_grounding_neg_cat.json"
    train_path = output / "train_label_embeddings.pt"
    negatives_path = output / "global_grounding_neg_embeddings.pt"
    for path in (categories_path, train_path, negatives_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite pilot text artifact: {path}")
    all_names, negative_names = collect_names(yaml_path, cache_paths)
    if len(negative_names) < 80:
        raise ValueError(f"Pilot threshold {PILOT_THRESHOLD} yields only {len(negative_names)} global negatives; official RandomLoadText needs at least 80")
    output.mkdir(parents=True, exist_ok=True)
    with open(categories_path, "x", encoding="utf-8", newline="\n") as stream:
        json.dump(negative_names, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    all_features = encode_names(all_names, checkpoint, args.device, args.batch)
    torch.save(dict(zip(all_names, all_features)), train_path)
    negative_features = encode_names(negative_names, checkpoint, args.device, args.batch)
    torch.save(negative_features, negatives_path)
    text_entry = {"text_model": "mobileclip:blt", "mobileclip_checkpoint": str(checkpoint),
                  "mobileclip_checkpoint_sha256": sha256_file(checkpoint),
                  "train_label_embeddings": str(train_path), "train_label_embeddings_sha256": sha256_file(train_path),
                  "global_negative_categories": str(categories_path),
                  "global_negative_categories_sha256": sha256_file(categories_path),
                  "global_negative_embeddings": str(negatives_path),
                  "global_negative_embeddings_sha256": sha256_file(negatives_path),
                  "global_negative_threshold": PILOT_THRESHOLD,
                  "global_negative_count": len(negative_names), "train_label_count": len(all_names)}
    from pilot_text import verify_text_artifacts
    verify_text_artifacts(text_entry, checkpoint)
    write_lock_sections(lock_path, {"text_artifacts": text_entry, "evaluation": evaluation})
    print(json.dumps({"train_label_count": len(all_names), "global_negative_count": len(negative_names),
                      "threshold": PILOT_THRESHOLD, "output": str(output)}, indent=2))


def main():
    repo_parent = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--objects-manifest", required=True)
    parser.add_argument("--gqa-json", required=True)
    parser.add_argument("--flickr-json", required=True)
    parser.add_argument("--mobileclip-weights", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lvis-minival-list", default=str(repo_parent / "datasets/lvis/minival.txt"))
    parser.add_argument("--lvis-annotation", default=str(repo_parent / "datasets/lvis/annotations/lvis_v1_minival.json"))
    parser.add_argument("--device", default="cpu", help="Later server may explicitly select cuda")
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
