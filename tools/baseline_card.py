"""Write results/baseline/tv12.json: what is needed to rerun and to compare against the baseline model, in one file.

    python tools/baseline_card.py            # rewrite the record from model.pth and the files it points to
    python tools/baseline_card.py --check    # exit 1 when the record no longer matches the files (also run by pytest)

Everything is read from files already in the repository: the checkpoint's own record (seed, settings, torch version,
Valid / Test metrics), the manifest, the stored head data, the blind-test score files and the development comparison
(tools/compare_on_dev.py). Nothing is trained or scored here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/baseline/tv12.json"
DEV = REPO / "results/baseline/tv12_dev.json"
FILES = ["model.pth", "dataset/manifest.csv", "experiments/new_vegetables/v7/head_data.npz",
         "experiments/sweetpotato_prompts/dev_features.npz", "requirements.txt"]
BLIND = ["results/t3_test", "results/t4_test", "results/t5_test", "results/t6_test", "results/t7_test", "results/t8_test", "results/t9_test"]
METRICS = ("sample_count", "accuracy", "macro_f1", "species_accuracy", "condition_accuracy", "rotten_recall", "fresh_recall")


def sha256(path: str) -> str:
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def last_commit(path: str) -> str:
    done = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=REPO, capture_output=True, text=True)
    return done.stdout.strip()


def build() -> dict:
    ck = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    clip_head, new_head = ck["clip_head"], ck["new_vegetable_head"]
    splits = Counter(r["split"] for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")))
    dev = json.load(open(DEV))["systems"]["TV-12"]["_all_labelled"]
    return {
        "version": ck["version"],
        "files_sha256": {path: sha256(path) for path in FILES},
        "data": {"manifest": "dataset/manifest.csv", "split_column": "split", "rows_per_split": dict(sorted(splits.items())),
                 "split_made_by": "dataset/tools/build_dataset_v3.py (rebuild check: tools/rebuild_and_compare.py)"},
        "vegetable_model": {"architecture": ck["architecture"], "seed": ck["seed"], "best_epoch": ck["best_epoch"],
                            "epochs_completed": ck["epochs_completed"], "selection": ck["selection"],
                            "hyperparameters": ck["hyperparameters"], "pretrained_source": ck["pretrained_source"],
                            "command": "python src/" + ck["watermark_round"]["vegetable_model"] + " --manifest dataset/manifest.csv --out OUT "
                                       "--arch efficientnet_b0 --num-classes 10 --mode finetune --epochs 30 --patience 8 "
                                       f"--threads {ck['hyperparameters']['threads']}",
                            "metrics": {s: {k: ck["metrics"][s][k] for k in METRICS} for s in ("val", "test")}},
        "clip_head": {"clip": clip_head["clip"], "trained_on": clip_head["head"],
                      "command": "python src/" + ck["watermark_round"]["clip_head"] + " --model CANDIDATE.pth --extra-photos LVIS --extra-train EXTRA.csv --clip-check-only",
                      "photos": "not in the repository (licences): tools/rebuild_head_photos.py recovers 4,591 of them; "
                                "experiments/root_and_tomato (B0') shows how close the rebuilt head comes"},
        "new_vegetable_head": {"command": "python src/train_new_veg_head.py --model CANDIDATE.pth --data experiments/new_vegetables/v7/head_data.npz --tau 0.8",
                               "trained_on": new_head["head"], "text_prompts": new_head["prompts"]},
        "decision_settings": {"vegetable_model_temperature": ck["calibration"]["temperature"],
                              "combined_temperature": clip_head["temperature"],
                              "not_five_threshold": clip_head["unknown_threshold"], "text_evidence_alpha": clip_head["text_evidence"]["alpha"],
                              "new_vegetable_text_alpha": new_head["alpha"], "new_vegetable_switch_probability": new_head["switch_probability"],
                              "not_eight_threshold": new_head["unknown_threshold"], "hold_back_below": 0.9},
        "environment": {"python": "3.11", "torch": ck["runtime"]["torch_version"], "device": ck["hyperparameters"]["device"],
                        "packages": "requirements.txt (exact versions)"},
        "commits": {path: last_commit(path) for path in ("model.pth", "dataset/manifest.csv", "src/decision.py")},
        "development_baseline": {"file": "results/baseline/tv12_dev.json", "made_by": "python tools/compare_on_dev.py results/baseline/tv12_dev.json",
                                 "all_labelled": {k: v for k, v in dev.items() if k != "classes"}},
        "blind_tests_scored_once": {path: {f.name: sha256(f"{path}/{f.name}") for f in sorted((REPO / path).glob("scores*.json"))}
                                    for path in BLIND},
        "blind_tests_note": "T3-T9 are used up: never score a candidate on them again (their files are kept as the record).",
    }


def main():
    record = build()
    if "--check" in sys.argv:
        stored = json.load(open(OUT, encoding="utf-8"))
        stored.pop("commits"); record.pop("commits")     # history is not available in shallow checkouts
        different = [key for key in record if record[key] != stored.get(key)]
        if different:
            raise SystemExit(f"results/baseline/tv12.json is out of date: {different} (rerun tools/baseline_card.py)")
        print("results/baseline/tv12.json matches the files")
        return
    json.dump(record, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
