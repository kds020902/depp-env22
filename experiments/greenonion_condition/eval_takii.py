"""Field-disease preliminary evaluation of the experimental green onion condition (fixed threshold, nothing tuned here).

    python experiments/greenonion_condition/eval_takii.py PHOTO_DIR

PHOTO_DIR holds the Takii atlas photos listed in takii_manifest.csv (copyrighted: kept outside the repository).
Labels are by eye (Claude). Two numbers per group:
- state score only: share above the threshold (the score alone, whatever the system answers);
- whole web page: share where the page shows "이상 징후 후보", i.e. the system answers green onion, not held back,
  not turned away, and the score is above the threshold (app.answer's rule).
Writes takii_eval.json next to this file.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

# sheet index (order of takii_manifest.csv) -> label; every other index is excluded for the reason given
DISEASED = {0, 5, 6, 8, 9, 11, 12, 21, 23, 24, 25, 28, 29, 30, 32, 33, 34, 36, 37, 44, 48, 49, 50, 52, 53, 56, 57, 60, 61}
NORMAL_LOOKING = {4, 10, 16, 20, 22, 40}   # look healthy; health is not confirmed by the source
EXCLUDE_REASON = "not opened as an image (GIF or not a photo) or not used"


def main():
    photo_dir = Path(sys.argv[1])
    import app
    from images import open_image
    from labels import SPECIES_NAMES
    from predict import Classifier
    system = Classifier(REPO / "model.pth", torch.device("cpu"))
    hint = np.load(app.GREENONION_HINT)
    th = float(hint["threshold"])
    rows = list(csv.DictReader(open(HERE / "takii_manifest.csv", encoding="utf-8")))
    seen, per_photo = set(), []
    for i, r in enumerate(rows):
        group = "diseased" if i in DISEASED else "normal_looking" if i in NORMAL_LOOKING else None
        rec = {"index": i, "file": r["file"], "disease_page": r["disease"], "sha256": r["sha256"], "group": group}
        if group is None:
            per_photo.append({**rec, "excluded": EXCLUDE_REASON}); continue
        if r["sha256"] in seen:
            per_photo.append({**rec, "excluded": "same file as an earlier photo (sha256)"}); continue
        seen.add(r["sha256"])
        try:
            photo = open_image(photo_dir / r["file"])
        except Exception as e:
            per_photo.append({**rec, "excluded": f"could not open: {type(e).__name__}"}); continue
        res = system.classify(photo, app.HOLD_BACK_BELOW)
        f = system.clip.image_features([photo]).cpu().numpy().reshape(-1)
        score = float((hint["spoiled"] @ f).mean() - (hint["fresh"] @ f).mean())
        is_green = (not res["unknown"]) and SPECIES_NAMES[res["index"] // 2] == "greenonion"
        page_flag = is_green and not res["held"] and score >= th
        per_photo.append({**rec, "score": round(score, 4), "score_flag": score >= th, "answer_species": None if res["unknown"] else SPECIES_NAMES[res["index"] // 2],
                          "unknown": res["unknown"], "held": res["held"], "page_flag": page_flag})
    summary = {}
    for g in ("diseased", "normal_looking"):
        used = [p for p in per_photo if p["group"] == g and "excluded" not in p]
        summary[g] = {"n": len(used), "score_flagged": sum(p["score_flag"] for p in used), "page_flagged": sum(p["page_flag"] for p in used),
                      "answered_green_onion_not_held": sum(p["answer_species"] == "greenonion" and not p["held"] for p in used)}
    out = {"threshold": th, "summary": summary, "photos": per_photo}
    (HERE / "takii_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
