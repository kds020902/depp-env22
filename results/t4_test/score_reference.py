"""Reference scoring of the blind test set T4 at the user's request (2026-11-02): TV-12 and every candidate of
experiments/t3_weak_spots (K1-K4, L1, L2), once each. No candidate was chosen on the development photos, so this is not an
adoption step; it records how the candidates do on photos nobody had looked at. After this run T4 counts as used.

    python results/t4_test/score_reference.py OUT.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import score as t4  # noqa: E402  (seal check, feature extraction and per-class scoring of the one-time script)
from common import W  # noqa: E402

VEG_MODEL = {"TV-12": "WC", "K1": "WC", "K3": "WC", "L1": "WC", "K2": "K2", "K4": "K2", "L2": "L2"}


def main():
    digest = t4.seal_ok()
    rows = list(csv.DictReader(open(HERE / "t4_test.csv", encoding="utf-8")))
    paths = [str(t4.IMG / f"{r['id']}.jpg") for r in rows]
    F, B_wc = t4.features(paths, REPO / "model.pth")
    B = {"WC": B_wc}
    for tag in ("K2", "L2"):
        model, ck = t4.load_model(W / f"{tag}.pth", torch.device("cpu")); tf = t4.make_transform(t4.input_size(ck))
        with torch.inference_mode():
            B[tag] = torch.cat([model(torch.stack([tf(t4.open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
    res = {"seal": digest, "n": len(rows), "note": "reference only; no candidate was chosen on development photos"}
    for tag, vm in VEG_MODEL.items():
        ck = torch.load(REPO / "model.pth" if tag == "TV-12" else W / f"{tag}.pth", map_location="cpu", weights_only=True)
        res[tag] = t4.t4_scores(ck, B[vm], F, rows)
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for tag in VEG_MODEL:
        r = res[tag]
        print(tag, "weak", r["weak_total"], "pair", r["pair_total"], {c: f"{v['good']}/{v['n']}" for c, v in r["per_class"].items()})


if __name__ == "__main__":
    main()
