"""TV-12 (= ZS8: alpha 0.25, threshold -0.3204, tau 0.8) on the earlier test sets, for the record after it was adopted by
the sealed test T3 (results/t3_test). Not used for any choice. Also checks that the TV-12 checkpoint (model.pth, text
evidence inside the CLIP head) gives exactly the ZS8 decisions of evaluate.decide() on every development and test set.

    python experiments/zero_shot_reject/final_zs8.py   -> final_zs8.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, W, dev_sets, system  # noqa: E402

import importlib.util  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zs = _load("zs_evaluate", HERE / "evaluate.py")
zf = _load("zs_final", HERE / "final.py")
TV11 = W / "TV11_model.pth"   # the TV-11 checkpoint (git f18a159), kept in the working folder


def main():
    dev = json.load(open(HERE / "dev_results.json"))
    tv11 = torch.load(TV11, map_location="cpu", weights_only=True)
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    assert "text_evidence" in tv12["clip_head"] and "text_evidence" not in tv11["clip_head"]
    veg, other = zs.text_embeddings()
    alpha, threshold, tau = 0.25, dev["0.25"]["threshold"], 0.8
    # same decisions from the checkpoint as from the experiment code, on every cached set
    checked = {}
    for F, B in ((dict(np.load(W / "clip_cache.npz")), dict(np.load(W / "b_WC.npz"))),
                 (dict(np.load(W / "final_clip.npz")), dict(np.load(W / "final_b_WC.npz")))):
        for name in B:
            if name not in F:
                continue
            tv11["new_vegetable_head"]["switch_probability"] = tau
            idx, unk, _, _ = zs.decide(tv11, B[name], F[name], zs.evidence(F[name], veg, other), alpha, threshold)
            r = system(tv12, B[name], F[name])
            assert (r["index"] == idx).all() and (r["unknown"] == unk).all(), name
            checked[name] = len(idx)
    sets = zf.photo_sets(); F = dict(np.load(W / "final_clip.npz")); B = dict(np.load(W / "final_b_WC.npz"))
    E = {n: zs.evidence(F[n], veg, other) for n in sets}
    res = {"checkpoint_matches_experiment_code": checked,
           "TV-11": zf.score(tv11, 0.0, dev["0.0"]["threshold"], 0.8, sets, F, B, E),
           "TV-12": zf.score(tv11, alpha, threshold, tau, sets, F, B, E)}
    json.dump(res, open(HERE / "final_zs8.json", "w"), indent=1, ensure_ascii=False, default=dict)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
