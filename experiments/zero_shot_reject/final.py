"""Final scoring of PREREG.md section 4: the alpha chosen on development photos (dev_results.json) against TV-11,
once on the any-photo set v2, the new internet test 241, Test 300 and new-vegetable tests 1-4. TV-11 (alpha 0) must give
the recorded numbers (pipeline check). Uses the features cached for the watermark round's final scoring (record script).

    python experiments/zero_shot_reject/final.py   -> final_once.json
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

import importlib.util

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, W  # noqa: E402
from model_utils import CLASS_NAMES  # noqa: E402


def _load(name, path):  # both experiment folders have an evaluate.py / final.py, so load them by path
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_zs = _load("zs_evaluate", HERE / "evaluate.py")
_wm = _load("wm_final", HERE.parent / "watermark/final.py")
decide, evidence, text_embeddings = _zs.decide, _zs.evidence, _zs.text_embeddings
any_photo, photo_sets = _wm.any_photo, _wm.photo_sets


def score(ck, alpha, threshold, tau, sets, F, B, E):
    ck["new_vegetable_head"]["switch_probability"] = tau
    out = {}
    for name, (paths, labels) in sets.items():
        idx, unk, _, _ = decide(ck, B[name], F[name], E[name], alpha, threshold)
        r = {"index": idx, "unknown": unk}
        if name == "any_photo_v2":
            out[name] = any_photo(r, labels); continue
        y = np.array([CLASS_NAMES.index(c) for c in labels])
        if name.startswith("new_veg_test"):
            kind = (idx // 2 == y // 2) & ~unk
            out[name] = {"n": len(y), "kind_right_and_kept": int(kind.sum()), "rejected": int(unk.sum()),
                         "sweetpotato_rotten": [int(kind[y == 13].sum()), int((y == 13).sum())]}
            continue
        right = idx == y
        out[name] = {"n": len(y), "accuracy": int(right.sum()), "kept_correct": int((right & ~unk).sum()), "turned_away": int(unk.sum()),
                     "to_sweet_potato": collections.Counter(CLASS_NAMES[a] for a, q, u in zip(y, idx, unk) if not u and q in (12, 13) and a < 10),
                     "tomato_rotten": f"{int((right & ~unk)[y == 5].sum())}/{int((y == 5).sum())}"}
    out["new_veg_tests_total"] = sum(out[f"new_veg_test{k}"]["kind_right_and_kept"] for k in range(1, 5))
    out["sweetpotato_rotten_total"] = [sum(out[f"new_veg_test{k}"]["sweetpotato_rotten"][i] for k in range(1, 5)) for i in (0, 1)]
    return out


def main():
    dev = json.load(open(HERE / "dev_results.json")); alpha = dev["chosen"]; assert alpha, "no candidate chosen"
    ck = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    veg, other = text_embeddings()
    sets = photo_sets(); F = dict(np.load(W / "final_clip.npz")); B = dict(np.load(W / "final_b_WC.npz"))
    E = {n: evidence(F[n], veg, other) for n in sets}
    base = dev["0.0"]; cand = dev[str(alpha)]
    res = {"alpha": alpha, "TV-11": score(ck, 0.0, base["threshold"], base["tau"], sets, F, B, E)}
    t = res["TV-11"]
    assert (t["test_300"]["accuracy"], t["test_300"]["kept_correct"]) == (287, 284), t["test_300"]
    assert (t["new_test_241"]["accuracy"], t["new_test_241"]["kept_correct"]) == (199, 189), t["new_test_241"]
    assert t["any_photo_v2"]["good_old_categories"] == 205 and t["new_veg_tests_total"] == 176
    res["candidate"] = c = score(ck, alpha, cand["threshold"], cand["tau"], sets, F, B, E)
    rules = {"any-photo old 247 good >= 210": c["any_photo_v2"]["good_old_categories"] >= 210,
             "any-photo new 6 good >= 5": c["any_photo_v2"]["good_new_veg"] >= 5,
             "internet 241 accuracy >= 197": c["new_test_241"]["accuracy"] >= 197,
             "internet 241 kept >= 187": c["new_test_241"]["kept_correct"] >= 187,
             "Test accuracy >= 284": c["test_300"]["accuracy"] >= 284, "Test kept >= 281": c["test_300"]["kept_correct"] >= 281,
             "new-vegetable tests 1-4 >= 173": c["new_veg_tests_total"] >= 173}
    res["rules"] = rules; res["adopt"] = all(rules.values())
    json.dump(res, open(HERE / "final_once.json", "w"), indent=1, ensure_ascii=False, default=dict)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
