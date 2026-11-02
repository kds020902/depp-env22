"""Part 2 of PREREG.md: TV-12 with alpha x CLIP zero-shot class scores added to the CLIP head's 10 class columns (candidates Z).

    python experiments/improve_without_photos/text_mix.py OUT.json

For every alpha the two-model temperature is refitted on Valid and the not-five threshold so that 95% of Valid + 192 LVIS
crops pass (TV-12's rules); the vegetable model, the text-evidence check and the new-vegetable head stay TV-12's. Every
candidate is scored on the development photos with the 8 rules of experiments/t3_weak_spots (no blind test set).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3WS_LISTS", os.environ["DEPP_WORK"] + "/t5/lists")
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, W, clip_features, dev_sets, kept_right_per_class  # noqa: E402
from model_utils import (  # noqa: E402
    clip_head_logits, clip_unknown_score, combined_logits, fit_temperature,
)

spec = importlib.util.spec_from_file_location("t3ws_evaluate", HERE.parent / "t3_weak_spots/evaluate.py")
t3ws = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3ws)

ALPHAS = (0.1, 0.25, 0.5, 1.0)
FIVE_DEV = ("valid", "odev", "o160", "asian37")


def candidate(tv12: dict, alpha: float, F: dict, B: dict, y_valid: np.ndarray) -> dict:
    ck = copy.deepcopy(tv12); head = ck["clip_head"]; nv = tv12["new_vegetable_head"]
    head.update(alpha=float(alpha), text_embeddings=nv["text_embeddings"][:10].clone(), prompts=list(nv["prompts"][:10]),
                text_mix_note="alpha x CLIP zero-shot class scores (V7 prompts of the 10 classes), experiments/improve_without_photos")
    fv = torch.tensor(F["valid"])
    head["temperature"] = fit_temperature(combined_logits(torch.tensor(B["valid"]), clip_head_logits(fv, ck), ck), torch.tensor(y_valid))
    scores = [clip_unknown_score(clip_head_logits(torch.tensor(F[n]), ck), ck, torch.tensor(F[n])) for n in ("valid", "lvis")]
    head["unknown_threshold"] = float(torch.quantile(torch.cat(scores), 0.95))
    ck["version"] = f"candidate Z{int(round(alpha * 100)):03d} (experiments/improve_without_photos)"
    return ck


def main():
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz"))
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    res = {"TV-12": t3ws.score("TV-12", REPO / "model.pth", "WC")}
    res["TV-12"]["five_dev_total"] = sum(res["TV-12"][n]["kept_correct"] for n in FIVE_DEV)
    res["TV-12"]["per_class"] = kept_right_per_class(tv12, FIVE_DEV)
    # reference only (not used to choose): the text prompts alone, argmax over the 10 classes, no turning away
    text = tv12["new_vegetable_head"]["text_embeddings"][:10]
    res["text_prompts_alone_accuracy"] = {n: f"{int(((torch.tensor(F[n]) @ text.T).argmax(1).numpy() == sets[n][1]).sum())}/{len(sets[n][1])}" for n in FIVE_DEV}
    for alpha in ALPHAS:
        tag = f"Z{int(round(alpha * 100)):03d}"
        ck = candidate(tv12, alpha, F, B, sets["valid"][1]); torch.save(ck, W / f"{tag}.pth")
        c = res[tag] = t3ws.score(tag, W / f"{tag}.pth", "WC")
        c["five_dev_total"] = sum(c[n]["kept_correct"] for n in FIVE_DEV)
        c["per_class"] = kept_right_per_class(ck, FIVE_DEV)
        c["checks"] = t3ws.dev_rules(c, res["TV-12"]); c["all_kept"] = all(c["checks"].values())
        c["temperature"], c["unknown_threshold"] = ck["clip_head"]["temperature"], ck["clip_head"]["unknown_threshold"]
        print(tag, "five-dev", c["five_dev_total"], "weak", c["weak_total"], "all rules", c["all_kept"], flush=True)
    ok = [t for t in res if t.startswith("Z") and res[t]["all_kept"]]
    best = max(ok, key=lambda t: (res[t]["five_dev_total"], res[t]["weak_total"], -int(t[1:]))) if ok else None
    gain = res[best]["five_dev_total"] - res["TV-12"]["five_dev_total"] if best else None
    res["chosen"] = best; res["five_dev_gain"] = gain
    res["verdict"] = None if best is None else ("improvement (>= +5)" if gain >= 5 else "small effect (< +5)")
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("text_prompts_alone_accuracy", "chosen", "five_dev_gain", "verdict")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
