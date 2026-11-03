"""PREREG.md: candidate A (kind decided with CLIP's text knowledge) and candidate B (a rotten crop holds the answer back),
on Valid, the outside development photos and Test 300 (no longer blind: its errors were looked at). Nothing is adopted here.

    python experiments/shape_and_parts/evaluate.py experiments/shape_and_parts/dev_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
sys.path.insert(0, str(HERE.parent / "improve_without_photos"))
from common import W, clip_features, dev_sets, photo_sets  # noqa: E402
from crops import crop_features  # noqa: E402  (five crops: corners and centre, 2/3 of the photo)
from decision import Answers, clip_head_logits, decide, held_back, new_vegetable_decision  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

BETAS = (0.25, 0.5, 1.0)
THRESHOLDS = (0.8, 0.9, 0.95)
OUTSIDE = ("odev", "o160", "asian37")
SETS = ("valid",) + OUTSIDE + ("test_300",)


def kind_by_text(ck: dict, a: Answers, f: torch.Tensor, beta: float) -> Answers:
    """Candidate A: five-vegetable probabilities x (CLIP zero-shot probability of the kind)^beta, renormalised."""
    text = ck["new_vegetable_head"]["text_embeddings"][:10].float()
    z = (100 * f.float() @ text.T).softmax(1)
    kind = z.reshape(len(f), 5, 2).sum(2).repeat_interleave(2, dim=1)
    five = a.five * kind ** beta
    five = five / five.sum(1, keepdim=True)
    probabilities, index, unknown, switched = new_vegetable_decision(five, a.five_unknown, f, ck)
    return Answers(probabilities, index, probabilities[torch.arange(len(index)), index], unknown, switched, five, a.five_unknown)


def crop_alarm(ck: dict, a: Answers, crops: np.ndarray, t: float) -> np.ndarray:
    """Candidate B: answers "fresh X" for which some crop gives "rotten X" a share of t or more."""
    z = clip_head_logits(torch.tensor(crops.reshape(-1, 512)), ck)[:, :10].reshape(len(crops), 5, 10)
    index, unknown = a.index.numpy(), a.unknown.numpy()
    alarm = np.zeros(len(index), bool)
    for i in np.where(~unknown & (index < 10) & (index % 2 == 0))[0]:
        alarm[i] = float(z[i, :, index[i]:index[i] + 2].softmax(1)[:, 1].max()) >= t
    return alarm


def metrics(a: Answers, y: np.ndarray, extra_hold: np.ndarray | None = None) -> dict:
    index, unknown = a.index.numpy(), a.unknown.numpy()
    right = (index == y) & ~unknown
    held = held_back(a).numpy() | (extra_hold if extra_hold is not None else False)
    answered = ~unknown & ~held
    rotten_to_fresh = answered & (y % 2 == 1) & (index < 10) & (index % 2 == 0)
    return {"n": len(y), "kept_right": int(right.sum()), "answered": int(answered.sum()), "known": int((~unknown).sum()),
            "answered_right": int((answered & right).sum()), "confident_wrong": int((answered & ~right).sum()),
            "rotten_to_fresh_answered": int(rotten_to_fresh.sum()), "held_back": int((~unknown & held).sum())}


def pooled(rows: list[dict]) -> dict:
    return {k: sum(r[k] for r in rows) for k in rows[0]}


def main():
    ck = torch.load(Path(__file__).resolve().parents[2] / "model.pth", map_location="cpu", weights_only=True)
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz"))
    ph = photo_sets(); Fe = dict(np.load(W / "final_clip.npz")); Be = dict(np.load(W / "final_b_WC.npz"))
    data = {n: (torch.tensor(B[n]), torch.tensor(F[n]), sets[n][1], sets[n][0]) for n in ("valid",) + OUTSIDE}
    data["test_300"] = (torch.tensor(Be["test_300"]), torch.tensor(Fe["test_300"]),
                        np.array([CLASS_NAMES.index(c) for c in ph["test_300"][1]]), ph["test_300"][0])
    crops = {n: crop_features(n, data[n][3]) for n in SETS}
    base = {n: decide(data[n][0], ck, data[n][1]) for n in SETS}
    res = {"TV-12": {n: metrics(base[n], data[n][2]) for n in SETS}}
    for beta in BETAS:
        res[f"A{beta}"] = {n: metrics(kind_by_text(ck, base[n], data[n][1], beta), data[n][2]) for n in SETS}
    for t in THRESHOLDS:
        res[f"B{t}"] = {n: metrics(base[n], data[n][2], crop_alarm(ck, base[n], crops[n], t)) for n in SETS}
    for tag in res:
        res[tag]["outside"] = pooled([res[tag][n] for n in OUTSIDE])
        res[tag]["choose_total"] = res[tag]["valid"]["kept_right"] + res[tag]["outside"]["kept_right"]
    # pre-registered choices
    best_a = max((f"A{b}" for b in BETAS), key=lambda t: res[t]["choose_total"])
    gain_a = res[best_a]["choose_total"] - res["TV-12"]["choose_total"]
    cover = lambda t: res[t]["outside"]["answered"] / res[t]["outside"]["known"]
    ok_b = [f"B{t}" for t in THRESHOLDS if cover("TV-12") - cover(f"B{t}") <= 0.05]
    best_b = min(ok_b, key=lambda t: (res[t]["outside"]["rotten_to_fresh_answered"], -res[t]["outside"]["answered"])) if ok_b else None
    res["chosen"] = {"A": best_a, "A_gain": gain_a, "A_verdict": "improvement (>= +3)" if gain_a >= 3 else "small effect (< +3)", "B": best_b}
    # examples: answers that change
    examples = []
    for n in SETS:
        b0, y, paths = base[n], data[n][2], data[n][3]
        aa = kind_by_text(ck, b0, data[n][1], float(best_a[1:]))
        for i in np.where(aa.index.numpy() != b0.index.numpy())[0]:
            examples.append({"fix": "A", "set": n, "path": Path(paths[i]).name, "label": CLASS_NAMES[y[i]],
                             "before": "unknown" if b0.unknown[i] else CLASS_NAMES[int(b0.index[i])], "before_p": round(float(b0.probability[i]), 3),
                             "after": "unknown" if aa.unknown[i] else CLASS_NAMES[int(aa.index[i])], "after_p": round(float(aa.probability[i]), 3)})
        if best_b:
            alarm = crop_alarm(ck, b0, crops[n], float(best_b[1:])) & ~held_back(b0).numpy()
            for i in np.where(alarm)[0]:
                examples.append({"fix": "B", "set": n, "path": Path(paths[i]).name, "label": CLASS_NAMES[y[i]],
                                 "before": CLASS_NAMES[int(b0.index[i])], "before_p": round(float(b0.probability[i]), 3), "after": "판단 보류", "after_p": None})
    res["examples"] = examples
    res["note"] = ("Test 300 is no longer blind (its errors were looked at); reported for reference only. "
                   "examples: answers that change (A: chosen beta; B: newly held back), path = file name only.")
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for tag in [t for t in res if t not in ("chosen", "examples", "note")]:
        print(tag, {n: res[tag][n]["kept_right"] for n in SETS}, "outside answered", res[tag]["outside"]["answered"],
              "right", res[tag]["outside"]["answered_right"], "conf wrong", res[tag]["outside"]["confident_wrong"],
              "rot->fresh", res[tag]["outside"]["rotten_to_fresh_answered"], "| test answered", res[tag]["test_300"]["answered"],
              "conf wrong", res[tag]["test_300"]["confident_wrong"])
    print(json.dumps(res["chosen"], ensure_ascii=False), "examples", len(examples))


if __name__ == "__main__":
    main()
