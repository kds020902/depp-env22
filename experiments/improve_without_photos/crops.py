"""Part 3 of PREREG.md: look at five crops (four corners and the centre, 2/3 of the width and height) of every photo. When the
system answers "fresh X" for the whole photo, the answer becomes "rotten X" if, for some crop, the CLIP head's share of
"rotten X" within X's two classes is t or more. Nothing else changes (no rotten -> fresh, no other kind, no turning away).

    python experiments/improve_without_photos/crops.py BASE_MODEL OUT.json

Crop features are cached per development set (working folder wm/crops_<set>.npz).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3WS_LISTS", (os.environ.get("DEPP_WORK", "/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad") + "/t5/lists"))
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import W, clip_features, dev_sets, system  # noqa: E402
from model_utils import CLASS_NAMES, clip_head_logits, open_image  # noqa: E402

spec = importlib.util.spec_from_file_location("t3ws_evaluate", HERE.parent / "t3_weak_spots/evaluate.py")
t3ws = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3ws)

THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
FIVE_DEV = ("valid", "odev", "o160", "asian37")


def five_crops(image):
    w, h = image.size; cw, ch = round(w * 2 / 3), round(h * 2 / 3)
    boxes = [(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch), ((w - cw) // 2, (h - ch) // 2)]
    return [image.crop((x, y, x + cw, y + ch)) for x, y in boxes]


def crop_features(name: str, paths: list[str]) -> np.ndarray:
    cache = W / f"crops_{name}.npz"
    if cache.exists():
        return np.load(cache)["f"]
    from clip_model import ClipModel
    clip = ClipModel("cpu"); out = []
    with torch.inference_mode():
        for i in range(0, len(paths), 8):
            crops = [c for p in paths[i:i + 8] for c in five_crops(open_image(p))]
            out.append(clip.image_features(crops).cpu().numpy().reshape(-1, 5, 512))
    f = np.concatenate(out); np.savez(cache, f=f)
    return f


def answers(ck, b, f, fc, t):
    """Indices and unknown flags of the system, with the crop rule applied at threshold t (None = no rule)."""
    r = system(ck, b, f); index, unknown = r["index"].copy(), r["unknown"]
    if t is None:
        return index, unknown
    z = clip_head_logits(torch.tensor(fc.reshape(-1, 512)), ck)[:, :10].reshape(len(fc), 5, 10)
    for i in np.where(~unknown & (index < 10) & (index % 2 == 0))[0]:
        pair = z[i, :, index[i]:index[i] + 2].softmax(1)[:, 1]
        if float(pair.max()) >= t:
            index[i] += 1
    return index, unknown


def score(ck, sets, F, B, FC, dev, t):
    out = {}
    for name in FIVE_DEV:
        index, unknown = answers(ck, B[name], F[name], FC[name], t); y = sets[name][1]
        out[name] = int(((index == y) & ~unknown).sum())
        out[name + "_per_class"] = {CLASS_NAMES[c]: f"{int(((index == c) & ~unknown)[y == c].sum())}/{int((y == c).sum())}"
                                     for c in range(10) if (y == c).any()}
        fresh, rotten = (y % 2 == 0), (y % 2 == 1)
        out[name + "_rotten_called_fresh"] = int((rotten & ~unknown & (index == y - 1)).sum())
        out[name + "_fresh_called_rotten"] = int((fresh & ~unknown & (index == y + 1)).sum())
    out["five_dev_total"] = sum(out[n] for n in FIVE_DEV)
    rows, Fn, Bn, FCn = dev
    index, unknown = answers(ck, Bn, Fn, FCn, t); per = {}
    for c in sorted({x["class_name"] for x in rows}):
        m = np.array([x["class_name"] == c for x in rows])
        if c == "rotten_fruit":
            good = unknown[m]
        elif c.startswith("sweetpotato"):
            good = (index[m] // 2 == CLASS_NAMES.index(c) // 2) & ~unknown[m]
        else:
            good = (index[m] == CLASS_NAMES.index(c)) & ~unknown[m]
        per[c] = [int(good.sum()), int(m.sum())]
    out["new_dev"] = per
    out["weak_total"] = sum(per.get(c, [0])[0] for c in t3ws.WEAK)
    out["pair_total"] = sum(per.get(c, [0])[0] for c in t3ws.PAIR)
    return out


def main():
    base_path = Path(sys.argv[1]); ck = torch.load(base_path, map_location="cpu", weights_only=True)
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz"))
    FC = {n: crop_features(n, sets[n][0]) for n in FIVE_DEV}
    rows, Fn, Bn = t3ws.new_dev("WC", base_path)
    dev = (rows, Fn, Bn, crop_features("t5dev", [r["path"] for r in rows]))
    res = {"base": str(base_path.name), "none": score(ck, sets, F, B, FC, dev, None)}
    base = res["none"]
    for t in THRESHOLDS:
        c = res[f"t{t}"] = score(ck, sets, F, B, FC, dev, t)
        d = lambda n: c[n] - base[n]
        # coco400 / new-vegetable Valid / old dev -> new are unchanged by construction (only fresh X -> rotten X)
        c["checks"] = {"Valid kept >= -3": d("valid") >= -3, "internet 221 kept >= -3": d("odev") >= -3, "160 kept >= -3": d("o160") >= -3,
                       "Asian 37 kept >= -2": d("asian37") >= -2, "pair classes >= -2": c["pair_total"] - base["pair_total"] >= -2,
                       "everyday 400 / new-vegetable Valid / old dev to new": True}
        c["all_kept"] = all(c["checks"].values())
        print(f"t={t} five-dev {c['five_dev_total']} (base {base['five_dev_total']}) weak {c['weak_total']} (base {base['weak_total']}) "
              f"rules {c['all_kept']}", flush=True)
    ok = [t for t in THRESHOLDS if res[f"t{t}"]["all_kept"]]
    best = max(ok, key=lambda t: (res[f"t{t}"]["five_dev_total"], res[f"t{t}"]["weak_total"], t)) if ok else None
    gain = res[f"t{best}"]["five_dev_total"] - base["five_dev_total"] if best else None
    res.update(chosen=best, five_dev_gain=gain, verdict=None if best is None else ("improvement (>= +5)" if gain >= 5 else "small effect (< +5)"))
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("chosen", "five_dev_gain", "verdict")}))


if __name__ == "__main__":
    main()
