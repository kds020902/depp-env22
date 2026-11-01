"""Development checks of PREREG.md: add alpha x CLIP text evidence e to the not-five rejection score of TV-11.

    e = logsumexp(100 cos(photo, words that are not the five vegetables)) - logsumexp(100 cos(photo, the five vegetables))

The rejection threshold is refitted by the same rule (95% of Valid + 192 real-scene vegetable crops pass) and tau of the
new-vegetable head by its rule. Uses cached CLIP features / vegetable-model logits of the working folder (record script).

    python experiments/zero_shot_reject/evaluate.py   -> dev_results.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import OLD_DEV, REPO, S, TAU_GRID, W, dev_sets  # noqa: E402
from model_utils import (  # noqa: E402
    clip_head_logits, clip_unknown_score, combined_logits, combined_probabilities, new_vegetable_decision,
)

ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
VEG_WORDS = {"cucumber": "cucumber", "potato": "potato", "tomato": "tomato", "bellpepper": "bell pepper", "carrot": "carrot"}


def text_embeddings():
    """(five-vegetable words, other words) text embeddings, cached."""
    cache = W / "zs_text.npz"
    if cache.exists():
        d = np.load(cache)
        return d["veg"], d["other"]
    import open_clip
    from clip_model import FRUITS, OTHER_THINGS, OTHER_VEGETABLES
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="datacomp_xl_s13b_b90k"); tok = open_clip.get_tokenizer("ViT-B-16")

    def enc(prompts):
        with torch.inference_mode():
            t = torch.nn.functional.normalize(model.encode_text(tok(prompts)), dim=-1).mean(0)
        return torch.nn.functional.normalize(t, dim=0).numpy()
    four = lambda w: [f"a photo of a {w}.", f"a photo of a fresh {w}.", f"a photo of a rotten {w}.", f"a photo of a moldy {w}."]
    veg = np.stack([enc(four(w)) for w in VEG_WORDS.values()])
    other = np.stack([enc(four(w)) for w in {**FRUITS, **OTHER_VEGETABLES}] + [enc([f"a photo of {w}."]) for w in OTHER_THINGS])
    np.savez(cache, veg=veg, other=other)
    return veg, other


def evidence(f: np.ndarray, veg: np.ndarray, other: np.ndarray) -> np.ndarray:
    from scipy.special import logsumexp
    return logsumexp(100 * f @ other.T, 1) - logsumexp(100 * f @ veg.T, 1)


def rotten_fruit_features() -> dict[str, np.ndarray]:
    pool = list(csv.DictReader(open(S / "arb/pool/pool.csv", encoding="utf-8"))); pf = np.load(S / "arb/pool/feats.npz")["f"]
    keep = set(np.load(S / "arb/pool/keep_idx.npy").tolist())
    inf = np.load(S / "arb/inat/features.npz"); idx = {str(n): i for i, n in enumerate(inf["ids"])}
    return {"kaggle_fruit": pf[[i for i, r in enumerate(pool) if i in keep and r["group"] == "kaggle_fruit"]],
            "inat_rotten_fruit": inf["f"][[idx[i] for i in json.load(open(S / "arb/pool/inat_train_keep.json"))]],
            "openverse_rotten_fruit": np.load(S / "arb/ov_train_fruit.npz")["f"]}


def decide(ck, b, f, e, alpha, threshold):
    logits, feats = torch.tensor(b), torch.tensor(f)
    clip_logits = clip_head_logits(feats, ck)
    probabilities = combined_probabilities(combined_logits(logits, clip_logits, ck), ck)
    unknown = (clip_unknown_score(clip_logits, ck) + alpha * torch.tensor(e)) > threshold
    _, index, still_unknown, switched = new_vegetable_decision(probabilities, unknown, feats, ck)
    return index.numpy(), still_unknown.numpy(), switched.numpy(), unknown.numpy()


def main():
    ck = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    veg, other = text_embeddings()
    sets = dev_sets(); F = dict(np.load(W / "clip_cache.npz")); B = dict(np.load(W / "b_WC.npz"))
    for name, X in rotten_fruit_features().items():
        sets[name] = (None, None); F[name] = X; B[name] = np.zeros((len(X), 10), np.float32)  # rejection does not use B
    E = {n: evidence(F[n], veg, other) for n in F}
    base_score = lambda n: clip_unknown_score(clip_head_logits(torch.tensor(F[n]), ck), ck).numpy()
    out = {}
    for alpha in ALPHAS:
        pooled = np.concatenate([base_score(n) + alpha * E[n] for n in ("valid", "lvis")])
        threshold = float(np.quantile(pooled, 0.95))
        head = ck["new_vegetable_head"]; keep = head["switch_probability"]; grid = {}
        for tau in TAU_GRID:  # the V5-V7 rule: photos the five-vegetable system answered, moved to a new vegetable
            head["switch_probability"] = tau
            grid[tau] = sum(int((~decide(ck, B[n], F[n], E[n], alpha, threshold)[3] &
                                 decide(ck, B[n], F[n], E[n], alpha, threshold)[2]).sum()) for n in OLD_DEV)
        head["switch_probability"] = min([t for t, m in grid.items() if m <= 1] or [TAU_GRID[-1]])
        res = {"threshold": threshold, "tau": head["switch_probability"], "tau_grid": grid}
        for name in sets:
            idx, unk, sw, five_unk = decide(ck, B[name], F[name], E[name], alpha, threshold)
            y = sets[name][1]
            if y is None:
                res[name] = {"turned_away": int(unk.sum()), "n": len(unk), "to_new_vegetable": int(((idx >= 10) & ~unk).sum())}
            elif name == "newval":
                kind = (idx // 2 == y // 2) & ~unk
                res[name] = {"kind_right_kept": int(kind.sum()), "n": int(len(y)),
                             "sweetpotato_rotten": f"{int(kind[y == 13].sum())}/{int((y == 13).sum())}"}
            else:
                right = idx == y
                res[name] = {"accuracy": int(right.sum()), "kept_correct": int((right & ~unk).sum()), "n": int(len(y)),
                             "to_new_vegetable": int(((idx >= 10) & ~unk).sum())}
        res["rotten_fruit_turned_away"] = sum(res[n]["turned_away"] for n in ("kaggle_fruit", "inat_rotten_fruit", "openverse_rotten_fruit"))
        res["old_dev_to_new"] = sum(res[n]["to_new_vegetable"] for n in OLD_DEV)
        head["switch_probability"] = keep
        out[str(alpha)] = res
        print(alpha, json.dumps({k: v for k, v in res.items() if k != "tau_grid"}, ensure_ascii=False), flush=True)
    base = out["0.0"]; rules = {}
    for a in ALPHAS[1:]:
        r = out[str(a)]; d = lambda k, f: r[k][f] - base[k][f]
        checks = {"Valid kept >= -3": d("valid", "kept_correct") >= -3,
                  "internet 221 accuracy >= -3": d("odev", "accuracy") >= -3, "internet 221 kept >= -3": d("odev", "kept_correct") >= -3,
                  "160 kept >= -3": d("o160", "kept_correct") >= -3, "Asian 37 kept >= -2": d("asian37", "kept_correct") >= -2,
                  "everyday 400 turned away >= -5": d("coco400", "turned_away") >= -5,
                  "new-vegetable Valid >= -2": d("newval", "kind_right_kept") >= -2,
                  "old-vegetable dev to new <= +1": r["old_dev_to_new"] - base["old_dev_to_new"] <= 1}
        rules[str(a)] = {"checks": checks, "pass": all(checks.values()),
                         "rotten_fruit_gain": r["rotten_fruit_turned_away"] - base["rotten_fruit_turned_away"]}
    ok = [a for a in ALPHAS[1:] if rules[str(a)]["pass"]]
    best = max(ok, key=lambda a: (rules[str(a)]["rotten_fruit_gain"], -a)) if ok else None
    chosen = best if best is not None and rules[str(best)]["rotten_fruit_gain"] >= 20 else None
    out.update(rules=rules, chosen=chosen)
    json.dump(out, open(HERE / "dev_results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"rules": rules, "chosen": chosen}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
