"""experiments/sharper_eye (PREREG.md): TV-12, P1, H and HP on the development photos, T3 and T4 included (development now).
No blind test set is read.

    python experiments/sharper_eye/evaluate.py OUT.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import DEV_LISTS, S, W, dev_new, dev_rules, dev_sets, vegetable_logits  # noqa: E402
from model_utils import CLASS_NAMES, input_size, load_model, make_transform, open_image  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eye = _load("eye_evaluate", REPO / "experiments/bigger_eye/evaluate.py")
ref = _load("eye_t3_t4", REPO / "experiments/bigger_eye/t3_t4_reference.py")
dp = _load("dirty_potato", REPO / "experiments/dirty_potato/evaluate.py")
MODELS = {"TV-12": (REPO / "model.pth", "WC"), "P1": (W / "L1.pth", "WC"), "H": (W / "H.pth", "H320"), "HP": (W / "HP.pth", "H320")}


def logits_of(path: Path, paths: list[str]) -> np.ndarray:
    model, ck = load_model(path, torch.device("cpu")); tf = make_transform(input_size(ck))
    with torch.inference_mode():
        return torch.cat([model(torch.stack([tf(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()


def vegetable_outputs(path: Path, tag: str, lists: dict, rows: dict) -> dict:
    """The vegetable model's logits on every photo the comparison needs (cached per vegetable model)."""
    if tag == "WC":
        z = dict(np.load(W / "dirty_potato_t3t4.npz"))
        return {"dev": dict(np.load(W / "b_WC.npz")), "t5_dev": np.load(W / "t3ws_dev_b_WC_t5.npz")["b"],
                "test_300": np.load(W / "final_b_WC.npz")["test_300"], "t3": z["t3_b"], "t4": z["t4_b"]}
    out = {"dev": vegetable_logits(path, tag, dev_sets()), "t5_dev": dev_new(DEV_LISTS, tag, path)[2]}
    cache = W / f"sharper_{tag}.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    for key, paths in (("test_300", lists["test_300"]), ("t3", [str(S / f"t3/img/{r['id']}.jpg") for r in rows["t3"]]),
                       ("t4", [str(S / f"t4/img/{r['id']}.jpg") for r in rows["t4"]])):
        if key not in have:
            have[key] = logits_of(path, paths); np.savez(cache, **have)
        out[key] = have[key]
    return out


def ms_per_photo(path: Path, paths: list[str]) -> float:
    model, ck = load_model(path, torch.device("cpu")); tf = make_transform(input_size(ck))
    x = torch.stack([tf(open_image(p)) for p in paths])
    with torch.inference_mode():
        model(x[:4]); t = time.time()
        for i in range(len(x)):
            model(x[i:i + 1])
    return round(1000 * (time.time() - t) / len(x), 1)


def main():
    torch.set_num_threads(4)
    lists = json.load(open(W / "eye_lists.json")); feats = eye.load("B0")
    rows = {"t3": list(csv.DictReader(open(REPO / "results/t3_test/t3_test.csv", encoding="utf-8"))),
            "t4": list(csv.DictReader(open(REPO / "results/t4_test/t4_test.csv", encoding="utf-8")))}
    z = dict(np.load(W / "dirty_potato_t3t4.npz"))   # CLIP (ViT-B-16) features of T3 / T4
    valid_y = np.array(dev_sets()["valid"][1])
    res = {}
    for name, (path, tag) in MODELS.items():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        veg = vegetable_outputs(path, tag, lists, rows)
        dev, answers = eye.score(ck, feats, lists, veg)
        r = {"dev": dev, "input_size": input_size(ck)}
        a = answers["valid"]; idx, unk = a.index.numpy(), a.unknown.numpy()
        r["valid_per_class"] = {c: f"{int(((idx == k) & ~unk & (valid_y == k)).sum())}/{int((valid_y == k).sum())}" for k, c in enumerate(CLASS_NAMES[:10])}
        for s in ("t3", "t4"):
            r[s] = dp.per_class(rows[s], ref.names(ck, veg[s], z[f"{s}_f"]))
        nd = dev["new_dev"]
        for c, n in (("tomato_rotten", 42), ("tomato_fresh", 42), ("potato_fresh", 60), ("potato_rotten", 36)):
            r[f"{c}_{n}"] = nd[c][0] + r["t3"][c][0] + r["t4"][c][0]
        res[name] = r
    for cand, base in (("H", "TV-12"), ("HP", "P1")):
        c, b = res[cand], res[base]
        c["rules"] = dev_rules(c["dev"], b["dev"])
        c["verdict"] = {"compared with": base, "rotten tomato gain (>= +5)": c["tomato_rotten_42"] - b["tomato_rotten_42"],
                        "fresh tomato change (>= -2)": c["tomato_fresh_42"] - b["tomato_fresh_42"], "8 guard rules": all(c["rules"].values())}
        v = c["verdict"]
        v["improvement"] = v["rotten tomato gain (>= +5)"] >= 5 and v["fresh tomato change (>= -2)"] >= -2 and v["8 guard rules"]
    sample = lists["dev"]["valid"][:48]
    res["ms_per_photo_vegetable_model"] = {"224 (TV-12)": ms_per_photo(REPO / "model.pth", sample), "320 (H)": ms_per_photo(W / "H.pth", sample)}
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for name in MODELS:
        r = res[name]; d = r["dev"]; o = d["outside"]
        print(f"{name:5} dev {d['dev_total']} (valid {d['valid']['kept_correct']} odev {d['odev']['kept_correct']} o160 {d['o160']['kept_correct']} "
              f"asian {d['asian37']['kept_correct']}) | conf-wrong {o['confident_wrong']} rot->fresh {o['rotten_to_fresh_answered']} answered {o['answered']} "
              f"| rotten tomato 42: {r['tomato_rotten_42']} fresh tomato 42: {r['tomato_fresh_42']} fresh potato 60: {r['potato_fresh_60']} "
              f"rotten potato 36: {r['potato_rotten_36']} | valid tomato_rotten {r['valid_per_class']['tomato_rotten']} | test {d['test_300_reference']}")
    for cand in ("H", "HP"):
        print(cand, json.dumps(res[cand]["verdict"], ensure_ascii=False), {k: v for k, v in res[cand]["rules"].items() if not v})
    print("ms per photo", res["ms_per_photo_vegetable_model"])


if __name__ == "__main__":
    main()
