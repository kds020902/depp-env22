"""experiments/bigger_eye (PREREG.md): TV-12's CLIP parts rebuilt on another image model; the vegetable model is unchanged.

    python experiments/bigger_eye/build.py B1 ViT-B-16-SigLIP2 webli
    python experiments/bigger_eye/build.py B0 ViT-B-16 datacomp_xl_s13b_b90k --seed-from-caches   # TV-12's own image model

Writes $DEPP_WORK/wm/eye_<TAG>_feats.npz (image features by photo path; a stopped run continues) and eye_<TAG>.pth:
1. image features of every photo in eye_lists.json (lists.py);
2. the CLIP head: src/train_clip_head.py with TV-12's photos and settings (watermark round 2, WC);
3. the text evidence: the 91 words and prompts of experiments/zero_shot_reject, alpha 0.25, threshold refitted by the same rule;
4. the new-vegetable head: src/train_new_veg_head.py on the same 6,271 rows and prompts, tau 0.8 (TV-12 = ZS8).
Text scores use the image model's own logit scale (100 for TV-12's model).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "watermark"))
from common import HEAD_ARGS, LVIS, REPO, S, W, rotten_fruit_features, run  # noqa: E402
from clip_model import FRUITS, OTHER_THINGS, OTHER_VEGETABLES, ClipModel  # noqa: E402
from decision import clip_head_logits, clip_unknown_score  # noqa: E402
from train_clip_head import clip_features  # noqa: E402

EXTRA = S / "negx/extra_train_N3.csv"
NEW_VEG_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
VEG_WORDS = ["cucumber", "potato", "tomato", "bell pepper", "carrot"]
DEV = ("valid", "odev", "o160", "asian37", "fruit374", "coco400", "lvis", "newval")


def all_paths(lists: dict) -> list[str]:
    paths = [p for name in DEV for p in lists["dev"][name]] + [p for v in lists["rotten_fruit"].values() for p in v]
    paths += [p for p, _ in lists["newveg_train"]] + lists["t5_dev"] + lists["test_300"]
    return list(dict.fromkeys(paths))


def save_store(path: Path, store: dict):
    np.savez(path, keys=np.array(list(store)), features=np.stack([np.asarray(v, np.float32) for v in store.values()]))


def load_store(path: Path) -> dict:
    with np.load(path) as saved:
        return dict(zip(saved["keys"].tolist(), saved["features"]))


def seed_from_caches(path: Path, lists: dict):
    """B0 only: TV-12's own ViT-B-16 features, taken from the caches the earlier rounds made (no new image is read)."""
    store = load_store(W / "head_features.npz")
    sets = json.load(open(S / "clip/sets.json")); F = dict(np.load(W / "clip_cache.npz"))
    for name, key in (("valid", "valid"), ("odev", "odev"), ("o160", "o160"), ("asian37", "asian37"), ("fruit374", "fruit374"),
                      ("coco400", "coco400"), ("lvis", "lvis_veg")):
        store.update(zip([p for p, _ in sets[key]], F[name]))
    pool = json.load(open(S / "v7/pool.json")); store.update(zip([r["path"] for r in pool], np.load(S / "v7/pool_feats.npz")["f"]))
    for k in ("fruit", "everyday", "otherveg"):
        z = np.load(S / f"neg/{k}_feats.npz"); store.update(zip([str(p) for p in z["paths"]], z["f"]))
    for k, f in rotten_fruit_features().items():
        store.update(zip(lists["rotten_fruit"][k], f))
    store.update(zip(lists["t5_dev"], np.load(W / "t3ws_dev_clip_t5.npz")["f"]))
    store.update(zip(lists["test_300"], np.load(W / "final_clip.npz")["test_300"]))
    missing = [p for p in all_paths(lists) if p not in store]
    assert not missing, f"{len(missing)} photos have no cached feature, e.g. {missing[:3]}"
    save_store(path, store)


def text_embeddings(clip: ClipModel, arch: str, prompt_lists: list[list[str]]) -> torch.Tensor:
    import open_clip
    tok = open_clip.get_tokenizer(arch)
    with torch.inference_mode():
        rows = [torch.nn.functional.normalize(clip.model.encode_text(tok(p)).float(), dim=-1).mean(0) for p in prompt_lists]
    return torch.nn.functional.normalize(torch.stack(rows), dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag"); parser.add_argument("arch"); parser.add_argument("pretrained")
    parser.add_argument("--seed-from-caches", action="store_true", help="B0: take TV-12's features from the earlier caches.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    lists = json.load(open(W / "eye_lists.json"))
    store_path, out = W / f"eye_{args.tag}_feats.npz", W / f"eye_{args.tag}.pth"
    if args.seed_from_caches and not store_path.exists():
        seed_from_caches(store_path, lists)

    # 1. image features of every photo (missing ones only)
    clip = ClipModel("cpu", arch=args.arch, pretrained=args.pretrained)
    scale = round(float(clip.model.logit_scale.exp()), 4)
    paths = all_paths(lists)
    clip_features(clip, [Path(p) for p in paths], cache=store_path)
    print(f"{args.tag}: features of {len(paths)} photos, logit scale {scale}", flush=True)

    # 2. CLIP head with TV-12's photos and settings (the script adds the light-watermark copies to the same feature file)
    shutil.copy(REPO / "model.pth", out)
    head_args = [a for a in HEAD_ARGS[:HEAD_ARGS.index("--feature-cache")]] + ["--feature-cache", str(store_path)]
    run("src/train_clip_head.py", "--model", out, "--extra-photos", LVIS, "--extra-train", EXTRA, *head_args,
        "--clip-arch", args.arch, "--clip-pretrained", args.pretrained)

    # 3. text evidence: the same words and prompts, this model's text side; threshold by the same rule (95% of Valid + 192 pass)
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    ck = torch.load(out, map_location="cpu", weights_only=True)
    four = lambda w: [f"a photo of a {w}.", f"a photo of a fresh {w}.", f"a photo of a rotten {w}.", f"a photo of a moldy {w}."]
    evidence = copy.deepcopy(tv12["clip_head"]["text_evidence"])
    evidence["vegetables"] = text_embeddings(clip, args.arch, [four(w) for w in VEG_WORDS])
    evidence["other"] = text_embeddings(clip, args.arch, [four(w) for w in {**FRUITS, **OTHER_VEGETABLES}]
                                        + [[f"a photo of {w}."] for w in OTHER_THINGS])
    if scale != 100.0:
        evidence["logit_scale"] = scale
    ck["clip_head"]["text_evidence"] = evidence
    store = load_store(store_path)
    feats = lambda ps: torch.tensor(np.stack([store[p] for p in ps]))
    scores = [clip_unknown_score(clip_head_logits(feats(lists["dev"][n]), ck), ck, feats(lists["dev"][n])) for n in ("valid", "lvis")]
    ck["clip_head"]["tv11_unknown_threshold"] = ck["clip_head"]["unknown_threshold"]
    ck["clip_head"]["unknown_threshold"] = float(torch.quantile(torch.cat(scores), 0.95))
    ck["version"] = f"bigger_eye {args.tag} ({args.arch}, {args.pretrained})"
    torch.save(ck, out)

    # 4. new-vegetable head: same rows, prompts and development photos, this model's features and text side
    data = dict(np.load(NEW_VEG_DATA))
    data["train_X"] = np.stack([store[p] for p, _ in lists["newveg_train"]]).astype(np.float32)
    assert np.array_equal(data["train_y"], np.array([y for _, y in lists["newveg_train"]], dtype=data["train_y"].dtype))
    for name in DEV:
        data[f"dev_{name}_X"] = np.stack([store[p] for p in lists["dev"][name]]).astype(np.float32)
    data["text16"] = text_embeddings(clip, args.arch, [str(p).split("|") for p in data["prompts"]]).numpy().astype(np.float32)
    data_path = W / f"head_data_eye_{args.tag}.npz"
    np.savez_compressed(data_path, **data)
    run("src/train_new_veg_head.py", "--model", out, "--data", data_path, "--logit-scale", scale, "--tau", 0.8)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
