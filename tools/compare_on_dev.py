"""Compare the baseline (model.pth = TV-12) with candidate checkpoints on the same development photos.

    python tools/compare_on_dev.py OUT.json [NAME=candidate.pth ...]

Without candidates it records the baseline alone. No photo and no CLIP weights are needed: the development sets are stored
as model inputs (b = the vegetable model's logits, X = CLIP image features):
- experiments/new_vegetables/v7/head_data.npz: Valid 300, internet 221, field 160, Asian cucumbers 37, new-vegetable Valid 89,
  and three sets without a class label (rotten/other fruit 374 and everyday things 400 should be turned away; the 192
  real-scene vegetable crops should not);
- experiments/sweetpotato_prompts/dev_features.npz: 162 potato / sweet potato web photos.
No blind test set (T3-T8) is read. Because b is stored, a candidate must keep model.pth's vegetable model (checked);
a candidate with a retrained vegetable model has to be compared on photos (experiments/root_and_tomato/evaluate.py).

Per set and class (tally()):
- right: the answer is the class; for green onions the kind is enough (the system does not judge their condition);
- rotten_called_fresh: a rotten photo answered "fresh" of any vegetable, as results/scorecard counts it;
  rotten_called_fresh_same_kind: of the same vegetable only; rotten_called_fresh_shown: those among the shown answers.
  A green onion answer claims no condition, so it is a wrong kind, never a "fresh" answer;
- turned_away, held_back (answered, but below 0.9), and shown = neither: the answers a user actually sees, and how many are right.
A candidate may change the heads, thresholds and text embeddings. It must keep what the stored inputs were made with:
model.pth's vegetable-model weights and preprocessing, and the CLIP image encoder of both heads (cached_inputs_apply()).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from decision import HOLD_BACK_BELOW, decide, held_back  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import state_checksum  # noqa: E402

HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
ROOT_DATA = REPO / "experiments/sweetpotato_prompts/dev_features.npz"
LABELLED = {"valid": "Valid 300", "odev": "internet 221", "o160": "field 160", "asian37": "Asian cucumbers 37",
            "newval": "new-vegetable Valid 89"}
UNLABELLED = {"fruit374": ("fruit 374", True), "coco400": ("everyday things 400", True), "lvis": ("real-scene vegetable crops 192", False)}
AWAY = "turned_away"
BASELINE = "TV-12"
SPECIES_ONLY = ("greenonion",)   # the system does not judge the condition of green onions


def development_sets() -> dict[str, tuple[torch.Tensor, torch.Tensor, list[str] | None]]:
    """name -> (vegetable-model logits, CLIP image features, class names or None)."""
    stored, sets = np.load(HEAD_DATA), {}
    for key, name in LABELLED.items():
        sets[name] = (torch.tensor(stored[f"dev_{key}_b"]), torch.tensor(stored[f"dev_{key}_X"]),
                      [CLASS_NAMES[int(i)] for i in stored[f"dev_{key}_y"]])
    for key, (name, _) in UNLABELLED.items():
        sets[name] = (torch.tensor(stored[f"dev_{key}_b"]), torch.tensor(stored[f"dev_{key}_X"]), None)
    roots = np.load(ROOT_DATA)
    sets["potato / sweet potato web 162"] = (torch.tensor(roots["b"]), torch.tensor(roots["X"]), [CLASS_NAMES[int(i)] for i in roots["y"]])
    return sets


def right(label: str, answer: str) -> bool:
    if label.startswith(SPECIES_ONLY):
        return answer.rsplit("_", 1)[0] == label.rsplit("_", 1)[0]
    return answer == label


def answers_of(checkpoint: dict, logits: torch.Tensor, features: torch.Tensor) -> tuple[list[str], list[bool]]:
    a = decide(logits, checkpoint, features)
    return [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)], held_back(a).tolist()


def says_fresh(answer: str) -> bool:
    """The answer claims a fresh vegetable (a green onion answer claims no condition)."""
    return answer.endswith("_fresh") and not answer.startswith(SPECIES_ONLY)


def tally(labels: list[str], answers: list[str], held: list[bool]) -> dict:
    """Counts for one group of labelled photos (definitions in the module text)."""
    n = len(labels)
    shown = [i for i in range(n) if answers[i] != AWAY and not held[i]]
    rotten = [i for i in range(n) if labels[i].endswith("_rotten")]
    return {"n": n, "right": sum(right(y, a) for y, a in zip(labels, answers)),
            "rotten_n": len(rotten), "rotten_called_fresh": sum(says_fresh(answers[i]) for i in rotten),
            "rotten_called_fresh_same_kind": sum(answers[i] == labels[i].replace("_rotten", "_fresh") for i in rotten),
            "rotten_called_fresh_shown": sum(says_fresh(answers[i]) for i in rotten if not held[i]),
            "turned_away": answers.count(AWAY), "held_back": sum(held),
            "shown": len(shown), "shown_right": sum(right(labels[i], answers[i]) for i in shown)}


def evaluate(checkpoint: dict, sets: dict) -> dict:
    out, every = {}, ([], [], [])
    for name, (logits, features, labels) in sets.items():
        answers, held = answers_of(checkpoint, logits, features)
        if labels is None:
            out[name] = {"n": len(answers), "turned_away": answers.count(AWAY), "answers": answers}
            continue
        classes = {c: tally(*zip(*[(y, a, h) for y, a, h in zip(labels, answers, held) if y == c]))
                   for c in CLASS_NAMES if c in labels}
        out[name] = {**tally(labels, answers, held), "classes": classes, "answers": answers}
        for store, values in zip(every, (labels, answers, held)):
            store.extend(values)
    out["_all_labelled"] = {**tally(*every), "classes": {c: tally(*zip(*[(y, a, h) for y, a, h in zip(*every) if y == c]))
                                                          for c in CLASS_NAMES if c in every[0]}}
    return out


def changed(base: dict, other: dict, sets: dict) -> dict:
    """Photos whose answer differs from the baseline's, per set."""
    return {name: sum(x != y for x, y in zip(base[name]["answers"], other[name]["answers"])) for name in sets}


def percent(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.1f}%)" if n else "-"


def table(results: dict[str, dict]) -> str:
    """Markdown: one row per class of all labelled development photos, the systems side by side."""
    names = list(results)
    lines = ["| class | n | " + " | ".join(f"{m} right" for m in names) + " | " + " | ".join(f"{m} rotten->fresh (any kind / same kind)" for m in names) + " |",
             "|---|---:|" + "---:|" * (2 * len(names))]
    first = results[names[0]]["_all_labelled"]
    for c, t in list(first["classes"].items()) + [("all labelled", first)]:
        cells = [results[m]["_all_labelled"]["classes"][c] if c != "all labelled" else results[m]["_all_labelled"] for m in names]
        lines.append(f"| {c} | {t['n']} | " + " | ".join(percent(x["right"], x["n"]) for x in cells) + " | "
                     + " | ".join(f"{x['rotten_called_fresh']} / {x['rotten_called_fresh_same_kind']}" if x["rotten_n"] else "-" for x in cells) + " |")
    lines += ["", "| system | held back (below %.1f) | turned away | shown answers right | rotten->fresh among shown answers |" % HOLD_BACK_BELOW,
              "|---|---:|---:|---:|---:|"]
    for m in names:
        t = results[m]["_all_labelled"]
        lines.append(f"| {m} | {percent(t['held_back'], t['n'])} | {percent(t['turned_away'], t['n'])} | {percent(t['shown_right'], t['shown'])} | {t['rotten_called_fresh_shown']} |")
    return "\n".join(lines)


def cached_inputs_apply(baseline: dict, candidate: dict) -> list[str]:
    """Why the stored logits and CLIP features cannot be used for `candidate` (empty = they can)."""
    reasons = []
    if state_checksum(baseline["state_dict"]) != state_checksum(candidate["state_dict"]):
        reasons.append("the vegetable model's weights differ")
    for key in ("architecture", "pooling", "preprocess"):
        if baseline.get(key) != candidate.get(key):
            reasons.append(f"the vegetable model's {key} differs")
    for head in ("clip_head", "new_vegetable_head"):
        if head in candidate and candidate[head].get("clip") != baseline["clip_head"]["clip"]:
            reasons.append(f"{head} uses another CLIP image encoder: {candidate[head].get('clip')}")
    return reasons


def parse_candidates(arguments: list[str]) -> dict[str, Path]:
    candidates = {}
    for argument in arguments:
        name, separator, path = argument.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"{argument!r}: write a candidate as NAME=checkpoint.pth")
        if name == BASELINE:
            raise SystemExit(f"{BASELINE} is the baseline's name (model.pth): give the candidate another name")
        if name in candidates:
            raise SystemExit(f"the name {name} is given twice")
        candidates[name] = Path(path)
    return candidates


def main():
    out, candidates = Path(sys.argv[1]), parse_candidates(sys.argv[2:])
    baseline = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    sets = development_sets()
    results = {BASELINE: evaluate(baseline, sets)}
    for name, path in candidates.items():
        candidate = torch.load(path, map_location="cpu", weights_only=True)
        if reasons := cached_inputs_apply(baseline, candidate):
            raise SystemExit(f"{name}: the stored development inputs do not apply ({'; '.join(reasons)}). "
                             "Compute the answers from the photos instead (experiments/root_and_tomato/evaluate.py).")
        results[name] = evaluate(candidate, sets)
    record = {"inputs": {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO / "model.pth", HEAD_DATA, ROOT_DATA)},
              "candidates": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in candidates.items()},
              "hold_back_below": HOLD_BACK_BELOW, "systems": results,
              "answers_changed_from_TV-12": {name: changed(results[BASELINE], results[name], sets) for name in candidates}}
    json.dump(record, open(out, "w"), indent=1, ensure_ascii=False)
    print(table(results))
    for name, counts in record["answers_changed_from_TV-12"].items():
        print(f"\n{name}: answers changed from TV-12:", counts)


if __name__ == "__main__":
    main()
