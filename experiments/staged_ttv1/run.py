"""TTV_1: a staged example system (the plan and the definitions are in PREREG.md).

    python experiments/staged_ttv1/run.py experiments/staged_ttv1/results.json

Reads experiments/new_vegetables/v7/head_data.npz (training rows) and the development sets of tools/compare_on_dev.py.
No blind test set is read. model.pth is only read for the TV-13 reference column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from compare_on_dev import AWAY, UNLABELLED, answers_of, development_sets, right, says_fresh  # noqa: E402
from labels import CLASS_NAMES, SPECIES_NAMES  # noqa: E402

OTHER = 16
NEW = {"onion": 5, "sweetpotato": 6, "greenonion": 7}
STEPS = ["not_eight + train1260", "+ reviewed_extra", "+ asian_cucumber", "+ weak_spots", "+ onion", "+ sweetpotato", "+ greenonion"]


def rows_of(step: int, y: np.ndarray, tag: np.ndarray) -> np.ndarray:
    keep = np.isin(tag, ["not_eight", "train1260"] + ["reviewed_extra", "asian_cucumber", "weak_spots"][:max(0, step)][:3])
    new = tag == "new_vegetables"
    kind = y // 2
    if step >= 4:
        keep |= new & (kind <= NEW["onion"])
    if step >= 5:
        keep |= new & (kind == NEW["sweetpotato"])
    if step >= 6:
        keep |= new & (kind == NEW["greenonion"])
    return keep


def fit(X, y):
    return LogisticRegression(C=10, max_iter=3000).fit(X, y)


def threshold(scores: np.ndarray) -> float:
    """The value 95% of the Valid photos of the kinds trained so far pass."""
    return float(np.quantile(scores, 0.05))


class Staged:
    def __init__(self, X, y):
        self.kinds = sorted(set((y[y != OTHER] // 2).tolist()))
        self.is_vegetable = fit(X, (y != OTHER).astype(int))
        v = y != OTHER
        self.kind = fit(X[v], y[v] // 2)
        self.condition = {k: fit(X[y // 2 == k], y[y // 2 == k] % 2) for k in self.kinds if len(set(y[y // 2 == k] % 2)) == 2}

    def score(self, X):
        return self.is_vegetable.decision_function(X)

    def classes(self, X):
        kind = self.kind.predict(X)
        out = kind * 2
        for k, model in self.condition.items():
            m = kind == k
            if m.any():
                out[m] = k * 2 + model.predict(X[m])
        return out


class Flat:
    def __init__(self, X, y):
        self.model = fit(X, y)
        self.other = list(self.model.classes_).index(OTHER)

    def score(self, X):
        z = self.model.decision_function(X)
        return np.delete(z, self.other, 1).max(1) - z[:, self.other]

    def classes(self, X):
        z = self.model.decision_function(X).copy()
        z[:, self.other] = -np.inf
        return self.model.classes_[z.argmax(1)]


def answers(system, X, cut):
    names = [CLASS_NAMES[int(c)] for c in system.classes(X)]
    return [AWAY if s < cut else n for s, n in zip(system.score(X), names)]


def measure(get_answers, sets, kinds):
    """Counts over the development photos of the kinds trained so far."""
    per_class, flags, away, kept = {}, {}, {}, {}
    wrong_fresh = 0
    for name, (b, X, labels) in sets.items():
        a = get_answers(name, b, X)
        if labels is None:
            should_reject = dict(UNLABELLED.values())[name]
            (away if should_reject else kept)[name] = [a.count(AWAY) if should_reject else len(a) - a.count(AWAY), len(a)]
            continue
        for i, (label, answer) in enumerate(zip(labels, a)):
            if SPECIES_NAMES.index(label.rsplit("_", 1)[0]) not in kinds:
                continue
            ok = right(label, answer)
            c = per_class.setdefault(label, [0, 0])
            c[0] += ok
            c[1] += 1
            flags[f"{name}#{i}"] = [bool(ok), answer, label]
            wrong_fresh += label.endswith("_rotten") and says_fresh(answer)
    return {"classes": per_class, "right": sum(c[0] for c in per_class.values()), "n": sum(c[1] for c in per_class.values()),
            "rotten_called_fresh": int(wrong_fresh), "turned_away_should": away, "kept_should": kept, "photos": flags}


def against_previous(now, before):
    """Among the classes both steps answer: classes whose count fell, photos lost there, and right -> wrong on the same photos."""
    fell = {c: before["classes"][c][0] - now["classes"][c][0] for c in before["classes"] if now["classes"][c][0] < before["classes"][c][0]}
    flips = sum(before["photos"][p][0] and not now["photos"][p][0] for p in before["photos"])
    gains = sum(not before["photos"][p][0] and now["photos"][p][0] for p in before["photos"])
    return {"classes_that_fell": fell, "photos_lost_in_them": sum(fell.values()), "right_to_wrong": int(flips), "wrong_to_right": int(gains)}


def stages_of_errors(system: Staged, cut, sets):
    """Where the staged system loses a labelled photo: turned away, wrong kind, or right kind and wrong condition."""
    out = {}
    for name, (b, X, labels) in sets.items():
        if labels is None:
            continue
        a = answers(system, X.numpy(), cut)
        for label, answer in zip(labels, a):
            if right(label, answer):
                continue
            where = "turned_away" if answer == AWAY else "wrong_kind" if answer.rsplit("_", 1)[0] != label.rsplit("_", 1)[0] else "wrong_condition"
            out.setdefault(label, {"turned_away": 0, "wrong_kind": 0, "wrong_condition": 0})[where] += 1
    return out


def main():
    out = Path(sys.argv[1])
    data = np.load(REPO / "experiments/new_vegetables/v7/head_data.npz", allow_pickle=True)
    X, y, tag = data["train_X"], data["train_y"].astype(int), data["train_tag"]
    sets = development_sets()
    valid = {n: sets[n] for n in ("Valid 300", "new-vegetable Valid 89")}
    checkpoint = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    record = {"steps": [], "training_rows": []}
    previous = {}
    for step, title in enumerate(STEPS):
        keep = rows_of(step, y, tag)
        kinds = sorted(set((y[keep & (y != OTHER)] // 2).tolist()))
        record["training_rows"].append({CLASS_NAMES[c] if c != OTHER else "other": int((y[keep] == c).sum()) for c in sorted(set(y[keep].tolist()))})
        entry = {"step": title, "kinds": [SPECIES_NAMES[k] for k in kinds]}
        for name, system in (("TTV_1", Staged(X[keep], y[keep])), ("FLAT", Flat(X[keep], y[keep]))):
            scores = np.concatenate([system.score(f.numpy())[[SPECIES_NAMES.index(l.rsplit("_", 1)[0]) in kinds for l in labels]]
                                     for _, f, labels in valid.values()])
            cut = threshold(scores)
            m = measure(lambda _n, _b, f, s=system, c=cut: answers(s, f.numpy(), c), sets, kinds)
            if name in previous:
                m["against_previous_step"] = against_previous(m, previous[name])
            previous[name] = m
            if name == "TTV_1" and step == len(STEPS) - 1:
                entry["TTV_1_errors_by_stage"] = stages_of_errors(system, cut, sets)
            entry[name] = {k: v for k, v in m.items() if k != "photos"}
        if step == len(STEPS) - 1:
            entry["TV-13"] = {k: v for k, v in measure(lambda _n, b, f: answers_of(checkpoint, b, f)[0], sets, kinds).items() if k != "photos"}
        record["steps"].append(entry)
    json.dump(record, open(out, "w"), indent=1, ensure_ascii=False)
    for e in record["steps"]:
        for name in ("TTV_1", "FLAT", "TV-13"):
            if name in e:
                m = e[name]
                print(f"{e['step']:26s} {name:6s} right {m['right']}/{m['n']}  rotten->fresh {m['rotten_called_fresh']}  "
                      f"away {m['turned_away_should']}  kept {m['kept_should']}  vs previous {m.get('against_previous_step', '')}")


if __name__ == "__main__":
    main()
