"""TTV_2: split kind classifiers and a second stage with the vegetable model's rot evidence (PREREG.md, addendum 1).

    python experiments/staged_ttv1/run2.py experiments/staged_ttv1/results2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import NEW, OTHER, REPO, STEPS, Staged, against_previous, answers, fit, measure, rows_of, threshold  # noqa: E402
from compare_on_dev import AWAY, development_sets, right  # noqa: E402
from labels import CLASS_NAMES, SPECIES_NAMES  # noqa: E402

POTATO, SWEET = 1, NEW["sweetpotato"]
SWITCH = 0.8
WEIGHTS = (0, 0.5, 1, 2, 4)


class OneVsRest(Staged):
    def __init__(self, X, y):
        super().__init__(X, y)
        v = y != OTHER
        self.heads = {k: fit(X[v], (y[v] // 2 == k).astype(int)) for k in self.kinds}

    def kind_of(self, X):
        return np.array(self.kinds)[np.stack([self.heads[k].decision_function(X) for k in self.kinds], 1).argmax(1)]

    def classes(self, X):
        return with_condition(self, self.kind_of(X), X)


class Frozen(Staged):
    """The kinds before `new_kinds` keep the kind classifier of `old`; each new kind takes a photo at probability >= SWITCH."""

    def __init__(self, X, y, old: Staged, new_kinds: list[int]):
        super().__init__(X, y)
        self.old, v = old, y != OTHER
        self.takers = {k: fit(X[v & (y // 2 <= k)], (y[v & (y // 2 <= k)] // 2 == k).astype(int)) for k in new_kinds}

    def kind_of(self, X):
        kind = self.old.kind_of(X) if isinstance(self.old, Frozen) else self.old.kind.predict(X)
        for k, head in self.takers.items():
            kind = np.where(head.predict_proba(X)[:, 1] >= SWITCH, k, kind)
        return kind

    def classes(self, X):
        return with_condition(self, self.kind_of(X), X)


class Pair(Staged):
    def __init__(self, X, y):
        super().__init__(X, y)
        m = np.isin(y // 2, [POTATO, SWEET]) & (y != OTHER)
        self.pair = fit(X[m], (y[m] // 2 == SWEET).astype(int)) if SWEET in self.kinds else None

    def classes(self, X):
        kind = self.kind.predict(X)
        if self.pair is not None:
            top2 = self.kind.classes_[np.argsort(self.kind.decision_function(X), 1)[:, -2:]]
            both = np.isin(top2, [POTATO, SWEET]).all(1)
            kind = np.where(both, np.where(self.pair.predict(X) == 1, SWEET, POTATO), kind)
        return with_condition(self, kind, X)


def with_condition(system: Staged, kind, X, extra=None, w=0.0):
    out = kind * 2
    for k, model in system.condition.items():
        m = kind == k
        if m.any():
            z = model.decision_function(X[m])
            if extra is not None and k < 5:
                z = z + w * (extra[m, 2 * k + 1] - extra[m, 2 * k])
            out[m] = k * 2 + (z > 0)
    return out


def cut_for(system, valid, kinds):
    return threshold(np.concatenate([system.score(f.numpy())[[SPECIES_NAMES.index(l.rsplit("_", 1)[0]) in kinds for l in labels]]
                                     for _, f, labels in valid.values()]))


def summary(m):
    c = m["classes"]
    return {"right": m["right"], "n": m["n"], "rotten_called_fresh": m["rotten_called_fresh"],
            "potato_fresh": c.get("potato_fresh"), "potato_rotten": c.get("potato_rotten"),
            "sweetpotato_fresh": c.get("sweetpotato_fresh"), "sweetpotato_rotten": c.get("sweetpotato_rotten"),
            "greenonion_fresh": c.get("greenonion_fresh"), "turned_away_should": m["turned_away_should"], "kept_should": m["kept_should"],
            "against_previous_step": m.get("against_previous_step")}


def main():
    out = Path(sys.argv[1])
    data = np.load(REPO / "experiments/new_vegetables/v7/head_data.npz", allow_pickle=True)
    X, y, tag = data["train_X"], data["train_y"].astype(int), data["train_tag"]
    sets = development_sets()
    valid = {n: sets[n] for n in ("Valid 300", "new-vegetable Valid 89")}
    record = {"kind_variants": {}, "second_stage": {}}

    # A. kind classifiers, steps 4 (before any new kind) to 6
    built = {"TTV_1": lambda Xs, ys, prev, new: Staged(Xs, ys), "K-OVR": lambda Xs, ys, prev, new: OneVsRest(Xs, ys),
             "K-PAIR": lambda Xs, ys, prev, new: Pair(Xs, ys),
             "K-FROZEN": lambda Xs, ys, prev, new: Frozen(Xs, ys, prev, new) if prev is not None else Staged(Xs, ys)}
    for name, make in built.items():
        previous, system, steps = None, None, []
        for step in range(3, len(STEPS)):
            keep = rows_of(step, y, tag)
            kinds = sorted(set((y[keep & (y != OTHER)] // 2).tolist()))
            new = [k for k in kinds if system is not None and k not in system.kinds]
            system = make(X[keep], y[keep], system, new)
            cut = cut_for(system, valid, kinds)
            m = measure(lambda _n, _b, f, s=system, c=cut: answers(s, f.numpy(), c), sets, kinds)
            if previous is not None:
                m["against_previous_step"] = against_previous(m, previous)
            previous = m
            steps.append({"step": STEPS[step], **summary(m)})
        record["kind_variants"][name] = steps

    # B. second stage with the vegetable model's evidence, all groups
    keep = rows_of(len(STEPS) - 1, y, tag)
    system = Staged(X[keep], y[keep])
    kinds = system.kinds
    cut = cut_for(system, valid, kinds)

    def condition_right(w, names):
        ok = n = 0
        for name in names:
            b, f, labels = sets[name]
            kind = system.kind.predict(f.numpy())
            pred = with_condition(system, kind, f.numpy(), b.numpy(), w)
            for label, k, p in zip(labels, kind, pred):
                true = CLASS_NAMES.index(label)
                if true // 2 == k and k < 5:
                    ok += p == true
                    n += 1
        return int(ok), n
    scores = {w: condition_right(w, ["Valid 300"]) for w in WEIGHTS}
    w = max(WEIGHTS, key=lambda v: (scores[v][0], -v))
    record["second_stage"]["condition_right_on_valid_by_w"] = {str(k): v for k, v in scores.items()}
    record["second_stage"]["w"] = w
    for label, weight in (("TTV_1 (w=0)", 0.0), (f"S2+E (w={w})", w)):
        def get(name, b, f, weight=weight):
            names = [CLASS_NAMES[int(c)] for c in with_condition(system, system.kind.predict(f.numpy()), f.numpy(), b.numpy(), weight)]
            return [AWAY if s < cut else n for s, n in zip(system.score(f.numpy()), names)]
        m = measure(get, sets, kinds)
        per_set = {}
        for name, (b, f, labels) in sets.items():
            if labels:
                a = get(name, b, f)
                per_set[name] = [sum(right(l, x) for l, x in zip(labels, a)), len(labels)]
        fresh_as_rotten = sum((not ok) and lab.endswith("_fresh") and ans == lab.replace("_fresh", "_rotten") for ok, ans, lab in m["photos"].values())
        record["second_stage"][label] = {**summary(m), "classes": m["classes"], "per_set": per_set, "fresh_called_rotten_same_kind": int(fresh_as_rotten)}
    json.dump(record, open(out, "w"), indent=1, ensure_ascii=False)
    for name, steps in record["kind_variants"].items():
        for s in steps:
            print(name, s["step"], s["right"], s["n"], "pf", s["potato_fresh"], "sp", s["sweetpotato_fresh"], s["sweetpotato_rotten"], "go", s["greenonion_fresh"],
                  "away", list(s["turned_away_should"].values())[0], "kept", list(s["kept_should"].values())[0], s["against_previous_step"])
    print(record["second_stage"]["condition_right_on_valid_by_w"], "w", w)
    for k, v in record["second_stage"].items():
        if isinstance(v, dict) and "right" in v:
            print(k, v["right"], v["n"], "r->f", v["rotten_called_fresh"], "f->r", v["fresh_called_rotten_same_kind"], v["per_set"])


if __name__ == "__main__":
    main()
