"""Measure stored-development metrics; T9 is read only as an existing aggregate."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from compare_on_dev import development_sets, right
from decision import clip_head_logits, clip_unknown_score, decide, new_vegetable_logits
from labels import CLASS_NAMES

Z95 = 1.959963984540054


def wilson(k, n):
    if not n:
        return {"k": int(k), "n": 0, "rate": None, "ci95": None}
    p = k/n
    a = Z95**2/n
    center = (p + a/2)/(1+a)
    d = Z95 * math.sqrt(p*(1-p)/n + Z95**2/(4*n*n))/(1+a)
    return {"k": int(k), "n": int(n), "rate": float(p), "ci95": [max(0., center-d), min(1., center+d)]}


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def ece_bins(p, y, mode="width", bins=10):
    if not len(p):
        return []
    if mode == "width":
        groups = [np.flatnonzero(np.minimum((p*bins).astype(int), bins-1) == i) for i in range(bins)]
    else:
        # Repeated probabilities stay in the same bin.
        cuts = np.unique(np.quantile(p, np.linspace(0, 1, bins+1)))[1:-1]
        keys = np.searchsorted(cuts, p, side="right")
        groups = [np.flatnonzero(keys == i) for i in range(len(cuts)+1)]
    return [{"p_min": float(p[ix].min()), "p_max": float(p[ix].max()),
             "p_mean": float(p[ix].mean()), "accuracy": wilson(int(y[ix].sum()), len(ix))}
            for ix in groups if len(ix)]


def ece(p, y, mode="width", bins=10):
    if mode == "width":
        group = np.minimum((p*bins).astype(int), bins-1)
    else:
        cuts = np.unique(np.quantile(p, np.linspace(0, 1, bins+1)))[1:-1]
        group = np.searchsorted(cuts, p, side="right")
    return float(np.abs(np.bincount(group, weights=p-y)).sum()/len(p))


def calibration(p, y, rng, repeats):
    if not len(p):
        return {"n": 0}
    summary = {"n": len(p), "mean_probability": float(p.mean()), "accuracy": wilson(int(y.sum()), len(y)),
               "ece": ece(p, y), "ece_15": ece(p, y, bins=15), "adaptive_ece": ece(p, y, "adaptive"),
               "brier_top_label": float(np.mean((p-y)**2))}
    dist = []
    for i in range(repeats):
        sample = rng.integers(0, len(p), size=len(p))
        q, r = p[sample], y[sample]
        dist.append((q.mean(), ece(q,r), ece(q,r,"adaptive"), np.mean((q-r)**2)))
    for key, values in zip(("mean_probability", "ece", "adaptive_ece", "brier_top_label"), np.array(dist).T):
        summary[key+"_ci95"] = interval(values)
    return summary


def aurc(p, ok):
    order = np.argsort(-p, kind="stable")
    prob = p[order]
    ending = np.r_[np.flatnonzero(prob[:-1] != prob[1:])+1, len(prob)]
    risk = 1 - np.cumsum(ok[order])[ending-1]/ending
    return float(np.dot(np.diff(ending, prepend=0)/len(p), risk))


def paired(old, new):
    lost, gained = int(np.sum(old & ~new)), int(np.sum(~old & new))
    return paired_counts(lost, gained, len(old))


def paired_counts(lost, gained, n):
    return {"right_to_wrong": wilson(lost,n), "wrong_to_right": wilson(gained,n),
            "p_exact_two_sided": float(binomtest(gained,lost+gained).pvalue) if lost+gained else 1.}


def reject_margin(features, a, checkpoint):
    # Positive first-stage margin means unknown. A passed second head can rescue a qualifying new kind.
    first = clip_unknown_score(clip_head_logits(features, checkpoint), checkpoint, features)
    first = first - checkpoint["clip_head"]["unknown_threshold"]
    assert not checkpoint["clip_head"].get("vegetable_model_check", True)
    head = checkpoint["new_vegetable_head"]
    z = new_vegetable_logits(features,head)
    m = z[:,-1] - torch.logsumexp(z[:,:-1],dim=1) - head["unknown_threshold"]
    species = z[:,:-1].softmax(1).reshape(len(z),-1,2).sum(2).argmax(1)
    margin = torch.where(species >= head["first_new_species"], torch.minimum(first,m), first)
    assert torch.equal(margin > 0, a.unknown)
    return -margin.numpy()


def detection(pos, neg, rng, repeats):
    target = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores = np.r_[pos, neg]
    def fpr95(a,b):
        fpr,tpr,_ = roc_curve(a,b,drop_intermediate=False)
        return float(fpr[tpr >= .95][0])
    boot = []
    for _ in range(repeats):
        s = np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        boot.append((roc_auc_score(target,s),fpr95(target,s)))
    a,b = np.array(boot).T
    return {"AUROC": float(roc_auc_score(target,scores)), "AUROC_ci95": interval(a),
            "FPR_at_TPR_ge_95": fpr95(target,scores), "FPR95_ci95": interval(b),
            "vegetable_rejected": wilson(int((pos<0).sum()),len(pos)),
            "out_of_scope_accepted": wilson(int((neg>=0).sum()),len(neg))}


def run(bootstrap=1000):
    torch.set_num_threads(2)
    rng = np.random.default_rng(421)
    checkpoint = torch.load(REPO/"model.pth", map_location="cpu", weights_only=True)
    previous = {k:v for k,v in checkpoint.items() if k!="potato_check"}
    groups, samples, good, bad = {}, [], [], []
    all_finite = 0
    with torch.inference_mode():
        for name,(logits,features,labels) in development_sets().items():
            a = decide(logits,checkpoint,features)
            score = reject_margin(features,a,checkpoint)
            all_finite += int(torch.isfinite(a.probabilities).all(1).sum())
            if labels is None:
                (good if name.startswith("real-scene") else bad).extend(score)
                continue
            good.extend(score)
            v = decide(logits,previous,features)
            p,u = a.probability.numpy(), a.unknown.numpy()
            ok = np.array([not away and right(y,CLASS_NAMES[i]) for y,i,away in zip(labels,a.index.tolist(),u)])
            prior_ok = np.array([not away and right(y,CLASS_NAMES[i]) for y,i,away in zip(labels,v.index.tolist(),v.unknown.tolist())])
            kept = ~u
            show = kept & (p>=.9)
            groups[name] = {"calibration":calibration(p[kept],ok[kept],rng,bootstrap),
                            "all_accuracy":wilson(ok.sum(),len(ok)), "shown_accuracy":wilson(ok[show].sum(),show.sum()),
                            "rejected":wilson(u.sum(),len(u))}
            samples.append((np.array(labels),p,kept,ok,prior_ok))
    labels,p,kept,ok,prior_ok = [np.concatenate(x) for x in zip(*samples)]
    by_class = {}
    for label in sorted(set(labels)):
        take = (labels==label)
        x = take & kept
        by_class[str(label)] = {"all_accuracy":wilson(ok[take].sum(),take.sum()), "calibration":calibration(p[x],ok[x],rng,bootstrap)}
    grid = {}
    for threshold in np.arange(50,100)/100:
        accepted = kept & (p>=threshold)
        grid[f"{threshold:.2f}"] = {"coverage":wilson(accepted.sum(),len(p)),
                                   "accuracy":wilson(ok[accepted].sum(),accepted.sum())}
    p2,y2 = p[kept],ok[kept]
    draw = []
    for _ in range(bootstrap):
        ix = rng.integers(0,len(p2),size=len(p2))
        draw.append(aurc(p2[ix],y2[ix]))
    t9 = json.loads((REPO/"results/t9_test/scores_once.json").read_text(encoding="utf-8"))
    t9_pair = t9["right_and_wrong_against_TV-12"]
    first = sum(x["right_to_wrong"] for x in t9_pair.values())
    second = sum(x["wrong_to_right"] for x in t9_pair.values())
    finite_total = len(good)+len(bad)
    return {"labelled_n":len(p), "unlabelled_n":finite_total-len(p), "finite_probability_rows":wilson(all_finite,finite_total),
            "pooled":calibration(p2,y2,rng,bootstrap), "bins10":ece_bins(p2,y2),
            "adaptive_bins10":ece_bins(p2,y2,"adaptive"), "groups":groups, "classes":by_class,
            "coverage_grid":grid, "aurc_given_not_rejected":aurc(p2,y2), "aurc_ci95":interval(draw),
            "max_coverage":wilson(kept.sum(),len(kept)),
            "reject_metrics":detection(np.array(good),np.array(bad),rng,bootstrap),
            "mcnemar_development":paired(prior_ok,ok), "mcnemar_t9_saved_main":paired_counts(first,second,t9["n_main"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=1000)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.bootstrap),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
