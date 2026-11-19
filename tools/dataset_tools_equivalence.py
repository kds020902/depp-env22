"""Old against new dataset/tools on synthetic inputs with a mocked ORB (no photos needed).

    mkdir OLD && for v in 2 3; do git show b202b39:dataset/tools/build_dataset_v$v.py > OLD/build_dataset_v$v.py; done
    python tools/dataset_tools_equivalence.py OLD dataset/tools

120 cases (v2 assign_splits, v3 assign, v3 drop_old_duplicates): split of every photo and its order, duplicate groups,
sessions, train quotas and the state of `random` afterwards must be equal. Exit code 0 only when all are.
"""
import importlib.util, random, sys, types
import numpy as np

def load(folder, name, alias):
    sys.path.insert(0, folder)
    for m in ("build_dataset_v2", "build_dataset_v3"): sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location(alias, f"{folder}/{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    v2 = sys.modules.get("build_dataset_v2"); sys.path.remove(folder)
    return mod, v2

class Img:
    def __init__(self, size): self.size = size

def fake_orb(nodes, pairs):
    out = {}
    for a, b in pairs:
        r = random.Random(int(a) * 100003 + int(b))
        out[(int(a), int(b))] = r.choice([0, 0, 0, 12, 35, 90])
    return out

def make(seed, classes, n, v3):
    rng = np.random.default_rng(seed); py = random.Random(seed)
    centres = rng.normal(size=(max(4, n // 9), 512))
    nodes, var = [], np.zeros((n, 8, 512), np.float32)
    for i in range(n):
        c = py.choice(classes); k = py.randrange(len(centres))
        base = centres[k] + rng.normal(scale=py.choice([0.05, 0.3, 0.6, 1.5]), size=512)
        for j in range(8):
            f = base + rng.normal(scale=0.05, size=512); var[i, j] = f / np.linalg.norm(f)
        h = py.getrandbits(64) if py.random() < 0.8 else nodes[py.randrange(len(nodes))]["_h"] ^ (1 << py.randrange(64)) if nodes else py.getrandbits(64)
        nd = dict(class_name=c, source=py.choice(["muhriddinmuxiddinov"] * 9 + ["swoyam2609"]) if not v3 else "muhriddinmuxiddinov",
                  archive_path=f"p/{i}.jpg", phash=f"{h:016x}", dhash=f"{h ^ py.getrandbits(3):016x}", _h=h,
                  image=Img((224, 224) if py.random() < 0.35 else (300, 200)))
        if not v3 and py.random() < 0.3: nd["old_group"] = f"g{py.randrange(n // 6)}"
        nodes.append(nd)
    return nodes, var

OLD, NEW = sys.argv[1], sys.argv[2]
results = []
for which in ("old", "new"):
    folder = OLD if which == "old" else NEW
    m3, m2 = load(folder, "build_dataset_v3", f"v3_{which}")
    m2.orb_inliers = fake_orb; m3.orb_inliers = fake_orb
    out = []
    for seed in range(40):
        nodes, var = make(seed, m2.CLASS_NAMES, 700 + 40 * (seed % 5), False)
        m2.QUOTA.update(train=40 + seed % 3 * 10, val=6 + seed % 4, test=6 + seed % 4)
        try:
            split, dup, ses = m2.assign_splits(nodes, var); r = (sorted(split.items()), list(split), dup.tolist(), ses.tolist())
        except RuntimeError as e: r = ("error", str(e))
        out.append(("v2", seed, r, random.getstate()))
        nodes, var = make(1000 + seed, m3.NEW_CLASSES, 500 + 30 * (seed % 4), True)
        m3.EVAL_QUOTA = 5 + seed % 4; m3.TRAIN_MAX = 30 + seed % 3 * 15
        pairs = [(f"p/{a}.jpg", f"p/{a + 7}.jpg") for a in range(0, 60, 9)]
        try:
            split, dup, ses, shoot, quota = m3.assign(nodes, var, pairs); r = (sorted(split.items()), list(split), dup.tolist(), ses.tolist(), shoot.tolist(), quota)
        except RuntimeError as e: r = ("error", str(e))
        out.append(("v3", seed, r, random.getstate()))
        old_n, var_all = make(2000 + seed, m3.CLASS_NAMES, 400, True)
        m3.fingerprints = lambda both, v=var_all: v
        kept, kv, dropped = m3.drop_old_duplicates(old_n[:150], old_n[150:]); out.append(("drop", seed, [k["archive_path"] for k in kept], kv.tobytes(), dropped))
    results.append(out)
same = sum(a == b for a, b in zip(*results)); kinds = {}
for a in results[0]:
    key = (a[0], "error" if a[2] and a[2][0] == "error" else "ok"); kinds[key] = kinds.get(key, 0) + 1
print("cases", len(results[0]), "identical", same, kinds)
v2ok = [a for a in results[0] if a[0] == "v2" and a[2][0] != "error"]
import collections; print("v2 split mix of first case:", collections.Counter(v for _, v in v2ok[0][2][0]))
sys.exit(0 if same == len(results[0]) else 1)
