# Development only (no sealed set is touched): 16-slot CLIP head trained with the v5 pool, and two ways to use it:
#   R (router, as V4-R): the new head picks the kind; old five -> the current system's answer; rejection by the new head.
#   C (cascade): the current system first; only photos it turns away go to the new head, which may answer onion / sweet potato /
#     green onion (when its "not five" log-odds is under its threshold and the most likely kind is a new one); otherwise still turned away.
import sys, csv, json, pickle, collections, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v5/'
src = open(S + 'negx/evaluate.py').read(); exec(src[:src.index("ck = torch.load")])
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size, CLASS_NAMES
torch.set_num_threads(4)
K = 16
nf_paths = [str(p) for k in ('fruit', 'everyday', 'otherveg') for p in np.load(S + f'neg/{k}_feats.npz')['paths']]
nf_paths = [p for p, d in zip(nf_paths, dn) if not d]
Xn16 = Xn[np.array(['/otherveg/onion__' not in p and '/otherveg/sweetpotato__' not in p for p in nf_paths])]
pool = json.load(open(V + 'pool.json')); pf = np.load(V + 'pool_feats.npz')['f']; assert len(pool) == len(pf)
tr = [i for i, r in enumerate(pool) if r['split'] == 'train']; va = [i for i, r in enumerate(pool) if r['split'] == 'val']
print('copies of check photos in the v5 pool train:', int(drop(pf[tr]).sum()))
X = np.concatenate([Xt, Xp, Xw, Xa, pf[tr], Xn16]); y = np.concatenate([yt, yp, yw, np.full(len(Xa), 0), [CLASS_NAMES.index(pool[i]['class_name']) for i in tr], np.full(len(Xn16), K)])
lr = LogisticRegression(C=10, max_iter=5000).fit(X, y)
coef = np.zeros((K + 1, X.shape[1])); icp = np.full(K + 1, lr.intercept_.min() - 50.0); coef[lr.classes_] = lr.coef_; icp[lr.classes_] = lr.intercept_
hz = lambda f: f @ coef.T + icp
B0, ck0 = load_model(REPO + 'model.pth', torch.device('cpu')); tf0 = make_transform(input_size(ck0)); h0 = ck0['clip_head']; TB0 = ck0['calibration']['temperature']
def blog(paths):
    with torch.inference_mode(): return torch.cat([B0(torch.stack([tf0(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
sets = json.load(open(S + 'clip/sets.json')); Y = json.load(open(S + 'post/labels.json'))
D = {'valid': (blog([p for p, _ in sets['valid']]), F['valid'], np.array([CLASS_NAMES.index(c) for c in Y['valid']])),
     'newval': (blog([pool[i]['path'] for i in va]), pf[va], np.array([CLASS_NAMES.index(pool[i]['class_name']) for i in va]))}
for k in ('odev', 'o160', 'asian37'): D[k] = (blog([p for p, _ in sets[k]]), F[k], np.array([CLASS_NAMES.index(c) for c in Y[k]]))
for k, fk in (('fruit374', 'fruit374'), ('coco400', 'coco400'), ('lvis', 'lvis_veg')): D[k] = (blog([p for p, _ in sets[fk]]), F[fk], None)
nfo = lambda z: z[:, K] - logsumexp(z[:, :K], 1)
EC = float(np.quantile(np.concatenate([nfo(hz(D['valid'][1])), nfo(hz(D['newval'][1])), nfo(hz(D['lvis'][1]))]), 0.95))
def now(b, f):
    z0 = f @ h0['coef'].numpy().T + h0['intercept'].numpy()
    return (log_softmax(b / TB0, 1) + log_softmax(z0[:, :10], 1)).argmax(1), z0[:, 10] - logsumexp(z0[:, :10], 1) > h0['unknown_threshold']
def newhead(f):
    z = hz(f); n = len(f); sp = np.exp(log_softmax(z[:, :K], 1)).reshape(n, 8, 2).sum(2).argmax(1)
    return sp, sp * 2 + z[:, :K].reshape(n, 8, 2)[np.arange(n), sp].argmax(1), nfo(z) > EC
def decide(name, b, f):
    p0, r0 = now(b, f)
    if name == 'now': return p0, r0
    sp, pn, rn = newhead(f)
    if name == 'R': return np.where(sp < 5, p0, pn), rn
    take = r0 & ~rn & (sp >= 5)            # C
    return np.where(take, pn, p0), r0 & ~take
res = {'train': {'new_veg_train': int(sum(pool[i]['class_name'].startswith(('onion', 'sweet', 'green')) for i in tr)), 'new_veg_val': len(va)}, 'EC_new': EC}
for name in ('now', 'R', 'C'):
    out = {}
    for k, (b, f, yy) in D.items():
        p, rej = decide(name, b, f)
        if yy is None: out[k] = f'rejected {int(rej.sum())}/{len(rej)}' + ('' if k != 'lvis' else f', answered new species {int(((p >= 10) & ~rej).sum())}'); continue
        right = p == yy
        if k == 'newval':
            kind = (p // 2 == yy // 2) & ~rej
            out[k] = {'kind_right_kept': f'{int(kind.sum())}/{len(yy)}', 'answers': collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in zip(yy, p, rej) if not (q == a and not j)).most_common()}
        else:
            out[k] = f'acc {int(right.sum())} kept_correct {int((right & ~rej).sum())} rejected {int(rej.sum())} new_species {int(((p >= 10) & ~rej).sum())} / {len(yy)}'
    res[name] = out
pickle.dump({'coef': coef, 'intercept': icp, 'EC': EC}, open(V + 'head_v5.pkl', 'wb'))
json.dump(res, open(V + 'explore.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(res, indent=1, ensure_ascii=False))
