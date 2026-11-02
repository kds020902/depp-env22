# v7, development only (no blind set is touched): ways to help abnormal sweet potato and green onion in the new-vegetable head.
#   base : V6-H as adopted (logistic regression, C=10)
#   zs(a): + a x CLIP zero-shot class logits (100 x cosine to text prompts, centred per photo) added to the 16 class logits
#   cb   : class-balanced sample weights for the six new-vegetable classes (each weighs as much as their average)
# Each variant: "not eight" threshold by the 95% rule and tau by the V5/V6 rule, then the V6-H decision.
# Measured on the development sets (old five: Valid 300, internet 221, 160, Asian 37, fruit 374, everyday 400, LVIS 192; new: Valid 73)
# and by 5-fold grouped cross-validation over the new-vegetable Train photos (folds by uploader / observer group).
import sys, json, pickle, hashlib, collections, numpy as np, torch, open_clip, os
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp, softmax
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V6 = S + 'v6/'; V = S + 'v7/'
src = open(S + 'negx/evaluate.py').read(); exec(src[:src.index("ck = torch.load")])
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size, CLASS_NAMES
torch.set_num_threads(4); K = 16
POOL = sys.argv[1] if len(sys.argv) > 1 else V6 + 'pool.json'
nf_paths = [str(p) for k in ('fruit', 'everyday', 'otherveg') for p in np.load(S + f'neg/{k}_feats.npz')['paths']]
nf_paths = [p for p, d in zip(nf_paths, dn) if not d]
Xn16 = Xn[np.array(['/otherveg/onion__' not in p and '/otherveg/sweetpotato__' not in p for p in nf_paths])]
pool = json.load(open(POOL)); pf = np.load(POOL.replace('pool.json', 'pool_feats.npz'))['f']; assert len(pool) == len(pf)
tr = [i for i, r in enumerate(pool) if r['split'] == 'train']; va = [i for i, r in enumerate(pool) if r['split'] == 'val']
X0 = np.concatenate([Xt, Xp, Xw, Xa, Xn16]); y0 = np.concatenate([yt, yp, yw, np.full(len(Xa), 0), np.full(len(Xn16), K)]).astype(int)
ytr = np.array([CLASS_NAMES.index(pool[i]['class_name']) for i in tr])
# ---- CLIP text prompts (zero-shot class logits)
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval(); tok = open_clip.get_tokenizer('ViT-B-16')
P = {'cucumber': 'cucumber', 'potato': 'potato', 'tomato': 'tomato', 'bellpepper': 'bell pepper', 'carrot': 'carrot', 'onion': 'onion', 'sweetpotato': 'sweet potato'}
prompts = []
for c in CLASS_NAMES:
    sp, cond = c.rsplit('_', 1)
    if sp == 'greenonion':
        prompts.append(['a photo of green onions.', 'a photo of scallions.', 'a photo of leeks.', 'a photo of a bunch of green onions.', 'a photo of welsh onions.'] if cond == 'fresh' else ['a photo of rotten green onions.'])
    elif cond == 'fresh':
        w = P[sp]; prompts.append([f'a photo of a fresh {w}.', f'a photo of {w}s.' if not w.endswith('o') else f'a photo of {w}es.', f'a close-up photo of a fresh {w}.'])
    else:
        w = P[sp]; prompts.append([f'a photo of a rotten {w}.', f'a photo of a moldy {w}.', f'a photo of a sprouting {w}.', f'a photo of a spoiled {w}.'])
with torch.inference_mode():
    T = torch.stack([torch.nn.functional.normalize(torch.nn.functional.normalize(clip.encode_text(tok(p)), dim=-1).mean(0), dim=0) for p in prompts]).numpy()
np.save(V + 'text16.npy', T)
def zs(f):
    z = 100 * f @ T.T; z[:, CLASS_NAMES.index('greenonion_rotten')] = 0; z = z - z[:, :15].mean(1, keepdims=True); z[:, 15] = 0
    return z
# ---- vegetable model logits for the development sets and the pool (cached)
cache = V + 'dev_cache.npz'
B0, ck0 = load_model(REPO + 'model.pth', torch.device('cpu')); tf0 = make_transform(input_size(ck0)); h0 = ck0['clip_head']; TB0 = ck0['calibration']['temperature']
def blog(paths):
    with torch.inference_mode(): return torch.cat([B0(torch.stack([tf0(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
sets = json.load(open(S + 'clip/sets.json')); Y = json.load(open(S + 'post/labels.json'))
if os.path.exists(cache): C = dict(np.load(cache, allow_pickle=True))
else: C = {}
def cached(key, paths):
    if key not in C or len(C[key]) != len(paths): C[key] = blog(paths); np.savez(cache, **C)
    return C[key]
D = {'valid': (cached('valid', [p for p, _ in sets['valid']]), F['valid'], np.array([CLASS_NAMES.index(c) for c in Y['valid']]))}
for k in ('odev', 'o160', 'asian37'): D[k] = (cached(k, [p for p, _ in sets[k]]), F[k], np.array([CLASS_NAMES.index(c) for c in Y[k]]))
for k, fk in (('fruit374', 'fruit374'), ('coco400', 'coco400'), ('lvis', 'lvis_veg')): D[k] = (cached(k, [p for p, _ in sets[fk]]), F[fk], None)
bpool = cached('pool_' + str(len(pool)), [r['path'] for r in pool])
D['newval'] = (bpool[va], pf[va], np.array([CLASS_NAMES.index(pool[i]['class_name']) for i in va]))
def now(b, f):
    z0 = f @ h0['coef'].numpy().T + h0['intercept'].numpy()
    return (log_softmax(b / TB0, 1) + log_softmax(z0[:, :10], 1)).argmax(1), z0[:, 10] - logsumexp(z0[:, :10], 1) > h0['unknown_threshold']
def fit(X, y, w=None):
    lr = LogisticRegression(C=10, max_iter=5000).fit(X, y, sample_weight=w)
    coef = np.zeros((K + 1, X.shape[1])); icp = np.full(K + 1, lr.intercept_.min() - 50.0); coef[lr.classes_] = lr.coef_; icp[lr.classes_] = lr.intercept_
    return coef, icp
def weights(y):
    w = np.ones(len(y)); new = [c for c in range(10, 15)]; n = {c: (y == c).sum() for c in new}; avg = np.mean(list(n.values()))
    for c in new: w[y == c] = avg / max(1, n[c])
    return w
def head_logits(f, coef, icp, a):
    z = f @ coef.T + icp
    if a: z[:, :K] += a * zs(f)
    return z
def decide(b, f, coef, icp, a, EC, tau):
    p0, r0 = now(b, f); z = head_logits(f, coef, icp, a); n = len(f)
    sp = np.exp(log_softmax(z[:, :K], 1)).reshape(n, 8, 2).sum(2).argmax(1)
    pn = sp * 2 + z[:, :K].reshape(n, 8, 2)[np.arange(n), sp].argmax(1); rn = z[:, K] - logsumexp(z[:, :K], 1) > EC
    q = softmax(z[:, :K], 1)[:, 10:].sum(1)
    take = (r0 & ~rn & (sp >= 5)) | (~r0 & (q >= tau) & (sp >= 5))
    return np.where(take, pn, p0), r0 & ~take
def calibrate(coef, icp, a):
    nfo = lambda f: (lambda z: z[:, K] - logsumexp(z[:, :K], 1))(head_logits(f, coef, icp, a))
    EC = float(np.quantile(np.concatenate([nfo(D['valid'][1]), nfo(D['newval'][1]), nfo(D['lvis'][1])]), 0.95))
    def moved(tau):
        m = 0
        for k in ('valid', 'odev', 'o160', 'asian37', 'lvis'):
            b, f, _ = D[k]; p0, r0 = now(b, f); p, rej = decide(b, f, coef, icp, a, EC, tau); m += int((~r0 & (p >= 10) & ~rej).sum())
        return m
    grid = {t: moved(t) for t in (0.5, 0.7, 0.8, 0.9, 0.95)}
    return EC, min([t for t, m in grid.items() if m <= 1] or [0.95]), grid
NEWC = list(range(10, 15))
def per_class(p, rej, yy):
    kind = (p // 2 == yy // 2) & ~rej; right = (p == yy) & ~rej
    return {CLASS_NAMES[c]: (int(kind[yy == c].sum()), int(right[yy == c].sum()), int((yy == c).sum())) for c in NEWC if (yy == c).any()}
VARIANTS = {'base': (0, False), 'zs0.1': (0.1, False), 'zs0.25': (0.25, False), 'zs0.5': (0.5, False), 'cb': (0, True), 'cb+zs0.25': (0.25, True)}
if len(sys.argv) > 2: VARIANTS = {k: VARIANTS[k] for k in sys.argv[2].split(',')}
groups = [pool[i]['group'] for i in tr]; fold = np.array([int(hashlib.sha1(('cv:' + g).encode()).hexdigest()[:8], 16) % 5 for g in groups])
res = {}
for name, (a, cb) in VARIANTS.items():
    Xall = np.concatenate([X0, pf[tr]]); yall = np.concatenate([y0, ytr]); w = weights(yall) if cb else None
    coef, icp = fit(Xall, yall, w); EC, tau, grid = calibrate(coef, icp, a)
    out = {'EC': EC, 'tau': tau, 'grid': grid}
    for k, (b, f, yy) in D.items():
        p, rej = decide(b, f, coef, icp, a, EC, tau)
        if yy is None: out[k] = f'rej {int(rej.sum())} new {int(((p >= 10) & ~rej).sum())}'; continue
        if k == 'newval': out[k] = {'kind': int(((p // 2 == yy // 2) & ~rej).sum()), 'of': len(yy), 'by': per_class(p, rej, yy)}; continue
        right = p == yy; out[k] = f'{int(right.sum())}/{int((right & ~rej).sum())} new {int(((p >= 10) & ~rej).sum())}'
    # grouped 5-fold CV over the new-vegetable Train photos (the head is refit without the fold; tau and EC as calibrated above)
    cvp, cvr = np.zeros(len(tr), int), np.zeros(len(tr), bool)
    for k in range(5):
        m = fold != k; Xk = np.concatenate([X0, pf[tr][m]]); yk = np.concatenate([y0, ytr[m]])
        ck_, ik_ = fit(Xk, yk, weights(yk) if cb else None)
        idx = np.where(~m)[0]; p, rej = decide(bpool[np.array(tr)[idx]], pf[np.array(tr)[idx]], ck_, ik_, a, EC, tau); cvp[idx], cvr[idx] = p, rej
    newm = ytr >= 10
    out['cv'] = {'kind': int(((cvp // 2 == ytr // 2) & ~cvr)[newm].sum()), 'of': int(newm.sum()), 'by': per_class(cvp[newm], cvr[newm], ytr[newm])}
    res[name] = out; print(name, json.dumps(out, ensure_ascii=False), flush=True)
    pickle.dump({'coef': coef, 'intercept': icp, 'EC': EC, 'tau': tau, 'alpha': a, 'cb': cb}, open(V + f'head_{name}.pkl', 'wb'))
json.dump(res, open(V + 'explore_' + os.path.basename(os.path.dirname(POOL)) + '.json', 'w'), indent=1, ensure_ascii=False)
