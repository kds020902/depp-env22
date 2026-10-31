# Pre-registered (candF/PREREG.md): CLIP head N3 + farm tomatoes (vegetable side) + rotten/fresh fruit (not-five side).
# F1 keeps the 1:1 combination; F2 refits the vegetable model's weight w and temperature T as in tom/PREREG.md (W).
# Development checks only. Rejection stays R1 (CLIP not-five log-odds above the threshold).
import sys, csv, json, pickle, collections, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
from scipy.optimize import minimize_scalar
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'
src = open(S + 'negx/evaluate.py').read(); exec(src[:src.index("ck = torch.load")])   # Xt, Xp, Xw, Xa, Xn and copy checks exactly as N3
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size, fit_temperature
torch.set_num_threads(4)
TB = 0.5283377766609192
C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json')); yv = np.array([NAMES.index(c) for c in Y['valid']])
any_test = np.load(S + 'arb/anyphoto_features.npz')['f']           # the sealed "any photo" set: no copies in training
chk = np.concatenate([checks, any_test])
dropx = lambda X: (X @ chk.T).max(1) >= 0.95

# --- additions ---
pool = list(csv.DictReader(open(S + 'arb/pool/pool.csv', encoding='utf-8'))); pf = np.load(S + 'arb/pool/feats.npz')['f']
keep = set(np.load(S + 'arb/pool/keep_idx.npy').tolist()); bad = set(json.load(open(S + 'arb/pool/review.json'))['mendeley_drop'])
mi = [i for i, r in enumerate(pool) if i in keep and r['group'] == 'mendeley_tomato' and r['path'] not in bad]
Xm = pf[mi]; ym = np.array([NAMES.index('tomato_fresh' if pool[i]['label'] == 'Fresh' else 'tomato_rotten') for i in mi]); mpaths = [pool[i]['path'] for i in mi]
ki = [i for i, r in enumerate(pool) if i in keep and r['group'] == 'kaggle_fruit']; Xk = pf[ki]
inf = np.load(S + 'arb/inat/features.npz'); iidx = {str(n): i for i, n in enumerate(inf['ids'])}
Xi = inf['f'][[iidx[i] for i in json.load(open(S + 'arb/pool/inat_train_keep.json'))]]
ov = np.load(S + 'arb/ov_train_fruit.npz'); Xo = ov['f']                # Openverse rotten-fruit photos from training-side uploaders, reviewed
dm, dk, di, do = dropx(Xm), dropx(Xk), dropx(Xi), dropx(Xo)
print('copies of check photos: farm tomato', int(dm.sum()), 'kaggle fruit', int(dk.sum()), 'inat fruit', int(di.sum()), 'openverse fruit', int(do.sum()))
Xm, ym, mpaths = Xm[~dm], ym[~dm], [p for p, d in zip(mpaths, dm) if not d]
Xf = np.concatenate([Xk[~dk], Xi[~di], Xo[~do]])
print('farm tomato', collections.Counter(NAMES[c] for c in ym), 'fruit (not five)', len(Xf), '= kaggle', int((~dk).sum()), 'inat', int((~di).sum()), 'openverse', int((~do).sum()))

# vegetable-model logits for the farm tomatoes (other extras are cached from tom/)
try: BM = np.load(S + 'candF/farm_b_logits.npy')
except FileNotFoundError:
    B, ck = load_model(REPO + 'model.pth', torch.device('cpu')); tf = make_transform(input_size(ck))
    with torch.inference_mode(): BM = torch.cat([B(torch.stack([tf(open_image(p)) for p in mpaths[i:i + 32]])) for i in range(0, len(mpaths), 32)]).numpy()
    np.save(S + 'candF/farm_b_logits.npy', BM)
BE = np.concatenate([np.load(S + 'tom/extra_b_logits.npy'), BM])

Xe = np.concatenate([Xp, Xw, Xa, Xm]); ye = np.concatenate([yp, yw, np.full(len(Xa), NAMES.index('cucumber_fresh')), ym])
XN = np.concatenate([Xn, Xf]); yN = np.full(len(XN), OTHER)
X = np.concatenate([Xt, Xe, XN]); y = np.concatenate([yt, ye, yN])
lr = LogisticRegression(C=10, max_iter=5000).fit(X, y)
key = {'valid': ('valid', 'valid'), 'lvis': ('lvis192', 'lvis_veg'), 'odev': ('odev', 'odev'), 'o160': ('o160', 'o160'), 'asian': ('asian37', 'asian37'),
       'fruit': ('fruit374', 'fruit374'), 'kfruit': ('mu400', 'mu400'), 'coco': ('coco400', 'coco400')}
other = lambda z: z[:, OTHER] - logsumexp(z[:, :OTHER], 1)
cl = lambda k: lr.decision_function(F[key[k][1]])
EC = float(np.quantile(np.concatenate([other(cl('valid')), other(cl('lvis'))]), 0.95))

# F2: w and T on held-out predictions (Valid + out-of-fold CLIP logits for the vegetable-side extras), as in tom/PREREG.md
fold = np.random.default_rng(0).permutation(len(ye)) % 5; ZE = np.zeros((len(ye), 11))
for k in range(5):
    tr = fold != k
    m = LogisticRegression(C=10, max_iter=5000).fit(np.concatenate([Xt, Xe[tr], XN]), np.concatenate([yt, ye[tr], yN]))
    ZE[fold == k] = m.decision_function(Xe[fold == k]); print('fold', k, flush=True)
LB = np.concatenate([log_softmax(C['B/valid/logit'] / TB, 1), log_softmax(BE / TB, 1)])
LC = np.concatenate([log_softmax(cl('valid')[:, :OTHER], 1), log_softmax(ZE[:, :OTHER], 1)]); YF = np.concatenate([yv, ye])
nll = lambda w, T: float(-log_softmax((w * LB + LC) / T, 1)[np.arange(len(YF)), YF].mean())
grid = {}
for w in np.round(np.arange(0, 1.5001, 0.05), 2):
    r = minimize_scalar(lambda T: nll(w, T), bounds=(0.2, 5), method='bounded'); grid[float(w)] = (float(r.x), float(r.fun))
w2 = min(grid, key=lambda w: grid[w][1]); T2 = grid[w2][0]
T1 = fit_temperature(torch.tensor(log_softmax(C['B/valid/logit'] / TB, 1) + log_softmax(cl('valid')[:, :OTHER], 1)), torch.tensor(yv))

def check(w):
    out = {}
    for k, (bk, _) in key.items():
        bl = C[f'B/{bk}/logit']; c = cl(k); rej = other(c) > EC
        if k in ('fruit', 'kfruit', 'coco', 'lvis'): out[k] = {'rejected': int(rej.sum()), 'of': len(c)}; continue
        yy = np.array([NAMES.index(v) for v in Y[bk]]); right = (w * log_softmax(bl / TB, 1) + log_softmax(c[:, :OTHER], 1)).argmax(1) == yy
        out[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()),
                  'tomato_fresh': f'{int(right[yy == 4].sum())}/{int((yy == 4).sum())}', 'tomato_rotten': f'{int(right[yy == 5].sum())}/{int((yy == 5).sum())}'}
    return out
res = {'train_photos': {'vegetable': int((y < OTHER).sum()), 'not_five': int((y == OTHER).sum()), 'farm_tomato': len(ym), 'fruit_added': len(Xf)},
       'clip_threshold': EC, 'F1': {'w': 1.0, 'T': T1, **check(1.0)}, 'F2': {'w': w2, 'T': T2, 'fusion_nll': grid[w2][1], 'nll_w1': grid[1.0][1], **check(w2)}}
def passes(r, need_odev_up):
    return (r['valid']['accuracy'] >= 289 and r['valid']['kept_correct'] >= 285 and r['o160']['accuracy'] >= 148 and r['o160']['kept_correct'] >= 148
            and r['asian']['accuracy'] >= 32 and r['odev']['kept_correct'] >= 188 and (r['odev']['accuracy'] > 199 if need_odev_up else r['odev']['accuracy'] >= 197)
            and r['fruit']['rejected'] >= 337 and r['kfruit']['rejected'] >= 284 and r['coco']['rejected'] >= 398)
res['F1']['checks_pass'] = passes(res['F1'], False); res['F2']['checks_pass'] = passes(res['F2'], True)
res['chosen'] = 'F2' if res['F2']['checks_pass'] else 'F1' if res['F1']['checks_pass'] else None
pickle.dump({'coef': lr.coef_, 'intercept': lr.intercept_, 'EC': EC, 'T1': T1, 'w2': w2, 'T2': T2}, open(S + 'candF/head_F.pkl', 'wb'))
json.dump(res, open(S + 'candF/check_results.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items()}, indent=1))
