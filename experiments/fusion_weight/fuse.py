# Pre-registered (tom/PREREG.md): fit the vegetable model's weight w (and temperature T) in the combination on held-out predictions
# (Valid + out-of-fold CLIP predictions on the 1,773 extra training photos), then run the development checks.
import sys, csv, json, pickle, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
from scipy.optimize import minimize_scalar
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'
src = open(S + 'negx/evaluate.py').read(); exec(src[:src.index("ck = torch.load")])
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size
torch.set_num_threads(4)
C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json'))
yv = np.array([NAMES.index(c) for c in Y['valid']])
Xe = np.concatenate([Xp, Xw, Xa]); ye = np.concatenate([yp, yw, np.zeros(len(Xa), int)])
paths = [r['path'] for r in csv.DictReader(open(S + 'negx/extra_train_N3.csv', encoding='utf-8')) if r['class_name'] != 'not_five']
labels = [r['class_name'] for r in csv.DictReader(open(S + 'negx/extra_train_N3.csv', encoding='utf-8')) if r['class_name'] != 'not_five']
assert len(paths) == len(ye) and all(NAMES.index(c) == k for c, k in zip(labels, ye)), (len(paths), len(ye))
try:
    BE = np.load(S + 'tom/extra_b_logits.npy')
except FileNotFoundError:
    B, ck = load_model(REPO + 'model.pth', torch.device('cpu')); tf = make_transform(input_size(ck))
    with torch.inference_mode(): BE = torch.cat([B(torch.stack([tf(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
    np.save(S + 'tom/extra_b_logits.npy', BE)
TB = 0.5283377766609192
# out-of-fold CLIP logits for the extra photos
fold = np.random.default_rng(0).permutation(len(ye)) % 5
try:
    ZE = np.load(S + 'tom/extra_oof_clip.npy')
except FileNotFoundError:
    ZE = np.zeros((len(ye), 11))
    for k in range(5):
        tr = fold != k
        lr = LogisticRegression(C=10, max_iter=5000).fit(np.concatenate([Xt, Xe[tr], Xn]), np.concatenate([yt, ye[tr], np.full(len(Xn), OTHER)]))
        ZE[fold == k] = lr.decision_function(Xe[fold == k]); print('fold', k, flush=True)
    np.save(S + 'tom/extra_oof_clip.npy', ZE)
h = pickle.load(open(S + 'negx/head_N3.pkl', 'rb'))
ZV = F['valid'] @ h['coef'].T + h['intercept']
LB = np.concatenate([log_softmax(C['B/valid/logit'] / TB, 1), log_softmax(BE / TB, 1)])
LC = np.concatenate([log_softmax(ZV[:, :10], 1), log_softmax(ZE[:, :10], 1)])
YF = np.concatenate([yv, ye])
def nll(w, T): return float(-log_softmax((w * LB + LC) / T, 1)[np.arange(len(YF)), YF].mean())
best = None; grid = {}
for w in np.round(np.arange(0, 1.5001, 0.05), 2):
    r = minimize_scalar(lambda T: nll(w, T), bounds=(0.2, 5), method='bounded'); grid[float(w)] = (float(r.x), float(r.fun))
    if best is None or r.fun < best[2]: best = (float(w), float(r.x), float(r.fun))
w, T, loss = best
print('chosen w %.2f T %.3f NLL %.4f (w=1: NLL %.4f)' % (w, T, loss, grid[1.0][1]), flush=True)
print('fusion-set accuracy: w=1 %d, chosen %d of %d' % (int(((LB + LC).argmax(1) == YF).sum()), int(((w * LB + LC).argmax(1) == YF).sum()), len(YF)))
# development checks (rejection unchanged: CLIP not-five check only)
out = {'w': w, 'T': T, 'fusion_nll': loss, 'nll_grid': grid}
for name, bk in (('valid', 'valid'), ('odev', 'odev'), ('o160', 'o160'), ('asian37', 'asian37')):
    y = np.array([NAMES.index(v) for v in Y[bk]]); bl = C[f'B/{bk}/logit']; z = F[name] @ h['coef'].T + h['intercept']
    rej = z[:, 10] - logsumexp(z[:, :10], 1) > h['EC']
    r = {}
    for tag, ww in (('now', 1.0), ('W', w)):
        p = (ww * log_softmax(bl / TB, 1) + log_softmax(z[:, :10], 1)).argmax(1); right = p == y
        r[tag] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()),
                  'tomato_rotten': f"{int(right[y == 5].sum())}/{int((y == 5).sum())}", 'tomato_fresh': f"{int(right[y == 4].sum())}/{int((y == 4).sum())}"}
    out[name] = r; print(name, r, flush=True)
out['checks_pass'] = (out['valid']['W']['accuracy'] >= out['valid']['now']['accuracy'] - 2 and out['o160']['W']['accuracy'] >= out['o160']['now']['accuracy'] - 2
                      and out['asian37']['W']['accuracy'] >= out['asian37']['now']['accuracy'] - 2 and out['odev']['W']['accuracy'] > out['odev']['now']['accuracy'])
print('checks_pass', out['checks_pass'])
json.dump(out, open(S + 'tom/check_results.json', 'w'), indent=1)
