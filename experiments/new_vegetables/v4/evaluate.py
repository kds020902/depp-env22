# Pre-registered (v4/PREREG.md): 8 vegetables (16 slots). Development checks only: Test, the new internet test, the any-photo
# set and the new-vegetable test are not touched here. Uses the chosen vegetable model (argv[1]: checkpoint with calibration).
import sys, csv, json, pickle, collections, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v4/'
src = open(S + 'negx/evaluate.py').read(); exec(src[:src.index("ck = torch.load")])   # Xt, Xp, Xw, Xa, Xn (+ copy checks) exactly as N3
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size, fit_temperature, CLASS_NAMES, UNTRAINED_CLASSES
torch.set_num_threads(4)
K = len(CLASS_NAMES); OTHER16 = K
# not-five photos minus the kritikseth onions and sweet potatoes (now trained vegetables)
nf_paths = [str(p) for k in ('fruit', 'everyday', 'otherveg') for p in np.load(S + f'neg/{k}_feats.npz')['paths']]
assert len(nf_paths) == len(Xn) + int(dn.sum())
nf_paths = [p for p, d in zip(nf_paths, dn) if not d]
keep_n = np.array(['/otherveg/onion__' not in p and '/otherveg/sweetpotato__' not in p for p in nf_paths])
Xn16 = Xn[keep_n]
pool = json.load(open(V + 'pool_with_ids.json')); pf = np.load(V + 'pool_feats.npz')['f']
tr = [i for i, r in enumerate(pool) if r['split'] == 'train']; va = [i for i, r in enumerate(pool) if r['split'] == 'val']
Xv4 = pf[tr]; yv4 = np.array([CLASS_NAMES.index(pool[i]['class_name']) for i in tr])
X = np.concatenate([Xt, Xp, Xw, Xa, Xv4, Xn16]); y = np.concatenate([yt, yp, yw, np.full(len(Xa), 0), yv4, np.full(len(Xn16), OTHER16)])
lr = LogisticRegression(C=10, max_iter=5000).fit(X, y)
coef = np.zeros((K + 1, X.shape[1])); icp = np.full(K + 1, lr.intercept_.min() - 50.0)
coef[lr.classes_] = lr.coef_; icp[lr.classes_] = lr.intercept_
head = lambda f: f @ coef.T + icp
other = lambda z: z[:, K] - logsumexp(z[:, :K], 1)
# vegetable model logits for every development photo
B, ck = load_model(sys.argv[1], torch.device('cpu')); tf = make_transform(input_size(ck)); TB = ck['calibration']['temperature']
def blog(paths):
    with torch.inference_mode(): return torch.cat([B(torch.stack([tf(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
sets = json.load(open(S + 'clip/sets.json')); Y = json.load(open(S + 'post/labels.json'))
D = {}
vpaths = [p for p, _ in sets['valid']] + [V + 'ds/' + pool[i]['image_path'] for i in va]
D['valid'] = (blog(vpaths), np.concatenate([F['valid'], pf[va]]), np.array([CLASS_NAMES.index(c) for c in Y['valid']] + [CLASS_NAMES.index(pool[i]['class_name']) for i in va]))
for k in ('odev', 'o160', 'asian37'):
    D[k] = (blog([p for p, _ in sets[k]]), F[k], np.array([CLASS_NAMES.index(c) for c in Y[k]]))
for k, fk in (('fruit374', 'fruit374'), ('coco400', 'coco400'), ('lvis', 'lvis_veg')):
    D[k] = (blog([p for p, _ in sets[fk]]), F[fk], None)
EC = float(np.quantile(np.concatenate([other(head(D['valid'][1])), other(head(D['lvis'][1]))]), 0.95))
comb = lambda b, f: log_softmax(b / TB, 1) + log_softmax(head(f)[:, :K], 1)
T = fit_temperature(torch.tensor(comb(*D['valid'][:2])), torch.tensor(D['valid'][2]))
res = {'train_photos': {'clip_vegetable': int((y < K).sum()), 'not_five': int((y == K).sum()), 'new_vegetables_train': len(tr)}, 'clip_threshold': EC, 'temperature': T,
       'vegetable_model': {k: ck.get(k) for k in ('best_epoch',)}}
for k, (b, f, yy) in D.items():
    rej = other(head(f)) > EC
    if yy is None: res[k] = {'rejected': int(rej.sum()), 'of': len(rej)}; continue
    p = comb(b, f).argmax(1); right = p == yy
    r = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'of': len(yy)}
    if k == 'valid':
        old = np.arange(len(yy)) < 300
        r = {'old300': {'accuracy': int(right[old].sum()), 'kept_correct': int((right & ~rej)[old].sum())},
             'new35': {'species_kept': int(((p // 2 == yy // 2) & ~rej)[~old].sum()), 'class_right': int(right[~old].sum()), 'of': int((~old).sum()),
                       'answers': collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in zip(yy[~old], p[~old], rej[~old])).most_common()},
             'old300_new_species_answers': int((p[old] >= 10).sum())}
    if k in ('odev', 'o160', 'asian37'): r['answered_new_species'] = int((p >= 10).sum())
    res[k] = r
v = res['valid']['old300']
res['checks_pass'] = bool(v['accuracy'] >= 288 and v['kept_correct'] >= 284 and res['o160']['accuracy'] >= 147 and res['odev']['accuracy'] >= 195
                          and res['odev']['kept_correct'] >= 186 and res['asian37']['accuracy'] >= 31 and res['fruit374']['rejected'] >= 320 and res['coco400']['rejected'] >= 395)
pickle.dump({'coef': coef, 'intercept': icp, 'EC': EC, 'T': T}, open(V + 'head_v4.pkl', 'wb'))
json.dump(res, open(V + 'check_results.json', 'w'), indent=1); print(json.dumps(res, indent=1))
