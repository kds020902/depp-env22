# v6 development check (no sealed set is touched). 16-slot CLIP head trained with the v6 pool; classes under the average Train count
# of the five new classes are topped up to that average with augmented copies of their own Train photos (random crop 60-100%,
# flip, mild colour jitter; seed 0). The decision is V5-H's; tau is chosen by the same rule as V5-H (smallest of 0.5/0.7/0.8/0.9/0.95
# that moves at most 1 old-vegetable development photo the current system answered to a new vegetable). argv[1] = 'aug' or 'noaug'.
import sys, csv, json, pickle, random, collections, numpy as np, torch, open_clip
from sklearn.linear_model import LogisticRegression
from scipy.special import log_softmax, logsumexp, softmax
from torchvision import transforms
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; V = S + 'v6/'; MODE = sys.argv[1]
src = open(S + 'negx/evaluate.py').read(); exec(src[:src.index("ck = torch.load")])
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size, CLASS_NAMES
torch.set_num_threads(4); K = 16
nf_paths = [str(p) for k in ('fruit', 'everyday', 'otherveg') for p in np.load(S + f'neg/{k}_feats.npz')['paths']]
nf_paths = [p for p, d in zip(nf_paths, dn) if not d]
Xn16 = Xn[np.array(['/otherveg/onion__' not in p and '/otherveg/sweetpotato__' not in p for p in nf_paths])]
pool = json.load(open(V + 'pool.json')); pf = np.load(V + 'pool_feats.npz')['f']; assert len(pool) == len(pf)
tr = [i for i, r in enumerate(pool) if r['split'] == 'train']; va = [i for i, r in enumerate(pool) if r['split'] == 'val']
print('copies of check photos in the pool train:', int(drop(pf[tr]).sum()))
NEWC = ['onion_fresh', 'onion_rotten', 'sweetpotato_fresh', 'sweetpotato_rotten', 'greenonion_fresh']
ntr = {c: sum(pool[i]['class_name'] == c for i in tr) for c in NEWC}; target = round(np.mean(list(ntr.values())))
Xa_, ya_ = [], []
if MODE == 'aug':
    clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
    aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.1)])
    random.seed(0); torch.manual_seed(0)
    for c in NEWC:
        paths = [pool[i]['path'] for i in tr if pool[i]['class_name'] == c]
        for k in range(max(0, target - len(paths))):
            im = open_image(paths[k % len(paths)]); im = im.resize((max(224, im.width), max(224, im.height))) if min(im.size) < 224 else im
            with torch.inference_mode(): Xa_.append(torch.nn.functional.normalize(clip.encode_image(prep(aug(im)).unsqueeze(0)), dim=-1)[0].numpy())
            ya_.append(CLASS_NAMES.index(c))
print('new-class Train', ntr, 'average', target, 'augmented copies', collections.Counter(CLASS_NAMES[c] for c in ya_))
Xaug = np.array(Xa_).reshape(-1, Xt.shape[1])
X = np.concatenate([Xt, Xp, Xw, Xa, pf[tr], Xaug, Xn16]); y = np.concatenate([yt, yp, yw, np.full(len(Xa), 0), [CLASS_NAMES.index(pool[i]['class_name']) for i in tr], ya_, np.full(len(Xn16), K)]).astype(int)
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
def dec(b, f, tau):
    p0, r0 = now(b, f)
    if tau is None: return p0, r0
    z = hz(f); n = len(f); sp = np.exp(log_softmax(z[:, :K], 1)).reshape(n, 8, 2).sum(2).argmax(1)
    pn = sp * 2 + z[:, :K].reshape(n, 8, 2)[np.arange(n), sp].argmax(1); rn = nfo(z) > EC; q = softmax(z[:, :K], 1)[:, 10:].sum(1)
    take = (r0 & ~rn & (sp >= 5)) | (~r0 & (q >= tau) & (sp >= 5))
    return np.where(take, pn, p0), r0 & ~take
def moved(tau):   # old-vegetable dev photos the current system answered that tau's rule (b) moves to a new vegetable
    m = 0
    for k in ('valid', 'odev', 'o160', 'asian37', 'lvis'):
        b, f, _ = D[k]; p0, r0 = now(b, f); p, rej = dec(b, f, tau); m += int((~r0 & (p >= 10) & ~rej).sum())
    return m
grid = {t: moved(t) for t in (0.5, 0.7, 0.8, 0.9, 0.95)}; TAU = min([t for t, m in grid.items() if m <= 1] or [0.95])
res = {'mode': MODE, 'new_class_train': ntr, 'average': target, 'augmented': dict(collections.Counter(CLASS_NAMES[c] for c in ya_)), 'EC_new': EC, 'tau_grid_moved': grid, 'tau': TAU}
for name, tau in (('now', None), ('V6-H', TAU)):
    out = {}
    for k, (b, f, yy) in D.items():
        p, rej = dec(b, f, tau)
        if yy is None: out[k] = {'rejected': int(rej.sum()), 'of': len(rej), 'answered_new_species': int(((p >= 10) & ~rej).sum())}; continue
        right = p == yy
        if k == 'newval':
            kind = (p // 2 == yy // 2) & ~rej
            out[k] = {'kind_right_kept': int(kind.sum()), 'of': len(yy), 'by_class': {CLASS_NAMES[c]: f'{int(kind[yy == c].sum())}/{int((yy == c).sum())}' for c in sorted(set(yy))},
                      'misses': collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in zip(yy, p, rej) if not (q // 2 == a // 2 and not j)).most_common()}
        else: out[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'answered_new_species': int(((p >= 10) & ~rej).sum()), 'of': len(yy)}
    res[name] = out
h = res['V6-H']
res['checks_pass'] = bool(h['valid']['accuracy'] >= 288 and h['valid']['kept_correct'] >= 284 and h['o160']['accuracy'] >= 147 and h['odev']['accuracy'] >= 195
                          and h['odev']['kept_correct'] >= 186 and h['asian37']['accuracy'] >= 31 and h['fruit374']['rejected'] >= 320 and h['coco400']['rejected'] >= 395)
pickle.dump({'coef': coef, 'intercept': icp, 'EC': EC, 'tau': TAU}, open(V + f'head_v6_{MODE}.pkl', 'wb'))
json.dump(res, open(V + f'check_{MODE}.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ('new_class_train', 'average', 'augmented', 'tau_grid_moved', 'tau', 'checks_pass')}, ensure_ascii=False)); print(json.dumps(res['V6-H'], ensure_ascii=False))
