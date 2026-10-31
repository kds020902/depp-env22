# Development only: V5-H and V6-H (aug / noaug) on the same new-vegetable Valid photos (v6 pool, 73).
import sys, json, pickle, collections, numpy as np, torch
from scipy.special import log_softmax, logsumexp, softmax
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; REPO = '/home/user/depp-env22/'
sys.path.insert(0, REPO + 'src')
from model_utils import load_model, make_transform, open_image, input_size, CLASS_NAMES
pool = json.load(open(S + 'v6/pool.json')); pf = np.load(S + 'v6/pool_feats.npz')['f']
va = [i for i, r in enumerate(pool) if r['split'] == 'val']; y = np.array([CLASS_NAMES.index(pool[i]['class_name']) for i in va]); f = pf[va]
B0, ck0 = load_model(REPO + 'model.pth', torch.device('cpu')); tf0 = make_transform(input_size(ck0)); h0 = ck0['clip_head']; TB0 = ck0['calibration']['temperature']
with torch.inference_mode(): b = torch.cat([B0(torch.stack([tf0(open_image(pool[i]['path'])) for i in va[k:k + 32]])) for k in range(0, len(va), 32)]).numpy()
z0 = f @ h0['coef'].numpy().T + h0['intercept'].numpy(); p0 = (log_softmax(b / TB0, 1) + log_softmax(z0[:, :10], 1)).argmax(1); r0 = z0[:, 10] - logsumexp(z0[:, :10], 1) > h0['unknown_threshold']
for name, hv in (('V5-H', dict(pickle.load(open(S + 'v5/head_v5.pkl', 'rb')), tau=0.7)), ('V6-H noaug', pickle.load(open(S + 'v6/head_v6_noaug.pkl', 'rb'))), ('V6-H aug', pickle.load(open(S + 'v6/head_v6_aug.pkl', 'rb')))):
    z = f @ hv['coef'].T + hv['intercept']; n = len(f); sp = np.exp(log_softmax(z[:, :16], 1)).reshape(n, 8, 2).sum(2).argmax(1)
    pn = sp * 2 + z[:, :16].reshape(n, 8, 2)[np.arange(n), sp].argmax(1); rn = z[:, 16] - logsumexp(z[:, :16], 1) > hv['EC']; q = softmax(z[:, :16], 1)[:, 10:].sum(1)
    take = (r0 & ~rn & (sp >= 5)) | (~r0 & (q >= hv['tau']) & (sp >= 5)); p = np.where(take, pn, p0); rej = r0 & ~take
    kind = (p // 2 == y // 2) & ~rej; cls = (p == y) & ~rej
    print(name, 'kind', int(kind.sum()), '/', n, {CLASS_NAMES[c]: f'{int(kind[y == c].sum())}/{int(cls[y == c].sum())}/{int((y == c).sum())}' for c in sorted(set(y))})
