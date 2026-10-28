# Pre-registered final scoring (negx/PREREG.md, N3): new sealed test set (241) and Test (300), once each.
# The current system is recomputed with the same features as a pipeline check (it must reproduce 191 / 165 on the new test set).
import sys, csv, json, pickle, numpy as np, torch, open_clip
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; REPO = '/home/user/depp-env22/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
torch.set_num_threads(4)
B, ck = load_model(REPO + 'model.pth', torch.device('cpu')); tf = make_transform(input_size(ck))
TB, EB = ck['calibration']['temperature'], ck['unknown_check']['threshold']; cur = ck['clip_head']
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
def feats(paths):
    bl, cf = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            bl.append(B(torch.stack([tf(im) for im in ims]))); cf.append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
    return torch.cat(bl).numpy(), torch.cat(cf).numpy()
h = pickle.load(open(S + 'negx/head_N3.pkl', 'rb'))
def score(bl, f, y):
    out = {}
    # current system
    zc = f @ np.asarray(cur['coef']).T + np.asarray(cur['intercept'])
    rc = (-logsumexp(bl, 1) > EB) | (-logsumexp(zc, 1) > cur['unknown_threshold'])
    pc = (log_softmax(bl / TB, 1) + log_softmax(zc, 1)).argmax(1)
    # N3
    z = f @ h['coef'].T + h['intercept']
    rn = (-logsumexp(bl, 1) > EB) | (z[:, 10] - logsumexp(z[:, :10], 1) > h['EC'])
    pn = (log_softmax(bl / TB, 1) + log_softmax(z[:, :10], 1)).argmax(1)
    for name, p, r in (('current', pc, rc), ('N3', pn, rn)):
        right = p == y
        out[name] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~r).sum()), 'rejected': int(r.sum()),
                     'by_class': {n: f'{int(right[y == i].sum())}/{int((y == i).sum())} (kept {int((right & ~r)[y == i].sum())})' for i, n in enumerate(NAMES) if (y == i).any()}}
    out['fixed'] = int((pn == y).sum() - ((pn == y) & (pc == y)).sum()); out['broken'] = int((pc == y).sum() - ((pn == y) & (pc == y)).sum())
    return out
# new sealed test set
t2 = list(csv.DictReader(open(S + 'ext3/outside_test2/outside_test2.csv', encoding='utf-8')))
bl, f = feats([S + f"weak/t2img/{r['id']}.jpg" for r in t2]); y = np.array([NAMES.index(r['class_name']) for r in t2])
res = {'new_test_241': score(bl, f, y)}
assert res['new_test_241']['current']['accuracy'] == 191 and res['new_test_241']['current']['kept_correct'] == 165, res['new_test_241']['current']
by_src = {}
for src in ('openverse', 'inaturalist'):
    m = np.array([r['source'] == src for r in t2]); by_src[src] = score(bl[m], f[m], y[m])
res['new_test_by_source'] = {k: {'current': v['current']['accuracy'], 'N3': v['N3']['accuracy'], 'n': int(sum(r['source'] == k for r in t2))} for k, v in by_src.items()}
# Test
T = np.load(S + 'clipcls/test_features.npz')
rows = [r for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test']
yt = np.array([NAMES.index(r['class_name']) for r in rows])
res['test_300'] = score(T['b_logits'], T['clip'], yt)
assert res['test_300']['current']['accuracy'] == 284, res['test_300']['current']
n, t = res['new_test_241']['N3'], res['test_300']['N3']
res['adopt'] = n['accuracy'] >= 194 and n['kept_correct'] >= 165 and t['accuracy'] >= 284 and t['kept_correct'] >= 280
json.dump(res, open(S + 'negx/final_once.json', 'w'), indent=1); print(json.dumps(res, indent=1))
