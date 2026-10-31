# Pre-registered (candF/PREREG.md): the chosen candidate (F1 or F2) and the current system (N3 + R1), scored once on the sealed sets:
# new internet test 241, Test 300, and the new sealed "any photo" set (scored for both systems at the same time).
import sys, csv, json, pickle, collections, numpy as np, torch, open_clip
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; REPO = '/home/user/depp-env22/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
REJECT = ('rotten_fruit', 'fresh_fruit', 'other_veg', 'food', 'drawing_toy', 'everyday')
torch.set_num_threads(4)
chosen = 'F1'   # user decision recorded at the end of PREREG.md (development check skipped for F1 only)
hf = pickle.load(open(S + 'candF/head_F.pkl', 'rb'))
B, ck = load_model(REPO + 'model.pth', torch.device('cpu')); tf = make_transform(input_size(ck)); TB, h = ck['calibration']['temperature'], ck['clip_head']
systems = {'now': dict(W=h['coef'].numpy(), I=h['intercept'].numpy(), EC=h['unknown_threshold'], w=1.0),
           chosen: dict(W=hf['coef'], I=hf['intercept'], EC=hf['EC'], w=1.0 if chosen == 'F1' else hf['w2'])}
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
def feats(paths):
    bl, cf = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            bl.append(B(torch.stack([tf(im) for im in ims]))); cf.append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
    return torch.cat(bl).numpy(), torch.cat(cf).numpy()
def decide(s, bl, f):
    z = f @ s['W'].T + s['I']
    pred = (s['w'] * log_softmax(bl / TB, 1) + log_softmax(z[:, :10], 1)).argmax(1)
    return pred, z[:, 10] - logsumexp(z[:, :10], 1) > s['EC']
def score_veg(bl, f, y):
    out = {}
    for k, s in systems.items():
        pred, rej = decide(s, bl, f); right = pred == y
        out[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'kept_wrong': int((~right & ~rej).sum()),
                  'by_class': {n: f'{int(right[y == i].sum())}/{int((y == i).sum())}' for i, n in enumerate(NAMES) if (y == i).any()}}
    return out
res = {'chosen': chosen}
t2 = list(csv.DictReader(open(REPO + 'results/10class/outside_test2/outside_test2.csv', encoding='utf-8')))
bl, f = feats([S + f"weak/t2img/{r['id']}.jpg" for r in t2]); y = np.array([NAMES.index(r['class_name']) for r in t2])
res['new_test_241'] = score_veg(bl, f, y)
assert (res['new_test_241']['now']['accuracy'], res['new_test_241']['now']['kept_correct']) == (203, 193), res['new_test_241']['now']
T = np.load(S + 'clipcls/test_features.npz')
yt = np.array([NAMES.index(r['class_name']) for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test'])
res['test_300'] = score_veg(T['b_logits'], T['clip'], yt)
assert (res['test_300']['now']['accuracy'], res['test_300']['now']['kept_correct']) == (288, 285), res['test_300']['now']
# any-photo set
rows = list(csv.DictReader(open(S + 'arb/anyphoto_test.csv', encoding='utf-8'))); files = json.load(open(S + 'arb/local_files.json'))
bl, f = feats([files[r['id']] for r in rows])
res['any_photo'] = {}
for k, s in systems.items():
    pred, rej = decide(s, bl, f); good = []
    for r, p, j in zip(rows, pred, rej):
        lab = r['label'].split('|') if r['label'] else []
        if r['category'] in REJECT: g = bool(j)
        elif r['category'] == 'five_daily': g = (not j) and NAMES[p] == lab[0]
        elif r['category'] == 'cut_five': g = bool(j) or NAMES[p] == lab[0]
        else: g = bool(j) or NAMES[p] in lab                     # several
        good.append(g)
    good = np.array(good); cats = collections.Counter(r['category'] for r in rows)
    res['any_photo'][k] = {'good': int(good.sum()), 'of': len(rows),
                           'by_category': {c: f"{int(good[[i for i, r in enumerate(rows) if r['category'] == c]].sum())}/{n}" for c, n in cats.items()},
                           'answers': [{'id': r['id'], 'category': r['category'], 'label': r['label'], 'pred': NAMES[p], 'rejected': bool(j), 'good': bool(g)}
                                       for r, p, j, g in zip(rows, pred, rej, good)]}
n, nn, t, a, a0 = res['new_test_241'][chosen], res['new_test_241']['now'], res['test_300'][chosen], res['any_photo'][chosen], res['any_photo']['now']
daily = lambda x: int(x['by_category']['five_daily'].split('/')[0])
res['adopt'] = bool(n['accuracy'] >= (206 if chosen == 'F2' else 203) and n['kept_correct'] >= 191 and t['accuracy'] >= 286 and t['kept_correct'] >= 283
                    and a['good'] >= a0['good'] + 8 and daily(a) >= daily(a0) - 2)
json.dump(res, open(S + 'candF/final_once.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: (v if k != 'any_photo' else {s: {kk: vv for kk, vv in x.items() if kk != 'answers'} for s, x in v.items()}) for k, v in res.items()}, indent=1, ensure_ascii=False))
