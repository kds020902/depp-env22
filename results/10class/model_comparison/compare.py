# Same photos, five systems: initial recipe (ResNet18, last layer only), improved 1 (EfficientNet-B0), improved 2 (+ CLIP head, 10 classes),
# current (N3 + R1), and candidate F1. Reporting only: no system is chosen from these numbers.
import sys, csv, json, pickle, collections, numpy as np, torch, open_clip
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; REPO = '/home/user/depp-env22/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
REJECT = ('rotten_fruit', 'fresh_fruit', 'other_veg', 'food', 'drawing_toy', 'everyday')
torch.set_num_threads(4)
R, ckr = load_model(S + 'cmp/resnet18_linear_10/model.pth', torch.device('cpu')); tfr = make_transform(input_size(ckr))
B, ck = load_model(REPO + 'model.pth', torch.device('cpu')); tf = make_transform(input_size(ck))
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
sets = json.load(open(S + 'clip/sets.json')); F = np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz'); C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json'))
def run(paths, need=('R', 'B', 'F')):
    out = {k: [] for k in need}
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            if 'R' in need: out['R'].append(R(torch.stack([tfr(im) for im in ims])))
            if 'B' in need: out['B'].append(B(torch.stack([tf(im) for im in ims])))
            if 'F' in need: out['F'].append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
    return {k: torch.cat(v).numpy() for k, v in out.items()}
try: D = dict(np.load(S + 'cmp/outputs.npz', allow_pickle=True))['D'].item()
except FileNotFoundError:
    D = {}
    test = [r for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test']
    T = np.load(S + 'clipcls/test_features.npz')
    D['test'] = dict(run([REPO + 'dataset/' + r['image_path'] for r in test], ('R',)), B=T['b_logits'], F=T['clip'], y=[r['class_name'] for r in test])
    t2 = list(csv.DictReader(open(REPO + 'results/10class/outside_test2/outside_test2.csv', encoding='utf-8')))
    D['new'] = dict(run([S + f"weak/t2img/{r['id']}.jpg" for r in t2]), y=[r['class_name'] for r in t2])
    for k in ('odev', 'o160', 'fruit374', 'coco400'):
        D[k] = dict(run([p for p, _ in sets[k]], ('R',)), B=C[f'B/{k}/logit'], F=F[k], y=Y.get(k))
    rows = list(csv.DictReader(open(S + 'arb/anyphoto_test.csv', encoding='utf-8'))); files = json.load(open(S + 'arb/local_files.json'))
    D['any'] = dict(run([files[r['id']] for r in rows]), rows=rows)
    np.savez(S + 'cmp/outputs.npz', D=np.array(D, dtype=object))
TB, EB = ck['calibration']['temperature'], ck['unknown_check']['threshold']
prev = torch.load(S + 'clipcls/model_with_head.pth', map_location='cpu', weights_only=True)['clip_head']
h = ck['clip_head']; hf = pickle.load(open(S + 'candF/head_F.pkl', 'rb'))
energy = lambda z: -logsumexp(z, 1)
def system(name, d):
    b, f = d.get('B'), d.get('F')
    if name == 'initial': return d['R'].argmax(1), np.zeros(len(d['R']), bool)
    if name == 'improved1': return b.argmax(1), energy(b) > EB
    if name == 'improved2':
        z = f @ prev['coef'].numpy().T + prev['intercept'].numpy()
        return (log_softmax(b / TB, 1) + log_softmax(z, 1)).argmax(1), (energy(b) > EB) | (energy(z) > prev['unknown_threshold'])
    W, I, EC = (h['coef'].numpy(), h['intercept'].numpy(), h['unknown_threshold']) if name == 'current' else (hf['coef'], hf['intercept'], hf['EC'])
    z = f @ W.T + I
    return (log_softmax(b / TB, 1) + log_softmax(z[:, :10], 1)).argmax(1), z[:, 10] - logsumexp(z[:, :10], 1) > EC
res = {}
for name in ('initial', 'improved1', 'improved2', 'current', 'F1'):
    r = {}
    for k in ('test', 'new', 'odev', 'o160'):
        p, rej = system(name, D[k]); y = np.array([NAMES.index(c) for c in D[k]['y']]); right = p == y
        r[k] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'n': len(y),
                'species': int((p // 2 == y // 2).sum()), 'condition': int((p % 2 == y % 2).sum()),
                'tomato_rotten': f'{int(right[y == 5].sum())}/{int((y == 5).sum())}'}
    for k in ('fruit374', 'coco400'):
        _, rej = system(name, D[k]); r[k] = {'rejected': int(rej.sum()), 'n': len(rej)}
    p, rej = system(name, D['any']); good = []
    for row, pp, j in zip(D['any']['rows'], p, rej):
        lab = row['label'].split('|') if row['label'] else []; c = row['category']
        good.append(bool(j) if c in REJECT else ((not j) and NAMES[pp] == lab[0]) if c == 'five_daily' else (bool(j) or NAMES[pp] == lab[0]) if c == 'cut_five' else (bool(j) or NAMES[pp] in lab))
    good = np.array(good); cats = [row['category'] for row in D['any']['rows']]
    r['any'] = {'good': int(good.sum()), 'n': len(good), 'by_category': {c: f'{int(good[[i for i, x in enumerate(cats) if x == c]].sum())}/{cats.count(c)}' for c in dict.fromkeys(cats)}}
    res[name] = r
assert (res['current']['test']['accuracy'], res['current']['test']['kept_correct']) == (288, 285)
assert (res['current']['new']['accuracy'], res['current']['new']['kept_correct']) == (203, 193)
assert (res['improved2']['test']['accuracy'], res['improved2']['test']['kept_correct']) == (284, 280), res['improved2']['test']
assert (res['improved1']['test']['accuracy']) == 275
json.dump(res, open(S + 'cmp/compare.json', 'w'), indent=1, ensure_ascii=False)
for name, r in res.items():
    print(name, {k: (v.get('accuracy'), v.get('kept_correct'), v.get('rejected')) for k, v in r.items() if k != 'any'}, 'any', r['any']['good'], r['any']['by_category'])
