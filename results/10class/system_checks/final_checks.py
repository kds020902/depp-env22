# Final check (development photos only): fit gap, calibration and turned/flipped photos for the current system (N3 + R1)
# and candidate F1. Analysis only; nothing is tuned on it.
import sys, json, pickle, numpy as np, torch, open_clip
from PIL import Image
from scipy.special import log_softmax, logsumexp, softmax
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size
torch.set_num_threads(4)
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
B, ck = load_model('/home/user/depp-env22/model.pth', torch.device('cpu')); tf = make_transform(input_size(ck))
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
TB, h = ck['calibration']['temperature'], ck['clip_head']; hf = pickle.load(open(S + 'candF/head_F.pkl', 'rb'))
SYS = {'current': (h['coef'].numpy(), h['intercept'].numpy(), h['unknown_threshold'], h['temperature']), 'F1': (hf['coef'], hf['intercept'], hf['EC'], hf['T1'])}
def decide(name, bl, f):
    W, I, EC, T = SYS[name]; z = f @ W.T + I; comb = log_softmax(bl / TB, 1) + log_softmax(z[:, :10], 1)
    return comb.argmax(1), z[:, 10] - logsumexp(z[:, :10], 1) > EC, softmax(comb / T, 1)
def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)
sets = json.load(open(S + 'clip/sets.json')); F = np.load(S + 'clip/feat_datacomp_xl_s13b_b90k.npz'); C = np.load(S + 'post/cache.npz'); Y = json.load(open(S + 'post/labels.json'))
out = {'fit': {}, 'rotation': {}}
# 1. fit gap and calibration: training photos (1,260) vs Valid vs photos from other sources
import csv
rows = list(csv.DictReader(open('/home/user/depp-env22/dataset/manifest.csv', encoding='utf-8'))); lab = {'/home/user/depp-env22/dataset/' + r['image_path']: r['class_name'] for r in rows if r['split'] == 'train'}
ti = [i for i, (p, _) in enumerate(sets['probe_train']) if p in lab]; ytr = [lab[sets['probe_train'][i][0]] for i in ti]
data = {'train': (C['B/train/logit'], F['probe_train'][ti], ytr)}
assert len(C['B/train/logit']) == len(ti) and Y['train'] == ytr
for k in ('valid', 'o160', 'odev'): data[k] = (C[f'B/{k}/logit'], F[k], Y[k])
for name in SYS:
    out['fit'][name] = {}
    for k, (bl, f, yy) in data.items():
        y = np.array([NAMES.index(c) for c in yy]); p, rej, prob = decide(name, bl, f)
        nll = float(-np.log(prob[np.arange(len(y)), y] + 1e-12).mean())
        out['fit'][name][k] = {'n': len(y), 'accuracy': int((p == y).sum()), 'kept_correct': int(((p == y) & ~rej).sum()), 'loss': round(nll, 3), 'ece': round(ece(prob, y), 3)}
    print(name, out['fit'][name], flush=True)
# 2. turned / flipped photos
FILL = (124, 116, 104)
VARIANTS = {'original': lambda im: im, 'rotate 90': lambda im: im.transpose(Image.ROTATE_90), 'rotate 180': lambda im: im.transpose(Image.ROTATE_180),
            'rotate 270': lambda im: im.transpose(Image.ROTATE_270), 'tilt +15': lambda im: im.rotate(15, resample=Image.BILINEAR, expand=True, fillcolor=FILL),
            'tilt -15': lambda im: im.rotate(-15, resample=Image.BILINEAR, expand=True, fillcolor=FILL),
            'tilt +30': lambda im: im.rotate(30, resample=Image.BILINEAR, expand=True, fillcolor=FILL),
            'flip left-right': lambda im: im.transpose(Image.FLIP_LEFT_RIGHT), 'flip up-down': lambda im: im.transpose(Image.FLIP_TOP_BOTTOM)}
for k in ('valid', 'odev'):
    paths, labels = zip(*sets[k]); y = np.array([NAMES.index(l) for l in labels]); images = [open_image(p) for p in paths]
    res = {name: {} for name in SYS}; base = {}
    for v, fn in VARIANTS.items():
        bl, cf = [], []
        with torch.inference_mode():
            for i in range(0, len(images), 32):
                ims = [fn(im) for im in images[i:i + 32]]
                bl.append(B(torch.stack([tf(im) for im in ims]))); cf.append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
        bl, cf = torch.cat(bl).numpy(), torch.cat(cf).numpy()
        for name in SYS:
            p, rej, _ = decide(name, bl, cf)
            if v == 'original': base[name] = p
            res[name][v] = {'accuracy': int((p == y).sum()), 'kept_correct': int(((p == y) & ~rej).sum()), 'rejected': int(rej.sum()), 'same_answer_as_original': int((p == base[name]).sum())}
        print(k, v, {n: res[n][v] for n in SYS}, flush=True)
    out['rotation'][k] = {'n': len(images), **res}
json.dump(out, open(S + 'final/checks.json', 'w'), indent=1)
