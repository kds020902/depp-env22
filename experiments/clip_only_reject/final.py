# Pre-registered (rej/PREREG.md): rejection by the CLIP head's not-five check only (R1) vs N3. New test set 241 and Test 300, once each.
import sys, csv, json, numpy as np, torch, open_clip
from scipy.special import log_softmax, logsumexp
sys.path.insert(0, '/home/user/depp-env22/src')
from model_utils import load_model, make_transform, open_image, input_size
S = '/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad/'; REPO = '/home/user/depp-env22/'
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
torch.set_num_threads(4)
B, ck = load_model(REPO + 'model.pth', torch.device('cpu')); tf = make_transform(input_size(ck))
TB, EB, h = ck['calibration']['temperature'], ck['unknown_check']['threshold'], ck['clip_head']
W, I, EC = h['coef'].numpy(), h['intercept'].numpy(), h['unknown_threshold']
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k'); clip.eval()
def feats(paths):
    bl, cf = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            bl.append(B(torch.stack([tf(im) for im in ims]))); cf.append(torch.nn.functional.normalize(clip.encode_image(torch.stack([prep(im) for im in ims])), dim=-1))
    return torch.cat(bl).numpy(), torch.cat(cf).numpy()
def score(bl, f, y):
    z = f @ W.T + I; p = (log_softmax(bl / TB, 1) + log_softmax(z[:, :10], 1)).argmax(1); right = p == y
    rc = z[:, 10] - logsumexp(z[:, :10], 1) > EC; rb = -logsumexp(bl, 1) > EB
    out = {}
    for name, rej in (('N3', rb | rc), ('R1', rc)):
        out[name] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~rej).sum()), 'rejected': int(rej.sum()), 'kept_wrong': int((~right & ~rej).sum()),
                     'rejected_by_class': {n: int((rej & (y == i)).sum()) for i, n in enumerate(NAMES) if (y == i).any()}}
    return out
t2 = list(csv.DictReader(open(REPO + 'results/10class/outside_test2/outside_test2.csv', encoding='utf-8')))
bl, f = feats([S + f"weak/t2img/{r['id']}.jpg" for r in t2]); y = np.array([NAMES.index(r['class_name']) for r in t2])
res = {'new_test_241': score(bl, f, y)}
assert res['new_test_241']['N3']['accuracy'] == 203 and res['new_test_241']['N3']['kept_correct'] == 169, res['new_test_241']['N3']
T = np.load(S + 'clipcls/test_features.npz')
yt = np.array([NAMES.index(r['class_name']) for r in csv.DictReader(open(REPO + 'dataset/manifest.csv', encoding='utf-8')) if r['split'] == 'test'])
res['test_300'] = score(T['b_logits'], T['clip'], yt)
assert res['test_300']['N3']['accuracy'] == 288 and res['test_300']['N3']['kept_correct'] == 282, res['test_300']['N3']
n, n0, t = res['new_test_241']['R1'], res['new_test_241']['N3'], res['test_300']['R1']
res['adopt'] = (n['kept_correct'] >= 172 and (n['kept_wrong'] - n0['kept_wrong']) < (n['kept_correct'] - n0['kept_correct']) and t['kept_correct'] >= 282)
json.dump(res, open(S + 'rej/final_once.json', 'w'), indent=1); print(json.dumps(res, indent=1))
