# Does the adopted system (vegetable model + CLIP head N3) give the same answer when the photo is turned or flipped?
# Development photos only: Valid 300 and the first internet set 221. Analysis only; nothing is tuned on it.
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))  # the repository
import sys, json, torch
from PIL import Image
sys.path.insert(0, ROOT + "/src")
from model_utils import (load_model, make_transform, open_image, input_size, clip_head_logits, combined_logits, is_unknown_combined)
from clip_model import ClipModel
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
S = os.environ["DEPP_WORK"] + "/"
NAMES = ['cucumber_fresh','cucumber_rotten','potato_fresh','potato_rotten','tomato_fresh','tomato_rotten','bellpepper_fresh','bellpepper_rotten','carrot_fresh','carrot_rotten']
B, ck = load_model(ROOT + "/model.pth", torch.device('cpu')); tf = make_transform(input_size(ck)); clip = ClipModel('cpu')
FILL = (124, 116, 104)  # the same grey-brown the training rotation used for empty corners
VARIANTS = {
    'original': lambda im: im,
    'rotate 90': lambda im: im.transpose(Image.ROTATE_90),
    'rotate 180': lambda im: im.transpose(Image.ROTATE_180),
    'rotate 270': lambda im: im.transpose(Image.ROTATE_270),
    'tilt +15': lambda im: im.rotate(15, resample=Image.BILINEAR, expand=True, fillcolor=FILL),
    'tilt -15': lambda im: im.rotate(-15, resample=Image.BILINEAR, expand=True, fillcolor=FILL),
    'tilt +30': lambda im: im.rotate(30, resample=Image.BILINEAR, expand=True, fillcolor=FILL),
    'flip left-right': lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    'flip up-down': lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
}
sets = json.load(open(S + 'clip/sets.json'))
out = {}
for name in ('valid', 'odev'):
    paths, labels = zip(*sets[name]); y = torch.tensor([NAMES.index(l) for l in labels])
    images = [open_image(p) for p in paths]
    res, base = {}, None
    for v, fn in VARIANTS.items():
        preds, rejs = [], []
        for i in range(0, len(images), 32):
            ims = [fn(im) for im in images[i:i + 32]]
            with torch.inference_mode():
                fe = clip.image_features(ims); bl = B(torch.stack([tf(im) for im in ims])); cl = clip_head_logits(fe, ck)
                preds.append(combined_logits(bl, cl, ck).argmax(1)); rejs.append(is_unknown_combined(bl, cl, ck, fe))
        p, r = torch.cat(preds), torch.cat(rejs)
        if base is None: base = p
        right = p == y
        res[v] = {'accuracy': int(right.sum()), 'kept_correct': int((right & ~r).sum()), 'rejected': int(r.sum()), 'same_answer_as_original': int((p == base).sum())}
        print(name, v, res[v], flush=True)
    out[name] = {'n': len(images), **res}
json.dump(out, open(S + 'checks/rotation.json', 'w'), indent=1)
