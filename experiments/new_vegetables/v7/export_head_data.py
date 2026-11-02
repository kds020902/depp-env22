# Save everything the new-vegetable head needs, so it can be rebuilt without the photos (experiments/new_vegetables/v7/head_data.npz):
# - train: CLIP image features (ViT-B-16, DataComp-XL, L2-normalised) and labels (0-15 = model_utils.CLASS_NAMES, 16 = not one of the
#   eight vegetables), in the order the adopted head was fitted, with a short source tag per row;
# - dev: features (+ vegetable-model logits) of the development sets used for the "not eight" threshold and the tau rule;
# - text: the 16 class text embeddings (CLIP zero-shot prompts) and the prompts themselves.
import os
import sys, json, numpy as np
XS = open((os.environ.get("DEPP_WORK", "/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad") + "/v7/explore.py")).read()
sys.argv = ['x', (os.environ.get("DEPP_WORK", "/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad") + "/v7/pool.json")]
exec(XS[:XS.index("VARIANTS = {")])
X = np.concatenate([X0, pf[tr]]).astype(np.float32); y = np.concatenate([y0, ytr]).astype(np.int16)
tags = np.array(['train1260'] * len(Xt) + ['reviewed_extra'] * len(Xp) + ['weak_spots'] * len(Xw) + ['asian_cucumber'] * len(Xa) + ['not_eight'] * len(Xn16) + ['new_vegetables'] * len(tr))
assert len(tags) == len(X)
out = {'train_X': X, 'train_y': y, 'train_tag': tags, 'text16': T.astype(np.float32), 'prompts': np.array(['|'.join(p) for p in prompts])}
for k, (b, f, yy) in D.items():
    out[f'dev_{k}_X'] = f.astype(np.float32); out[f'dev_{k}_b'] = b.astype(np.float32)
    if yy is not None: out[f'dev_{k}_y'] = yy.astype(np.int16)
np.savez_compressed(V + 'head_data.npz', **out)
print({k: v.shape for k, v in out.items()})
