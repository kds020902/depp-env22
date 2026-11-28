# Constrained head repair C — stopped, no adoption

The user assigned existing-model repair to Codex. Plan commit: 512d701, before training. CPU only, three predetermined candidates, no T10/T11 scoring. The existing TV-13 model.pth was not changed.

Freeze all second-head classifier rows except sweet-potato fresh/rotten, keep all temperatures/thresholds fixed, and penalize divergence from TV-13 on original training features while fitting the 259 new learning rows. New data was taken only from candidate B's verified training cache; its original training prefix was checked byte-for-value against v7/head_data.npz. No development rows enter the fitting loss.

| System | Development correct / 969 | Fresh potato / 126 | Rotten potato / 133 | Rotten called fresh |
|---|---:|---:|---:|---:|
| TV-13 | 869 | 107 | 126 | 17 |
| C1 (preservation weight 1) | 846 | 99 | 108 | 21 |
| C4 (weight 4) | 857 | 103 | 115 | 20 |
| C16 (weight 16) | 868 | 106 | 124 | 19 |

All three candidates fail the pre-registered gates. Increasing the preservation weight reduced losses, but did not produce a qualifying improvement. Stop here; do not search more weights or weaken gates. Existing-model replay distillation did not solve the outside-photo trade-off in this fixed candidate family. This does not establish that the encoder or all possible CPU approaches cannot improve.

Verification: only the intended classifier rows changed; all other classifier rows and switch/unknown/display thresholds stayed equal; all optimizers stopped before the iteration cap; the baseline file SHA-256 stayed unchanged. Full metrics, per-photo changes, input hashes and losses are in result.json. Candidate checkpoints stay in ignored work/.

Rerun: python experiments/codex_repair/run.py. The script refuses to overwrite an existing result.json. No rerun is needed for the recorded results.
