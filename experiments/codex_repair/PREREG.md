# Codex repair C: preserve old boundaries while learning the new sweet-potato photos

Registered before fitting or evaluating candidate C. CPU only.

## Authorization and scope
The user assigned model repair to Codex after A2/B were stopped. Claude owns green-onion condition work; this experiment changes no production code or model.pth.

## Hypothesis
Adding sweet-potato photos and freely refitting all 17 rows displaces potato decisions. Test a smaller update: freeze every second-head row except sweetpotato_fresh/rotten, and penalize deviations from TV-13 predictions on the ORIGINAL training features. This may preserve learned boundaries better. This is a hypothesis, not a demonstrated cause or improvement.

## Inputs and exclusions
- Teacher: current model.pth (TV-13).
- Replay: v7/head_data.npz train_X only (6271 rows). No dev features enter fitting.
- New learning: retrain_b/work/head_data_b.npz appended training rows (259). Verify its prefix equals the original training X/y exactly.
- All 969 labelled development rows and the existing unlabelled fruit/everyday/LVIS sets are used ONLY for candidate selection and gate evaluation.
- T3 through T10 are spent and never rescored. T11 stays untouched during this experiment. No held-out feature files or IDs are loaded.

## Fixed candidate family (maximum three fits)
Start from TV-13's second head. Fit additive residuals on ONLY rows 12 and 13 with CPU LBFGS, max 150 iterations. Freeze all other rows, zero-shot prompts, first head, potato_check, thresholds, switch probability and temperature.
Loss: mean supervised 17-way cross entropy on the 259 new train rows + lambda * mean KL(teacher || student) on original train rows + 0.001 * squared residual parameter norm. lambda in [1,4,16]. The teacher targets include the existing zero-shot contribution. No development labels in loss. Random seed 0, torch threads 4.

## Development gates (must pass all)
- Whole 969-row exact class/kind-for-green-onion correctness >= TV-13.
- Fresh potato and rotten potato exact correctness each >= TV-13.
- Rotten-to-fresh errors of ANY vegetable kind <= TV-13, and the number of confidently shown rotten-to-fresh errors <= TV-13.
- Rejected fruit + everyday objects must not decrease; retained LVIS vegetable crops must not decrease.
- At least one improvement on the 969 labelled rows; a candidate identical to TV-13 does not count as progress.
- Save every right-to-wrong and wrong-to-right transition per dataset and class, all input/checkpoint hashes, optimizer status and loss components.

Select only among passing candidates by total correct, then fewer rotten-to-fresh, then larger lambda. An acceptable result is a development candidate only, NOT proof of outside generalization. Do not open T11 or replace model.pth in this run; independent review comes first.

## Stop
If none passes, stop after these three fits and report failure. Do not adjust the lambda grid, relax gates, or start another family without reporting. Keep TV-13 active.
