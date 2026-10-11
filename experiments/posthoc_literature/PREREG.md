# Post-hoc package from the literature, pre-registered before any candidate is scored (2026-10-11)

Ideas taken from the literature review (no new training of the vegetable model; all four members below were trained earlier
on the same 3rd-round Train with no Test rows):
 a. weight averaging ("model soup", Wortsman et al. 2022) of B, S7, L, R (EfficientNet-B0, same ImageNet start), with
    BatchNorm statistics recomputed on Train; fallback: logit ensemble of the four.
 b. a better unknown score (OpenOOD; KNN Sun et al. 2022; ViM Wang et al. 2022; Mahalanobis; MaxLogit; Energy).
 c. threshold calibrated on photos closer to real use (Bates et al. 2023): Valid plus 192 raw real-scene vegetable crops
    from LVIS (kitchens/markets; reviewed by eye, copies of check photos removed; never used by the vegetable model).
 d. hierarchical decision rule (Conditional Risk Minimization, Karthik et al. 2021): wrong species costs c, wrong condition 1.

Selection (Valid / dev halves only; checks are not looked at until everything is fixed):
 1. classifier = soup if its Valid accuracy >= 278 (B 281 - 3); else the 4-model ensemble if >= 278; else B.
 2. unknown score = the one with the highest AUROC separating {Valid + LVIS 192} from the dev halves of the 400 Kaggle fruit
    and 400 COCO photos (same protocol as calibration_unknown); KNN k=10 on L2-normalised Train features, ViM D=300.
 3. threshold = 95th percentile of the score on {Valid + LVIS 192}.
 4. CRM cost c in {1, 1.5, 2, 3} chosen by Valid accuracy with temperature fitted on Valid; ties -> smaller c (c=1 = argmax).

Adoption (final pipeline vs B with its current energy check; "kept correct" = not rejected and right class):
 - outside-dev 221 kept correct >= 143 (B 138 + 5)
 - 160 set kept correct >= 140 (B 142 - 2); Asian cucumbers 37 kept correct >= 27 (B 28 - 1)
 - internet fruit 374 rejected >= 233 (B); COCO 400 rejected >= 378 (B 386 - 8)
 - then Test once: Test accuracy >= 272 and Test kept correct >= 268 (B 275 / 268).
