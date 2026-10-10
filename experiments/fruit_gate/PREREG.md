# Fruit gate, pre-registered before any run (2026-10-10T15:35Z)

Request (from the user): option 2 - keep the vegetable model (model.pth, "B") exactly as it is and put a separate small
"fruit or vegetable?" model in front of it.

Pipeline: gate first. If the gate's fruit probability is above t, answer "학습된 데이터가 아닙니다". Otherwise B answers as
today (including its own energy check). B is not retrained, so vegetable accuracy (argmax) cannot change; what can change is
how many vegetable photos the gate wrongly turns away.

Gate model: MobileNetV3-Small (ImageNet weights, 2.5M parameters), fine-tuned for 2 classes, input 224, same augmentation as
train.py, AdamW (head 1e-3, backbone 1e-4), cosine, up to 20 epochs, early stop on Valid loss (patience 5), class-weighted
cross-entropy (vegetable photos outnumber fruit photos).

Gate photos
 - vegetable, Train: the 1,260 3rd-round Train photos; the 886 more_data and 325 fix_carrot_cucumber (K) photos (reviewed
   earlier, never used for model.pth); the five vegetables from kritikseth fruit-and-vegetable-image-recognition (CC0,
   web-style photos from the same dataset as part of the fruit photos, so "web photo" does not mean "fruit").
 - fruit, Train: the fruit pool of the fruit_other experiment minus its 30 Valid photos (~670; muhriddin, kritikseth,
   Fruits-262; apricot, cherry, nectarine, plumcot, damson never trained).
 - Valid: the 300 Valid vegetable photos + the same 30 Valid fruit photos as fruit_other.
 - copies (hash-near or cosine >= 0.95) of Valid/Test vegetables, outside-dev, 160 set, Asian cucumbers, carrot check, the
   374 internet fruit photos, the 400 calibration_unknown fruit photos or the demo fruit samples are dropped.

Threshold t (Valid only): t = max(0.5, the 99th percentile of the gate's fruit probability on the 300 Valid vegetable photos),
so at most 1% of Valid vegetables are turned away.

Checks: the same frozen sets as fruit_other (internet fruit 374, sha256 8e78a231...4eeb; 400 Kaggle fruit; 400 COCO; demo
16-20; Valid 300; outside-dev 221; 160 set; Asian cucumbers 37). "kept correct" = not rejected and the right class.
Baseline B alone: internet fruit rejected 233; COCO rejected 386; Valid kept correct 271; outside-dev kept correct 138;
160 set kept correct 142.

Rule (gate seeds 42 and 7, each in front of B; sums, B's values doubled in brackets)
 1. internet fruit rejected >= 598 of 748 (80%) [466]
 2. Valid kept correct >= 536 [542]; outside-dev kept correct >= 270 [276]; 160 set kept correct >= 278 [284];
    COCO rejected >= 762 [772]
 3. if 1 and 2 hold: with the gate seed that has the lower Valid loss, score Test once; adopt only if Test kept correct
    >= 265 (B alone: 268). Test accuracy itself cannot change (275).

Note added after the data build (before any training run): Train = 2,783 vegetable (1,260 + 886 + 325 + 312 kritikseth; 174 kritikseth photos dropped as copies of Valid/Test or check photos) and 671 fruit; Valid = 300 vegetable + 30 fruit.
