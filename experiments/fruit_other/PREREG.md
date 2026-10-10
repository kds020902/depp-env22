# Candidate O ("other" class for fruit), pre-registered before any run (2026-10-10T12:58Z)

Request (from the user): tell fruit from the five vegetables reliably. Today a fruit photo is only rejected when the energy
score is high, and fruit that looks like a vegetable (orange -> tomato, rotten apple -> rotten tomato) gets through.

Change: an 11th class "other" trained on fruit photos. A photo is answered "학습된 데이터가 아닙니다" when the top class is
"other" OR the energy score is above the threshold (threshold now fitted on the Valid vegetable photos only, still 95%).
Model, recipe and the vegetable photos stay the same (EfficientNet-B0, --epochs 30 --patience 8; Test vegetables untouched).

"other" photos (all CC0, reviewed by eye, copies of any evaluation or vegetable photo removed):
 - Train ~293: muhriddin Fruits (same Kaggle archive as the vegetables, groups touching the 400 calibration_unknown fruit
   photos or the demo fruit samples removed) 10 per folder x 10 folders (fresh/rotten apple, banana, mango, orange,
   strawberry); kritikseth fruit-and-vegetable-image-recognition, 8 per fruit x 11 fruits; Fruits-262, 3 per fruit x 35 fruits.
 - Valid 30: 10 muhriddin, 11 kritikseth, 9 Fruits-262 (separate photos).
 - Fruit kinds never trained: apricot, cherry, nectarine, plumcot, damson (and peach, plum, Korean melon are not in any source).

Checks (frozen before training):
 - Internet fruit check: 374 Flickr/Wikimedia photos of 19 fruits (276 fresh, 98 rotten), incl. persimmon, tangerine, Korean
   melon, Asian pear, jujube (`fruit_check.csv`, sha256 8e78a231...4eeb). 62 are kinds never trained (apricot, cherry, peach,
   plum, Korean melon).
 - 400 muhriddin fruit photos of calibration_unknown (same source as part of the training fruit, so easier); 400 COCO photos.
 - Vegetables: Valid 300, outside-dev 221, 160 set, Asian cucumbers 37.
"kept correct" = not rejected and the right vegetable class.

Baselines B / S7 (S7 given the same Valid-fitted temperature and threshold with src/calibrate.py), sums in brackets:
internet fruit rejected 233 / 226 [459 of 748]; COCO rejected 386 / 380 [766]; Valid kept correct 271 / 264 [535];
outside-dev kept correct 138 / 140 [278]; 160 set kept correct 142 / 145 [287]; Valid accuracy 281 / 274 [555].

Rule (sums over seeds 42 and 7)
 1. internet fruit rejected >= 598 of 748 (80%)
 2. Valid kept correct >= 529; outside-dev kept correct >= 272; 160 set kept correct >= 281; COCO rejected >= 756;
    Valid accuracy >= 549
 3. if 1 and 2 hold: score Test once with the higher-Valid seed; replace model.pth only if Test accuracy >= 272/300 and
    Test kept correct >= 265 (B: 275 and 268)
Also reported: the 400 muhriddin fruit photos, fruit kinds never trained, per fruit, Asian cucumbers, demo samples 16-20.

Note added after the build (before any training run): the build gave Train 287 (Fruits-262 has 33 usable fruit kinds after the held-out ones, so 99 not 105) and Valid 30.

Note added after the runs: the frozen check list is saved here as `fruit_check.csv` (same sha256). A 1-epoch smoke run (seed 1) was scored once before the real runs only to test the pipeline; it was discarded.
