# More rotten (and matching fresh) phone/field photos in Train, pre-registered before any run (2026-10-09T04:37Z)

Question: does adding real phone/field photos of fresh and rotten vegetables (same source for both labels) make the
model better on photos people take, without hurting the web-photo Valid/Test?

Train additions (Valid/Test rows untouched)
- cucumber: Kaggle sujaykapadnis "Cucumber Disease Recognition" (CC BY 4.0) fresh / belly rot + pythium fruit rot
- potato, tomato, carrot: Kaggle mdatikurrahman3111 "vegetables-quality-dataset-2" (CC0) good / bad quality
  (good carrots: phone photos only; the web/watermarked ones are left out so the source style matches)
- bell pepper: nothing (no source with real rot for both labels; VegNet's rotten peppers are red and its fresh green)
- per class +60 after visual review (3rd-round rules; cracks/cuts alone are not "rotten"; sprouts count for potato),
  one photo per copy group, drawn round-robin over look-alike sessions in random order (seed 0) until 60 pass;
  any copy of a dataset photo or an outside-test photo is dropped

Outside test set ("real photos"): Openverse (Flickr etc.) + Wikimedia Commons search results, labelled by eye before
any model sees them; copies of dataset photos removed; never used for training or for choosing settings.

Runs: same recipe as model.pth (efficientnet_b0 finetune, epochs 30, patience 8), manifest without Test rows,
seeds 42 (R42) and 7 (R7). Baseline: B (model.pth, seed 42) and S7 (seed 7, 3rd-round Train).

Rule
1. adopt only if BOTH
   a. outside set: correct(R42) + correct(R7) >= correct(B) + correct(S7) + 2 * ceil(0.03 * N)   (mean +3%p)
   b. Valid: Valid(R42) + Valid(R7) >= 281 + 274 - 6 = 549 of 600   (mean drop at most 1.0%p)
2. if adopted, score Test once with the run with the higher Valid (tie: R42); replace model.pth only if
   Test >= 272/300 (at most 1.0%p below the current 91.7%), since the change targets photos unlike the Test web photos
3. otherwise keep model.pth and dataset/, record as not adopted (the outside-set scores are reported either way)

Outside set frozen 2026-10-09T04:48Z: ext/outside_test.csv, N = 221 (sha256 ddc8ccb29572a793...), so the margin in 1a is 2 * ceil(0.03 * 221) = 14 correct.
