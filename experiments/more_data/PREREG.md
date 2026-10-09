# More varied Train data for every class, pre-registered before any run (2026-10-09T07:59Z)

Model stays EfficientNet-B0 (same recipe as model.pth); only Train data changes. Valid/Test rows untouched.

Train additions (same source for both labels of a vegetable, visual review with the 3rd-round rules)
- cucumber: Cucumber Disease Recognition (CC BY 4.0) 60 fresh / 60 rotten (reviewed in experiments/rotten_phone)
- potato:   vegetables-quality-dataset-2 (CC0) 60 / 60 (reviewed in experiments/rotten_phone)
- tomato:   vegetables-quality-dataset-2 60 / 60 (reviewed) + razwansultan tomato fruit (MIT) healthy vs mold/anthracnose/
            blight/blossom-end rot, up to 50 / 50 + jimpax "Hybrid Tomato Disease" (CC BY 4.0) healthy vs blossom-end rot,
            up to 40 / 40 (field photos of tomatoes on the plant)
- carrot:   vegetables-quality-dataset-2 60 / 60 (reviewed) + vetion vege-quality (ODbL) up to 40 / 40
- pepper:   vetion vege-quality 73 fresh / 34 rotten + user2036 rotten capsicum (CC0) 39 (reviewed in experiments/pepper_web)
- every addition that copies (hash-near or cosine >= 0.95) a dataset photo, an outside-dev photo or an outside-test photo
  is dropped; look-alikes (cosine >= 0.90) of Valid/Test/outside photos are checked by eye

Evaluation sets
- Valid 300 (epoch selection, guard)
- outside-dev: the 221 Flickr/Wikimedia photos (results/10class/outside_test), used for the decision
- outside-test: 160 photos from sources never used for training (AgriFreshNET cucumber/tomato, VegNet pepper, Mendeley
  carrot, TriModal Ripeness carrot, VegQual potato, Potato Disease Recognition), frozen 2026-10-09T07:58Z
  (experiments/more_data/outside_test_160.csv; scratch file sha256 0c2f89518ba00317...), labels set by eye; no model has been run on it yet
- Test 300

Runs: G42 (seed 42), G7 (seed 7), manifest without Test rows. Baselines: B (model.pth) and S7.

Rule
1. outside-dev: correct(G42) + correct(G7) >= 167 + 162 + 14 = 343
2. Valid: Valid(G42) + Valid(G7) >= 281 + 274 - 6 = 549
3. only if 1 and 2 hold: score once, with the run that has the higher Valid (tie: G42), on outside-test and Test, and
   score B on outside-test at the same time. Replace model.pth only if
   outside-test(G) >= outside-test(B) + 5 (about 3%p of 160) AND Test(G) >= 272/300
4. otherwise keep model.pth and dataset/, record as not adopted
