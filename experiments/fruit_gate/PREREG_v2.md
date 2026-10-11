# Fruit gate v2, pre-registered before any run (2026-10-11)

Second try of the fruit gate (experiments/fruit_gate). Gate v1 kept the vegetables (160 set unchanged) but rejected only
77% of the internet fruit; persimmon passed in more than half of the photos. Likely cause: few fruit photos (671, about a
dozen per fruit). Only the training photos change; model, recipe, threshold rule and checks stay exactly as in v1.

Added to the gate's Train (Valid unchanged: 300 vegetables + 30 fruit):
 - fruit: up to 60 more Fruits-262 photos per fruit for the 34 trainable fruits (persimmon, mandarine, clementine, yali pear =
   Asian pear, jujube, yuzu, kumquat, apple, pear, grape, ...). Apricot, cherry, nectarine, plumcot, damson stay untrained.
 - vegetable: 120 Fruits-262 tomato and 120 Fruits-262 bell pepper photos (same web source as the fruit, against
   "web photo = fruit"; tomatoes were the most common vegetable the v1 gate called fruit), and the 140 East Asian cucumber
   crops of experiments/asian_cucumber J2 (the v1 gate called up to 5 Asian cucumbers fruit).
 - Fruits-262 photos are checked by contact sheets of samples, not one by one.
 - copies (hash-near or cosine >= 0.95) of every check photo are dropped, as in v1.

Rule: identical to v1 (gate seeds 42 and 7, each in front of B; sums):
 1. internet fruit rejected >= 598 of 748 (80%)
 2. Valid kept correct >= 536; outside-dev kept correct >= 270; 160 set kept correct >= 278; COCO rejected >= 762
 3. if 1 and 2 hold: with the gate seed that has the lower Valid loss, score Test once; adopt only if Test kept correct >= 265.
This is the second attempt against the same checks, so a pass is slightly more likely by chance; reported as such.

Note added after the downloads (before the data build and any run): yali pear (Asian pear) and bell pepper sit in folders with a space in the name and were fetched in a second pass (72 yali pear, 120 bell pepper). Contact sheet of 6 photos per kind checked.
