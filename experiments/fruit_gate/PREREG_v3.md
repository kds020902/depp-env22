# Fruit gate v3, pre-registered before any run (2026-10-11)

Third try of the fruit gate. v2 passed the fruit target (85%) but outside-dev kept correct was 267 < 270: the gate turned
away 10-15 online vegetable photos, mostly fresh tomatoes and potatoes on market tables, gardens and kitchens. The gate's
vegetable photos are mostly single vegetables on plain backgrounds, so real scenes are missing on the vegetable side.

Only the training photos change; model, recipe, threshold rule, Valid, checks and rule stay exactly as in v1/v2.

Added to the gate's Train (on top of v2):
 - crops from the LVIS fruits-and-vegetables subset (henningheyen, MIT; COCO photos of kitchens, markets, tables):
   vegetable side = boxes of tomato, potato, cucumber, bell pepper, carrot; fruit side = boxes of the trained fruit kinds
   (apple, orange, mandarin, clementine, persimmon, pear, banana, lemon, lime, grape, kiwi, strawberry, melon, watermelon,
   pineapple, papaya, fig, ...). Peach, apricot and cherry boxes are not used (they stay untrained kinds).
   Both sides get real scenes, so "real scene" cannot mean "vegetable". Boxes at least 100 px on the short side, 10% margin,
   at most 2 boxes per photo, up to 700 per side, balanced over kinds.
 - A whole LVIS photo is dropped if it copies (hash-near or cosine >= 0.95) any check photo (the LVIS photos come from
   Flickr like the internet checks). COCO check photos have no food, so they cannot be LVIS photos.
 - checked by contact sheets of samples, not one by one.

Rule: identical to v1/v2 (gate seeds 42 and 7, each in front of B; sums):
 1. internet fruit rejected >= 598 of 748 (80%)
 2. Valid kept correct >= 536; outside-dev kept correct >= 270; 160 set kept correct >= 278; COCO rejected >= 762
 3. if 1 and 2 hold: with the gate seed that has the lower Valid loss, score Test once; adopt only if Test kept correct >= 265.
Third attempt against the same checks: a pass is somewhat more likely by chance than in v1; reported as such.

Note added after reviewing the crops (before any training run): most LVIS vegetable boxes are cooked or on dishes (pizza, salad, roast). All 700 vegetable crops were checked on contact sheets and only raw vegetables were kept: 192 (carrot 106, tomato 40, bell pepper 16, cucumber 15, potato 15). Persimmons and oranges labelled "tomato" in LVIS were left out. The fruit side gets the same number, 192 random fruit crops after dropping 17 that showed people, food or a vegetable.
