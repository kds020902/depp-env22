# Candidate J (Asian cucumbers), pre-registered before any run (2026-10-10T05:20Z)

Question (from the user): Korean, Japanese and Chinese cucumbers look different from the Western-style cucumbers that make
up most of the training photos. Does training on East Asian cucumbers help?

Asian check, frozen before any training (`asian_check_frozen.csv`, sha256 932a0007...ad92):
 - 37 fresh East Asian cucumbers people posted online (Flickr / Wikimedia Commons; Korea, Japan, China, Taiwan, Hong Kong,
   Thailand), reviewed by eye; copies of dataset / outside-dev / 160-set photos dropped (2 were copies of dataset photos).
 - No rotten East Asian cucumbers could be found under an open licence, so the check has fresh photos only.
 - Never used for training. Baseline: B 30/37, S7 32/37 (B misses 3 spiny Korean/Japanese cucumbers as rotten).

Train additions to the 3rd-round Train (Valid/Test untouched), all reviewed by eye:
 - cucumber_fresh +60: Zenodo "Cucumber Dataset" (10.5281/zenodo.10081197, Jin Tao, CC BY 4.0), cucumber germplasm photographed
   on black cloth with a ruler. Cropped to the fruit (ruler and colour cards removed); one photo per accession; green fruits
   only (no yellow/white mature seed fruits), favouring long, slender, spiny East Asian types.
 - cucumber_rotten +60: Mendeley "CUCUMBER" (zncrtr2yhn, CC BY 4.0) bad cucumbers, also on black background, randomly drawn
   (seed 0) from the 76 that passed review last round. Purpose: black background on both sides, so it cannot mean "fresh".
 - copies (hash-near or cosine >= 0.95) of dataset, outside-dev, 160-set, carrot-check or Asian-check photos are dropped.

Runs J42, J7 (same recipe as model.pth: efficientnet_b0 finetune, --epochs 30 --patience 8).

Rule (sums over the two seeds; B+S7 in brackets)
 1. Asian check >= 68 of 74 [62]
 2. Valid >= 549 [555]; outside-dev >= 329 [329]; 160 set >= 294 [294]
 3. if 1 and 2 hold: score Test once with the higher-Valid seed; replace model.pth only if Test >= 272/300
Also reported, not part of the rule: carrot check, the 10 brown rotten cucumbers in the 160 set.

Note added after the runs: in this folder the frozen check list is saved as `asian_check.csv` (same sha256).
The country list above is wrong: the 37 photos are from Korea, Japan and China, plus a few Asian varieties grown in the US.
