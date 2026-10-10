# Candidate J2 (Western : East Asian cucumbers 5:5), pre-registered before any run (2026-10-10T08:12Z)

Request (from the user): make the Western and Asian cucumbers in the training photos 5:5.

Current Train cucumber_fresh (140), sorted by eye: Western style (thick, smooth, dark green; stock and slice photos) 72,
South Asian market cucumbers (thick, light green; Bangladesh) 66, on the vine / unclear 2, East Asian style (long, slender,
spiny) 0. The South Asian ones look like the Western ones, not like Korean/Japanese/Chinese cucumbers, so the 5:5 split is
"existing 140 : East Asian 140".

Train additions to the 3rd-round Train (Valid/Test untouched):
 - cucumber_fresh +140 East Asian style: the 60 Zenodo 10081197 crops of candidate J plus 80 more from the same photos
   (40 more accessions + 22 DSC + 18 IMG photos), shortlisted by shape (long, green, not pale) and then checked by eye.
 - cucumber_rotten: every distinct Mendeley "CUCUMBER" bad photo (black background, same 76-photo pool as J; after removing
   near-copies about 52-60 remain). The 500 bad photos hold only about 68 different shots, so the black background cannot be
   matched 140:140 on the rotten side. Known risk: "black background = fresh"; watched through the rotten cucumbers below.
 - copies (hash-near or cosine >= 0.95) of dataset, outside-dev, 160-set, carrot-check or Asian-check photos are dropped;
   near-copies inside the additions (cosine >= 0.97) keep only the first.

Runs J2_42, J2_7 (same recipe as model.pth: efficientnet_b0 finetune, --epochs 30 --patience 8).

Rule (sums over the two seeds; B+S7 in brackets) - same as candidate J
 1. Asian check (37 photos, frozen 2026-10-10) >= 68 of 74 [62]
 2. Valid >= 549 [555]; outside-dev >= 329 [329]; 160 set >= 294 [294]
 3. if 1 and 2 hold: score Test once with the higher-Valid seed; replace model.pth only if Test >= 272/300
Also reported, not part of the rule: rotten cucumbers in Valid (30) and in the 160 set (10), carrot check.
