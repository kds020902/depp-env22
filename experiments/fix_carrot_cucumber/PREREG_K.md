# Candidate K, pre-registered before any run (2026-10-09T16:54Z)

Goal: (b) brown shriveled rotten cucumbers recognised as cucumbers; (a)/(a') soil must not decide fresh vs rotten for carrots and
potatoes. Lesson from H: fresh carrots with soil alone made dirty rotten potatoes look fresh, so soil now appears on both sides.

Train additions to the 3rd-round Train (Valid/Test untouched), all reviewed by eye, same source for both labels where possible:
 - cucumber: Mendeley "CUCUMBER" (zncrtr2yhn, CC BY 4.0), one phone camera, black background:
   good 79 (green, clearly fresh) / bad 76 (mostly brown, shriveled)
 - potato: Zenodo "Hybrid Potato Tuber Dataset" (20616991, CC BY 4.0), own photos only: good 59 / rotten or sprouted 60
   (manual cuts and cut-open tubers from the incorporated Potato Disease Recognition set excluded); plus Flickr/Wikimedia
   potatoes with soil, fresh 7 / diseased 2
 - carrot: fresh with soil 30 (Flickr/Wikimedia, same as last round) / rotten with soil 10 (vege-quality, previously dropped
   for being soil-covered) + 4 (Flickr/Wikimedia) + clearly rotten web carrots 19 (last round)
 - copies (hash-near or cosine >= 0.95) of dataset, outside-dev, 160-set or carrot-check photos are dropped

Checks: carrot check 27 (frozen earlier); the 10 brown rotten AgriFreshNET cucumbers in the 160 set (different source and
background from the Mendeley training photos; they motivated the change, never trained on); the 160 set (now a dev set);
Valid 300; outside-dev 221. Runs K42, K7 (same recipe as model.pth).

Rule (sums over the two seeds; B+S7 in brackets)
 1. brown cucumbers recognised as cucumber >= 6 of 20 [2]  AND  carrot check >= 44 [46]
 2. 160 set >= 294 [294]; Valid >= 549 [555]; outside-dev >= 329 [329]
 3. if 1 and 2 hold: score Test once with the higher-Valid seed; replace model.pth only if Test >= 272/300
