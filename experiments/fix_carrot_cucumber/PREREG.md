# Fixing two error types, pre-registered before any run (2026-10-09T12:34Z)

Errors found when the 160-photo outside-test set was scored once (experiments/more_data):
 (a) G42 (more_data model) called 11/20 fresh carrots with soil/rough skin "carrot_rotten"; nearest Train neighbours were
     the added vq2 "bad" carrots, many of which are dry/rough rather than rotten
 (b) both B and G42 call brown, shriveled rotten cucumbers "potato_rotten" (B 8/10 wrong, G42 9/10 wrong);
     no Train photo shows a cucumber in that state

Changes (Train only; Valid/Test untouched)
 - targeted additions for both candidates: brown/shriveled rotten cucumbers (all usable ones found: 6), fresh carrots with
   soil from Flickr/Wikimedia (30, the other 14 held out by uploader), clearly rotten carrot web photos (29)
 - H : 3rd-round Train (1,260) + targeted additions
 - Gp: more_data Train (2,146) minus the 57 added rotten carrots that are only dry/rough (re-reviewed, 43 kept) + targeted

Evaluation
 - carrot check (27 fresh carrots, frozen now): 13 Mendeley fresh carrots not in the 160 set + 14 held-out muddy carrots
 - the 160 outside-test set is now a development set (its errors motivated the change); reported, not used as an unbiased score
 - cucumber: the 10 brown rotten AgriFreshNET cucumbers in the 160 set (the photos that showed the problem; optimistic),
   counted as species correct
 - Valid 300, outside-dev 221 (Flickr/Wikimedia)
 Runs: H42, H7, Gp42, Gp7 (same recipe as model.pth). Baselines B, S7, G42, G7.

Rule
 1. a candidate passes if, summed over its two seeds:
    carrot check >= B42 + S7 - 2 (no regression vs the current model's seeds)
    160 set >= B + S7 on the same set
    Valid >= 549, outside-dev >= 329 (no regression)
 2. among passing candidates take the one with the higher 160-set sum (tie: H); score Test once with its higher-Valid seed;
    replace model.pth only if Test >= 272/300 and its cucumber species errors on the 10 brown cucumbers < B's
 3. otherwise keep model.pth; record everything

Note added after the build (not part of the rule): the final copy check dropped 11 of the 65 targeted photos as copies of dataset photos (8 Train, 2 Test, 1 Valid), leaving 5 cucumbers, 30 fresh carrots with soil and 19 rotten carrots.
