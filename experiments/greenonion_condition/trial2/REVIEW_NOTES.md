# Trial 2 - review corrections (codex recompute, 2026-11-28)

Codex recomputed from the five saved fold candidates (no retraining, final test not read)
and reproduced spoiled 37/63 and fresh 49/59. Corrections to the trial-2 report:

1. G1 passed as preregistered (969 dev: 869 -> 871), but this is NOT "no regression on the
   other seven species". The +2 includes green onion gains. Excluding green onion,
   the seven species went 855 -> 854; rotten sweet potato 18 -> 17. Potatoes unchanged.
2. G2/G3 counts exclude only `unknown`. Photos withheld as low-confidence are still counted,
   so these are not the rates a user would see as confirmed results on the web.
3. Integrity: the 130 photos match the list hashes, and each fold's training set excludes
   that fold's evaluation features. The feature cache stores no source hash or extractor
   record, so the link to the photos at extraction time cannot be checked independently.
4. trial2/run.py did not call `build_user_lists.py --check` (claude's omission). A post-hoc
   check passed for the user photos only.

Status unchanged: trial 2 FAILS G3 (spoiled 37/63 = 58.7% < 60%).
Before any trial 3: run.py must check user photos, collected photos and the feature cache
(source hashes + extractor id) and stop on any mismatch. Whether more photos help is a hypothesis.
