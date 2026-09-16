# Session log

One line per working session. This is the memory of the project — in month four
you will need to know what you tried in week six.

Format: `YYYY-MM-DD | what ran | what broke | what's next`

---

## Week 1 — 7–13 Sep 2026
**Goal:** Build the ruler before you measure anything.

- 2026-09-16 | repo scaffolded, metrics frozen, CF implemented, 40 tests green | — | pseudo-mask pipeline & datasets

---

## Week 2 — 14–20 Sep 2026
**Goal:** Build and validate the real pseudo-mask pipeline and data infrastructure.

- 2026-09-16 | pseudo_masks.py (LAB diff+Otsu+morphology), acquire.py (DFBench remote zip confirmed: 18,871 edits match 11,922 partial_sources by stem), datasets.py (PairedEditDataset, MaskSupervisedDataset), checkpoint.py (Rule 1 atomic resume), 40/40 tests pass | — | validate on real SID-Set, implement baseline architectures in src/models/


<!-- Template for each new week:

## Week N — dates
**Goal:** one sentence.

- YYYY-MM-DD | ran | broke | next

**Friday review:** did the goal happen? Gate status? GPU hours used?
-->
