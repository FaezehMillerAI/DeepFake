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
- 2026-09-16 | shards.py (WebDataset writer/reader, 384px resize, 5K pristine PNG preservation), freq.py (SRM 1st/2nd/3rd order high-pass filters + 2D DCT fp16 .npy shards), notebooks/00_data_prep.ipynb, DIFFERENTIATION.md (INP-X and GenShield novelty boundaries), scripts/prepare_dfbench_shards.py, scripts/validate_gate_g0.py | — | explanation extraction & baseline training
- 2026-09-16 | explain.py (Grad-CAM CNN+ViT, Score-CAM, Integrated Gradients, Attention Rollout for ViT-B/16), evaluate_drift.py (end-to-end clean + 25 corruptions + PGD-10/FGSM evaluation, ESI, ESI_rho, GAcc, IoU, ECE, Brier), 57/57 tests green | — | Kaggle execution of 00_data_prep.ipynb, training baselines and logging Table 1


<!-- Template for each new week:

## Week N — dates
**Goal:** one sentence.

- YYYY-MM-DD | ran | broke | next

**Friday review:** did the goal happen? Gate status? GPU hours used?
-->
