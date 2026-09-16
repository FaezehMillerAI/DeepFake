# Notebooks

Kaggle notebooks live here. Keep them **thin**: a notebook imports from `src/`,
sets a config, and runs. It does not contain logic.

Why: notebooks are not diffable, not testable, and not reviewable. Anything that
matters belongs in `src/` with a test beside it.

Suggested notebooks:

| Notebook | Purpose | Week |
|---|---|---|
| `00_data_shards.ipynb` | Build and upload the Kaggle dataset shards | 2 |
| `01_pseudo_masks.ipynb` | Pseudo-mask pipeline + SID-Set validation (Gate G0) | 2 |
| `02_train_baseline.ipynb` | Train one baseline, resume-safe | 3 |
| `03_drift_sweep.ipynb` | E2 corruption sweep | 5 |
