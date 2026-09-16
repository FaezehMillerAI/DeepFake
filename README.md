# GRACE-DF

**Grounded, Robust And Calibrated Evidence for Deepfake Detection**

Research code for a study of *evidence drift* in deepfake detection — the phenomenon
where a detector keeps its label under degradation or attack while its explanation
stops pointing at the manipulated region — and a grounded, frequency-aware model
that resists it.

See `docs/` for the full research plan and the 24-week schedule.

---

## Environment

All experiments run on **Kaggle**. Fill these in on day one and keep them current —
every scheduling decision depends on them.

| Constraint | Value | Checked on |
|---|---|---|
| GPU types | P100 16GB / 2×T4 16GB | |
| Weekly GPU quota | ? h | |
| Max session length | ? h | |
| `/kaggle/working` cap | ? GB | |
| Max dataset size | ? GB | |
| vCPUs / RAM | ? | |

> **No bf16 on P100 or T4.** Use fp16 + `GradScaler`, and watch for NaNs in the
> contrastive loss (clamp logits, `eps=1e-8`).

---

## Layout

```
configs/        YAML experiment configs. One file per experiment, never edit in place
                once a run has used it — copy and version instead.
src/
  config.py     Dataclass + YAML loader. Single source of truth for hyperparameters.
  data/         Dataset construction, sharding, pseudo-mask pipeline.
  models/       Backbones, the frequency encoder, GRACE-DF itself.
  metrics/      GAcc, ESI, IoU, ECE, Brier. Frozen in Week 1 — do not change
                definitions after experiments start.
  attacks/      Corruption bank and adversarial attacks.
  train/        Training loops and the resume-from-checkpoint harness.
  eval/         Evaluation drivers and statistical tests.
  utils/        Seeding and JSON run logging.
tests/          Unit tests for the metrics and the corruption bank.
notebooks/      Kaggle notebooks. Thin — they import from src/, they don't contain logic.
runs/           One JSON per run: git SHA, config, seed, all metrics.
                Every table in the paper is generated from these files.
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                             # must pass before any experiment
```

---

## The two rules

1. **Every training script resumes from `runs/<name>/ckpt_last.pt`.** Assume every
   Kaggle session dies at 11h58m.
2. **No number is ever typed by hand.** Results go to `runs/*.json`; tables and
   figures are generated from those files by script.

---

## Running

```bash
python -m src.train.baseline --config configs/baselines/effnet_b0.yaml
python -m src.eval.run       --config configs/baselines/effnet_b0.yaml --corruptions all
```
