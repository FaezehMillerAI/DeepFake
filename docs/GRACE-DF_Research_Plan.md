# GRACE‑DF — Step‑by‑Step Research Plan

**Grounded, Robust And Calibrated Evidence for Deepfake Detection**

Built from: `Rigorous Critical Analysis of the Manuscript Limitations.docx` (Ideas 2 + 3, fused)
Baselines: DFBench/MoA‑DF · DeepFakeBuster · Shakya et al. (adversarial) · ARC‑Net · CT‑DFID
Constraints assumed: **Kaggle only** (P100 16 GB *or* 2×T4 16 GB, ~30 GPU‑h/week, 12 h session cap)
Target: **one top‑tier paper in 3–6 months**

---

## 0. The one‑sentence thesis

> Every deepfake detector in the literature is evaluated on *whether it says fake*; none is evaluated on *whether it still points at the right pixels* once the image is compressed, blurred or attacked — and we show that the label survives while the evidence does not.

**Headline claim to aim for:**
A **0.8 B‑parameter** grounded model matches a **24 B** mixture‑of‑agents (MoA‑DF) on DFBench classification accuracy, *additionally* produces manipulation masks, and retains ≥ X % of its localisation fidelity under degradation and PGD attack — a property no existing detector has ever been measured on, because no existing detector can localise at all.

That claim is (a) novel, (b) falsifiable, (c) achievable on Kaggle, (d) simultaneously closes four of the six limitations in your critique doc.

---

## 1. What the five baselines actually establish

Read this table as your evidence base — every number here is a sentence in your Related Work, and every "Limitation" column is a hole you are allowed to claim.

| # | Paper | What it proves | The exact hole |
|---|-------|----------------|----------------|
| 1 | **DFBench / MoA‑DF** (ACM MM '25) | 540 K images, 12 generators, bidirectional protocol. MoA‑DF (Qwen2.5‑7B + InternVL2.5‑8B + InternVL3‑9B ≈ **24 B**) hits 98.41 % real / 97.05 % AI‑edit / 99.92 % AI‑gen after fine‑tuning | Static softmax sum over a **fixed** pool. **Zero‑shot LMM average on "Object Enhance" = 15.89 % Acc / 0.211 F1** — catastrophic on fine‑grained local edits. **No localisation whatsoever** despite the dataset containing partially edited images. Accuracy drops on CSIQ/TID2013/KADID distortions |
| 2 | **DeepFakeBuster** (Sci. Rep. 2026) | 11‑detector heterogeneous ensemble, confidence‑calibrated adaptive fusion, 97.8 % on 192 K images; beats majority voting (96.7), static weighted (97.1), stacking (97.4) | Self‑declared: accuracy falls to **~67 % at 32×32**, degrades under repeated JPEG (`A_{q,n}=A_0 e^{-λn(100-q)/100}`), **adversarially untested**, forensic module is *qualitative only* — "not part of the current study to assess the formal validation of the interpretability" |
| 3 | **Shakya et al.** (IEEE ISDFS 2026) | 5 architectures, PGD + cross‑dataset. Clean CelebA = perfect for all 5. **Exp‑3 (clean‑train → adversarial‑test): accuracy exactly 0.50 for all five models.** Exp‑5 cross‑dataset under attack: Acc 0.41–0.60, **AUC 0.38–0.52 (below chance)** | Proves *"in‑distribution accuracy substantially overestimates real‑world reliability."* But it is accuracy‑only — it never asks what happened to the model's *evidence*. Adversarial training did not transfer across datasets |
| 4 | **ARC‑Net** (PLOS ONE 2026) | Attention+residual on EfficientNet‑B0. 99.0 / 97.6 / 99.3 % across three sets. +500 South‑Asian images cut FPR **22.0 % → 10.0 %**, Brier **0.180 → 0.110**. McNemar **p = 0.028**. Grad‑CAM + LIME | Explainability is **post‑hoc and unvalidated** — Grad‑CAM maps are shown, never scored against ground‑truth manipulated regions. Face‑only. No adversarial test |
| 5 | **CT‑DFID** (IEEE TIP 2026) | DTA (gradient‑reversal domain‑tag adversarial) + DFA (MMD alignment) for **extra‑type** cross‑domain. Acc 96.20 %, AUC 97.71 %, works with **< 100 target labels** | Face‑swap only; classification only; assumes an unlabelled target *pool* exists; robustness to post‑processing is asserted, not stress‑tested jointly with adversarial attack |

**Methodological assets to borrow (not to re‑invent):**

- ARC‑Net → McNemar's test, Brier score, subgroup FPR, OOD protocol design.
- CT‑DFID → MMD + gradient‑reversal layer, few‑shot target adaptation, extra‑type split definition.
- Shakya → the five‑experiment matrix (Exp‑1…Exp‑5) as a *ready‑made* robustness protocol you can extend rather than design.
- DeepFakeBuster → the confidence‑calibration formulation (reliability priors `w_i^(g)`, input‑conditioned `w_i(I)`) as the fusion baseline you must beat.
- DFBench → your primary test bed and your efficiency comparison target.

---

## 2. The gap that survives all five papers

Stack them and exactly one region of the space is empty:

```
                 classify   localise   degradation-   adversarial   calibrated   cheap
                            (masks)    robust         robust        (ECE/Brier)  (<1B)
DFBench/MoA-DF      Y          N            N              N            N          N (24B)
DeepFakeBuster      Y          N          partial          N            Y          N (11 nets)
Shakya et al.       Y          N            N            measured       N          Y
ARC-Net             Y      post-hoc         N              N            Y          Y
CT-DFID             Y          N          partial          N            N          Y
-------------------------------------------------------------------------------------
GRACE-DF (yours)    Y          Y            Y              Y            Y          Y
```

Three things nobody has done, in order of novelty:

1. **Nobody has ever measured whether an explanation survives a perturbation.** All robustness work measures label stability. This is a free, defensible, first‑of‑its‑kind contribution.
2. **Nobody has built a grounded detector that is trained to be degradation‑invariant in the frequency domain.** (Your Idea 3 supplies the mechanism; Idea 2 supplies the output format.)
3. **Nobody has shown that a sub‑1 B model can replace a 24 B agent ensemble.** DFBench hands you the exact number to beat.

---

## 3. The four claims your paper will make

| Claim | Statement | Killed by which experiment |
|-------|-----------|----------------------------|
| **C1 — Evidence drift exists** | Under degradation and PGD, detectors retain label accuracy while their evidence maps decorrelate from the true manipulated region | E2, E3 (must show ESI collapse in ≥ 2 baselines) |
| **C2 — Frequency tokens fix it** | Injecting DCT/wavelet tokens + degradation‑invariant contrastive alignment restores localisation fidelity under degradation | E5 ablation (frequency tokens ON/OFF) |
| **C3 — Grounding is cheap** | 0.23–0.8 B grounded model ≥ MoA‑DF (24 B) accuracy on DFBench AI‑edit, at ~30× fewer parameters | E6 |
| **C4 — Calibrated abstention converts robustness into utility** | With conformal abstention at 10 % coverage loss, adversarial error drops from ~50 % (Shakya's floor) to < Y % | E7 |

If C1 fails you still have a paper (C2–C4 = a good method paper). If C1 holds you have a *strong* paper, because C1 is a claim about the whole field, not about your model.

---

## 4. New metrics — define these formally in Section 3 of the paper

Let `M(I) ∈ [0,1]^{H×W}` be a model's evidence map for image `I` (Grad‑CAM for CNNs, attention rollout for ViTs, predicted mask for grounded models), `G(I)` the ground‑truth manipulation mask, `T` a corruption operator (JPEG q, Gaussian blur σ, noise, resize) or an adversarial attack.

**1. Grounded Accuracy (GAcc)** — right for the right reason:

```
GAcc = (1/N) Σ_i  1[ŷ_i = y_i] · 1[ IoU(M(I_i), G(I_i)) ≥ τ ]      τ = 0.5
```

**2. Evidence Stability Index (ESI)** — the core novel metric:

```
ESI(T) = (1/N) Σ_i  SSIM( M(I_i), M(T(I_i)) )          ∈ [0,1]
```
Report also rank‑correlation variant `ESI_ρ` (Spearman over pixels) for robustness to intensity shifts.

**3. Evidence–Label Divergence (ELD)** — quantifies C1 directly:

```
ELD(T) = [Acc(I) − Acc(T(I))]  −  [ESI(T)-implied drop]
```
Simpler and cleaner to present: plot `ΔAcc` vs `ΔESI` per corruption level. A point in the *high ΔESI / low ΔAcc* quadrant **is** evidence drift. Make this Figure 1 of the paper.

**4. Localisation‑under‑attack (LuA)** — mean IoU on adversarial inputs, reported alongside AUC.

**5. Calibration** — ECE (15 bins) + Brier, following ARC‑Net, but reported *per corruption level*, which ARC‑Net does not do.

**6. Efficiency** — params, GFLOPs, ms/image on a single T4, following DeepFakeBuster's deployment critique.

> **Do this before anything else:** these six definitions plus their reference implementation are ~200 lines of code and they are the spine of the paper. Write them in Week 1, unit‑test them, freeze them.

---

## 5. Data plan

### 5.1 What you need

| Purpose | Source | Masks? | Size to use |
|---------|--------|--------|-------------|
| Primary benchmark, classification | **DFBench** (github.com/IntMeGroup/DFBench) — real 45 K, AI‑edit 15 K, AI‑gen 480 K | No | Curated **60 K** subset |
| Localisation supervision | **SID‑Set** (SIDA) — tampered images with pixel masks | **Yes** | ~20 K |
| Localisation supervision | **MagicBrush**, **AutoSplice**, **CocoGlide**, **DEFACTO** | **Yes** | ~15 K combined |
| Face‑domain transfer | **FaceForensics++** (masks available), **Celeb‑DFv2** | Partial | ~10 K frames |
| Degradation reference | **CSIQ, TID2013, KADID‑10k, KonIQ‑10k, LIVE, CLIVE** (already inside DFBench) | n/a | as provided |
| Fairness / OOD replication | ARC‑Net's 140 K Real/Fake Faces (Kaggle) | No | ~10 K |

### 5.2 Pseudo‑masks — the trick that makes this feasible

DFBench's AI‑edit subset is built from *real source images that were then edited*. You therefore have `(I_real, I_edit)` pairs. Derive a mask without any annotation:

```
D   = |I_edit − I_real|                       # per-pixel, in LAB space
D   = gaussian_blur(D, σ=2)
G   = binarise(D, threshold = Otsu(D))
G   = morphological_open → close → largest_k_components
reject if area(G) > 0.6·HW  or  area(G) < 0.001·HW   # style-change / no-op cases
```

- Validate on SID‑Set where real masks exist: report pseudo‑mask IoU vs true mask. **If mean IoU > 0.7, the trick is publishable**; state the number in the paper.
- This converts DFBench's four edit categories (Object Enhance / Object Operation / Semantic Change / Style Change) into a **localisation benchmark that does not exist today**. That alone is a contribution worth a dataset section.
- Style Change will mostly be rejected (global edits) — that's correct behaviour, note it explicitly.

### 5.3 Kaggle data logistics (this kills projects — plan it now)

- **Never** put 540 K images in `/kaggle/working` (20 GB cap, and it must persist between sessions).
- Build **private Kaggle Datasets**, one per split, each ≤ ~40 GB, uploaded via `kaggle datasets create/version` from your local machine.
- Store images as **WebDataset `.tar` shards** or a single **LMDB / HDF5** per split. Thousands of small files make Kaggle input mounts crawl.
- Pre‑resize once to 384 px shortest side and re‑encode as JPEG q=95 — **and keep a small pristine PNG subset (~5 K)** for the degradation experiments, because you cannot study compression artefacts on already‑compressed data. This is a real methodological trap; the paper should say you avoided it.
- Pre‑compute and cache DCT/wavelet maps as `float16` `.npy` shards. Computing them on the fly will make you dataloader‑bound on 2 vCPUs.

**Target working set: ≈ 60 K train / 8 K val / 15 K test, ~35 GB.** Resist the urge to use all 540 K — you cannot afford it and you do not need it.

---

## 6. Model plan — what actually fits on Kaggle

You **cannot** fine‑tune Qwen2.5‑7B / InternVL‑8B / 9B on a P100 or 2×T4. Do not try. Reframe the constraint as your efficiency contribution.

### Tier A — primary model (build this)

**Grounded backbone: `Florence-2-base` (0.23 B) or `Florence-2-large` (0.77 B)**
Reasons: natively emits bounding boxes and region tokens (`<CAPTION_TO_PHRASE_GROUNDING>`, `<REGION_TO_SEGMENTATION>`); trainable in fp16 on 16 GB; fast. This is the single most Kaggle‑appropriate grounded VLM available.

Alternatives if Florence‑2 misbehaves: `Qwen2‑VL‑2B‑Instruct` (QLoRA 4‑bit, r=16, grad‑checkpointing — tight but feasible on 2×T4), or `PaliGemma‑3B‑mix‑448` (QLoRA), or a non‑VLM route: **SegFormer‑B2 / ConvNeXt‑T + FPN dual‑head** (classify + segment) which is guaranteed to fit and trains in hours.

### The GRACE‑DF architecture

```
                    ┌─────────────────────────────────────────┐
 I (RGB) ───────────► ViT / DaViT vision encoder ──► v_tokens  │
                    └─────────────────────────────────────────┘
                                                          │
 DCT(8×8 blockwise) ─┐                                    │
 DWT (Haar, 2 lvl)  ─┤► FreqEncoder (3-layer conv) ──► f_tokens
 High-pass residual ─┘   (shared spatial grid)            │
                                                          ▼
                              [v_tokens ; f_tokens ; task_prompt]
                                          │
                                   Language / fusion decoder
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              y ∈ {real, edited,     mask / boxes         rationale text
                 fully-generated}    (localisation)       (natural language)
                                          │
                                     Calibration head
                                  (temperature + conformal)
```

**Losses:**

```
L = L_cls  +  λ_loc · L_loc  +  λ_div · L_DIC  +  λ_cal · L_cal

L_loc  = Dice + BCE  (or L1 on box coords for Florence-2's token format)
L_DIC  = degradation-invariant contrastive:
         pull  z(I)  ↔  z(T(I))     for the same image under corruption T
         push  z(I)  ↔  z(I')       for different images
         (InfoNCE, τ=0.07, on the pooled fused representation)
L_cal  = focal / label-smoothing term; temperature fitted post-hoc on val
```

`L_DIC` is the mechanism for C2 and is directly transplanted from CT‑DFID's philosophy (align distributions) but applied to **corruption domains rather than generator domains** — that reframing is your novelty over CT‑DFID, and you should say so explicitly.

**Optional +1 (only if Weeks 1–14 go well):** a lightweight two‑agent debate — a *texture agent* (frequency branch) and a *semantic agent* (RGB branch) each emit a box + a short justification, and the decoder arbitrates. This recovers your doc's Idea 2 "debate" element at 2 % of MoA‑DF's cost. **Treat this as a stretch goal, not a dependency.**

### Tier B — baselines you must reproduce (non‑negotiable for a top‑tier venue)

| Baseline | Why | Effort |
|----------|-----|--------|
| EfficientNet‑B0, ResNet‑18, ConvNeXt‑T, ViT‑B/16, MobileNetV3‑S | Exactly Shakya's five — lets you *extend* their protocol rather than compete with it | 1 week, cheap |
| ARC‑Net (attention+residual EfficientNet‑B0) | Reproduce from PLOS ONE; it's small and the paper is detailed | 3–4 days |
| DeepFakeBuster‑style adaptive fusion | Implement the `w_i(I)` fusion over your 5 CNNs — do **not** rebuild all 11 detectors | 3 days |
| CT‑DFID (DTA+DFA) | Code released at `github.com/QinQin741/DTA-DFA-DA-model` — run as‑is | 2 days |
| MoA‑DF | **Zero‑shot inference only**, on a 5 K test subset, in 4‑bit. Do not fine‑tune | 4–5 days incl. debugging |
| LGrad, UnivFD, CnnSpott | Standard AIGC detectors, public weights, zero‑shot | 3 days |

> MoA‑DF zero‑shot in 4‑bit on 5 K images across three 7–9 B models will take roughly 12–20 GPU‑hours. Budget an entire week's quota for it and run it **once**, caching all logits to disk. Never re‑run it.

---

## 7. Experiment plan

Each experiment states what it proves and what you do if it fails.

### E0 — Infrastructure (Week 1–2)
Metric implementations (§4), pseudo‑mask pipeline + IoU validation on SID‑Set, corruption suite (JPEG q ∈ {90,70,50,30,10}, Gaussian blur σ ∈ {0.5,1,2,3}, Gaussian noise σ ∈ {5,10,20}, downscale ∈ {256,128,64,32}, WebP, screenshot‑resave chain), Kaggle dataset shards, checkpoint/resume harness.
*Fail‑safe:* if pseudo‑mask IoU < 0.5 on SID‑Set, drop DFBench localisation and use SID‑Set + MagicBrush + AutoSplice only. Paper survives.

### E1 — Baseline reproduction (Week 3–4)
Train the 5 CNNs/ViTs + ARC‑Net on your split. Reproduce Shakya's Exp‑1 (clean ≈ perfect) and Exp‑3 (clean‑train → PGD‑test ≈ 0.50). **You must reproduce the 0.50 collapse** — it validates your pipeline and it is the launchpad for C1.

### E2 — Evidence drift under degradation ★ (Week 5–6)
For every baseline, at every corruption level: measure `Acc`, `ESI`, `GAcc`, `IoU`, `ECE`.
**The money plot:** x = corruption severity, twin y‑axes = Accuracy (flat‑ish) and ESI (collapsing).
*This is the experiment that makes the paper interesting. Run it before you build your model.* If you see the divergence, you have C1 in Week 6 and everything after is upside.

### E3 — Evidence drift under attack ★ (Week 6–7)
PGD (ε = 2/255, 4/255, 8/255; 10 steps), FGSM, and two novel variants worth defining:
- **Evidence‑targeted attack:** maximise `1 − SSIM(M(I), M(I+δ))` *subject to* `ŷ` unchanged. Shows an attacker can corrupt the forensic explanation while leaving the verdict intact — a genuinely alarming, quotable result for a forensics venue.
- **Transfer attack:** perturbations crafted on ResNet‑18, evaluated on all others (Shakya showed adversarial *training* doesn't transfer; show whether *attacks* do).

### E4 — GRACE‑DF training (Week 7–11)
Stage 1: freeze vision encoder, train frequency encoder + fusion + heads (3–4 epochs).
Stage 2: unfreeze with LoRA (r=16, α=32, on attention projections), add `L_DIC` (2–3 epochs).
Stage 3: fit temperature + conformal threshold on val.
Mixed precision fp16, grad‑checkpointing, effective batch 32 via accumulation, cosine LR 1e‑4 → 1e‑6, AdamW wd 0.01.

### E5 — Ablation (Week 11–12)
Rows: `base` → `+freq tokens` → `+L_DIC` → `+loc head` → `+calibration` → `full`.
Columns: Acc, GAcc, IoU, ESI(JPEG50), ESI(PGD 4/255), ECE, params, ms/img.
Also ablate frequency representation: DCT only / DWT only / high‑pass only / all three.
*This table is what reviewers grade you on. Budget real time for it.*

### E6 — Efficiency vs MoA‑DF (Week 12)
GRACE‑DF (0.23 B / 0.77 B) vs MoA‑DF (24 B) on the DFBench AI‑edit and real subsets. Report Acc, F1, params, latency, VRAM. Specifically target **Object Enhance and Object Operation**, where DFBench reports zero‑shot LMM averages of 15.89 % and 18.68 % Acc — the fine‑grained local categories your localisation head is designed for. A large win here is your strongest single result.

### E7 — Calibrated abstention (Week 13)
Split conformal prediction on the val set; sweep coverage 100 % → 70 %; plot risk–coverage curves for clean / degraded / adversarial. Show the operating point where adversarial error falls below a usable threshold.

### E8 — Cross‑domain and fairness (Week 13–14)
- Extra‑type transfer following CT‑DFID's protocol (train on diffusion, test on GAN/AR generators, and vice versa).
- Face vs general‑content transfer (FF++ → DFBench and back).
- ARC‑Net‑style subgroup FPR on the South‑Asian / demographic slice if obtainable; otherwise report per‑generator FPR spread as the fairness proxy and say so honestly.

---

## 8. Twenty‑four‑week schedule

| Wk | Milestone | Kaggle GPU‑h |
|----|-----------|--------------|
| 1 | Metrics coded + unit‑tested; corruption suite; repo + seeds + config system | 2 |
| 2 | Pseudo‑mask pipeline; validate on SID‑Set; build Kaggle dataset shards | 5 |
| 3 | Train 5 CNN/ViT baselines (clean) | 20 |
| 4 | ARC‑Net + DeepFakeBuster fusion reproduction; **Shakya Exp‑1/3 reproduced** | 25 |
| 5 | **E2 evidence drift — degradation** | 20 |
| 6 | E2 finished; **E3 attacks** begin; ← **GO/NO‑GO on C1** | 25 |
| 7 | E3 done incl. evidence‑targeted attack; write Fig. 1 + Sec. 1 draft | 20 |
| 8 | GRACE‑DF Stage‑1 training; frequency encoder debugging | 28 |
| 9 | Stage‑2 LoRA + `L_DIC`; first end‑to‑end numbers | 28 |
| 10 | Hyper‑parameter sweep (λ_loc, λ_div, τ); pick final config | 28 |
| 11 | Final training runs × 3 seeds | 28 |
| 12 | **E5 ablation** + **E6 MoA‑DF comparison** (cache all logits) | 28 |
| 13 | E7 conformal + E8 cross‑domain | 22 |
| 14 | Buffer / re‑runs / failed‑experiment recovery | 25 |
| 15 | All figures at publication quality; all tables auto‑generated from JSON | 8 |
| 16–18 | **Write the paper.** Method → Experiments → Related Work → Intro → Abstract, in that order | 5 |
| 19 | Internal review; supervisor + one external reader | 2 |
| 20 | Address review; rebuttal‑proofing (pre‑empt the 5 questions in §11) | 8 |
| 21 | Supplementary: full ablations, failure cases, dataset card, licence audit | 3 |
| 22 | Code release prep (anonymised repo, model card, reproduce.sh) | 3 |
| 23 | **Submit** | — |
| 24 | arXiv preprint + slack for deadline slippage | — |

Total ≈ **330 GPU‑hours** against a ~30 h/week quota over 24 weeks (~720 h available). Roughly 2.2× headroom — which is the right margin, because half of it will be eaten by crashed sessions and mistakes.

---

## 9. Kaggle engineering rules (obey these or lose weeks)

1. **Every training script resumes from `/kaggle/working/ckpt_last.pt` on start.** Assume every session dies at 11 h 58 m. Write this on day one.
2. Push checkpoints to a **Kaggle Dataset** at the end of every session (`kaggle datasets version`) — `/kaggle/working` is capped at 20 GB and is easy to lose.
3. **No bf16** on P100/T4. Use fp16 + `GradScaler`, and watch for NaNs in the contrastive loss (clamp logits, use `eps=1e-8`).
4. 2×T4 → use `DataParallel` or `accelerate` with `device_map`; a P100 is a *single* faster card. Pick per‑job, don't mix.
5. Only 2–4 vCPUs: `num_workers=2`, pre‑decode to shards, cache frequency maps. You will be I/O‑bound otherwise.
6. Log every run to a JSON in `/kaggle/working/runs/` with git SHA, config, seed, and all metrics. **Tables in the paper get generated from these files, never typed by hand.**
7. Set `torch.use_deterministic_algorithms(True)` where possible; run 3 seeds for headline numbers; report mean ± std.
8. Never run MoA‑DF twice. Cache logits.
9. Keep a `SESSION_LOG.md` — one line per session: what ran, what broke, what's next. Future‑you in month 4 will need it.

---

## 10. Statistical protocol (this is what separates a top‑tier paper from a mid one)

- **3 seeds minimum** for every headline model; report mean ± std, not a single best run.
- **McNemar's test** for paired classification comparisons (follow ARC‑Net; report the p‑value in the table).
- **Bootstrap 95 % CIs** (10 000 resamples) for IoU, ESI, AUC.
- **Holm–Bonferroni** correction across the ablation family — reviewers at TIFS/CVPR do check.
- **Report ECE and Brier per corruption level**, which no baseline does.
- Pre‑register your test split and never touch it until Week 12. Use val for everything.

---

## 11. Risk register — with kill criteria

| Risk | Probability | Mitigation | Kill criterion |
|------|-------------|------------|----------------|
| Evidence drift (C1) doesn't appear | Medium | Try multiple explanation methods (Grad‑CAM, Score‑CAM, attention rollout, IG); drift may be method‑dependent, which is *itself* a finding | If no drift by **end of Week 6**, drop C1 to a secondary observation and lead with C2+C3 (method paper). Do not spend Week 7+ chasing it |
| Pseudo‑masks too noisy | Medium | Validate on SID‑Set first | IoU < 0.5 → use annotated datasets only; DFBench becomes classification‑only |
| Florence‑2 won't fine‑tune stably in fp16 | Medium‑high | Fall back to Qwen2‑VL‑2B QLoRA, then to SegFormer/ConvNeXt dual‑head | Two weeks of instability → switch. The dual‑head CNN route *will* work and still supports every claim except natural‑language rationale |
| MoA‑DF too slow to run at all | Medium | 4‑bit, 5 K subset, batch inference, flash attention off on T4 | Can't finish in one week's quota → cite DFBench's published numbers on their splits and evaluate on a matching subset, clearly stating the caveat |
| DFBench download / storage burden | Medium | Use the AI‑edit + real subsets fully (60 K), sample AI‑gen | — |
| Kaggle quota exhaustion in a critical week | High | 2.2× headroom; keep Weeks 14 and 20 as buffer; register a second Kaggle account only if permitted by their ToS (**check — do not risk a ban**) | — |
| Scooped | Low‑medium | Post an arXiv preprint the day you submit; the ESI metric is the defensible flag in the ground | — |

**Five questions a reviewer will ask — have answers by Week 15:**
1. "Is ESI just measuring that Grad‑CAM is unstable, not that the model is?" → answer with the multi‑explanation‑method study and with your grounded model's *predicted* masks (not post‑hoc maps).
2. "Are pseudo‑masks reliable enough to draw conclusions from?" → the SID‑Set validation number.
3. "Why not compare against the full 24 B fine‑tuned MoA‑DF?" → compute statement + zero‑shot comparison + DFBench's own published fine‑tuned numbers.
4. "Is frequency‑domain input novel?" → it isn't, and don't claim it is; *what's* novel is frequency tokens inside a grounded VLM sequence trained with a degradation‑invariance objective, evaluated on localisation stability.
5. "Does this hold on video?" → no, state it as a limitation and cite your own Idea 4 as future work.

---

## 12. Paper skeleton and venue

**Title (working):** *Right for the Wrong Pixels: Evidence Drift in Deepfake Detection and a Grounded, Frequency‑Aware Remedy*

```
1  Introduction            — Fig.1 = ΔAcc vs ΔESI divergence plot
2  Related Work            — LMM benchmarks / ensembles / adversarial / UDA / grounding (your 5 baselines are the spine)
3  Evidence Drift          — definitions: GAcc, ESI, ELD, LuA  (§4 above)
4  GRACE-DF                — architecture, frequency tokens, L_DIC, calibration
5  Experimental Setup      — data, pseudo-masks + validation, corruption suite, baselines
6  Results                 — E2, E3, E5, E6, E7, E8
7  Limitations             — image-only; pseudo-mask noise; no video/audio; single-language rationales
8  Conclusion
```

**Venues (verify current dates — these are typical windows, not confirmed 2027 deadlines):**

| Venue | Typical window | Fit |
|-------|----------------|-----|
| **IEEE TIFS** | Rolling | ★★★★★ Best fit: forensics, robustness, statistical rigour, no deadline pressure. Where CT‑DFID's sibling work lives |
| **ICCV / CVPR** | ~Mar / ~Nov | ★★★★ Needs the strongest version of C1 + C3 |
| **ACM MM** | ~Apr | ★★★★ DFBench's own venue — reviewers will already know the baseline |
| **IEEE TIP** | Rolling | ★★★★ CT‑DFID's venue |
| **WACV / ICASSP / WIFS** | varies | ★★★ Good fallback / earlier workshop version |

Given a 3–6 month horizon and Kaggle‑scale compute, **IEEE TIFS is the honest primary target** — rolling submission removes the deadline gamble, and the venue rewards exactly the kind of rigorous robustness protocol this plan produces. Aim a short version at a workshop (WIFS / CVPRW media forensics) around Week 14 if you want an early flag.

---

## 13. Your first seven days — concrete checklist

- [ ] **Day 1** — Create the repo. `configs/`, `src/{data,models,metrics,attacks,train,eval}/`, `notebooks/`, `runs/`. Set up seeds, a config dataclass, and JSON run‑logging. Write `SESSION_LOG.md`.
- [ ] **Day 1** — Kaggle account check: confirm current GPU quota, session limit, and `/kaggle/working` cap. Write the numbers into the README.
- [ ] **Day 2** — Implement and unit‑test the six metrics from §4. Synthetic test cases: identical maps → ESI = 1; random maps → ESI ≈ 0.
- [ ] **Day 2** — Implement the corruption suite as a deterministic, seeded transform bank.
- [ ] **Day 3** — Download SID‑Set + MagicBrush + AutoSplice. Get a small DFBench slice (real + AI‑edit).
- [ ] **Day 3–4** — Build the pseudo‑mask pipeline. **Validate against SID‑Set masks. Record the IoU. This number decides the shape of the project.**
- [ ] **Day 5** — Build Kaggle Dataset shards (WebDataset tars). Verify a training loop reads them at > 200 img/s.
- [ ] **Day 5** — Write the resume‑from‑checkpoint harness and test it by killing a job halfway.
- [ ] **Day 6** — Fine‑tune EfficientNet‑B0 on a 10 K subset as a smoke test; run the *whole* eval pipeline on it (Acc, ESI, GAcc, ECE) end to end.
- [ ] **Day 6** — Implement PGD and the evidence‑targeted attack; sanity‑check that PGD drops clean accuracy to ~0.5, reproducing Shakya.
- [ ] **Day 7** — Produce a first, ugly version of Figure 1 (ΔAcc vs ΔESI on one model). **If the divergence is visible in that scrappy plot, you have your paper.**

---

## 14. Reading list before Week 3

Beyond your five baselines:

- **Grounding / localisation:** HiFi‑IFDL, TruFor, IML‑ViT, SIDA (SID‑Set), MMTD‑Set, CAT‑Net, PSCC‑Net, MVSS‑Net.
- **Frequency for forgery:** F3‑Net, SPSL, FreqNet, LGrad, NPR, Frank et al. 2020, Durall et al. 2020.
- **Robustness / explanation stability:** Ghorbani et al. *Interpretation of NNs is Fragile* (AAAI'19) — **the closest prior art to your ESI idea; read it first and position against it carefully** (they attack saliency in classification; you measure natural degradation in forensics with ground‑truth masks).
- **Calibration / abstention:** Guo et al. temperature scaling; Angelopoulos & Bates conformal prediction tutorial.
- **VLM efficiency:** Florence‑2, Qwen2‑VL, PaliGemma, LoRA/QLoRA.
- **Deepfake surveys 2025–26** for Related Work coverage.

---

## 15. What to do right now

1. Confirm the primary model choice (Florence‑2 vs a CNN dual‑head) — this determines whether the paper can claim natural‑language rationales.
2. Confirm you can obtain **SID‑Set**; the pseudo‑mask validation depends on it.
3. Start §13 Day 1. The metric code and the pseudo‑mask IoU are the two things that decide whether this plan is real, and both are done inside a week.

---

*Prepared from the critical analysis of DFBench/MoA‑DF and the five supplied baseline papers. Ideas 2 (visual grounding) and 3 (frequency‑token injection) are fused here; Idea 1 (MoE routing) is absorbed as the efficiency argument in C3; Idea 4 (video) is reserved as declared future work.*
