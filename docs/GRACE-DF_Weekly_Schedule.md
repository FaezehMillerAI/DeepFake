# GRACE‑DF — Weekly Execution Schedule

Companion to `GRACE-DF_Research_Plan.md`
**Start:** Monday 7 September 2026 · **Submit:** week of 8 February 2027 · **Slack:** to 21 February 2027
Compute: Kaggle (~30 GPU‑h/week) · Budget: **~330 GPU‑h of ~720 available (2.2× headroom)**

---

## How to use this document

- Each week has **one goal**. If you can't state it in a sentence on Friday, the week failed — say so in `SESSION_LOG.md` and use slack.
- **Gates (G0–G5)** are hard decision points. At a gate you either continue on the primary path or switch to the named fallback. *Do not negotiate with a gate.* The whole point of writing them down now is that in Week 9, at 1 a.m., you will want to.
- Checkboxes are the unit of work. If a task can't be ticked in ≤ 1 day, split it.
- GPU hours are budgets, not targets. Underspending is fine; overspending steals from Week 14 and Week 21, which are your only real buffers.

**Weekly rituals**

| When | Ritual | Time |
|------|--------|------|
| Mon 09:00 | Read last week's log. Write this week's one‑sentence goal at the top of `SESSION_LOG.md`. Check remaining GPU quota. | 15 min |
| Every session | One line in `SESSION_LOG.md`: what ran, what broke, what's next. Push checkpoints to Kaggle Dataset before the session dies. | 2 min |
| Fri 16:00 | Did the goal happen? Update the gate status. Tick the boxes below. Commit results JSON. | 30 min |
| Fri (biweekly) | Supervisor update: one page — plot, number, blocker. Never a status essay. | 30 min |

**Slack policy:** Weeks **14** and **21** are protected buffer. Do not schedule work into them in advance. If you're more than 5 days behind by Week 8, cut Tier‑B baselines (LGrad/UnivFD/CnnSpot) before you cut experiments.

---

## Phase map

| Phase | Weeks | Dates | Purpose | GPU‑h |
|-------|-------|-------|---------|-------|
| **I — Foundations** | 1–2 | 7–20 Sep | Metrics, masks, data infrastructure | 7 |
| **II — Baselines** | 3–4 | 21 Sep – 4 Oct | Reproduce the field, validate the pipeline | 45 |
| **III — The Discovery** | 5–7 | 5–25 Oct | Evidence drift: the paper's core claim | 65 |
| **IV — The Method** | 8–11 | 26 Oct – 22 Nov | Build and train GRACE‑DF | 112 |
| **V — Evidence** | 12–13 | 23 Nov – 6 Dec | Ablation, MoA‑DF comparison, cross‑domain | 50 |
| **VI — Buffer** | 14 | 7–13 Dec | Protected recovery week | 25 |
| **VII — Writing** | 15–20 | 14 Dec – 24 Jan | Figures, draft, review, revision | 20 |
| **VIII — Hardening** | 21–24 | 25 Jan – 21 Feb | Extra experiments, release, submit | 8 |

> **Holiday note:** Weeks 16–17 (21 Dec – 3 Jan) are deliberately loaded with low‑intensity, offline‑friendly work — reading, Related Work prose, dataset cards. Do not plan training runs there.

---

# PHASE I — FOUNDATIONS

## Week 1 · 7–13 Sep · *Build the ruler before you measure anything*

**Goal:** The six metrics from §4 of the plan are coded, tested and frozen.

- [ ] Create repo: `configs/ src/{data,models,metrics,attacks,train,eval}/ notebooks/ runs/ SESSION_LOG.md`
- [ ] Config system (dataclass + YAML), global seeding, git‑SHA stamping on every run
- [ ] JSON run‑logger — **every table in the paper will be generated from these files, never typed**
- [ ] Implement `GAcc`, `ESI` (SSIM + Spearman variants), `ELD`, `LuA`, `ECE`(15‑bin), `Brier`
- [ ] Unit tests: identical maps → ESI = 1.0 · random maps → ESI ≈ 0 · shifted‑intensity maps → ESI_ρ stable
- [ ] Corruption suite as a deterministic seeded transform bank: JPEG q∈{90,70,50,30,10}, blur σ∈{0.5,1,2,3}, noise σ∈{5,10,20}, downscale∈{256,128,64,32}, WebP, screenshot‑resave chain
- [ ] Confirm and record current Kaggle limits (GPU quota, session cap, `/kaggle/working` cap) in README
- [ ] Decide: **Florence‑2 vs CNN dual‑head** — this determines whether you can claim NL rationales

**Deliverable:** `src/metrics/` with green tests · `README.md` with Kaggle constraints
**GPU:** 2 h

---

## Week 2 · 14–20 Sep · *The number that decides the project*

**Goal:** Know whether pseudo‑masks work.

- [ ] Acquire **SID‑Set** (annotated masks), MagicBrush, AutoSplice, CocoGlide
- [ ] Acquire DFBench real + AI‑edit subsets; sample AI‑gen
- [ ] Build pseudo‑mask pipeline: LAB difference → blur σ=2 → Otsu → open/close → largest‑k components → area rejection (>0.6 HW or <0.001 HW)
- [ ] **Validate pseudo‑masks against SID‑Set ground truth. Record mean IoU.**
- [ ] Report pseudo‑mask yield per DFBench edit category (expect Style Change to be mostly rejected — that's correct)
- [ ] Build WebDataset `.tar` shards; pre‑resize to 384px; **keep a 5 K pristine‑PNG subset** for degradation work
- [ ] Pre‑compute DCT / DWT / high‑pass maps as fp16 `.npy` shards
- [ ] Upload as private Kaggle Datasets (≤ 40 GB each); verify a loop reads > 200 img/s
- [ ] Write and test the resume‑from‑checkpoint harness — **kill a job at 50 % and prove it resumes**

### ▶ GATE G0 — *Pseudo‑mask viability* (Fri 18 Sep)

| Mean IoU vs SID‑Set | Decision |
|---|---|
| **≥ 0.70** | Full plan. DFBench becomes a localisation benchmark — this is itself a contribution |
| **0.50 – 0.70** | Use pseudo‑masks for training only; evaluate localisation on annotated sets only. Report the IoU honestly |
| **< 0.50** | **Drop DFBench localisation.** Train and evaluate masks on SID‑Set + MagicBrush + AutoSplice. DFBench stays classification‑only. Plan survives; C1 and C3 unaffected |

**Deliverable:** `pseudo_mask_validation.json` · shards live on Kaggle · resume harness proven
**GPU:** 5 h

---

# PHASE II — BASELINES

## Week 3 · 21–27 Sep · *Reproduce the field*

**Goal:** Five baselines trained and evaluated on your split.

- [ ] Train ResNet‑18, MobileNetV3‑S, EfficientNet‑B0, ConvNeXt‑T, ViT‑B/16 (Shakya's exact five)
- [ ] Confirm Shakya **Exp‑1**: near‑perfect clean accuracy on all five
- [ ] Implement explanation extraction: Grad‑CAM, Score‑CAM, Integrated Gradients, attention rollout (ViT)
- [ ] Run the full eval pipeline on one model end‑to‑end (Acc, ESI, GAcc, IoU, ECE) — smoke test the whole stack
- [ ] 3 seeds for each; mean ± std into `runs/`

**Deliverable:** Table 1 of the paper (clean baselines) auto‑generated
**GPU:** 20 h

---

## Week 4 · 28 Sep – 4 Oct · *Prove the pipeline is trustworthy*

**Goal:** Reproduce the published collapse. This validates everything downstream.

- [ ] Implement PGD (ε ∈ {2,4,8}/255, 10 steps) and FGSM
- [ ] Run Shakya **Exp‑3**: clean‑train → adversarial‑test
- [ ] Reproduce ARC‑Net (attention + residual EfficientNet‑B0) from the PLOS ONE description
- [ ] Implement DeepFakeBuster‑style adaptive fusion `w_i(I)` over your five CNNs (**not** all 11 detectors)
- [ ] Clone and run CT‑DFID from `github.com/QinQin741/DTA-DFA-DA-model`
- [ ] Set up McNemar / bootstrap‑CI / Holm–Bonferroni helpers in `src/eval/stats.py`

### ▶ GATE G1 — *Pipeline validated* (Fri 2 Oct)

**Pass condition:** Exp‑3 accuracy collapses to **≈ 0.50** across all five models, matching Shakya et al.
**If it doesn't:** your attack implementation or your data split is wrong. **Stop and fix before Week 5.** Every result after this depends on it.

**Deliverable:** Reproduction table with published‑vs‑yours columns
**GPU:** 25 h

---

# PHASE III — THE DISCOVERY

## Week 5 · 5–11 Oct · *Does the evidence move?*

**Goal:** Run E2 — evidence drift under natural degradation.

- [ ] Sweep every baseline × every corruption × every severity
- [ ] Log `Acc`, `ESI`, `ESI_ρ`, `GAcc`, `IoU`, `ECE`, `Brier` at each point
- [ ] Repeat for **all four explanation methods** — drift may be method‑dependent, which is *itself* a finding
- [ ] Draft the money plot: x = severity, twin y = Accuracy (flat) vs ESI (collapsing)
- [ ] Compute the ΔAcc / ΔESI quadrant scatter

**Deliverable:** First version of **Figure 1**
**GPU:** 20 h

---

## Week 6 · 12–18 Oct · *The gate that defines the paper*

**Goal:** Finish E2, start E3, decide what the paper is about.

- [ ] Finish degradation sweep incl. compression chains and the pristine‑PNG subset
- [ ] Begin E3: PGD / FGSM evidence measurement
- [ ] Implement the **evidence‑targeted attack**: maximise `1 − SSIM(M(I), M(I+δ))` subject to `ŷ` unchanged
- [ ] Begin transfer attacks (craft on ResNet‑18, evaluate on all)

### ▶ GATE G2 — *Evidence drift confirmed?* ★ (Fri 16 Oct)

| Outcome | Decision |
|---|---|
| **Drift visible** (ESI falls ≫ Acc across ≥ 2 baselines) | **C1 is your lead claim.** Paper becomes a field‑level finding + a remedy. Retitle around it |
| **Drift only in some explanation methods** | Still publishable — reframe as "post‑hoc explanations are unreliable under degradation; predicted masks are not." Slightly weaker, still novel |
| **No drift anywhere** | **Demote C1 to a secondary observation.** Lead with C2 + C3 (method paper: cheap grounded detector beats 24 B ensemble). **Do not spend Week 7 chasing it** |

**Deliverable:** Gate decision written into `SESSION_LOG.md` with the plot that justifies it
**GPU:** 25 h

---

## Week 7 · 19–25 Oct · *Bank the finding, then start building*

**Goal:** Phase III closed and written up while it's fresh.

- [ ] Complete E3 including evidence‑targeted and transfer attacks
- [ ] Publication‑quality Figure 1 + Figure 2 (attack version)
- [ ] **Write Section 1 (Intro) and Section 3 (Evidence Drift) now** — you will never have more clarity about this than this week
- [ ] Read Ghorbani et al. *Interpretation of NNs is Fragile* and write the positioning paragraph against it
- [ ] Scaffold GRACE‑DF: frequency encoder, token concatenation, head stubs
- [ ] Supervisor checkpoint: present Figure 1

**Deliverable:** ~4 pages of real paper prose · model scaffold compiles
**GPU:** 20 h

---

# PHASE IV — THE METHOD

## Week 8 · 26 Oct – 1 Nov · *Get it training at all*

**Goal:** Stage‑1 training runs without NaNs.

- [ ] Load Florence‑2‑base; verify grounding tokens work out of the box on sample images
- [ ] Wire FreqEncoder (3‑layer conv over DCT + DWT + high‑pass, shared spatial grid) → `f_tokens`
- [ ] Concatenate `[v_tokens ; f_tokens ; prompt]`; freeze vision encoder
- [ ] Implement `L_cls` + `L_loc` (Dice + BCE, or L1 on box tokens)
- [ ] fp16 + GradScaler + grad‑checkpointing; **watch for NaN in fp16 — clamp logits, eps=1e‑8, no bf16 on P100/T4**
- [ ] Stage‑1: 3–4 epochs, frozen backbone
- [ ] Overfit a 200‑image subset to loss ≈ 0 as a correctness check **before** any full run

**Deliverable:** First GRACE‑DF checkpoint with non‑trivial IoU
**GPU:** 28 h

---

## Week 9 · 2–8 Nov · *The mechanism*

**Goal:** Degradation‑invariant contrastive loss working; first end‑to‑end numbers.

- [ ] Implement `L_DIC`: InfoNCE, τ = 0.07, positives = same image under corruption `T`, negatives = other images
- [ ] Stage‑2: unfreeze with LoRA (r=16, α=32, attention projections), 2–3 epochs
- [ ] Effective batch 32 via accumulation; cosine LR 1e‑4 → 1e‑6; AdamW wd 0.01
- [ ] First full eval: Acc, GAcc, IoU, ESI(JPEG50), ESI(PGD 4/255), ECE
- [ ] Compare against best CNN baseline from Week 3

### ▶ GATE G3 — *Method viability* (Fri 6 Nov)

| Outcome | Decision |
|---|---|
| GRACE‑DF **> best baseline on GAcc and ESI** | Continue on Florence‑2 |
| Comparable but unstable / fp16 problems persist | Switch to **Qwen2‑VL‑2B QLoRA** (one week) |
| Two weeks of instability total | **Switch to SegFormer‑B2 / ConvNeXt‑T + FPN dual‑head.** Guaranteed to train; supports every claim except natural‑language rationale. Take this exit without regret |

**Deliverable:** First honest comparison table
**GPU:** 28 h

---

## Week 10 · 9–15 Nov · *Tune it*

**Goal:** Final configuration chosen and justified.

- [ ] Sweep λ_loc, λ_div, τ (contrastive), LoRA rank — coarse grid, 3–4 points each, **not** a full factorial
- [ ] Ablate frequency representation: DCT only / DWT only / high‑pass only / all three
- [ ] Fit temperature scaling on val; compute ECE before/after
- [ ] Freeze the final config into `configs/grace_df_final.yaml` and **do not touch it again**

**Deliverable:** `grace_df_final.yaml` + sweep table for the supplementary
**GPU:** 28 h

---

## Week 11 · 16–22 Nov · *Final runs*

**Goal:** Three seeds of the final model, done properly.

- [ ] Train 3 seeds end‑to‑end with the frozen config
- [ ] Full eval per seed; mean ± std everywhere
- [ ] Efficiency measurements: params, GFLOPs, ms/image on a single T4, peak VRAM
- [ ] Archive all three checkpoints to a Kaggle Dataset
- [ ] Supervisor checkpoint: main results

**Deliverable:** Main results table with error bars
**GPU:** 28 h

---

# PHASE V — EVIDENCE

## Week 12 · 23–29 Nov · *The table reviewers grade you on*

**Goal:** E5 ablation + E6 the headline comparison.

- [ ] **E5 ablation ladder:** `base` → `+freq tokens` → `+L_DIC` → `+loc head` → `+calibration` → `full`
- [ ] Columns: Acc · GAcc · IoU · ESI(JPEG50) · ESI(PGD 4/255) · ECE · params · ms/img
- [ ] **E6 — MoA‑DF zero‑shot** (Qwen2.5‑7B + InternVL2.5‑8B + InternVL3‑9B, 4‑bit, 5 K subset)
  - [ ] **Cache every logit to disk. Run this once. Never re‑run it.**
  - [ ] Target the **Object Enhance / Object Operation** categories where DFBench reports zero‑shot LMM averages of 15.89 % / 18.68 % Acc
- [ ] Efficiency comparison: 0.23–0.8 B vs 24 B

**Deliverable:** Ablation table + headline efficiency table
**GPU:** 28 h — *this week is fully committed; protect it*

---

## Week 13 · 30 Nov – 6 Dec · *Close the remaining claims*

**Goal:** E7 + E8 done. All experiments complete.

- [ ] **E7:** split conformal on val; sweep coverage 100 % → 70 %; risk–coverage curves for clean / degraded / adversarial
- [ ] Identify the operating point where adversarial error falls below usable threshold (vs Shakya's 0.50 floor)
- [ ] **E8a:** extra‑type transfer (diffusion→GAN, GAN→diffusion) following CT‑DFID's protocol
- [ ] **E8b:** face↔general transfer (FF++ ↔ DFBench)
- [ ] **E8c:** fairness — subgroup FPR if the demographic slice is obtainable, otherwise per‑generator FPR spread (state the substitution honestly)
- [ ] Run the full statistical protocol: McNemar, bootstrap CIs, Holm–Bonferroni

### ▶ GATE G4 — *Experiments complete* (Fri 4 Dec)

**Pass condition:** Every table and figure in §12 of the plan has real numbers. **Test set is now frozen — no further peeking.**
**If not:** use Week 14 buffer. If still incomplete after Week 14, cut E8c (fairness) and the debate module, in that order.

**Deliverable:** Complete results directory
**GPU:** 22 h

---

# PHASE VI — BUFFER

## Week 14 · 7–13 Dec · *Protected*

**Goal:** Recover whatever slipped. If nothing slipped, use it for the stretch goal.

- [ ] Re‑run any failed / underpowered experiment
- [ ] Fill missing seeds
- [ ] **Only if genuinely ahead:** two‑agent debate module (texture agent + semantic agent + arbiter) — the Idea 2 stretch goal
- [ ] Failure‑case gallery: 20 images where GRACE‑DF localises wrongly, categorised

**Deliverable:** No open experimental questions
**GPU:** 25 h

---

# PHASE VII — WRITING

## Week 15 · 14–20 Dec · *Freeze the visuals*

**Goal:** Every figure and table publication‑ready and auto‑generated.

- [ ] All figures at final quality (vector, consistent palette, colour‑blind safe, legible at print size)
- [ ] All tables generated from `runs/*.json` by script — zero hand‑typed numbers
- [ ] Write Section 4 (GRACE‑DF method) and Section 5 (Experimental Setup)
- [ ] Build the figure/table manifest so a late re‑run regenerates everything with one command

**Deliverable:** `make figures` reproduces the paper's visuals
**GPU:** 8 h

---

## Week 16 · 21–27 Dec · *Reduced capacity — offline work only*

**Goal:** Related Work, written away from a GPU.

- [ ] Section 2 (Related Work) full draft — the five baselines are the spine
- [ ] Reading list: HiFi‑IFDL, TruFor, IML‑ViT, SIDA, CAT‑Net, PSCC‑Net, F3‑Net, SPSL, FreqNet, LGrad, NPR
- [ ] Reference manager tidy; consistent citation keys
- [ ] Section 7 (Limitations) draft — image‑only, pseudo‑mask noise, no video/audio

**GPU:** 0 h

---

## Week 17 · 28 Dec – 3 Jan · *Reduced capacity — supplementary*

**Goal:** Everything that isn't the main paper.

- [ ] Supplementary: full hyper‑parameter sweeps, extended ablations, failure gallery
- [ ] Dataset card: sources, licences, pseudo‑mask method, known limitations
- [ ] Model card
- [ ] Licence audit for every dataset used — **do this before submission, not after acceptance**

**GPU:** 0 h

---

## Week 18 · 4–10 Jan · *Assemble*

**Goal:** Complete first draft, end to end.

- [ ] Section 6 (Results) written against the frozen tables
- [ ] Rewrite Section 1 (Intro) now that you know what the paper actually found
- [ ] Abstract — written last, as always
- [ ] Section 8 (Conclusion)
- [ ] Full read‑through for narrative consistency: does every claim C1–C4 have a named experiment?

**Deliverable:** **Complete draft v1**
**GPU:** 2 h

---

## Week 19 · 11–17 Jan · *External eyes*

**Goal:** Honest review before reviewers get it.

- [ ] Supervisor review
- [ ] One external reader outside the project — ideally someone who will be blunt
- [ ] Self‑review against the **five reviewer questions** in §11 of the plan; write the answers into the paper
- [ ] Check: is every number in the text traceable to a JSON in `runs/`?

### ▶ GATE G5 — *Submission‑readiness* (Fri 15 Jan)

**Pass condition:** Draft complete, reviewed, no known holes.
**If not:** Weeks 20–21 absorb it; submission slips to Week 24 (still inside the 6‑month target).

**GPU:** 2 h

---

## Week 20 · 18–24 Jan · *Revise*

- [ ] Address all review comments
- [ ] Rebuttal‑proofing: pre‑empt the pseudo‑mask reliability question, the "is ESI just Grad‑CAM instability" question, and the "why not full fine‑tuned MoA‑DF" question **in the main text**
- [ ] Tighten to the page limit
- [ ] Draft v2

**GPU:** 8 h *(reserved for review‑driven extra experiments)*

---

# PHASE VIII — HARDENING

## Week 21 · 25–31 Jan · *Protected — the "one more experiment" week*

**Goal:** Absorb the experiment your reviewers will inevitably ask for.

- [ ] Run whatever Week 19–20 review demanded
- [ ] Regenerate affected figures via the manifest
- [ ] Update text

**GPU:** 8 h

---

## Week 22 · 1–7 Feb · *Reproducibility*

- [ ] Anonymised public repo
- [ ] `reproduce.sh` — clone → download → train → evaluate, tested from scratch
- [ ] Release checkpoints and the pseudo‑mask pipeline
- [ ] **Have someone else run `reproduce.sh` on a fresh Kaggle account**
- [ ] Final licence and ethics statement

---

## Week 23 · 8–14 Feb · *Submit*

- [ ] Final proofread (fresh eyes, printed, out loud)
- [ ] Format check against venue template — **IEEE TIFS primary** (rolling submission, no deadline gamble)
- [ ] Supplementary packaged
- [ ] **Submit**
- [ ] arXiv preprint the same day

---

## Week 24 · 15–21 Feb · *Slack*

- [ ] Absorbs any slip from Weeks 19–23
- [ ] If unused: prepare the workshop short version (WIFS / CVPRW media forensics)
- [ ] Write the retrospective — what you'd do differently. This becomes the design of the follow‑on video study (Idea 4)

---

# Milestone summary

| # | Milestone | Due | Evidence it happened |
|---|-----------|-----|----------------------|
| M1 | Metrics frozen and tested | Fri 11 Sep | Green test suite |
| **G0** | **Pseudo‑mask viability decided** | **Fri 18 Sep** | `pseudo_mask_validation.json` |
| M2 | Five baselines trained | Fri 25 Sep | Table 1 |
| **G1** | **Shakya collapse reproduced (≈0.50)** | **Fri 2 Oct** | Reproduction table |
| M3 | Degradation sweep complete | Fri 9 Oct | Figure 1 v1 |
| **G2** | **Evidence drift confirmed / demoted** ★ | **Fri 16 Oct** | Gate decision in log |
| M4 | Intro + Sec 3 written | Fri 23 Oct | 4 pages of prose |
| M5 | GRACE‑DF trains stably | Fri 30 Oct | First checkpoint |
| **G3** | **Method viability / fallback decision** | **Fri 6 Nov** | Comparison table |
| M6 | Config frozen | Fri 13 Nov | `grace_df_final.yaml` |
| M7 | 3 seeds complete | Fri 20 Nov | Main table ± std |
| M8 | Ablation + MoA‑DF comparison | Fri 27 Nov | Headline tables |
| **G4** | **All experiments complete; test set frozen** | **Fri 4 Dec** | Results directory |
| M9 | Figures publication‑ready | Fri 18 Dec | `make figures` |
| M10 | Complete draft v1 | Fri 8 Jan | Full PDF |
| **G5** | **Submission‑readiness** | **Fri 15 Jan** | Reviewed draft |
| M11 | Reproducibility verified | Fri 6 Feb | Independent `reproduce.sh` run |
| **M12** | **SUBMITTED** | **Fri 12 Feb** | Submission receipt + arXiv |

---

# GPU budget tracker

| Phase | Weeks | Budgeted | Running total | Available (30/wk) |
|-------|-------|----------|---------------|-------------------|
| I | 1–2 | 7 | 7 | 60 |
| II | 3–4 | 45 | 52 | 120 |
| III | 5–7 | 65 | 117 | 210 |
| IV | 8–11 | 112 | 229 | 330 |
| V | 12–13 | 50 | 279 | 390 |
| VI | 14 | 25 | 304 | 420 |
| VII | 15–20 | 20 | 324 | 600 |
| VIII | 21–24 | 8 | **332** | **720** |

**Danger zone: Weeks 8–12.** Five consecutive weeks at 28 h against a 30 h quota — a single crashed multi‑hour session costs you a week. This is exactly why the resume harness in Week 2 is not optional, and why MoA‑DF logits get cached and never recomputed.

---

# If you fall behind — cut in this order

1. Tier‑B extras: LGrad, UnivFD, CnnSpot *(saves ~1 week)*
2. E8c fairness analysis → replace with per‑generator FPR spread *(saves 3 days)*
3. The two‑agent debate stretch goal — it was never a dependency *(saves 1 week)*
4. Explanation‑method breadth: keep Grad‑CAM + predicted masks, drop Score‑CAM and IG *(saves 4 days)*
5. Seeds: 3 → 2 on ablation rows only, never on headline numbers *(saves 4 days)*

**Never cut:** the metric definitions, the pseudo‑mask validation number, G1's reproduction check, the 3 seeds on headline results, or the statistical tests. Those are what make it a top‑tier paper rather than a preprint.
