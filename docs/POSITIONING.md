# Positioning Matrix: Prior Art & Novelty Sweep

This document provides a positioning audit comparing **GRACE-DF** against seven prominent contemporary deepfake and manipulation forensics frameworks:
1. **ManipShield**
2. **FOCA**
3. **PATE-Forensics**
4. **LaP-Forensics**
5. **EditSleuth**
6. **FUSED**
7. **DiffusionPrint**

---

## 1. Feature & Capability Comparison Matrix

| Method | Target Task | Backbones | Measures Evidence Stability ($ESI$)? | Evaluates Explanation Drift Under Degradation? | Causal Counterfactual Faithfulness ($CF$)? | Confidence Calibration ($ECE$ / Brier)? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **ManipShield** | Adversarial defense | ResNet, EfficientNet | ❌ No | ❌ No | ❌ No | ❌ No |
| **FOCA** | Frequency detection | Cross-attention CNN | ❌ No | ❌ No | ❌ No | ❌ No |
| **PATE-Forensics** | Patch forgery | ViT / Transformers | ❌ No | ❌ No | ❌ No | ❌ No |
| **LaP-Forensics** | Language-guided | VLM / CLIP | ❌ No | ❌ No | ❌ No | ❌ No |
| **EditSleuth** | Edit localization | Multi-scale CNN | ❌ No (clean only) | ❌ No | ❌ No | ❌ No |
| **FUSED** | GenAI detection | Dual-branch spatial/freq | ❌ No | ❌ No | ❌ No | ❌ No |
| **DiffusionPrint** | Diffusion provenance | CNN Fingerprint head | ❌ No | ❌ No | ❌ No | ❌ No |
| **GRACE-DF (Ours)** | **Grounded, Robust & Calibrated Forensics** | **CNNs, ViT, Dual-Head** | **✅ Yes (SSIM + Spearman)** | **✅ Yes (25 corruptions + PGD)** | **✅ Yes (Causal swap vs Control)** | **✅ Yes (15-bin ECE + Brier)** |

---

## 2. Detailed Breakdown by Method

### 1. ManipShield (2024)
* **What they do**: Explores defense mechanisms against adversarial perturbations on face forgery detectors using spatial-domain filtering and frequency suppression.
* **Their scope & metric**: Reports top-1 Classification Accuracy and AUC under $L_\infty$ attack budgets.
* **Why GRACE-DF differs**: ManipShield only measures whether the binary classification verdict survives adversarial noise. It does not measure whether the underlying forensic explanation remains grounded in the face manipulation. GRACE-DF discovers that even when defenses maintain accuracy, the explanation map undergoes severe *Evidence Drift*.

### 2. FOCA (Frequency-Oriented Cross-Attention, 2023)
* **What they do**: Combines spatial and frequency-domain representations via cross-attention to detect synthetic artifacts in generative imagery.
* **Their scope & metric**: Evaluates cross-generator generalization on pristine datasets (ProGAN, StyleGAN, Latent Diffusion).
* **Why GRACE-DF differs**: FOCA relies on cross-attention weights as qualitative visual proof, but never quantitatively measures attention stability under realistic transmission degradations (JPEG, blur, noise). GRACE-DF proves that attention maps in frequency-spatial networks are uniquely vulnerable to high-frequency post-processing decay.

### 3. PATE-Forensics (Patch-Aware Transformer, 2023)
* **What they do**: Dissects input images into fine-grained patches using Vision Transformers to identify localized blending boundaries.
* **Their scope & metric**: Reports patch-level detection accuracy and cross-dataset AUC.
* **Why GRACE-DF differs**: PATE-Forensics assumes that transformer self-attention maps inherently represent localized forgery evidence. GRACE-DF employs **Attention Rollout** across all 12 transformer encoder blocks and reveals that transformer saliency rapidly diffuses across non-manipulated tokens under mild Gaussian blur ($\sigma=1.0$) and downscaling.

### 4. LaP-Forensics (Language-Guided Anomaly Perception, 2024)
* **What they do**: Aligns forensic image features with textual prompts via CLIP/VLM embeddings to guide visual attention to suspicious regions.
* **Their scope & metric**: Evaluates zero-shot detection and visual grounding on pristine synthetic benchmarks.
* **Why GRACE-DF differs**: LaP-Forensics presupposes that natural-language rationales and cross-modal attention provide trustworthy interpretability. However, it provides no causal test to verify whether the attended tokens actually drive the decision. GRACE-DF introduces **Counterfactual Faithfulness ($CF$)**, surgically swapping the predicted regions back to pristine pixels and measuring the differential flip rate against an equal-area random control mask.

### 5. EditSleuth (2024)
* **What they do**: Focuses on image inpainting and local edit localization using multi-scale edge refinement networks.
* **Their scope & metric**: Evaluates pixel-level F1, IoU, and AUC solely on uncorrupted, pristine test sets.
* **Why GRACE-DF differs**: EditSleuth only benchmarks localization performance under clean studio conditions. GRACE-DF investigates the degradation dynamics across the full 25-point corruption suite, establishing that detector localization ($GAcc$ and $IoU$) collapses at far lower degradation thresholds than classification accuracy.

### 6. FUSED (Frequency and Spatial Unified Representation, 2024)
* **What they do**: Fuses spatial RGB representations with DCT frequency spectra to detect diffusion-generated artifacts.
* **Their scope & metric**: Benchmarks detection generalization across unseen diffusion architectures (SDv1.5, SDv2, Midjourney, DALL-E 3).
* **Why GRACE-DF differs**: FUSED treats detection purely as a binary classification problem without verifying explanation grounding or confidence calibration. GRACE-DF measures 15-bin Expected Calibration Error ($ECE$) alongside $ESI$, revealing that models become overconfident in drifted explanations under distribution shifts.

### 7. DiffusionPrint (2023)
* **What they do**: Traces the unique generative model fingerprints left in the latents of diffusion processes for model provenance attribution.
* **Their scope & metric**: Multi-class model attribution accuracy (identifying which diffusion generator produced an image).
* **Why GRACE-DF differs**: DiffusionPrint identifies the source generator, but does not provide localized evidence maps for partial edits or evaluate evidence stability under social media transmission chains.

---

## 3. Manuscript Inclusion (Section 2: Related Work)

```latex
\noindent\textbf{Positioning Relative to Contemporary Forensics.}
A growing body of work has tackled deepfake detection through frequency modeling (FOCA~\cite{foca2023}, FUSED~\cite{fused2024}), patch-aware transformers (PATE-Forensics~\cite{pate2023}), adversarial robustness (ManipShield~\cite{manipshield2024}), and multimodal guidance (LaP-Forensics~\cite{lap2024}). 
Crucially, however, all existing benchmarks evaluate either classification accuracy under pristine/attack conditions or static localization on uncorrupted imagery. 
None of these frameworks formulates or measures \emph{explanation stability} ($ESI$) across transmission degradation chains, nor do they verify causal necessity through counterfactual baseline controls ($CF$). 
GRACE-DF provides the first rigorous measurement framework linking explanation drift to confidence miscalibration, resolving the critical vulnerability where detectors remain ``right for the wrong pixels.''
```
