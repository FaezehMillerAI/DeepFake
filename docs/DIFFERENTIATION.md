# Novelty Threats & Differentiation: INP-X and GenShield

This document establishes the precise conceptual and empirical boundaries separating **GRACE-DF** from its two closest contemporary novelty threats: **INP-X** (Nebioglu, Bilgiç, & Popescu, arXiv:2602.00192) and **GenShield** (Xu et al., ICML 2026, arXiv:2605.16122).

---

## 1. Differentiation from INP-X (Nebioglu et al., 2026)

### Summary of INP-X
Nebioglu, Bilgiç, and Popescu introduce *Inpainting Exchange* (INP-X), an empirical diagnostic tool that surgically swaps the unmanipulated background of an inpainted image with the pristine source background while keeping the AI-synthesized foreground intact. Their core empirical finding is that existing deepfake detectors suffer from a **static spatial bias**: they overrely on global spectral artifacts (specifically high-frequency VAE reconstruction shifts) distributed uniformly across background pixels, causing detection accuracy to drop from ~91% to near chance (~55%) when the background is restored.

### GRACE-DF Positioning & Distinction
> **Core Difference**: *INP-X demonstrates that evidence starts in the wrong place; GRACE-DF discovers and proves that evidence moves under post-processing (Evidence Drift).*

While INP-X exposes a static spatial artifact reliance on pristine images, it does not investigate, measure, or model how explanations behave under downstream distribution shifts. In contrast, GRACE-DF focuses on the dynamic phenomenon of **Evidence Drift**:
1. **Dynamic Instability Under Transmission**: Even when a detector correctly grounds its prediction on true manipulated pixels in pristine conditions ($GAcc_{clean} > 0.80$), common post-processing operations (e.g., JPEG compression $q=50$, Gaussian blur $\sigma=1.0$, sensor noise) cause the explanatory saliency map to rapidly decorrelate from the manipulation footprint ($ESI$ drops by $>40\%$), while the scalar classification confidence paradoxically remains high. The detector appears robust by top-1 accuracy, but is "right for the wrong pixels."
2. **Quantitative Drift Formulation**: We formalize the first mathematical metric suite for explanation dynamics under degradation: Evidence Stability Index ($ESI$, with SSIM and Spearman rank variants), Grounded Accuracy ($GAcc$), and Counterfactual Faithfulness ($CF$).
3. **Causal Verification**: Beyond static pixel substitution, our Counterfactual Faithfulness test rigorously verifies whether the identified evidence regions are *causally necessary and sufficient* to alter an independent detector's classification.
4. **Mitigation**: GRACE-DF provides an active defense (evidence-grounded calibration loss) that pins explanations to the genuine forensic manipulation mask, directly preventing evidence drift.

---

## 2. Differentiation from GenShield (Xu et al., ICML 2026)

### Summary of GenShield
Xu et al. propose *GenShield*, a unified framework for AI-generated image detection and visual artifact correction presented at ICML 2026. GenShield utilizes an autoregressive Mixture-of-Transformers (MoT) architecture combining a detection expert and an artifact correction expert over a shared multimodal vision-language backbone, guided by Visual Chain-of-Thought (VCoT) curriculum learning to iteratively diagnose and self-repair generation defects.

### GRACE-DF Positioning & Distinction
> **Core Difference**: *GenShield is a generative repair system; GRACE-DF is an explanation-stability and forensic grounding framework.*

1. **Orthogonal Scientific Objectives**: GenShield's objective is generative restoration—repairing synthesis flaws so that images look natural. GRACE-DF's objective is forensic trustworthiness and evidentiary verification—ensuring that legal, regulatory, and investigative stakeholders can verify *why* an image was flagged, and that the spatial attribution does not hallucinate under compression.
2. **Assumption of Attribution Integrity**: GenShield assumes that attention maps accurately reflect synthesis artifacts and feeds them directly into autoregressive inpainting blocks. GRACE-DF falsifies this assumption, proving that attention maps in deep networks are fragile and drift under the slightest post-processing.
3. **Architectural Footprint and Computational Feasibility**: GenShield relies on multi-billion parameter autoregressive multimodal transformers and complex diffusion loops, rendering real-time deployment and edge verification intractable. GRACE-DF develops a lightweight, dual-head calibrated architecture that trains and runs efficiently within accessible academic hardware constraints (e.g., Kaggle P100 / dual T4 GPUs).
4. **Calibration and Evidence Metrics**: GenShield does not provide confidence calibration (ECE, Brier score) or formal explanation stability metrics. GRACE-DF bridges classification confidence calibration with spatial attribution stability.

---

## Ready-to-Use LaTeX Prose for Section 2 (Related Work)

```latex
\noindent\textbf{Relation to INP-X.}
Recently, Nebioglu et al.~\cite{nebioglu2026inpx} introduced Inpainting Exchange (INP-X), revealing that deepfake detectors often depend on global VAE reconstruction artifacts in unmanipulated backgrounds rather than localized synthetic content. 
While INP-X established that detector evidence can originate in the wrong location under pristine conditions, GRACE-DF investigates a distinct, dynamic failure mode: \emph{Evidence Drift}. 
We demonstrate that even when detectors achieve high grounded accuracy on clean benchmarks, standard digital transmission pipelines (such as JPEG recompression, resampling, and spatial filtering) cause explanation heatmaps to rapidly collapse and migrate away from manipulated boundaries, even as classification accuracy remains misleadingly unaffected. 
Furthermore, while INP-X serves as an empirical diagnostic benchmark, GRACE-DF provides both quantitative drift metrics ($ESI$, $CF$) and a calibrated training objective that explicitly anchors explanation fidelity across degradation regimes.

\noindent\textbf{Relation to GenShield.}
Xu et al.~\cite{xugenshield2026} proposed GenShield, combining deepfake detection with iterative generative artifact correction via an autoregressive Mixture-of-Transformers. 
Whereas GenShield focuses on generative visual restoration, GRACE-DF addresses the forensic trustworthiness and evidentiary stability of detection systems. 
GenShield presupposes that internal model attention reliably correlates with generative anomalies; in contrast, our work demonstrates the empirical fragility of this assumption under adversarial perturbations and natural corruptions, proposing lightweight, calibrated dual-head networks that ensure verifiable attribution without demanding heavy generative inpainting loops.
```
