"""Projected Gradient Descent, and the evidence-targeted attack (Week 6).

`pgd` is the standard L-inf attack used to reproduce Shakya et al. Exp-3, where
all five baselines collapse to ~0.50 accuracy. If your reproduction does not
show that collapse, the bug is here or in the data split — fix it before Week 5.
"""
from __future__ import annotations


def pgd(model, x, y, eps: float = 4 / 255, alpha: float = 1 / 255,
        steps: int = 10, loss_fn=None, clip: tuple = (0.0, 1.0)):
    """L-inf PGD. `x` is a batch in [0, 1]; returns the perturbed batch.

    Args:
        eps:   perturbation budget (2/255, 4/255, 8/255 in the plan)
        alpha: step size
        steps: iterations
    """
    import torch
    import torch.nn.functional as F

    loss_fn = loss_fn or F.cross_entropy
    x_adv = x.clone().detach()

    # random start inside the eps-ball
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, *clip).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, *clip).detach()

    return x_adv


def fgsm(model, x, y, eps: float = 4 / 255, loss_fn=None, clip: tuple = (0.0, 1.0)):
    """Single-step FGSM."""
    return pgd(model, x, y, eps=eps, alpha=eps, steps=1, loss_fn=loss_fn, clip=clip)


# Aliases
pgd_attack = pgd
fgsm_attack = fgsm


# ---------------------------------------------------------------------------
# Week 6 — the novel attack. Implement when E3 starts.
#
# def evidence_attack(model, explain_fn, x, y, eps, steps):
#     """Maximise 1 - SSIM(M(x), M(x+delta)) subject to the prediction being
#     unchanged. Shows an attacker can corrupt the forensic explanation while
#     leaving the verdict intact — the quotable result for a forensics venue.
#
#     Sketch: differentiable saliency (e.g. input gradients or a differentiable
#     Grad-CAM), loss = -ssim(M_clean, M_adv) + lambda * CE(f(x+delta), y_pred),
#     the second term pinning the label. Requires explain_fn to be differentiable.
#     """
#     raise NotImplementedError
