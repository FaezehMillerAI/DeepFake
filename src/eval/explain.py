"""Explanation extraction module for GRACE-DF evidence drift experiments.

Implements:
  1. GradCAM: Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).
     Supports both 4D CNN feature maps and 3D Vision Transformer token maps.
  2. ScoreCAM: Gradient-free perturbation-based activation mapping (Wang et al., 2020).
  3. IntegratedGradients: Axiomatic path-integrated gradients (Sundararajan et al., 2017).
  4. AttentionRollout: Multi-layer attention flow for Vision Transformers (Abnar & Zuidema, 2020).
  5. Unified `explain()` interface producing normalized heatmaps H in [0, 1]^{B x 1 x H x W}.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _min_max_normalize(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize heatmaps per sample to range [0, 1].

    Parameters
    ----------
    maps : torch.Tensor
        Shape [B, 1, H, W] or [B, H, W].
    """
    if maps.ndim == 3:
        maps = maps.unsqueeze(1)
    b = maps.shape[0]
    mins = maps.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
    maxs = maps.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    diff = maxs - mins
    normalized = torch.where(diff > eps, (maps - mins) / (diff + eps), torch.zeros_like(maps))
    return normalized.detach()


class GradCAM:
    """Grad-CAM for CNNs and Vision Transformers."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_last_conv_layer"):
            self.target_layer = model.get_last_conv_layer()
        else:
            raise ValueError(
                "Must provide target_layer or model must implement get_last_conv_layer()."
            )

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

    def _register_hooks(self) -> None:
        self._remove_hooks()
        self.activations = None
        self.gradients = None

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self.activations = out

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(
        self,
        images: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Generate Grad-CAM heatmaps for input batch.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images [B, 3, H, W].
        target_class : int or Tensor, optional
            Target class index. If None, uses the argmax of model logits.

        Returns
        -------
        heatmaps : torch.Tensor
            Normalized attribution heatmaps [B, 1, H, W] in [0, 1].
        """
        self.model.eval()
        self._register_hooks()

        try:
            b, c, h, w = images.shape
            images_req = images.clone().detach().requires_grad_(True)
            logits = self.model(images_req)

            if target_class is None:
                targets = logits.argmax(dim=-1)
            elif isinstance(target_class, int):
                targets = torch.full((b,), target_class, dtype=torch.long, device=images.device)
            else:
                targets = target_class.to(images.device)

            self.model.zero_grad()
            one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).float()
            score = (logits * one_hot).sum()
            score.backward(retain_graph=False)

            assert self.activations is not None and self.gradients is not None

            # Handle 4D CNN activations [B, C, H_a, W_a]
            if self.activations.dim() == 4:
                # Global average pooling of gradients
                weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # [B, 1, H_a, W_a]
            # Handle 3D ViT activations [B, Seq_len, Dim]
            elif self.activations.dim() == 3:
                # Token 0 is CLS, tokens 1..196 are spatial patches
                spatial_acts = self.activations[:, 1:, :]  # [B, 196, D]
                spatial_grads = self.gradients[:, 1:, :]  # [B, 196, D]
                weights = spatial_grads.mean(dim=1, keepdim=True)  # [B, 1, D]
                cam_seq = F.relu((spatial_acts * weights).sum(dim=-1))  # [B, 196]
                grid_sz = int(spatial_acts.shape[1] ** 0.5)
                cam = cam_seq.view(b, 1, grid_sz, grid_sz)
            else:
                raise ValueError(f"Unsupported activation dimension: {self.activations.dim()}")

            # Upsample to original image resolution
            cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
            return _min_max_normalize(cam_upsampled)

        finally:
            self._remove_hooks()


class ScoreCAM:
    """Score-CAM: Gradient-free perturbation-based activation mapping."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        max_channels: int = 32,
    ):
        self.model = model
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_last_conv_layer"):
            self.target_layer = model.get_last_conv_layer()
        else:
            raise ValueError("Must provide target_layer or model must implement get_last_conv_layer().")
        self.max_channels = max_channels

    def generate(
        self,
        images: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Generate Score-CAM heatmaps."""
        self.model.eval()
        b, c, h, w = images.shape

        acts = []
        hook = self.target_layer.register_forward_hook(lambda m, inp, out: acts.append(out))
        with torch.no_grad():
            logits = self.model(images)
        hook.remove()

        act = acts[0]
        if act.dim() == 3:
            # Reshape ViT spatial tokens
            spatial_act = act[:, 1:, :]
            grid_sz = int(spatial_act.shape[1] ** 0.5)
            act = spatial_act.permute(0, 2, 1).view(b, -1, grid_sz, grid_sz)

        if target_class is None:
            targets = logits.argmax(dim=-1)
        elif isinstance(target_class, int):
            targets = torch.full((b,), target_class, dtype=torch.long, device=images.device)
        else:
            targets = target_class.to(images.device)

        heatmaps = []
        for i in range(b):
            img_i = images[i : i + 1]  # [1, 3, H, W]
            act_i = act[i]  # [C, H_a, W_a]
            target_i = targets[i].item()

            num_channels = act_i.shape[0]
            # Select top-K channels by activation energy
            channel_energy = act_i.view(num_channels, -1).sum(dim=1)
            k = min(self.max_channels, num_channels)
            top_k_indices = torch.topk(channel_energy, k=k).indices

            selected_acts = act_i[top_k_indices]  # [K, H_a, W_a]
            # Upsample masks
            masks = F.interpolate(
                selected_acts.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
            )  # [K, 1, H, W]
            masks_norm = _min_max_normalize(masks)  # [K, 1, H, W]

            # Masked images forward
            masked_inputs = img_i * masks_norm  # [K, 3, H, W]
            with torch.no_grad():
                masked_logits = self.model(masked_inputs)
                scores = F.softmax(masked_logits, dim=-1)[:, target_i]  # [K]

            weights = scores.view(k, 1, 1, 1)
            cam = F.relu((weights * masks_norm).sum(dim=0, keepdim=True))  # [1, 1, H, W]
            cam_norm = _min_max_normalize(cam)
            heatmaps.append(cam_norm)

        return torch.cat(heatmaps, dim=0)


class IntegratedGradients:
    """Integrated Gradients attribution (Sundararajan et al., 2017)."""

    def __init__(self, model: nn.Module, steps: int = 20):
        self.model = model
        self.steps = steps

    def generate(
        self,
        images: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate Integrated Gradients attribution map."""
        self.model.eval()
        b, c, h, w = images.shape

        if baseline is None:
            baseline = torch.zeros_like(images)

        with torch.no_grad():
            logits = self.model(images)

        if target_class is None:
            targets = logits.argmax(dim=-1)
        elif isinstance(target_class, int):
            targets = torch.full((b,), target_class, dtype=torch.long, device=images.device)
        else:
            targets = target_class.to(images.device)

        accum_grads = torch.zeros_like(images)

        for step in range(1, self.steps + 1):
            alpha = float(step) / self.steps
            x_step = (baseline + alpha * (images - baseline)).clone().detach().requires_grad_(True)

            out = self.model(x_step)
            one_hot = F.one_hot(targets, num_classes=out.shape[-1]).float()
            score = (out * one_hot).sum()

            self.model.zero_grad()
            score.backward()

            assert x_step.grad is not None
            accum_grads += x_step.grad.detach()

        avg_grads = accum_grads / self.steps
        attr = (images - baseline) * avg_grads  # [B, 3, H, W]
        # Sum of absolute attributions across color channels
        saliency = attr.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        return _min_max_normalize(saliency)


class AttentionRollout:
    """Attention Rollout for Vision Transformers (Abnar & Zuidema, 2020)."""

    def __init__(self, model: nn.Module):
        self.model = model
        # Find underlying ViT module if wrapped in BaselineClassifier
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            self.vit = model.model
        elif hasattr(model, "encoder"):
            self.vit = model
        else:
            raise ValueError("AttentionRollout requires a Vision Transformer model with .encoder.")

    def generate(self, images: torch.Tensor) -> torch.Tensor:
        """Compute recursive attention rollout map from input images."""
        self.model.eval()
        b, c, h, w = images.shape

        with torch.no_grad():
            # 1. Patch projection
            p = self.vit.conv_proj(images)
            p = p.reshape(b, p.shape[1], -1).permute(0, 2, 1)  # [B, Num_patches, D]
            cls_token = self.vit.class_token.expand(b, -1, -1)
            seq = torch.cat([cls_token, p], dim=1)
            seq = seq + self.vit.encoder.pos_embedding
            seq = self.vit.encoder.dropout(seq)

            # 2. Extract attention matrices across all layers
            attns = []
            for layer in self.vit.encoder.layers:
                norm_seq = layer.ln_1(seq)
                _, weights = layer.self_attention(
                    norm_seq, norm_seq, norm_seq, need_weights=True, average_attn_weights=True
                )
                attns.append(weights)  # [B, Seq_len, Seq_len]
                seq = layer(seq)

            seq_len = attns[0].shape[1]
            eye = torch.eye(seq_len, device=images.device).unsqueeze(0)
            rollout = eye.repeat(b, 1, 1)

            # 3. Rollout recursion: R_l = (0.5 A_l + 0.5 I) @ R_{l-1}
            for A in attns:
                A_hat = 0.5 * A + 0.5 * eye
                A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
                rollout = torch.bmm(A_hat, rollout)

            # 4. Extract CLS token attention to image patches (excluding CLS self-attention)
            cls_attn = rollout[:, 0, 1:]  # [B, Num_patches]
            grid_sz = int(cls_attn.shape[1] ** 0.5)
            cam = cls_attn.view(b, 1, grid_sz, grid_sz)

            # 5. Upsample and normalize
            cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
            return _min_max_normalize(cam_upsampled)


def explain(
    model: nn.Module,
    images: torch.Tensor,
    target_class: Optional[Union[int, torch.Tensor]] = None,
    method: str = "gradcam",
    **kwargs: Any,
) -> torch.Tensor:
    """Unified explanation interface.

    Parameters
    ----------
    model : nn.Module
        Classification model.
    images : torch.Tensor
        Batch of images [B, 3, H, W], values in [0, 1].
    target_class : int or Tensor, optional
        Target class to explain. If None, uses predicted class.
    method : str
        One of: 'gradcam', 'scorecam', 'integrated_gradients' / 'ig', 'rollout'.

    Returns
    -------
    attribution : torch.Tensor
        Normalized heatmap [B, 1, H, W] in [0, 1].
    """
    method_clean = method.lower().replace("-", "_")

    if method_clean == "gradcam":
        extractor = GradCAM(model, target_layer=kwargs.get("target_layer"))
        return extractor.generate(images, target_class=target_class)

    elif method_clean == "scorecam":
        max_ch = kwargs.get("max_channels", 32)
        extractor = ScoreCAM(model, target_layer=kwargs.get("target_layer"), max_channels=max_ch)
        return extractor.generate(images, target_class=target_class)

    elif method_clean in ["integrated_gradients", "ig"]:
        steps = kwargs.get("steps", 20)
        extractor = IntegratedGradients(model, steps=steps)
        return extractor.generate(images, target_class=target_class, baseline=kwargs.get("baseline"))

    elif method_clean == "rollout":
        extractor = AttentionRollout(model)
        return extractor.generate(images)

    else:
        raise ValueError(
            f"Unknown explanation method '{method}'. Choose from: 'gradcam', 'scorecam', 'integrated_gradients', 'rollout'."
        )
