from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def reduce_weighted_loss(losses: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return losses.mean()
    weights = sample_weight.view(-1).to(losses.device, dtype=losses.dtype)
    total_weight = torch.clamp_min(torch.sum(weights), 1e-6)
    return torch.sum(losses * weights) / total_weight


def dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probs * target, dim=(1, 2, 3)) + eps
    denominator = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps
    per_sample = 1.0 - numerator / denominator
    if reduction == "none":
        return per_sample
    return reduce_weighted_loss(per_sample, sample_weight)


def vf_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_vf = probs.mean(dim=(1, 2, 3))
    gt_vf = target.mean(dim=(1, 2, 3))
    per_sample = (pred_vf - gt_vf) ** 2
    if reduction == "none":
        return per_sample
    return reduce_weighted_loss(per_sample, sample_weight)


class MBULoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.use_edge_loss = bool(config.get("use_edge_loss", False))
        self.use_vf_loss = bool(config.get("use_vf_loss", False))
        self.edge_weight = float(config.get("edge_weight", 0.3))
        self.deep_weight = float(config.get("deep_weight", 0.2))
        self.vf_weight = float(config.get("vf_weight", 0.1))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def bce_loss(self, logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
        per_pixel = self.bce(logits, target)
        per_sample = per_pixel.mean(dim=(1, 2, 3))
        return reduce_weighted_loss(per_sample, sample_weight)

    def seg_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        dice_term = dice_loss_from_logits(logits, target, sample_weight=sample_weight)
        bce_term = self.bce_loss(logits, target, sample_weight=sample_weight)
        return dice_term + bce_term

    def forward(self, outputs: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target_mask = batch["mask"]
        sample_weight = batch.get("sample_weight")
        total_loss = self.seg_loss(outputs["seg_logits"], target_mask, sample_weight=sample_weight)
        components: dict[str, float] = {"seg": float(total_loss.detach().cpu())}

        if self.use_edge_loss and outputs.get("edge_logits") is not None:
            edge_term = self.bce_loss(outputs["edge_logits"], batch["edge"], sample_weight=sample_weight)
            total_loss = total_loss + self.edge_weight * edge_term
            components["edge"] = float(edge_term.detach().cpu())

        deep_logits = outputs.get("deep_logits") or []
        if deep_logits:
            deep_term = torch.stack(
                [self.seg_loss(logits, target_mask, sample_weight=sample_weight) for logits in deep_logits]
            ).mean()
            total_loss = total_loss + self.deep_weight * deep_term
            components["deep"] = float(deep_term.detach().cpu())

        if self.use_vf_loss:
            vf_term = vf_loss_from_logits(outputs["seg_logits"], target_mask, sample_weight=sample_weight)
            total_loss = total_loss + self.vf_weight * vf_term
            components["vf"] = float(vf_term.detach().cpu())

        components["total"] = float(total_loss.detach().cpu())
        return total_loss, components
