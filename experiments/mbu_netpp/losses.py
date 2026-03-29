from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probs * target, dim=(1, 2, 3)) + eps
    denominator = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3)) + eps
    return (1.0 - numerator / denominator).mean()


def vf_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_vf = probs.mean(dim=(1, 2, 3))
    gt_vf = target.mean(dim=(1, 2, 3))
    return torch.mean((pred_vf - gt_vf) ** 2)


class MBULoss(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.use_edge_loss = bool(config.get("use_edge_loss", False))
        self.use_vf_loss = bool(config.get("use_vf_loss", False))
        self.edge_weight = float(config.get("edge_weight", 0.3))
        self.deep_weight = float(config.get("deep_weight", 0.2))
        self.vf_weight = float(config.get("vf_weight", 0.1))
        self.bce = nn.BCEWithLogitsLoss()

    def seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss_from_logits(logits, target) + self.bce(logits, target)

    def forward(self, outputs: dict[str, Any], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target_mask = batch["mask"]
        total_loss = self.seg_loss(outputs["seg_logits"], target_mask)
        components: dict[str, float] = {"seg": float(total_loss.detach().cpu())}

        if self.use_edge_loss and outputs.get("edge_logits") is not None:
            edge_term = self.bce(outputs["edge_logits"], batch["edge"])
            total_loss = total_loss + self.edge_weight * edge_term
            components["edge"] = float(edge_term.detach().cpu())

        deep_logits = outputs.get("deep_logits") or []
        if deep_logits:
            deep_term = torch.stack([self.seg_loss(logits, target_mask) for logits in deep_logits]).mean()
            total_loss = total_loss + self.deep_weight * deep_term
            components["deep"] = float(deep_term.detach().cpu())

        if self.use_vf_loss:
            vf_term = vf_loss_from_logits(outputs["seg_logits"], target_mask)
            total_loss = total_loss + self.vf_weight * vf_term
            components["vf"] = float(vf_term.detach().cpu())

        components["total"] = float(total_loss.detach().cpu())
        return total_loss, components
