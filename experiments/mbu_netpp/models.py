from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SegmentationOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.encoder = getattr(model, "encoder", None)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        return {
            "seg_logits": self.model(x),
            "edge_logits": None,
            "deep_logits": [],
        }


class UnetPlusPlusExperimentModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weights: str | None,
        in_channels: int,
        classes: int,
        decoder_channels: list[int],
        use_edge_head: bool,
        use_deep_supervision: bool,
    ) -> None:
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=tuple(decoder_channels),
            activation=None,
        )
        self.encoder = self.backbone.encoder
        self.use_edge_head = use_edge_head
        self.use_deep_supervision = use_deep_supervision

        final_channels = int(self.backbone.decoder.out_channels[-1])
        if use_edge_head:
            hidden_channels = max(16, final_channels)
            self.edge_head = nn.Sequential(
                nn.Conv2d(final_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            self.edge_head = None

        if use_deep_supervision:
            aux_channels = list(self.backbone.decoder.out_channels[1:4])
            aux_keys = ["x_0_1", "x_0_2", "x_0_3"]
            aux_upsampling = [8, 4, 2]
            self.deep_keys = aux_keys
            self.deep_heads = nn.ModuleDict(
                {
                    key: smp.base.SegmentationHead(
                        in_channels=channels,
                        out_channels=1,
                        kernel_size=3,
                        activation=None,
                        upsampling=upsampling,
                    )
                    for key, channels, upsampling in zip(aux_keys, aux_channels, aux_upsampling, strict=True)
                }
            )
        else:
            self.deep_keys = []
            self.deep_heads = nn.ModuleDict()

    def _decode_with_dense_connections(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        decoder = self.backbone.decoder
        decoder_features = features[1:][::-1]
        dense_x: dict[str, torch.Tensor] = {}
        for layer_idx in range(len(decoder.in_channels) - 1):
            for depth_idx in range(decoder.depth - layer_idx):
                if layer_idx == 0:
                    output = decoder.blocks[f"x_{depth_idx}_{depth_idx}"](
                        decoder_features[depth_idx],
                        decoder_features[depth_idx + 1],
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_level = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_level}"]
                        for idx in range(depth_idx + 1, dense_level + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [decoder_features[dense_level + 1]],
                        dim=1,
                    )
                    dense_x[f"x_{depth_idx}_{dense_level}"] = decoder.blocks[f"x_{depth_idx}_{dense_level}"](
                        dense_x[f"x_{depth_idx}_{dense_level - 1}"],
                        cat_features,
                    )
        dense_x[f"x_0_{decoder.depth}"] = decoder.blocks[f"x_0_{decoder.depth}"](dense_x[f"x_0_{decoder.depth - 1}"])
        return dense_x[f"x_0_{decoder.depth}"], dense_x

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        features = self.backbone.encoder(x)
        decoder_output, dense_features = self._decode_with_dense_connections(features)
        seg_logits = self.backbone.segmentation_head(decoder_output)

        edge_logits = self.edge_head(decoder_output) if self.edge_head is not None else None
        deep_logits = [self.deep_heads[key](dense_features[key]) for key in self.deep_keys]
        return {
            "seg_logits": seg_logits,
            "edge_logits": edge_logits,
            "deep_logits": deep_logits,
        }


def _normalize_checkpoint_key(key: str) -> list[str]:
    stripped = key
    for prefix in ("state_dict.", "model.", "module.", "encoder.", "backbone."):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
    candidates = {stripped}
    if stripped.startswith("conv1."):
        candidates.add("layer0." + stripped)
    if stripped.startswith("bn1."):
        candidates.add("layer0." + stripped)
    if ".se." in stripped:
        candidates.add(stripped.replace(".se.", ".se_module."))
    if ".se_module." in stripped:
        candidates.add(stripped.replace(".se_module.", ".se."))
    if stripped.startswith("layer0.conv1."):
        candidates.add(stripped[len("layer0.") :])
    if stripped.startswith("layer0.bn1."):
        candidates.add(stripped[len("layer0.") :])
    expanded = set()
    for candidate in candidates:
        expanded.add(candidate)
        if ".se." in candidate:
            expanded.add(candidate.replace(".se.", ".se_module."))
        if ".se_module." in candidate:
            expanded.add(candidate.replace(".se_module.", ".se."))
    return list(expanded)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("encoder_state_dict", "state_dict", "model_state_dict", "model", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise ValueError("无法从 checkpoint 中提取 state_dict")


def _adapt_tensor_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.shape == target.shape:
        return source
    if source.ndim == 4 and target.ndim == 4 and source.shape[0] == target.shape[0] and source.shape[2:] == target.shape[2:]:
        if source.shape[1] == 3 and target.shape[1] == 1:
            return source.mean(dim=1, keepdim=True)
        if source.shape[1] == 1 and target.shape[1] == 3:
            return source.repeat(1, 3, 1, 1) / 3.0
    return None


def load_microscopy_pretrained_encoder(encoder: nn.Module, checkpoint_path: str | Path) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = _extract_state_dict(checkpoint)
    target_state = encoder.state_dict()

    matched = 0
    loaded_state = {}
    for raw_key, raw_value in source_state.items():
        if not isinstance(raw_value, torch.Tensor):
            continue
        for candidate in _normalize_checkpoint_key(raw_key):
            if candidate not in target_state:
                continue
            adapted = _adapt_tensor_shape(raw_value, target_state[candidate])
            if adapted is None:
                continue
            loaded_state[candidate] = adapted
            matched += 1
            break

    missing = len(target_state) - len(loaded_state)
    encoder.load_state_dict(loaded_state, strict=False)
    return {"matched": matched, "missing": missing, "total": len(target_state)}


def resolve_encoder_weights(mode: str) -> str | None:
    normalized = str(mode or "none").strip().lower()
    if normalized == "imagenet":
        return "imagenet"
    return None


def build_model(model_config: dict[str, Any]) -> nn.Module:
    name = str(model_config["name"]).strip().lower()
    encoder_name = str(model_config.get("encoder_name", "se_resnext50_32x4d"))
    encoder_weights_mode = str(model_config.get("encoder_weights_mode", "none"))
    encoder_weights = resolve_encoder_weights(encoder_weights_mode)
    in_channels = int(model_config.get("in_channels", 1))
    classes = int(model_config.get("classes", 1))
    decoder_channels = [int(value) for value in model_config.get("decoder_channels", [256, 128, 64, 32, 16])]

    if name == "unet":
        model = SegmentationOnlyWrapper(
            smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                decoder_channels=tuple(decoder_channels),
                activation=None,
            )
        )
    elif name in {"unetpp", "micronet_unetpp", "mbu_netpp"}:
        use_edge_head = bool(model_config.get("use_edge_head", False)) if name == "mbu_netpp" else False
        use_deep_supervision = bool(model_config.get("use_deep_supervision", False)) if name == "mbu_netpp" else False
        model = UnetPlusPlusExperimentModel(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=decoder_channels,
            use_edge_head=use_edge_head,
            use_deep_supervision=use_deep_supervision,
        )
    else:
        raise ValueError(f"不支持的模型名称: {name}")

    if encoder_weights_mode.strip().lower() == "micronet":
        checkpoint_path = str(model_config.get("micronet_checkpoint", "")).strip()
        if not checkpoint_path:
            raise ValueError("当前配置要求使用 MicroNet 预训练，但 micronet_checkpoint 为空")
        info = load_microscopy_pretrained_encoder(model.encoder, checkpoint_path)
        print(f"MicroNet 预训练 encoder 加载完成: matched={info['matched']}, missing={info['missing']}, total={info['total']}")

    return model


def build_model_for_checkpoint(model_config: dict[str, Any]) -> nn.Module:
    cfg = dict(model_config)
    cfg["encoder_weights_mode"] = "none"
    cfg["micronet_checkpoint"] = ""
    return build_model(cfg)


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int,
    overlap: float,
) -> torch.Tensor:
    if image_tensor.shape[0] != 1:
        raise ValueError("当前滑窗推理只支持 batch size = 1")

    _, _, height, width = image_tensor.shape
    stride = max(1, int(patch_size * (1.0 - overlap)))
    pad_bottom = max(0, patch_size - height)
    pad_right = max(0, patch_size - width)
    padded = nn.functional.pad(image_tensor, (0, pad_right, 0, pad_bottom), mode="reflect")
    padded_height, padded_width = padded.shape[-2:]

    y_positions = list(range(0, max(1, padded_height - patch_size + 1), stride))
    x_positions = list(range(0, max(1, padded_width - patch_size + 1), stride))
    if y_positions[-1] != padded_height - patch_size:
        y_positions.append(padded_height - patch_size)
    if x_positions[-1] != padded_width - patch_size:
        x_positions.append(padded_width - patch_size)

    device = padded.device
    logits_sum = torch.zeros((1, 1, padded_height, padded_width), device=device)
    logits_count = torch.zeros((1, 1, padded_height, padded_width), device=device)
    for top in y_positions:
        for left in x_positions:
            patch = padded[:, :, top : top + patch_size, left : left + patch_size]
            outputs = model(patch)
            patch_logits = outputs["seg_logits"]
            logits_sum[:, :, top : top + patch_size, left : left + patch_size] += patch_logits
            logits_count[:, :, top : top + patch_size, left : left + patch_size] += 1.0

    logits = logits_sum / torch.clamp_min(logits_count, 1.0)
    return logits[:, :, :height, :width]
