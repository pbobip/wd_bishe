from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_build_model_supports_deeplabv3plus() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "mbu_netpp"
        / "configs"
        / "real53_cv5_deeplabv3plus_imagenet.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"]["name"] == "deeplabv3plus"
    assert config["model"]["encoder_name"] == "se_resnext50_32x4d"
    assert config["model"]["encoder_weights_mode"] == "imagenet"
    assert config["model"]["decoder_channels_head"] == 256


def test_real53_cv5_configs_share_same_prepared_dataset() -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "experiments" / "mbu_netpp" / "configs"
    expected_files = [
        "real53_cv5_unet_gpu.yaml",
        "real53_cv5_unetpp_nopretrain.yaml",
        "real53_cv5_unetpp_imagenet.yaml",
        "real53_cv5_unetpp_micronet.yaml",
        "real53_cv5_mbu_edge.yaml",
        "real53_cv5_mbu_edge_deep.yaml",
        "real53_cv5_mbu_edge_deep_vf.yaml",
        "real53_cv5_deeplabv3plus_imagenet.yaml",
        "real53_cv5_deeplabv3plus_micronet.yaml",
    ]

    for file_name in expected_files:
        config = yaml.safe_load((config_dir / file_name).read_text(encoding="utf-8"))
        assert config["data"]["prepared_root"] == "experiments/mbu_netpp/workdir/prepared_real53_cv5"
        assert config["data"]["num_folds"] == 5
        assert config["data"]["fold_manifest_name"] == "folds_5_seed42.json"
        assert config["training"]["epochs"] == 80
        assert config["training"]["batch_size"] == 6
