"""Tests for configuration merging and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from player_detect.config import build_settings, load_yaml_config


def _base(config: dict, overrides: dict | None = None):
    return build_settings(
        video=Path("videos/a.mp4"),
        out_dir=Path("runs/a"),
        config=config,
        overrides=overrides or {},
    )


def test_defaults_apply_with_empty_config() -> None:
    settings = _base({})

    assert settings.model == "yolo11n.pt"
    assert settings.tile == 1
    assert settings.effective_batch == 8
    assert settings.video == Path("videos/a.mp4")


def test_overrides_win_over_config() -> None:
    settings = _base({"conf": 0.2, "tile": 2}, {"conf": 0.6})

    assert settings.conf == 0.6
    assert settings.tile == 2
    assert settings.effective_batch == 1


def test_none_overrides_are_ignored() -> None:
    settings = _base({"conf": 0.2}, {"conf": None, "imgsz": None})

    assert settings.conf == 0.2


def test_detector_classes_are_normalised_to_a_tuple() -> None:
    settings = _base({"detector_classes": [0, 1, 2]})

    assert settings.detector_classes == (0, 1, 2)


def test_detector_classes_reject_empty_or_negative() -> None:
    with pytest.raises(ValueError, match="detector_classes"):
        _base({"detector_classes": []})
    with pytest.raises(ValueError, match="detector_classes"):
        _base({"detector_classes": [0, -1]})


def test_unknown_config_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        _base({"nope": 1})


def test_invalid_values_raise() -> None:
    with pytest.raises(ValueError, match="conf"):
        _base({"conf": 1.5})
    with pytest.raises(ValueError, match="tile"):
        _base({"tile": 0})
    with pytest.raises(ValueError, match="stride"):
        _base({"stride": 0})
    with pytest.raises(ValueError, match="tile_auto_min_side"):
        _base({"tile_auto_min_side": 320})
    with pytest.raises(ValueError, match="tracker_params"):
        _base({"tracker_params": ["track_high_thresh", 0.1]})


def test_tile_auto_forces_single_frame_batches() -> None:
    settings = _base({"tile_auto": True, "batch": 8})

    assert settings.tile_auto is True
    assert settings.tile_auto_min_side == 1920
    assert settings.effective_batch == 1


def test_tracker_params_are_copied() -> None:
    params = {"track_high_thresh": 0.15}
    settings = _base({"tracker_params": params})
    params["track_high_thresh"] = 0.9

    assert settings.tracker_params == {"track_high_thresh": 0.15}


def test_load_yaml_config_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("conf: 0.4\ntile: 3\n", encoding="utf-8")

    config = load_yaml_config(path)

    assert config == {"conf": 0.4, "tile": 3}


def test_load_yaml_config_missing_file_returns_empty() -> None:
    assert load_yaml_config(Path("does/not/exist.yaml")) == {}
    assert load_yaml_config(None) == {}


def test_default_config_file_is_valid() -> None:
    from player_detect.config import DEFAULT_CONFIG_PATH

    settings = _base(load_yaml_config(DEFAULT_CONFIG_PATH))

    assert settings.model.endswith(".pt")
    assert 0.0 < settings.conf < 1.0
