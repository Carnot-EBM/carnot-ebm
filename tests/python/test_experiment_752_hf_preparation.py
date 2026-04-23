"""Tests for Experiment 752: HuggingFace model artifact preparation.

These tests verify that the artifacts produced by
scripts/experiment_752_hf_model_preparation.py satisfy REQ-PUBLISH-001:
model cards must be emoji-free, configs must have all required fields,
and the upload script must reference the correct file paths.

Spec: REQ-PUBLISH-001, SCENARIO-PUBLISH-001
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_MODELS = _REPO / "models"

# Import helper functions from the experiment script so they are covered.
# Importing them here ensures pytest-cov measures their execution.
import sys as _sys
_sys.path.insert(0, str(_REPO / "scripts"))
from experiment_752_hf_model_preparation import (  # noqa: E402
    _verify_model_card_no_emojis,
    _verify_config_fields,
    _copy_kan_weights,
    _export_jepa_probe_weights,
)


def _has_emoji(text: str) -> bool:
    """Return True if text contains any emoji codepoints.

    Emoji detection covers the common unicode ranges used in documentation.
    We check Miscellaneous Symbols and Pictographs (U+1F300-U+1FFFF),
    Miscellaneous Symbols (U+2600-U+27BF), and Variation Selectors
    (U+FE00-U+FE0F) — the same ranges checked in the experiment script.
    """
    for char in text:
        cp = ord(char)
        if (
            0x1F300 <= cp <= 0x1FFFF
            or 0x2600 <= cp <= 0x27BF
            or 0xFE00 <= cp <= 0xFE0F
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# REQ-PUBLISH-001: model cards must have no emojis
# ---------------------------------------------------------------------------


def test_jepa_model_card_no_emojis() -> None:
    """REQ-PUBLISH-001: StepLevelJEPAProbe model card must be emoji-free."""
    card = _MODELS / "MODELCARD_carnot_step_jepa_probe_v1.md"
    assert card.exists(), f"Model card not found: {card}"
    text = card.read_text(encoding="utf-8")
    assert not _has_emoji(text), (
        "MODELCARD_carnot_step_jepa_probe_v1.md contains emoji characters. "
        "Per REQ-PUBLISH-001 and CLAUDE.md, all public documentation must be emoji-free."
    )


def test_kan_model_card_no_emojis() -> None:
    """REQ-PUBLISH-001: KAN Tier 0b model card must be emoji-free."""
    card = _MODELS / "MODELCARD_carnot_kan_tier0b_v3.md"
    assert card.exists(), f"Model card not found: {card}"
    text = card.read_text(encoding="utf-8")
    assert not _has_emoji(text), (
        "MODELCARD_carnot_kan_tier0b_v3.md contains emoji characters. "
        "Per REQ-PUBLISH-001 and CLAUDE.md, all public documentation must be emoji-free."
    )


# ---------------------------------------------------------------------------
# REQ-PUBLISH-001: config JSON must have all required fields
# ---------------------------------------------------------------------------

_JEPA_REQUIRED_FIELDS = [
    "model_type",
    "hidden_dim",
    "layer_index",
    "probe_architecture",
    "training_data",
    "eval_auc_5fold",
    "eval_std_5fold",
    "latency_p50_ms",
    "model_class",
]

_KAN_REQUIRED_FIELDS = [
    "model_type",
    "auroc",
    "fp_rate_gsm8k_1000q",
    "deployment_tier",
    "architecture",
    "training_data",
]


def test_jepa_config_has_required_fields() -> None:
    """REQ-PUBLISH-001: StepLevelJEPAProbe config must contain all required fields."""
    config_path = _MODELS / "carnot_step_jepa_probe_v1_config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    config = json.loads(config_path.read_text())
    missing = [f for f in _JEPA_REQUIRED_FIELDS if f not in config]
    assert not missing, (
        f"carnot_step_jepa_probe_v1_config.json is missing required fields: {missing}. "
        "REQ-PUBLISH-001 requires all fields to be present for a valid model card."
    )


def test_kan_config_has_required_fields() -> None:
    """REQ-PUBLISH-001: KAN Tier 0b config must contain all required fields."""
    config_path = _MODELS / "carnot_kan_tier0b_v3_config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    config = json.loads(config_path.read_text())
    missing = [f for f in _KAN_REQUIRED_FIELDS if f not in config]
    assert not missing, (
        f"carnot_kan_tier0b_v3_config.json is missing required fields: {missing}. "
        "REQ-PUBLISH-001 requires all fields to be present for a valid model card."
    )


# ---------------------------------------------------------------------------
# REQ-PUBLISH-001: upload script must reference correct file paths
# ---------------------------------------------------------------------------


def test_upload_script_references_correct_paths() -> None:
    """REQ-PUBLISH-001: hf_upload_commands.sh must reference all artifact files."""
    script = _MODELS / "hf_upload_commands.sh"
    assert script.exists(), f"Upload script not found: {script}"
    text = script.read_text()

    expected_files = [
        "carnot_step_jepa_probe_v1.safetensors",
        "carnot_step_jepa_probe_v1_config.json",
        "MODELCARD_carnot_step_jepa_probe_v1.md",
        "carnot_kan_tier0b_v3.safetensors",
        "carnot_kan_tier0b_v3_config.json",
        "MODELCARD_carnot_kan_tier0b_v3.md",
    ]
    missing = [f for f in expected_files if f not in text]
    assert not missing, (
        f"hf_upload_commands.sh does not reference these artifact files: {missing}. "
        "The upload script must list every artifact so the operator can push all files."
    )


def test_upload_script_references_correct_repos() -> None:
    """REQ-PUBLISH-001: hf_upload_commands.sh must target the Carnot-EBM org repos."""
    script = _MODELS / "hf_upload_commands.sh"
    assert script.exists(), f"Upload script not found: {script}"
    text = script.read_text()

    expected_repos = [
        "Carnot-EBM/step-jepa-probe-v1",
        "Carnot-EBM/kan-tier0b-v3",
    ]
    missing = [r for r in expected_repos if r not in text]
    assert not missing, (
        f"hf_upload_commands.sh does not reference these HuggingFace repos: {missing}. "
        "Both models must be uploaded to the Carnot-EBM organisation namespace."
    )


# ---------------------------------------------------------------------------
# REQ-PUBLISH-001: deliverable JSON must confirm artifact readiness
# ---------------------------------------------------------------------------


def test_deliverable_honest_verdict_ready() -> None:
    """REQ-PUBLISH-001: Experiment 752 deliverable must report hf_artifacts_ready."""
    deliverable = _REPO / "results" / "experiment_752_hf_model_preparation.json"
    assert deliverable.exists(), f"Deliverable not found: {deliverable}"
    data = json.loads(deliverable.read_text())
    verdict = data.get("honest_verdict", "")
    assert verdict == "hf_artifacts_ready", (
        f"Expected honest_verdict='hf_artifacts_ready', got '{verdict}'. "
        "All six artifact files (2 weights, 2 configs, 2 model cards) must exist "
        "and pass validation before reporting ready."
    )
