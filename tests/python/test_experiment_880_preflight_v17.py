"""Tests for Exp 880 — Pre-flight v17: HalluSAE retirement + RETRO audit.

**Why these tests exist:**
    Exp 880 performs two governance-critical writes:
      1. Retiring HalluSAEGeometricProbe in ops/exclusion_manifest.yaml
         (retire_if_same_verdict triggered by Exp 878 below_v1 verdict).
      2. Appending the Milestone 2026.04.68 Pre-flight section to
         MILESTONE_PREREQS.md with 6 open RETROs after HalluSAE retirement.
    These tests verify both writes and the resulting artifact schema.

Spec: REQ-INFRA-072, SCENARIO-INFRA-081
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_880_preflight_v17 import (  # noqa: E402
    _ALL_67_RETROS,
    _RETIRING_THIS_MILESTONE,
    _hallusae_already_retired,
    _retire_hallusae,
    _append_prereqs_68,
    _load_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_manifest(tmp_path: Path) -> Path:
    """Minimal exclusion_manifest.yaml WITHOUT a HalluSAE entry."""
    manifest = tmp_path / "exclusion_manifest.yaml"
    manifest.write_text(
        "retired:\n"
        "  - experiment_id: 260\n"
        "    completed_milestone: '2026.04.37'\n"
        "    reason: 'test entry'\n"
    )
    return manifest


@pytest.fixture()
def tmp_manifest_with_hallusae(tmp_path: Path) -> Path:
    """exclusion_manifest.yaml that already contains the HalluSAE scope tag."""
    manifest = tmp_path / "exclusion_manifest.yaml"
    manifest.write_text(
        "retired:\n"
        "  - experiment_id: 260\n"
        "    completed_milestone: '2026.04.37'\n"
        "    reason: 'test entry'\n"
        "retired_experiments:\n"
        "  - experiment_scope: 'HalluSAEGeometricProbe (python/carnot/verify/hallusae_probe.py)'\n"
        "    reason: 'retire_if_same_verdict triggered by Exp 878'\n"
        "    retired_milestone: '2026.04.67'\n"
        "    retire_if_same_verdict: true\n"
    )
    return manifest


@pytest.fixture()
def tmp_prereqs(tmp_path: Path) -> Path:
    """Minimal MILESTONE_PREREQS.md WITHOUT a .68 section."""
    prereqs = tmp_path / "MILESTONE_PREREQS.md"
    prereqs.write_text("# Milestone Prerequisites\n\nExisting content preserved.\n")
    return prereqs


@pytest.fixture()
def tmp_prereqs_with_68(tmp_path: Path) -> Path:
    """MILESTONE_PREREQS.md that already contains the .68 section."""
    prereqs = tmp_path / "MILESTONE_PREREQS.md"
    prereqs.write_text(
        "# Milestone Prerequisites\n\n## Milestone 2026.04.68 Pre-flight\n\nalready there\n"
    )
    return prereqs


# ---------------------------------------------------------------------------
# Tests: _hallusae_already_retired
# ---------------------------------------------------------------------------


def test_hallusae_not_yet_retired(tmp_manifest: Path) -> None:
    """_hallusae_already_retired returns False when HalluSAE is absent."""
    # REQ-INFRA-072: manifest must be authoritative source for retired scope
    assert _hallusae_already_retired(tmp_manifest) is False


def test_hallusae_already_retired_returns_true(tmp_manifest_with_hallusae: Path) -> None:
    """_hallusae_already_retired returns True when scope tag is present."""
    assert _hallusae_already_retired(tmp_manifest_with_hallusae) is True


# ---------------------------------------------------------------------------
# Tests: _retire_hallusae
# ---------------------------------------------------------------------------


def test_retire_hallusae_writes_entry(tmp_manifest: Path) -> None:
    """_retire_hallusae appends the HalluSAE retirement block to the manifest.

    After the call, the manifest must contain the scope tag and the
    retire_if_same_verdict flag (SCENARIO-INFRA-081).
    """
    result = _retire_hallusae(tmp_manifest)
    assert result is True
    text = tmp_manifest.read_text()
    assert "HalluSAEGeometricProbe" in text
    assert "retire_if_same_verdict" in text
    assert "2026.04.67" in text


def test_retire_hallusae_idempotent(tmp_manifest_with_hallusae: Path) -> None:
    """_retire_hallusae is idempotent — does not duplicate when already present."""
    result = _retire_hallusae(tmp_manifest_with_hallusae)
    assert result is True
    text = tmp_manifest_with_hallusae.read_text()
    # Must appear exactly once.
    assert text.count("HalluSAEGeometricProbe") == 1


# ---------------------------------------------------------------------------
# Tests: exclusion manifest updated with HalluSAE entry (live file)
# ---------------------------------------------------------------------------


def test_live_exclusion_manifest_contains_hallusae() -> None:
    """ops/exclusion_manifest.yaml must contain the HalluSAE scope tag.

    This verifies the actual write performed by Exp 880 on the live repo file.
    REQ-INFRA-072: manifest is the authoritative governance source.
    """
    live_manifest = _REPO_ROOT / "ops" / "exclusion_manifest.yaml"
    assert live_manifest.exists(), "ops/exclusion_manifest.yaml missing"
    text = live_manifest.read_text()
    assert "HalluSAEGeometricProbe" in text, (
        "HalluSAE not found in ops/exclusion_manifest.yaml — "
        "retire_if_same_verdict enforcement failed"
    )
    assert "retire_if_same_verdict: true" in text


# ---------------------------------------------------------------------------
# Tests: _append_prereqs_68
# ---------------------------------------------------------------------------


def test_append_prereqs_68_writes_section(tmp_prereqs: Path) -> None:
    """_append_prereqs_68 appends the .68 section when absent."""
    open_retros = [
        "RETRO-MANIFEST-FULL-SCOPE",
        "RETRO-JEPA-OOD",
        "RETRO-SVAMP-ZERO-AUC",
        "RETRO-XILINX-TOOLS-UNAVAILABLE",
        "RETRO-SOTA-MODEL-DOWNLOAD",
        "RETRO-INERTIA-SWEEPS-TARGET-MISSED",
    ]
    _append_prereqs_68(tmp_prereqs, open_retros)
    text = tmp_prereqs.read_text()
    assert "## Milestone 2026.04.68 Pre-flight" in text
    assert "open_retros_count: 6" in text
    assert "RETRO-MANIFEST-FULL-SCOPE" in text
    # Existing content is preserved (CLAUDE.md: never remove existing content).
    assert "Existing content preserved." in text


def test_append_prereqs_68_idempotent(tmp_prereqs_with_68: Path) -> None:
    """_append_prereqs_68 is idempotent — does not duplicate when already present."""
    _append_prereqs_68(tmp_prereqs_with_68, [])
    text = tmp_prereqs_with_68.read_text()
    assert text.count("## Milestone 2026.04.68 Pre-flight") == 1


# ---------------------------------------------------------------------------
# Tests: MILESTONE_PREREQS.md contains .68 section (live file)
# ---------------------------------------------------------------------------


def test_live_milestone_prereqs_contains_68_section() -> None:
    """MILESTONE_PREREQS.md must contain the .68 pre-flight section.

    Verifies the actual write from Exp 880 on the live repo file.
    SCENARIO-INFRA-081: prereqs doc must be updated before .68 experiments run.
    """
    live_prereqs = _REPO_ROOT / "MILESTONE_PREREQS.md"
    assert live_prereqs.exists(), "MILESTONE_PREREQS.md missing"
    text = live_prereqs.read_text()
    assert "## Milestone 2026.04.68 Pre-flight" in text


# ---------------------------------------------------------------------------
# Tests: open_retros_count == 6 after HalluSAE retirement
# ---------------------------------------------------------------------------


def test_open_retros_count_after_retirement() -> None:
    """open_retros_count must equal 6 after retiring RETRO-HALLUSAE-AUC-BELOW-THRESHOLD.

    The .67 retro reported 7 open RETROs.  HalluSAE retirement reduces this to 6.
    REQ-INFRA-072: RETRO accounting must be accurate at milestone boundary.
    """
    remaining = [r for r in _ALL_67_RETROS if r not in _RETIRING_THIS_MILESTONE]
    assert len(remaining) == 6, (
        f"Expected 6 open RETROs after HalluSAE retirement; got {len(remaining)}: {remaining}"
    )
    assert "RETRO-HALLUSAE-AUC-BELOW-THRESHOLD" not in remaining


# ---------------------------------------------------------------------------
# Tests: result artifact schema
# ---------------------------------------------------------------------------


def test_result_artifact_exists() -> None:
    """results/experiment_880_preflight_v17.json must exist on disk."""
    artifact_path = _REPO_ROOT / "results" / "experiment_880_preflight_v17.json"
    assert artifact_path.exists(), "Exp 880 result artifact not written"


def test_result_artifact_required_fields() -> None:
    """Result artifact must contain all REQUIRED_RESULT_FIELDS."""
    artifact_path = _REPO_ROOT / "results" / "experiment_880_preflight_v17.json"
    artifact = _load_json(artifact_path)
    required = [
        "experiment", "schema", "run_date", "started_at", "finished_at",
        "duration_s", "status", "title",
    ]
    for field in required:
        assert field in artifact, f"Required field '{field}' missing from artifact"


def test_result_artifact_honest_verdict() -> None:
    """honest_verdict must be 'hallusae_retired_governance_ready'."""
    artifact_path = _REPO_ROOT / "results" / "experiment_880_preflight_v17.json"
    artifact = _load_json(artifact_path)
    assert artifact["honest_verdict"] == "hallusae_retired_governance_ready"


def test_result_artifact_open_retros_count() -> None:
    """open_retros_count in artifact must equal 6."""
    artifact_path = _REPO_ROOT / "results" / "experiment_880_preflight_v17.json"
    artifact = _load_json(artifact_path)
    assert artifact["open_retros_count"] == 6


def test_result_artifact_hallusae_retired_true() -> None:
    """hallusae_retired must be True in the artifact."""
    artifact_path = _REPO_ROOT / "results" / "experiment_880_preflight_v17.json"
    artifact = _load_json(artifact_path)
    assert artifact["hallusae_retired"] is True


def test_result_artifact_prereqs_updated_true() -> None:
    """prereqs_updated must be True in the artifact."""
    artifact_path = _REPO_ROOT / "results" / "experiment_880_preflight_v17.json"
    artifact = _load_json(artifact_path)
    assert artifact["prereqs_updated"] is True
