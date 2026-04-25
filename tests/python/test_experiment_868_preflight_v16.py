"""Tests for Exp 868 pre-flight v16 — ExclusionManifestEnforcer.

Traces to:
  REQ-INFRA-072: ExclusionManifestEnforcer must prevent retired experiments from
    re-entering the queue by writing gate entries to MILESTONE_PREREQS.md.
  SCENARIO-INFRA-081: Given ops/exclusion_manifest.yaml with retired IDs,
    write_prereqs_section() must append "## Exclusion Manifest Gate" section.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from carnot.pipeline.manifest_enforcer import ExclusionManifestEnforcer, _parse_yaml_fallback


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_YAML = textwrap.dedent("""\
    retired:
      - experiment_id: 527
        completed_milestone: "2026.04.57"
        reason: "RETRO-033 resolved by Exp 720; no remaining research mandate"
      - experiment_id: 491
        completed_milestone: "2026.04.58"
        reason: "JEPA curriculum diagnostic, findings incorporated"
      - experiment_id: 99999
        completed_milestone: "2026.04.99"
        reason: "hypothetical test entry"
""")


@pytest.fixture
def manifest_yaml(tmp_path: Path) -> str:
    """Write sample YAML to a temp file and return its path."""
    p = tmp_path / "exclusion_manifest.yaml"
    p.write_text(SAMPLE_YAML)
    return str(p)


@pytest.fixture
def prereqs_file(tmp_path: Path) -> str:
    """Create an empty prereqs file and return its path."""
    p = tmp_path / "MILESTONE_PREREQS.md"
    p.write_text("# Prereqs\n\nExisting content must be preserved.\n")
    return str(p)


# ---------------------------------------------------------------------------
# Test: load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    """REQ-INFRA-072: load_manifest() reads YAML and returns id->reason map."""

    def test_loads_integer_ids(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        result = enforcer.load_manifest(manifest_yaml)
        assert 527 in result
        assert 491 in result
        assert 99999 in result

    def test_correct_reasons(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        result = enforcer.load_manifest(manifest_yaml)
        assert "RETRO-033" in result[527]
        assert "JEPA" in result[491]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        enforcer = ExclusionManifestEnforcer()
        result = enforcer.load_manifest(str(tmp_path / "nonexistent.yaml"))
        assert result == {}

    def test_returns_dict(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        result = enforcer.load_manifest(manifest_yaml)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: is_retired
# ---------------------------------------------------------------------------


class TestIsRetired:
    """REQ-INFRA-072: is_retired() returns True for retired IDs only."""

    def test_retired_id_returns_true(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        assert enforcer.is_retired(527) is True

    def test_unknown_id_returns_false(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        assert enforcer.is_retired(9999) is False

    def test_not_loaded_returns_false(self) -> None:
        enforcer = ExclusionManifestEnforcer()
        # No load_manifest called — safe default.
        assert enforcer.is_retired(527) is False


# ---------------------------------------------------------------------------
# Test: get_retirement_reason
# ---------------------------------------------------------------------------


class TestGetRetirementReason:
    """REQ-INFRA-072: get_retirement_reason() returns reason or empty string."""

    def test_known_id_returns_reason(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        reason = enforcer.get_retirement_reason(491)
        assert "JEPA" in reason

    def test_unknown_id_returns_empty(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        assert enforcer.get_retirement_reason(9999) == ""


# ---------------------------------------------------------------------------
# Test: write_prereqs_section
# ---------------------------------------------------------------------------


class TestWritePrereqsSection:
    """SCENARIO-INFRA-081: write_prereqs_section appends Exclusion Manifest Gate."""

    def test_appends_gate_section(self, manifest_yaml: str, prereqs_file: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        enforcer.write_prereqs_section(prereqs_file)
        content = Path(prereqs_file).read_text()
        assert "## Exclusion Manifest Gate" in content

    def test_preserves_existing_content(self, manifest_yaml: str, prereqs_file: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        enforcer.write_prereqs_section(prereqs_file)
        content = Path(prereqs_file).read_text()
        assert "Existing content must be preserved." in content

    def test_lists_retired_ids(self, manifest_yaml: str, prereqs_file: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        enforcer.write_prereqs_section(prereqs_file)
        content = Path(prereqs_file).read_text()
        assert "527" in content
        assert "491" in content

    def test_manifest_enforcer_deployed_flag(self, manifest_yaml: str, prereqs_file: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        enforcer.write_prereqs_section(prereqs_file)
        content = Path(prereqs_file).read_text()
        assert "manifest_enforcer_deployed: true" in content

    def test_multiple_writes_append_not_replace(self, manifest_yaml: str, prereqs_file: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        enforcer.write_prereqs_section(prereqs_file)
        enforcer.write_prereqs_section(prereqs_file)
        content = Path(prereqs_file).read_text()
        # Both writes appended; the original content still present.
        assert content.count("## Exclusion Manifest Gate") == 2


# ---------------------------------------------------------------------------
# Test: check_queue
# ---------------------------------------------------------------------------


class TestCheckQueue:
    """REQ-INFRA-072: check_queue() returns blocked IDs from a candidate list."""

    def test_returns_blocked_ids(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        blocked = enforcer.check_queue([527, 491, 999])
        assert 527 in blocked
        assert 491 in blocked
        assert 999 not in blocked

    def test_empty_queue_returns_empty(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        assert enforcer.check_queue([]) == []

    def test_no_retired_in_queue(self, manifest_yaml: str) -> None:
        enforcer = ExclusionManifestEnforcer()
        enforcer.load_manifest(manifest_yaml)
        assert enforcer.check_queue([1, 2, 3]) == []


# ---------------------------------------------------------------------------
# Test: _parse_yaml_fallback
# ---------------------------------------------------------------------------


class TestParseYamlFallback:
    """Fallback parser produces same result as PyYAML for ops/exclusion_manifest.yaml format."""

    def test_parses_integer_ids(self) -> None:
        result = _parse_yaml_fallback(SAMPLE_YAML)
        assert 527 in result
        assert 491 in result

    def test_parses_reasons(self) -> None:
        result = _parse_yaml_fallback(SAMPLE_YAML)
        assert "RETRO-033" in result[527]

    def test_skips_invalid_ids(self) -> None:
        yaml_with_string = "  - experiment_id: jepa_v15_cascade\n    reason: 'test'\n"
        result = _parse_yaml_fallback(yaml_with_string)
        assert "jepa_v15_cascade" not in result

    def test_empty_input(self) -> None:
        assert _parse_yaml_fallback("") == {}


# ---------------------------------------------------------------------------
# Test: deliverable artifact exists and is valid
# ---------------------------------------------------------------------------


class TestDeliverableArtifact:
    """Verify the Exp 868 deliverable JSON was written with required fields."""

    ARTIFACT_PATH = "results/experiment_868_preflight_v16.json"

    def test_artifact_exists(self) -> None:
        assert os.path.exists(self.ARTIFACT_PATH), (
            f"Deliverable not found at {self.ARTIFACT_PATH}"
        )

    def test_artifact_is_valid_json(self) -> None:
        with open(self.ARTIFACT_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_required_fields_present(self) -> None:
        with open(self.ARTIFACT_PATH) as f:
            data = json.load(f)
        for field in ("experiment", "schema", "run_date", "started_at",
                      "finished_at", "duration_s", "status", "title"):
            assert field in data, f"Required field '{field}' missing"

    def test_manifest_enforcer_deployed(self) -> None:
        with open(self.ARTIFACT_PATH) as f:
            data = json.load(f)
        assert data.get("manifest_enforcer_deployed") is True

    def test_open_retros_count(self) -> None:
        with open(self.ARTIFACT_PATH) as f:
            data = json.load(f)
        assert data.get("open_retros_count") == 7

    def test_honest_verdict_governance_ready(self) -> None:
        with open(self.ARTIFACT_PATH) as f:
            data = json.load(f)
        assert data.get("honest_verdict") == "governance_ready"
