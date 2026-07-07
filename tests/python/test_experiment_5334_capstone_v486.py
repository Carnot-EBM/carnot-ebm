"""Tests for Exp 5334 .486 capstone synthesis.

Spec refs: REQ-CAPSTONE-5334, SCENARIO-CAPSTONE-5334,
SCENARIO-CAPSTONE-5334-BLOCKED-MISSING-INPUT,
SCENARIO-CAPSTONE-5334-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile

import pytest

from carnot import experiment_5334_capstone_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _copy_expected_artifacts(root: Path) -> None:
    for source in mod.EXPECTED_ARTIFACTS:
        destination = root / source.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(REPO / source.relative_path, destination)


def test_req_capstone_5334_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5334: OpenSpec anchors the .486 no-laundering contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5334") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5334",
        "SCENARIO-CAPSTONE-5334",
        "SCENARIO-CAPSTONE-5334-BLOCKED-MISSING-INPUT",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "runtime",
        "SOTA quality",
        "internal-signal receipts",
        "KAN localization",
        "reachability-only hardware",
        "`hardware_speedup_claim=false`",
        "`active_roadmap_modified=false`",
        "`conductor_modified=false`",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5334_builds_gate_reconciliation() -> None:
    """SCENARIO-CAPSTONE-5334: checked-in .486 artifacts produce honest gates."""

    artifact = mod.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit gate reconciliation", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.MILESTONE
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["artifacts_read"]["value"]) == len(mod.EXPECTED_ARTIFACTS)
    assert artifact["runtime_stable"] is True
    assert artifact["sota_quality_measured"] is True
    assert artifact["rewrite_state_ready"] is True
    assert artifact["smt_corrigendum_clean"] is True
    assert artifact["context_lifecycle_ready"] is True
    assert artifact["certificate_self_learning_ready"] is True
    assert artifact["internal_signal_path_open"] is True
    assert artifact["kan_localization_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False

    gates = {row["gate"]: row for row in artifact["gate_table"]["value"]}
    assert gates["runtime"]["ready"] is True
    assert gates["runtime"]["claim_boundary"] == "runtime stability only; no quality claim"
    assert gates["sota_quality"]["ready"] is True
    assert gates["sota_quality"]["claim_boundary"] == "bounded fixture-scored smoke; no headline quality claim"
    assert gates["internal_signal_receipts"]["ready"] is True
    assert gates["internal_signal_receipts"]["classification"] == "open_but_flagged"
    assert gates["hardware"]["ready"] is False
    assert gates["hardware"]["classification"] == "reachability_only_no_speedup"

    blocked = artifact["missing_or_blocked_artifacts"]["value"]
    assert any(row["experiment_number"] == 5321 and row["classification"] == "blocked" for row in blocked)
    assert any(row["experiment_number"] == 5331 and row["classification"] == "flagged" for row in blocked)
    assert any(row["experiment_number"] == 5333 and row["classification"] == "blocked" for row in blocked)

    recommendation = artifact["next_milestone_recommendation"]["value"]
    assert recommendation["recommendation"] == "self_learning_scale_up"
    assert "hardware_speedup" in recommendation["do_not_claim"]
    assert "headline_sota_quality" in recommendation["do_not_claim"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_capstone_5334_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5334: run writes a deterministic capstone artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=REPO,
        result_path=result_path,
        tests_run=[{"command": "unit run", "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == mod.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit run", "outcome": "passed"}],
    )
    mod.validate_artifact(artifact)


def test_scenario_capstone_5334_missing_artifact_blocks_without_false_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5334-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    _copy_expected_artifacts(tmp_path)
    (tmp_path / mod.EXP5331.relative_path).unlink()
    (tmp_path / mod.EXP5330.relative_path).write_text("{", encoding="utf-8")
    (tmp_path / mod.EXP5332.relative_path).write_text("[]", encoding="utf-8")

    artifact = mod.build_result_artifact(root=tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked_missing_required"
    assert artifact["honest_verdict"]["value"].startswith("blocked_missing_required")
    assert artifact["internal_signal_path_open"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert any(
        row["experiment_number"] == 5331 and row["classification"] == "missing"
        for row in artifact["missing_or_blocked_artifacts"]["value"]
    )
    assert any(
        row["experiment_number"] == 5330
        and row["classification"] == "malformed"
        and row["reason"].startswith("malformed_json:")
        for row in artifact["missing_or_blocked_artifacts"]["value"]
    )
    assert any(
        row["experiment_number"] == 5332
        and row["classification"] == "malformed"
        and row["reason"] == "not_json_object"
        for row in artifact["missing_or_blocked_artifacts"]["value"]
    )
    assert artifact["gate_table"]["value"][6]["gate"] == "internal_signal_receipts"
    assert artifact["gate_table"]["value"][6]["ready"] is False


def test_req_capstone_5334_validation_rejects_schema_drift() -> None:
    """REQ-CAPSTONE-5334: validator rejects missing principles and false claims."""

    artifact = mod.build_result_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("artifacts_read")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_field)

    bad_wrapped = deepcopy(artifact)
    bad_wrapped["gate_table"] = bad_wrapped["gate_table"]["value"]
    with pytest.raises(ValueError, match="gate_table"):
        mod.validate_artifact(bad_wrapped)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(bad_speedup)

    bad_bool = deepcopy(artifact)
    bad_bool["runtime_stable"] = {"value": True}
    with pytest.raises(ValueError, match="runtime_stable"):
        mod.validate_artifact(bad_bool)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["active_roadmap_modified"] = True
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(bad_roadmap)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:not-real"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_capstone_5334_repository_artifact_matches_schema() -> None:
    """REQ-CAPSTONE-5334: checked-in deliverable is a valid .486 capstone."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.MILESTONE
    assert artifact["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert artifact["hardware_speedup_claim"] is False
