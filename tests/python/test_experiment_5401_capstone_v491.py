"""Tests for Exp5401 .491 capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5401, SCENARIO-CAPSTONE-5401,
SCENARIO-CAPSTONE-5401-MISSING-OR-FLAGGED-INPUT,
SCENARIO-CAPSTONE-5401-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile

import pytest

from carnot import experiment_5401_capstone_v491 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _copy_upstream_inputs(root: Path) -> None:
    for relative in (*exp.UPSTREAM_ARTIFACT_PATHS, *exp.SIDECAR_ARTIFACT_PATHS):
        source = REPO / relative
        if not source.exists():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
    conductor = root / exp.CONDUCTOR_LOG_PATH
    conductor.parent.mkdir(parents=True, exist_ok=True)
    copyfile(REPO / exp.CONDUCTOR_LOG_PATH, conductor)


def test_req_capstone_5401_spec_declares_truth_table_contract() -> None:
    """REQ-CAPSTONE-5401: OpenSpec anchors the .491 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5401") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5401",
        "SCENARIO-CAPSTONE-5401",
        "SCENARIO-CAPSTONE-5401-MISSING-OR-FLAGGED-INPUT",
        "SCENARIO-CAPSTONE-5401-FIELD-PRINCIPLES",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5389 through Exp5400",
        "Exp5392 SHALL be marked blocked",
        "Exp5397 SHALL remain an honest no-bank ARC result",
        "`hardware_speedup_claim` = \"must remain false unless Exp5398 proves repeated",
        "`future_token_signal_allowed` = \"must remain false",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5401_builds_truth_table_without_overclaiming() -> None:
    """SCENARIO-CAPSTONE-5401: checked-in .491 artifacts yield honest gates."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone 5401", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["artifact_count_expected"] == exp.EXPECTED_TASK_COUNT
    assert artifact["artifact_count_found"] == exp.EXPECTED_TASK_COUNT
    assert artifact["missing_artifacts"] == []
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False

    assert artifact["structured_scaleup_ready"] is True
    assert artifact["formal_encoding_fixture_ready"] is False
    assert artifact["overwrite_guidance_corrigendum_clean"] is True
    assert artifact["pbit_boundary_ablation_ready"] is True
    assert artifact["continuous_self_learning_router_ready"] is True
    assert artifact["raw_episode_guard_ready"] is True
    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_repeatability_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["dynamic_counterexample_certificate_ready"] is True
    assert artifact["future_token_signal_allowed"] is False

    assert artifact["flagged_artifacts"] == [
        {
            "path": exp.EXP5392,
            "task_id": "exp5392-v491-formal-encoding-safety-fixture",
            "reasons": [
                "artifact flagged_adversarial=true",
                "conductor log status FLAGGED",
                "critical TAUTOLOGY corrigendum pending",
            ],
            "headline_eligible": False,
        }
    ]

    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert rows["structured_constraint_tax_scaleup"]["classification"] == "headline_ready"
    assert rows["formal_encoding_safety_fixture"]["classification"] == "blocked"
    assert rows["formal_encoding_safety_fixture"]["headline_ready"] is False
    assert rows["formal_encoding_safety_fixture"]["blocked_reason"] == "flagged_adversarial_tautology"
    assert rows["overwrite_guidance_corrigendum"]["classification"] == "headline_ready"
    assert rows["pbit_boundary_ablation"]["classification"] == "bounded_ready"
    assert rows["pbit_boundary_ablation"]["headline_ready"] is True
    assert rows["pbit_boundary_ablation"]["claim_boundary"] == "cpu_only_no_hardware_speedup"
    assert rows["continuous_self_learning_router"]["classification"] == "headline_ready"
    assert rows["raw_episode_memory_guard"]["classification"] == "headline_ready"
    assert rows["arc_level_up"]["classification"] == "honest_null"
    assert rows["hardware_repeatability"]["classification"] == "blocked"
    assert rows["kan_dynamic_certificate"]["classification"] == "headline_ready"
    assert rows["prd_evidence_table"]["classification"] == "partial"

    blocked = {row["lane"]: row for row in artifact["retired_or_blocked_lanes"]}
    assert blocked["formal_encoding_safety_fixture"]["state"] == "blocked_flagged_adversarial"
    assert blocked["arc_level_up"]["state"] == "honest_null_no_bank"
    assert blocked["hardware_repeatability"]["state"] == "blocked_no_board_local_repeats"
    assert blocked["hardware_speedup_claim"]["state"] == "blocked_no_repeatable_timing_speedup"
    assert blocked["future_token_internal_signal"]["state"] == "retired_until_backend_feature_artifact"

    assert artifact["headline_ready_lanes"] == [
        "structured_constraint_tax_scaleup",
        "overwrite_guidance_corrigendum",
        "pbit_boundary_ablation",
        "continuous_self_learning_router",
        "raw_episode_memory_guard",
        "kan_dynamic_certificate",
    ]
    assert artifact["honest_verdict"].startswith("complete:")
    assert "Exp5392 flagged" in artifact["honest_verdict"]
    assert "Exp5397 no-bank" in artifact["honest_verdict"]
    assert "no hardware speedup" in artifact["honest_verdict"]


def test_scenario_capstone_5401_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5401-MISSING-OR-FLAGGED-INPUT: gaps never imply success."""

    _copy_upstream_inputs(tmp_path)
    (tmp_path / exp.EXP5391).unlink()
    (tmp_path / exp.EXP5400).write_text("{", encoding="utf-8")

    artifact = exp.build_artifact(root=tmp_path)

    exp.validate_artifact(artifact)
    assert artifact["status"] == "honest_partial"
    assert artifact["artifact_count_found"] == exp.EXPECTED_TASK_COUNT - 2
    assert artifact["structured_scaleup_ready"] is False
    assert artifact["missing_artifacts"] == [
        exp.EXP5391,
        exp.EXP5400,
    ]
    assert artifact["artifact_read_errors"] == [
        {
            "path": exp.EXP5400,
            "classification": "malformed_json:Expecting property name enclosed in double quotes",
            "line": 1,
            "column": 2,
        }
    ]
    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert rows["structured_constraint_tax_scaleup"]["classification"] == "missing_inputs"
    assert rows["prd_evidence_table"]["classification"] == "missing_inputs"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "missing 2 upstream artifact" in artifact["honest_verdict"]


def test_req_capstone_5401_run_writes_stable_json(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5401: run writes the deterministic capstone JSON."""

    tests_run = [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5401_capstone_v491.py -q",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5401_capstone_v491.py "
                "-m pytest tests/python/test_experiment_5401_capstone_v491.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5401_capstone_v491.py "
                "--fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH

    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == exp.build_artifact(root=REPO, tests_run=tests_run)
    exp.validate_artifact(artifact)


def test_req_capstone_5401_committed_result_matches_replay() -> None:
    """REQ-CAPSTONE-5401: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_capstone_5401_validation_rejects_claim_drift() -> None:
    """REQ-CAPSTONE-5401: validator rejects inflated or inconsistent claims."""

    artifact = exp.build_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("artifact_count_expected")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.490"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_count = deepcopy(artifact)
    bad_count["artifact_count_found"] = 99
    with pytest.raises(ValueError, match="artifact_count_found"):
        exp.validate_artifact(bad_count)

    bad_missing = deepcopy(artifact)
    bad_missing["missing_artifacts"] = [exp.EXP5391]
    bad_missing["artifact_count_found"] = exp.EXPECTED_TASK_COUNT - 1
    bad_missing["status"] = "complete"
    with pytest.raises(ValueError, match="honest_partial"):
        exp.validate_artifact(bad_missing)

    bad_formal = deepcopy(artifact)
    bad_formal["formal_encoding_fixture_ready"] = True
    with pytest.raises(ValueError, match="formal_encoding_fixture_ready"):
        exp.validate_artifact(bad_formal)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_speedup)

    bad_token = deepcopy(artifact)
    bad_token["future_token_signal_allowed"] = True
    with pytest.raises(ValueError, match="future_token_signal_allowed"):
        exp.validate_artifact(bad_token)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["active_roadmap_modified"] = True
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        exp.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(artifact)
    bad_conductor["conductor_modified"] = True
    with pytest.raises(ValueError, match="conductor_modified"):
        exp.validate_artifact(bad_conductor)

    bad_truth = deepcopy(artifact)
    bad_truth["truth_table"][0]["classification"] = "success"
    with pytest.raises(ValueError, match="truth_table"):
        exp.validate_artifact(bad_truth)

    bad_flag = deepcopy(artifact)
    bad_flag["flagged_artifacts"] = []
    with pytest.raises(ValueError, match="flagged_artifacts"):
        exp.validate_artifact(bad_flag)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    assert exp.unwrap({"principle": "p", "value": False}) is False
    assert exp.unwrap("plain") == "plain"
