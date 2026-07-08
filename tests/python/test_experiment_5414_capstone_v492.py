"""Tests for Exp5414 .492 terminal capstone synthesis.

Spec refs: REQ-CAPSTONE-5414, SCENARIO-CAPSTONE-5414,
SCENARIO-CAPSTONE-5414-MISSING-OR-BLOCKED-INPUT,
SCENARIO-CAPSTONE-5414-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
import subprocess

import pytest

from carnot import experiment_5414_capstone_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _copy_inputs(root: Path) -> None:
    for relative in (*exp.EXPECTED_ARTIFACT_PATHS, *exp.SIDECAR_ARTIFACT_PATHS):
        source = REPO / relative
        if not source.exists():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)
    for relative in exp.STATUS_CONTEXT_PATHS:
        source = REPO / relative
        if not source.exists() or source.is_dir():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)


def test_req_capstone_5414_spec_declares_terminal_truth_table_contract() -> None:
    """REQ-CAPSTONE-5414: OpenSpec anchors the .492 capstone fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5414") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5414",
        "SCENARIO-CAPSTONE-5414",
        "SCENARIO-CAPSTONE-5414-MISSING-OR-BLOCKED-INPUT",
        "SCENARIO-CAPSTONE-5414-FIELD-PRINCIPLES",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5402 through Exp5413",
        "Exp5410 as an honest no-bank ARC result",
        "Exp5411 as restored same-workload PolarFire repeatability",
        "`hardware_speedup_claim` = \"must remain false",
        "`future_token_signal_allowed` = \"closed token/internal lane",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5414_builds_terminal_truth_table() -> None:
    """SCENARIO-CAPSTONE-5414: actual .492 artifacts produce honest gates."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone 5414", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["artifacts_read"] == [
        *(str(path) for path in exp.EXPECTED_ARTIFACT_PATHS),
        *(str(path) for path in exp.SIDECAR_ARTIFACT_PATHS),
        *exp.CONDUCTOR_STATUS_INPUTS,
    ]
    assert artifact["missing_artifacts"] == []
    assert artifact["artifact_read_errors"] == []
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["ops_docs_modified"] is False
    assert artifact["traceability_modified"] is False

    assert artifact["formal_encoding_corrigendum_clean"] is True
    assert artifact["structured_safety_action_panel_ready"] is True
    assert artifact["active_constraint_warmstart_ready"] is True
    assert artifact["pbit_qubo_stress_ready"] is True
    assert artifact["resource_accounted_csl_ready"] is True
    assert artifact["uncertainty_gated_promotion_ready"] is True
    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_repeatability_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["kan_active_constraint_certificate_ready"] is True
    assert artifact["future_token_signal_allowed"] is False
    assert artifact["local_sota_inference_ready"] is True

    assert artifact["headline_ready_lanes"] == [
        "formal_encoding_corrigendum",
        "structured_safety_action_panel",
        "resource_accounted_csl",
        "uncertainty_gated_promotion",
    ]
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "ARC no-bank" in artifact["honest_verdict"]
    assert "no hardware speedup" in artifact["honest_verdict"]
    assert "token/internal lane closed" in artifact["honest_verdict"]

    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert list(rows) == list(exp.TRUTH_TABLE_LANES)
    assert rows["formal_corrigendum"]["classification"] == "headline_ready"
    assert rows["structured_safety_action_scaleup"]["classification"] == "headline_ready"
    assert rows["active_constraint_guidance"]["classification"] == "bounded_ready"
    assert rows["pbit_qubo_stress"]["classification"] == "bounded_ready"
    assert rows["pbit_qubo_stress"]["claim_boundary"] == "cpu_only_no_hardware_speedup"
    assert rows["resource_accounted_csl"]["classification"] == "headline_ready"
    assert rows["uncertainty_gated_promotion"]["classification"] == "headline_ready"
    assert rows["arc_live_levelup"]["classification"] == "honest_null"
    assert rows["arc_live_levelup"]["headline_ready"] is False
    assert rows["hardware_repeatability"]["classification"] == "partial"
    assert rows["hardware_repeatability"]["evidence"]["repeated_same_workload_ready"] is True
    assert rows["hardware_repeatability"]["evidence"]["hardware_speedup_claim"] is False
    assert rows["kan_active_constraint_certificate"]["classification"] == "bounded_ready"
    assert rows["local_sota_inference"]["classification"] == "bounded_ready"
    assert rows["token_internal_lane"]["classification"] == "blocked"
    assert rows["token_internal_lane"]["evidence"]["future_token_signal_allowed"] is False

    assert artifact["non_headline_lanes"] == [
        "active_constraint_guidance",
        "pbit_qubo_stress",
        "arc_live_levelup",
        "hardware_repeatability",
        "kan_active_constraint_certificate",
        "local_sota_inference",
        "token_internal_lane",
    ]
    assert [row["target"] for row in artifact["next_recommendations"]] == [
        "arc_live_levelup",
        "hardware_speedup",
        "hardware_reachability",
        "active_constraint_scale",
        "pbit_hardware_transfer",
        "kan_certificate_family",
        "token_internal_backend",
        "next_milestone",
    ]
    assert artifact["claim_boundary_checks"] == exp.CLAIM_BOUNDARY_CHECKS


def test_scenario_capstone_5414_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5414-MISSING-OR-BLOCKED-INPUT: missing inputs block."""

    _copy_inputs(tmp_path)
    (tmp_path / exp.EXP5405).unlink()
    (tmp_path / exp.EXP5413).write_text("{", encoding="utf-8")

    artifact = exp.build_artifact(root=tmp_path)

    exp.validate_artifact(artifact)
    assert artifact["status"] == "blocked_missing_inputs"
    assert artifact["missing_artifacts"] == [exp.EXP5405, exp.EXP5413]
    assert artifact["artifact_read_errors"] == [
        {
            "path": exp.EXP5413,
            "classification": "malformed_json:Expecting property name enclosed in double quotes",
            "line": 1,
            "column": 2,
        }
    ]
    assert artifact["structured_safety_action_panel_ready"] is False
    assert artifact["headline_ready_lanes"] == [
        "formal_encoding_corrigendum",
        "resource_accounted_csl",
        "uncertainty_gated_promotion",
    ]
    assert artifact["honest_verdict"].startswith("blocked:")

    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert rows["structured_safety_action_scaleup"]["classification"] == "missing_inputs"
    assert rows["structured_safety_action_scaleup"]["headline_ready"] is False


def test_req_capstone_5414_run_writes_stable_json(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5414: run() writes the deterministic deliverable JSON."""

    tests_run = [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5414_capstone_v492.py -q",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5414_capstone_v492.py "
                "-m pytest tests/python/test_experiment_5414_capstone_v492.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5414_capstone_v492.py "
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
    assert artifact["result_path"] == str(exp.RESULT_RELATIVE_PATH)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_capstone_5414_committed_result_matches_replay() -> None:
    """REQ-CAPSTONE-5414: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_capstone_5414_validation_rejects_overclaims() -> None:
    """REQ-CAPSTONE-5414: validator rejects claim drift and missing guards."""

    artifact = exp.build_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("artifacts_read")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.491"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_field_principles = deepcopy(artifact)
    bad_field_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_field_principles)

    bad_status_mismatch = deepcopy(artifact)
    bad_status_mismatch["missing_artifacts"] = [exp.EXP5405]
    with pytest.raises(ValueError, match="status mismatch"):
        exp.validate_artifact(bad_status_mismatch)

    bad_formal = deepcopy(artifact)
    bad_formal["formal_encoding_corrigendum_clean"] = "yes"
    with pytest.raises(ValueError, match="formal_encoding_corrigendum_clean"):
        exp.validate_artifact(bad_formal)

    bad_arc = deepcopy(artifact)
    bad_arc["arc_new_level_banked"] = True
    with pytest.raises(ValueError, match="arc_new_level_banked"):
        exp.validate_artifact(bad_arc)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_speedup)

    bad_token = deepcopy(artifact)
    bad_token["future_token_signal_allowed"] = True
    with pytest.raises(ValueError, match="future_token_signal_allowed"):
        exp.validate_artifact(bad_token)

    bad_headline = deepcopy(artifact)
    bad_headline["headline_ready_lanes"].append("arc_live_levelup")
    with pytest.raises(ValueError, match="headline_ready_lanes"):
        exp.validate_artifact(bad_headline)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["active_roadmap_modified"] = True
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        exp.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(artifact)
    bad_conductor["conductor_modified"] = True
    with pytest.raises(ValueError, match="conductor_modified"):
        exp.validate_artifact(bad_conductor)

    bad_ops = deepcopy(artifact)
    bad_ops["ops_docs_modified"] = True
    with pytest.raises(ValueError, match="ops_docs_modified"):
        exp.validate_artifact(bad_ops)

    bad_traceability = deepcopy(artifact)
    bad_traceability["traceability_modified"] = True
    with pytest.raises(ValueError, match="traceability_modified"):
        exp.validate_artifact(bad_traceability)

    bad_lane_order = deepcopy(artifact)
    bad_lane_order["truth_table"] = list(reversed(bad_lane_order["truth_table"]))
    with pytest.raises(ValueError, match="truth_table lane order"):
        exp.validate_artifact(bad_lane_order)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    bad_classification = deepcopy(artifact)
    bad_classification["truth_table"][0]["classification"] = "success"
    with pytest.raises(ValueError, match="truth_table classification"):
        exp.validate_artifact(bad_classification)

    bad_row = deepcopy(artifact)
    bad_row["truth_table"][0].pop("evidence")
    with pytest.raises(ValueError, match="truth_table missing evidence"):
        exp.validate_artifact(bad_row)

    bad_non_headline = deepcopy(artifact)
    bad_non_headline["non_headline_lanes"] = []
    with pytest.raises(ValueError, match="non_headline_lanes"):
        exp.validate_artifact(bad_non_headline)

    bad_claim_checks = deepcopy(artifact)
    bad_claim_checks["claim_boundary_checks"] = {}
    with pytest.raises(ValueError, match="claim_boundary_checks"):
        exp.validate_artifact(bad_claim_checks)

    bad_blocked_verdict = deepcopy(artifact)
    bad_blocked_verdict["missing_artifacts"] = [exp.EXP5405]
    bad_blocked_verdict["status"] = "blocked_missing_inputs"
    bad_blocked_verdict["honest_verdict"] = "complete: wrong prefix"
    with pytest.raises(ValueError, match="honest_verdict must start with blocked"):
        exp.validate_artifact(bad_blocked_verdict)

    bad_complete_verdict = deepcopy(artifact)
    bad_complete_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict must start with complete"):
        exp.validate_artifact(bad_complete_verdict)

    assert exp.unwrap({"principle": "p", "value": False}) is False
    assert exp.unwrap("plain") == "plain"
    assert exp.json_ready((Path("a"), Path("b"))) == ["a", "b"]


def test_req_capstone_5414_helper_branches_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-5414: helper branches stay covered and fail closed."""

    _copy_inputs(tmp_path)
    (tmp_path / exp.EXP5406).write_text("[]", encoding="utf-8")
    payloads, artifacts_read, missing, errors = exp.read_inputs(tmp_path)
    assert exp.EXP5406 not in payloads
    assert exp.EXP5406 not in artifacts_read
    assert exp.EXP5406 in missing
    assert {"path": exp.EXP5406, "classification": "not_json_object"} in errors

    flagged = exp.flagged_inputs(
        {
            exp.EXP5404: {
                "flagged_adversarial": True,
                "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
            }
        }
    )
    assert flagged == [
        {
            "path": exp.EXP5404,
            "reasons": ["flagged_adversarial=true", "corrigendum_pending_present"],
            "headline_eligible": False,
        }
    ]
    assert exp.classification(exp.EXP5404, (), {exp.EXP5404}, "headline_ready") == "blocked"
    assert exp.joint_classification((exp.EXP5404, exp.EXP5405), (), {exp.EXP5405}, "bounded_ready") == "blocked"
    assert exp.future_token_signal_allowed(
        {exp.EXP5413: {"claim_boundary_checks": {"token_internal_backend_claimed_without_receipt": True}}}
    ) is True
    assert exp.future_token_signal_allowed({}) is False

    for helper in (
        exp.formal_evidence,
        exp.structured_evidence,
        exp.active_constraint_evidence,
        exp.pbit_evidence,
        exp.resource_csl_evidence,
        exp.uncertainty_evidence,
        exp.arc_evidence,
        exp.hardware_evidence,
        exp.kan_evidence,
    ):
        assert helper(None) == {}
    assert exp.conductor_status_summary(tmp_path / "missing-root") == {
        "path": "ops/conductor-log.md",
        "present": False,
        "v492_rows": 0,
        "flagged_rows": 0,
    }

    def _raise_oserror(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp.subprocess, "run", _raise_oserror)
    assert exp.git_path_modified(REPO, "research-roadmap.yaml") is False

    cli_result = tmp_path / "cli" / exp.RESULT_RELATIVE_PATH
    assert exp.main(["--root", str(REPO), "--result-path", str(cli_result)]) == 0
    assert json.loads(cli_result.read_text(encoding="utf-8"))["milestone"] == exp.MILESTONE
