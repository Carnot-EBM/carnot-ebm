"""Tests for Exp5453 .495 terminal capstone truth table.

Spec refs: REQ-CAPSTONE-5453, SCENARIO-CAPSTONE-5453,
SCENARIO-CAPSTONE-5453-MISSING-INPUT,
SCENARIO-CAPSTONE-5453-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile

import pytest

from carnot import experiment_5453_capstone_v495 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _copy_inputs(root: Path) -> None:
    for relative in exp.EXPECTED_INPUT_PATHS:
        source = REPO / relative
        if not source.exists() or source.is_dir():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)


def _lane_names(rows: list[dict[str, object]]) -> list[str]:
    return [str(row["lane"]) for row in rows]


def test_req_capstone_5453_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5453: OpenSpec anchors the .495 capstone schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5453") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5453",
        "SCENARIO-CAPSTONE-5453",
        "SCENARIO-CAPSTONE-5453-MISSING-INPUT",
        "SCENARIO-CAPSTONE-5453-FIELD-PRINCIPLES",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5443 verifier-potential fixtures",
        "Exp5444 local SOTA decoding as blocked",
        "Exp5450 ARC as honest-null/no-bank",
        "`hardware_speedup_claim` = \"hardware honesty",
        "`token_internal_lane_reopened` = \"closed-lane discipline",
    ):
        assert marker in section or marker in normalized

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5453_builds_truth_table_from_actual_artifacts() -> None:
    """SCENARIO-CAPSTONE-5453: actual .495 artifacts produce honest lane buckets."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone 5453", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["milestone"] == "2026.07.495"
    assert artifact["task_range"] == "exp5441-exp5453"
    assert artifact["artifacts_found"] == list(exp.RESULT_ARTIFACT_PATHS)
    assert artifact["artifact_read_errors"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")

    assert _lane_names(artifact["headline_ready_lanes"]) == [
        "verifier_potential_generation",
        "ast_kb_witnesses",
        "governed_csl",
        "memory_stress",
        "prd_gap_synthesis",
    ]
    assert _lane_names(artifact["bounded_lanes"]) == [
        "active_constraint_pbit_bridge",
        "hardware_receipts",
        "kan_certificates",
    ]
    assert _lane_names(artifact["blocked_lanes"]) == [
        "local_sota_decoding",
        "token_internal_access",
    ]
    assert _lane_names(artifact["honest_null_lanes"]) == [
        "arc_live_progress",
        "hardware_speedup_claim",
    ]
    assert artifact["missing_lanes"] == []

    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert rows["local_sota_decoding"]["classification"] == "blocked"
    assert rows["local_sota_decoding"]["blocked_reason"] == "flagged_adversarial_and_tautology"
    assert rows["local_sota_decoding"]["terminal_evidence"]["flagged_adversarial"] is True
    assert rows["arc_live_progress"]["classification"] == "honest_null"
    assert rows["hardware_speedup_claim"]["classification"] == "honest_null"
    assert rows["token_internal_access"]["classification"] == "blocked"

    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["token_internal_lane_reopened"] is False
    assert artifact["roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE

    assert [row["target"] for row in artifact["next_recommendations"]] == [
        "structured_decoding_corrigendum",
        "governed_memory_scale",
        "arc_live_levelup",
        "hardware_repeatability",
        "token_internal_backend",
    ]
    assert "ARC no-bank" in artifact["honest_verdict"]
    assert "no hardware speedup" in artifact["honest_verdict"]
    assert "token/internal lane closed" in artifact["honest_verdict"]


def test_scenario_capstone_5453_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5453-MISSING-INPUT: absent inputs become missing lanes."""

    _copy_inputs(tmp_path)
    (tmp_path / exp.EXP5446).unlink()
    (tmp_path / exp.EXP5444).write_text("{", encoding="utf-8")

    artifact = exp.build_artifact(root=tmp_path)

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["artifacts_missing"] == [exp.EXP5444, exp.EXP5446]
    assert _lane_names(artifact["headline_ready_lanes"]) == [
        "verifier_potential_generation",
        "ast_kb_witnesses",
        "memory_stress",
        "prd_gap_synthesis",
    ]
    assert _lane_names(artifact["blocked_lanes"]) == ["token_internal_access"]
    assert any(
        row["lane"] == "local_sota_decoding" and row["classification"] == "missing"
        for row in artifact["missing_lanes"]
    )
    assert any(
        row["lane"] == "governed_csl" and row["classification"] == "missing"
        for row in artifact["missing_lanes"]
    )


def test_req_capstone_5453_run_writes_stable_json(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5453: run() writes a deterministic capstone deliverable."""

    tests_run = [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5453_capstone_v495.py -q",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5453_capstone_v495.py "
                "-m pytest tests/python/test_experiment_5453_capstone_v495.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5453_capstone_v495.py "
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


def test_req_capstone_5453_committed_result_matches_replay() -> None:
    """REQ-CAPSTONE-5453: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_capstone_5453_validation_rejects_overclaims() -> None:
    """REQ-CAPSTONE-5453: validation rejects schema drift and overclaims."""

    artifact = exp.build_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("artifacts_found")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.494"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_task_range = deepcopy(artifact)
    bad_task_range["task_range"] = "exp5441-exp5452"
    with pytest.raises(ValueError, match="task_range"):
        exp.validate_artifact(bad_task_range)

    bad_field_principles = deepcopy(artifact)
    bad_field_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(bad_field_principles)

    bad_arc = deepcopy(artifact)
    bad_arc["arc_new_level_banked"] = True
    with pytest.raises(ValueError, match="arc_new_level_banked"):
        exp.validate_artifact(bad_arc)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_speedup)

    bad_token = deepcopy(artifact)
    bad_token["token_internal_lane_reopened"] = True
    with pytest.raises(ValueError, match="token_internal_lane_reopened"):
        exp.validate_artifact(bad_token)

    bad_headline = deepcopy(artifact)
    bad_headline["headline_ready_lanes"].append(bad_headline["blocked_lanes"][0])
    with pytest.raises(ValueError, match="lane bucket"):
        exp.validate_artifact(bad_headline)

    bad_local_sota = deepcopy(artifact)
    bad_local_sota["truth_table"][1]["classification"] = "headline_ready"
    with pytest.raises(ValueError, match="flagged local SOTA"):
        exp.validate_artifact(bad_local_sota)


def test_req_capstone_5453_defensive_branches_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-5453: defensive branches stay covered for malformed evidence."""

    assert exp.unwrap({"value": "wrapped"}) == "wrapped"
    assert exp.lane_classification("local_sota_decoding", {"verifier_guided_decoding_ready": True}) == "bounded"
    assert exp.lane_classification("unknown_lane", {}) == "blocked"
    assert exp.flag_reasons({"corrigendum_pending": [{"kind": "DURATION"}]}) == ["duration"]
    assert exp.flag_reasons({"unsupported_claims_detected": [{"rejected": False}]}) == ["unsupported_claim"]
    assert exp.recursive_key_true({"nested": [{"backend_receipt_present": True}]}, "backend_receipt_present")
    assert exp.json_ready(Path("relative/path")) == "relative/path"

    _copy_inputs(tmp_path)
    (tmp_path / "AGENTS.md").unlink()
    (tmp_path / exp.EXP5445).write_text("[]", encoding="utf-8")
    artifact = exp.build_artifact(root=tmp_path)
    assert artifact["source_context_missing"] == ["AGENTS.md"]
    assert artifact["artifacts_missing"] == [exp.EXP5445]
    assert artifact["artifact_read_errors"][0]["classification"] == "not_json_object"

    base = exp.build_artifact(root=REPO)

    bad_substrate = deepcopy(base)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_bool = deepcopy(base)
    bad_bool["arc_new_level_banked"] = "false"
    with pytest.raises(ValueError, match="arc_new_level_banked must be boolean"):
        exp.validate_artifact(bad_bool)

    bad_roadmap = deepcopy(base)
    bad_roadmap["roadmap_yaml_unchanged"] = False
    with pytest.raises(ValueError, match="roadmap_yaml_unchanged"):
        exp.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(base)
    bad_conductor["conductor_unchanged"] = False
    with pytest.raises(ValueError, match="conductor_unchanged"):
        exp.validate_artifact(bad_conductor)

    bad_prefix = deepcopy(base)
    bad_prefix["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict must start"):
        exp.validate_artifact(bad_prefix)

    bad_missing_prefix = deepcopy(base)
    bad_missing_prefix["artifacts_missing"] = [exp.EXP5444]
    with pytest.raises(ValueError, match="blocked: when inputs are missing"):
        exp.validate_artifact(bad_missing_prefix)

    bad_complete_prefix = deepcopy(base)
    bad_complete_prefix["honest_verdict"] = "blocked: wrong prefix"
    with pytest.raises(ValueError, match="complete: when all inputs are readable"):
        exp.validate_artifact(bad_complete_prefix)

    bad_order = deepcopy(base)
    bad_order["truth_table"] = list(reversed(bad_order["truth_table"]))
    with pytest.raises(ValueError, match="truth_table lane order"):
        exp.validate_artifact(bad_order)

    bad_classification = deepcopy(base)
    bad_classification["truth_table"][0]["classification"] = "closed"
    with pytest.raises(ValueError, match="classification invalid"):
        exp.validate_artifact(bad_classification)

    bad_evidence = deepcopy(base)
    bad_evidence["truth_table"][0]["terminal_evidence"] = []
    with pytest.raises(ValueError, match="terminal_evidence"):
        exp.validate_artifact(bad_evidence)

    bad_tautology_headline = deepcopy(base)
    bad_tautology_headline["truth_table"][1]["terminal_evidence"]["flagged_adversarial"] = False
    bad_tautology_headline["truth_table"][1]["classification"] = "headline_ready"
    with pytest.raises(ValueError, match="flagged local SOTA"):
        exp.validate_artifact(bad_tautology_headline)

    bad_expected_headlines = deepcopy(base)
    bad_expected_headlines["truth_table"][0]["classification"] = "bounded"
    buckets = exp.bucket_lanes(bad_expected_headlines["truth_table"])
    for bucket_name, rows in buckets.items():
        bad_expected_headlines[bucket_name] = rows
    with pytest.raises(ValueError, match="headline_ready_lanes"):
        exp.validate_artifact(bad_expected_headlines)

    bad_recommendations = deepcopy(base)
    bad_recommendations["next_recommendations"][0]["target"] = "wrong"
    with pytest.raises(ValueError, match="next_recommendations"):
        exp.validate_artifact(bad_recommendations)

    bad_checksum = deepcopy(base)
    bad_checksum["tests_run"] = [{"command": "changed", "outcome": "failed"}]
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    def _raise_oserror(*_args: object, **_kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp.subprocess, "run", _raise_oserror)
    assert exp.git_path_unchanged(REPO, "research-roadmap.yaml") is True

    output_path = tmp_path / exp.RESULT_RELATIVE_PATH
    assert exp.main(["--root", str(REPO), "--result-path", str(output_path)]) == 0
    assert output_path.exists()
