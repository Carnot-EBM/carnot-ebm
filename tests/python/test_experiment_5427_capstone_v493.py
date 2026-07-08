"""Tests for Exp5427 .493 terminal capstone truth table.

Spec refs: REQ-CAPSTONE-5427, SCENARIO-CAPSTONE-5427,
SCENARIO-CAPSTONE-5427-MISSING-INPUT,
SCENARIO-CAPSTONE-5427-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
import subprocess

import pytest

from carnot import experiment_5427_capstone_v493 as exp


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


def test_req_capstone_5427_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-5427: OpenSpec anchors the .493 capstone schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5427") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5427",
        "SCENARIO-CAPSTONE-5427",
        "SCENARIO-CAPSTONE-5427-MISSING-INPUT",
        "SCENARIO-CAPSTONE-5427-FIELD-PRINCIPLES",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp5417 risk-calibrated structured",
        "verification and Exp5418 predictive prefix/action safety",
        "Exp5423 ARC level-up as honest-null/no-bank",
        "`hardware_speedup_claim` = \"no unsupported acceleration",
        "`future_token_signal_allowed` = \"token/internal lane closure",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5427_builds_terminal_truth_table_from_actual_artifacts() -> None:
    """SCENARIO-CAPSTONE-5427: actual .493 artifacts produce honest lane buckets."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone 5427", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["upstream_artifacts_missing"] == []
    assert artifact["upstream_artifacts_read"] == list(exp.EXPECTED_INPUT_PATHS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")

    assert _lane_names(artifact["headline_ready_lanes"]) == [
        "evidence_reliance_csl",
        "gated_csl_promotion",
    ]
    assert _lane_names(artifact["bounded_lanes"]) == [
        "active_constraint_lns_scale",
        "pbit_hardware_transfer_preflight",
        "comparable_hardware_timing",
        "kan_measurement_access_certificates",
    ]
    assert _lane_names(artifact["honest_null_lanes"]) == ["arc_levelup"]
    assert _lane_names(artifact["blocked_lanes"]) == [
        "risk_calibrated_structured_verification",
        "predictive_prefix_action_safety",
        "token_internal_feature_lane_closed",
    ]

    rows = {row["lane"]: row for row in artifact["truth_table"]}
    assert rows["risk_calibrated_structured_verification"]["classification"] == "blocked"
    assert rows["risk_calibrated_structured_verification"]["blocked_reason"] == (
        "flagged_adversarial_and_corrigendum_pending"
    )
    assert rows["predictive_prefix_action_safety"]["classification"] == "blocked"
    assert rows["active_constraint_lns_scale"]["classification"] == "bounded"
    assert rows["pbit_hardware_transfer_preflight"]["classification"] == "bounded"
    assert rows["evidence_reliance_csl"]["classification"] == "headline_ready"
    assert rows["gated_csl_promotion"]["classification"] == "headline_ready"
    assert rows["arc_levelup"]["classification"] == "honest_null"
    assert rows["comparable_hardware_timing"]["classification"] == "bounded"
    assert rows["kan_measurement_access_certificates"]["classification"] == "bounded"
    assert rows["token_internal_feature_lane_closed"]["classification"] == "blocked"

    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["future_token_signal_allowed"] is False
    assert artifact["local_sota_gguf_receipts_valid"] is True
    assert artifact["research_roadmap_yaml_unchanged"] is True
    assert artifact["conductor_unchanged"] is True

    assert [row["target"] for row in artifact["next_recommendations"]] == [
        "arc_live_levelup",
        "pbit_hardware_transfer",
        "continuous_self_learning",
        "token_internal_backend",
    ]
    assert "ARC no-bank" in artifact["honest_verdict"]
    assert "no hardware speedup" in artifact["honest_verdict"]
    assert "token/internal lane closed" in artifact["honest_verdict"]


def test_scenario_capstone_5427_missing_inputs_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5427-MISSING-INPUT: absent inputs become missing lanes."""

    _copy_inputs(tmp_path)
    (tmp_path / exp.EXP5422).unlink()
    (tmp_path / exp.EXP5419).write_text("{", encoding="utf-8")

    artifact = exp.build_artifact(root=tmp_path)

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["upstream_artifacts_missing"] == [exp.EXP5419, exp.EXP5422]
    assert _lane_names(artifact["headline_ready_lanes"]) == ["evidence_reliance_csl"]
    assert _lane_names(artifact["bounded_lanes"]) == [
        "pbit_hardware_transfer_preflight",
        "comparable_hardware_timing",
        "kan_measurement_access_certificates",
    ]
    assert any(
        row["lane"] == "active_constraint_lns_scale" and row["classification"] == "missing"
        for row in artifact["missing_lanes"]
    )
    assert any(
        row["lane"] == "gated_csl_promotion" and row["classification"] == "missing"
        for row in artifact["missing_lanes"]
    )


def test_req_capstone_5427_run_writes_stable_json(tmp_path: Path) -> None:
    """REQ-CAPSTONE-5427: run() writes a deterministic capstone deliverable."""

    tests_run = [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5427_capstone_v493.py -q",
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5427_capstone_v493.py "
                "-m pytest tests/python/test_experiment_5427_capstone_v493.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5427_capstone_v493.py "
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


def test_req_capstone_5427_committed_result_matches_replay() -> None:
    """REQ-CAPSTONE-5427: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay


def test_req_capstone_5427_validation_rejects_overclaims() -> None:
    """REQ-CAPSTONE-5427: validation rejects schema drift and overclaims."""

    artifact = exp.build_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("upstream_artifacts_read")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing_field)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.492"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

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
    bad_token["future_token_signal_allowed"] = True
    with pytest.raises(ValueError, match="future_token_signal_allowed"):
        exp.validate_artifact(bad_token)

    bad_headline = deepcopy(artifact)
    bad_headline["headline_ready_lanes"].append(bad_headline["blocked_lanes"][0])
    with pytest.raises(ValueError, match="lane bucket"):
        exp.validate_artifact(bad_headline)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["research_roadmap_yaml_unchanged"] = False
    with pytest.raises(ValueError, match="research_roadmap_yaml_unchanged"):
        exp.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(artifact)
    bad_conductor["conductor_unchanged"] = False
    with pytest.raises(ValueError, match="conductor_unchanged"):
        exp.validate_artifact(bad_conductor)

    bad_lanes = deepcopy(artifact)
    bad_lanes["truth_table"] = list(reversed(bad_lanes["truth_table"]))
    with pytest.raises(ValueError, match="truth_table lane order"):
        exp.validate_artifact(bad_lanes)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)


def test_req_capstone_5427_helper_branches_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-5427: helper branches stay covered and fail closed."""

    _copy_inputs(tmp_path)
    payloads, read, missing, errors = exp.read_inputs(tmp_path)
    assert len(payloads) == len(exp.RESULT_ARTIFACT_PATHS)
    assert read == list(exp.EXPECTED_INPUT_PATHS)
    assert missing == []
    assert errors == []

    scalar = tmp_path / exp.EXP5420
    scalar.write_text("[]", encoding="utf-8")
    _, read, missing, errors = exp.read_inputs(tmp_path)
    assert exp.EXP5420 not in read
    assert exp.EXP5420 in missing
    assert {"path": exp.EXP5420, "classification": "not_json_object"} in errors

    assert exp.unwrap({"principle": "p", "value": False}) is False
    assert exp.unwrap("plain") == "plain"
    assert exp.json_ready((Path("a"), Path("b"))) == ["a", "b"]
    assert exp.lane_bucket_name("headline_ready") == "headline_ready_lanes"
    assert exp.lane_bucket_name("bounded") == "bounded_lanes"
    assert exp.lane_bucket_name("honest_null") == "honest_null_lanes"
    assert exp.lane_bucket_name("blocked") == "blocked_lanes"
    assert exp.lane_bucket_name("missing") == "missing_lanes"
    assert exp.lane_bucket_name("other") == "blocked_lanes"

    assert exp.flag_reasons({}) == []
    assert exp.flag_reasons({"flagged_adversarial": True, "corrigendum_pending": [{"kind": "x"}]}) == [
        "flagged_adversarial",
        "corrigendum_pending",
    ]

    assert exp.local_sota_gguf_receipts_valid({}) is False
    assert exp.classify_lane(
        exp.LANE_SPECS[0],
        {},
        [exp.EXP5417],
    )["classification"] == "missing"

    def _raise_oserror(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp.subprocess, "run", _raise_oserror)
    assert exp.git_path_unchanged(REPO, "research-roadmap.yaml") is True

    cli_result = tmp_path / "cli" / exp.RESULT_RELATIVE_PATH
    assert exp.main(["--root", str(REPO), "--result-path", str(cli_result)]) == 0
    assert json.loads(cli_result.read_text(encoding="utf-8"))["milestone"] == exp.MILESTONE
