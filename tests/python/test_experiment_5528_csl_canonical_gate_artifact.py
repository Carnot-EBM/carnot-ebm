"""Tests for Exp5528 canonical CSL gate artifact discipline.

Spec refs: REQ-LEARN-5528,
SCENARIO-LEARN-5528-SIDECAR-FAILURE,
SCENARIO-LEARN-5528-CANONICAL-GATE.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_5528_csl_canonical_gate_artifact as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
UPSTREAM_PATH = REPO / exp.UPSTREAM_ARTIFACT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5528_csl_canonical_gate_artifact.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5528_csl_canonical_gate_artifact.py "
    "-m pytest tests/python/test_experiment_5528_csl_canonical_gate_artifact.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5528_csl_canonical_gate_artifact.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TESTS_ADDED_OR_REUSED = [TEST_COMMAND, COVERAGE_COMMAND, FULL_TEST_COMMAND]


def test_req_learn_5528_spec_declares_canonical_gate_contract() -> None:
    """REQ-LEARN-5528: OpenSpec anchors the canonical sidecar-safe gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5528") :]

    for marker in (
        "REQ-LEARN-5528",
        "SCENARIO-LEARN-5528-SIDECAR-FAILURE",
        "SCENARIO-LEARN-5528-CANONICAL-GATE",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "scripts/conductor_gates.py",
        "same-experiment sidecar",
        "Exp5529 and Exp5530 roadmap gates",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5528_sidecar_failure_reproduces_with_conductor(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5528-SIDECAR-FAILURE: newest sidecars hide gates."""

    results_dir = tmp_path / "results"
    primary = results_dir / "experiment_5515_csl_independent_outcome_gate_repair.json"
    sidecar = results_dir / "experiment_5515_csl_independent_outcome_stream_fixture.json"
    exp.write_json(
        primary,
        {
            "metric_independence_clean": True,
            "csl_gate_fields_resolvable": True,
            "csl_experience_graph_ready": True,
        },
    )
    exp.write_json(sidecar, {"fixture_id": "exp5515-sidecar-without-gates"})
    os.utime(primary, (100.0, 100.0))
    os.utime(sidecar, (200.0, 200.0))

    reproduction = exp.reproduce_5515_sidecar_selection(results_dir)

    assert reproduction["gate_check_passed"] is False
    assert reproduction["selected_artifact"] == sidecar.as_posix()
    assert reproduction["primary_fields_visible_through_newest_artifact"] is False
    assert reproduction["gates_evaluated"][0]["actual"] is None
    assert "actual=None == expected=True" in reproduction["gate_check_summary"]


def test_scenario_learn_5528_canonical_artifact_fields_and_gate_probe(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5528-CANONICAL-GATE: Exp5528 gates resolve downstream."""

    results_dir = tmp_path / "results"
    result_path = results_dir / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        results_dir=results_dir,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert exp.find_artifact_for_task(exp.TASK_ID, results_dir) == result_path
    assert exp.same_exp_sidecar_after_primary(result_path, results_dir) is False
    assert artifact["metric_independence_clean"] is True
    assert artifact["csl_gate_fields_resolvable"] is True
    assert artifact["csl_experience_graph_ready"] is True
    assert artifact["continuous_self_learning_evidence"] is True
    assert artifact["conductor_gate_probe_passed"] is True
    assert artifact["csl_gate_fields_conductor_visible"] is True
    assert artifact["same_exp_sidecar_after_primary"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    for field in exp.COPIED_EVIDENCE_FIELDS:
        assert artifact[field] == upstream[field]
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact["field_principles"]
        assert artifact["field_principles"][field]

    probe = artifact["downstream_gate_probe"]
    assert probe["exp5529_full"]["passed"] is True
    assert probe["exp5530_exp5528_only"]["passed"] is True
    assert probe["exp5530_full"]["passed"] is False
    assert "exp5529-gated-csl-event-topic-residue-stress" in probe["exp5530_full"]["summary"]


def test_req_learn_5528_validation_fails_on_artifact_drift(tmp_path: Path) -> None:
    """REQ-LEARN-5528-1/2/3/4: validation rejects hidden or stale gates."""

    results_dir = tmp_path / "results"
    result_path = results_dir / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        results_dir=results_dir,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    exp.validate_artifact(artifact)
    assert exp._resolve_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH
    assert exp._resolve_path(REPO, result_path) == result_path
    assert exp.same_exp_sidecar_after_primary(results_dir / "missing.json", results_dir) is False

    drift_cases = [
        ("heldout_delta", 0.0, "heldout_delta"),
        ("metric_independence_clean", False, "metric_independence_clean"),
        ("csl_gate_fields_resolvable", False, "csl_gate_fields_resolvable"),
        ("csl_experience_graph_ready", False, "csl_experience_graph_ready"),
        (
            "continuous_self_learning_evidence",
            False,
            "continuous_self_learning_evidence",
        ),
        ("conductor_gate_probe_passed", False, "conductor_gate_probe_passed"),
        (
            "csl_gate_fields_conductor_visible",
            False,
            "csl_gate_fields_conductor_visible",
        ),
        ("same_exp_sidecar_after_primary", True, "same_exp_sidecar_after_primary"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("honest_verdict", "ready", "honest_verdict"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("csl_gate_fields_resolvable")
    missing["reproducibility_checksum"] = exp.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("heldout_delta")
    missing_principle["reproducibility_checksum"] = exp.reproducibility_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(missing_principle)

    no_tests = deepcopy(artifact)
    no_tests["tests_added_or_reused"] = []
    no_tests["reproducibility_checksum"] = exp.reproducibility_checksum(no_tests)
    with pytest.raises(ValueError, match="tests_added_or_reused"):
        exp.validate_artifact(no_tests)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)

    blocked_verdict = deepcopy(artifact)
    blocked_verdict["conductor_gate_probe_passed"] = False
    assert exp.honest_verdict(blocked_verdict).startswith("blocked:")

    sidecar = results_dir / "experiment_5528_late_sidecar.json"
    exp.write_json(sidecar, {"sidecar": True})
    os.utime(result_path, (100.0, 100.0))
    os.utime(sidecar, (200.0, 200.0))
    assert exp.same_exp_sidecar_after_primary(result_path, results_dir) is True
