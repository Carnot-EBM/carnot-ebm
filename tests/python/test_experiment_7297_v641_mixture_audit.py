"""Audit delayed-feedback mixtures under REQ-CL-7297 and its scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7297_v641_mixture_audit as audit


@pytest.fixture(scope="module")
def measured_audit(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[audit.ExperimentPaths, dict[str, object]]:
    """Build one stream from each stratum once for all measured assertions."""

    paths = audit.ExperimentPaths.under(tmp_path_factory.mktemp("exp7297-audit"))
    artifact = audit.build_and_seal(
        audit.REPO_ROOT,
        paths,
        stream_ids=("evaluation-01", "evaluation-13"),
        bootstrap_draws=200,
        progress=True,
    )
    return paths, artifact


def test_req_cl_7297_freezes_no_llm_audit_contract() -> None:
    """REQ-CL-7297 fixes the seven arms, bounds, seeds, and current invocation state."""

    spec = (audit.REPO_ROOT / audit.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7297" in spec
    assert len(set(audit.SCENARIO_PATTERN.findall(spec))) == 7
    assert audit.ARMS == audit.learning.ARMS
    assert audit.EVALUATION_STREAM_SEEDS == audit.fixture.EVALUATION_STREAM_SEEDS
    assert audit.MODEL_SPECS == []
    assert audit.MODEL_INVOKED is False
    assert set(audit.INVOCATION_COUNTS.values()) == {0}
    assert audit.INFERENCE_SUBSTRATE == "cpu_exact_solver_or_simulator"
    assert audit.REDUCER_INFERENCE_SUBSTRATE == "aggregation_from_upstream_artifacts"
    assert set(audit.REQUIRED_ARTIFACT_FIELDS) <= set(audit.FIELD_PRINCIPLES)


def test_scenario_cl_7297_preconditions_authenticate_both_frozen_contracts(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7297-PRECONDITIONS binds capture and the frozen fixture."""

    paths = audit.ExperimentPaths.under(tmp_path)
    checks, hashes, capture, contract = audit.collect_preconditions(audit.REPO_ROOT, paths)

    assert all(row["passed"] is True for row in checks)
    assert capture["mixture_capture_complete_score"] == 1
    assert capture["mixture_value_score"] == 0
    assert contract["mixture_fixture_ready_score"] == 1
    assert contract["learning_contract"]["eta"] == 0.5
    assert contract["learning_contract"]["fixed_share"] == 0.02
    assert hashes[str(audit.REPO_ROOT / audit.UPSTREAM_CAPTURE)] is not None
    assert hashes[str(audit.REPO_ROOT / audit.UPSTREAM_CONTRACT)] is not None

    missing = tmp_path / "missing.json"
    checks, hashes, capture, contract = audit.collect_preconditions(
        audit.REPO_ROOT,
        paths,
        capture_path=missing,
    )
    blocked = audit.build_blocked_artifact(
        checks,
        hashes,
        ("evaluation-01",),
        started_at="2026-09-14T00:00:00+00:00",
        duration_s=0.01,
    )
    assert capture == {}
    assert contract["mixture_fixture_ready_score"] == 1
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["first_failure"]["field"] == (
        "mixture_capture_complete_score"
    )
    assert audit.validate_artifact(blocked) == []


def test_scenario_cl_7297_cold_replay_matches_every_prediction_and_transition(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7297-COLD-REPLAY checks all arms without producer aggregates."""

    _, artifact = measured_audit
    parity = artifact["cold_replay_parity"]
    assert parity["fresh_process"] is True
    assert parity["producer_aggregate_accessed"] is False
    assert parity["arm_count"] == 7
    assert parity["prediction_mismatch_count"] == 0
    assert parity["state_hash_mismatch_count"] == 0
    assert parity["transition_mismatch_count"] == 0
    assert parity["matched_prediction_rows"] == 2 * 7 * 896
    assert parity["matched_transition_rows"] == 2 * 128
    assert parity["mismatch_rows"] == []


def test_scenario_cl_7297_intervals_keep_strata_and_feedback_subsets(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7297-INTERVALS prevents pooled masking of frozen criteria."""

    _, artifact = measured_audit
    rows = artifact["independent_interval_rows"]
    assert rows
    assert {row["stratum"] for row in rows} == {
        "overall",
        "separated_recurrence",
        "overlapping_recurrence",
    }
    assert {row["independent_unit"] for row in rows} == {"stream"}
    assert {row["future_subset"] for row in rows} >= {
        "all_future",
        "feedback_selected",
        "non_feedback_future",
        "recurrence",
    }
    assert all(row["paired_differences"] for row in rows)
    assert all(row["bootstrap_seed"] == audit.BOOTSTRAP_SEED for row in rows)
    assert (
        artifact["acceptance_gate_results"]["separated_recurrence_vs_frozen_warmup"]["passed"]
        is False
    )


def test_scenario_cl_7297_interventions_reject_all_invalid_copies(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7297-INTERVENTIONS detects six distinct invalid mechanisms."""

    _, artifact = measured_audit
    rows = artifact["causal_intervention_rows"]
    assert {row["control"] for row in rows} == set(audit.CONTROL_NAMES)
    assert all(row["passed"] is True for row in rows)
    assert all(row["invalid_copy_rejected"] is True for row in rows)
    assert all(row["evaluation_rows_immutable"] is True for row in rows)
    assert len({row["detection"] for row in rows}) == len(audit.CONTROL_NAMES)


def test_scenario_cl_7297_causality_excludes_current_label_fitting(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7297-CAUSALITY counts only post-update unrevealed changes."""

    _, artifact = measured_audit
    rows = artifact["causal_change_rows"]
    assert rows
    assert all(row["prediction_index"] > row["causing_release_index"] for row in rows)
    assert all(row["prediction_index"] < row["own_label_release_index"] for row in rows)
    assert all(row["legitimate_feedback_update"] is True for row in rows)
    assert all(row["changed_from_uniform"] is True for row in rows)
    assert artifact["causal_summary"]["pre_update_changed_prediction_count"] == 0
    assert artifact["memory_label_accounting"]["uncharged_state_reference_count"] == 0


def test_scenario_cl_7297_e2e_and_terminal_keep_complete_null(
    measured_audit: tuple[audit.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7297-E2E and TERMINAL preserve a complete efficacy null."""

    paths, artifact = measured_audit
    assert [row["stage"] for row in artifact["e2e_rows"]] == list(audit.E2E_STAGES)
    assert all(row["passed"] is True for row in artifact["e2e_rows"])
    assert artifact["status"] == "complete"
    assert artifact["mixture_audit_complete_score"] == 1
    assert artifact["mixture_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["retirement_scope"] == audit.RETIREMENT_SCOPE
    assert (
        audit.validate_artifact(
            artifact,
            expected_stream_ids=("evaluation-01", "evaluation-13"),
            check_files=True,
        )
        == []
    )

    changed = deepcopy(artifact)
    changed["cold_replay_parity"]["prediction_mismatch_count"] = 1
    assert "checksum" in audit.validate_artifact(changed)
    assert "cold_replay_parity" in audit.validate_artifact(changed)

    receipt = {
        "command": "focused-test",
        "exit_code": 0,
        "classification": "passed",
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "0" * 64,
    }
    sealed = audit.attach_validation_receipts(artifact, [receipt])
    output = paths.artifact
    audit.write_artifact(
        output,
        sealed,
        expected_stream_ids=("evaluation-01", "evaluation-13"),
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"


def test_req_cl_7297_thin_entrypoint_and_score_logic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7297 keeps orchestration reusable and promotion conjunctive."""

    assert audit.derive_terminal_scores(True, False, True) == (1, 0, "null")
    assert audit.derive_terminal_scores(True, True, True) == (1, 1, "circular_positive")
    assert audit.derive_terminal_scores(False, True, True) == (0, 0, "partial")

    commands = audit._validation_commands(tmp_path / "candidate.json")
    flat = [part for command in commands for part in command]
    assert "scripts/check_spec_coverage.py" in flat
    assert "scripts/adversarial_verify.py" in flat
    assert "scripts/verdict_row_consistency_lint.py" in flat
    assert "mypy" in flat
    assert "ruff" in flat

    wrapper = audit.REPO_ROOT / audit.WRAPPER_PATH
    source = wrapper.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 24
    monkeypatch.setattr(audit, "main", lambda: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 0
