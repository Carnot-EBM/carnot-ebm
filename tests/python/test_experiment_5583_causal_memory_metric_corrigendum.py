"""Tests for Exp5583 cached-row causal memory metric corrigendum.

Spec refs: REQ-LEARN-5583,
SCENARIO-LEARN-5583-ROWS,
SCENARIO-LEARN-5583-ESTIMANDS,
SCENARIO-LEARN-5583-CONTROLS,
SCENARIO-LEARN-5583-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5583_causal_memory_metric_corrigendum as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5583_causal_memory_metric_corrigendum.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5583_causal_memory_metric_corrigendum.py "
    "-m pytest tests/python/test_experiment_5583_causal_memory_metric_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5583_causal_memory_metric_corrigendum.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5583_causal_memory_metric_corrigendum.json"
)
TESTS_ADDED_OR_REUSED = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    ADVERSARIAL_COMMAND,
]


def _source() -> dict[str, object]:
    return exp.load_json(REPO / exp.SOURCE_5569_RELATIVE_PATH)


def _rows() -> list[dict[str, object]]:
    return exp.reconstruct_cached_rows(_source())


def test_req_learn_5583_spec_declares_corrigendum_contract() -> None:
    """REQ-LEARN-5583: OpenSpec anchors the cached-row corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5583") : spec.index("## REQ-LEARN-5570")]

    for marker in (
        "REQ-LEARN-5583",
        "SCENARIO-LEARN-5583-ROWS",
        "SCENARIO-LEARN-5583-ESTIMANDS",
        "SCENARIO-LEARN-5583-CONTROLS",
        "SCENARIO-LEARN-5583-ARTIFACT",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "forward transfer is the optimized-minus-static success delta",
        "backward retention is the optimized-minus-static success",
        "forgetting is the optimized within-family delayed loss",
    ):
        assert marker in section
    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_learn_5583_rows_reconstruct_membership_or_block() -> None:
    """SCENARIO-LEARN-5583-ROWS: cached rows reconstruct or block."""

    source = _source()
    rows = exp.reconstruct_cached_rows(source)
    summary = exp.rows_reconstructed(rows, source)

    assert summary["source_artifact"] == str(exp.SOURCE_5569_RELATIVE_PATH)
    assert summary["complete"] is True
    assert summary["total_rows"] == 3000
    assert summary["seeds"] == [5569, 5570, 5571, 5572, 5573]
    assert summary["arms"] == list(exp.ARM_NAMES)
    assert summary["rows_per_arm_seed"] == 120
    assert summary["sessions"] == 6
    assert summary["families"] == 6
    assert all(row["family_name"] for row in rows)
    assert all("local_index" in row for row in rows)

    missing_row = deepcopy(source)
    missing_row["tournament"]["seed_results"][0]["arm_rows"][exp.NO_MEMORY_ARM].pop()
    with pytest.raises(ValueError, match="row-level outcomes"):
        exp.reconstruct_cached_rows(missing_row)

    missing_arm = deepcopy(source)
    missing_arm["tournament"]["seed_results"][0]["arm_rows"].pop(exp.NO_MEMORY_ARM)
    with pytest.raises(ValueError, match="tournament arms"):
        exp.reconstruct_cached_rows(missing_arm)

    missing_membership = deepcopy(source)
    missing_membership["sessions"][0]["event_ids"].pop(0)
    with pytest.raises(ValueError, match="session membership"):
        exp.reconstruct_cached_rows(missing_membership)

    mismatched_membership = deepcopy(source)
    mismatched_membership["tournament"]["seed_results"][0]["arm_rows"][exp.NO_MEMORY_ARM][0][
        "session_index"
    ] = 99
    with pytest.raises(ValueError, match="session membership mismatch"):
        exp.reconstruct_cached_rows(mismatched_membership)

    incomplete_grid = [row for row in rows if row["arm"] != exp.NO_MEMORY_ARM]
    with pytest.raises(ValueError, match="incomplete seed/arm grid"):
        exp._require_complete_grid(incomplete_grid, expected_n_events=120)


def test_scenario_learn_5583_estimands_are_distinct_from_cached_rows() -> None:
    """SCENARIO-LEARN-5583-ESTIMANDS: corrected metrics are not aliases."""

    rows = _rows()
    estimands = exp.compute_estimands(rows)
    comparison = exp.arm_comparison(rows)

    assert estimands["forward_transfer_delta"] == 0.0
    assert estimands["backward_retention_delta"] == pytest.approx(0.3333333333)
    assert estimands["forgetting_delta"] == pytest.approx(0.25)
    assert estimands["forward_transfer"]["denominator_per_arm"] == 80
    assert estimands["backward_retention"]["denominator_per_arm"] == 120
    assert estimands["forgetting"]["first_delayed_denominator"] == 120
    assert estimands["forgetting"]["final_delayed_denominator"] == 120
    assert (
        len(
            {
                estimands["forward_transfer_delta"],
                estimands["backward_retention_delta"],
                estimands["forgetting_delta"],
            }
        )
        == 3
    )

    assert set(comparison) == set(exp.ARM_NAMES)
    assert (
        comparison[exp.SELF_OPTIMIZED_CAUSAL_ARM]["heldout_success"]
        > comparison[exp.STATIC_CAUSAL_ARM]["heldout_success"]
    )
    assert comparison[exp.SELF_OPTIMIZED_CAUSAL_ARM]["later_family_first_exposure_success"] == 0.0
    assert comparison[exp.SELF_OPTIMIZED_CAUSAL_ARM]["read_rate"] == 0.8
    assert exp.policy_ready(estimands, controls_passed=True) is False


def test_scenario_learn_5583_controls_prove_metric_independence() -> None:
    """SCENARIO-LEARN-5583-CONTROLS: controls gate policy promotion."""

    rows = _rows()
    control = exp.permutation_control(rows)
    positive = exp.metric_independence_positive_control()

    assert control["passed"] is True
    assert control["permuted_policy_ready"] is False
    assert (
        control["permuted_metrics"]["backward_retention_delta"]
        != control["original_metrics"]["backward_retention_delta"]
    )
    assert (
        control["permuted_metrics"]["forgetting_delta"]
        != control["original_metrics"]["forgetting_delta"]
    )

    fixture = exp.metric_independence_fixture()
    before = exp.compute_estimands(fixture)
    after = exp.flip_forward_positive_control(fixture)
    after_metrics = exp.compute_estimands(after)
    assert after_metrics["forward_transfer_delta"] > before["forward_transfer_delta"]
    assert after_metrics["backward_retention_delta"] == before["backward_retention_delta"]
    assert positive["passed"] is True
    assert (
        positive["before"]["backward_retention_delta"]
        == positive["after"]["backward_retention_delta"]
    )


def test_scenario_learn_5583_artifact_fields_and_stable_write(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5583-ARTIFACT: receipt exposes the retired gate."""

    destination = tmp_path / exp.RESULT_RELATIVE_PATH.name
    artifact = exp.run(
        root=REPO,
        result_path=destination,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=True,
    )
    no_write = exp.run(
        root=REPO,
        result_path=exp.RESULT_RELATIVE_PATH,
        tests_added_or_reused=TESTS_ADDED_OR_REUSED,
        write=False,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == artifact
    assert no_write["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert exp.validate_artifact(artifact) is True
    assert exp.resolve_path(REPO, destination) == destination
    assert exp.resolve_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in artifact
        assert artifact["field_principles"][field] == principle

    assert artifact["rows_reconstructed"]["complete"] is True
    assert artifact["forward_transfer_delta"] == 0.0
    assert artifact["backward_retention_delta"] > 0.0
    assert artifact["forgetting_delta"] > 0.0
    assert artifact["permutation_control_passed"] is True
    assert artifact["metric_independence_positive_control"]["passed"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["source_artifacts"]["exp5569"]["flagged_adversarial"] is True
    assert artifact["policy_ready"] is False
    assert artifact["policy_gate"]["policy_benefit_passed"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_reused"] == TESTS_ADDED_OR_REUSED
