"""Tests for the matched-budget multiscale sampler benchmark.

Spec refs: REQ-SAMPLER-7134, REQ-SAMPLER-7134-GATE,
REQ-SAMPLER-7134-DESIGN, REQ-SAMPLER-7134-BUDGET,
REQ-SAMPLER-7134-CHAINS, REQ-SAMPLER-7134-PLAN,
REQ-SAMPLER-7134-METRICS, REQ-SAMPLER-7134-CLAIMS,
REQ-SAMPLER-7134-ARTIFACT, REQ-SAMPLER-7134-BOUNDARY,
SCENARIO-SAMPLER-7134-GATE, SCENARIO-SAMPLER-7134-MATCHED-BUDGET,
SCENARIO-SAMPLER-7134-FINITE-PARITY, SCENARIO-SAMPLER-7134-FAILURES,
SCENARIO-SAMPLER-7134-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7134_v626_multiscale_sampler_benchmark as exp


REPO = Path(__file__).resolve().parents[2]


def _rehash(payload: dict) -> dict:
    payload["reproducibility_checksum"] = exp.artifact_checksum(payload)
    return payload


@pytest.fixture(scope="module")
def benchmark_artifact() -> dict:
    """Build the full benchmark once without writing tracked evidence."""

    return exp.build_artifact(root=REPO, run_date="20260908")


def test_req_sampler_7134_spec_precedes_implementation() -> None:
    """REQ-SAMPLER-7134 fixes the gate, budget, plan, rows, and scope."""

    text = (REPO / exp.SAMPLER_SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-SAMPLER-7134", 1)[1]
    for anchor in (
        "REQ-SAMPLER-7134-GATE",
        "REQ-SAMPLER-7134-DESIGN",
        "REQ-SAMPLER-7134-BUDGET",
        "REQ-SAMPLER-7134-CHAINS",
        "REQ-SAMPLER-7134-PLAN",
        "REQ-SAMPLER-7134-METRICS",
        "REQ-SAMPLER-7134-CLAIMS",
        "REQ-SAMPLER-7134-ARTIFACT",
        "REQ-SAMPLER-7134-BOUNDARY",
        "SCENARIO-SAMPLER-7134-GATE",
        "SCENARIO-SAMPLER-7134-MATCHED-BUDGET",
        "SCENARIO-SAMPLER-7134-FINITE-PARITY",
        "SCENARIO-SAMPLER-7134-FAILURES",
        "SCENARIO-SAMPLER-7134-ARTIFACT",
    ):
        assert anchor in section


def test_req_sampler_7134_design_freezes_seeds_cells_and_plan() -> None:
    """REQ-SAMPLER-7134-DESIGN covers a fixed frustration-temperature grid."""

    cells = exp.frozen_cells()
    plan = exp.frozen_analysis_plan()
    assert len(exp.FROZEN_SEEDS) >= 5
    assert len(set(exp.FROZEN_SEEDS)) == len(exp.FROZEN_SEEDS)
    assert len(cells) == 8
    assert {cell.size for cell in cells} == {2, 4}
    assert {cell.frustration for cell in cells} == {"low", "high"}
    assert {cell.temperature for cell in cells} == {0.8, 1.6}
    assert {cell.enumerated for cell in cells} == {False, True}
    assert len({cell.fixture_hash for cell in cells}) == len(cells)
    assert plan == exp.frozen_analysis_plan()
    assert plan.observables == ("energy", "magnetization", "positive_mode_indicator")
    assert plan.burn_in_steps > 0
    assert plan.thinning > 0
    assert plan.lag_window > 0
    assert plan.energy_evaluation_budget % 2 == 0
    for cell in cells:
        receipt = exp.validate_cell(cell)
        assert receipt["passed"] is True
        assert receipt["frustrated_plaquettes"] > 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"size": 3}, "size"),
        ({"temperature": 0.0}, "temperature"),
        ({"fields": (0.0,)}, "field count"),
        ({"fields": (float("nan"),) * 4}, "finite"),
        ({"edges": ((0, 0, 1.0),)}, "self-loop"),
        ({"edges": ((0, 1, 1.0), (1, 0, 1.0))}, "duplicate"),
        ({"edges": ((0, 9, 1.0),)}, "endpoint"),
        ({"edges": ((0, 1, float("nan")),)}, "finite"),
        ({"edges": ((0, 1, 1.0),)}, "edge set"),
    ],
)
def test_req_sampler_7134_invalid_cells_fail_closed(change: dict, message: str) -> None:
    """REQ-SAMPLER-7134-DESIGN rejects cells outside the frozen lattice scope."""

    changed = exp.replace_cell(exp.frozen_cells()[0], **change)
    with pytest.raises(ValueError, match=message):
        exp.validate_cell(changed)


def test_scenario_sampler_7134_gate_rechecks_exact_producer_fields() -> None:
    """SCENARIO-SAMPLER-7134-GATE checks readiness and producer code identity."""

    checks = exp.collect_preconditions(REPO)
    assert all(row["available"] is True for row in checks)
    by_resource = {row["resource"]: row for row in checks}
    assert by_resource["multiscale_sampler_ready_score"]["expected_value"] == 1
    assert by_resource["multiscale_sampler_ready_score"]["observed_value"] == 1
    assert by_resource["prototype_code_hash"]["observed_value"] == by_resource[
        "prototype_code_hash"
    ]["expected_value"]
    assert by_resource["sealed_finite_laws"]["available"] is True
    assert by_resource["fixed_seeds"]["observed_value"] == list(exp.FROZEN_SEEDS)


def test_scenario_sampler_7134_gate_failure_is_terminal_and_names_producer_field() -> None:
    """SCENARIO-SAMPLER-7134-GATE writes exact blocked evidence before chains."""

    checks = exp.collect_preconditions(REPO)
    index = next(
        idx for idx, row in enumerate(checks) if row["resource"] == "multiscale_sampler_ready_score"
    )
    checks[index] = {
        **checks[index],
        "available": False,
        "observed_value": 0,
    }
    artifact = exp.build_artifact(root=REPO, run_date="20260908", preconditions=checks)
    assert exp.validate_artifact(artifact) == []
    assert artifact["rows"] == artifact["chain_rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"] == {
        "failed_check": "multiscale_sampler_ready_score",
        "producer_field": "multiscale_sampler_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "passed": False,
    }


def test_req_sampler_7134_energy_counter_and_corrected_arms_match_budget() -> None:
    """REQ-SAMPLER-7134-BUDGET charges exact energies for every arm."""

    cell = exp.frozen_cells()[0]
    plan = exp.frozen_analysis_plan()
    for arm in exp.eligible_arms(cell):
        result = exp.run_arm(cell, arm=arm, seed=exp.FROZEN_SEEDS[0], plan=plan)
        assert result["status"] == "complete"
        assert result["energy_evaluations"] == plan.energy_evaluation_budget
        assert result["sample_count"] > plan.lag_window
        assert len(result["states"]) == len(result["energies"]) == result["sample_count"]
        if arm == "multiscale_mh":
            assert result["uses_mh_correction"] is True
            assert result["uses_forward_probability"] is True
            assert result["uses_reverse_probability"] is True
        if arm == "exact_law":
            assert result["reference_source"] == "independent_exp6657_enumerator"
    large = next(cell for cell in exp.frozen_cells() if not cell.enumerated)
    assert exp.eligible_arms(large) == ("multiscale_mh", "local_gibbs")
    with pytest.raises(ValueError, match="exact_law"):
        exp.run_arm(large, arm="exact_law", seed=exp.FROZEN_SEEDS[0], plan=plan)
    with pytest.raises(ValueError, match="arm"):
        exp.run_arm(cell, arm="unknown", seed=exp.FROZEN_SEEDS[0], plan=plan)


def test_req_sampler_7134_replay_metrics_and_autocorrelation_are_complete() -> None:
    """REQ-SAMPLER-7134-METRICS retains the frozen lag window and observables."""

    cell = exp.frozen_cells()[0]
    plan = exp.frozen_analysis_plan()
    first = exp.run_arm(cell, arm="multiscale_mh", seed=exp.FROZEN_SEEDS[1], plan=plan)
    replay = exp.run_arm(cell, arm="multiscale_mh", seed=exp.FROZEN_SEEDS[1], plan=plan)
    assert first["trace_sha256"] == replay["trace_sha256"]
    metrics = exp.compute_metrics(cell, "multiscale_mh", exp.FROZEN_SEEDS[1], first, plan)
    assert len(metrics["effective_sample_size_rows"]) == len(plan.observables)
    assert len(metrics["autocorrelation_rows"]) == len(plan.observables)
    for row in metrics["autocorrelation_rows"]:
        assert len(row["autocorrelations"]) == plan.lag_window + 1
        assert row["autocorrelations"][0] == pytest.approx(1.0)
        assert 1 <= row["integration_stop_lag"] <= plan.lag_window
        assert row["integrated_autocorrelation"] >= 1.0
    assert 0.0 <= metrics["total_variation_rows"][0]["total_variation"] <= 1.0
    assert math.isfinite(metrics["energy_moment_rows"][0]["mean_energy"])
    assert sum(
        metrics["mode_occupancy_rows"][0][name]
        for name in ("negative_fraction", "zero_fraction", "positive_fraction")
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lag window"):
        exp.autocorrelation_series([1.0, 2.0], 2)
    constant = exp.autocorrelation_series([1.0] * 8, 3)
    assert constant == [1.0, 0.0, 0.0, 0.0]


def test_scenario_sampler_7134_artifact_has_complete_per_unit_evidence(
    benchmark_artifact: dict,
) -> None:
    """SCENARIO-SAMPLER-7134-ARTIFACT keeps every metric linked to one unit."""

    assert exp.validate_artifact(benchmark_artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(benchmark_artifact)
    assert set(benchmark_artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(benchmark_artifact["field_principles"].values())
    expected_units = sum(
        len(exp.FROZEN_SEEDS) * len(exp.eligible_arms(cell)) for cell in exp.frozen_cells()
    )
    assert expected_units == 100
    assert len(benchmark_artifact["rows"]) == expected_units
    assert len(benchmark_artifact["chain_rows"]) == expected_units
    for field in (
        "budget_rows",
        "acceptance_rows",
        "total_variation_rows",
        "energy_moment_rows",
        "mode_occupancy_rows",
        "wall_time_rows",
        "failure_rows",
    ):
        assert len(benchmark_artifact[field]) == expected_units
    observables = len(exp.frozen_analysis_plan().observables)
    assert len(benchmark_artifact["effective_sample_size_rows"]) == expected_units * observables
    assert len(benchmark_artifact["autocorrelation_rows"]) == expected_units * observables
    assert benchmark_artifact["matched_budget_verified"] is True
    assert benchmark_artifact["finite_parity_verified"] is True
    assert benchmark_artifact["hardware_execution_claimed"] is False
    assert benchmark_artifact["asymptotic_scaling_claimed"] is False
    assert benchmark_artifact["sampler_benchmark_complete_score"] == 1
    assert benchmark_artifact["verdict_class"] in {"positive", "null"}
    assert benchmark_artifact["honest_verdict"].startswith(("complete:", "null:"))
    assert benchmark_artifact["gate_check_summary"]["passed"] is True
    assert benchmark_artifact["duration_s"] > 0.0


def test_scenario_sampler_7134_finite_and_large_scope_is_explicit(
    benchmark_artifact: dict,
) -> None:
    """SCENARIO-SAMPLER-7134-FINITE-PARITY limits exact laws to finite cells."""

    conditions = {row["cell_id"]: row for row in benchmark_artifact["condition_rows"]}
    for row in benchmark_artifact["arm_rows"]:
        condition = conditions[row["cell_id"]]
        assert (row["arm"] == "exact_law") is condition["enumerated"] or row[
            "arm"
        ] != "exact_law"
    for row in benchmark_artifact["total_variation_rows"]:
        if conditions[row["cell_id"]]["enumerated"]:
            assert row["total_variation"] is not None
        else:
            assert row["arm"] != "exact_law"
            assert row["total_variation"] is None
    assert benchmark_artifact["scope"] == {
        "language": "Python",
        "venue": "host",
        "bounded_sizes": [2, 4],
        "rust_parity": False,
        "fpga_execution": False,
        "tsu_execution": False,
        "power_claim": False,
        "logarithmic_scaling_inference": False,
    }


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda row: row.pop("prototype_hash"), "missing_required_fields"),
        (lambda row: row["field_principles"].pop("rows"), "field_principles_invalid"),
        (
            lambda row: row["budget_rows"][0].update(energy_evaluations=1),
            "budget_mismatch",
        ),
        (
            lambda row: row["analysis_plan"].update(burn_in_steps=0),
            "analysis_plan_drift",
        ),
        (lambda row: row["rows"][0].update(seed="pooled"), "seed_pooling_or_row_loss"),
        (
            lambda row: row["autocorrelation_rows"][0]["autocorrelations"].pop(),
            "autocorrelation_truncated",
        ),
        (lambda row: row["failure_rows"].pop(), "failure_rows_incomplete"),
        (
            lambda row: next(
                item for item in row["chain_rows"] if item["arm"] == "multiscale_mh"
            ).update(uses_mh_correction=False),
            "uncorrected_multiscale_proposal",
        ),
        (lambda row: row.update(hardware_execution_claimed=True), "claim_boundary_invalid"),
        (lambda row: row.update(asymptotic_scaling_claimed=True), "claim_boundary_invalid"),
        (lambda row: row.update(execution_venue="fpga"), "claim_boundary_invalid"),
        (
            lambda row: row["pooled_comparison"].update(mean_paired_ess_delta=999.0),
            "pooled_claim_mismatch",
        ),
        (lambda row: row.update(sampler_benchmark_complete_score=0), "completion_score_invalid"),
        (lambda row: row.update(verdict_class="partial"), "verdict_invalid"),
        (lambda row: row.update(honest_verdict="positive without prefix"), "verdict_invalid"),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "checksum_mismatch"),
    ],
)
def test_req_sampler_7134_validator_rejects_preregistered_attacks(
    benchmark_artifact: dict, mutator: object, error: str
) -> None:
    """REQ-SAMPLER-7134-ARTIFACT detects every required evidence mutation."""

    changed = deepcopy(benchmark_artifact)
    mutator(changed)  # type: ignore[operator]
    if error != "checksum_mismatch":
        _rehash(changed)
    assert error in exp.validate_artifact(changed)


def test_scenario_sampler_7134_failed_units_remain_rows() -> None:
    """SCENARIO-SAMPLER-7134-FAILURES preserves a simulated chain failure."""

    unit = exp.unit_id(exp.frozen_cells()[0].cell_id, exp.FROZEN_SEEDS[0], "local_gibbs")
    artifact = exp.build_artifact(
        root=REPO,
        run_date="20260908",
        forced_failures={unit: "simulated_failure_for_policy_test"},
    )
    assert exp.validate_artifact(artifact) == []
    chain = next(row for row in artifact["chain_rows"] if row["unit_id"] == unit)
    failure = next(row for row in artifact["failure_rows"] if row["unit_id"] == unit)
    assert chain["status"] == "failed"
    assert failure["failed"] is True
    assert failure["failure_reason"] == "simulated_failure_for_policy_test"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["sampler_benchmark_complete_score"] == 0


def test_req_sampler_7134_atomic_writer_and_cli_use_redirected_paths(
    benchmark_artifact: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-SAMPLER-7134-ARTIFACT writes only the selected complete JSON."""

    output = tmp_path / "artifact.json"
    receipt = exp.write_json_atomic(output, benchmark_artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == benchmark_artifact
    assert receipt["atomic_replace"] is True
    assert receipt["sha256"] == exp.sha256_file(output)
    assert not list(tmp_path.glob("*.tmp"))
    assert exp.main(["--validate", str(output)]) == 0
    assert "validated" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert exp.main(["--validate", str(bad)]) == 2
    assert "artifact_read_error" in capsys.readouterr().out
    generated = tmp_path / "generated.json"
    assert exp.main(["--date", "20260908", "--output", str(generated)]) in {0, 2}
    emitted = json.loads(generated.read_text(encoding="utf-8"))
    assert exp.validate_artifact(emitted) == []
    assert "sampler_benchmark_complete_score" in capsys.readouterr().out
