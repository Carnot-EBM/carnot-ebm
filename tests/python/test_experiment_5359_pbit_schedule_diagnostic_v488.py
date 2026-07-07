"""Tests for Exp 5359 CPU p-bit schedule diagnostic.

Spec refs: REQ-VERIFY-5359, SCENARIO-VERIFY-5359.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5359_pbit_schedule_diagnostic_v488 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _row(
    rows: list[dict[str, object]],
    *,
    schedule: str,
    instance_class: str,
) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["schedule_variant"] == schedule and row["instance_class"] == instance_class
    )


def test_req_verify_5359_spec_declares_schedule_diagnostic_contract() -> None:
    """REQ-VERIFY-5359: OpenSpec anchors the CPU schedule diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5359") : spec.index("### REQ-VERIFY-5345")
    ]

    for marker in (
        "REQ-VERIFY-5359",
        "SCENARIO-VERIFY-5359",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "baseline sequential",
        "partial deactivation",
        "fully parallel inertia",
        "cost-aware anneal",
        "misleading-assumption guard",
        "`hardware_speedup_claim` SHALL be false",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5359_reuses_bounded_pbit_cdcl_fixtures() -> None:
    """REQ-VERIFY-5359: instances come from bounded Exp 5292/5300 classes."""

    instances = mod.build_schedule_instances()
    by_class = {instance.instance_class: instance for instance in instances}

    assert mod.SCHEDULE_VARIANTS == (
        "baseline_sequential",
        "partial_deactivation",
        "fully_parallel_inertia",
        "cost_aware_anneal",
        "misleading_assumption_guard",
    )
    assert set(by_class) == {
        "aligned_factor_sat",
        "misleading_factor_sat",
        "neutral_factor_sat",
        "aligned_repair",
        "misleading_repair",
        "neutral_noop_repair",
        "malformed_control",
        "semantic_wrong_control",
    }
    assert {instance.source_experiment for instance in instances} == {"exp5292", "exp5299"}
    assert all(instance.source_fixture_id == "small_pair_sum" for instance in instances)
    assert all(instance.hardware_execution is False for instance in instances)
    assert by_class["aligned_factor_sat"].seed_literals == (3, 8)
    assert by_class["misleading_factor_sat"].seed_literals == (4, 7)
    assert by_class["semantic_wrong_control"].seed_literals == (2, 8)


def test_scenario_verify_5359_runs_all_schedules_with_solver_authority() -> None:
    """SCENARIO-VERIFY-5359: every schedule row stays CPU and CDCL-authoritative."""

    benchmark = mod.run_benchmark()
    rows = benchmark["per_schedule_results"]

    assert benchmark["schedule_variant_count"] == len(mod.SCHEDULE_VARIANTS)
    assert benchmark["fixture_count"] == len(mod.EXPECTED_INSTANCE_CLASSES)
    assert len(rows) == len(mod.SCHEDULE_VARIANTS) * len(mod.EXPECTED_INSTANCE_CLASSES)
    assert set(benchmark["schedule_summaries"]) == set(mod.SCHEDULE_VARIANTS)
    assert benchmark["correctness_preserved"] is True
    assert benchmark["false_accept_count"] == 0

    for row in rows:
        assert row["hardware_execution"] is False
        assert row["solver_authoritative"] is True
        assert row["false_accept"] is False
        assert isinstance(row["sweeps_to_solution"], int)
        assert isinstance(row["energy_trace"], list)
        assert row["energy_trace"]
        assert set(row["cdcl_metrics"]) == {
            "conflicts",
            "decisions",
            "propagations",
            "restarts",
            "wall_clock_s",
        }
        assert row["final_status"] == row["solver_only"]["status"]


def test_req_verify_5359_schedule_comparison_records_benefit_and_instability() -> None:
    """REQ-VERIFY-5359: deltas expose help, harm, and monotonicity violations."""

    benchmark = mod.run_benchmark()
    summaries = benchmark["schedule_summaries"]
    effects = benchmark["class_schedule_effects"]

    assert summaries["cost_aware_anneal"]["sweeps_to_solution_delta_vs_baseline"] > 0
    assert summaries["cost_aware_anneal"]["conflict_delta_vs_baseline"] > 0
    assert summaries["fully_parallel_inertia"]["energy_monotonicity_violation_count"] > 0
    assert summaries["misleading_assumption_guard"]["misleading_class_harm_count"] == 0
    assert benchmark["energy_monotonicity_violation_count"] > 0
    assert benchmark["sweeps_to_solution_delta"] > 0
    assert benchmark["conflict_delta_vs_baseline"] > 0
    assert benchmark["pbit_schedule_signal_ready"] is True

    assert "cost_aware_anneal" in effects["aligned_factor_sat"]["helps"]
    assert "fully_parallel_inertia" in effects["aligned_factor_sat"]["harms"]
    assert "misleading_assumption_guard" in effects["misleading_factor_sat"]["helps"]
    assert effects["neutral_factor_sat"]["inconclusive"]


def test_scenario_verify_5359_guard_blocks_misleading_classes_without_false_accepts() -> None:
    """SCENARIO-VERIFY-5359: guard routes misleading basins to CDCL fallback."""

    rows = mod.run_benchmark()["per_schedule_results"]

    for instance_class in mod.MISLEADING_CLASSES:
        row = _row(
            rows,
            schedule="misleading_assumption_guard",
            instance_class=instance_class,
        )

        assert row["route"] == "fallback_solver_only"
        assert row["final_model"] == row["solver_only"]["model"]
        assert row["cdcl_metrics"] == row["solver_only"]["metrics"]
        assert row["false_accept"] is False
        assert row["misleading_class_harm"] == 0


def test_req_verify_5359_artifact_schema_and_required_bare_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5359: artifact exposes required bare schedule metrics."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5359", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        cpu_runtime_s=0.42,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["schedule_variant_count"] == len(mod.SCHEDULE_VARIANTS)
    assert artifact["fixture_count"] == len(mod.EXPECTED_INSTANCE_CLASSES)
    assert isinstance(artifact["sweeps_to_solution_delta"], (int, float))
    assert isinstance(artifact["conflict_delta_vs_baseline"], (int, float))
    assert isinstance(artifact["misleading_class_harm_rate"], (int, float))
    assert isinstance(artifact["energy_monotonicity_violation_count"], int)
    assert artifact["cpu_runtime_s"] == 0.42
    assert artifact["false_accept_count"] == 0
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["pbit_schedule_signal_ready"] is True
    assert artifact["tests_run"] == tests_run
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert "REQ-VERIFY-5359" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5359_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5359: invalid substrate, speedup, or false accepts fail."""

    artifact = mod.build_artifact(
        cpu_runtime_s=0.1,
        tests_run=[{"command": "unit exp5359", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["hardware_speedup_claim"] = True
    with pytest.raises(AssertionError, match="hardware"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = "hardware_sampler"
    with pytest.raises(AssertionError, match="substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["false_accept_count"] = 1
    with pytest.raises(AssertionError, match="false accept"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["schedule_variant_count"] = 1
    with pytest.raises(AssertionError, match="schedule"):
        mod.validate_artifact(broken)


def test_req_verify_5359_defensive_branches_on_tiny_custom_fixture() -> None:
    """REQ-VERIFY-5359: tiny fixtures cover early exits and false-signal guards."""

    instance = mod.ScheduleInstance(
        instance_id="tiny_one_var",
        instance_class="neutral_factor_sat",
        source_experiment="unit",
        n_vars=1,
        clauses=((1,),),
        source_fixture_id="unit",
        source_artifact="unit",
        seed_literals=(),
        seed_method="unit",
        lns_repair_agreement="neutral",
        candidate_format_valid=True,
    )

    baseline = mod.simulate_schedule(instance, mod.BASELINE_VARIANT)
    partial = mod.simulate_schedule(instance, "partial_deactivation")
    solver_only = mod.cdcl.CdclRun(
        status="sat",
        model=(1,),
        metrics={
            "conflicts": 0,
            "decisions": 1,
            "propagations": 1,
            "restarts": 1,
            "wall_clock_s": 0.0,
        },
    )
    bad_proposal = mod.ScheduleProposal(
        schedule_variant="unit",
        final_state=(False,),
        energy_trace=(1,),
        sweeps_to_solution=1,
        solution_found=False,
    )
    summaries = {
        schedule: {
            "conflict_delta_vs_baseline": 0,
            "misleading_class_harm_count": 0,
            "false_accept_count": 0,
        }
        for schedule in mod.SCHEDULE_VARIANTS
    }
    effects = {
        instance_class: {"helps": [], "harms": [], "inconclusive": []}
        for instance_class in mod.EXPECTED_INSTANCE_CLASSES
    }

    assert baseline.energy_trace[-1] == 0
    assert partial.energy_trace[-1] == 0
    assert mod._false_accept(
        instance,
        solver_only,
        final_status="unsat",
        final_model=(),
        proposal=bad_proposal,
        route="fallback_solver_only",
    )
    assert mod._false_accept(
        instance,
        solver_only,
        final_status="sat",
        final_model=(-1,),
        proposal=bad_proposal,
        route="fallback_solver_only",
    )
    assert not mod._pbit_schedule_signal_ready(
        summaries,
        effects,
        false_accept_count=1,
    )
    assert not mod._pbit_schedule_signal_ready(
        summaries,
        effects,
        false_accept_count=0,
    )


def test_deliverable_file_validates_for_scenario_verify_5359() -> None:
    """SCENARIO-VERIFY-5359: checked-in deliverable satisfies the V488 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["false_accept_count"] == 0
    assert artifact["pbit_schedule_signal_ready"] is True
