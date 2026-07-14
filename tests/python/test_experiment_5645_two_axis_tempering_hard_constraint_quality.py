"""Tests for Exp5645 two-axis hard-constraint quality comparison.

Spec refs: REQ-SAMPLE-5645, SCENARIO-SAMPLE-5645.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5645_two_axis_tempering_hard_constraint_quality as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5645_two_axis_tempering_hard_constraint_quality.py")


def test_req_sample_5645_spec_declares_preregistered_quality_gate() -> None:
    """REQ-SAMPLE-5645: OpenSpec freezes fields, controls, and promotion gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5645") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "SCENARIO-SAMPLE-5645",
        "two_axis_tempering",
        "one_axis_temperature_exchange",
        "independent_corrected_cdls",
        "at least five paired seeds",
        "disabled-penalty-swap",
        "collapsed-ladder",
        "shuffled-label",
        "fixed-weak-penalty",
        "fixed-strong-penalty",
        "invalid-state controls",
        "`timing_claimed` SHALL be false",
        "`hardware_speedup_claimed` SHALL be false",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5645_protocol_panel_and_budget_are_frozen() -> None:
    """REQ-SAMPLE-5645: families, ladders, seeds, controls, and budgets are preregistered."""

    panel = mod.frozen_instance_panel()
    protocol = mod.preregistered_protocol(mod.DEFAULT_RANDOM_SEEDS)
    hashes = mod.instance_hashes(panel)
    configs = mod.sampler_configs()
    budget = mod.transition_budget_parity(
        panel,
        mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=3,
        sample_sweeps=9,
    )

    assert len(panel) >= 4
    assert {item.family for item in panel} == {"frustrated_ising", "exact_verifier_csp"}
    assert len({item.size_stratum for item in panel}) >= 2
    assert all(item.preregistered for item in panel)
    assert all(item.penalty_definition.startswith("C(x)=") for item in panel)
    assert set(hashes) == {item.instance_id for item in panel}
    assert protocol["outcome_driven_tuning_excluded"] is True
    assert protocol["paired_seed_count"] >= mod.MIN_PAIRED_SEEDS
    assert protocol["control_ids"] == list(mod.CONTROL_IDS)
    assert set(configs) == set(mod.ARM_IDS)
    assert configs["two_axis_tempering"]["penalty_swaps_enabled"] is True
    assert configs["one_axis_temperature_exchange"]["penalty_swaps_enabled"] is False
    assert budget["budget_equal"] is True
    assert len(set(budget["within_replica_proposals_by_arm"].values())) == 1
    assert budget["swap_work_accounted_separately"] is True


def test_scenario_sample_5645_builds_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5645: paired hard-constraint trial emits valid evidence."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=4,
        sample_sweeps=16,
        tests_added_or_reused=[TEST_PATH.as_posix()],
        wall_clock=lambda: 5645.0,
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["schema"] == mod.SCHEMA
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["upstream_gate_receipts"]["exactness_eligibility_explicit"] is True
    assert saved["upstream_gate_receipts"]["exp5622"]["ready"] is True
    assert saved["upstream_gate_receipts"]["exp5634"]["ready"] is True
    assert saved["upstream_gate_receipts"]["exp5644"]["ready"] is True
    assert saved["preregistered_protocol"]["outcome_driven_tuning_excluded"] is True
    assert set(saved["instance_hashes"]) == {
        row["instance_id"] for row in saved["instance_manifest"]
    }
    assert set(saved["sampler_configs"]) == set(mod.ARM_IDS)
    assert saved["transition_budget_parity"]["budget_equal"] is True
    assert len(set(saved["transition_budget_parity"]["within_replica_proposals_by_arm"].values())) == 1
    assert saved["successful_seed_count"] == len(mod.DEFAULT_RANDOM_SEEDS)
    assert saved["failed_seed_reasons"] == []
    for field in (
        "constraint_validity_by_arm",
        "feasible_hit_rate_by_arm",
        "violation_distribution_by_arm",
        "first_feasible_transition_by_arm",
        "temperature_round_trips",
        "penalty_round_trips",
        "barrier_crossings_by_arm",
        "ess_by_arm",
        "autocorrelation_by_arm",
        "feasible_energy_by_arm",
        "solve_probability_by_arm",
    ):
        assert set(saved[field]) == set(mod.ARM_IDS)
    assert set(saved["paired_intervals"]) == {
        "two_axis_tempering_vs_one_axis_temperature_exchange",
        "two_axis_tempering_vs_independent_corrected_cdls",
    }
    assert set(saved["control_diagnostics"]) == set(mod.CONTROL_IDS)
    assert all(row["control_passed"] is True for row in saved["control_diagnostics"].values())
    assert saved["zero_solve_instances_by_arm"]
    assert saved["invalid_execution_count"] == 0
    assert saved["material_quality_regression_count"] >= 0
    assert saved["timing_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["two_axis_quality_ready_score"] == mod.quality_ready_score(saved)
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seeds"] == list(mod.DEFAULT_RANDOM_SEEDS)
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith(("complete:", "blocked:"))
    mod.validate_artifact(saved)


def test_req_sample_5645_quality_gate_requires_interval_improvement_without_regression() -> None:
    """REQ-SAMPLE-5645: the scalar gate cannot be manually promoted."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=3,
        sample_sweeps=12,
        wall_clock=lambda: 5645.0,
    )
    comparison = "two_axis_tempering_vs_one_axis_temperature_exchange"

    promotable = deepcopy(artifact)
    promotable["paired_intervals"][comparison]["feasible_hit_rate_delta_interval_95"] = [0.10, 0.20]
    promotable["paired_intervals"][comparison]["constraint_validity_delta_interval_95"] = [0.0, 0.0]
    promotable["paired_intervals"][comparison]["feasible_energy_improvement_interval_95"] = [
        0.0,
        0.15,
    ]
    promotable["material_quality_regression_count"] = 0
    promotable["two_axis_quality_ready_score"] = mod.quality_ready_score(promotable)
    promotable["honest_verdict"] = mod.honest_verdict(promotable)
    promotable["reproducibility_checksum"] = mod.payload_checksum(promotable)

    assert promotable["two_axis_quality_ready_score"] == 1.0
    assert promotable["honest_verdict"].startswith("complete:")
    mod.validate_artifact(promotable)

    regressed = deepcopy(promotable)
    regressed["paired_intervals"][comparison]["constraint_validity_delta_interval_95"] = [
        -0.20,
        -0.10,
    ]
    regressed["material_quality_regression_count"] = mod.material_quality_regression_count(regressed)
    regressed["two_axis_quality_ready_score"] = mod.quality_ready_score(regressed)
    regressed["honest_verdict"] = mod.honest_verdict(regressed)
    regressed["reproducibility_checksum"] = mod.payload_checksum(regressed)

    assert regressed["material_quality_regression_count"] > 0
    assert regressed["two_axis_quality_ready_score"] == 0.0
    assert regressed["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(regressed)


def test_req_sample_5645_validation_fails_closed_on_invalid_edits() -> None:
    """REQ-SAMPLE-5645: validation rejects missing fields, bad controls, and false claims."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=3,
        sample_sweeps=12,
        wall_clock=lambda: 5645.0,
    )
    mutations = [
        ("missing required field", lambda data: data.pop("instance_manifest")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        (
            "inference_substrate",
            lambda data: data.__setitem__("inference_substrate", "deterministic_verifier"),
        ),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        (
            "transition_budget_parity",
            lambda data: data["transition_budget_parity"].__setitem__("budget_equal", False),
        ),
        (
            "control_diagnostics",
            lambda data: data["control_diagnostics"]["invalid_state_control"].__setitem__(
                "control_passed", False
            ),
        ),
        (
            "two_axis_quality_ready_score",
            lambda data: data.__setitem__(
                "two_axis_quality_ready_score", 1.0 - mod.quality_ready_score(data)
            ),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"missing required field", "reproducibility_checksum"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)
