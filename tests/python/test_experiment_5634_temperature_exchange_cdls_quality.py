"""Tests for Exp 5634 paired temperature-exchange cDLS quality trial.

Spec refs: REQ-SAMPLE-5634, SCENARIO-SAMPLE-5634.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5634_temperature_exchange_cdls_quality as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5634_temperature_exchange_cdls_quality.py")


def test_req_sample_5634_spec_declares_quality_trial_contract() -> None:
    """REQ-SAMPLE-5634: OpenSpec anchors the panel, controls, gates, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5634") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-SAMPLE-5634",
        str(mod.RESULT_RELATIVE_PATH),
        "frustrated Ising instances",
        "exact-verifier CSP encodings",
        "at least five paired seeds",
        "fixed total corrected-kernel transition budget",
        "temperature_exchange_cdls",
        "independent_corrected_cdls_replicas",
        "single_corrected_cold_chain",
        "beta-ladder ablation",
        "disabled-exchange control",
        "label-shuffle diagnostic",
        "transition-budget audit",
        "seed-order permutation",
        "exact-verifier replay",
        "`hardware_speedup_claimed` SHALL be false",
        "`timing_claimed` SHALL be false",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5634_frozen_panel_and_paired_schedule_are_preregistered() -> None:
    """REQ-SAMPLE-5634: hard instances, strata, seeds, and arms are frozen."""

    panel = mod.frozen_instance_panel()
    schedule = mod.paired_seed_schedule(mod.DEFAULT_RANDOM_SEEDS)

    assert len(panel) >= 4
    assert {item.family for item in panel} == {"frustrated_ising", "exact_verifier_csp"}
    assert len({item.size_stratum for item in panel}) >= 2
    assert all(item.preregistered is True for item in panel)
    assert all(item.barrier_description for item in panel)
    assert len(schedule["paired_seeds"]) >= mod.MIN_PAIRED_SEEDS
    assert schedule["seed_order_locked"] is True
    assert schedule["arms"] == list(mod.ARM_IDS)
    assert schedule["burn_in_rule"] == "fixed_before_treatment_result"


def test_scenario_sample_5634_builds_valid_quality_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5634: paired trial emits terminal quality/mixing evidence."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=4,
        sample_sweeps=16,
        tests_added_or_reused=[TEST_PATH.as_posix()],
        wall_clock=lambda: 5634.0,
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["schema"] == mod.SCHEMA
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["upstream_gate_receipts"]["exp5622"]["ready"] is True
    assert saved["upstream_gate_receipts"]["exp5633"]["ready"] is True
    assert saved["paired_seed_schedule"]["paired_seed_count"] == len(mod.DEFAULT_RANDOM_SEEDS)
    assert saved["transition_budget_receipt"]["budget_equal"] is True
    assert (
        len(set(saved["transition_budget_receipt"]["corrected_kernel_transitions_by_arm"].values()))
        == 1
    )
    assert len(set(saved["transition_budget_receipt"]["cold_target_samples_by_arm"].values())) == 1
    assert (
        len(set(saved["transition_budget_receipt"]["exact_validation_calls_by_arm"].values())) == 1
    )
    assert {row["arm_id"] for row in saved["method_arms"]} == set(mod.ARM_IDS)
    assert set(saved["round_trip_stats"]) == set(mod.ARM_IDS)
    assert saved["round_trip_stats"]["temperature_exchange_cdls"]["exchange_attempts"] > 0
    assert saved["round_trip_stats"]["temperature_exchange_cdls"]["accepted_exchanges"] >= 0
    assert set(saved["barrier_crossing_stats"]) == set(mod.ARM_IDS)
    assert set(saved["ess_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["autocorrelation_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["energy_distribution_diagnostics"]) == set(mod.ARM_IDS)
    assert set(saved["best_energy_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["mean_energy_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["solve_probability_by_arm"]) == set(mod.ARM_IDS)
    assert set(saved["exact_valid_rate_by_arm"]) == set(mod.ARM_IDS)
    assert saved["control_diagnostics"]["beta_ladder_ablation"]["control_passed"] is True
    assert saved["control_diagnostics"]["disabled_exchange_control"]["control_passed"] is True
    assert saved["control_diagnostics"]["label_shuffle_diagnostic"]["control_passed"] is True
    assert saved["control_diagnostics"]["transition_budget_audit"]["control_passed"] is True
    assert saved["control_diagnostics"]["seed_order_permutation"]["control_passed"] is True
    assert saved["control_diagnostics"]["exact_verifier_replay"]["control_passed"] is True
    assert saved["wall_time_provenance_only"]["speedup_computed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["timing_claimed"] is False
    assert saved["quality_mixing_ready"] is mod.promotion_gate(saved)
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seeds"] == list(mod.DEFAULT_RANDOM_SEEDS)
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith(("complete:", "blocked:"))
    mod.validate_artifact(saved)


def test_req_sample_5634_promotion_gate_requires_improvement_and_no_regression() -> None:
    """REQ-SAMPLE-5634: quality_mixing_ready is mechanical, not manually set."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=3,
        sample_sweeps=12,
        wall_clock=lambda: 5634.0,
    )
    promotable = deepcopy(artifact)
    promotable["paired_deltas_and_intervals"][
        "temperature_exchange_cdls_vs_independent_corrected_cdls_replicas"
    ]["barrier_crossings_delta_interval_95"] = [0.25, 0.5]
    promotable["paired_deltas_and_intervals"][
        "temperature_exchange_cdls_vs_single_corrected_cold_chain"
    ]["barrier_crossings_delta_interval_95"] = [0.2, 0.45]
    promotable["quality_mixing_ready"] = mod.promotion_gate(promotable)
    promotable["honest_verdict"] = mod.honest_verdict(promotable)
    promotable["reproducibility_checksum"] = mod.payload_checksum(promotable)

    assert promotable["quality_mixing_ready"] is True
    assert promotable["honest_verdict"].startswith("complete:")
    mod.validate_artifact(promotable)

    regressed = deepcopy(promotable)
    regressed["paired_deltas_and_intervals"][
        "temperature_exchange_cdls_vs_single_corrected_cold_chain"
    ]["exact_valid_rate_delta_interval_95"] = [-0.2, -0.1]
    regressed["quality_mixing_ready"] = mod.promotion_gate(regressed)
    regressed["honest_verdict"] = mod.honest_verdict(regressed)
    regressed["reproducibility_checksum"] = mod.payload_checksum(regressed)

    assert regressed["quality_mixing_ready"] is False
    assert regressed["honest_verdict"].startswith("complete:")
    mod.validate_artifact(regressed)


def test_req_sample_5634_validation_fails_closed_on_manual_or_invalid_edits() -> None:
    """REQ-SAMPLE-5634: validation rejects missing fields, timing claims, and bad budgets."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=3,
        sample_sweeps=12,
        wall_clock=lambda: 5634.0,
    )
    mutations = [
        ("missing required field", lambda data: data.pop("instance_panel")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        (
            "inference_substrate",
            lambda data: data.__setitem__("inference_substrate", "deterministic_verifier"),
        ),
        (
            "hardware_speedup_claimed",
            lambda data: data.__setitem__("hardware_speedup_claimed", True),
        ),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        (
            "transition_budget_receipt",
            lambda data: data["transition_budget_receipt"].__setitem__("budget_equal", False),
        ),
        (
            "quality_mixing_ready",
            lambda data: data.__setitem__("quality_mixing_ready", not mod.promotion_gate(data)),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "reproducibility_checksum",
            lambda data: data.__setitem__("reproducibility_checksum", "bad"),
        ),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"missing required field", "reproducibility_checksum"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_sample_5634_closed_gate_helper_edges(tmp_path: Path) -> None:
    """REQ-SAMPLE-5634: helper edge cases stay deterministic and fail closed."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=mod.DEFAULT_RANDOM_SEEDS,
        burn_in_sweeps=3,
        sample_sweeps=12,
        wall_clock=lambda: 5634.0,
    )

    assert mod._autocorrelation_time([2.0, 2.0]) == 1.0
    assert mod._interval_95([3.0]) == [3.0, 3.0]
    assert (
        mod._one_upstream_receipt(
            tmp_path / "missing.json", validator=lambda payload: None, ready_field="ready"
        )["ready"]
        is False
    )

    missing_intervals = deepcopy(artifact)
    missing_intervals["paired_deltas_and_intervals"] = []
    assert mod.promotion_gate(missing_intervals) is False

    timing_claim = deepcopy(artifact)
    timing_claim["timing_claimed"] = True
    assert mod.promotion_gate(timing_claim) is False

    bad_receipts = deepcopy(artifact)
    bad_receipts["upstream_gate_receipts"]["exp5633"]["ready"] = False
    assert mod.promotion_gate(bad_receipts) is False

    bad_budget = deepcopy(artifact)
    bad_budget["transition_budget_receipt"]["budget_equal"] = False
    assert mod.promotion_gate(bad_budget) is False

    bad_target = deepcopy(artifact)
    bad_target["target_diagnostics_within_exp5633_bounds"] = False
    assert mod.promotion_gate(bad_target) is False
