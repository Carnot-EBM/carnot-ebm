"""Tests for Exp5644 exact two-axis temperature-by-penalty label exchange.

Spec refs: REQ-SAMPLE-5644, SCENARIO-SAMPLE-5644.
"""

from __future__ import annotations

from copy import deepcopy
import json
from math import exp
from pathlib import Path

import pytest

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5644_two_axis_parallel_tempering_exact_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5644_two_axis_parallel_tempering_exact_audit.py")


def test_req_sample_5644_spec_declares_two_axis_exact_contract() -> None:
    """REQ-SAMPLE-5644: OpenSpec anchors two-axis exactness and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5644") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "SCENARIO-SAMPLE-5644",
        "corrected_cdls_projection_mh",
        "swap parameter labels",
        "SHALL NOT copy or swap replica states",
        "exp(-beta * (E(x) + lambda * C(x)))",
        "min(1, exp((beta_a - beta_b)",
        "min(1, exp(beta * (lambda_a - lambda_b)",
        "missing penalty terms",
        "sign reversal",
        "state swapping",
        "asymmetric scheduling",
        "extreme lambda",
        "disabled swaps",
        "`timing_claimed` SHALL be false",
        "`hardware_speedup_claimed` SHALL be false",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5644_horizontal_and_vertical_swap_math_balances() -> None:
    """REQ-SAMPLE-5644: derived swap ratios match the joint target."""

    system = mod.constrained_ising_fixtures()[0]
    states = exp5622.enumerate_states(system.n_spins)
    slots = mod.parameter_slots()
    state_indices = (0, 1, 2, 3)
    labels = (0, 1, 2, 3)
    horizontal_pair = mod.horizontal_exchange_pairs()[0]["slot_pair"]
    vertical_pair = mod.vertical_exchange_pairs()[-1]["slot_pair"]
    energies = exp5622.energy_vector(system, states)
    penalties = mod.constraint_penalty_vector(system, states)

    h_left = mod.slot_by_id(slots, horizontal_pair[0])
    h_right = mod.slot_by_id(slots, horizontal_pair[1])
    h_left_pos = labels.index(h_left.slot_id)
    h_right_pos = labels.index(h_right.slot_id)
    h_energy_left = (
        energies[state_indices[h_left_pos]] + h_left.penalty * penalties[state_indices[h_left_pos]]
    )
    h_energy_right = (
        energies[state_indices[h_right_pos]]
        + h_right.penalty * penalties[state_indices[h_right_pos]]
    )
    expected_horizontal = min(
        1.0, exp((h_left.beta - h_right.beta) * (h_energy_left - h_energy_right))
    )

    v_left = mod.slot_by_id(slots, vertical_pair[0])
    v_right = mod.slot_by_id(slots, vertical_pair[1])
    v_left_pos = labels.index(v_left.slot_id)
    v_right_pos = labels.index(v_right.slot_id)
    expected_vertical = min(
        1.0,
        exp(
            v_left.beta
            * (v_left.penalty - v_right.penalty)
            * (penalties[state_indices[v_left_pos]] - penalties[state_indices[v_right_pos]])
        ),
    )

    assert h_left.penalty == h_right.penalty
    assert v_left.beta == v_right.beta
    assert mod.label_exchange_candidate(
        state_indices=state_indices, labels=labels, slot_pair=horizontal_pair
    )["proposed_state_indices"] == list(state_indices)
    assert mod.horizontal_swap_acceptance_probability(
        system=system,
        states=states,
        state_indices=state_indices,
        labels=labels,
        slot_pair=horizontal_pair,
    ) == pytest.approx(expected_horizontal)
    assert mod.vertical_swap_acceptance_probability(
        system=system,
        states=states,
        state_indices=state_indices,
        labels=labels,
        slot_pair=vertical_pair,
    ) == pytest.approx(expected_vertical)
    assert mod.horizontal_detailed_balance_residual(
        system=system, states=states, slot_pair=horizontal_pair
    ) <= (mod.DETAILED_BALANCE_TOLERANCE)
    assert mod.vertical_detailed_balance_residual(
        system=system, states=states, slot_pair=vertical_pair
    ) <= (mod.DETAILED_BALANCE_TOLERANCE)


def test_req_sample_5644_transition_exactness_and_reconstructable_fixtures() -> None:
    """REQ-SAMPLE-5644: fixtures and exact transition audits are reconstructable."""

    systems = mod.constrained_ising_fixtures()
    assert len(systems) >= 3

    for system in systems:
        states = exp5622.enumerate_states(system.n_spins)
        audit = mod.audit_one_system(system)

        assert audit["descriptor"]["barrier_summary"]["energy_level_count"] >= 3
        assert audit["descriptor"]["barrier_summary"]["penalty_level_count"] >= 3
        assert audit["transition_row_error"] <= mod.TRANSITION_ROW_TOLERANCE
        assert audit["transition_probability_min"] >= -mod.TRANSITION_ROW_TOLERANCE
        assert audit["horizontal_detailed_balance_error"] <= mod.DETAILED_BALANCE_TOLERANCE
        assert audit["vertical_detailed_balance_error"] <= mod.DETAILED_BALANCE_TOLERANCE
        assert audit["exact_joint_target_tv"] <= mod.EXACT_TV_TOLERANCE
        assert audit["exact_target_replica_tv"] <= mod.EXACT_TV_TOLERANCE
        assert audit["target_feasibility_marginal_error"] <= mod.FEASIBILITY_MARGINAL_TOLERANCE
        assert mod.target_feasibility_marginal(system, states, mod.target_slot()) == pytest.approx(
            audit["descriptor"]["target_slot"]["exact_feasibility_marginal"]
        )


def test_req_sample_5644_broken_controls_are_rejected() -> None:
    """REQ-SAMPLE-5644: every required invalid two-axis kernel is detected."""

    system = mod.constrained_ising_fixtures()[1]
    rows = mod.audit_broken_controls(system=system, states=exp5622.enumerate_states(system.n_spins))
    by_id = {row["control_id"]: row for row in rows}

    assert set(by_id) == set(mod.BROKEN_CONTROL_IDS)
    assert all(row["detected"] is True for row in rows)
    assert (
        by_id["missing_penalty_terms"]["max_detailed_balance_error"]
        > mod.DETAILED_BALANCE_TOLERANCE
    )
    assert by_id["sign_reversal"]["max_detailed_balance_error"] > mod.DETAILED_BALANCE_TOLERANCE
    assert by_id["state_swapping"]["state_mutation_detected"] is True
    assert (
        by_id["asymmetric_scheduling"]["max_detailed_balance_error"]
        > mod.DETAILED_BALANCE_TOLERANCE
    )
    assert (
        by_id["extreme_lambda"]["target_feasibility_marginal_delta"]
        > mod.FEASIBILITY_MARGINAL_TOLERANCE
    )
    assert by_id["disabled_swaps"]["scheduler_missing_required_swaps"] is True


def test_scenario_sample_5644_builds_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5644: exact audit writes gated JSON evidence."""

    artifact = mod.build_artifact(
        root=REPO,
        random_seeds=(5644, 5645, 5646),
        replay_sweeps=18,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["schema"] == mod.SCHEMA
    assert saved["openspec_requirement_ids"] == list(mod.SPEC_REFS)
    assert saved["temperature_ladder"] == list(mod.TEMPERATURE_LADDER)
    assert saved["penalty_ladder"] == list(mod.PENALTY_LADDER)
    assert saved["horizontal_swap_rule"]["state_update"] == "parameter_labels_only"
    assert saved["vertical_swap_rule"]["state_update"] == "parameter_labels_only"
    assert len(saved["fixture_definitions"]) >= 3
    assert saved["transition_row_error_max"] <= mod.TRANSITION_ROW_TOLERANCE
    assert saved["transition_probability_min"] >= -mod.TRANSITION_ROW_TOLERANCE
    assert saved["horizontal_detailed_balance_error_max"] <= mod.DETAILED_BALANCE_TOLERANCE
    assert saved["vertical_detailed_balance_error_max"] <= mod.DETAILED_BALANCE_TOLERANCE
    assert saved["exact_joint_target_tv"] <= mod.EXACT_TV_TOLERANCE
    assert saved["exact_target_replica_tv"] <= mod.EXACT_TV_TOLERANCE
    assert saved["target_feasibility_marginal_error"] <= mod.FEASIBILITY_MARGINAL_TOLERANCE
    assert saved["deterministic_replay_pass"] is True
    assert saved["broken_control_rejected"] is True
    assert all(row["detected"] is True for row in saved["broken_controls"])
    assert saved["exactness_comparators"]["promoted_one_axis_temperature_exchange"][
        "exact_distribution_tv_max"
    ] <= (mod.EXACT_TV_TOLERANCE)
    assert (
        saved["exactness_comparators"]["equal_transition_independent_corrected_chains"][
            "exact_distribution_tv_max"
        ]
        <= mod.EXACT_TV_TOLERANCE
    )
    assert saved["timing_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["two_axis_invariant_ready_score"] == 1.0
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seeds"] == [5644, 5645, 5646]
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5644_validation_fails_closed_on_manual_readiness() -> None:
    """REQ-SAMPLE-5644: readiness cannot be manually set past failed gates."""

    artifact = mod.build_artifact(root=REPO, random_seeds=(5644, 5645), replay_sweeps=12)
    mutations = [
        ("missing required field", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        (
            "openspec_requirement_ids",
            lambda data: data.__setitem__("openspec_requirement_ids", ["REQ-SAMPLE-0000"]),
        ),
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
            "deterministic_replay_pass",
            lambda data: data.__setitem__("deterministic_replay_pass", False),
        ),
        (
            "broken_controls",
            lambda data: data.__setitem__("broken_controls", data["broken_controls"][1:]),
        ),
        ("broken_controls", lambda data: data["broken_controls"][0].__setitem__("detected", False)),
        (
            "two_axis_invariant_ready_score",
            lambda data: data.__setitem__("two_axis_invariant_ready_score", 0.5),
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
        if expected not in {"reproducibility_checksum", "missing required field"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["exact_joint_target_tv"] = mod.EXACT_TV_TOLERANCE * 10.0
    blocked["two_axis_invariant_ready_score"] = mod.ready_score(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)

    assert blocked["two_axis_invariant_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(blocked)
