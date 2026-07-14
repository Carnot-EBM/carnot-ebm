"""Tests for Exp 5633 exact temperature-label exchange cDLS audit.

Spec refs: REQ-SAMPLE-5633, SCENARIO-SAMPLE-5633.
"""

from __future__ import annotations

from copy import deepcopy
import json
from math import exp
from pathlib import Path

import pytest

from carnot import experiment_5622_cdls_exact_kernel_audit as exp5622
from carnot import experiment_5633_temperature_exchange_cdls_exact_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5633_temperature_exchange_cdls_exact_audit.py")


def test_req_sample_5633_spec_declares_exact_label_exchange_contract() -> None:
    """REQ-SAMPLE-5633: OpenSpec anchors fixed ladder, label swaps, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5633") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-SAMPLE-5633",
        str(mod.RESULT_RELATIVE_PATH),
        "corrected_cdls_projection_mh",
        "source module/hash receipt",
        "swap temperature labels",
        "min(1, exp((beta_a - beta_b) * (E(x_a) - E(x_b))))",
        "equal-transition single-chain",
        "independent-replica no-exchange",
        "missing beta factors",
        "wrong energy sign",
        "state-copy swap",
        "asynchronous stale energy",
        "one-way exchange",
        "biased proposal schedules",
        "`timing_claimed` SHALL be false",
        "`hardware_speedup_claimed` SHALL be false",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5633_corrected_kernel_receipt_pins_exp5622_source() -> None:
    """REQ-SAMPLE-5633: corrected cDLS substrate is inherited unchanged."""

    receipt = mod.corrected_kernel_receipt(REPO)

    assert receipt["final_kernel"] == "corrected_cdls_projection_mh"
    assert receipt["source_path"] == mod.CORRECTED_KERNEL_SOURCE_RELATIVE_PATH.as_posix()
    assert receipt["source_sha256"] == mod.file_sha256(REPO / receipt["source_path"])
    assert receipt["result_path"] == exp5622.RESULT_RELATIVE_PATH.as_posix()
    assert receipt["proposal_std"] == exp5622.CDLS_PROPOSAL_STD
    assert receipt["drift_scale"] == exp5622.CDLS_DRIFT_SCALE
    assert receipt["continuous_bound"] == exp5622.CDLS_CONTINUOUS_BOUND
    assert receipt["large_n_timing_tuned"] is False
    assert receipt["substrate_unchanged"] is True


def test_req_sample_5633_label_exchange_preserves_states_and_balances() -> None:
    """REQ-SAMPLE-5633: exchange swaps labels only and satisfies detailed balance."""

    system = mod.enumerable_frustrated_systems()[0]
    states = exp5622.enumerate_states(system.n_spins)
    state_indices = (0, 3, 7)
    labels = (0, 1, 2)
    label_pair = (1, 2)
    move = mod.label_exchange_candidate(
        state_indices=state_indices,
        labels=labels,
        label_pair=label_pair,
    )

    beta_left = mod.BETA_LADDER[label_pair[0]]
    beta_right = mod.BETA_LADDER[label_pair[1]]
    energy_left = float(exp5622.energy_vector(system, states[[state_indices[1]]])[0])
    energy_right = float(exp5622.energy_vector(system, states[[state_indices[2]]])[0])
    expected_acceptance = min(1.0, exp((beta_left - beta_right) * (energy_left - energy_right)))

    assert move["proposed_state_indices"] == list(state_indices)
    assert move["proposed_labels"] == [0, 2, 1]
    assert sorted(move["proposed_labels"]) == sorted(labels)
    assert mod.swap_acceptance_probability(
        system=system,
        states=states,
        state_indices=state_indices,
        labels=labels,
        label_pair=label_pair,
        variant="correct",
    ) == pytest.approx(expected_acceptance)

    correct_residual = mod.swap_detailed_balance_residual(
        system=system,
        states=states,
        beta_ladder=mod.BETA_LADDER,
        label_pair=label_pair,
        variant="correct",
    )
    assert correct_residual <= mod.SWAP_DETAILED_BALANCE_TOLERANCE


def test_req_sample_5633_broken_exchange_controls_are_rejected() -> None:
    """REQ-SAMPLE-5633: every required broken control is detected."""

    system = mod.enumerable_frustrated_systems()[0]
    rows = mod.audit_broken_controls(
        system=system,
        states=exp5622.enumerate_states(system.n_spins),
        beta_ladder=mod.BETA_LADDER,
    )
    by_id = {row["control_id"]: row for row in rows}

    assert set(by_id) == set(mod.BROKEN_CONTROL_IDS)
    assert all(row["detected"] is True for row in rows)
    assert by_id["missing_beta_factors"]["swap_detailed_balance_residual"] > mod.SWAP_DETAILED_BALANCE_TOLERANCE
    assert by_id["wrong_energy_sign"]["swap_detailed_balance_residual"] > mod.SWAP_DETAILED_BALANCE_TOLERANCE
    assert by_id["state_copy_swap"]["state_mutation_detected"] is True
    assert by_id["asynchronous_stale_energy"]["swap_detailed_balance_residual"] > mod.SWAP_DETAILED_BALANCE_TOLERANCE
    assert by_id["one_way_exchange"]["swap_detailed_balance_residual"] > mod.SWAP_DETAILED_BALANCE_TOLERANCE
    assert by_id["biased_proposal_schedule"]["swap_detailed_balance_residual"] > mod.SWAP_DETAILED_BALANCE_TOLERANCE


def test_scenario_sample_5633_builds_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5633: exact audit writes gated JSON evidence."""

    artifact = mod.build_artifact(
        root=REPO,
        replay_sweeps=24,
        random_seeds=(5633, 5634, 5635),
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["schema"] == mod.SCHEMA
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert saved["corrected_kernel_receipt"]["substrate_unchanged"] is True
    assert saved["beta_ladder"] == list(mod.BETA_LADDER)
    assert saved["within_replica_schedule"] == mod.within_replica_schedule()
    assert saved["exchange_schedule"] == mod.exchange_schedule()
    assert saved["swap_rule"]["state_update"] == "temperature_labels_only"
    assert len(saved["enumerable_targets"]) >= 2
    assert saved["transition_normalization_error_max"] <= mod.TRANSITION_NORMALIZATION_TOLERANCE
    assert saved["swap_detailed_balance_residual_max"] <= mod.SWAP_DETAILED_BALANCE_TOLERANCE
    assert saved["exact_distribution_tv_max"] <= mod.EXACT_DISTRIBUTION_TV_THRESHOLD
    assert saved["cold_replica_energy_error"] <= mod.COLD_ENERGY_ERROR_TOLERANCE
    assert saved["round_trip_accounting_error"] == 0.0
    assert all(row["detected"] is True for row in saved["broken_controls"])
    assert saved["deterministic_replay_pass"] is True
    assert saved["timing_claimed"] is False
    assert saved["hardware_speedup_claimed"] is False
    assert saved["replica_exchange_kernel_ready_score"] == 1.0
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["random_seeds"] == [5633, 5634, 5635]
    assert saved["baselines"]["single_chain_equal_transition"]["exact_distribution_tv_max"] <= mod.EXACT_DISTRIBUTION_TV_THRESHOLD
    assert saved["baselines"]["independent_replicas_no_exchange"]["exact_distribution_tv_max"] <= mod.EXACT_DISTRIBUTION_TV_THRESHOLD
    assert "duration_s" not in saved
    assert "cuda_device_receipt" not in saved
    assert "crossover_size" not in saved
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5633_validation_fails_closed_on_manual_readiness() -> None:
    """REQ-SAMPLE-5633: readiness cannot be manually set past failed gates."""

    artifact = mod.build_artifact(root=REPO, replay_sweeps=12, random_seeds=(5633, 5634))
    mutations = [
        ("missing required field", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].__setitem__("x", "y")),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "deterministic_verifier")),
        ("timing_claimed", lambda data: data.__setitem__("timing_claimed", True)),
        ("hardware_speedup_claimed", lambda data: data.__setitem__("hardware_speedup_claimed", True)),
        ("deterministic_replay_pass", lambda data: data.__setitem__("deterministic_replay_pass", False)),
        ("broken_controls", lambda data: data.__setitem__("broken_controls", data["broken_controls"][1:])),
        (
            "broken_controls",
            lambda data: data["broken_controls"][0].__setitem__("detected", False),
        ),
        (
            "replica_exchange_kernel_ready_score",
            lambda data: data.__setitem__("replica_exchange_kernel_ready_score", 0.5),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"reproducibility_checksum", "missing required field"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["exact_distribution_tv_max"] = mod.EXACT_DISTRIBUTION_TV_THRESHOLD * 2.0
    blocked["replica_exchange_kernel_ready_score"] = mod.ready_score(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)

    assert blocked["replica_exchange_kernel_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(blocked)
