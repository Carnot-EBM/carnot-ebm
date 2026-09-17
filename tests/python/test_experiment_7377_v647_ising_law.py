"""Tests for the exact original-clause finite-temperature Ising law fixture.

Spec refs: REQ-ISING-7377 and SCENARIO-ISING-7377-*.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest

from carnot import experiment_7377_v647_ising_law as exp


ROOT = Path(__file__).resolve().parents[2]


def _passing_receipts() -> list[dict[str, object]]:
    """Build the complete receipt name set used by the cold reducer."""

    rows = []
    for name in (*exp.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES):
        rows.append(
            {
                "name": name,
                "exit_code": 0,
                "passed": True,
                "timed_out": False,
                "command_argv": [name],
                "command_environment": {},
                "scope": "unit_fixture",
                "duration_s": 0.01,
                "log_sha256": "sha256:" + "a" * 64,
            }
        )
    return rows


def test_compile_2cnf_matches_independent_clause_enumeration() -> None:
    """SCENARIO-ISING-7377-SOURCE-LAW: all compiled state energies are exact."""

    clauses = ((1, 2), (-2, 3), (-3, -1), (2, 2), (1, -1))
    law = exp.compile_2cnf(3, clauses)

    assert law.energy_convention == "E(s)=offset-sum_i(h_i*s_i)-sum_i<j(J_ij*s_i*s_j)"
    assert law.bit_spin_convention == "s_i=2*x_i-1; x_i=(s_i+1)/2"
    assert len(law.to_dict()["biases"]) == 3
    for bits in exp.enumerate_bits(3):
        spins = exp.bits_to_spins(bits)
        expected = exp.independent_clause_energy(bits, clauses)
        assert law.energy(spins) == pytest.approx(expected, abs=1e-12)

    with pytest.raises(ValueError, match="n_vars_out_of_range"):
        exp.compile_2cnf(13, ((1, 2),))
    with pytest.raises(ValueError, match="clause_not_two_literals"):
        exp.compile_2cnf(2, ((1,),))
    with pytest.raises(ValueError, match="literal_out_of_range"):
        exp.compile_2cnf(2, ((0, 1),))
    with pytest.raises(ValueError, match="spin_state_length"):
        law.energy((1,))


def test_conditioning_is_a_clamp_and_empty_support_is_explicit() -> None:
    """SCENARIO-ISING-7377-CONDITIONING: assumptions do not become penalties."""

    states = exp.enumerate_bits(3)
    mask = exp.condition_mask(states, (1, -3))
    assert sum(mask) == 2
    assert all(bits[0] == 1 and bits[2] == 0 for bits, keep in zip(states, mask) if keep)

    conflict = exp.condition_mask(states, (2, -2))
    probabilities, normalizer = exp.normalized_probabilities([0.0] * len(states), conflict, 1.0)
    assert normalizer == 0.0
    assert probabilities == [0.0] * len(states)

    probabilities, normalizer = exp.normalized_probabilities(
        [exp.independent_clause_energy(bits, ((-1, 2),)) for bits in states],
        [True] * len(states),
        1.0,
    )
    assert normalizer > 0.0
    assert math.fsum(probabilities) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="beta_must_be_positive"):
        exp.normalized_probabilities([0.0], [True], 0.0)
    with pytest.raises(ValueError, match="energy_mask_length"):
        exp.normalized_probabilities([0.0], [True, False], 1.0)


def test_frozen_fixture_has_paths_duplicates_and_conflicting_assumptions() -> None:
    """REQ-ISING-7377: the 24-formula panel freezes every required edge case."""

    formulas = exp.build_frozen_fixture()
    assert len(formulas) == 24
    assert {row["seed"] for row in formulas} == set(exp.FORMULA_SEEDS)
    assert all(row["n_vars"] == 6 for row in formulas)
    assert all(exp.validate_implication_certificate(row) == [] for row in formulas)
    assert all(len(row["certificate"]["path_literals"]) >= 4 for row in formulas)
    assert any(row["feature"] == "duplicate_clause" for row in formulas)
    assert any(row["feature"] == "duplicate_assumption" for row in formulas)
    assert any(row["feature"] == "conflicting_assumptions" for row in formulas)
    assert any(
        not any(exp.condition_mask(exp.enumerate_bits(row["n_vars"]), row["assumptions"]))
        for row in formulas
    )

    forged = deepcopy(formulas[0])
    forged["certificate"]["path_literals"][2] *= -1
    assert exp.validate_implication_certificate(forged)

    mutations = []
    changed = deepcopy(formulas[0])
    changed["certificate"]["source_hash"] = "sha256:" + "0" * 64
    mutations.append((changed, "source_hash_mismatch"))
    changed = deepcopy(formulas[0])
    changed["certificate"]["antecedent"] = 2
    mutations.append((changed, "certificate_endpoints_mismatch"))
    changed = deepcopy(formulas[0])
    changed["certificate"]["implied_clause"] = [-1, 5]
    mutations.append((changed, "implied_clause_mismatch"))
    changed = deepcopy(formulas[0])
    changed["certificate"]["source_clause_indices"].pop()
    mutations.append((changed, "path_index_count_mismatch"))
    changed = deepcopy(formulas[0])
    changed["certificate"]["source_clause_indices"][0] = 99
    mutations.append((changed, "source_clause_index_invalid"))
    changed = deepcopy(formulas[0])
    changed["original_clauses"][4] = [-5, -6]
    changed["source_hash"] = exp._source_hash(6, changed["original_clauses"])
    changed["certificate"]["source_hash"] = changed["source_hash"]
    mutations.append((changed, "enumerated_entailment_failed"))
    for changed, expected in mutations:
        assert expected in exp.validate_implication_certificate(changed)

    assert exp._minimum_states([(0,)], [0.0], [False]) == []


def test_exact_panel_preserves_source_law_and_exposes_wrong_law() -> None:
    """SCENARIO-ISING-7377-NEGATIVE-CONTROL: minima alone cannot pass."""

    formulas = exp.build_frozen_fixture()
    rows = exp.run_exact_panel(formulas)

    assert len(rows) == 24 * len(exp.BETA_GRID) * 3
    assert {row["condition"] for row in rows} == {
        "source_only",
        "proof_assisted_source_only",
        "appended_implied_clause",
    }
    faithful = [row for row in rows if row["condition"] != "appended_implied_clause"]
    appended = [row for row in rows if row["condition"] == "appended_implied_clause"]
    assert max(row["max_abs_energy_residual"] for row in faithful) <= 1e-12
    assert max(row["enumeration_total_variation"] for row in faithful) <= 1e-10
    assert max(row["source_law_total_variation"] for row in faithful) <= 1e-10
    assert all(row["source_minimum_states"] == row["condition_minimum_states"] for row in appended)
    assert any(row["source_law_total_variation"] > 1e-10 for row in appended)
    assert all(row["proof_zeroed_positive_source_mass"] == 0.0 for row in faithful)
    assert all(row["censored"] is False and row["failures"] == [] for row in rows)


def test_independent_mutations_detect_every_required_boundary() -> None:
    """REQ-ISING-7377: offset, sign, beta, multiplicity, support, and clauses bite."""

    controls = exp.run_negative_controls(exp.build_frozen_fixture()[0])
    assert {row["mutation"] for row in controls} == {
        "offset",
        "sign",
        "beta",
        "clause_multiplicity",
        "conditioning",
        "extra_implied_clause",
    }
    assert all(row["detected"] for row in controls)
    assert all(row["censored"] is False and row["failures"] == [] for row in controls)
    offset = next(row for row in controls if row["mutation"] == "offset")
    assert offset["max_abs_energy_residual"] > 0
    assert offset["probability_change"] <= 1e-12
    conditioning = next(row for row in controls if row["mutation"] == "conditioning")
    assert conditioning["positive_source_mass_removed"] > 0


def test_artifact_reduction_scores_only_complete_validated_evidence() -> None:
    """SCENARIO-ISING-7377-TERMINAL: raw rows and receipts determine readiness."""

    formulas = exp.build_frozen_fixture()
    finite_rows = exp.run_exact_panel(formulas)
    controls = exp.run_negative_controls(formulas[0])
    artifact = exp.build_artifact_for_test(finite_rows, controls, _passing_receipts())

    assert artifact["law_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"]["current"] == exp.ZERO_CURRENT_INVOCATIONS
    assert artifact["frozen_sampling_protocol"]["recorded_samples_per_chain"] == 4_000
    assert artifact["frozen_sampling_protocol"]["fixture_sha256"] == exp.fixture_hash(formulas)
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["finite_law_rows"][0]["compiled_normalizer"] += 1.0
    assert "stored_reduction_mismatch" in exp.validate_artifact(changed)
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)

    failed = deepcopy(_passing_receipts())
    failed[0]["passed"] = False
    failed[0]["exit_code"] = 1
    disqualified = exp.build_artifact_for_test(finite_rows, controls, failed)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["law_fixture_ready_score"] == 0
    assert disqualified["gate_check_summary"]["blocking_failed_count"] >= 1

    blocked = exp.build_artifact_for_test(
        finite_rows,
        controls,
        _passing_receipts(),
        preconditions_passed=False,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["law_fixture_ready_score"] == 0


def test_cold_validation_rejects_each_terminal_contract_mutation() -> None:
    """SCENARIO-ISING-7377-TERMINAL: malformed declarations fail closed."""

    formulas = exp.build_frozen_fixture()
    artifact = exp.build_artifact_for_test(
        exp.run_exact_panel(formulas),
        exp.run_negative_controls(formulas[0]),
        _passing_receipts(),
    )
    mutations = (
        ("schema", "wrong", "identity_mismatch"),
        ("run_date", "20260918", "run_identity_mismatch"),
        ("verdict_class", "success", "verdict_class_invalid"),
        ("MODEL_SPECS", ["forbidden"], "current_model_declaration_invalid"),
        ("invocation_counts", {"current": {}}, "current_invocation_counts_invalid"),
        ("inference_substrate_class", "gpu", "substrate_class_invalid"),
        ("execution_venue", "board", "execution_venue_invalid"),
        ("promotion_score", 1, "promotion_nonzero"),
        ("verifier_is_oracle", False, "circularity_declaration_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

    failed = deepcopy(artifact)
    failed["validation_receipts"][0]["passed"] = False
    failed["validation_receipts"][0]["exit_code"] = 1
    failed["verdict_class"] = "disqualified"
    failed["law_fixture_ready_score"] = 1
    assert "failed_state_readiness_nonzero" in exp.validate_artifact(failed)


def test_preconditions_authenticate_sources_and_driving_requirement() -> None:
    """REQ-ISING-7377: exact local sources are checked before dependent work."""

    checks, hashes, sidecars = exp.collect_preconditions(ROOT)
    blocking = [row for row in checks if row["terminal_blocking"]]
    assert blocking and all(row["passed"] for row in blocking)
    assert any(row["check"] == "driving_requirement" for row in checks)
    assert all(value.startswith("sha256:") for value in hashes.values())
    assert sidecars == [
        {
            "label": "no_historical_model_inputs",
            "counted_as_current": False,
            "receipts": [],
        }
    ]
