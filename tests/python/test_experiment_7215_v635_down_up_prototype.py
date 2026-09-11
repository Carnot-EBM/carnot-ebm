"""Tests for the exact down-up sampler prototype.

Spec: REQ-ISING-7215 and SCENARIO-ISING-7215-*.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot.samplers import experiment_7215_down_up as exp


REPO = Path(__file__).resolve().parents[2]


def test_req_ising_7215_frozen_method_and_roster() -> None:
    """REQ-ISING-7215 freezes the paper version, method, and all 90 cells."""

    assert exp.PAPER_VERSION == "arXiv:2609.08873v1"
    assert exp.PAPER_LOCATIONS == ("Algorithm 1 in Section 3", "Remark 2 in Section 3.2")
    assert exp.ENERGY_CONVENTION == slices.EDGE_COUNTING_CONVENTION
    assert exp.SIZES == (8,)
    assert exp.CARDINALITIES == (1, 2, 4)
    assert exp.BETAS == (0.0, 1.0, 2.0)
    assert exp.SEEDS == tuple(range(7215001, 7215011))
    assert len(exp.SEEDS) * len(exp.CARDINALITIES) * len(exp.BETAS) == 90
    assert exp.MODEL_SPECS == []


def test_scenario_ising_7215_boundaries_and_invalid_inputs() -> None:
    """SCENARIO-ISING-7215-BOUNDARIES handles absorbing states and bad input."""

    instance = slices.make_frustrated_instance(8, 7215001)
    assert exp.down_up_step(instance, (), 1.0, down_uniform=0.0, up_uniform=0.0) == ()
    full = tuple(range(instance.n))
    assert exp.down_up_step(instance, full, 1.0, down_uniform=0.75, up_uniform=0.2) == full
    for k in (0, instance.n):
        matrix, states = exp.transition_matrix(instance, k, 1.0)
        assert states == ((full if k == instance.n else ()),)
        assert np.array_equal(matrix, np.ones((1, 1)))

    with pytest.raises(ValueError, match="0 <= k <= n"):
        exp.enumerate_subsets(8, -1)
    with pytest.raises(ValueError, match="0 <= k <= n"):
        exp.transition_matrix(instance, 9, 1.0)
    with pytest.raises(ValueError, match="sorted unique"):
        exp.down_up_step(instance, (1, 1), 1.0, down_uniform=0.0, up_uniform=0.0)
    with pytest.raises(ValueError, match="outside"):
        exp.down_up_step(instance, (1, 8), 1.0, down_uniform=0.0, up_uniform=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        exp.down_up_step(instance, (0, 1), -1.0, down_uniform=0.0, up_uniform=0.0)
    with pytest.raises(ValueError, match="finite"):
        exp.down_up_step(instance, (0, 1), math.inf, down_uniform=0.0, up_uniform=0.0)
    with pytest.raises(ValueError, match="uniform"):
        exp.down_up_step(instance, (0, 1), 1.0, down_uniform=1.0, up_uniform=0.0)
    with pytest.raises(ValueError, match="uniform"):
        exp.down_up_step(instance, (0, 1), 1.0, down_uniform=0.0, up_uniform=-0.1)

    with pytest.raises(ValueError, match="nonzero"):
        exp.transition_matrix(slices.replace_instance(instance, fields=(0.0,) * 8), 2, 1.0)
    asymmetric = slices.replace_instance(
        instance,
        edges=((1, 0, 1.0),) + instance.edges[1:],
    )
    with pytest.raises(ValueError, match="left < right"):
        exp.transition_matrix(asymmetric, 2, 1.0)


def test_req_ising_7215_log_sum_exp_and_optional_tapes() -> None:
    """REQ-ISING-7215 uses stable conditionals and caller-owned random tapes."""

    instance = slices.make_frustrated_instance(8, 7215002)
    sites, probabilities, energies = exp.replacement_distribution(instance, (0,), 1000.0)
    assert sites == tuple(index for index in range(8) if index != 0)
    assert all(math.isfinite(value) for value in probabilities)
    assert all(value >= 0.0 for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
    assert len(energies) == len(sites)

    sites_zero, probabilities_zero, _ = exp.replacement_distribution(instance, (0,), 0.0)
    assert sites_zero == sites
    assert probabilities_zero == pytest.approx((1.0 / 7.0,) * 7)

    source = (0, 2, 5, 7)
    first = exp.down_up_step(instance, source, 2.0, down_uniform=0.51, up_uniform=0.0)
    second = exp.down_up_step(instance, source, 2.0, down_uniform=0.51, up_uniform=0.0)
    assert first == second
    assert len(first) == len(source)
    assert set(first) - set(source) or first == source

    rng_a = random.Random(17)
    rng_b = random.Random(17)
    assert exp.down_up_step(instance, source, 1.0, rng=rng_a) == exp.down_up_step(
        instance, source, 1.0, rng=rng_b
    )


def test_scenario_ising_7215_transition_matches_direct_construction() -> None:
    """SCENARIO-ISING-7215-KERNEL retains every down path and self-transition."""

    instance = slices.make_frustrated_instance(8, 7215003)
    matrix, states = exp.transition_matrix(instance, 2, 1.0)
    source_index = states.index((0, 1))
    expected = np.zeros(len(states))
    for removed in (0, 1):
        core = tuple(site for site in (0, 1) if site != removed)
        candidates, probabilities, _ = exp.replacement_distribution(instance, core, 1.0)
        for candidate, probability in zip(candidates, probabilities, strict=True):
            target = tuple(sorted((*core, candidate)))
            expected[states.index(target)] += 0.5 * probability

    assert matrix[source_index] == pytest.approx(expected)
    assert matrix[source_index, source_index] > 0.0
    assert np.count_nonzero(matrix[source_index]) == 13


def test_scenario_ising_7215_exact_law_and_diagnostics_are_independent() -> None:
    """SCENARIO-ISING-7215-FINITE-LAW checks the target with scalar energies."""

    instance = slices.make_frustrated_instance(8, 7215004)
    law = exp.independent_exact_law(instance, 4, 2.0)
    matrix, states = exp.transition_matrix(instance, 4, 2.0)
    diagnostics = exp.transition_diagnostics(law, matrix, states)

    assert law.states == states
    assert diagnostics["row_stochasticity_residual"] <= 1.0e-12
    assert diagnostics["minimum_probability"] >= 0.0
    assert diagnostics["fixed_cardinality"] is True
    assert diagnostics["detailed_balance_residual"] <= exp.TOLERANCE
    assert diagnostics["stationarity_residual"] <= exp.TOLERANCE
    assert diagnostics["passed"] is True
    assert law.energies == pytest.approx(
        tuple(
            slices.reference_energy(instance, exp.subset_to_spins(instance.n, state))
            for state in states
        )
    )


def test_req_ising_7215_empirical_draws_follow_the_exact_row() -> None:
    """REQ-ISING-7215 compares sampled one-step traces with an exact matrix row."""

    instance = slices.make_frustrated_instance(8, 7215005)
    matrix, states = exp.transition_matrix(instance, 2, 1.0)
    comparison = exp.empirical_one_step_comparison(
        instance,
        states[0],
        1.0,
        matrix[0],
        states,
        seed=99,
        draws=20_000,
    )

    assert comparison["draws"] == 20_000
    assert comparison["cardinality_violations"] == 0
    assert comparison["max_absolute_error"] < 0.02
    assert comparison["passed"] is True


def test_scenario_ising_7215_mutations_are_all_detected() -> None:
    """SCENARIO-ISING-7215-MUTATIONS rejects all three incorrect transitions."""

    rows = exp.run_mutation_checks()
    assert {row["mutation"] for row in rows} == {
        "omit_removed_site",
        "wrong_energy_sign",
        "drop_self_transitions",
    }
    assert all(row["control_detected"] is True for row in rows)
    assert all(row["passed"] is False for row in rows)


def test_req_ising_7215_wrapper_and_quarantine_are_fail_closed() -> None:
    """REQ-ISING-7215 unwraps only declared wrappers and checks quarantine first."""

    wrapped = {"value": 1, "principle": "A reason."}
    arbitrary = {"value": 1, "metadata": "not a wrapper"}
    assert exp.unwrap_principled_value(wrapped) == 1
    assert exp.unwrap_principled_value(arbitrary) is arbitrary
    assert exp.unwrap_principled_value(3) == 3

    clean = exp.upstream_quarantine_observation({}, manifest_match=False)
    dirty = exp.upstream_quarantine_observation({"flagged_adversarial": True}, manifest_match=False)
    manifest = exp.upstream_quarantine_observation({}, manifest_match=True)
    assert clean["quarantined"] is False
    assert dirty["quarantined"] is True
    assert manifest["quarantined"] is True
    assert exp.gated_upstream_value(wrapped, clean, "unused") == 1
    assert exp.gated_upstream_value(wrapped, dirty, "unused") == "not_consumed_due_to_quarantine"
    assert exp.gated_upstream_value(arbitrary, clean, "unused") is arbitrary


def test_req_ising_7215_preconditions_bind_source_and_context() -> None:
    """REQ-ISING-7215 records paper bytes and the non-promoted Exp7202 null."""

    paper_bytes = Path("/tmp/carnot-exp7215-source/2609.08873v1.html").read_bytes()
    checks, hashes, paper = exp.collect_preconditions(REPO, paper_bytes=paper_bytes)
    by_name = {row["check"]: row for row in checks}

    assert all(row["passed"] is True for row in checks)
    assert by_name["driving_capability_spec"]["observed_value"] == "REQ-ISING-7215 present"
    assert by_name["roadmap_task_contract"]["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    assert by_name["upstream_quarantine"]["observed_value"]["quarantined"] is False
    assert by_name["upstream_authentication"]["observed_value"]["producer_valid"] is False
    assert by_name["upstream_known_failed_values"]["observed_value"]["promoted"] is False
    assert paper["sha256"] == "sha256:" + exp.PAPER_HTML_SHA256
    assert paper["locations"] == list(exp.PAPER_LOCATIONS)
    assert set(hashes) == {str(path) for path in exp.REQUIRED_SOURCE_PATHS} | {exp.PAPER_URL}


def test_scenario_ising_7215_blocked_artifact_names_failed_gate(tmp_path: Path) -> None:
    """SCENARIO-ISING-7215-ARTIFACT keeps an external block diagnostic and terminal."""

    failed = {
        "check": "paper_source",
        "upstream": exp.PAPER_URL,
        "field": "sha256",
        "expected_value": exp.PAPER_HTML_SHA256,
        "observed_value": "wrong",
        "passed": False,
    }
    artifact = exp.build_artifact(
        REPO,
        preconditions=[failed],
        source_hashes={},
        paper_source={},
    )
    output = tmp_path / "blocked.json"
    exp.atomic_write(output, artifact)

    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "paper_source"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", str(output)]) == 0
    assert exp.main(["--date", "19000101"]) == 2


def test_scenario_ising_7215_terminal_artifact_recomputes() -> None:
    """SCENARIO-ISING-7215-ARTIFACT validates the durable full-roster evidence."""

    payload = json.loads((REPO / exp.RESULT_PATH).read_text(encoding="utf-8"))
    assert exp.validate_artifact(payload, root=REPO) == []
    assert payload["down_up_kernel_ready_score"] == 1
    assert payload["verdict_class"] == "circular_positive"
    assert len(payload["transition_rows"]) == 90
    assert len(payload["mutation_rows"]) == 3
    assert payload["MODEL_SPECS"] == []
    assert payload["model_invoked"] is False
    assert payload["paper_replication_claimed"] is False
    assert payload["general_mixing_theorem_claimed"] is False
    assert payload["hardware_speed_claimed"] is False


def test_req_ising_7215_validator_rejects_row_and_claim_mutations() -> None:
    """REQ-ISING-7215 makes readiness, row hashes, and claim limits tamper-evident."""

    payload = json.loads((REPO / exp.RESULT_PATH).read_text(encoding="utf-8"))
    broken = copy.deepcopy(payload)
    broken["transition_rows"][0]["stationarity_residual"] = 1.0
    broken["rows"][0]["stationarity_residual"] = 1.0
    broken["hardware_speed_claimed"] = True
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    errors = exp.validate_artifact(broken)

    assert "transition_rows_invalid" in errors
    assert "rows_invalid" in errors
    assert "claim_limits_invalid" in errors
