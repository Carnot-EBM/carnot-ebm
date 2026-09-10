"""Tests for the fixed-width transition audit.

Spec refs: REQ-SAMPLER-7188, REQ-SAMPLER-7188-QUANTIZER,
REQ-SAMPLER-7188-LAWS, REQ-SAMPLER-7188-EXACT,
REQ-SAMPLER-7188-CONTROL, REQ-SAMPLER-7188-TRAJECTORIES,
REQ-SAMPLER-7188-COST, REQ-SAMPLER-7188-PREFLIGHT,
REQ-SAMPLER-7188-ARTIFACT, REQ-SAMPLER-7188-READINESS,
REQ-SAMPLER-7188-BOUNDARY, SCENARIO-SAMPLER-7188-QUANTIZER,
SCENARIO-SAMPLER-7188-CORRECTION, SCENARIO-SAMPLER-7188-DISTORTION,
SCENARIO-SAMPLER-7188-TRAJECTORIES, SCENARIO-SAMPLER-7188-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7187_v633_slice_sampler as slices
from carnot import experiment_7188_v633_quantized_transition_audit as exp


REPO = Path(__file__).resolve().parents[2]


def _rehash(payload: dict) -> dict:
    payload["reproducibility_checksum"] = exp.artifact_checksum(payload)
    return payload


@pytest.fixture(scope="module")
def ready_artifact() -> dict:
    """Build the complete frozen roster once for artifact tests."""

    return exp.build_artifact(root=REPO, run_date=exp.RUN_DATE)


def test_req_sampler_7188_spec_precedes_implementation() -> None:
    """REQ-SAMPLER-7188 fixes each law, metric, roster, and claim boundary."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-SAMPLER-7188", 1)[1]
    for anchor in (
        "REQ-SAMPLER-7188-QUANTIZER",
        "REQ-SAMPLER-7188-LAWS",
        "REQ-SAMPLER-7188-EXACT",
        "REQ-SAMPLER-7188-CONTROL",
        "REQ-SAMPLER-7188-TRAJECTORIES",
        "REQ-SAMPLER-7188-COST",
        "REQ-SAMPLER-7188-PREFLIGHT",
        "REQ-SAMPLER-7188-ARTIFACT",
        "REQ-SAMPLER-7188-READINESS",
        "REQ-SAMPLER-7188-BOUNDARY",
        "SCENARIO-SAMPLER-7188-QUANTIZER",
        "SCENARIO-SAMPLER-7188-CORRECTION",
        "SCENARIO-SAMPLER-7188-DISTORTION",
        "SCENARIO-SAMPLER-7188-TRAJECTORIES",
        "SCENARIO-SAMPLER-7188-ARTIFACT",
    ):
        assert anchor in section


def test_scenario_sampler_7188_quantizer_rounds_then_saturates() -> None:
    """SCENARIO-SAMPLER-7188-QUANTIZER checks ties, endpoints, and clipping."""

    result = exp.quantize_values((-20.0, -1.5, -0.5, 0.5, 1.5, 20.0), bits=4, scale=1.0)
    assert result.qmin == -8
    assert result.qmax == 7
    assert result.codes == (-8, -2, 0, 0, 2, 7)
    assert result.dequantized == (-8.0, -2.0, 0.0, 0.0, 2.0, 7.0)
    assert result.saturation_count == 2
    with pytest.raises(ValueError, match="bits"):
        exp.quantize_values((1.0,), bits=3, scale=1.0)
    with pytest.raises(ValueError, match="scale"):
        exp.quantize_values((1.0,), bits=4, scale=0.0)
    with pytest.raises(ValueError, match="finite"):
        exp.quantize_values((float("nan"),), bits=4, scale=1.0)


@pytest.mark.parametrize("bits", exp.QUANTIZER_BITS)
def test_req_sampler_7188_instance_uses_one_scale_and_independent_energy(bits: int) -> None:
    """REQ-SAMPLER-7188-QUANTIZER shares scale without changing the authority path."""

    instance = slices.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[0])
    quantized = exp.quantize_instance(instance, bits)
    coefficients = [edge[2] for edge in instance.edges] + list(instance.fields)
    assert quantized.scale == pytest.approx(max(map(abs, coefficients)) / quantized.qmax)
    assert len(quantized.edge_codes) == len(instance.edges)
    assert len(quantized.field_codes) == instance.n
    assert quantized.saturation_count == 0
    state = slices.enumerate_slice(8, 2)[7]
    authority = exp.full_precision_authority_energy(instance, state)
    assert authority == pytest.approx(slices.ising_energy(instance, state), abs=1e-12)
    assert math.isfinite(exp.quantized_energy(quantized, state))
    with pytest.raises(ValueError, match="state"):
        exp.quantized_energy(quantized, state[:-1])


def test_req_sampler_7188_delayed_acceptance_product_derivation() -> None:
    """SCENARIO-SAMPLER-7188-CORRECTION verifies the forward/reverse law."""

    beta = 2.3
    for delta_full, delta_quant in ((1.2, 0.3), (-0.8, 1.1), (0.4, -0.7)):
        forward = exp.delayed_acceptance_log_terms(beta, delta_full, delta_quant)
        reverse = exp.delayed_acceptance_log_terms(beta, -delta_full, -delta_quant)
        assert forward[2] - reverse[2] == pytest.approx(-beta * delta_full)
        assert forward[2] == pytest.approx(forward[0] + forward[1])
    with pytest.raises(ValueError, match="beta"):
        exp.delayed_acceptance_log_terms(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        exp.delayed_acceptance_log_terms(1.0, float("nan"), 1.0)


def test_scenario_sampler_7188_exact_matrices_distinguish_targets() -> None:
    """SCENARIO-SAMPLER-7188-CORRECTION keeps full fidelity and exposes naive bias."""

    instance = slices.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[0])
    states = slices.enumerate_slice(8, 2)
    full = tuple(exp.full_precision_authority_energy(instance, state) for state in states)
    quantized = exp.quantize_instance(instance, 4)
    approximate = tuple(exp.quantized_energy(quantized, state) for state in states)
    full_target = exp.distribution_from_energies(full, 2.0)
    quantized_target = exp.distribution_from_energies(approximate, 2.0)

    full_matrix = exp.build_transition_matrix(states, 2.0, full, approximate, exp.FULL_ARM)
    naive_matrix = exp.build_transition_matrix(states, 2.0, full, approximate, exp.NAIVE_ARM)
    corrected_matrix = exp.build_transition_matrix(
        states, 2.0, full, approximate, exp.CORRECTED_ARM
    )
    for matrix in (full_matrix, naive_matrix, corrected_matrix):
        assert np.max(np.abs(matrix.sum(axis=1) - 1.0)) <= exp.TOLERANCE
        assert np.min(matrix) >= 0.0
    full_diagnostics = exp.transition_diagnostics(full_matrix, full_target)
    corrected_diagnostics = exp.transition_diagnostics(corrected_matrix, full_target)
    naive_own = exp.transition_diagnostics(naive_matrix, quantized_target)
    naive_full = exp.transition_diagnostics(naive_matrix, full_target)
    assert full_diagnostics["stationary_residual_max"] <= exp.TOLERANCE
    assert corrected_diagnostics["detailed_balance_error_max"] <= exp.TOLERANCE
    assert corrected_diagnostics["stationary_residual_max"] <= exp.TOLERANCE
    assert naive_own["stationary_residual_max"] <= exp.TOLERANCE
    assert naive_full["stationary_residual_max"] > exp.TOLERANCE
    with pytest.raises(ValueError, match="arm"):
        exp.build_transition_matrix(states, 2.0, full, approximate, "missing")
    with pytest.raises(ValueError, match="length"):
        exp.build_transition_matrix(states, 2.0, full[:-1], approximate, exp.FULL_ARM)
    singleton = ((-1, -1),)
    assert np.array_equal(
        exp.build_transition_matrix(singleton, 1.0, (0.0,), (0.0,), exp.FULL_ARM),
        np.ones((1, 1)),
    )
    with pytest.raises(ValueError, match="shape"):
        exp.transition_diagnostics(np.eye(1), (0.5, 0.5))
    with pytest.raises(ValueError, match="same shape"):
        exp.total_variation((1.0,), (0.5, 0.5))
    with pytest.raises(ValueError, match="beta"):
        exp.distribution_from_energies((0.0,), 0.0)
    with pytest.raises(ValueError, match="nonempty"):
        exp.distribution_from_energies((), 1.0)


def test_req_sampler_7188_matched_random_control_matches_error_norms() -> None:
    """REQ-SAMPLER-7188-CONTROL matches coupling and field error separately."""

    instance = slices.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[0])
    quantized = exp.quantize_instance(instance, 4)
    perturbed = exp.matched_perturbation(instance, quantized, seed=77)
    quant_edge_error = np.asarray(
        [
            code * quantized.scale - edge[2]
            for edge, code in zip(instance.edges, quantized.edge_codes)
        ]
    )
    control_edge_error = np.asarray(
        [changed[2] - edge[2] for edge, changed in zip(instance.edges, perturbed.edges)]
    )
    quant_field_error = np.asarray(quantized.field_codes) * quantized.scale - np.asarray(
        instance.fields
    )
    control_field_error = np.asarray(perturbed.fields) - np.asarray(instance.fields)
    assert np.linalg.norm(control_edge_error) == pytest.approx(np.linalg.norm(quant_edge_error))
    assert np.linalg.norm(control_field_error) == pytest.approx(np.linalg.norm(quant_field_error))
    exact = exp.quantize_instance(slices.make_frustrated_instance(8, 1), 4)
    zero = exp._matched_error_vector(np.zeros(3), seed=1)
    assert np.array_equal(zero, np.zeros(3))
    assert exact.bits == 4


def test_req_sampler_7188_order_and_moment_metrics_validate_shapes() -> None:
    """REQ-SAMPLER-7188-EXACT records order, TV, and two moment orders."""

    order = exp.energy_order_metrics((0.0, 1.0, 2.0), (0.0, 2.0, 1.0))
    assert order["energy_order_inversion_count"] == 1
    assert order["energy_order_comparable_pair_count"] == 3
    assert order["quantized_tie_count"] == 0
    states = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    first, second = exp.state_moments(states, (0.25,) * 4)
    assert np.array_equal(first, np.zeros(2))
    assert second == pytest.approx((0.0,))
    bias = exp.moment_bias(states, (0.25,) * 4, (0.05, 0.15, 0.3, 0.5))
    assert bias["first_moment_bias_max"] > 0.0
    assert bias["second_moment_bias_max"] > 0.0
    with pytest.raises(ValueError, match="length"):
        exp.energy_order_metrics((0.0,), (0.0, 1.0))
    with pytest.raises(ValueError, match="probabilities"):
        exp.state_moments(states, (1.0,))


def test_scenario_sampler_7188_trajectories_separate_error_and_cost() -> None:
    """SCENARIO-SAMPLER-7188-TRAJECTORIES reports fidelity and calls per law."""

    instance = slices.make_frustrated_instance(8, exp.TRAJECTORY_GRAPH_SEED)
    states = slices.enumerate_slice(8, exp.TRAJECTORY_K)
    full = tuple(exp.full_precision_authority_energy(instance, state) for state in states)
    quantized = exp.quantize_instance(instance, 4)
    approximate = tuple(exp.quantized_energy(quantized, state) for state in states)
    full_target = exp.distribution_from_energies(full, exp.TRAJECTORY_BETA)
    quantized_target = exp.distribution_from_energies(approximate, exp.TRAJECTORY_BETA)
    rows = {
        arm: exp.run_trajectory(
            states=states,
            beta=exp.TRAJECTORY_BETA,
            full_energies=full,
            approximate_energies=approximate,
            full_target=full_target,
            own_target=full_target
            if arm in {exp.FULL_ARM, exp.CORRECTED_ARM}
            else quantized_target,
            arm=arm,
            seed=19,
            proposals=120,
            burn_in=20,
        )
        for arm in (exp.FULL_ARM, exp.NAIVE_ARM, exp.CORRECTED_ARM)
    }
    assert rows[exp.FULL_ARM]["full_energy_calls"] == 121
    assert rows[exp.NAIVE_ARM]["full_energy_calls"] == 0
    assert 1 <= rows[exp.CORRECTED_ARM]["full_energy_calls"] <= 121
    assert rows[exp.CORRECTED_ARM]["full_energy_calls_saved"] >= 0
    assert rows[exp.NAIVE_ARM]["exact_distortion_tv_from_full"] > 0.0
    assert rows[exp.FULL_ARM]["exact_distortion_tv_from_full"] == 0.0
    assert all(row["monte_carlo_tv_to_own_target"] >= 0.0 for row in rows.values())
    with pytest.raises(ValueError, match="burn"):
        exp.run_trajectory(
            states=states,
            beta=2.0,
            full_energies=full,
            approximate_energies=approximate,
            full_target=full_target,
            own_target=full_target,
            arm=exp.FULL_ARM,
            seed=1,
            proposals=10,
            burn_in=10,
        )
    with pytest.raises(ValueError, match="arm"):
        exp.run_trajectory(
            states=states,
            beta=2.0,
            full_energies=full,
            approximate_energies=approximate,
            full_target=full_target,
            own_target=full_target,
            arm="missing",
            seed=1,
            proposals=10,
            burn_in=1,
        )
    with pytest.raises(ValueError, match="matching lengths"):
        exp.run_trajectory(
            states=states,
            beta=2.0,
            full_energies=full[:-1],
            approximate_energies=approximate,
            full_target=full_target,
            own_target=full_target,
            arm=exp.FULL_ARM,
            seed=1,
            proposals=10,
            burn_in=1,
        )


def test_req_sampler_7188_preconditions_bind_task_and_upstream_gate(tmp_path: Path) -> None:
    """REQ-SAMPLER-7188-PREFLIGHT records exact expected and observed contracts."""

    checks, hashes = exp.collect_preconditions(
        REPO,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    assert all(row["passed"] is True for row in checks)
    by_check = {row["check"]: row for row in checks}
    task = by_check["same_milestone_gate_fields"]
    assert task["expected_value"] == task["observed_value"] == exp.EXPECTED_TASK_CONTRACT
    upstream = by_check["upstream_slice_sampler_ready"]
    assert upstream["expected_value"] == 1
    assert upstream["observed_value"] == 1
    assert by_check["driving_capability_spec"]["observed_value"]["req_present"] is True
    assert set(hashes) == {str(path) for path in exp.REQUIRED_SOURCE_PATHS}


def test_req_sampler_7188_preflight_missing_and_malformed_contracts(tmp_path: Path) -> None:
    """REQ-SAMPLER-7188-PREFLIGHT names unreadable external inputs instead of guessing."""

    assert exp._task_contract(tmp_path) is None
    roadmap = tmp_path / exp.ROADMAP_PATH
    roadmap.write_text("[]\n", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    roadmap.write_text("tasks:\n  - id: another-task\n", encoding="utf-8")
    assert exp._task_contract(tmp_path) is None
    checks, hashes = exp.collect_preconditions(tmp_path)
    assert set(hashes) == {str(exp.ROADMAP_PATH)}
    assert (
        next(row for row in checks if row["check"] == "upstream_slice_sampler_ready")[
            "observed_value"
        ]
        is None
    )


def test_scenario_sampler_7188_artifact_is_complete_and_bounded(ready_artifact: dict) -> None:
    """SCENARIO-SAMPLER-7188-ARTIFACT recomputes complete fidelity evidence."""

    assert exp.validate_artifact(ready_artifact) == []
    assert set(ready_artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert len(ready_artifact["quantizer_rows"]) == 18
    assert len(ready_artifact["law_comparison_rows"]) == 648
    assert len(ready_artifact["trajectory_rows"]) == 120
    assert len(ready_artifact["cost_rows"]) == 12
    assert len(ready_artifact["rows"]) == 798
    assert ready_artifact["quantized_audit_complete_score"] == 1
    assert ready_artifact["corrected_kernel_ready_score"] == 1
    assert ready_artifact["verdict_class"] == "positive"
    assert ready_artifact["hardware_execution_claimed"] is False
    assert ready_artifact["soft_spin_execution_claimed"] is False
    assert ready_artifact["tsu_execution_claimed"] is False
    assert ready_artifact["paper_replication_claimed"] is False
    assert ready_artifact["useful_acceleration_claimed"] in {False, True}
    assert "different target" in ready_artifact["honest_verdict"]
    assert ready_artifact["duration_s"] > 0.0


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda item: item.pop("rows"), "missing_required_fields"),
        (lambda item: item["field_principles"].pop("rows"), "field_principles_invalid"),
        (lambda item: item["quantizer_rows"].pop(), "quantizer_rows_incomplete"),
        (lambda item: item["law_comparison_rows"].pop(), "law_rows_incomplete"),
        (
            lambda item: item["law_comparison_rows"][0].update(passed=False),
            "law_rows_invalid",
        ),
        (lambda item: item["trajectory_rows"].pop(), "trajectory_rows_incomplete"),
        (lambda item: item["cost_rows"].pop(), "cost_rows_incomplete"),
        (lambda item: item["rows"].pop(), "rows_incomplete"),
        (
            lambda item: next(
                row for row in item["law_comparison_rows"] if row["arm"] == exp.CORRECTED_ARM
            ).update(full_target_stationary_residual_max=1.0),
            "corrected_law_invalid",
        ),
        (lambda item: item.update(hardware_execution_claimed=True), "claim_boundary_invalid"),
        (lambda item: item.update(verifier_is_oracle=True), "verifier_authority_invalid"),
        (
            lambda item: item.update(useful_acceleration_claimed=True),
            "acceleration_claim_invalid",
        ),
        (lambda item: item.update(corrected_kernel_ready_score=0), "readiness_invalid"),
        (lambda item: item.update(status="wrong"), "terminal_verdict_invalid"),
        (
            lambda item: item.update(inference_substrate_class="wrong"),
            "substrate_class_invalid",
        ),
        (
            lambda item: item["gate_check_summary"].update(passed=False),
            "gate_summary_invalid",
        ),
        (
            lambda item: item.update(delayed_acceptance_derivation="wrong"),
            "correction_derivation_invalid",
        ),
    ],
)
def test_scenario_sampler_7188_artifact_mutations_fail_closed(
    ready_artifact: dict, mutator, error: str
) -> None:
    """SCENARIO-SAMPLER-7188-ARTIFACT rejects omissions and inflated claims."""

    changed = deepcopy(ready_artifact)
    mutator(changed)
    _rehash(changed)
    assert error in exp.validate_artifact(changed)


def test_req_sampler_7188_external_failure_is_terminal_blocked(tmp_path: Path) -> None:
    """REQ-SAMPLER-7188-PREFLIGHT stops before measurement on an external block."""

    checks, hashes = exp.collect_preconditions(
        REPO,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    checks[-1] = {**checks[-1], "passed": False, "observed_value": 0}
    artifact = exp.build_artifact(
        root=REPO,
        run_date=exp.RUN_DATE,
        preconditions=checks,
        source_hashes=hashes,
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_check"] == checks[-1]["check"]


def test_req_sampler_7188_atomic_write_and_validation_cli(
    ready_artifact: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-SAMPLER-7188-ARTIFACT validates durable bytes through the entry API."""

    path = tmp_path / "artifact.json"
    receipt = exp.atomic_write(path, ready_artifact)
    assert receipt["atomic_replace"] is True
    assert receipt["sha256"] == exp.sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == ready_artifact
    assert exp.main(["--validate", str(path)]) == 0
    assert "validation" in capsys.readouterr().out
    path.write_text("not json", encoding="utf-8")
    assert exp.main(["--validate", str(path)]) == 2
    path.write_text("{}\n", encoding="utf-8")
    assert exp.main(["--validate", str(path)]) == 2


def test_req_sampler_7188_checksum_and_defensive_validation(ready_artifact: dict) -> None:
    """REQ-SAMPLER-7188-ARTIFACT binds bytes and rejects invalid terminal states."""

    changed = deepcopy(ready_artifact)
    changed["run_date"] = "wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    with pytest.raises(ValueError, match="nonfinite"):
        exp.canonical_json({"bad": float("nan")})
    blocked = deepcopy(ready_artifact)
    blocked.update(verdict_class="blocked", status="complete")
    _rehash(blocked)
    assert "blocked_state_invalid" in exp.validate_artifact(blocked)


def test_req_sampler_7188_publication_and_main_paths(
    ready_artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLER-7188-ARTIFACT covers atomic publication and CLI failures."""

    payload = deepcopy(ready_artifact)
    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: payload)
    published = exp.run_experiment(
        root=REPO, output=tmp_path / "published.json", run_date=exp.RUN_DATE
    )
    assert published == payload
    assert (tmp_path / "published.json").is_file()
    monkeypatch.setattr(exp, "validate_artifact", lambda _payload: ["forced"])
    with pytest.raises(ValueError, match="invalid Exp7188"):
        exp.run_experiment(root=REPO, output=tmp_path / "bad.json", run_date=exp.RUN_DATE)

    monkeypatch.setattr(exp, "run_experiment", lambda **_kwargs: payload)
    assert exp.main([]) == 0

    def fail_run(**_kwargs) -> dict:
        raise ValueError("forced")

    monkeypatch.setattr(exp, "run_experiment", fail_run)
    assert exp.main(["--output", str(tmp_path / "never.json")]) == 2
    assert "experiment_error" in capsys.readouterr().out
