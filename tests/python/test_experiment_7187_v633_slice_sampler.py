"""Tests for the fixed-cardinality pair-swap Metropolis baseline.

Spec refs: REQ-SAMPLER-7187, REQ-SAMPLER-7187-PROPOSAL,
REQ-SAMPLER-7187-FINITE-LAW, REQ-SAMPLER-7187-CONTROLS,
REQ-SAMPLER-7187-BENCHMARK, REQ-SAMPLER-7187-PREFLIGHT,
REQ-SAMPLER-7187-ARTIFACT, REQ-SAMPLER-7187-READINESS,
REQ-SAMPLER-7187-BOUNDARY, SCENARIO-SAMPLER-7187-SYMMETRY,
SCENARIO-SAMPLER-7187-SINGLETON, SCENARIO-SAMPLER-7187-EXACT,
SCENARIO-SAMPLER-7187-MUTATIONS, SCENARIO-SAMPLER-7187-BENCHMARK,
SCENARIO-SAMPLER-7187-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest

from carnot import experiment_7187_v633_slice_sampler as exp


REPO = Path(__file__).resolve().parents[2]


def _rehash(payload: dict) -> dict:
    payload["reproducibility_checksum"] = exp.artifact_checksum(payload)
    return payload


@pytest.fixture(scope="module")
def ready_artifact() -> dict:
    """Build the complete fixed roster without writing tracked evidence."""

    return exp.build_artifact(root=REPO, run_date=exp.RUN_DATE)


def test_req_sampler_7187_spec_precedes_implementation() -> None:
    """REQ-SAMPLER-7187 fixes the complete bounded evidence contract."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-SAMPLER-7187", 1)[1]
    for anchor in (
        "REQ-SAMPLER-7187-PROPOSAL",
        "REQ-SAMPLER-7187-FINITE-LAW",
        "REQ-SAMPLER-7187-CONTROLS",
        "REQ-SAMPLER-7187-BENCHMARK",
        "REQ-SAMPLER-7187-PREFLIGHT",
        "REQ-SAMPLER-7187-ARTIFACT",
        "REQ-SAMPLER-7187-READINESS",
        "REQ-SAMPLER-7187-BOUNDARY",
        "SCENARIO-SAMPLER-7187-SYMMETRY",
        "SCENARIO-SAMPLER-7187-SINGLETON",
        "SCENARIO-SAMPLER-7187-EXACT",
        "SCENARIO-SAMPLER-7187-MUTATIONS",
        "SCENARIO-SAMPLER-7187-BENCHMARK",
        "SCENARIO-SAMPLER-7187-ARTIFACT",
    ):
        assert anchor in section


def test_req_sampler_7187_graphs_are_fixed_frustrated_and_nonzero_field() -> None:
    """REQ-SAMPLER-7187-FINITE-LAW freezes valid frustrated graph inputs."""

    instances = [exp.make_frustrated_instance(8, seed) for seed in exp.SMALL_GRAPH_SEEDS]
    assert len(instances) == 3
    assert len({instance.instance_hash for instance in instances}) == 3
    for instance in instances:
        receipt = exp.validate_instance(instance)
        assert receipt["passed"] is True
        assert receipt["frustrated_triangle"] is True
        assert all(field != 0.0 for field in instance.fields)
        assert all(left < right for left, right, _ in instance.edges)
        assert len({(left, right) for left, right, _ in instance.edges}) == len(instance.edges)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n": 0}, "n"),
        ({"fields": (0.1,)}, "field count"),
        ({"fields": (float("nan"),) * 8}, "finite"),
        ({"fields": (0.0,) * 8}, "nonzero"),
        ({"edges": ((0, 0, 1.0),)}, "self-loop"),
        ({"edges": ((0, 1, 1.0), (0, 1, -1.0))}, "duplicate"),
        ({"edges": ((0, 9, 1.0),)}, "endpoint"),
        ({"edges": ((0, 1, float("inf")),)}, "finite"),
        ({"edges": ((0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0))}, "frustrated"),
    ],
)
def test_req_sampler_7187_invalid_graphs_fail_closed(changes: dict, message: str) -> None:
    """REQ-SAMPLER-7187-FINITE-LAW rejects invalid exact-law fixtures."""

    instance = exp.replace_instance(exp.make_frustrated_instance(8, 17), **changes)
    with pytest.raises(ValueError, match=message):
        exp.validate_instance(instance)


def test_scenario_sampler_7187_slice_support_and_singletons() -> None:
    """SCENARIO-SAMPLER-7187-SINGLETON keeps exactly the requested cardinality."""

    assert exp.enumerate_slice(4, 0) == ((-1, -1, -1, -1),)
    assert exp.enumerate_slice(4, 4) == ((1, 1, 1, 1),)
    states = exp.enumerate_slice(8, 2)
    assert len(states) == math.comb(8, 2)
    assert all(state.count(1) == 2 for state in states)
    with pytest.raises(ValueError, match="k"):
        exp.enumerate_slice(8, -1)
    with pytest.raises(ValueError, match="k"):
        exp.enumerate_slice(8, 9)
    with pytest.raises(ValueError, match="n"):
        exp.enumerate_slice(0, 0)


def test_req_sampler_7187_energy_counts_each_edge_once() -> None:
    """REQ-SAMPLER-7187-FINITE-LAW uses the declared sign and edge convention."""

    instance = exp.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[0])
    state = exp.enumerate_slice(8, 2)[7]
    favorable = sum(
        coupling * state[left] * state[right] for left, right, coupling in instance.edges
    ) + sum(field * state[index] for index, field in enumerate(instance.fields))
    assert exp.ising_energy(instance, state) == pytest.approx(-favorable)
    assert exp.reference_energy(instance, state) == pytest.approx(-favorable)
    with pytest.raises(ValueError, match="state length"):
        exp.ising_energy(instance, state[:-1])
    with pytest.raises(ValueError, match="spins"):
        exp.reference_energy(instance, (0,) * 8)


def test_scenario_sampler_7187_pair_swap_is_symmetric_and_stable() -> None:
    """SCENARIO-SAMPLER-7187-SYMMETRY proves support, symmetry, and log acceptance."""

    state = (1, 1, -1, -1)
    target = (-1, 1, 1, -1)
    expected = 1.0 / 4.0
    assert exp.pair_swap_proposal_probability(state, target) == expected
    assert exp.pair_swap_proposal_probability(target, state) == expected
    assert exp.pair_swap_proposal_probability(state, state) == 0.0
    assert exp.pair_swap_proposal_probability(state, (1, -1, -1, 1)) == expected
    assert exp.pair_swap_proposal_probability(state, (1, 1, 1, -1)) == 0.0
    assert exp.log_acceptance(1.0e308, 1.0e308) == -math.inf
    assert exp.log_acceptance(1.0e308, -1.0e308) == 0.0
    with pytest.raises(ValueError, match="beta"):
        exp.log_acceptance(0.0, 1.0)

    rng = random.Random(9)
    proposed, forward, reverse = exp.propose_pair_swap(state, rng)
    assert proposed.count(1) == state.count(1)
    assert forward == reverse == expected
    assert exp.propose_pair_swap((-1, -1), rng) == ((-1, -1), 1.0, 1.0)
    with pytest.raises(ValueError, match="spins"):
        exp.propose_pair_swap((0, -1), rng)


def test_req_sampler_7187_exact_law_and_all_transition_arms() -> None:
    """SCENARIO-SAMPLER-7187-EXACT checks three explicit finite matrices."""

    instance = exp.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[0])
    law = exp.independent_exact_law(instance, 2, 5.0)
    assert sum(law.probabilities) == pytest.approx(1.0, abs=exp.TOLERANCE)
    assert min(law.probabilities) > 0.0
    assert law.states == exp.enumerate_slice(8, 2)
    for arm in exp.ARMS:
        matrix = exp.transition_matrix(instance, 2, 5.0, arm=arm)
        diagnostics = exp.transition_diagnostics(law, matrix)
        assert diagnostics["transition_normalization_error_max"] <= exp.TOLERANCE
        assert diagnostics["transition_support_min"] >= 0.0
        assert diagnostics["detailed_balance_error_max"] <= exp.TOLERANCE
        assert diagnostics["stationary_law_error"] <= exp.TOLERANCE
        if arm == exp.INVALID_SINGLE_SPIN_ARM:
            assert np.array_equal(matrix, np.eye(len(law.states)))
    with pytest.raises(ValueError, match="arm"):
        exp.transition_matrix(instance, 2, 1.0, arm="missing")


def test_scenario_sampler_7187_singleton_matrix_is_identity() -> None:
    """SCENARIO-SAMPLER-7187-SINGLETON makes both boundary slices exact."""

    instance = exp.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[1])
    for k in (0, instance.n):
        law = exp.independent_exact_law(instance, k, 2.0)
        matrix = exp.transition_matrix(instance, k, 2.0, arm=exp.PAIR_SWAP_ARM)
        assert law.probabilities == (1.0,)
        assert np.array_equal(matrix, np.ones((1, 1)))


def test_scenario_sampler_7187_mutations_are_detected() -> None:
    """SCENARIO-SAMPLER-7187-MUTATIONS retains every negative result."""

    rows = exp.run_mutations()
    assert {row["mutation_id"] for row in rows} == exp.REQUIRED_MUTATIONS
    assert all(row["detected"] is True and row["passed"] is True for row in rows)
    by_id = {row["mutation_id"]: row for row in rows}
    for mutation_id in (
        "energy_sign_reversal",
        "double_counted_edges",
        "asymmetric_proposal_without_hastings",
    ):
        assert by_id[mutation_id]["observed_value"] > exp.TOLERANCE
    assert by_id["invalid_k"]["observed_value"] == "ValueError"


def test_req_sampler_7187_chain_budget_metrics_and_frozen_control() -> None:
    """SCENARIO-SAMPLER-7187-BENCHMARK charges equal work and keeps nulls."""

    instance = exp.make_frustrated_instance(32, exp.BENCHMARK_SEEDS[0])
    rows = [
        exp.run_chain(
            instance,
            k=2,
            beta=exp.BENCHMARK_BETA,
            arm=arm,
            seed=123,
            energy_budget=80,
        )
        for arm in exp.ARMS
    ]
    assert {row["energy_evaluations"] for row in rows} == {80}
    assert all(row["attempts"] == 80 for row in rows)
    frozen = next(row for row in rows if row["arm"] == exp.INVALID_SINGLE_SPIN_ARM)
    assert frozen["frozen"] is True
    assert frozen["acceptance"] == 0.0
    assert frozen["energy_ess"] is None
    assert frozen["ess_per_second"] is None
    assert frozen["autocorrelation"] is None
    active = [row for row in rows if not row["frozen"]]
    assert all(row["autocorrelation"] is not None for row in active)
    assert all(row["energy_ess"] is not None for row in active)
    with pytest.raises(ValueError, match="exactly one budget"):
        exp.run_chain(instance, k=2, beta=2.0, arm=exp.PAIR_SWAP_ARM, seed=1)
    with pytest.raises(ValueError, match="exactly one budget"):
        exp.run_chain(
            instance,
            k=2,
            beta=2.0,
            arm=exp.PAIR_SWAP_ARM,
            seed=1,
            energy_budget=1,
            wall_time_s=0.01,
        )
    with pytest.raises(ValueError, match="positive"):
        exp.run_chain(
            instance,
            k=2,
            beta=2.0,
            arm=exp.PAIR_SWAP_ARM,
            seed=1,
            energy_budget=0,
        )


def test_req_sampler_7187_autocorrelation_and_ess_boundaries() -> None:
    """REQ-SAMPLER-7187-BENCHMARK reports no invented statistic for constants."""

    assert exp.autocorrelation([2.0] * 20, 4) is None
    assert exp.effective_sample_size([2.0] * 20, 4) is None
    series = [float(index % 3) for index in range(30)]
    correlations = exp.autocorrelation(series, 5)
    assert correlations is not None
    assert correlations[0] == pytest.approx(1.0)
    ess = exp.effective_sample_size(series, 5)
    assert ess is not None and 1.0 <= ess <= len(series)
    with pytest.raises(ValueError, match="lag"):
        exp.autocorrelation([1.0, 2.0], 2)


def test_req_sampler_7187_preconditions_bind_exact_v633_contract(tmp_path: Path) -> None:
    """REQ-SAMPLER-7187-PREFLIGHT records sources, hashes, tools, and paths."""

    checks, hashes = exp.collect_preconditions(
        REPO,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
    )
    assert all(row["passed"] is True for row in checks)
    by_check = {row["check"]: row for row in checks}
    milestone = by_check["same_milestone_gate_fields"]
    assert (
        milestone["expected_value"]
        == milestone["observed_value"]
        == {
            "id": exp.TASK_ID,
            "milestone": exp.MILESTONE,
            "deliverable": str(exp.RESULT_PATH),
            "gated_on": [],
        }
    )
    assert by_check["driving_capability_spec"]["observed_value"]["req_present"] is True
    assert by_check["required_source_bytes"]["passed"] is True
    assert by_check["required_tools"]["passed"] is True
    assert by_check["output_directories"]["passed"] is True
    assert set(hashes) == {str(path) for path in exp.REQUIRED_SOURCE_PATHS}
    assert all(exp.HASH_PATTERN.fullmatch(value) for value in hashes.values())


def test_scenario_sampler_7187_artifact_is_complete_and_bounded(ready_artifact: dict) -> None:
    """SCENARIO-SAMPLER-7187-ARTIFACT recomputes the complete positive claim."""

    assert exp.validate_artifact(ready_artifact) == []
    assert set(ready_artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert len(ready_artifact["finite_law_rows"]) == 54
    assert len(ready_artifact["transition_rows"]) == 162
    assert len(ready_artifact["benchmark_rows"]) == 240
    assert len(ready_artifact["rows"]) == 54 + 162 + 240
    assert ready_artifact["slice_sampler_ready_score"] == 1
    assert ready_artifact["verdict_class"] == "positive"
    assert ready_artifact["hardware_execution_claimed"] is False
    assert ready_artifact["paper_replication_claimed"] is False
    assert ready_artifact["speed_win_claimed"] in {False, True}
    assert ready_artifact["gate_check_summary"]["passed"] is True
    assert ready_artifact["duration_s"] > 0.0
    assert ready_artifact["edge_counting_convention"] == (
        "E(s)=-sum_{(i,j) in E} J_ij*s_i*s_j-sum_i h_i*s_i; "
        "each undirected edge is stored once with i<j and counted once"
    )
    frozen_rows = [
        row for row in ready_artifact["benchmark_rows"] if row["arm"] == exp.INVALID_SINGLE_SPIN_ARM
    ]
    assert frozen_rows and all(row["frozen"] is True for row in frozen_rows)
    assert all(row["energy_ess"] is None for row in frozen_rows)
    assert all(row["autocorrelation"] is None for row in frozen_rows)


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda item: item.pop("rows"), "missing_required_fields"),
        (lambda item: item["field_principles"].pop("rows"), "field_principles_invalid"),
        (lambda item: item["finite_law_rows"].pop(), "finite_law_rows_incomplete"),
        (lambda item: item["transition_rows"][0].update(passed=False), "transition_failure"),
        (lambda item: item["benchmark_rows"].pop(), "benchmark_rows_incomplete"),
        (
            lambda item: next(
                row for row in item["benchmark_rows"] if row["arm"] == exp.INVALID_SINGLE_SPIN_ARM
            ).update(energy_ess=1.0),
            "frozen_control_metrics_invalid",
        ),
        (lambda item: item.update(hardware_execution_claimed=True), "claim_boundary_invalid"),
        (lambda item: item.update(paper_replication_claimed=True), "claim_boundary_invalid"),
        (lambda item: item.update(slice_sampler_ready_score=0), "readiness_invalid"),
    ],
)
def test_scenario_sampler_7187_artifact_mutations_fail_closed(
    ready_artifact: dict, mutator, error: str
) -> None:
    """SCENARIO-SAMPLER-7187-ARTIFACT rejects incomplete or inflated evidence."""

    changed = deepcopy(ready_artifact)
    mutator(changed)
    _rehash(changed)
    assert error in exp.validate_artifact(changed)


def test_req_sampler_7187_checksum_and_json_reject_nonfinite(ready_artifact: dict) -> None:
    """REQ-SAMPLER-7187-ARTIFACT binds content and rejects nonfinite evidence."""

    changed = deepcopy(ready_artifact)
    changed["run_date"] = "wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    with pytest.raises(ValueError, match="nonfinite"):
        exp.canonical_json({"bad": float("nan")})


def test_req_sampler_7187_external_failure_writes_valid_blocked_state(tmp_path: Path) -> None:
    """REQ-SAMPLER-7187-PREFLIGHT stops before all measurement rows."""

    checks, hashes = exp.collect_preconditions(
        REPO,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint" / "running.json",
    )
    checks[0] = {**checks[0], "passed": False, "observed_value": "missing"}
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
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_check"] == checks[0]["check"]


def test_req_sampler_7187_atomic_writer_and_validation_cli(
    ready_artifact: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-SAMPLER-7187-ARTIFACT validates caller-selected durable bytes."""

    path = tmp_path / "artifact.json"
    receipt = exp.atomic_write(path, ready_artifact)
    assert receipt["atomic_replace"] is True
    assert receipt["sha256"] == exp.sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8")) == ready_artifact
    assert exp.main(["--validate", str(path)]) == 0
    assert "validation_end" in capsys.readouterr().out
    path.write_text("not json", encoding="utf-8")
    assert exp.main(["--validate", str(path)]) == 2


def test_req_sampler_7187_defensive_input_paths_fail_closed() -> None:
    """REQ-SAMPLER-7187 rejects malformed inputs on each public boundary."""

    with pytest.raises(ValueError, match="n"):
        exp.make_frustrated_instance(2, 1)
    assert exp.make_frustrated_instance(3, 1).n == 3
    instance = exp.make_frustrated_instance(8, exp.SMALL_GRAPH_SEEDS[0])
    reversed_edge = ((1, 0, 1.0), *instance.edges[1:])
    with pytest.raises(ValueError, match="left < right"):
        exp.validate_instance(exp.replace_instance(instance, edges=reversed_edge))
    with pytest.raises(ValueError, match="beta"):
        exp.independent_exact_law(instance, 2, 0.0)
    assert exp.pair_swap_proposal_probability((1, -1), (1,)) == 0.0
    with pytest.raises(ValueError, match="NaN"):
        exp.log_acceptance(1.0, float("nan"))
    law = exp.independent_exact_law(instance, 2, 1.0)
    with pytest.raises(ValueError, match="shape"):
        exp.transition_diagnostics(law, np.eye(1))
    with pytest.raises(ValueError, match="unknown mutation"):
        exp._mutation_matrix(instance, 2, 1.0, "missing")
    with pytest.raises(ValueError, match="unknown arm"):
        exp.run_chain(instance, k=2, beta=1.0, arm="missing", seed=1, energy_budget=2)
    with pytest.raises(ValueError, match="wall-time"):
        exp.run_chain(
            instance,
            k=2,
            beta=1.0,
            arm=exp.PAIR_SWAP_ARM,
            seed=1,
            wall_time_s=0.0,
        )


def test_req_sampler_7187_validator_defensive_branches(ready_artifact: dict) -> None:
    """SCENARIO-SAMPLER-7187-ARTIFACT rejects every derived readiness input."""

    mutations = (
        (lambda item: item.update(verifier_is_oracle=True), "verifier_authority_invalid"),
        (
            lambda item: item.update(
                verdict_class="blocked", status="complete", slice_sampler_ready_score=0
            ),
            "blocked_state_invalid",
        ),
        (lambda item: item["finite_law_rows"][0].update(passed=False), "finite_law_failure"),
        (lambda item: item["transition_rows"].pop(), "transition_rows_incomplete"),
        (lambda item: item["rows"].pop(), "rows_incomplete"),
        (lambda item: item["mutation_rows"].pop(), "mutation_controls_invalid"),
        (lambda item: item.update(status="wrong"), "terminal_verdict_invalid"),
        (
            lambda item: item.update(inference_substrate_class="wrong"),
            "substrate_class_invalid",
        ),
        (lambda item: item["gate_check_summary"].update(passed=False), "gate_summary_invalid"),
    )
    for mutator, expected in mutations:
        changed = deepcopy(ready_artifact)
        mutator(changed)
        _rehash(changed)
        assert expected in exp.validate_artifact(changed)


def test_req_sampler_7187_run_experiment_and_main_paths(
    ready_artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLER-7187-ARTIFACT covers validated publication and CLI failures."""

    payload = deepcopy(ready_artifact)
    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: payload)
    published = exp.run_experiment(root=tmp_path)
    assert published == payload
    assert (tmp_path / exp.RESULT_PATH).is_file()
    monkeypatch.setattr(exp, "validate_artifact", lambda _payload: ["forced"])
    with pytest.raises(ValueError, match="artifact validation"):
        exp.run_experiment(root=tmp_path)

    monkeypatch.setattr(exp, "run_experiment", lambda **_kwargs: payload)
    assert exp.main([]) == 0

    def fail_run(**_kwargs) -> dict:
        raise ValueError("forced")

    monkeypatch.setattr(exp, "run_experiment", fail_run)
    assert exp.main([]) == 2
    assert "experiment_error" in capsys.readouterr().out
