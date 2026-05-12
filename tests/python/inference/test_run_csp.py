"""Tests for the RUN-CSP unsupervised binary-CSP solver.

Spec traces: REQ-SAMPLE-1972, SCENARIO-SAMPLE-1972.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.inference.run_csp import (
    BinaryCSPConstraint,
    RUNCSPSolver,
    RUNCSPSolver,
    RUNCSPSolverConfig,
    RunCSPGraph,
    build_planted_binary_csp,
    run_experiment,
    validator_graph_to_run_csp,
)


def test_req_sample_1972_spec_anchor_exists() -> None:
    """REQ-SAMPLE-1972, SCENARIO-SAMPLE-1972: solver is spec-anchored."""

    spec = Path("openspec/capabilities/samplers/spec.md").read_text(encoding="utf-8")

    assert "REQ-SAMPLE-1972" in spec
    assert "SCENARIO-SAMPLE-1972" in spec
    assert "results/experiment_1972_run_csp_unsupervised.json" in spec


def test_req_sample_1972_maps_validator_graph_to_bipartite_messages() -> None:
    """REQ-SAMPLE-1972-1: validator graph constraints become RUN-CSP edges."""

    graph = validator_graph_to_run_csp(
        3,
        [
            {"name": "same_0_1", "scope": [0, 1], "allowed": [[0, 0], [1, 1]]},
            {"name": "diff_1_2", "variables": [1, 2], "allowed": [[0, 1], [1, 0]]},
        ],
    )

    architecture = graph.message_passing_architecture()

    assert graph.num_variables == 3
    assert [constraint.name for constraint in graph.constraints] == ["same_0_1", "diff_1_2"]
    assert architecture["num_variable_nodes"] == 3
    assert architecture["num_constraint_nodes"] == 2
    assert architecture["bipartite_edges"] == [(0, 0), (0, 1), (1, 1), (1, 2)]
    assert graph.satisfaction_rate([1, 1, 0]) == pytest.approx(1.0)
    assert graph.satisfaction_rate([1, 0, 0]) == pytest.approx(0.0)


def test_req_sample_1972_rejects_non_binary_validator_constraints() -> None:
    """REQ-SAMPLE-1972-1: malformed binary-CSP validator rows fail clearly."""

    with pytest.raises(ValueError, match="scope"):
        validator_graph_to_run_csp(3, [{"scope": [0, 1, 2], "allowed": [[0, 1, 0]]}])
    with pytest.raises(ValueError, match="binary values"):
        validator_graph_to_run_csp(3, [{"scope": [0, 1], "allowed": [[0, 2]]}])
    with pytest.raises(ValueError, match="out of range"):
        validator_graph_to_run_csp(2, [{"scope": [0, 2], "allowed": [[0, 1]]}])


def test_req_sample_1972_trains_unsupervised_against_energy_loss() -> None:
    """REQ-SAMPLE-1972-2: training calls Carnot energy and reduces loss."""

    base_graph = build_planted_binary_csp(num_variables=12, edge_factor=2, seed=1972)
    energy_calls = 0

    def counting_energy(probabilities):
        nonlocal energy_calls
        energy_calls += 1
        return base_graph.table_energy(probabilities)

    graph = validator_graph_to_run_csp(
        base_graph.num_variables,
        [constraint.to_validator_row() for constraint in base_graph.constraints],
        energy_fn=counting_energy,
    )
    solver = RUNCSPSolver(
        RUNCSPSolverConfig(
            epochs=3,
            message_steps=12,
            seed=7,
            candidate_gains=(0.35, 0.7, 1.4),
        )
    )

    trained = solver.train(graph)

    assert energy_calls > 0
    assert trained.labels_used is False
    assert trained.history[0]["energy"] >= trained.history[-1]["energy"]
    assert trained.final_energy <= trained.initial_energy
    assert trained.satisfaction_rate >= 0.95
    assert len(trained.assignment) == graph.num_variables


def test_req_sample_1972_reuses_learned_message_parameters_on_larger_graph() -> None:
    """SCENARIO-SAMPLE-1972: learned message parameters transfer by graph size."""

    train_graph = build_planted_binary_csp(num_variables=40, edge_factor=2, seed=1972)
    eval_graph = build_planted_binary_csp(num_variables=160, edge_factor=2, seed=1973)
    solver = RUNCSPSolver(
        RUNCSPSolverConfig(
            epochs=4,
            message_steps=16,
            seed=11,
            candidate_gains=(0.4, 0.8, 1.6),
        )
    )

    trained = solver.train(train_graph)
    evaluation = solver.evaluate(eval_graph, trained.parameters)

    assert evaluation.parameters == trained.parameters
    assert evaluation.num_variables == 160
    assert evaluation.satisfaction_rate >= 0.95
    assert evaluation.normalized_energy <= 0.05
