"""Tests for the Exp 2353 VERGE MCS repair engine.

Spec: REQ-VERIFY-2353, SCENARIO-VERIFY-2353
"""

from __future__ import annotations

import json
from pathlib import Path

import z3

from carnot.repair.verge_repair import (
    RANDOM_SEED,
    VergeRepairEngine,
    build_experiment_2353_scenarios,
    evaluate_verge_repair_scenarios,
    run_experiment_2353,
)


def test_verge_find_mcs_returns_single_relaxed_claim() -> None:
    """REQ-VERIFY-2353: a single false claim is the minimal correction subset."""

    engine = VergeRepairEngine()
    constraints = [
        z3.IntVal(12) + z3.IntVal(7) == z3.IntVal(20),
        z3.IntVal(19) + z3.IntVal(3) == z3.IntVal(22),
    ]

    assert engine.find_mcs(constraints, violated=[0]) == [0]


def test_verge_find_mcs_handles_multiple_independent_false_claims() -> None:
    """REQ-VERIFY-2353: MCS search grows until remaining constraints are SAT."""

    engine = VergeRepairEngine()
    constraints = [
        z3.IntVal(2) + z3.IntVal(2) == z3.IntVal(5),
        z3.IntVal(3) + z3.IntVal(3) == z3.IntVal(7),
        z3.IntVal(4) + z3.IntVal(4) == z3.IntVal(8),
    ]

    assert engine.find_mcs(constraints, violated=[0, 1]) == [0, 1]


def test_verge_suggest_repair_uses_nsvif_violation_claim() -> None:
    """SCENARIO-VERIFY-2353: NSVIF violation text becomes a concrete edit."""

    engine = VergeRepairEngine()
    response = "We have 4 times 6 equals 25. Thus 24 - 5 = 19."
    violations = engine.extractor.verify(response)["violations"]

    suggestion = engine.suggest_repair(response, violations)

    assert suggestion == "Change '4 * 6 = 25' to '4 * 6 = 24'"


def test_verge_ten_scenario_evaluation_localizes_actual_errors() -> None:
    """REQ-VERIFY-2353: Exp 2353 evaluates exactly ten repair scenarios."""

    scenarios = build_experiment_2353_scenarios()
    metrics = evaluate_verge_repair_scenarios(scenarios)

    assert len(scenarios) == 10
    assert metrics["n_repair_scenarios"] == 10
    assert metrics["mcs_repair_success_rate"] >= 0.50
    assert all(row["mcs_correct"] for row in metrics["case_results"])
    assert all(row["suggestion_correct"] for row in metrics["case_results"])


def test_verge_run_experiment_2353_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2353: Exp 2353 writes the terminal JSON deliverable."""

    artifact_path = tmp_path / "experiment_2353_verge_repair.json"

    payload = run_experiment_2353(artifact_path=artifact_path)
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert persisted == payload
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["verge_repair_validated"] is True
    assert payload["mcs_repair_success_rate"] >= 0.50
    assert payload["n_repair_scenarios"] == 10
    assert payload["random_seed"] == RANDOM_SEED == 42
    assert payload["field_principles"]["honest_verdict"] == "Terminal-prefix required."
