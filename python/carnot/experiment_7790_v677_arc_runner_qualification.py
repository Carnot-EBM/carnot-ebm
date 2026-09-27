"""V677 scored ARC runner qualification controls.

The scored agent and archive remain in their existing modules. These checks
bind the repaired validation scope to the actual SDK probe evidence.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any

from carnot.experiment_7776_v676_arc_runner_qualification import gate_decision


def classify_readiness(
    receipts: Sequence[Mapping[str, Any]],
    required: Sequence[str],
    sdk_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Require all named checks and a clean scored transition in each arm."""
    sdk_ok = len(sdk_rows) == 3 and {row.get("arm") for row in sdk_rows} == {
        "off",
        "total",
        "organic",
    }
    sdk_ok = sdk_ok and all(
        row.get("error") is None
        and row.get("counts", {}).get("sdk_transitions", 0) > 0
        and row.get("policy_entry", {}).get("policy_class") == "E3AgentPolicy"
        and all(action.get("induction_attempt_count", 0) == 0 for action in row.get("actions", []))
        for row in sdk_rows
    )
    return gate_decision(receipts, required, sdk_ok=sdk_ok)


def current_metadata(value: Mapping[str, Any], run_date: str) -> dict[str, Any]:
    """Give V677 its own identity and leave unmeasured benefit fields null."""
    if run_date != "20260927":
        raise ValueError("run_date_must_match_milestone")
    current = dict(value)
    gates = dict(current.get("acceptance_gate_results", {}))
    gates.update(
        validity=False,
        readiness=False,
        probability_quality=None,
        decision_benefit=None,
        retention=None,
        efficiency=None,
    )
    current.update(
        schema="carnot.exp7790.v677.arc_runner_qualification.v1",
        experiment_id=7790,
        milestone="2026.09.677",
        run_date=run_date,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        planned_inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts={
            "loads": 0,
            "generations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "loaded_files": [],
        },
        organic_runner_ready_score=0,
        acceptance_gate_results=gates,
    )
    return current


def verify_selector_assertions(
    source: str, *, expected_tests: int = 16, expected_assertions: int = 39
) -> bool:
    """Reject a test repair that drops any of the historical selector assertions."""
    tree = ast.parse(source)
    tests = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if (len(tests), assertions) != (expected_tests, expected_assertions):
        raise ValueError("selector_assertions_changed")
    return True
