"""Tests for Exp 2995 FR-11 verifier-grounded trace memory v2.

Spec: REQ-LEARN-2995,
      SCENARIO-LEARN-2995,
      SCENARIO-LEARN-2995-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_verifier_grounded_trace_memory_v2 as exp


REQUIRED_FIELDS = {
    "independent_self_learning_boundary_preserved",
    "continuous_self_learning_task",
    "trace_memory_ready",
    "n_trace_memories",
    "independent_metric_names",
    "utility_metric_names",
    "no_identical_metric_flag",
    "negative_control_deltas",
    "forgetting_guard_passed",
    "heldout_metric_deltas",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _solver_row(
    item_id: str,
    *,
    skills: list[str],
    initial_parseable: bool,
    initial_answer_correct: bool = False,
    initial_solver_correct: bool = False,
    initial_failure_category: str = "unparseable",
    parse_error: str | None = "no_json_object",
) -> dict[str, Any]:
    return {
        "prompt_hash": "a" * 64,
        "z3_transcript_sha256": "b" * 64,
        "final_z3_input_sha256": "c" * 64,
        "initial_result": {
            "item_id": item_id,
            "answer_correct": initial_answer_correct,
            "solver_formula_correct": initial_solver_correct,
            "parseable": initial_parseable,
            "z3_executed": initial_solver_correct,
            "failure_category": initial_failure_category,
            "parse_error": parse_error,
        },
        "final_result": {
            "item_id": item_id,
            "check_kind": "symbolization",
            "skill_labels": skills,
            "answer_correct": True,
            "solver_formula_correct": True,
            "parseable": True,
            "z3_executed": True,
            "failure_category": "solver_verified_correct",
            "z3_result": {
                "solver_status_matches_expected": True,
                "answer_extraction_matches_expected": True,
            },
        },
    }


def _validator_fixture(
    fixture_id: str,
    *,
    authority: str,
    failure_kind: str,
    rejection_reason: str,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "compiled": True,
        "known_good_feedback": {
            "accepted": True,
            "llm_judge_used": False,
            "failing_node_ids": [],
            "rejection_reasons": [],
            "node_results": [
                {
                    "node_id": f"{fixture_id}:good",
                    "authority": authority,
                    "kind": failure_kind,
                    "accepted": True,
                    "rejection_reason": None,
                }
            ],
        },
        "known_bad_feedback": {
            "accepted": False,
            "llm_judge_used": False,
            "failing_node_ids": [f"{fixture_id}:bad"],
            "rejection_reasons": [rejection_reason],
            "node_results": [
                {
                    "node_id": f"{fixture_id}:bad",
                    "authority": authority,
                    "kind": failure_kind,
                    "accepted": False,
                    "rejection_reason": rejection_reason,
                }
            ],
        },
        "validator_tree": {
            "tree_id": fixture_id,
            "nodes": [{"node_id": f"{fixture_id}:bad", "authority": authority}],
        },
    }


def _write_ready_inputs(root: Path) -> None:
    _write_json(
        root,
        exp.EXP2982_REL_PATH,
        {
            "honest_verdict": "complete: fr11_independent_self_learning_ready",
            "continuous_self_learning_task": True,
            "fr11_independent_self_learning_ready": True,
            "forgetting_guard_passed": True,
            "no_identical_metric_flag": True,
            "independent_metrics": [{"name": name} for name in exp.independent_metric_names()],
        },
    )
    _write_json(
        root,
        exp.EXP2983_REL_PATH,
        {
            "honest_verdict": "complete: trace_to_skill_memory_ready",
            "continuous_self_learning_task": True,
            "trace_to_skill_memory_ready": True,
            "headline_result": False,
        },
    )
    _write_json(
        root,
        exp.EXP2992_REL_PATH,
        {
            "honest_verdict": "reproduced: solver provenance ready",
            "solver_provenance_reproduced": True,
            "formalization_clean": True,
            "per_item_results": [
                _solver_row(
                    "train-1",
                    skills=["symbolization", "validity"],
                    initial_parseable=False,
                ),
                _solver_row(
                    "train-2",
                    skills=["symbolization", "satisfiability"],
                    initial_parseable=True,
                    initial_answer_correct=True,
                    initial_failure_category="z3_exception",
                    parse_error=None,
                ),
                _solver_row(
                    "heldout-1",
                    skills=["symbolization", "validity"],
                    initial_parseable=False,
                ),
                _solver_row(
                    "heldout-2",
                    skills=["symbolization", "satisfiability"],
                    initial_parseable=True,
                    initial_answer_correct=True,
                    initial_failure_category="z3_exception",
                    parse_error=None,
                ),
                _solver_row(
                    "heldout-3",
                    skills=["symbolization", "schema"],
                    initial_parseable=False,
                    parse_error="missing_schema_field:variables",
                ),
            ],
        },
    )
    _write_json(
        root,
        exp.EXP2994_REL_PATH,
        {
            "honest_verdict": "complete: prompt-validator dialogue protocol ready",
            "prompt_validator_protocol_ready": True,
            "exact_verifier_authority_preserved": True,
            "validator_tree_fixtures": [
                _validator_fixture(
                    "json-final-answer-confidence",
                    authority="runtime_json_parser",
                    failure_kind="json_number_between",
                    rejection_reason="numeric_range_violation",
                ),
                _validator_fixture(
                    "python-normalize-slug-ast",
                    authority="python_ast_parser",
                    failure_kind="python_function_signature",
                    rejection_reason="function_signature_mismatch",
                ),
                _validator_fixture(
                    "z3-linear-integer-assignment",
                    authority="z3_solver",
                    failure_kind="z3_linear_integer_relation",
                    rejection_reason="z3_unsatisfied",
                ),
            ],
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=100.0,
        clock=lambda: 103.25,
        tests_run=("focused-req-2995",),
    )


def test_req_learn_2995_spec_anchor_exists() -> None:
    """REQ-LEARN-2995: OpenSpec declares verifier-grounded trace memory v2."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/self-learning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-LEARN-2995" in spec
    assert "SCENARIO-LEARN-2995" in spec
    assert "SCENARIO-LEARN-2995-BLOCKED" in spec
    assert exp.OUTPUT_FILENAME in spec


def test_scenario_learn_2995_writes_ready_trace_memory_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2995: verifier-grounded memories improve held-out metrics."""

    _write_ready_inputs(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == "ready: verifier_grounded_trace_memory_ready"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["independent_self_learning_boundary_preserved"] is True
    assert artifact["trace_memory_ready"] is True
    assert 0 < artifact["n_trace_memories"] <= exp.MAX_SELECTED_TRACE_MEMORIES
    assert set(artifact["independent_metric_names"]) == set(exp.independent_metric_names())
    assert set(artifact["utility_metric_names"]) == set(exp.utility_metric_names())
    assert not set(artifact["independent_metric_names"]) & set(artifact["utility_metric_names"])
    assert artifact["no_identical_metric_flag"] is True
    assert artifact["forgetting_guard_passed"] is True
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["selection_rule"]["disable_supported"] is True

    assert all(delta > 0.0 for delta in artifact["heldout_metric_deltas"].values())
    for control in artifact["negative_control_deltas"].values():
        assert all(delta == pytest.approx(0.0) for delta in control.values())
    assert artifact["controls_improve_equally"] is False
    assert artifact["trace_memory_metrics"]["pass_at_1"] > artifact["random_control_metrics"][
        "pass_at_1"
    ]
    assert artifact["trace_memory_metrics"]["verifier_false_accept_rate"] < artifact[
        "random_control_metrics"
    ]["verifier_false_accept_rate"]
    assert {memory["source"] for memory in artifact["selected_trace_memories"]} == {
        "exp2992",
        "exp2994",
    }
    assert artifact["tests_run"] == ["focused-req-2995"]


def test_req_learn_2995_candidates_are_process_verified_and_label_safe(tmp_path: Path) -> None:
    """REQ-LEARN-2995-2/3: memory selection uses exact evidence without leakage."""

    _write_ready_inputs(tmp_path)
    exp2992 = exp.read_json_object(tmp_path / exp.EXP2992_REL_PATH)
    exp2994 = exp.read_json_object(tmp_path / exp.EXP2994_REL_PATH)

    candidates = exp.build_trace_memory_candidates(exp2992, exp2994)
    heldout = exp.build_heldout_tasks(exp2992)
    selected = exp.select_trace_memories(candidates, enabled=True)

    assert candidates
    assert selected
    assert exp.select_trace_memories(candidates, enabled=False) == []
    assert all(exp.validate_trace_memory(memory) == memory for memory in selected)
    assert all(memory["process_verifiable"] is True for memory in selected)
    assert all(memory["selection_utility"]["process_verification_score"] > 0 for memory in selected)
    assert exp.leakage_flag_for(selected, heldout) is False

    leaking = dict(selected[0])
    leaking["reuse_hint"] = f"memorize {heldout[0]['task_id']} expected_solver_status=sat"
    assert exp.leakage_flag_for([leaking], heldout) is True

    incomplete = dict(selected[0])
    incomplete.pop("process_evidence")
    with pytest.raises(ValueError, match="trace memory missing required fields"):
        exp.validate_trace_memory(incomplete)


def test_req_learn_2995_metric_separation_controls_and_directional_deltas(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-2995-3/4/5: controls and deltas gate promotion."""

    _write_ready_inputs(tmp_path)
    exp2992 = exp.read_json_object(tmp_path / exp.EXP2992_REL_PATH)
    candidates = exp.build_trace_memory_candidates(
        exp2992,
        exp.read_json_object(tmp_path / exp.EXP2994_REL_PATH),
    )
    heldout = exp.build_heldout_tasks(exp2992)
    selected = exp.select_trace_memories(candidates, enabled=True)

    random_metrics = exp.evaluate_heldout_metrics(heldout, selected, condition="random_control")
    trace_metrics = exp.evaluate_heldout_metrics(heldout, selected, condition="trace_memory")
    disabled_metrics = exp.evaluate_heldout_metrics(heldout, selected, condition="disabled_update")
    deltas = exp.directional_delta(trace_metrics, random_metrics)
    negative = exp.directional_delta(disabled_metrics, random_metrics)

    assert exp.metrics_improved(deltas) is True
    assert all(delta == pytest.approx(0.0) for delta in negative.values())
    assert exp.controls_improve_equally(deltas, {"disabled": negative}) is False
    assert exp.no_identical_metric_flag(exp.utility_metric_names(), exp.independent_metric_names())
    assert not exp.no_identical_metric_flag(("pass_at_1",), exp.independent_metric_names())

    with pytest.raises(ValueError, match="unknown held-out condition"):
        exp.evaluate_heldout_metrics(heldout, selected, condition="mystery")

    equal_control = {"disabled": dict(deltas)}
    assert exp.controls_improve_equally(deltas, equal_control) is True


def test_scenario_learn_2995_blocked_artifacts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2995-BLOCKED: missing or unready evidence blocks promotion."""

    missing = exp.build_artifact(_config(tmp_path))
    assert REQUIRED_FIELDS <= set(missing)
    assert missing["honest_verdict"] == "blocked_missing_exp2982_independent_boundary"
    assert missing["trace_memory_ready"] is False
    assert missing["n_trace_memories"] == 0
    assert missing["heldout_metric_deltas"] == {}
    assert missing["forgetting_guard_passed"] is False

    _write_ready_inputs(tmp_path)
    _write_json(tmp_path, exp.EXP2982_REL_PATH, {"fr11_independent_self_learning_ready": False})
    not_ready = exp.build_artifact(_config(tmp_path))
    assert not_ready["honest_verdict"] == "blocked_exp2982_independent_boundary_not_ready"

    _write_ready_inputs(tmp_path)
    (tmp_path / exp.EXP2992_REL_PATH).unlink()
    missing_solver = exp.build_artifact(_config(tmp_path))
    assert missing_solver["honest_verdict"] == "blocked_missing_exp2992_solver_traces"

    _write_ready_inputs(tmp_path)
    _write_json(tmp_path, exp.EXP2992_REL_PATH, {"solver_provenance_reproduced": False})
    solver_not_ready = exp.build_artifact(_config(tmp_path))
    assert solver_not_ready["honest_verdict"] == "blocked_exp2992_solver_provenance_not_ready"

    _write_ready_inputs(tmp_path)
    _write_json(tmp_path, exp.EXP2994_REL_PATH, {"prompt_validator_protocol_ready": False})
    validator_not_ready = exp.build_artifact(_config(tmp_path))
    assert validator_not_ready["honest_verdict"] == (
        "blocked_exp2994_validator_protocol_not_ready"
    )


def test_req_learn_2995_validation_and_defensive_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2995-5: validation rejects tautology, control, and schema drift."""

    _write_ready_inputs(tmp_path)
    artifact = exp.build_artifact(_config(tmp_path))

    assert exp.validate_artifact(artifact) == artifact

    incomplete = dict(artifact)
    incomplete.pop("trace_memory_ready")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(incomplete)

    bad_identical = dict(artifact, no_identical_metric_flag=False)
    with pytest.raises(ValueError, match="identical"):
        exp.validate_artifact(bad_identical)

    bad_ready = dict(artifact, trace_memory_ready=True, n_trace_memories=0)
    with pytest.raises(ValueError, match="selected trace memories"):
        exp.validate_artifact(bad_ready)

    bad_control = dict(artifact, controls_improve_equally=True)
    with pytest.raises(ValueError, match="controls improve equally"):
        exp.validate_artifact(bad_control)

    bad_boundary = dict(artifact, independent_self_learning_boundary_preserved=False)
    with pytest.raises(ValueError, match="boundary"):
        exp.validate_artifact(bad_boundary)

    malformed = tmp_path / exp.EXP2992_REL_PATH
    malformed.write_text("{", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}

    with pytest.raises(ValueError, match="process_evidence"):
        exp.validate_trace_memory(
            {
                "memory_id": "trace-bad",
                "source": "exp2992",
                "source_trace_id": "x",
                "trace_kind": "solver",
                "process_signature": "solver::x",
                "process_verifiable": True,
                "process_evidence": "bad",
                "selection_utility": {"process_verification_score": 1.0},
                "reuse_hint": "x",
                "forbidden_label_leakage": [],
            }
        )

    monkeypatch.setattr(exp, "write_artifact", lambda: {})
    assert exp.main() == 0
