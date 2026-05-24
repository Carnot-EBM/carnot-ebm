"""Tests for Exp 2981 Interwhen partial-monitor promotion metrics.

Spec refs: REQ-VERIFY-2981, SCENARIO-VERIFY-2981.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import interwhen_partial_monitor_promotion_v2 as exp


REQUIRED_FIELDS = {
    "honest_verdict",
    "partial_monitor_promoted",
    "full_streaming_verification_claim",
    "event_types",
    "coverage_by_event",
    "prefix_failure_localization_rate",
    "monitor_latency_ms",
    "false_alarm_rate",
    "fixture_count",
    "live_trace_count",
    "promotion_gates",
    "inference_substrate",
    "duration_s",
}


def test_req_verify_2981_spec_anchor_and_event_vocabulary_exist() -> None:
    """REQ-VERIFY-2981: promotion metrics are OpenSpec anchored."""
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-2981" in spec
    assert "SCENARIO-VERIFY-2981" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="deterministic_monitor_harness"' in spec
    assert "full_streaming_verification_claim=false" in spec
    assert exp.EVENT_TYPES == (
        "draft_intent",
        "constraint_emission",
        "parse_boundary",
        "verifier_call",
        "counterexample",
        "repair_step",
    )


def test_scenario_verify_2981_promotes_with_measured_fixture_and_live_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2981: coverage and localization gates control promotion."""
    _write_sources(tmp_path, include_exp2980=True)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["partial_monitor_promoted"] is True
    assert artifact["full_streaming_verification_claim"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(1.5)
    assert artifact["fixture_count"] >= 4
    assert artifact["live_trace_count"] == 2
    assert artifact["event_types"] == list(exp.EVENT_TYPES)
    assert all(artifact["coverage_by_event"][event]["count"] > 0 for event in exp.EVENT_TYPES)
    assert artifact["coverage_by_event"]["draft_intent"]["stream_kinds"] == ["code", "solver"]
    assert artifact["prefix_failure_localization_rate"] == pytest.approx(1.0)
    assert artifact["false_alarm_rate"] == pytest.approx(0.0)
    assert artifact["monitor_latency_ms"]["total"] > 0.0
    assert artifact["promotion_gates"]["event_coverage_broad"]["passed"] is True
    assert artifact["promotion_gates"]["false_alarm_rate_bounded"]["passed"] is True
    assert artifact["promotion_gates"]["full_streaming_claim_supported"]["passed"] is False
    exp.validate_artifact(artifact)


def test_req_verify_2981_uses_deterministic_fixtures_when_exp2980_absent(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2981: absent Exp 2980 leaves live count at zero without blocking."""
    _write_sources(tmp_path, include_exp2980=False)

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["partial_monitor_promoted"] is True
    assert artifact["fixture_count"] >= 4
    assert artifact["live_trace_count"] == 0
    assert artifact["source_artifacts"]["exp2980"]["present"] is False
    assert artifact["full_streaming_verification_claim"] is False
    assert artifact["promotion_gates"]["stream_coverage_broad"]["passed"] is True


def test_req_verify_2981_blocks_when_exp2979_frontier_is_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2981: Exp 2979 frontier readiness is a hard precondition."""
    _write_sources(tmp_path, exp2979_ready=False, include_exp2980=True)

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_precondition: exp2979_frontier_upgrade_not_ready"
    assert artifact["partial_monitor_promoted"] is False
    assert artifact["fixture_count"] == 0
    assert artifact["live_trace_count"] == 0
    assert artifact["prefix_failure_localization_rate"] == pytest.approx(0.0)
    assert artifact["false_alarm_rate"] == pytest.approx(0.0)
    assert artifact["promotion_gates"]["exp2979_frontier_upgrade_ready"]["passed"] is False

    missing_sources = exp.build_artifact(exp.ExperimentConfig(repo_root=tmp_path / "missing"))
    assert missing_sources["honest_verdict"] == "blocked_precondition: exp2979_frontier_upgrade_not_ready"

    sparse_root = tmp_path / "sparse"
    _write_json(
        sparse_root / "results" / exp.EXP2979_FILENAME,
        {
            "honest_verdict": "complete: sparse fixture",
            "frontier_upgrade_ready": True,
            "frontier_items": [],
        },
    )
    sparse = exp.build_artifact(exp.ExperimentConfig(repo_root=sparse_root))
    assert sparse["honest_verdict"] == "complete: deterministic partial monitor measured but not promoted"


def test_req_verify_2981_gates_and_validation_fail_closed() -> None:
    """REQ-VERIFY-2981: false alarms, late localization, and bad schemas fail closed."""
    traces = [
        {
            "trace_id": "ok-but-flagged",
            "stream_kind": "code",
            "trace_source": "fixture",
            "events": [
                {"event_type": "draft_intent", "expected_issue": False, "monitor_flag": False},
                {"event_type": "constraint_emission", "expected_issue": False, "monitor_flag": True},
            ],
        },
        {
            "trace_id": "late-failure",
            "stream_kind": "solver",
            "trace_source": "fixture",
            "events": [
                {"event_type": "draft_intent", "expected_issue": True, "monitor_flag": False},
                {"event_type": "repair_step", "expected_issue": False, "monitor_flag": True},
                {"event_type": "not_required", "expected_issue": False, "monitor_flag": False},
            ],
        },
    ]
    monitored = [exp.monitor_trace(trace) for trace in traces]
    metrics = exp.compute_metrics(monitored, exp2979_ready=True)

    assert metrics["false_alarm_rate"] == pytest.approx(0.5)
    assert metrics["prefix_failure_localization_rate"] == pytest.approx(0.0)
    assert metrics["promotion_gates"]["false_alarm_rate_bounded"]["passed"] is False
    assert metrics["promotion_gates"]["prefix_failure_localization_rate"]["passed"] is False
    assert metrics["partial_monitor_promoted"] is False

    valid = {
        "honest_verdict": "complete: deterministic partial monitor measured but not promoted",
        "partial_monitor_promoted": False,
        "full_streaming_verification_claim": False,
        "event_types": list(exp.EVENT_TYPES),
        "coverage_by_event": metrics["coverage_by_event"],
        "prefix_failure_localization_rate": metrics["prefix_failure_localization_rate"],
        "monitor_latency_ms": metrics["monitor_latency_ms"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "fixture_count": 2,
        "live_trace_count": 0,
        "promotion_gates": metrics["promotion_gates"],
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "duration_s": 0.1,
    }
    exp.validate_artifact(valid)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "x"})
    with pytest.raises(ValueError, match="full_streaming_verification_claim"):
        exp.validate_artifact(valid | {"full_streaming_verification_claim": True})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(valid | {"inference_substrate": "deterministic_wiring"})
    with pytest.raises(ValueError, match="event_types"):
        exp.validate_artifact(valid | {"event_types": ["draft_intent"]})
    with pytest.raises(ValueError, match="promotion_gates"):
        exp.validate_artifact(valid | {"promotion_gates": []})
    with pytest.raises(ValueError, match="partial_monitor_promoted"):
        exp.validate_artifact(valid | {"partial_monitor_promoted": True})

    malformed = exp.deterministic_fixture_traces(
        {"monitor_results": [{"trace_id": "malformed", "events": ["not-a-dict"]}]},
        {"frontier_items": []},
    )
    assert malformed[0]["trace_id"] == "fixture:malformed"


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 11.5,
    )


def _write_sources(
    root: Path,
    *,
    exp2979_ready: bool = True,
    include_exp2980: bool,
) -> None:
    _write_json(
        root / "results" / exp.EXP2968_FILENAME,
        {
            "honest_verdict": "complete: fixture",
            "partial_monitor_harness_ready": True,
            "full_streaming_verification_claim": False,
            "monitor_results": [
                {
                    "trace_id": "code-ok",
                    "trace_kind": "code",
                    "checks_passed": True,
                    "events": [
                        {
                            "event_name": "partial_code_block",
                            "checks": [{"check_name": "parser_prefix_validity", "passed": True}],
                        },
                        {
                            "event_name": "function_sig",
                            "checks": [{"check_name": "schema_field_coverage", "passed": True}],
                        },
                        {
                            "event_name": "assertion_or_formula_line",
                            "checks": [{"check_name": "symbol_consistency", "passed": True}],
                        },
                        {
                            "event_name": "final_answer",
                            "checks": [{"check_name": "symbol_consistency", "passed": True}],
                        },
                    ],
                },
                {
                    "trace_id": "code-bad-import",
                    "trace_kind": "code",
                    "checks_passed": False,
                    "events": [
                        {
                            "event_name": "partial_code_block",
                            "checks": [{"check_name": "parser_prefix_validity", "passed": True}],
                        },
                        {
                            "event_name": "import_line",
                            "checks": [{"check_name": "import_allow_list", "passed": False}],
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        root / "results" / exp.EXP2979_FILENAME,
        {
            "honest_verdict": "complete: fixture" if exp2979_ready else "blocked: fixture",
            "frontier_upgrade_ready": exp2979_ready,
            "frontier_items": [
                _frontier_item("solver-parse", parse_error="no_json_object"),
                _frontier_item("solver-counter", model_counterexample={"x": "1"}),
                _frontier_item("solver-z3", z3_exception="Z3Exception: parser error"),
            ],
        },
    )
    if include_exp2980:
        _write_json(
            root / "results" / exp.EXP2980_FILENAME,
            {
                "honest_verdict": "complete: fixture",
                "per_item_results": [
                    {
                        "item_id": "live-parse",
                        "repair_attempted": True,
                        "solver_feedback": _frontier_item("live-parse", parse_error="no_json_object")[
                            "solver_feedback"
                        ],
                        "initial_result": {
                            "parseable": False,
                            "z3_executed": False,
                            "parse_error": "no_json_object",
                            "failure_category": "unparseable",
                        },
                        "final_result": {
                            "parseable": True,
                            "z3_executed": True,
                            "solver_formula_correct": True,
                            "answer_correct": True,
                            "z3_result": {"z3_executed": True, "z3_error": None},
                        },
                    },
                    {
                        "item_id": "live-clean",
                        "repair_attempted": False,
                        "solver_feedback": _frontier_item("live-clean")["solver_feedback"],
                        "initial_result": {
                            "parseable": True,
                            "z3_executed": True,
                            "solver_formula_correct": True,
                            "answer_correct": True,
                            "z3_result": {"z3_executed": True, "z3_error": None},
                        },
                        "final_result": {
                            "parseable": True,
                            "z3_executed": True,
                            "solver_formula_correct": True,
                            "answer_correct": True,
                            "z3_result": {"z3_executed": True, "z3_error": None},
                        },
                    },
                ],
            },
        )


def _frontier_item(
    item_id: str,
    *,
    parse_error: str | None = None,
    z3_exception: str | None = None,
    model_counterexample: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "prompt": f"Formalize {item_id}",
        "skill_label": "symbolization",
        "skill_labels": ["symbolization", "satisfiability"],
        "expected_solver_status": "sat",
        "accepted_reference_formalization": {
            "format": "smt2",
            "assertions": "(declare-const x Int)\n(assert (= x 1))\n",
            "expected_solver_status": "sat",
            "expected_answer_values": {},
        },
        "solver_feedback": {
            "parse_error": parse_error,
            "z3_exception": z3_exception,
            "model_counterexample": model_counterexample,
            "unsat_core_or_mus": {"unsat_core": ["premise"]} if parse_error else None,
            "minimal_correction_hint": "Preserve symbols before repairing.",
            "skill_label": "symbolization",
            "accepted_reference_formalization": {
                "format": "smt2",
                "assertions": "(declare-const x Int)\n(assert (= x 1))\n",
                "expected_solver_status": "sat",
                "expected_answer_values": {},
            },
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
