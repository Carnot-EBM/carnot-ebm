"""Tests for Exp 2968 deterministic partial-output monitor harness.

Spec: REQ-VERIFY-2968, SCENARIO-VERIFY-2968.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.eval import interwhen_partial_monitor_harness_v1 as exp


def test_req_verify_2968_spec_anchor_exists() -> None:
    """REQ-VERIFY-2968: the partial monitor harness is OpenSpec anchored."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-2968" in spec
    assert "SCENARIO-VERIFY-2968" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="deterministic_wiring"' in spec


def test_req_verify_2968_event_and_check_vocabularies_are_complete() -> None:
    """REQ-VERIFY-2968: required monitor events and deterministic checks exist."""

    assert exp.MONITOR_EVENTS == (
        "partial_code_block",
        "import_line",
        "function_sig",
        "assertion_or_formula_line",
        "solver_query",
        "final_answer",
    )
    assert set(exp.DETERMINISTIC_CHECKS) == {
        "parser_prefix_validity",
        "import_allow_list",
        "symbol_consistency",
        "schema_field_coverage",
        "z3_parse_check",
    }


def test_req_verify_2968_import_and_symbol_failures_trigger_escalation() -> None:
    """REQ-VERIFY-2968: deterministic monitor failures produce escalation triggers."""

    trace = {
        "trace_id": "bad-code",
        "trace_kind": "code",
        "source_artifact": "synthetic",
        "source_record_id": "synthetic:bad",
        "code": "import os\n\ndef solve(x):\n    return x + 1\n",
        "assertions": ["assert missing(1) == 2"],
        "final_answer": "missing",
    }

    monitored = exp.monitor_trace(trace)
    failed = exp.failed_check_names(monitored)
    escalations = exp.escalation_triggers(monitored)

    assert "import_allow_list" in failed
    assert "symbol_consistency" in failed
    assert "disallowed_import" in escalations
    assert "symbol_inconsistency" in escalations


def test_scenario_verify_2968_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2968: fixture replay writes required ready artifact."""

    _write_source_artifacts(tmp_path)
    artifact_path = tmp_path / "results" / exp.OUTPUT_FILENAME

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=artifact_path,
            started_at=20.0,
            clock=lambda: 21.25,
        )
    )
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["partial_monitor_harness_ready"] is True
    assert artifact["full_streaming_verification_claim"] is False
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert artifact["fixture_trace_count"] >= 5
    assert artifact["fixture_checks_passed"] is True
    assert artifact["duration_s"] == 1.25
    assert artifact["latency_estimate_ms"] > 0.0
    assert set(artifact["monitor_events"]) == set(exp.MONITOR_EVENTS)
    assert set(artifact["deterministic_checks"]) == set(exp.DETERMINISTIC_CHECKS)
    assert all(artifact["coverage_by_event"][event]["count"] > 0 for event in exp.MONITOR_EVENTS)
    assert artifact["escalation_policy"]["target"] == "full_verify_repair_pipeline"
    assert artifact["false_positive_notes"]
    assert "scripts/research_conductor.py" not in artifact["files_changed"]


def test_req_verify_2968_missing_sources_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-2968: missing source artifacts do not fabricate readiness."""

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            started_at=5.0,
            clock=lambda: 5.5,
        )
    )

    assert artifact["partial_monitor_harness_ready"] is False
    assert artifact["fixture_trace_count"] == 0
    assert artifact["fixture_checks_passed"] is False
    assert artifact["honest_verdict"] == "blocked_source_artifacts_missing"
    assert any(source["present"] is False for source in artifact["source_artifacts"])


def test_req_verify_2968_fixture_builder_covers_source_and_synthetic_edges(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2968: fixture building tolerates sparse source records."""

    raw_dir = tmp_path / "results" / "raw" / "experiment_2952"
    raw_dir.mkdir(parents=True)
    for index in range(4):
        (raw_dir / f"code_{index}.txt").write_text(
            f"```python\ndef solve_{index}(x):\n    return x + {index}\n```\n",
            encoding="utf-8",
        )
    _write_json(
        tmp_path / "results" / exp.EXP2952_FILENAME,
        {
            "candidate_evaluations": [
                "malformed-row",
                {"syntax_success": True, "schema_valid": True, "static_checks": {}},
                *[
                    {
                        "task_id": f"task-{index}",
                        "syntax_success": True,
                        "schema_valid": True,
                        "raw_response_ref": str((raw_dir / f"code_{index}.txt").relative_to(tmp_path)),
                        "static_checks": {
                            "unsafe_imports": [],
                            "unsupported_api_calls": [],
                        },
                    }
                    for index in range(4)
                ],
            ]
        },
    )
    _write_json(
        tmp_path / "results" / exp.EXP2959_FILENAME,
        {
            "per_item_results": [
                "malformed-row",
                {
                    "item_id": "fallback-formalization",
                    "parseable": True,
                    "z3_executed": True,
                    "parsed_formalization": "not-a-dict",
                    "solver_answer": "necessary",
                },
                {
                    "item_id": "dict-formalization",
                    "parseable": True,
                    "z3_executed": True,
                    "parsed_formalization": {
                        "facts": [["p", "a"]],
                        "rules": [],
                        "exclusions": [],
                        "query": ["p", "a"],
                    },
                    "solver_answer": "necessary",
                },
                {
                    "item_id": "unused-after-break",
                    "parseable": True,
                    "z3_executed": True,
                    "parsed_formalization": {
                        "facts": [["q", "a"]],
                        "rules": [],
                        "exclusions": [],
                        "query": ["q", "a"],
                    },
                    "solver_answer": "necessary",
                },
            ]
        },
    )

    traces = exp.build_fixture_traces(tmp_path)
    source_code_traces = [
        trace for trace in traces if trace["source_artifact"] == f"results/{exp.EXP2952_FILENAME}"
    ]
    source_logic_traces = [
        trace for trace in traces if trace["source_artifact"] == f"results/{exp.EXP2959_FILENAME}"
    ]
    synthetic_missing = exp.build_fixture_traces(tmp_path / "missing")

    assert len(source_code_traces) == 3
    assert len(source_logic_traces) == 2
    assert source_logic_traces[0]["formalization"]["query"][0] == "is_athlete"
    assert len(synthetic_missing) >= 5


def test_req_verify_2968_defensive_checks_are_deterministic(monkeypatch: Any) -> None:
    """REQ-VERIFY-2968: malformed events fail closed without live inference."""

    unknown = exp.monitor_trace(
        {
            "trace_id": "unknown",
            "trace_kind": "other",
            "source_artifact": "synthetic",
            "source_record_id": "unknown",
            "final_answer": "x",
        }
    )
    malformed_code = exp.monitor_trace(
        {
            "trace_id": "bad-code-syntax",
            "trace_kind": "code",
            "source_artifact": "synthetic",
            "source_record_id": "bad-code",
            "code": "def broken(:\n",
            "assertions": ["assert x == 1"],
            "final_answer": "x",
        }
    )
    malformed_logic = exp.monitor_trace(
        {
            "trace_id": "bad-logic",
            "trace_kind": "logic",
            "source_artifact": "synthetic",
            "source_record_id": "bad-logic",
            "formalization": "not-a-dict",
            "final_answer": "maybe",
        }
    )
    unknown_coverage = exp.coverage_by_event(
        [{"events": [{"event_name": "not_required", "checks": []}]}]
    )
    malformed_coverage = exp.coverage_by_event([malformed_code])

    assert exp.failed_check_names(unknown) == ["schema_field_coverage"]
    assert "parser_prefix_validity" in exp.failed_check_names(malformed_code)
    assert "symbol_consistency" in exp.failed_check_names(malformed_code)
    assert "z3_parse_check" in exp.failed_check_names(malformed_logic)
    assert all(row["count"] == 0 for row in unknown_coverage.values())
    assert malformed_coverage["partial_code_block"]["passed"] is False
    assert "schema_gap" in exp.escalation_policy([unknown])["observed_triggers"]
    assert exp._honest_verdict(True, False) == "blocked_fixture_checks_failed"  # noqa: SLF001
    assert exp._import_allowed("import 1") is False  # noqa: SLF001
    assert exp._json_parse_ok("{") is False  # noqa: SLF001
    assert exp._query_symbol_is_grounded(  # noqa: SLF001
        {"facts": [], "rules": [{"head": ["p", "x"]}], "exclusions": [], "query": ["p", "x"]}
    )

    valid_logic = {
        "trace_id": "logic",
        "trace_kind": "logic",
        "source_artifact": "synthetic",
        "source_record_id": "logic",
        "formalization": {
            "facts": [["p", "a"]],
            "rules": [],
            "exclusions": [],
            "query": ["p", "a"],
        },
        "solver_answer": "necessary",
        "final_answer": "necessary",
    }

    monkeypatch.setattr(
        exp.z3mini,  # noqa: SLF001
        "execute_z3_checks",
        lambda _formalization: (_ for _ in ()).throw(ImportError("z3 missing")),
    )
    optional_z3 = exp.monitor_trace(valid_logic)
    assert optional_z3["checks_passed"] is True

    monkeypatch.setattr(
        exp.z3mini,  # noqa: SLF001
        "execute_z3_checks",
        lambda _formalization: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    runtime_z3 = exp.monitor_trace(valid_logic)
    assert "z3_parse_check" in exp.failed_check_names(runtime_z3)

    monkeypatch.setattr(
        exp.z3mini,  # noqa: SLF001
        "execute_z3_checks",
        lambda _formalization: {"z3_executed": False, "z3_error": "not executed"},
    )
    unexecuted_z3 = exp.monitor_trace(valid_logic)
    assert "z3_parse_failure" in exp.escalation_triggers(unexecuted_z3)


def _write_source_artifacts(root: Path) -> None:
    raw_dir = root / "results" / "raw" / "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1"
    raw_dir.mkdir(parents=True)
    code_path = raw_dir / "code_sample.txt"
    code_path.write_text(
        "```python\n"
        "from typing import Iterable\n\n"
        "def solve_total(values: Iterable[int]) -> int:\n"
        "    return sum(values)\n"
        "```\n",
        encoding="utf-8",
    )

    exp2952 = {
        "artifact": "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1",
        "candidate_evaluations": [
            {
                "task_id": "MBPP:synthetic-total",
                "stable_id": "synthetic-total",
                "mode": "taxonomy_guided",
                "raw_response_ref": str(code_path.relative_to(root)),
                "syntax_success": True,
                "schema_valid": True,
                "passed": True,
                "static_checks": {
                    "status": "passed",
                    "unsafe_imports": [],
                    "unsupported_api_calls": [],
                },
            }
        ],
    }
    exp2959 = {
        "artifact": "experiment_2959_nl_to_z3_execution_repair_mini_v2",
        "per_item_results": [
            {
                "item_id": "logic-synthetic-001",
                "parseable": True,
                "z3_executed": True,
                "parsed_formalization": {
                    "facts": [["is_athlete", "Nia"]],
                    "rules": [],
                    "exclusions": [],
                    "query": ["is_athlete", "Nia"],
                },
                "solver_answer": "necessary",
                "model_answer": "necessary",
                "failure_category": "solver_verified_correct",
            }
        ],
    }
    _write_json(root / "results" / exp.EXP2952_FILENAME, exp2952)
    _write_json(root / "results" / exp.EXP2959_FILENAME, exp2959)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
