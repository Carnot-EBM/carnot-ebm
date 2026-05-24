"""Tests for Exp 2983 trace-to-skill repair memory pilot.

Spec: REQ-LEARN-2983,
      SCENARIO-LEARN-2983,
      SCENARIO-LEARN-2983-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import trace_to_skill_repair_memory_pilot_v1 as exp


REQUIRED_FIELDS = {
    "honest_verdict",
    "continuous_self_learning_task",
    "trace_to_skill_memory_ready",
    "headline_result",
    "pilot_source",
    "models_used",
    "mandatory_headline_model_ids",
    "memory_schema",
    "extracted_memory_count",
    "heldout_task_count",
    "no_memory_metrics",
    "random_memory_metrics",
    "trace_memory_metrics",
    "heldout_skill_reuse_delta",
    "leakage_flag",
    "negative_control_delta",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repair_row(
    task_id: str,
    mode: str,
    *,
    passed: bool,
    schema_valid: bool = True,
    syntax_success: bool = True,
    category: str = "syntax_error",
    runtime_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "stable_id": task_id.lower().replace(":", "-"),
        "mode": mode,
        "corpus": "MBPP",
        "sample_id": f"{task_id}:sample",
        "passed": passed,
        "schema_valid": schema_valid,
        "syntax_success": syntax_success,
        "verifier_accepted": passed,
        "false_accept": False,
        "original_failure_categories": [category],
        "schema_errors": [] if schema_valid else ["missing repaired_code"],
        "schema_diagnostics": {
            "schema_valid": schema_valid,
            "schema_errors": [] if schema_valid else ["missing repaired_code"],
        },
        "syntax_diagnostics": {
            "syntax_success": syntax_success,
            "syntax_errors": [] if syntax_success else ["invalid syntax"],
        },
        "runtime_trace": runtime_trace or [],
        "runtime_trace_present": bool(runtime_trace),
        "verifier_output": {"accepted_by_verifier": passed, "score": 1.0 if passed else 0.0},
    }


def _write_ready_inputs(root: Path, *, include_exp2977: bool = True) -> None:
    _write_json(
        root,
        exp.EXP2976_REL_PATH,
        {
            "honest_verdict": "complete: protocol ready",
            "trace_execution_plan_ready": True,
            "mandatory_headline_model_ids": list(exp.MANDATORY_HEADLINE_MODEL_IDS),
        },
    )
    if include_exp2977:
        _write_json(
            root,
            exp.EXP2977_REL_PATH,
            {
                "honest_verdict": "blocked_cached_sota_pair_unavailable_cpu_smoke_only",
                "headline_result": False,
                "models_used": ["Qwen/Qwen3.5-0.8B"],
                "mandatory_headline_model_ids": list(exp.MANDATORY_HEADLINE_MODEL_IDS),
                "candidate_evaluations": [
                    _repair_row(
                        "MBPP:train-syntax",
                        "baseline",
                        passed=False,
                        schema_valid=True,
                        syntax_success=False,
                        category="syntax_error",
                    ),
                    _repair_row(
                        "MBPP:train-schema",
                        "schema_only_dccd",
                        passed=False,
                        schema_valid=False,
                        syntax_success=False,
                        category="schema_error",
                    ),
                    _repair_row("MBPP:train-passing", "baseline", passed=True),
                ],
            },
        )
    _write_json(
        root,
        exp.EXP2964_REL_PATH,
        {
            "honest_verdict": "complete: DCCD repair replication did not promote",
            "headline_models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
            "candidate_evaluations": [
                _repair_row(
                    "MBPP:train-syntax",
                    "baseline_no_taxonomy",
                    passed=False,
                    schema_valid=True,
                    syntax_success=False,
                    category="syntax_error",
                ),
                _repair_row(
                    "MBPP:train-schema",
                    "dccd_structured",
                    passed=False,
                    schema_valid=False,
                    syntax_success=False,
                    category="schema_error",
                ),
                _repair_row(
                    "MBPP:heldout-a",
                    "baseline_no_taxonomy",
                    passed=False,
                    syntax_success=False,
                    category="syntax_error",
                ),
                _repair_row(
                    "MBPP:heldout-a",
                    "taxonomy_guided",
                    passed=True,
                    category="syntax_error",
                ),
                _repair_row(
                    "MBPP:heldout-b",
                    "baseline_no_taxonomy",
                    passed=False,
                    schema_valid=False,
                    syntax_success=False,
                    category="schema_error",
                ),
                _repair_row(
                    "MBPP:heldout-b",
                    "taxonomy_guided",
                    passed=True,
                    category="schema_error",
                ),
                _repair_row("MBPP:heldout-c", "baseline_no_taxonomy", passed=True),
                _repair_row("MBPP:heldout-c", "taxonomy_guided", passed=True),
            ],
        },
    )
    _write_json(
        root,
        exp.EXP2968_REL_PATH,
        {
            "honest_verdict": "complete: deterministic partial monitor harness ready",
            "partial_monitor_harness_ready": True,
            "monitor_results": [
                {
                    "trace_id": "monitor-code",
                    "trace_kind": "code",
                    "checks_passed": True,
                    "events": [
                        {
                            "event_name": "import_line",
                            "payload": {"line": "from typing import Iterable"},
                            "checks": [
                                {"check_name": "import_allow_list", "passed": True},
                                {"check_name": "parser_prefix_validity", "passed": True},
                            ],
                        },
                        {
                            "event_name": "function_sig",
                            "payload": {"function_name": "solve"},
                            "checks": [{"check_name": "symbol_consistency", "passed": True}],
                        },
                    ],
                },
                {
                    "trace_id": "monitor-logic",
                    "trace_kind": "logic",
                    "checks_passed": True,
                    "events": [
                        {
                            "event_name": "solver_query",
                            "payload": {"query": ["p", "a"]},
                            "checks": [{"check_name": "z3_parse_check", "passed": True}],
                        }
                    ],
                },
            ],
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=30.0,
        clock=lambda: 33.5,
        tests_run=("focused-req-2983",),
    )


def test_req_learn_2983_spec_anchor_exists() -> None:
    """REQ-LEARN-2983: OpenSpec declares trace-to-skill memory replay."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/self-learning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-LEARN-2983" in spec
    assert "SCENARIO-LEARN-2983" in spec
    assert "SCENARIO-LEARN-2983-BLOCKED" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="artifact_replay_and_optional_live_llm"' in spec


def test_scenario_learn_2983_writes_ready_trace_memory_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2983: trace memories improve held-out replay without leakage."""

    _write_ready_inputs(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == "complete: trace_to_skill_memory_ready"
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["trace_to_skill_memory_ready"] is True
    assert artifact["headline_result"] is False
    assert artifact["pilot_source"] == ".280"
    assert artifact["models_used"] == [
        "Qwen/Qwen3.5-0.8B",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert artifact["mandatory_headline_model_ids"] == list(exp.MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(3.5)

    assert artifact["memory_schema"]["schema_version"] == exp.MEMORY_SCHEMA_VERSION
    assert set(exp.MEMORY_REQUIRED_FIELDS) <= set(artifact["memory_schema"]["required"])
    assert artifact["extracted_memory_count"] >= 5
    assert artifact["heldout_task_count"] == 3
    assert artifact["leakage_flag"] is False
    assert artifact["negative_control_delta"] == pytest.approx(0.0)
    assert artifact["heldout_skill_reuse_delta"] > 0.0
    assert (
        artifact["trace_memory_metrics"]["task_success_rate"]
        > artifact["random_memory_metrics"]["task_success_rate"]
    )
    assert artifact["random_memory_metrics"]["task_success_count"] == artifact[
        "no_memory_metrics"
    ]["task_success_count"]
    assert artifact["random_memory_metrics"]["task_success_rate"] == artifact[
        "no_memory_metrics"
    ]["task_success_rate"]
    assert artifact["leakage_audit"]["train_heldout_intersection"] == []
    assert artifact["leakage_audit"]["selection_metric_reused_as_heldout_metric"] is False
    assert artifact["source_artifacts"]["exp2976"]["present"] is True
    assert artifact["tests_run"] == ["focused-req-2983"]


def test_req_learn_2983_memory_schema_and_leakage_controls(tmp_path: Path) -> None:
    """REQ-LEARN-2983-3/5: memories are schema-valid and label leakage is detected."""

    _write_ready_inputs(tmp_path)
    exp2977 = exp.read_json_object(tmp_path / exp.EXP2977_REL_PATH)
    exp2968 = exp.read_json_object(tmp_path / exp.EXP2968_REL_PATH)
    memories = exp.extract_skill_memories(
        repair_payload=exp2977,
        monitor_payload=exp2968,
        repair_source="exp2977",
    )
    heldout = exp.build_heldout_tasks(
        exp.read_json_object(tmp_path / exp.EXP2964_REL_PATH),
        excluded_task_ids=exp.extraction_task_ids(memories),
    )

    assert memories
    assert heldout
    assert all(exp.validate_memory(memory) == memory for memory in memories)
    assert all("passed" not in json.dumps(memory) for memory in memories)
    assert exp.leakage_flag_for(memories, heldout) is False

    leaking = dict(memories[0])
    leaking["minimal_fix_pattern"] = f"memorize heldout {heldout[0]['task_id']} passed=True"
    assert exp.leakage_flag_for([leaking], heldout) is True

    incomplete = dict(memories[0])
    incomplete.pop("failure_signature")
    with pytest.raises(ValueError, match="memory missing required fields"):
        exp.validate_memory(incomplete)


def test_req_learn_2983_falls_back_to_exp2964_when_exp2977_absent(tmp_path: Path) -> None:
    """REQ-LEARN-2983-2: absent Exp 2977 uses Exp 2964 and marks pilot_source=.279."""

    _write_ready_inputs(tmp_path, include_exp2977=False)

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["pilot_source"] == ".279"
    assert artifact["trace_to_skill_memory_ready"] is True
    assert artifact["source_artifacts"]["exp2977"]["present"] is False
    assert artifact["extraction_source"] == "exp2964"
    assert artifact["heldout_task_count"] >= 1


def test_scenario_learn_2983_blocked_artifacts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2983-BLOCKED: missing protocol readiness blocks readiness."""

    missing = exp.build_artifact(_config(tmp_path))
    assert missing["honest_verdict"] == "blocked_missing_exp2976_protocol"
    assert REQUIRED_FIELDS <= set(missing)
    assert missing["trace_to_skill_memory_ready"] is False
    assert missing["extracted_memory_count"] == 0
    assert missing["heldout_task_count"] == 0
    assert missing["leakage_flag"] is True
    assert missing["inference_substrate"] == exp.INFERENCE_SUBSTRATE

    _write_ready_inputs(tmp_path)
    _write_json(tmp_path, exp.EXP2976_REL_PATH, {"trace_execution_plan_ready": False})
    not_ready = exp.build_artifact(_config(tmp_path))
    assert not_ready["honest_verdict"] == "blocked_exp2976_trace_execution_plan_not_ready"

    _write_ready_inputs(tmp_path)
    (tmp_path / exp.EXP2968_REL_PATH).unlink()
    missing_monitor = exp.build_artifact(_config(tmp_path))
    assert missing_monitor["trace_to_skill_memory_ready"] is True
    assert missing_monitor["source_artifacts"]["exp2968"]["present"] is False

    only_protocol = tmp_path / "only-protocol"
    _write_json(
        only_protocol,
        exp.EXP2976_REL_PATH,
        {"trace_execution_plan_ready": True},
    )
    missing_heldout = exp.build_artifact(_config(only_protocol))
    assert missing_heldout["honest_verdict"] == "blocked_missing_exp2964_heldout_source"


def test_req_learn_2983_validation_rejects_schema_and_claim_drift(tmp_path: Path) -> None:
    """REQ-LEARN-2983-5: artifact validation enforces fields and claim boundaries."""

    _write_ready_inputs(tmp_path)
    artifact = exp.build_artifact(_config(tmp_path))

    assert exp.validate_artifact(artifact) == artifact

    incomplete = dict(artifact)
    incomplete.pop("trace_memory_metrics")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(incomplete)

    bad_substrate = dict(artifact, inference_substrate="live_llm_inference")
    with pytest.raises(ValueError, match="substrate"):
        exp.validate_artifact(bad_substrate)

    bad_ready = dict(artifact, trace_to_skill_memory_ready=True, leakage_flag=True)
    with pytest.raises(ValueError, match="leakage"):
        exp.validate_artifact(bad_ready)

    bad_headline = dict(artifact, headline_result=True)
    with pytest.raises(ValueError, match="headline"):
        exp.validate_artifact(bad_headline)

    assert exp.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}

    assert exp._round(0.1234567899) == pytest.approx(0.12345679)


def test_req_learn_2983_defensive_branches_and_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-2983-3/5: defensive branches fail closed under malformed inputs."""

    noisy_repair = {
        "candidate_evaluations": [
            {"mode": "baseline", "passed": False},
            _repair_row(
                "MBPP:dup",
                "baseline",
                passed=False,
                syntax_success=False,
                category="syntax_error",
            ),
            _repair_row(
                "MBPP:dup",
                "baseline",
                passed=False,
                syntax_success=False,
                category="syntax_error",
            ),
            _repair_row(
                "MBPP:runtime",
                "baseline",
                passed=False,
                runtime_trace=[{"command": "pytest", "exit_code": 1}],
                category="runtime_error",
            ),
            _repair_row(
                "MBPP:limit",
                "baseline",
                passed=False,
                syntax_success=False,
                category="syntax_error",
            ),
        ]
    }
    noisy_monitor = {
        "monitor_results": [
            {
                "trace_id": "bad-events",
                "trace_kind": "code",
                "events": [
                    {"checks": []},
                    {
                        "event_name": "function_sig",
                        "checks": [{"check_name": "symbol_consistency"}],
                    },
                    {
                        "event_name": "function_sig",
                        "checks": [{"check_name": "symbol_consistency"}],
                    },
                    {"event_name": "final_answer", "checks": []},
                ],
            }
        ]
    }
    memories = exp.extract_skill_memories(
        repair_payload=noisy_repair,
        monitor_payload=noisy_monitor,
        repair_source="exp-test",
    )
    signatures = [memory["failure_signature"] for memory in memories]

    assert signatures.count("repair::syntax_invalid") == 1
    assert "repair::runtime_trace_failure" in signatures
    assert signatures.count("partial_monitor::function_sig") == 1
    assert "partial_monitor::final_answer" in signatures

    runtime_row = _repair_row(
        "MBPP:r",
        "baseline",
        passed=False,
        runtime_trace=[{"command": "pytest", "exit_code": 1}],
        category="runtime_error",
    )
    false_accept_row = dict(_repair_row("MBPP:f", "baseline", passed=False))
    false_accept_row["false_accept"] = True
    false_accept_row["original_failure_categories"] = []
    verifier_row = {
        "task_id": "MBPP:v",
        "passed": False,
        "schema_valid": True,
        "syntax_success": True,
        "verifier_accepted": False,
        "original_failure_categories": [],
    }
    unknown_row = {
        "task_id": "MBPP:u",
        "passed": False,
        "schema_valid": True,
        "syntax_success": True,
        "verifier_accepted": True,
        "original_failure_categories": [],
    }

    assert exp.failure_signatures_for_row(runtime_row) == ["repair::runtime_trace_failure"]
    assert exp.failure_signatures_for_row(false_accept_row) == ["repair::false_accept_risk"]
    assert exp.failure_signatures_for_row(verifier_row) == ["repair::verifier_rejected"]
    assert exp.failure_signatures_for_row(unknown_row) == ["repair::unknown_failed_trace"]
    assert "runtime trace" in exp.verifier_feedback_for_row(
        runtime_row, "repair::runtime_trace_failure"
    )
    assert "false-accept" in exp.verifier_feedback_for_row(
        false_accept_row, "repair::false_accept_risk"
    )
    assert "threshold" in exp.verifier_feedback_for_row(verifier_row, "repair::verifier_rejected")
    assert "failed trace" in exp.verifier_feedback_for_row(
        unknown_row, "repair::unknown_failed_trace"
    )
    assert exp.minimal_fix_pattern_for("unknown::signature").startswith("reuse")

    with pytest.raises(ValueError, match="field is empty"):
        exp.validate_memory(
            {
                "failure_signature": "",
                "verifier_feedback": "x",
                "minimal_fix_pattern": "x",
                "applicability_conditions": {},
                "forbidden_label_leakage": [],
            }
        )
    with pytest.raises(ValueError, match="applicability_conditions"):
        exp.validate_memory(
            {
                "failure_signature": "x",
                "verifier_feedback": "x",
                "minimal_fix_pattern": "x",
                "applicability_conditions": "bad",
                "forbidden_label_leakage": [],
            }
        )
    with pytest.raises(ValueError, match="forbidden_label_leakage"):
        exp.validate_memory(
            {
                "failure_signature": "x",
                "verifier_feedback": "x",
                "minimal_fix_pattern": "x",
                "applicability_conditions": {},
                "forbidden_label_leakage": "bad",
            }
        )

    heldout = exp.build_heldout_tasks(
        {
            "candidate_evaluations": [
                _repair_row("MBPP:no-baseline", "taxonomy_guided", passed=True),
                _repair_row("MBPP:real", "baseline_no_taxonomy", passed=False),
            ]
        },
        excluded_task_ids=set(),
    )
    assert [task["task_id"] for task in heldout] == ["MBPP:real"]
    with pytest.raises(ValueError, match="unknown replay condition"):
        exp.evaluate_replay(heldout, memories=(), condition="mystery")

    leak_memory = {
        "failure_signature": "repair::syntax_invalid",
        "verifier_feedback": "x",
        "minimal_fix_pattern": "expected_output should never appear",
        "applicability_conditions": {},
        "forbidden_label_leakage": [],
        "source_task_id": "MBPP:real",
    }
    assert exp.leakage_flag_for([leak_memory], heldout) is True
    no_id_leak = dict(leak_memory, source_task_id="MBPP:other")
    assert exp.leakage_flag_for([no_id_leak], [{"task_id": "MBPP:different"}]) is True

    _write_ready_inputs(tmp_path)
    artifact = exp.build_artifact(_config(tmp_path))
    for patch, message in [
        ({"heldout_skill_reuse_delta": 0.0}, "positive heldout delta"),
        ({"negative_control_delta": 0.1}, "negative control"),
        ({"extracted_memory_count": 0}, "extracted memories"),
        ({"heldout_task_count": 0}, "held-out tasks"),
        ({"memory_schema": {"required": []}}, "memory schema"),
    ]:
        bad = dict(artifact, **patch)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad)

    monkeypatch.setattr(exp, "write_artifact", lambda: {})
    assert exp.main() == 0
