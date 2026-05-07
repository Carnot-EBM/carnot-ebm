"""Tests for Exp 1486 CCTU executable constraint micro-benchmark.

Spec: REQ-VERIFY-1486, SCENARIO-VERIFY-1486.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import cctu_executable_constraint_microbenchmark as exp


def test_req_verify_1486_builds_20_cases_across_constraint_families() -> None:
    """REQ-VERIFY-1486: benchmark defines 20 auditable executable cases."""

    cases = exp.build_benchmark_cases()
    family_counts: dict[str, int] = {}

    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    for case in cases:
        family_counts[case.family] = family_counts.get(case.family, 0) + 1
        assert case.prompt
        assert case.tool_name in case.prompt
        assert case.expected_final_answer

        result = exp.validate_transcript(case, exp.compliant_transcript_for_case(case))
        assert result["validator_result"]["tool_call_structure_valid"] is True
        assert result["validator_result"]["tool_result_consistent"] is True
        assert result["validator_result"]["final_answer_valid"] is True
        assert result["validator_result"]["verifier_outcome_valid"] is True
        assert result["verifier_result"]["accepted"] is True

    assert family_counts == {
        "arithmetic": 5,
        "graph_path": 5,
        "string_constraint": 5,
        "table_filter": 5,
    }


def test_req_verify_1486_validators_fail_closed_on_bad_transcripts() -> None:
    """REQ-VERIFY-1486: validators detect bad calls, results, answers, and verifier claims."""

    case = exp.build_benchmark_cases()[0]
    bad_payload = json.loads(exp.compliant_transcript_for_case(case))
    bad_payload["tool_call"]["arguments"] = {"operation": "sum", "numbers": [1]}
    bad_payload["tool_result"] = {"value": 999}
    bad_payload["final_answer"] = "999"
    bad_payload["verifier"] = {"accept": True}

    bad = exp.validate_transcript(case, json.dumps(bad_payload))
    checks = bad["validator_result"]

    assert checks["parse_error"] is None
    assert checks["tool_call_structure_valid"] is False
    assert checks["tool_result_consistent"] is False
    assert checks["final_answer_valid"] is False
    assert checks["verifier_outcome_valid"] is False
    assert bad["verifier_result"]["accepted"] is False
    assert bad["verifier_result"]["caught_invalid"] is True

    valid_payload = json.loads(exp.compliant_transcript_for_case(case))
    valid_payload["verifier"] = {"accept": False}
    wrong_verifier = exp.validate_transcript(case, json.dumps(valid_payload))
    assert wrong_verifier["validator_result"]["verifier_outcome_valid"] is False
    assert wrong_verifier["verifier_result"]["accepted"] is False


def test_req_verify_1486_json_extraction_handles_wrappers_and_missing_json() -> None:
    """REQ-VERIFY-1486: extraction tolerates common model wrappers and rejects junk."""

    case = exp.build_benchmark_cases()[1]
    transcript = exp.compliant_transcript_for_case(case)
    wrapped = f"<think>done</think>\n```json\n{transcript}\n```\n"

    assert exp.extract_json_object(wrapped)["final_answer"] == case.expected_final_answer
    assert exp.extract_json_object("") is None
    assert exp.extract_json_object("prefix {not-json} suffix") is None

    missing = exp.validate_transcript(case, "I cannot call tools here.")
    assert missing["validator_result"]["parse_error"] == "no_json_object"
    assert missing["validator_result"]["tool_call_structure_valid"] is False
    assert missing["verifier_result"]["accepted"] is False

    malformed = exp.validate_transcript(case, json.dumps({"final_answer": "51"}))
    assert malformed["validator_result"]["tool_result_error"] == "missing_or_malformed_tool_call"
    bad_tool = exp.validate_transcript(
        case,
        json.dumps(
            {
                "tool_call": {"name": "not.a_tool", "arguments": {}},
                "tool_result": {},
                "final_answer": "51",
                "verifier": {"accept": False},
            }
        ),
    )
    assert bad_tool["validator_result"]["tool_result_error"].startswith("ValueError")


def test_req_verify_1486_tool_executor_is_deterministic_and_bounded() -> None:
    """REQ-VERIFY-1486: local tools execute without network or model dependencies."""

    arithmetic, table, string_case, graph = exp.build_benchmark_cases()[0::5]

    assert exp.execute_tool(arithmetic.tool_name, arithmetic.tool_arguments) == {
        "value": 45
    }
    assert exp.execute_tool(table.tool_name, table.tool_arguments) == {
        "rows": ["bolt", "gasket"]
    }
    assert exp.execute_tool(string_case.tool_name, string_case.tool_arguments) == {
        "value": "tfihstolonrac"
    }
    assert exp.execute_tool(graph.tool_name, graph.tool_arguments) == {
        "cost": 7,
        "path": ["A", "B", "D"],
    }

    with pytest.raises(ValueError, match="unknown tool"):
        exp.execute_tool("not.a_tool", {})
    with pytest.raises(ValueError, match="unknown arithmetic"):
        exp.execute_tool("arithmetic.evaluate", {"operation": "mystery"})
    with pytest.raises(ValueError, match="unknown string"):
        exp.execute_tool("string.transform", {"text": "abc", "operations": [{"op": "bad"}]})
    with pytest.raises(ValueError, match="no path"):
        exp.execute_tool(
            "graph.shortest_path",
            {"edges": [["A", "B", 1]], "start": "A", "end": "Z"},
        )
    with pytest.raises(ValueError, match="unknown family"):
        exp._final_answer_from_result("bad_family", {})

    tuple_case = json.loads(exp.compliant_transcript_for_case(arithmetic))
    tuple_case["tool_result"] = {"value": 45.0}
    assert exp.validate_transcript(arithmetic, json.dumps(tuple_case))["validator_result"][
        "tool_result_consistent"
    ]
    assert exp._canonical((1, 2)) == [1, 2]


def test_req_verify_1486_live_collector_can_be_unit_tested_without_loading_gguf() -> None:
    """REQ-VERIFY-1486: live GGUF collection has injectable resolver/import hooks."""

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            return {"choices": [{"text": '{"final_answer": "stub"}'}]}

        def close(self) -> None:
            self.closed = True

    cases = exp.build_benchmark_cases()[:2]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "gpu": 0}
    ok = exp.collect_live_model_outputs(
        spec,
        cases,
        resolver=lambda _hf_id: "/tmp/fake.gguf",
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    missing = exp.collect_live_model_outputs(
        spec,
        cases,
        resolver=lambda _hf_id: None,
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    import_failed = exp.collect_live_model_outputs(
        {**spec, "model_path": "/tmp/fake.gguf"},
        cases,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
        env_preparer=lambda: {},
    )

    assert ok["summary"]["model_used"] is True
    assert len(ok["rows"]) == 2
    assert ok["rows"][0]["generation_source"] == "live_sota_llamacpp"
    assert FakeLlama.prompts == [case.prompt for case in cases]
    assert missing["summary"]["model_used"] is False
    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"


def test_req_verify_1486_live_collector_reports_load_and_generation_errors() -> None:
    """REQ-VERIFY-1486: live collection records model-load and per-case blockers."""

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("generation failed")

    cases = exp.build_benchmark_cases()[:1]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/tmp/fake.gguf"}
    load_failed = exp.collect_live_model_outputs(
        spec,
        cases,
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )
    generate_failed = exp.collect_live_model_outputs(
        spec,
        cases,
        llama_importer=lambda: (True, GenerateFails, None),
        env_preparer=lambda: {},
    )

    assert load_failed["summary"]["model_used"] is False
    assert "load failed" in load_failed["summary"]["blocker"]
    assert generate_failed["summary"]["model_used"] is False
    assert generate_failed["rows"][0]["blocker"] == "RuntimeError: generation failed"


def test_req_verify_1486_helper_edges_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-1486: helper edge cases stay deterministic and local."""

    assert exp._completion_text("raw") == "raw"
    assert exp._completion_text(None) == ""
    assert exp._completion_text({"choices": []}) == ""
    assert exp._completion_text({"choices": ["bad"]}) == ""
    assert exp._completion_text({"choices": [{"text": "ok"}]}) == "ok"

    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/set")
    env = exp.prepare_llama_environment()
    assert "ld_library_path_after" in env
    assert "/already/set" in env["ld_library_path_after"]

    original_import = builtins.__import__

    class FakeModule:
        class Llama:
            pass

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            return FakeModule
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ok, llama_class, import_error = exp._default_llama_importer()
    assert ok is True
    assert llama_class is FakeModule.Llama
    assert import_error is None

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise RuntimeError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    blocked_ok, blocked_class, blocked_error = exp._default_llama_importer()
    assert blocked_ok is False
    assert blocked_class is None
    assert blocked_error == "RuntimeError: blocked"


def test_scenario_verify_1486_runner_writes_manifest_and_required_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1486: runner writes complete artifact and JSONL rows."""

    def fake_collect(spec: dict[str, Any], cases: list[exp.BenchmarkCase]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for index, case in enumerate(cases):
            if index % 2 == 0:
                output = exp.compliant_transcript_for_case(case)
            else:
                payload = json.loads(exp.compliant_transcript_for_case(case))
                payload["final_answer"] = "wrong"
                payload["verifier"] = {"accept": False}
                output = json.dumps(payload, sort_keys=True)
            rows.append(
                {
                    "case_id": case.case_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec.get("name"),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": output,
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
        rows.append(
            {
                "case_id": "not-a-known-case",
                "model_hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "generation_source": "live_sota_llamacpp",
                "output_text": "{}",
                "elapsed_seconds": 0.01,
                "blocker": None,
            }
        )
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": True,
                "blocker": None,
            },
            "rows": rows,
        }

    output_path = tmp_path / "experiment_1486.json"
    manifest_path = tmp_path / "manifest.jsonl"

    artifact = exp.run_microbenchmark(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260507",
        collect_model_outputs_fn=fake_collect,
        max_models=1,
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["benchmark_cases"] == 20
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["executable_constraint_benchmark_ready"] is True
    assert artifact["tool_call_validity_rate"] == pytest.approx(1.0)
    assert artifact["final_answer_validity_rate"] == pytest.approx(0.5)
    assert artifact["verifier_catch_rate"] == pytest.approx(1.0)
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == 20
    assert rows[0]["prompt"] == exp.build_benchmark_cases()[0].prompt
    assert rows[0]["validator_result"]["final_answer_valid"] is True
    assert len(artifact["model_attempts"]) == 3
    assert artifact["model_attempts"][1]["blocker"] == "not_attempted_runtime_budget"


def test_req_verify_1486_aggregate_metrics_handles_empty_rows() -> None:
    """REQ-VERIFY-1486: aggregate rates are defined even before generation rows exist."""

    assert exp.aggregate_manifest_metrics([]) == {
        "tool_call_validity_rate": 0.0,
        "final_answer_validity_rate": 0.0,
        "verifier_catch_rate": 1.0,
        "verifier_false_accept_rate": 0.0,
    }


def test_req_verify_1486_main_uses_all_models_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1486: CLI prints the same key fields as the artifact."""

    seen: dict[str, int] = {}

    def fake_run(*, max_models: int) -> dict[str, Any]:
        seen["max_models"] = max_models
        return {
            "executable_constraint_benchmark_ready": True,
            "benchmark_cases": 20,
            "models_used": ["m"],
            "final_answer_validity_rate": 0.25,
            "verifier_false_accept_rate": 0.0,
        }

    monkeypatch.setenv("CARNOT_CCTU_1486_MAX_MODELS", "1")
    monkeypatch.setattr(exp, "run_microbenchmark", fake_run)

    assert exp.main(["--all-models"]) == 0
    assert seen["max_models"] == len(exp.MANDATED_MODEL_SPECS)
    assert "ready=True" in capsys.readouterr().out
