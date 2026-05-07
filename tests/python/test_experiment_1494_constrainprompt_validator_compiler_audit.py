"""Tests for Exp 1494 bounded ConstrainPrompt validator compiler audit.

Spec: REQ-VERIFY-1494, SCENARIO-VERIFY-1494.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import constrainprompt_validator_compiler_audit as exp


def test_req_verify_1494_builds_30_prompt_set_from_cctu_plus_new_cases() -> None:
    """REQ-VERIFY-1494: audit prompt set has 30 fixed CCTU-style prompts."""

    prompts = exp.build_prompt_set()
    source_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}

    assert len(prompts) == 30
    assert len({prompt.prompt_id for prompt in prompts}) == 30

    for prompt in prompts:
        source_counts[prompt.source] = source_counts.get(prompt.source, 0) + 1
        family_counts[prompt.family] = family_counts.get(prompt.family, 0) + 1
        assert prompt.prompt_id in prompt.prompt
        assert prompt.known_good_output
        assert prompt.known_bad_output

    assert source_counts["exp1486"] == 20
    assert source_counts["exp1494_new"] == 10
    assert family_counts["arithmetic"] >= 8
    assert family_counts["json_schema"] == 3
    assert family_counts["simple_code"] == 2
    assert family_counts["graph_path"] >= 7


def test_req_verify_1494_compiler_uses_safe_dsl_for_supported_prompt_families() -> None:
    """REQ-VERIFY-1494: supported validators compile and reject known-bad outputs."""

    prompts = exp.build_prompt_set()
    samples = [
        next(prompt for prompt in prompts if prompt.prompt_id == "cctu-1486-arith-001"),
        next(prompt for prompt in prompts if prompt.prompt_id == "cctu-1494-arith-001"),
        next(prompt for prompt in prompts if prompt.prompt_id == "cctu-1494-schema-001"),
        next(prompt for prompt in prompts if prompt.prompt_id == "cctu-1494-code-001"),
        next(prompt for prompt in prompts if prompt.prompt_id == "cctu-1494-graph-001"),
    ]

    for prompt in samples:
        compiled = exp.compile_prompt(prompt)
        assert compiled.compiled is True
        assert compiled.manual_review_required is False
        assert compiled.dsl["kind"] in exp.SAFE_VALIDATOR_KINDS
        assert exp.compiler_uses_arbitrary_code_execution() is False

        good = exp.evaluate_compiled_validator(compiled, prompt.known_good_output)
        bad = exp.evaluate_compiled_validator(compiled, prompt.known_bad_output)

        assert good["accepted"] is True, prompt.prompt_id
        assert bad["accepted"] is False, prompt.prompt_id


def test_req_verify_1494_unsupported_prompts_fail_closed_for_manual_review() -> None:
    """REQ-VERIFY-1494: unsupported prompt constraints do not compile speculatively."""

    prompts = exp.build_prompt_set()
    unsupported = [
        prompt
        for prompt in prompts
        if prompt.prompt_id in {"cctu-1494-schema-003", "cctu-1494-code-002"}
    ]

    assert len(unsupported) == 2
    for prompt in unsupported:
        compiled = exp.compile_prompt(prompt)
        assert compiled.compiled is False
        assert compiled.manual_review_required is True
        assert compiled.failure_reason
        assert (
            exp.evaluate_compiled_validator(compiled, prompt.known_good_output)["accepted"] is False
        )


def test_req_verify_1494_code_validator_never_executes_candidate_code(tmp_path: Path) -> None:
    """REQ-VERIFY-1494: simple-code validation is AST-only, not arbitrary execution."""

    prompt = next(
        prompt for prompt in exp.build_prompt_set() if prompt.prompt_id == "cctu-1494-code-001"
    )
    compiled = exp.compile_prompt(prompt)
    marker = tmp_path / "should_not_exist"
    malicious = (
        "def normalize_slug(text):\n"
        f"    __import__('pathlib').Path({str(marker)!r}).write_text('bad')\n"
        "    return text\n"
    )

    result = exp.evaluate_compiled_validator(compiled, malicious)

    assert result["accepted"] is False
    assert marker.exists() is False


def test_scenario_verify_1494_runner_writes_manifest_and_ready_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1494: runner reports compile, rejection, and review metrics."""

    def fake_collect(spec: dict[str, Any], prompts: list[exp.PromptCase]) -> dict[str, Any]:
        rows = [
            {
                "prompt_id": prompt.prompt_id,
                "model_hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "generation_source": "live_sota_llamacpp",
                "output_text": json.dumps(
                    {
                        "prompt_id": prompt.prompt_id,
                        "validator_kind": prompt.expected_validator_kind,
                        "fields": ["final_answer"],
                    },
                    sort_keys=True,
                ),
                "elapsed_seconds": 0.01,
                "blocker": None,
            }
            for prompt in prompts
        ]
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": True,
                "blocker": None,
            },
            "rows": rows,
        }

    output_path = tmp_path / "experiment_1494.json"
    manifest_path = tmp_path / "manifest_1494.jsonl"
    artifact = exp.run_experiment(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_model_skeletons_fn=fake_collect,
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["validator_compiler_ready"] is True
    assert artifact["prompts_attempted"] == 30
    assert artifact["validator_skeletons_generated"] == 30
    assert artifact["validators_compiled"] == 28
    assert artifact["validator_compile_rate"] == pytest.approx(28 / 30)
    assert artifact["known_good_pass_rate"] == pytest.approx(1.0)
    assert artifact["known_bad_reject_rate"] == pytest.approx(1.0)
    assert artifact["verifier_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["manual_review_required_count"] == 2
    assert artifact["validator_manifest_path"] == str(manifest_path)
    assert artifact["models_used"] == [exp.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert len(manifest_rows) == 30
    assert sum(row["manual_review_required"] for row in manifest_rows) == 2
    assert all("model_skeleton" in row for row in manifest_rows)


def test_req_verify_1494_runner_blocks_without_mandated_sota_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-1494: missing mandated live SOTA skeletons is a terminal blocker."""

    artifact = exp.run_experiment(
        output_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.jsonl",
        run_date="20260507",
        model_specs=[],
        collect_model_skeletons_fn=lambda _spec, _prompts: pytest.fail("no model"),
        gpu_probe_fn=lambda: {"nvidia_smi_available": False, "gpu_count": 0},
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["validator_compiler_ready"] is False
    assert artifact["validator_skeletons_generated"] == 0
    assert artifact["blockers"] == ["no_mandated_sota_gguf_model_available"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1494_live_collector_resolution_and_error_paths() -> None:
    """REQ-VERIFY-1494: live GGUF skeleton collection is injectable and honest."""

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            prompt_id = prompt.split("Prompt id: ", 1)[1].split("\n", 1)[0]
            return {
                "choices": [
                    {
                        "text": json.dumps(
                            {
                                "prompt_id": prompt_id,
                                "validator_kind": "json_final_answer",
                                "fields": ["final_answer"],
                            },
                            sort_keys=True,
                        )
                    }
                ]
            }

        def close(self) -> None:
            self.closed = True

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("generation failed")

    prompts = exp.build_prompt_set()[:2]
    spec = {"hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "gpu": 0}

    missing = exp.collect_live_model_skeletons(
        spec,
        prompts,
        resolver=lambda _hf_id: None,
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    import_failed = exp.collect_live_model_skeletons(
        {**spec, "model_path": "/tmp/fake.gguf"},
        prompts,
        llama_importer=lambda: (False, None, "llama missing"),
        env_preparer=lambda: {},
    )
    load_failed = exp.collect_live_model_skeletons(
        {**spec, "model_path": "/tmp/fake.gguf"},
        prompts,
        llama_importer=lambda: (True, LoadFails, None),
        env_preparer=lambda: {},
    )
    ok = exp.collect_live_model_skeletons(
        spec,
        prompts,
        resolver=lambda _hf_id: "/tmp/fake.gguf",
        llama_importer=lambda: (True, FakeLlama, None),
        env_preparer=lambda: {},
    )
    generation_failed = exp.collect_live_model_skeletons(
        {**spec, "model_path": "/tmp/fake.gguf"},
        prompts[:1],
        llama_importer=lambda: (True, GenerateFails, None),
        env_preparer=lambda: {},
    )

    assert missing["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama missing"
    assert "load failed" in load_failed["summary"]["blocker"]
    assert ok["summary"]["model_used"] is True
    assert len(ok["rows"]) == 2
    assert FakeLlama.prompts == [exp._skeleton_prompt(prompt) for prompt in prompts]
    assert generation_failed["summary"]["model_used"] is False
    assert generation_failed["summary"]["blocker"] == "no_usable_skeleton_generations"
    assert generation_failed["rows"][0]["blocker"] == "RuntimeError: generation failed"


def test_req_verify_1494_model_resolution_gpu_probe_and_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1494: model and GPU provenance helpers expose bounded outcomes."""

    pair = [{"name": "A", "hf_id": exp.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "a"}]
    assert exp.resolve_model_specs(cached_pair_fn=lambda **_kwargs: pair) == pair

    calls: list[str] = []

    def fake_resolver(hf_id: str) -> str | None:
        calls.append(hf_id)
        return "/tmp/qwen.gguf" if hf_id == exp.MANDATED_MODEL_SPECS[0]["hf_id"] else None

    singleton = exp.resolve_model_specs(
        cached_pair_fn=lambda **_kwargs: None,
        resolver_fn=fake_resolver,
    )
    assert singleton == [
        {
            **exp.MANDATED_MODEL_SPECS[0],
            "model_path": "/tmp/qwen.gguf",
        }
    ]
    assert calls == [spec["hf_id"] for spec in exp.MANDATED_MODEL_SPECS]

    class Success:
        returncode = 0
        stdout = "GPU A, 24576 MiB\n\nGPU B, 24576 MiB\n"
        stderr = ""

    monkeypatch.setattr(exp.subprocess, "run", lambda *_args, **_kwargs: Success())
    available = exp.probe_gpu()
    assert available["nvidia_smi_available"] is True
    assert available["gpu_count"] == 2
    assert available["gpus"][0]["memory_total"] == "24576 MiB"

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "not available"

    monkeypatch.setattr(exp.subprocess, "run", lambda *_args, **_kwargs: Failed())
    failed = exp.probe_gpu()
    assert failed["nvidia_smi_available"] is False
    assert failed["error"] == "not available"

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("no nvidia-smi")

    monkeypatch.setattr(exp.subprocess, "run", boom)
    blocked = exp.probe_gpu()
    assert blocked["nvidia_smi_available"] is False
    assert "no nvidia-smi" in blocked["error"]

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda: {
            "validator_compiler_ready": True,
            "validator_compile_rate": 0.5,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": "complete: test",
        },
    )
    assert exp.main([]) == 0
    printed = capsys.readouterr().out
    assert "ready=True" in printed
    assert "compile_rate=0.5" in printed


def test_req_verify_1494_defensive_validator_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1494: unsupported or malformed constraints fail closed."""

    unknown_prompt = exp.PromptCase(
        "manual-1",
        "manual",
        "test",
        "Case: manual-1\nNo supported constraints here.",
        "{}",
        "{}",
        "manual_review",
    )
    arithmetic_bad = exp.PromptCase(
        "arith-bad",
        "arithmetic",
        "test",
        "Case: arith-bad\nArithmetic expression: bad()",
        "{}",
        "{}",
        "json_final_answer",
    )
    arithmetic_missing = exp.PromptCase(
        "arith-missing",
        "arithmetic",
        "test",
        "Case: arith-missing\nArithmetic expression:\n",
        "{}",
        "{}",
        "json_final_answer",
    )
    cctu_bad = exp.PromptCase(
        "cctu-bad",
        "arithmetic",
        "test",
        "Use exactly one local tool named arithmetic.evaluate but omit the JSON block.",
        "{}",
        "{}",
        "cctu_tool_transcript",
    )
    graph_bad = exp.PromptCase(
        "graph-bad",
        "graph_path",
        "test",
        'Case: graph-bad\nGraph edges JSON:\n[["A", "B", 1]]\nStart: A\n',
        "{}",
        "{}",
        "graph_path_json",
    )
    code_bad = exp.PromptCase(
        "code-bad",
        "simple_code",
        "test",
        "Case: code-bad\nCode constraints JSON:\n"
        '{"function_name": 7, "parameters": "x", "returns_expr": "x"}\n',
        "def f(x):\n    return x\n",
        "def f(x):\n    return 0\n",
        "python_ast_function",
    )

    assert exp.compile_prompt(unknown_prompt).failure_reason == "unsupported_prompt_pattern"
    assert exp.compile_prompt(arithmetic_bad).failure_reason.startswith("ValueError")
    assert exp.compile_prompt(arithmetic_missing).failure_reason == "arithmetic_expression_missing"
    assert exp.compile_prompt(cctu_bad).failure_reason == "cctu_prompt_parse_failed"
    assert exp.compile_prompt(graph_bad).failure_reason == "graph_endpoint_missing"
    assert exp.compile_prompt(code_bad).failure_reason == "code_signature_constraint_malformed"

    unknown_compiled = exp.CompiledValidator(
        prompt_id="x",
        compiled=True,
        dsl={"kind": "unknown"},
        manual_review_required=False,
    )
    assert exp.evaluate_compiled_validator(unknown_compiled, "{}")["reason"] == (
        "unknown_validator_kind:unknown"
    )
    assert exp.parse_model_skeleton("not json") is None

    arithmetic = exp.compile_prompt(
        next(
            prompt for prompt in exp.build_prompt_set() if prompt.prompt_id == "cctu-1494-arith-001"
        )
    )
    assert exp.evaluate_compiled_validator(arithmetic, "not json")["reason"] == "no_json_object"

    schema = exp.compile_prompt(
        next(
            prompt
            for prompt in exp.build_prompt_set()
            if prompt.prompt_id == "cctu-1494-schema-001"
        )
    )
    assert exp.evaluate_compiled_validator(schema, "not json")["reason"] == "no_json_object"
    assert exp.evaluate_compiled_validator(schema, "{}")["reason"].startswith("missing_field:")
    assert (
        exp.evaluate_compiled_validator(
            schema,
            json.dumps({"ticket_id": 7, "priority": 2, "tags": ["security"]}),
        )["reason"]
        == "ticket_id:not_string"
    )
    assert (
        exp.evaluate_compiled_validator(
            schema,
            json.dumps({"ticket_id": "INC-42", "priority": "2", "tags": ["security"]}),
        )["reason"]
        == "priority:not_integer"
    )
    assert (
        exp.evaluate_compiled_validator(
            schema,
            json.dumps({"ticket_id": "INC-42", "priority": 0, "tags": ["security"]}),
        )["reason"]
        == "priority:below_minimum"
    )
    assert (
        exp.evaluate_compiled_validator(
            schema,
            json.dumps({"ticket_id": "INC-42", "priority": 2, "tags": "security"}),
        )["reason"]
        == "tags:not_array_string"
    )
    assert (
        exp.evaluate_compiled_validator(
            schema,
            json.dumps({"ticket_id": "INC-42", "priority": 2, "tags": ["api"]}),
        )["reason"]
        == "tags:missing_required_member"
    )

    original_extract = exp.cctu.extract_json_object
    monkeypatch.setattr(exp.cctu, "extract_json_object", lambda _text: [])
    assert exp.evaluate_compiled_validator(schema, "[]")["reason"] == "json_not_object"
    monkeypatch.setattr(exp.cctu, "extract_json_object", original_extract)

    code = exp.compile_prompt(
        next(
            prompt for prompt in exp.build_prompt_set() if prompt.prompt_id == "cctu-1494-code-001"
        )
    )
    assert exp.evaluate_compiled_validator(code, "def bad(:")["reason"].startswith("syntax_error")
    assert exp.evaluate_compiled_validator(code, "x = 1")["reason"] == (
        "expected_single_function_def"
    )
    assert (
        exp.evaluate_compiled_validator(code, "def wrong(text):\n    return text\n")["reason"]
        == "function_signature_mismatch"
    )
    assert (
        exp.evaluate_compiled_validator(
            code,
            "def normalize_slug(text):\n    x = text\n    return x\n",
        )["reason"]
        == "function_body_not_single_return"
    )

    graph = exp.compile_prompt(
        next(
            prompt for prompt in exp.build_prompt_set() if prompt.prompt_id == "cctu-1494-graph-001"
        )
    )
    assert exp.evaluate_compiled_validator(graph, "not json")["reason"] == "no_json_object"

    assert exp._schema_compile_failure({"required": {}}) == "schema_required_fields_missing"
    assert exp._schema_compile_failure({"required": {7: {}}}) == "schema_field_malformed"
    assert (
        exp._schema_compile_failure({"required": {"x": {"type": "number"}}})
        == "unsupported_schema_type:number"
    )
    assert exp._safe_eval_arithmetic("-3 + 8 - 2") == 3
    with pytest.raises(ValueError, match="unsupported_arithmetic_expression"):
        exp._safe_eval_arithmetic("abs(-3)")
    with pytest.raises(ValueError, match="json_block_missing_after"):
        exp._extract_json_block("Marker:\ntrue\n", "Marker:")


def test_req_verify_1494_runner_records_blocked_skeleton_generation(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1494: collector blockers flow into the terminal artifact."""

    artifact = exp.run_experiment(
        output_path=tmp_path / "blocked_rows.json",
        manifest_path=tmp_path / "blocked_rows.jsonl",
        run_date="20260507",
        model_specs=[{**exp.MANDATED_MODEL_SPECS[0], "model_path": "/tmp/fake.gguf"}],
        collect_model_skeletons_fn=lambda spec, _prompts: {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "no_usable_skeleton_generations",
            },
            "rows": [],
        },
        gpu_probe_fn=lambda: {"nvidia_smi_available": True, "gpu_count": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["blockers"] == [
        "no_usable_skeleton_generations",
        "live_sota_skeleton_generation_unavailable",
    ]
