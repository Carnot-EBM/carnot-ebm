"""Tests for Exp 2952 taxonomy-guided live code repair evaluation.

Spec: REQ-CODE-2952, SCENARIO-CODE-2952.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import sota_taxonomy_guided_code_repair_eval as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(
    *,
    stable_id: str,
    corpus: str,
    candidate_index: int,
    seed: int,
    extracted_code: str,
    error_type: str,
    error_message: str,
    syntax_success: bool,
) -> dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "corpus": corpus,
        "error_message": error_message,
        "error_type": error_type,
        "executed": syntax_success,
        "extracted_code": extracted_code,
        "extraction_success": True,
        "raw_response": extracted_code,
        "raw_response_path": f"results/raw/{stable_id}_{seed}.txt",
        "raw_response_sha256": "a" * 64,
        "random_seed": seed,
        "row_status": "candidate_syntax_failed" if not syntax_success else "candidate_failed",
        "runtime_success": False,
        "stable_id": stable_id,
        "syntax_success": syntax_success,
        "passed": False,
    }


def _write_ready_sources(tmp_path: Path) -> None:
    _write_json(
        tmp_path / exp.EXP2940_REL_PATH,
        {
            "artifact": "experiment_2940_verifier_ensemble_auprc_code_corpora_v1",
            "max_f1_operating_point": {"threshold": 1.0},
        },
    )
    _write_json(
        tmp_path / exp.EXP2946_REL_PATH,
        {
            "artifact": "experiment_2946_sota_code_generation_continuation_v1",
            "honest_verdict": "complete: retain continuation executed",
            "inference_substrate": "live_llm_inference",
            "pass_at_1": 0.06,
            "pass_at_k": 0.16,
            "protocol_artifact_path": str(exp.NESTED_EXP2946_REL_PATH),
        },
    )
    _write_json(
        tmp_path / exp.NESTED_EXP2946_REL_PATH,
        {
            "artifact": "experiment_2946_nested_exp2910_protocol",
            "candidate_results": [
                _candidate(
                    stable_id="mbpp-alpha",
                    corpus="MBPP",
                    candidate_index=0,
                    seed=2910,
                    extracted_code="def solve_alpha(:\n",
                    error_type="SyntaxError",
                    error_message="invalid syntax",
                    syntax_success=False,
                ),
                _candidate(
                    stable_id="mbpp-beta",
                    corpus="MBPP",
                    candidate_index=0,
                    seed=2918,
                    extracted_code="def solve_beta(x):\n    return x\n",
                    error_type="AssertionError",
                    error_message="AssertionError: wrong answer",
                    syntax_success=True,
                ),
            ],
            "per_task_results": [
                {"stable_id": "mbpp-alpha", "pass_at_k": 0.0},
                {"stable_id": "mbpp-beta", "pass_at_k": 0.0},
            ],
        },
    )
    _write_json(
        tmp_path / exp.EXP2950_REL_PATH,
        {
            "artifact": "experiment_2950_code_taxonomy_repair_prompt_manifest_v1",
            "repair_prompt_manifest_ready": True,
            "repair_prompt_templates": {
                "syntax_error": {
                    "template_id": "syntax_error_repair_v1",
                    "template": (
                        "Taxonomy label: syntax_error\n"
                        "Sample: {sample_id}\n"
                        "Failure evidence: {failure_evidence}\n"
                        "Task context: {task_prompt}\n"
                        "Candidate code:\n{candidate_code}\n"
                        "Repair focus: fix malformed Python.\n"
                        "Return only corrected Python code."
                    ),
                },
                "failed_tests": {
                    "template_id": "failed_tests_repair_v1",
                    "template": (
                        "Taxonomy label: failed_tests\n"
                        "Sample: {sample_id}\n"
                        "Failure evidence: {failure_evidence}\n"
                        "Task context: {task_prompt}\n"
                        "Candidate code:\n{candidate_code}\n"
                        "Repair focus: satisfy assertions.\n"
                        "Return only corrected Python code."
                    ),
                },
            },
        },
    )
    _write_json(
        tmp_path / exp.EXP2951_REL_PATH,
        {
            "artifact": "experiment_2951_structured_candidate_manifest_adapter_v1",
            "structured_decode_manifest_ready": True,
            "schema_version": "carnot.structured_candidate_manifest.v1",
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        raw_response_dir=tmp_path / "results" / "raw" / "experiment_2952",
        n_tasks=2,
        samples_per_mode=2,
        started_at=10.0,
        clock=lambda: 75.5,
        tests_run=("focused-pytest",),
    )


def _preconditions() -> exp.PreconditionReport:
    return exp.PreconditionReport(
        checks=[
            {"resource": "dual_rtx_3090_host", "available": True, "detail": "2x RTX 3090"},
            {"resource": "llama_cpp_runtime", "available": True, "detail": "imported"},
            {"resource": "runsc_sandbox", "available": True, "detail": "runsc"},
        ],
        model_specs=[
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": "/models/gemma.gguf",
                "cached": True,
                "selected_for_live_repair": True,
            },
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": None,
                "cached": False,
                "selected_for_live_repair": False,
            },
        ],
        runnable_model_specs=[
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": "/models/gemma.gguf",
            }
        ],
    )


def _task_rows(_config: exp.ExperimentConfig) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        ("MBPP", "mbpp-alpha"): {
            "stable_id": "mbpp-alpha",
            "prompt": "Return x + 1.",
            "tests": ["assert solve_alpha(1) == 2"],
            "test_imports": [],
        },
        ("MBPP", "mbpp-beta"): {
            "stable_id": "mbpp-beta",
            "prompt": "Return x * 2.",
            "tests": ["assert solve_beta(3) == 6"],
            "test_imports": [],
        },
    }


def _generator(prompt: str, seed: int, _max_tokens: int, _model_spec: dict[str, Any]) -> exp.GenerationOutcome:
    assert seed in {2952, 2953, 2954, 2955, 3052, 3053, 3054, 3055}
    mode = "taxonomy" if "Taxonomy label:" in prompt else "baseline"
    is_alpha = "solve_alpha" in prompt
    if mode == "baseline" and is_alpha and seed == 2952:
        text = "```python\ndef solve_alpha(:\n```"
    elif mode == "taxonomy" and is_alpha:
        text = "```python\ndef solve_alpha(x):\n    return x + 1\n```"
    elif mode == "taxonomy":
        text = "```python\ndef solve_beta(x):\n    return x\n```"
    else:
        text = "```python\ndef solve_beta(x):\n    return x\n```"
    return exp.GenerationOutcome(
        text=text,
        tokens_generated=32,
        duration_s=1.25,
        backend="fake-live-llama",
        backend_detail="fake",
    )


def _executor(script: str, _timeout: float) -> exp.ExecutionOutcome:
    if "def solve_alpha(x):\n    return x + 1" in script:
        return exp.ExecutionOutcome(passed=True)
    if "def solve_beta(x):\n    return x" in script:
        return exp.ExecutionOutcome(
            passed=False,
            error_type="AssertionError",
            error_message="wrong answer",
        )
    return exp.ExecutionOutcome(
        passed=False,
        error_type="SyntaxError",
        error_message="invalid syntax",
    )


def test_req_code_2952_spec_anchor_exists() -> None:
    """REQ-CODE-2952, SCENARIO-CODE-2952: Exp 2952 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2952" in spec
    assert "SCENARIO-CODE-2952" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference"' in spec


def test_scenario_code_2952_equal_budget_taxonomy_repair_computes_guarded_delta(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2952: taxonomy repair wins only when false accepts do not grow."""

    _write_ready_sources(tmp_path)

    artifact = exp.write_artifact(
        _config(tmp_path),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=_task_rows,
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["n_tasks"] == 2
    assert artifact["sample_budget_per_mode"] == 4
    assert artifact["selected_task_ids"] == ["MBPP:mbpp-alpha", "MBPP:mbpp-beta"]
    assert artifact["baseline_pass_at_1"] == pytest.approx(0.0)
    assert artifact["repair_pass_at_1"] == pytest.approx(0.5)
    assert artifact["pass_at_1_delta"] == pytest.approx(0.5)
    assert artifact["baseline_pass_at_k"] == pytest.approx(0.0)
    assert artifact["repair_pass_at_k"] == pytest.approx(0.5)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(-0.25)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["false_accept_delta"] == pytest.approx(-0.25)
    assert artifact["taxonomy_repair_delta_pass"] is True
    assert artifact["baseline_verifier_acceptance_rate"] == pytest.approx(0.75)
    assert artifact["repair_verifier_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["candidate_manifest_sha256"] == exp._sha256_payload(
        artifact["candidate_manifests"]
    )
    assert len(artifact["candidate_manifest_sha256"]) == 64
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["headline_models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["legacy_models_only_for_smoke"] is False
    assert all(row["schema_valid"] for row in artifact["candidate_evaluations"])
    assert {
        tuple(row["original_failure_categories"])
        for row in artifact["candidate_evaluations"]
        if row["mode"] == "taxonomy_guided"
    } == {("syntax_error",), ("failed_tests",)}
    assert "false accepts decreased" in artifact["false_accept_audit_notes"][0]


def test_req_code_2952_blocks_when_upstream_artifact_is_missing(tmp_path: Path) -> None:
    """REQ-CODE-2952: missing gated artifacts prevent live repair."""

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2951_REL_PATH).unlink()

    artifact = exp.build_artifact(
        _config(tmp_path),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=_task_rows,
    )

    assert artifact["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert artifact["headline_models_used"] == []
    assert artifact["n_tasks"] == 0
    assert artifact["baseline_pass_at_1"] is None
    assert artifact["repair_pass_at_1"] is None
    assert artifact["taxonomy_repair_delta_pass"] is False
    assert any(
        row["resource"] == "exp2951_structured_candidate_manifest_adapter"
        and row["available"] is False
        for row in artifact["preconditions_checked"]
    )


def test_req_code_2952_blocks_when_no_headline_model_is_runnable(tmp_path: Path) -> None:
    """REQ-CODE-2952: headline repair cannot fall back to legacy small models."""

    _write_ready_sources(tmp_path)
    blocked = exp.PreconditionReport(
        checks=[{"resource": "headline_gguf_cache", "available": False, "detail": "none"}],
        model_specs=[],
        runnable_model_specs=[],
    )

    artifact = exp.build_artifact(
        _config(tmp_path),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: blocked,
        task_row_provider=_task_rows,
    )

    assert artifact["honest_verdict"] == "blocked_sota_gguf_precondition"
    assert artifact["legacy_models_only_for_smoke"] is False
    assert artifact["candidate_evaluations"] == []


def test_req_code_2952_covers_static_schema_and_selection_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2952: deterministic validation covers static and blocked branches."""

    _write_ready_sources(tmp_path)
    no_rows = exp.build_artifact(
        _config(tmp_path),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=lambda _config: {},
    )
    assert no_rows["honest_verdict"] == "blocked_no_failed_exp2946_candidates"

    nested = {
        "per_task_results": [
            {"stable_id": "skip-passed", "pass_at_k": 0.0},
            {"stable_id": "skip-missing-row", "pass_at_k": 0.0},
            {"stable_id": "keep", "pass_at_k": 0.0},
        ],
        "candidate_results": [
            {"stable_id": "", "corpus": "MBPP", "passed": False},
            {"stable_id": "not-failed", "corpus": "MBPP", "passed": False},
            {"stable_id": "skip-passed", "corpus": "MBPP", "passed": True},
            {"stable_id": "skip-missing-row", "corpus": "MBPP", "passed": False},
            _candidate(
                stable_id="keep",
                corpus="MBPP",
                candidate_index=0,
                seed=2999,
                extracted_code="def keep(x):\n    return x\n",
                error_type="AssertionError",
                error_message="wrong",
                syntax_success=True,
            ),
        ],
    }
    selected = exp.select_repair_set(
        nested,
        {("MBPP", "keep"): {"stable_id": "keep", "prompt": "Return x.", "tests": ["assert keep(1)==1"]}},
        n_tasks=1,
    )
    assert [row.stable_id for row in selected] == ["keep"]
    assert "Taxonomy label: failed_tests" in exp._repair_prompt(selected[0], exp.REPAIR_MODE, {})

    unsafe = exp._static_checks("import os\ndef keep(x):\n    return x\n")
    assert unsafe["status"] == "failed"
    assert exp.parser_or_static_error(unsafe) == "StaticCheckFailed"
    assert exp._post_repair_taxonomy(True, unsafe, False) == ["unsupported_import"]

    unsupported = exp._static_checks("import json\ndef keep(x):\n    return json.parse('{}')\n")
    assert unsupported["unsupported_api_calls"] == ["json.parse"]
    assert exp._post_repair_taxonomy(True, unsupported, False) == [
        "unsupported_api_hallucination"
    ]
    assert exp._static_checks("def broken(:\n")["status"] == "syntax_error"
    assert exp._attribute_root(__import__("ast").parse("foo.bar.baz()").body[0].value.func) == "foo"
    assert exp._attribute_root(__import__("ast").parse("42").body[0].value) == ""

    monkeypatch.setattr(exp.exp2950, "_candidate_labels", lambda _row: ())
    assert exp._original_failure_categories({"syntax_success": False}) == ("syntax_error",)
    assert exp._original_failure_categories({"syntax_success": True}) == ("failed_tests",)

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert exp._read_jsonl(jsonl) == [{"a": 1}, {"b": 2}]
    assert exp._display_corpus("humaneval") == "HumanEval"
    assert exp._display_corpus("custom") == "custom"
    assert exp._false_accept_notes({}, {}, {"false_accept_delta": 0}) == [
        "false accepts unchanged under taxonomy-guided repair"
    ]
    assert "increased" in exp._false_accept_notes({}, {}, {"false_accept_delta": 0.5})[0]
    assert exp._delta("new", 1.0) is None
    assert exp._negative(-0.1) is True
