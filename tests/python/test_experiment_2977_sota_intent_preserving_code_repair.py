"""Tests for Exp 2977 SOTA intent-preserving code-repair rerun.

Spec refs: REQ-VERIFY-2977, SCENARIO-VERIFY-2977.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_2977_sota_intent_preserving_code_repair as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
REQUIRED_FIELDS = {
    "honest_verdict",
    "repair_rerun_clean",
    "headline_result",
    "n_tasks",
    "models_used",
    "model_specs",
    "mandatory_headline_model_ids",
    "legacy_model_used_only_for_smoke",
    "baseline_pass_at_1",
    "schema_only_pass_at_1",
    "intent_preserving_pass_at_1",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "schema_failure_rate_delta",
    "syntax_failure_rate_delta",
    "false_accept_delta",
    "runtime_trace_coverage",
    "per_model_metrics",
    "failures_by_category",
    "inference_substrate",
    "duration_s",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sources(root: Path, *, n_tasks: int = 20, protocol_ready: bool = True) -> None:
    selected = []
    for index in range(n_tasks):
        selected.append(
            {
                "task_id": f"MBPP:mbpp-{index}",
                "stable_id": f"mbpp-{index}",
                "corpus": "MBPP",
                "sample_id": f"MBPP:mbpp-{index}:c0:s{2910 + index}",
                "candidate_index": 0,
                "candidate_seed": 2910 + index,
                "original_failure_categories": ["syntax_error"],
            }
        )
    _write_json(
        root / "results" / exp.PROTOCOL_FILENAME,
        {
            "artifact": exp.PROTOCOL_FILENAME,
            "intent_preserving_repair_protocol_ready": protocol_ready,
            "trace_execution_plan_ready": protocol_ready,
            "mandatory_headline_model_ids": list(exp.MANDATORY_HEADLINE_MODEL_IDS),
        },
    )
    _write_json(
        root / "results" / exp.UPSTREAM_REPAIR_FILENAME,
        {
            "artifact": "experiment_2964_sota_dccd_repair_replication_v1",
            "selected_repair_set": selected,
            "candidate_evaluations": [],
        },
    )
    _write_json(
        root / "results" / exp.THRESHOLD_FILENAME,
        {
            "artifact": "experiment_2953_code_verifier_threshold_policy_v1",
            "selected_default_threshold": 1.0,
            "expected_false_accept_rate_at_default": 0.010135135135,
        },
    )


def _config(tmp_path: Path, *, n_tasks: int = 20, smoke_tasks: int = 2) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        raw_response_dir=tmp_path / "results" / "raw" / "experiment_2977",
        n_tasks=n_tasks,
        smoke_tasks=smoke_tasks,
        samples_per_mode=1,
        started_at=10.0,
        clock=lambda: 72.5,
        tests_run=("focused-exp2977",),
    )


def _cached_pair() -> list[dict[str, Any]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma.gguf",
        },
    ]


def _clean_generator(
    task: exp.RepairTask,
    mode: str,
    sample_index: int,
    seed: int,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    if mode == exp.INTENT_PRESERVING_MODE:
        return {
            "raw_candidate": f"def repair_{task.stable_id.replace('-', '_')}(x):\n    return x + 1\n",
            "schema_valid": True,
            "syntax_success": True,
            "passed": True,
            "verifier_accepted": True,
            "verifier_score": 1.0,
            "execution_trace": [
                {
                    "command": "pytest",
                    "exit_code": 0,
                    "stdout": "passed",
                    "stderr": "",
                    "failing_assertions": [],
                }
            ],
            "tokens_generated": 48,
            "generation_backend": "fake-live-llama",
            "generation_backend_detail": model_spec["model_path"],
        }
    if mode == exp.SCHEMA_ONLY_MODE:
        return {
            "raw_candidate": '{"repaired_code": "unterminated"',
            "schema_valid": False,
            "syntax_success": False,
            "passed": False,
            "verifier_accepted": False,
            "schema_errors": ["invalid JSON object"],
            "tokens_generated": 8,
            "generation_backend": "fake-live-llama",
            "generation_backend_detail": model_spec["model_path"],
        }
    return {
        "raw_candidate": "def broken(:\n",
        "schema_valid": True,
        "syntax_success": False,
        "passed": False,
        "verifier_accepted": False,
        "syntax_errors": ["invalid syntax"],
        "tokens_generated": 8,
        "generation_backend": "fake-live-llama",
        "generation_backend_detail": model_spec["model_path"],
    }


def test_req_verify_2977_spec_anchor_exists() -> None:
    """REQ-VERIFY-2977: the rerun harness is anchored in OpenSpec."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-2977" in spec
    assert "SCENARIO-VERIFY-2977" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference"' in spec


def test_scenario_verify_2977_cpu_smoke_when_cached_pair_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2977: no cached SOTA pair produces a non-headline smoke result."""
    _write_sources(tmp_path, n_tasks=20)

    artifact = exp.write_artifact(
        _config(tmp_path, smoke_tasks=2),
        cached_pair_func=lambda: None,
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == "blocked_cached_sota_pair_unavailable_cpu_smoke_only"
    assert artifact["headline_result"] is False
    assert artifact["repair_rerun_clean"] is False
    assert artifact["legacy_model_used_only_for_smoke"] is True
    assert artifact["n_tasks"] == 2
    assert artifact["models_used"] == ["Qwen/Qwen3.5-0.8B"]
    assert artifact["mandatory_headline_model_ids"] == list(exp.MANDATORY_HEADLINE_MODEL_IDS)
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["runtime_trace_coverage"] == pytest.approx(1.0)
    assert len(artifact["candidate_evaluations"]) == 2 * len(exp.CONDITIONS)
    assert all(
        Path(tmp_path, row["raw_candidate_ref"]).is_file()
        for row in artifact["candidate_evaluations"]
    )


def test_scenario_verify_2977_clean_headline_requires_all_gates(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2977: clean promotion requires pass lift and no regressions."""
    _write_sources(tmp_path, n_tasks=20)

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        cached_pair_func=_cached_pair,
        generator=_clean_generator,
    )

    assert (
        artifact["honest_verdict"] == "complete: intent-preserving trace-aware repair rerun clean"
    )
    assert artifact["headline_result"] is True
    assert artifact["repair_rerun_clean"] is True
    assert artifact["legacy_model_used_only_for_smoke"] is False
    assert artifact["n_tasks"] == 20
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["model_specs"] == _cached_pair()
    assert artifact["baseline_pass_at_1"] == pytest.approx(0.0)
    assert artifact["schema_only_pass_at_1"] == pytest.approx(0.0)
    assert artifact["intent_preserving_pass_at_1"] == pytest.approx(1.0)
    assert artifact["pass_at_1_delta"] == pytest.approx(1.0)
    assert artifact["pass_at_k_delta"] == pytest.approx(1.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(-1.0)
    assert artifact["false_accept_delta"] == pytest.approx(0.0)
    assert artifact["runtime_trace_coverage"] == pytest.approx(1.0)
    assert artifact["failures_by_category"]["syntax_error"] == 20
    assert artifact["per_model_metrics"]["unsloth/Qwen3.6-35B-A3B-GGUF"][
        exp.INTENT_PRESERVING_MODE
    ]["pass_at_1"] == pytest.approx(1.0)


def test_req_verify_2977_cached_headline_uses_live_generator_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2977: cached headline models use the live GGUF generator, not smoke output."""
    _write_sources(tmp_path, n_tasks=20)
    calls: list[dict[str, Any]] = []

    def live_factory(*, model_path: str, main_gpu: int) -> exp.RepairGenerator:
        calls.append({"model_path": model_path, "main_gpu": main_gpu})
        return _clean_generator

    monkeypatch.setattr(exp, "llama_cpp_trace_aware_repair_generator", live_factory, raising=False)

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        cached_pair_func=_cached_pair,
    )

    assert calls == [{"model_path": "/models/qwen.gguf", "main_gpu": 0}]
    assert artifact["headline_result"] is True
    assert artifact["legacy_model_used_only_for_smoke"] is False
    assert (
        artifact["honest_verdict"] == "complete: intent-preserving trace-aware repair rerun clean"
    )
    assert {row["generation_backend"] for row in artifact["candidate_evaluations"]} == {
        "fake-live-llama"
    }


def test_req_verify_2977_llama_cpp_generator_records_trace_without_auto_accepting() -> None:
    """REQ-VERIFY-2977: live GGUF output keeps raw text, diagnostics, and trace evidence."""
    prompts: list[str] = []

    class FakeLlama:
        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            prompts.append(prompt)
            assert kwargs["seed"] == 123
            assert kwargs["max_tokens"] == 256
            return {
                "choices": [
                    {
                        "text": (
                            '{"repaired_code": "def fixed(x):\\n    return x + 1\\n"}'
                            if "schema-only" in prompt
                            else "```python\ndef fixed(x):\n    return x + 1\n```"
                        )
                    }
                ],
                "usage": {"completion_tokens": 17},
            }

    created: list[dict[str, Any]] = []

    def fake_factory(**kwargs: Any) -> FakeLlama:
        created.append(kwargs)
        return FakeLlama()

    generator = exp.llama_cpp_trace_aware_repair_generator(
        model_path="/models/qwen.gguf",
        main_gpu=1,
        llama_factory=fake_factory,
    )
    task = exp.RepairTask(
        task_id="MBPP:mbpp-1",
        stable_id="mbpp-1",
        corpus="MBPP",
        sample_id="MBPP:mbpp-1:c0:s2911",
        original_failure_categories=("syntax_error",),
    )

    schema_payload = generator(
        task,
        exp.SCHEMA_ONLY_MODE,
        0,
        123,
        {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_path": "/models/qwen.gguf"},
    )
    intent_payload = generator(
        task,
        exp.INTENT_PRESERVING_MODE,
        0,
        123,
        {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_path": "/models/qwen.gguf"},
    )

    assert created == [
        {
            "model_path": "/models/qwen.gguf",
            "n_ctx": 4096,
            "n_batch": 128,
            "n_ubatch": 128,
            "n_gpu_layers": -1,
            "main_gpu": 1,
            "verbose": False,
        }
    ]
    assert len(prompts) == 2
    assert "schema-only DCCD" in prompts[0]
    assert "intent-preserving trace-aware repair" in prompts[1]
    assert schema_payload["schema_valid"] is True
    assert schema_payload["syntax_success"] is True
    assert schema_payload["passed"] is False
    assert schema_payload["verifier_accepted"] is False
    assert schema_payload["tokens_generated"] == 17
    assert schema_payload["generation_backend"] == "llama_cpp"
    assert intent_payload["execution_trace"][0]["stage"] == "ast_parse"
    assert intent_payload["execution_trace"][0]["exit_code"] == 0
    assert intent_payload["passed"] is False
    assert intent_payload["verifier_accepted"] is False


def test_req_verify_2977_live_generator_defensive_diagnostics() -> None:
    """REQ-VERIFY-2977: malformed live outputs become diagnostics, not accepted repairs."""

    class FailingLlama:
        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    generator = exp.llama_cpp_trace_aware_repair_generator(
        model_path="/models/qwen.gguf",
        llama_factory=lambda **_kwargs: FailingLlama(),
    )
    task = exp.RepairTask(
        task_id="MBPP:bad",
        stable_id="bad",
        corpus="MBPP",
        sample_id="MBPP:bad:c0:s1",
        original_failure_categories=(),
    )

    payload = generator(task, exp.BASELINE_MODE, 0, 1, {"model_path": "/models/qwen.gguf"})

    assert payload["generation_error"] == "RuntimeError: boom"
    assert payload["schema_valid"] is False
    assert payload["syntax_success"] is False
    assert payload["execution_trace"] == []
    assert exp._parse_schema_repair("not-json")[1] == ["invalid JSON: Expecting value"]
    assert exp._parse_schema_repair("[]")[1] == ["schema repair output is not a JSON object"]
    assert exp._parse_schema_repair('{"repaired_code": ""}')[1] == [
        'schema repair output missing non-empty "repaired_code"'
    ]
    assert exp._extract_python_code("plain text") == "plain text"
    assert exp._syntax_diagnostics("") == (False, ["empty candidate"])
    assert exp._syntax_diagnostics("def bad(:") == (False, ["SyntaxError: invalid syntax"])
    assert exp._llama_text({"choices": ["not-a-mapping"]}).startswith("{'choices'")
    assert exp._llama_completion_tokens({"usage": "not-a-mapping"}) == 0


def test_req_verify_2977_blocks_when_protocol_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-2977: Exp 2976 readiness is a hard precondition."""
    _write_sources(tmp_path, n_tasks=20, protocol_ready=False)

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        cached_pair_func=_cached_pair,
        generator=_clean_generator,
    )

    assert artifact["honest_verdict"] == "blocked_exp2976_protocol_not_ready"
    assert artifact["headline_result"] is False
    assert artifact["repair_rerun_clean"] is False
    assert artifact["n_tasks"] == 0
    assert artifact["candidate_evaluations"] == []


def test_req_verify_2977_blocks_when_required_sources_are_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-2977: missing upstream repair inputs block evaluation."""
    _write_sources(tmp_path, n_tasks=20)
    (tmp_path / "results" / exp.THRESHOLD_FILENAME).unlink()

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        cached_pair_func=_cached_pair,
        generator=_clean_generator,
    )

    assert artifact["honest_verdict"] == "blocked_missing_upstream_repair_artifacts"
    assert artifact["headline_result"] is False
    assert artifact["repair_rerun_clean"] is False
    assert artifact["n_tasks"] == 0
    assert artifact["candidate_evaluations"] == []


def test_req_verify_2977_blocks_when_no_repair_tasks_remain(tmp_path: Path) -> None:
    """REQ-VERIFY-2977: an empty selected repair set blocks the rerun."""
    _write_sources(tmp_path, n_tasks=1)
    upstream_path = tmp_path / "results" / exp.UPSTREAM_REPAIR_FILENAME
    upstream_payload = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream_payload["selected_repair_set"] = ["not-a-mapping"]
    _write_json(upstream_path, upstream_payload)

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        cached_pair_func=_cached_pair,
        generator=_clean_generator,
    )

    assert artifact["honest_verdict"] == "blocked_no_repair_tasks_available"
    assert artifact["headline_result"] is False
    assert artifact["repair_rerun_clean"] is False
    assert artifact["n_tasks"] == 0


def test_scenario_verify_2977_normalizes_task_rows_and_external_raw_refs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2977: malformed rows are skipped and raw refs stay traceable."""
    _write_sources(tmp_path, n_tasks=20)
    upstream_path = tmp_path / "results" / exp.UPSTREAM_REPAIR_FILENAME
    upstream_payload = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream_payload["selected_repair_set"] = [
        "not-a-mapping",
        {
            "stable_id": "odd/id",
            "corpus": "MBPP",
            "original_failure_categories": [7],
        },
    ]
    _write_json(upstream_path, upstream_payload)
    outside_raw_dir = tmp_path.parent / "external-exp2977-raw"

    artifact = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
            raw_response_dir=outside_raw_dir,
            n_tasks=2,
            samples_per_mode=1,
            started_at=10.0,
            clock=lambda: 72.5,
        ),
        cached_pair_func=_cached_pair,
        generator=_clean_generator,
    )

    assert artifact["n_tasks"] == 1
    first_row = artifact["candidate_evaluations"][0]
    assert first_row["task_id"] == "MBPP:odd/id"
    assert first_row["stable_id"] == "odd/id"
    assert first_row["sample_id"] == "MBPP:odd/id"
    assert first_row["original_failure_categories"] == ["7"]
    assert first_row["raw_candidate_ref"].startswith(str(outside_raw_dir))
    assert "odd_id" in first_row["raw_candidate_ref"]


def test_req_verify_2977_nonclean_verdict_and_main_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-2977: non-clean gates remain explicit and main returns status."""
    assert (
        exp._complete_verdict(False)
        == "complete: intent-preserving trace-aware repair rerun did not clear gates"
    )

    monkeypatch.setattr(
        exp,
        "write_artifact",
        lambda _config: {"honest_verdict": "blocked_missing_upstream_repair_artifacts"},
    )
    assert exp.main() == 1

    monkeypatch.setattr(exp, "write_artifact", lambda _config: {"honest_verdict": "complete"})
    assert exp.main() == 0
