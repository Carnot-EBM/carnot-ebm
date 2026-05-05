"""Tests for Exp 1366 tag-first prefix-injection CRANE certificates.

Spec: REQ-VERIFY-1366, SCENARIO-VERIFY-1366
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import certificate_v8_tag_first_prefix_injection_crane as mod


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _exp1352() -> dict[str, Any]:
    return {
        "status": "complete",
        "sota_run_allowed": True,
        "runtime_settings_used": {"max_tokens": 96, "temperature": 0.0, "top_p": 1.0},
        "min_completion_tokens_by_state": {"SAT": 6, "UNSAT": 6, "UNKNOWN": 6, "REPAIR_HINT": 10},
        "max_token_budget_sufficient": True,
        "dynamic_dispatch_preserved": True,
        "structural_tag_supported": True,
        "honest_verdict": "preflight_allows_exp1353_pure_python_fallback_xgrammar_absent",
    }


def _exp1353() -> dict[str, Any]:
    return {
        "status": "complete",
        "certificate_parse_rate": 0.0,
        "trigger_token_hit_rate": 0.0,
        "certificate_truthfulness_rate": 0.0,
        "unknown_preservation_rate": 0.0,
        "honest_verdict": "sota_triggered_certificate_v7_measured",
    }


def _exp1364() -> dict[str, Any]:
    return {
        "status": "complete",
        "thinking_mode_blocker_confirmed": True,
        "prior_certificate_parse_rate": 0.0,
        "honest_verdict": "milestone_105_carryforward_confirms_thinking_mode_missing_structural_tag_blocker",
    }


def _sources() -> dict[str, dict[str, Any]]:
    return {"exp1352": _exp1352(), "exp1353": _exp1353(), "exp1364": _exp1364()}


def _perfect_generation(
    spec: dict[str, Any],
    case: mod.CertificateCase,
    prompts: mod.CranePrompts,
) -> mod.CraneGenerationResult:
    del prompts
    return mod.CraneGenerationResult(
        model_hf_id=spec["hf_id"],
        case_id=case.case_id,
        reasoning_text="The branch follows directly from the bounded fixture.",
        reasoning_token_count=9,
        certificate_prefix=mod.structural_tag(case.expected_state) + "\n",
        certificate_body=mod.json_certificate_text(case.expected_state),
        generation_source="live_sota_llamacpp",
        certificate_token_count=18,
        elapsed_reasoning_seconds=0.01,
        elapsed_certificate_seconds=0.02,
    )


def test_req1366_prefix_injected_headline_rows_compute_metrics() -> None:
    """REQ-VERIFY-1366-3/4/5/6/7/8: SOTA prefix rows clear headline metrics."""

    artifact = mod.build_experiment_artifact(
        source_artifacts=_sources(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=_perfect_generation,
        run_date="20260505",
        project_root="/repo",
        max_models=1,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["prefix_injection_supported"] is True
    assert artifact["prefix_injection_method"] == mod.PREFIX_INJECTION_METHOD
    assert artifact["certificate_case_count"] == 4
    assert artifact["trigger_token_hit_rate"] == pytest.approx(1.0)
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["certificate_truthfulness_rate"] == pytest.approx(1.0)
    assert artifact["unknown_preservation_rate"] == pytest.approx(1.0)
    assert artifact["parse_rate_delta_over_exp1353"] == pytest.approx(1.0)
    assert artifact["crane_reasoning_budget_tokens_used"] == [9, 9, 9, 9]
    assert artifact["terminal_blocker"] is None
    assert artifact["retire_trigger_before_constrain"] is False
    assert artifact["headline_result_allowed"] is True
    assert artifact["models_used"][0]["hf_id"] == QWEN_SPEC["hf_id"]
    assert artifact["models_used"][0]["quantization"] == "UD-Q4_K_M"
    assert artifact["honest_verdict"] == "tag_first_prefix_injection_crane_positive_parse_rate_1_0"


def test_req1366_retire_when_prefix_path_unavailable_or_parse_zero() -> None:
    """REQ-VERIFY-1366-8: unavailable prefix injection or repeated parse zero retires."""

    unavailable = mod.build_experiment_artifact(
        source_artifacts=_sources(),
        model_specs=None,
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=lambda *_args, **_kwargs: pytest.fail("generation must not run"),
        run_date="20260505",
        project_root="/repo",
    )

    assert unavailable["terminal_blocker"] == "cached_sota_pair_unavailable"
    assert unavailable["prefix_injection_supported"] is False
    assert unavailable["retire_trigger_before_constrain"] is True
    assert unavailable["retire_if_same_verdict"] is True
    assert unavailable["headline_result_allowed"] is False

    def malformed(
        spec: dict[str, Any],
        case: mod.CertificateCase,
        prompts: mod.CranePrompts,
    ) -> mod.CraneGenerationResult:
        del prompts
        return mod.CraneGenerationResult(
            model_hf_id=spec["hf_id"],
            case_id=case.case_id,
            reasoning_text="<think>the model reasoned first</think>",
            reasoning_token_count=7,
            certificate_prefix=mod.structural_tag(case.expected_state) + "\n",
            certificate_body="<think>still not a certificate</think>",
            generation_source="live_sota_llamacpp",
            certificate_token_count=7,
        )

    parse_zero = mod.build_experiment_artifact(
        source_artifacts=_sources(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=malformed,
        run_date="20260505",
        project_root="/repo",
        max_models=1,
    )

    assert parse_zero["prefix_injection_supported"] is True
    assert parse_zero["trigger_token_hit_rate"] == pytest.approx(1.0)
    assert parse_zero["certificate_parse_rate"] == pytest.approx(0.0)
    assert parse_zero["terminal_blocker"] == "prefix_injection_parse_rate_zero"
    assert parse_zero["retire_trigger_before_constrain"] is True
    assert parse_zero["headline_result_allowed"] is False


def test_req1366_live_generator_uses_prefix_and_certificate_grammar() -> None:
    """REQ-VERIFY-1366-4/5: live adapter separates reasoning from constrained body."""

    class FakeGrammar:
        grammars: list[str] = []

        @classmethod
        def from_string(cls, grammar: str, verbose: bool = False) -> "FakeGrammar":
            del verbose
            cls.grammars.append(grammar)
            return cls()

    class FakeLlama:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"prompt": prompt, "kwargs": kwargs, "init": self.kwargs})
            if kwargs.get("grammar") is None:
                return {
                    "choices": [{"text": "Brief unconstrained reasoning."}],
                    "usage": {"completion_tokens": 4},
                }
            return {
                "choices": [{"text": mod.json_certificate_text("SAT")}],
                "usage": {"completion_tokens": 18},
            }

    generator = mod.LlamaCppCraneGenerator(
        {"max_tokens": 96, "temperature": 0.0, "top_p": 1.0, "n_ctx": 128},
        llama_importer=lambda: FakeLlama,
        grammar_importer=lambda: FakeGrammar,
    )
    case = mod.bounded_certificate_suite()[0]
    prompts = mod.build_crane_prompts(case, {"max_tokens": 96})
    result = generator(QWEN_SPEC, case, prompts)

    assert result.reasoning_token_count == 4
    assert result.full_certificate_text.startswith("<CARNOT_CERT_STATE:SAT>\n")
    assert len(FakeLlama.calls) == 2
    assert FakeLlama.calls[0]["kwargs"]["max_tokens"] == mod.CRANE_REASONING_BUDGET_TOKENS
    assert "grammar" not in FakeLlama.calls[0]["kwargs"]
    assert FakeLlama.calls[1]["prompt"].endswith(result.certificate_prefix)
    assert FakeLlama.calls[1]["kwargs"]["grammar"] is not None
    assert FakeLlama.calls[1]["kwargs"]["max_tokens"] == 96
    assert FakeGrammar.grammars[0].startswith("root ::= ")

    with pytest.raises(RuntimeError, match="model_path missing"):
        generator({"hf_id": QWEN_SPEC["hf_id"]}, case, prompts)


def test_scenario1366_run_experiment_writes_in_progress_then_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1366: runner persists bootstrap and terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    paths = {
        "exp1352_path": results / "exp1352.json",
        "exp1353_path": results / "exp1353.json",
        "exp1364_path": results / "exp1364.json",
    }
    paths["exp1352_path"].write_text(json.dumps(_exp1352()), encoding="utf-8")
    paths["exp1353_path"].write_text(json.dumps(_exp1353()), encoding="utf-8")
    paths["exp1364_path"].write_text(json.dumps(_exp1364()), encoding="utf-8")
    output_path = results / "exp1366.json"
    writes: list[dict[str, Any]] = []
    real_write = mod._write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(payload)
        real_write(path, payload)

    monkeypatch.setattr(mod, "_write_json", recording_write)

    artifact = mod.run_experiment(
        output_path=output_path,
        run_date="20260505",
        project_root=tmp_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        gpu_health_fn=lambda: mod.GPUHealth(True, 2, []),
        generation_fn=_perfect_generation,
        max_models=1,
        **paths,
    )

    assert [write["status"] for write in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
