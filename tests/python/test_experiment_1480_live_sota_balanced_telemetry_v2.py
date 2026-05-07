"""Tests for Exp 1480 balanced live SOTA telemetry v2 manifest.

Spec: REQ-INFER-SOTA-010,
      SCENARIO-INFER-SOTA-010-001,
      SCENARIO-INFER-SOTA-010-002
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting.live_sota_balanced_telemetry_v2 import (
    REQUIRED_ARTIFACT_FIELDS,
    SUPERFICIAL_BASELINE_FIELDS,
    BalancedTelemetryCase,
    RawBalancedGeneration,
    _answer_lexical_overlap,
    _coerce_float,
    _completion_text,
    _display_path,
    _evaluate_row_labels,
    _extract_completion_telemetry,
    _label_counts,
    _logits_summary,
    _resolved_specs,
    _superficial_baselines,
    _write_json,
    build_balanced_telemetry_artifact,
    build_balanced_telemetry_cases,
    run_experiment,
)


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA = "unsloth/gemma-4-31B-it-GGUF"
QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": QWEN,
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": GEMMA,
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _cached_pair(*, gpu_indices: tuple[int, int], preferred_quant: str) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    assert preferred_quant == "Q4_K_M"
    return [dict(QWEN_SPEC), dict(GEMMA_SPEC)]


def _response_for_case(case: BalancedTelemetryCase) -> str:
    if case.intended_correct and case.intended_format_valid:
        return case.expected_answer
    if not case.intended_correct and case.intended_format_valid:
        return case.adversarial_wrong_answer
    if case.intended_correct and not case.intended_format_valid:
        return f"The answer is {case.expected_answer}."
    return f"The answer is {case.adversarial_wrong_answer}."


def _topk_response(text: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "text": text,
                "logprobs": {
                    "tokens": [text],
                    "token_logprobs": [-0.12],
                    "top_logprobs": [
                        {
                            text: -0.12,
                            " 4": -1.4,
                            " 2": -2.0,
                            " 5": -2.6,
                            " 1": -3.0,
                        }
                    ],
                },
            }
        ],
        "usage": {"completion_tokens": 1},
    }


def _raw(text: str, *, topk: bool = True, logits: bool = True) -> RawBalancedGeneration:
    return RawBalancedGeneration(
        response_text=text,
        raw_response=_topk_response(text) if topk else {"choices": [{"text": text}]},
        elapsed_seconds=0.01,
        logits_available=logits,
        logits_shape=[1, 8] if logits else [],
    )


def test_exp1480_case_set_is_30_to_40_and_intentionally_balanced() -> None:
    """REQ-INFER-SOTA-010: default prompt set has balanced known labels."""
    cases = build_balanced_telemetry_cases()

    assert 30 <= len(cases) <= 40
    assert len(cases) == 36
    assert {case.family for case in cases} == {
        "fover_claim",
        "arithmetic_word_problem",
        "constraint_check",
    }
    buckets = {
        (case.intended_correct, case.intended_format_valid): 0
        for case in cases
    }
    for case in cases:
        buckets[(case.intended_correct, case.intended_format_valid)] += 1
        assert "legacy" not in case.prompt.lower()
        assert case.expected_answer
        assert case.adversarial_wrong_answer != case.expected_answer
    assert buckets == {
        (True, True): 9,
        (False, True): 9,
        (True, False): 9,
        (False, False): 9,
    }


def test_exp1480_builds_claim_allowed_balanced_manifest(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-010-001: balanced live rows open downstream audit path."""
    manifest_path = tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"

    artifact = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=manifest_path,
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, case: _raw(_response_for_case(case)),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == [QWEN]
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["telemetry_cases_requested"] == 36
    assert artifact["telemetry_cases_completed"] == 36
    assert artifact["balanced_label_counts"] == {
        "correct": 18,
        "incorrect": 18,
        "format_valid": 18,
        "format_invalid": 18,
        "correct_format_valid": 9,
        "incorrect_format_valid": 9,
        "correct_format_invalid": 9,
        "incorrect_format_invalid": 9,
    }
    assert artifact["topk_logprobs_available"] is True
    assert artifact["logits_available"] is True
    assert artifact["superficial_baselines_recorded"] is True
    assert artifact["claim_allowed"] is True
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"] == "balanced_live_sota_telemetry_ready"
    assert len(rows) == 36
    assert all(set(SUPERFICIAL_BASELINE_FIELDS) <= set(row["superficial_baselines"]) for row in rows)
    assert rows[0]["hf_id"] == QWEN
    assert rows[0]["generation_source"] == "live_sota_llamacpp"
    assert rows[0]["token_logprobs_available"] is True
    assert rows[0]["topk_alternatives_available"] is True
    assert rows[0]["logits_available"] is True


def test_exp1480_missing_cached_pair_does_not_generate_or_use_legacy(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-010-002: missing SOTA cache blocks without legacy fallback."""
    artifact = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        generation_fn=lambda _spec, _case: pytest.fail("generation must not run"),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
    )

    assert artifact["models_used"] == []
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["telemetry_cases_completed"] == 0
    assert artifact["balanced_label_counts"] == {}
    assert artifact["claim_allowed"] is False
    assert artifact["blockers"] == ["cached_sota_pair_not_loadable"]


def test_exp1480_unbalanced_or_missing_telemetry_blocks_claim_but_keeps_rows(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-010-002: completed rows are not enough without balance and telemetry."""
    manifest_path = tmp_path / "results" / "unbalanced.jsonl"

    artifact = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=manifest_path,
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, case: _raw(case.expected_answer, topk=False, logits=False),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert artifact["telemetry_cases_completed"] == 36
    assert artifact["balanced_label_counts"]["correct"] == 36
    assert artifact["balanced_label_counts"]["incorrect"] == 0
    assert artifact["superficial_baselines_recorded"] is True
    assert artifact["topk_logprobs_available"] is False
    assert artifact["logits_available"] is False
    assert artifact["claim_allowed"] is False
    assert "balanced_labels_missing" in artifact["blockers"]
    assert "topk_logprobs_unavailable_or_insufficient" in artifact["blockers"]
    assert "logits_unavailable_from_llamacpp_response" in artifact["blockers"]
    assert len(rows) == 36


def test_exp1480_label_and_superficial_baseline_helpers() -> None:
    """REQ-INFER-SOTA-010: row labels and superficial confounds are deterministic."""
    case = BalancedTelemetryCase(
        case_id="case",
        family="fover_claim",
        prompt="Return exactly The answer is 3.",
        expected_answer="3",
        adversarial_wrong_answer="13",
        intended_correct=True,
        intended_format_valid=False,
    )

    labels = _evaluate_row_labels(case, "The answer is 3.")
    json_labels = _evaluate_row_labels(case, '{"answer": "3"}')
    leading_think_labels = _evaluate_row_labels(case, "13\n\n<think>\nunfinished")
    baselines = _superficial_baselines(
        case,
        response_text='{"answer": "3"}',
        completion_tokens=4,
        spec={"hf_id": QWEN},
    )

    assert labels == {"correct": True, "format_valid": False}
    assert json_labels == {"correct": True, "format_valid": False}
    assert leading_think_labels == {"correct": False, "format_valid": True}
    assert _answer_lexical_overlap("13", "3") == 0.0
    assert _answer_lexical_overlap("The answer is 3.", "3") == 1.0
    assert baselines["response_length"] == len('{"answer": "3"}')
    assert baselines["token_count"] == 4
    assert baselines["json_valid"] is True
    assert baselines["schema_valid"] is False
    assert baselines["prompt_family"] == "fover_claim"
    assert baselines["answer_lexical_overlap"] == 1.0
    assert baselines["model_family"] == "qwen_moe"


def test_exp1480_normalization_helpers_cover_alternate_runtime_shapes(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-010: defensive runtime normalization stays deterministic."""
    output = tmp_path / "nested" / "artifact.json"
    _write_json(output, {"status": "complete"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
    assert _display_path(tmp_path, Path("/outside/manifest.jsonl")).startswith("/outside/")
    assert _coerce_float(True) is None
    assert _coerce_float("not-a-number") is None
    assert _completion_text("plain") == "plain"
    assert _completion_text(123) == ""
    assert _completion_text({"choices": []}) == ""
    assert _completion_text({"choices": [{"message": {"content": "chat text"}}]}) == "chat text"
    assert _completion_text({"choices": [{"message": "not-a-chat-message"}]}) == ""

    telemetry = _extract_completion_telemetry(
        {
            "choices": [
                {
                    "message": {"content": "answer"},
                    "logprobs": {
                        "tokens": ["answer", None],
                        "token_logprobs": [None, True, "-0.2"],
                        "top_logprobs": [None, {"answer": "-0.2", "bad": False}],
                    },
                }
            ],
            "usage": {"completion_tokens": "bad"},
        }
    )
    assert telemetry["response_text"] == "answer"
    assert telemetry["completion_tokens"] == 0
    assert telemetry["token_texts"] == ["answer"]
    assert telemetry["token_logprobs"] == [-0.2]
    assert telemetry["top_logprobs"] == [{"answer": -0.2}]

    class ShapeScores:
        shape = (2, 3)

    class WithShape:
        scores = ShapeScores()

    class WithEvalLogits:
        scores = None
        eval_logits = [[0.0, 1.0], [2.0, 3.0]]

    class WithoutLogits:
        scores = None
        eval_logits = None

    assert _logits_summary(WithShape()) == (True, [2, 3])
    assert _logits_summary(WithEvalLogits()) == (True, [2, 2])
    assert _logits_summary(WithoutLogits()) == (False, [])
    assert _resolved_specs(None) == []
    assert _resolved_specs([{"hf_id": "legacy/small", "model_path": "/legacy.gguf"}, QWEN_SPEC]) == [
        QWEN_SPEC
    ]
    assert _label_counts([]) == {}
    assert _superficial_baselines(
        BalancedTelemetryCase("c", "constraint_check", "p", "1", "0", True, True),
        response_text="1",
        completion_tokens=1,
        spec={"hf_id": GEMMA},
    )["model_family"] == "gemma_dense"
    assert _superficial_baselines(
        BalancedTelemetryCase("c", "constraint_check", "p", "1", "0", True, True),
        response_text="1",
        completion_tokens=1,
        spec={"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"},
    )["model_family"] == "gemma_moe"
    assert _superficial_baselines(
        BalancedTelemetryCase("c", "constraint_check", "p", "1", "0", True, True),
        response_text="1",
        completion_tokens=1,
        spec={"hf_id": "unknown/model"},
    )["model_family"] == "unknown_or_non_sota"


def test_exp1480_run_experiment_writes_in_progress_then_final(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-010: runner persists bootstrap JSON before final manifest."""
    writes: list[dict[str, Any]] = []
    output_path = tmp_path / "results" / "experiment_1480_live_sota_balanced_telemetry_v2.json"
    manifest_path = tmp_path / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        output_path=output_path,
        manifest_path=manifest_path,
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, case: _raw(_response_for_case(case)),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
        write_json_fn=lambda path, payload: (
            writes.append(dict(payload)),
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_text(json.dumps(payload), encoding="utf-8"),
        ),
    )

    assert writes[0]["status"] == "in_progress"
    assert writes[-1] == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert manifest_path.is_file()


def test_exp1480_default_writer_persists_artifact(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-010: default artifact writer is stable JSON."""
    output_path = tmp_path / "results" / "experiment_1480_live_sota_balanced_telemetry_v2.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        output_path=output_path,
        manifest_path=tmp_path / "results" / "manifest.jsonl",
        cached_pair_fn=_cached_pair,
        cases=build_balanced_telemetry_cases()[:1],
        generation_fn=lambda _spec, case: _raw(case.expected_answer),
        generation_source="injected",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["live_sota_model_inference_used"] is False
    assert "live_sota_model_inference_not_used" in artifact["blockers"]


def test_exp1480_live_llama_path_requests_topk_and_logits(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-010: live llama.cpp collection asks for top-k and logits."""
    init_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}
    cases = [
        BalancedTelemetryCase(
            case_id="case-live",
            family="arithmetic_word_problem",
            prompt="Output exactly: 3",
            expected_answer="3",
            adversarial_wrong_answer="4",
            intended_correct=True,
            intended_format_valid=True,
        )
    ]

    class FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)
            self.scores = [[0.0, 1.0, 2.0]]
            self.closed = False

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert "Output exactly" in prompt
            call_kwargs.update(kwargs)
            return _topk_response("3")

        def close(self) -> None:
            self.closed = True

    artifact = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "live.jsonl",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, FakeLlama, None),
        cases=cases,
        generation_fn=None,
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    assert init_kwargs["model_path"] == QWEN_SPEC["model_path"]
    assert init_kwargs["logits_all"] is True
    assert init_kwargs["n_gpu_layers"] == -1
    assert call_kwargs["logprobs"] == 5
    assert call_kwargs["max_tokens"] == 24
    assert artifact["telemetry_cases_completed"] == 1
    assert artifact["topk_logprobs_available"] is True
    assert artifact["logits_available"] is True


def test_exp1480_error_paths_remain_explicit(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-010-002: cache, import, load, and generation failures are blockers."""
    one_case = build_balanced_telemetry_cases()[:1]

    cache_error = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "cache_error.jsonl",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: (_ for _ in ()).throw(
            RuntimeError("pair failed")
        ),
        generation_fn=lambda _spec, _case: pytest.fail("generation must not run"),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
    )
    assert cache_error["blockers"] == [
        "cached_sota_pair_not_loadable",
        "RuntimeError: pair failed",
    ]

    generation_error = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "generation_error.jsonl",
        cached_pair_fn=_cached_pair,
        cases=one_case,
        generation_fn=lambda _spec, _case: (_ for _ in ()).throw(RuntimeError("boom")),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )
    assert generation_error["telemetry_cases_completed"] == 0
    assert "no_live_sota_generation_completed" in generation_error["blockers"]
    assert "superficial_baselines_missing" in generation_error["blockers"]
    assert any("RuntimeError: boom" in blocker for blocker in generation_error["blockers"])

    import_error = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "import_error.jsonl",
        cached_pair_fn=_cached_pair,
        cases=one_case,
        generation_fn=None,
        llama_importer=lambda: (False, None, "ImportError: no llama_cpp"),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )
    assert any("ImportError: no llama_cpp" in blocker for blocker in import_error["blockers"])

    class BadInitLlama:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    load_error = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "load_error.jsonl",
        cached_pair_fn=_cached_pair,
        cases=one_case,
        generation_fn=None,
        llama_importer=lambda: (True, BadInitLlama, None),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )
    assert any("RuntimeError: load failed" in blocker for blocker in load_error["blockers"])


def test_exp1480_typeerror_fallback_records_logprob_blocker(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-010: llama.cpp logprob TypeError falls back honestly."""
    cases = [
        BalancedTelemetryCase(
            case_id="case-live",
            family="arithmetic_word_problem",
            prompt="Output exactly: 3",
            expected_answer="3",
            adversarial_wrong_answer="4",
            intended_correct=True,
            intended_format_valid=True,
        )
    ]

    class NoLogprobLlama:
        scores = None

        def __call__(self, _prompt: str, **kwargs: Any) -> dict[str, Any]:
            if "logprobs" in kwargs:
                raise TypeError("logprobs unsupported")
            return {"choices": [{"text": "3"}], "usage": {"completion_tokens": 1}}

    artifact = build_balanced_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "typeerror.jsonl",
        cached_pair_fn=_cached_pair,
        cases=cases,
        generation_fn=None,
        llama_importer=lambda: (True, lambda **_kwargs: NoLogprobLlama(), None),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "typeerror.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["response_text"] == "3"
    assert rows[0]["logprob_error"] == "logprobs_unavailable: logprobs unsupported"
    assert rows[0]["token_logprobs_available"] is False
    assert artifact["logits_available"] is False
