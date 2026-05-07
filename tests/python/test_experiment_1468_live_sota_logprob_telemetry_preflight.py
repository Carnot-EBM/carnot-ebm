"""Tests for Exp 1468 live SOTA logprob telemetry preflight.

Spec: REQ-INFER-SOTA-009,
      SCENARIO-INFER-SOTA-009-001,
      SCENARIO-INFER-SOTA-009-002
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting.live_sota_logprob_telemetry_preflight import (
    REQUIRED_ARTIFACT_FIELDS,
    RawTelemetryGeneration,
    TelemetryCase,
    _completion_text,
    _display_path,
    _extract_completion_telemetry,
    _generate_with_llama,
    _logits_summary,
    _write_json,
    build_telemetry_artifact,
    build_telemetry_cases,
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


def _topk_response(text: str = " 3") -> dict[str, Any]:
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


def _one_case() -> list[TelemetryCase]:
    return [
        TelemetryCase(
            case_id="fover_gsm8k_verified_001",
            family="gsm8k_style",
            prompt="Mia has 1 marble and gets 2 more. Answer with the final integer only.",
            expected_answer="3",
        )
    ]


def test_exp1468_case_set_is_bounded_and_verified_style() -> None:
    """REQ-INFER-SOTA-009: the default prompt set has 10-20 verified bounded cases."""
    cases = build_telemetry_cases()

    assert 10 <= len(cases) <= 20
    assert cases[0].case_id.startswith("fover_gsm8k_verified_")
    assert {case.family for case in cases} == {"gsm8k_style", "fover_style"}
    assert all(case.expected_answer for case in cases)
    assert all(len(case.prompt) < 500 for case in cases)


def test_exp1468_builds_complete_artifact_with_topk_manifest(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-009 / SCENARIO-INFER-SOTA-009-001: top-k telemetry opens path."""

    def generation_fn(spec: dict[str, Any], case: TelemetryCase) -> RawTelemetryGeneration:
        assert spec["hf_id"] == QWEN
        return RawTelemetryGeneration(
            response_text=f" {case.expected_answer}",
            raw_response=_topk_response(f" {case.expected_answer}"),
            elapsed_seconds=0.05,
            logits_available=True,
            logits_shape=[1, 8],
        )

    manifest_path = tmp_path / "results" / "live_sota_telemetry_manifest_1468.jsonl"
    artifact = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=manifest_path,
        cached_pair_fn=_cached_pair,
        generation_fn=generation_fn,
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["models_used"] == [QWEN]
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["telemetry_cases_requested"] == len(build_telemetry_cases())
    assert artifact["telemetry_cases_completed"] == len(build_telemetry_cases())
    assert artifact["topk_logprobs_available"] is True
    assert artifact["logits_available"] is True
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"] == "live_sota_topk_telemetry_ready"
    assert len(rows) == len(build_telemetry_cases())
    assert rows[0]["hf_id"] == QWEN
    assert rows[0]["response_text"].strip() == rows[0]["expected_answer"]
    assert rows[0]["token_texts"]
    assert rows[0]["token_logprobs_available"] is True
    assert rows[0]["topk_alternatives_available"] is True
    assert rows[0]["logits_available"] is True


def test_exp1468_missing_topk_stays_blocked_but_records_text(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-009-002: response text alone is not top-k readiness."""

    artifact = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "live_sota_telemetry_manifest_1468.jsonl",
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, case: RawTelemetryGeneration(
            response_text=f" {case.expected_answer}",
            raw_response={"choices": [{"text": f" {case.expected_answer}", "logprobs": None}]},
            elapsed_seconds=0.05,
        ),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "live_sota_telemetry_manifest_1468.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["telemetry_cases_completed"] == len(build_telemetry_cases())
    assert artifact["topk_logprobs_available"] is False
    assert artifact["logits_available"] is False
    assert "topk_logprobs_unavailable_or_insufficient" in artifact["blockers"]
    assert rows[0]["response_text"].strip() == rows[0]["expected_answer"]
    assert rows[0]["topk_alternatives_available"] is False


def test_exp1468_missing_cached_pair_does_not_generate(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-009: missing SOTA cache blocks without legacy model fallback."""

    artifact = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "live_sota_telemetry_manifest_1468.jsonl",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: None,
        generation_fn=lambda _spec, _case: pytest.fail("generation must not run"),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
    )

    assert artifact["status"] == "complete"
    assert artifact["models_used"] == []
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["telemetry_cases_completed"] == 0
    assert artifact["topk_logprobs_available"] is False
    assert "cached_sota_pair_not_loadable" in artifact["blockers"]


def test_exp1468_extracts_completion_telemetry_and_drops_nulls() -> None:
    """REQ-INFER-SOTA-009: token text, token logprobs, and top-k alternatives parse."""

    class FloatLike:
        def __init__(self, value: float) -> None:
            self.value = value

        def __float__(self) -> float:
            return self.value

    telemetry = _extract_completion_telemetry(
        {
            "choices": [
                {
                    "text": "answer",
                    "logprobs": {
                        "tokens": ["prompt", "answer"],
                        "token_logprobs": [None, True, FloatLike(-0.2)],
                        "top_logprobs": [
                            None,
                            {"skip_bool": False, "answer": FloatLike(-0.2), "wrong": -1.9},
                        ],
                    },
                }
            ],
            "usage": {"completion_tokens": 2},
        }
    )

    assert telemetry["response_text"] == "answer"
    assert telemetry["token_texts"] == ["prompt", "answer"]
    assert telemetry["token_logprobs"] == [-0.2]
    assert telemetry["top_logprobs"] == [{"answer": -0.2, "wrong": -1.9}]
    assert telemetry["completion_tokens"] == 2


def test_exp1468_run_experiment_writes_in_progress_then_final(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-009: runner persists bootstrap JSON and final manifest."""
    writes: list[dict[str, Any]] = []
    output_path = tmp_path / "results" / "experiment_1468_live_sota_logprob_telemetry_preflight.json"
    manifest_path = tmp_path / "results" / "live_sota_telemetry_manifest_1468.jsonl"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        output_path=output_path,
        manifest_path=manifest_path,
        cached_pair_fn=_cached_pair,
        generation_fn=lambda _spec, _case: RawTelemetryGeneration(
            response_text=" 3",
            raw_response=_topk_response(" 3"),
            elapsed_seconds=0.01,
        ),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
        write_json_fn=lambda path, payload: (writes.append(dict(payload)), path.write_text(json.dumps(payload), encoding="utf-8")),
    )

    assert writes[0]["status"] == "in_progress"
    assert writes[-1] == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert manifest_path.is_file()


def test_exp1468_live_llama_path_requests_logprobs_and_logits_all(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-009: live llama.cpp collection asks for top-k and logits."""
    init_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}

    class FakeLlama:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)
            self.scores = [[0.0, 1.0, 2.0]]
            self.closed = False

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert "final integer only" in prompt
            call_kwargs.update(kwargs)
            return _topk_response(" 3")

        def close(self) -> None:
            self.closed = True

    artifact = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "live_sota_telemetry_manifest_1468.jsonl",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, FakeLlama, None),
        cases=_one_case(),
        generation_fn=None,
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
    )

    assert init_kwargs["model_path"] == QWEN_SPEC["model_path"]
    assert init_kwargs["logits_all"] is True
    assert init_kwargs["n_gpu_layers"] == -1
    assert call_kwargs["logprobs"] == 5
    assert call_kwargs["max_tokens"] == 48
    assert artifact["telemetry_cases_completed"] == 1
    assert artifact["topk_logprobs_available"] is True
    assert artifact["logits_available"] is True


def test_exp1468_helper_branches_cover_completion_logits_and_writer(tmp_path: Path) -> None:
    """REQ-INFER-SOTA-009: helper normalization covers alternate runtime shapes."""
    output = tmp_path / "nested" / "artifact.json"
    _write_json(output, {"status": "complete"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
    assert _display_path(tmp_path, Path("/outside/live_sota_telemetry_manifest_1468.jsonl")).startswith(
        "/outside/"
    )
    assert _completion_text("plain") == "plain"
    assert _completion_text({"choices": []}) == ""
    assert _completion_text({"choices": [{"message": {"content": "chat text"}}]}) == "chat text"
    assert _completion_text({"choices": [{"message": "not-a-chat-message"}]}) == ""

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


def test_exp1468_typeerror_fallback_records_logprob_blocker() -> None:
    """REQ-INFER-SOTA-009: llama.cpp logprob TypeError falls back honestly."""

    class NoLogprobLlama:
        scores = None

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, _prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            if "logprobs" in kwargs:
                raise TypeError("logprobs unsupported")
            return {"choices": [{"text": " 3", "logprobs": None}], "usage": {"completion_tokens": 1}}

    raw = _generate_with_llama(NoLogprobLlama(), _one_case()[0])

    assert raw.response_text == " 3"
    assert raw.logprob_error == "logprobs_unavailable: logprobs unsupported"
    assert raw.logits_available is False


def test_exp1468_failure_branches_preserve_precise_blockers(tmp_path: Path) -> None:
    """SCENARIO-INFER-SOTA-009-002: cache/import/generation failures stay explicit."""
    case = _one_case()

    generation_error = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "generation_error.jsonl",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: [
            {"hf_id": "legacy/small-model", "model_path": "/legacy.gguf"},
            dict(QWEN_SPEC),
        ],
        generation_fn=lambda _spec, _case: (_ for _ in ()).throw(RuntimeError("boom")),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
        cases=case,
    )
    assert generation_error["telemetry_cases_completed"] == 0
    assert "no_live_sota_generation_completed" in generation_error["blockers"]
    assert any("RuntimeError: boom" in blocker for blocker in generation_error["blockers"])

    import_error = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "import_error.jsonl",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (False, None, "ImportError: no llama_cpp"),
        generation_fn=None,
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
        cases=case,
    )
    assert any("ImportError: no llama_cpp" in blocker for blocker in import_error["blockers"])

    class BadInitLlama:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    load_error = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "load_error.jsonl",
        cached_pair_fn=_cached_pair,
        llama_importer=lambda: (True, BadInitLlama, None),
        generation_fn=None,
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 2, "cuda_available": True},
        cases=case,
    )
    assert any("RuntimeError: load failed" in blocker for blocker in load_error["blockers"])

    cache_error = build_telemetry_artifact(
        project_root=tmp_path,
        run_date="20260507",
        manifest_path=tmp_path / "results" / "cache_error.jsonl",
        cached_pair_fn=lambda *, gpu_indices, preferred_quant: (_ for _ in ()).throw(
            RuntimeError("pair failed")
        ),
        generation_fn=lambda _spec, _case: pytest.fail("generation must not run"),
        generation_source="live_sota_llamacpp",
        gpu_probe_fn=lambda: {"gpu_count": 0, "cuda_available": False},
        cases=case,
    )
    assert cache_error["blockers"] == [
        "cached_sota_pair_not_loadable",
        "RuntimeError: pair failed",
    ]
