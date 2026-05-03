"""Tests for Exp 1170 BEAVER-lite llama.cpp completion logprobs.

Spec: REQ-VERIFY-1170, SCENARIO-VERIFY-1170
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_lite import CompletionCandidate  # noqa: E402
from carnot.verify.beaver_lite_live import (  # noqa: E402
    BEAVERLiteVerifier,
    LlamaCppCompletionLogprobProvider,
    run_beaver_live_logprobs_v2_experiment,
)


class _RecordingLlama:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"prompt": prompt, **kwargs})
        index = len(self.calls) - 1
        text = " Final answer: seven" if index == 0 else " Final answer: 7"
        token_logprobs = [None, math.log(0.6)] if index == 0 else [math.log(0.3)]
        return {
            "choices": [
                {
                    "text": text,
                    "logprobs": {
                        "tokens": [" Final", " answer"],
                        "token_logprobs": token_logprobs,
                    },
                }
            ]
        }


class _LiveFixtureProvider:
    mock_logprobs_used = False
    logprobs_source = "llama_cpp_logits_all"

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        del prompt, max_tokens
        candidates = (
            ("Final answer: seven", 0.35),
            ("No numeric final answer", 0.25),
            ("Final answer: 7", 0.20),
            ("Final answer: 8", 0.20),
        )
        return [
            CompletionCandidate(text, (text, "<eos>"), math.log(probability), True)
            for text, probability in candidates[:top_k]
        ]


def _write_fover_jsonl(path: Path, n_rows: int = 10) -> None:
    rows = [
        {
            "question_id": str(index),
            "step_text": f"FoVer arithmetic step {index}: {index} + 1 = {index + 1}.",
            "label": "correct",
        }
        for index in range(n_rows)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_completion_provider_sets_logits_all_and_extracts_token_logprobs() -> None:
    """REQ-VERIFY-1170-1, REQ-VERIFY-1170-2: live provider uses token logprobs."""

    created: list[_RecordingLlama] = []

    def llama_factory(**kwargs: object) -> _RecordingLlama:
        llama = _RecordingLlama(**kwargs)
        created.append(llama)
        return llama

    provider = LlamaCppCompletionLogprobProvider(
        "/models/tiny.gguf",
        llama_factory=llama_factory,
        n_ctx=128,
    )
    completions = provider.enumerate_completions("prompt", top_k=2, max_tokens=4)

    assert provider.mock_logprobs_used is False
    assert provider.logprobs_source == "llama_cpp_logits_all"
    assert created[0].kwargs["model_path"] == "/models/tiny.gguf"
    assert created[0].kwargs["logits_all"] is True
    assert created[0].kwargs["n_ctx"] == 128
    assert created[0].calls[0]["logprobs"] == 2
    assert completions[0].logprob == pytest.approx(math.log(0.6))
    assert completions[1].logprob == pytest.approx(math.log(0.3))
    assert completions[0].tokens == (" Final", " answer", "<eos>")


def test_beaver_lite_verifier_wrapper_reports_sound_live_bound() -> None:
    """REQ-VERIFY-1170-2: verifier wrapper feeds logprob candidates to BEAVER-lite."""

    verifier = BEAVERLiteVerifier(provider=_LiveFixtureProvider(), top_k=4, max_tokens=4)
    evaluation = verifier.evaluate_question("How much is 3 + 4?")

    assert evaluation.n_completions == 4
    assert evaluation.unsafe_mass_bound == pytest.approx(0.60)
    assert evaluation.empirical_violation_rate == pytest.approx(0.50)
    assert evaluation.bound_is_sound is True


def test_exp1170_live_provider_writes_required_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-1170-3, REQ-VERIFY-1170-4, REQ-VERIFY-1170-5: live artifact."""

    fover_path = tmp_path / "fover.jsonl"
    output_path = tmp_path / "experiment_1170_beaver_live_logprobs_v2.json"
    _write_fover_jsonl(fover_path)

    artifact = run_beaver_live_logprobs_v2_experiment(
        output_path=output_path,
        fover_corpus_path=fover_path,
        llama_cpp_available_override=True,
        model_path="/models/tiny.gguf",
        live_provider_factory=lambda model_path: _LiveFixtureProvider(),
        top_k=4,
        max_tokens=4,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["mock_logprobs_used"] is False
    assert artifact["logprobs_source"] == "llama_cpp_logits_all"
    assert artifact["bound_is_sound"] is True
    assert artifact["sample_bound_values"] == pytest.approx([0.60] * 5)
    assert artifact["n_test_prompts_run"] == 10
    assert artifact["honest_verdict"] == "live_logprobs_sound_bound"


def test_exp1170_zipf_fallback_is_honest_when_logprobs_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-1170-4, REQ-VERIFY-1170-5: unavailable live logprobs use fallback."""

    fover_path = tmp_path / "fover.jsonl"
    _write_fover_jsonl(fover_path)

    artifact = run_beaver_live_logprobs_v2_experiment(
        output_path=tmp_path / "fallback.json",
        fover_corpus_path=fover_path,
        llama_cpp_available_override=False,
        model_path=None,
        top_k=10,
        max_tokens=4,
    )

    assert artifact["mock_logprobs_used"] is True
    assert artifact["logprobs_source"] == "zipf_mock"
    assert artifact["bound_is_sound"] is True
    assert len(artifact["sample_bound_values"]) == 5
    assert artifact["n_test_prompts_run"] == 10
    assert artifact["honest_verdict"] == "logprobs_unavailable_mock_fallback"
