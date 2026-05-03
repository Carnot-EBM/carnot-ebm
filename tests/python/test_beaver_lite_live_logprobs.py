"""Tests for Exp 1158 BEAVER-lite live-or-Zipf logprob certificates.

Spec: REQ-VERIFY-1158, SCENARIO-VERIFY-1158
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_lite import BEAVERLiteBounder, CompletionCandidate  # noqa: E402
from carnot.verify.beaver_lite_live import (  # noqa: E402
    EXP1142_MOCK_PRIOR_BOUND,
    INSTALL_LLAMA_CPP_COMMAND,
    BounderSelection,
    QuestionEvaluation,
    ZipfMockLogprobProvider,
    build_experiment_1158_artifact,
    llama_cpp_available,
    resolve_cached_gguf_model_path,
    run_beaver_lite_live_logprob_experiment,
    select_beaver_lite_bounder,
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "experiment_1158_beaver_lite_live_logprobs",
        REPO_ROOT / "scripts" / "experiment_1158_beaver_lite_live_logprobs.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeterministicLiveProvider:
    """Small live-shaped provider for tests that avoids llama.cpp imports."""

    mock_logprobs_used = False

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        """Return non-uniform terminal completions with one invalid answer.

        Spec: REQ-VERIFY-1158-2
        """

        del prompt, top_k, max_tokens
        return [
            CompletionCandidate("Final answer: seven", ("seven", "<eos>"), math.log(0.5)),
            CompletionCandidate("Final answer: 8", ("8", "<eos>"), math.log(0.3)),
            CompletionCandidate("Final answer: 7", ("7", "<eos>"), math.log(0.2)),
        ]


def test_zipf_provider_is_nonuniform_sound_and_tighter_than_uniform_mock() -> None:
    """The fallback is Zipf-distributed rather than Exp 1142 uniform mass.

    Spec: REQ-VERIFY-1158-3, REQ-VERIFY-1158-5, SCENARIO-VERIFY-1158
    """

    provider = ZipfMockLogprobProvider(alpha=1.0)
    completions = provider.enumerate_completions("prompt", top_k=10, max_tokens=8)
    probabilities = [math.exp(candidate.logprob) for candidate in completions]

    assert sum(probabilities) == pytest.approx(1.0)
    assert max(probabilities) > min(probabilities)
    assert probabilities == sorted(probabilities, reverse=True)

    result = BEAVERLiteBounder(provider=provider, top_k=10).bound_prefix_violation(
        "Janet has 10 marbles and gives away 3. How many remain?"
    )
    expected_zipf_unsafe_mass = (1 / 2 + 1 / 3 + 1 / 10) / sum(1 / rank for rank in range(1, 11))

    assert result.n_completions == 10
    assert result.mock_logprobs_used is True
    assert result.empirical_rate == pytest.approx(0.3)
    assert result.upper_bound == pytest.approx(expected_zipf_unsafe_mass)
    assert result.bound_is_sound is True
    assert result.bound_gap > 0.0
    assert result.upper_bound < EXP1142_MOCK_PRIOR_BOUND


def test_llama_cpp_detection_and_cached_model_resolution(tmp_path: Path) -> None:
    """Environment probing is deterministic and does not require live downloads.

    Spec: REQ-VERIFY-1158-2, REQ-VERIFY-1158-3
    """

    assert llama_cpp_available(importer=lambda name: object()) is True

    def missing_import(name: str) -> object:
        del name
        raise ImportError("not installed")

    assert llama_cpp_available(importer=missing_import) is False

    env_model = tmp_path / "env-model.gguf"
    env_model.write_text("fake", encoding="utf-8")
    assert resolve_cached_gguf_model_path(
        env={"CARNOT_BEAVER_LITE_GGUF": str(env_model)}, cache_root=tmp_path / "empty"
    ) == str(env_model)

    cache_model = (
        tmp_path / "models--Qwen--Qwen3.5-0.8B-GGUF" / "snapshots" / "abc123" / "qwen35.gguf"
    )
    cache_model.parent.mkdir(parents=True)
    cache_model.write_text("fake", encoding="utf-8")

    assert resolve_cached_gguf_model_path(env={}, cache_root=tmp_path) == str(cache_model)
    assert resolve_cached_gguf_model_path(env={}, cache_root=tmp_path / "missing") is None


def test_select_bounder_uses_zipf_only_when_llama_cpp_is_unavailable() -> None:
    """The fallback path honestly reports mock and Zipf status.

    Spec: REQ-VERIFY-1158-3, REQ-VERIFY-1158-4
    """

    selection = select_beaver_lite_bounder(
        llama_cpp_is_available=False,
        model_path=None,
        top_k=10,
        max_tokens=8,
    )

    assert selection.bounder is not None
    assert selection.llama_cpp_available is False
    assert selection.model_used is None
    assert selection.mock_logprobs_used is True
    assert selection.zipf_mock_used is True
    assert selection.install_command == INSTALL_LLAMA_CPP_COMMAND
    assert selection.blocked_reason is None


def test_select_bounder_blocks_when_llama_cpp_exists_but_model_is_missing() -> None:
    """A missing cached GGUF is a blocked live run, not a mock-logprob claim.

    Spec: REQ-VERIFY-1158-2, REQ-VERIFY-1158-4, REQ-VERIFY-1158-5
    """

    selection = select_beaver_lite_bounder(
        llama_cpp_is_available=True,
        model_path=None,
        top_k=10,
        max_tokens=8,
    )

    assert selection.bounder is None
    assert selection.llama_cpp_available is True
    assert selection.model_used is None
    assert selection.mock_logprobs_used is False
    assert selection.zipf_mock_used is False
    assert selection.blocked_reason == "cached_gguf_model_not_found"


def test_live_provider_selection_and_artifact_schema(tmp_path: Path) -> None:
    """A live-shaped provider produces the required Exp 1158 artifact fields.

    Spec: REQ-VERIFY-1158-1, REQ-VERIFY-1158-2, REQ-VERIFY-1158-4,
          REQ-VERIFY-1158-5, SCENARIO-VERIFY-1158
    """

    model_path = tmp_path / "qwen35.gguf"
    model_path.write_text("fake", encoding="utf-8")

    artifact = run_beaver_lite_live_logprob_experiment(
        output_path=tmp_path / "experiment_1158_beaver_lite_live_logprobs.json",
        llama_cpp_available_override=True,
        model_path=str(model_path),
        live_provider_factory=lambda path: DeterministicLiveProvider(),
    )

    required = {
        "llama_cpp_available",
        "model_used",
        "mock_logprobs_used",
        "n_questions_evaluated",
        "unsafe_mass_bound_live",
        "empirical_violation_rate_live",
        "bound_gap_live",
        "bound_is_sound_live",
        "unsafe_mass_bound_mock_prior",
        "bound_tighter_than_mock",
        "beaver_lite_live_logprobs_sound_bound",
        "honest_verdict",
    }
    assert required <= set(artifact)
    assert artifact["llama_cpp_available"] is True
    assert artifact["model_used"] == str(model_path)
    assert artifact["mock_logprobs_used"] is False
    assert artifact["zipf_mock_used"] is False
    assert artifact["n_questions_evaluated"] == 3
    assert artifact["unsafe_mass_bound_live"] == pytest.approx(0.5)
    assert artifact["empirical_violation_rate_live"] == pytest.approx(1 / 3)
    assert artifact["bound_is_sound_live"] is True
    assert artifact["bound_tighter_than_mock"] is False
    assert artifact["honest_verdict"] == "sound_bound_live_logprobs"


def test_zipf_run_writes_stable_artifact_when_llama_cpp_is_absent(tmp_path: Path) -> None:
    """The no-llama path writes a sound Zipf artifact for three questions.

    Spec: REQ-VERIFY-1158-1, REQ-VERIFY-1158-3, REQ-VERIFY-1158-4,
          REQ-VERIFY-1158-5, SCENARIO-VERIFY-1158
    """

    output_path = tmp_path / "experiment_1158_beaver_lite_live_logprobs.json"
    artifact = run_beaver_lite_live_logprob_experiment(
        output_path=output_path,
        llama_cpp_available_override=False,
        top_k=10,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["llama_cpp_available"] is False
    assert artifact["model_used"] is None
    assert artifact["mock_logprobs_used"] is True
    assert artifact["zipf_mock_used"] is True
    assert artifact["llama_cpp_install_command"] == INSTALL_LLAMA_CPP_COMMAND
    assert artifact["n_questions_evaluated"] == 3
    assert artifact["bound_is_sound_live"] is True
    assert artifact["beaver_lite_live_logprobs_sound_bound"] is True
    assert artifact["bound_tighter_than_mock"] is True
    assert artifact["honest_verdict"] == "sound_bound_zipf_mock"


def test_blocked_and_bug_verdicts_are_encoded_explicitly(tmp_path: Path) -> None:
    """Blocked and violated cases use the constrained verdict vocabulary.

    Spec: REQ-VERIFY-1158-4, REQ-VERIFY-1158-5
    """

    blocked = run_beaver_lite_live_logprob_experiment(
        output_path=tmp_path / "blocked.json",
        llama_cpp_available_override=True,
        model_path=None,
    )
    assert blocked["honest_verdict"] == "llm_not_available_blocked"
    assert blocked["n_questions_evaluated"] == 0
    assert blocked["mock_logprobs_used"] is False

    violated = build_experiment_1158_artifact(
        selection=BounderSelection(
            bounder=BEAVERLiteBounder(provider=ZipfMockLogprobProvider(), top_k=10),
            llama_cpp_available=False,
            model_used=None,
            mock_logprobs_used=True,
            zipf_mock_used=True,
            install_command=INSTALL_LLAMA_CPP_COMMAND,
            blocked_reason=None,
        ),
        question_evaluations=(
            QuestionEvaluation(
                question="q",
                unsafe_mass_bound=0.1,
                empirical_violation_rate=0.2,
                bound_gap=-0.1,
                bound_is_sound=False,
                n_completions=10,
            ),
        ),
    )

    assert violated["honest_verdict"] == "bound_violated_bug"
    assert violated["beaver_lite_live_logprobs_sound_bound"] is False


def test_script_entrypoint_writes_requested_artifact(tmp_path: Path) -> None:
    """The conductor-facing script delegates to the tested Exp 1158 module.

    Spec: REQ-VERIFY-1158-4, SCENARIO-VERIFY-1158
    """

    script = _load_script_module()
    output_path = tmp_path / "script-artifact.json"
    artifact = script.run_experiment(
        output_path=output_path,
        llama_cpp_available_override=False,
        top_k=10,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "sound_bound_zipf_mock"
