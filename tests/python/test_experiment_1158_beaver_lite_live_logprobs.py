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

from carnot.verify.beaver_lite import CompletionCandidate  # noqa: E402
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
    while str(PYTHON_DIR) in sys.path:
        sys.path.remove(str(PYTHON_DIR))
    spec = importlib.util.spec_from_file_location(
        "experiment_1158_beaver_lite_live_logprobs",
        REPO_ROOT / "scripts" / "experiment_1158_beaver_lite_live_logprobs.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LiveFixtureProvider:
    mock_logprobs_used = False

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        del prompt, max_tokens
        candidates = (
            ("Final answer: seven", 0.5),
            ("Final answer: 7", 0.2),
            ("Final answer: 8", 0.2),
            ("Final answer: 9", 0.1),
        )
        return [
            CompletionCandidate(
                text=text,
                tokens=(text, "<eos>"),
                logprob=math.log(probability),
                terminal=True,
            )
            for text, probability in candidates[:top_k]
        ]


def test_zipf_fallback_uses_nonuniform_mock_when_llama_cpp_unavailable() -> None:
    """Unavailable llama.cpp uses a deterministic Zipf fallback.

    Spec: REQ-VERIFY-1158-3, SCENARIO-VERIFY-1158
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

    result = selection.bounder.bound_prefix_violation("How much is 4 plus 5?")

    assert result.mock_logprobs_used is True
    assert result.n_completions == 10
    assert result.upper_bound == pytest.approx(0.31865600867091187)
    assert result.empirical_rate == pytest.approx(0.3)
    assert result.bound_is_sound is True
    assert result.upper_bound < EXP1142_MOCK_PRIOR_BOUND


def test_live_selection_requires_cached_model_path() -> None:
    """llama.cpp availability without a cached model is honestly blocked.

    Spec: REQ-VERIFY-1158-2, REQ-VERIFY-1158-5
    """

    selection = select_beaver_lite_bounder(
        llama_cpp_is_available=True,
        model_path=None,
        top_k=10,
        max_tokens=8,
    )
    artifact = build_experiment_1158_artifact(selection, ())

    assert selection.bounder is None
    assert artifact["llama_cpp_available"] is True
    assert artifact["model_used"] is None
    assert artifact["blocked_reason"] == "cached_gguf_model_not_found"
    assert artifact["n_questions_evaluated"] == 0
    assert artifact["honest_verdict"] == "llm_not_available_blocked"
    assert artifact["beaver_lite_live_logprobs_sound_bound"] is False


def test_artifact_schema_and_bound_gap_match_reported_aggregate() -> None:
    """The Exp 1158 artifact exposes required fields and a coherent gap.

    Spec: REQ-VERIFY-1158-4, REQ-VERIFY-1158-5
    """

    selection = BounderSelection(
        bounder=None,
        llama_cpp_available=False,
        model_used=None,
        mock_logprobs_used=True,
        zipf_mock_used=True,
        install_command=INSTALL_LLAMA_CPP_COMMAND,
        blocked_reason=None,
    )
    evaluations = (
        QuestionEvaluation("q1", 0.35, 0.25, 0.10, True, 4),
        QuestionEvaluation("q2", 0.20, 0.20, 0.00, True, 4),
        QuestionEvaluation("q3", 0.31, 0.25, 0.06, True, 4),
    )

    artifact = build_experiment_1158_artifact(selection, evaluations)

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
    assert artifact["n_questions_evaluated"] == 3
    assert artifact["unsafe_mass_bound_live"] == pytest.approx(0.35)
    assert artifact["empirical_violation_rate_live"] == pytest.approx(0.25)
    assert artifact["bound_gap_live"] == pytest.approx(
        artifact["unsafe_mass_bound_live"] - artifact["empirical_violation_rate_live"]
    )
    assert artifact["unsafe_mass_bound_mock_prior"] == pytest.approx(0.400)
    assert artifact["bound_tighter_than_mock"] is True
    assert artifact["honest_verdict"] == "sound_bound_zipf_mock"
    assert len(artifact["question_evaluations"]) == 3


def test_live_logprob_path_evaluates_three_questions_and_writes_artifact(tmp_path: Path) -> None:
    """A live provider path records three real-logprob-mode question results.

    Spec: REQ-VERIFY-1158-1, REQ-VERIFY-1158-2, REQ-VERIFY-1158-4,
          SCENARIO-VERIFY-1158
    """

    output_path = tmp_path / "experiment_1158_beaver_lite_live_logprobs.json"
    artifact = run_beaver_lite_live_logprob_experiment(
        output_path=output_path,
        llama_cpp_available_override=True,
        model_path="/models/qwen3.5-0.8b.gguf",
        live_provider_factory=lambda model_path: _LiveFixtureProvider(),
        top_k=4,
        max_tokens=8,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == artifact
    assert artifact["llama_cpp_available"] is True
    assert artifact["model_used"] == "/models/qwen3.5-0.8b.gguf"
    assert artifact["mock_logprobs_used"] is False
    assert artifact["zipf_mock_used"] is False
    assert artifact["n_questions_evaluated"] == 3
    assert artifact["unsafe_mass_bound_live"] == pytest.approx(0.5)
    assert artifact["empirical_violation_rate_live"] == pytest.approx(0.25)
    assert artifact["bound_gap_live"] == pytest.approx(0.25)
    assert artifact["bound_is_sound_live"] is True
    assert artifact["honest_verdict"] == "sound_bound_live_logprobs"


def test_script_wrapper_runs_zipf_fallback_and_main_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The conductor-facing script writes the Exp 1158 JSON artifact.

    Spec: SCENARIO-VERIFY-1158
    """

    script = _load_script_module()
    output_path = tmp_path / "experiment_1158_beaver_lite_live_logprobs.json"

    artifact = script.run_experiment(
        output_path=output_path,
        llama_cpp_available_override=False,
        top_k=10,
        max_tokens=8,
    )
    assert json.loads(output_path.read_text()) == artifact
    assert artifact["zipf_mock_used"] is True

    monkeypatch.setattr(script, "run_experiment", lambda: artifact)
    monkeypatch.setattr(
        script,
        "OUTPUT_PATH",
        REPO_ROOT / "results" / "experiment_1158_test_tmp.json",
    )
    assert script.main() == 0
    captured = capsys.readouterr()
    assert "honest_verdict=sound_bound_zipf_mock" in captured.out
    assert "zipf_mock_used=True" in captured.out


def test_helpers_detect_llama_cpp_and_cached_gguf(tmp_path: Path) -> None:
    """Availability and cached-model helpers are deterministic and download-free.

    Spec: REQ-VERIFY-1158-2, REQ-VERIFY-1158-3
    """

    assert llama_cpp_available(lambda name: object()) is True

    def missing_importer(name: str) -> object:
        raise ImportError(name)

    assert llama_cpp_available(missing_importer) is False
    assert resolve_cached_gguf_model_path({}, tmp_path / "missing") is None
    assert resolve_cached_gguf_model_path({}, tmp_path) is None

    env_model = tmp_path / "explicit.gguf"
    env_model.write_text("fake", encoding="utf-8")
    assert resolve_cached_gguf_model_path(
        {"LLAMA_CPP_MODEL_PATH": str(env_model)}, tmp_path
    ) == str(env_model)

    cache_model = tmp_path / "hub" / "models--Qwen" / "snapshots" / "abc" / "model.gguf"
    cache_model.parent.mkdir(parents=True)
    cache_model.write_text("fake", encoding="utf-8")
    assert resolve_cached_gguf_model_path({}, tmp_path / "hub") == str(cache_model)


def test_bound_violation_sets_bug_verdict() -> None:
    """A negative bound gap is reported as a certificate bug.

    Spec: REQ-VERIFY-1158-5
    """

    selection = BounderSelection(
        bounder=None,
        llama_cpp_available=True,
        model_used="/models/qwen.gguf",
        mock_logprobs_used=False,
        zipf_mock_used=False,
        install_command=None,
        blocked_reason=None,
    )
    artifact = build_experiment_1158_artifact(
        selection,
        (QuestionEvaluation("q", 0.1, 0.4, -0.3, False, 4),),
    )

    assert artifact["honest_verdict"] == "bound_violated_bug"
    assert artifact["bound_is_sound_live"] is False
