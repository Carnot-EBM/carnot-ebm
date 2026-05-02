"""Tests for the Exp 1142 BEAVER-lite arithmetic certificate tier.

Spec: REQ-VERIFY-1142, SCENARIO-VERIFY-1142
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

from carnot.verify.beaver_lite import (  # noqa: E402
    BEAVERLiteBounder,
    BEAVERLiteResult,
    CompletionCandidate,
    FinalIntegerConstraint,
    MockLogprobProvider,
    ScoredCompletion,
    build_experiment_artifact,
    logsumexp,
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "experiment_1142_beaver_lite_certificate_tier",
        REPO_ROOT / "scripts" / "experiment_1142_beaver_lite_certificate_tier.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_integer_constraint_is_terminal_prefix_closed() -> None:
    """The constraint only becomes violated after a terminal prefix is known.

    Spec: REQ-VERIFY-1142-1, SCENARIO-VERIFY-1142
    """

    constraint = FinalIntegerConstraint()

    assert constraint.is_satisfied("Alice has 7")
    assert constraint.is_satisfied("Final answer: 9999")
    assert constraint.is_satisfied("Answer: 0007")
    assert not constraint.is_satisfied("Final answer: 10000")
    assert not constraint.is_satisfied("Final answer: -1")
    assert not constraint.is_satisfied("Final answer: 7.")
    assert not constraint.is_satisfied("Final answer: seven")

    assert constraint.prefix_violates("Final answer: seven", terminal=False) is False
    assert constraint.prefix_violates("Final answer: seven", terminal=True) is True
    assert constraint.prefix_violates("Final answer: 7", terminal=True) is False


def test_logsumexp_computes_probability_mass_stably() -> None:
    """Unsafe prefix mass is aggregated in log space.

    Spec: REQ-VERIFY-1142-3
    """

    assert logsumexp([]) == -math.inf
    assert logsumexp([-math.inf]) == -math.inf
    assert math.exp(logsumexp([math.log(0.2), math.log(0.3)])) == pytest.approx(0.5)


def test_beaver_lite_bounder_reports_sound_mock_bound() -> None:
    """Mock top-K enumeration reports empirical rate and BEAVER-lite bound.

    Spec: REQ-VERIFY-1142-2, REQ-VERIFY-1142-3, REQ-VERIFY-1142-4,
          SCENARIO-VERIFY-1142
    """

    bounder = BEAVERLiteBounder(provider=MockLogprobProvider(), top_k=50)
    result = bounder.bound_prefix_violation(
        "If Alice has 3 apples and Bob gives her 4, how many does she have?"
    )

    assert result.n_completions == 50
    assert result.mock_logprobs_used is True
    assert 0.0 <= result.upper_bound <= 1.0
    assert 0.0 <= result.empirical_rate <= 1.0
    assert result.bound_gap == pytest.approx(result.upper_bound - result.empirical_rate)
    assert result.bound_gap >= -1e-12
    assert result.upper_bound == pytest.approx(result.empirical_rate)
    assert any(completion.violates_constraint for completion in result.completions)
    assert any(completion.satisfies_constraint for completion in result.completions)


def test_bounder_validates_inputs_and_defaults_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CI path is deterministic when llama.cpp logprobs are unavailable.

    Spec: REQ-VERIFY-1142-4
    """

    with pytest.raises(ValueError, match="top_k"):
        BEAVERLiteBounder(top_k=0)
    with pytest.raises(ValueError, match="max_tokens"):
        BEAVERLiteBounder(max_tokens=0)
    with pytest.raises(ValueError, match="top_k"):
        MockLogprobProvider().enumerate_completions("", 0, 1)

    monkeypatch.delenv("CARNOT_BEAVER_LITE_GGUF", raising=False)
    monkeypatch.delenv("LLAMA_CPP_MODEL_PATH", raising=False)
    assert BEAVERLiteBounder().mock_logprobs_used is True

    monkeypatch.setenv("CARNOT_BEAVER_LITE_GGUF", "/missing/model.gguf")
    assert BEAVERLiteBounder().mock_logprobs_used is True


def test_bounder_falls_back_when_live_provider_errors() -> None:
    """A broken live provider falls back to mock logprobs honestly.

    Spec: REQ-VERIFY-1142-4
    """

    class FailingLiveProvider:
        mock_logprobs_used = False

        def enumerate_completions(
            self,
            prompt: str,
            top_k: int,
            max_tokens: int,
        ) -> list[CompletionCandidate]:
            del prompt, top_k, max_tokens
            raise RuntimeError("llama.cpp unavailable")

    bounder = BEAVERLiteBounder(provider=FailingLiveProvider())
    result = bounder.bound_prefix_violation("q")

    assert result.mock_logprobs_used is True
    assert bounder.mock_logprobs_used is True
    assert result.n_completions == 50


def test_mock_provider_errors_are_not_hidden() -> None:
    """Mock-provider failures are surfaced instead of being double-mocked.

    Spec: REQ-VERIFY-1142-4
    """

    class FailingMockProvider:
        mock_logprobs_used = True

        def enumerate_completions(
            self,
            prompt: str,
            top_k: int,
            max_tokens: int,
        ) -> list[CompletionCandidate]:
            del prompt, top_k, max_tokens
            raise RuntimeError("bad mock")

    with pytest.raises(RuntimeError, match="bad mock"):
        BEAVERLiteBounder(provider=FailingMockProvider()).bound_prefix_violation("q")


def test_custom_candidate_bound_uses_only_violating_terminal_prefixes() -> None:
    """The BEAVER-lite mass excludes valid terminal prefixes.

    Spec: REQ-VERIFY-1142-3
    """

    candidates = [
        CompletionCandidate(
            text="Final answer: 7", tokens=("Final answer: 7", "<eos>"), logprob=math.log(1 / 3)
        ),
        CompletionCandidate(
            text="Final answer: seven",
            tokens=("Final answer: seven", "<eos>"),
            logprob=math.log(1 / 3),
        ),
        CompletionCandidate(
            text="Final answer: 7.", tokens=("Final answer: 7.", "<eos>"), logprob=math.log(1 / 3)
        ),
    ]
    provider = MockLogprobProvider(candidates=candidates)
    result = BEAVERLiteBounder(provider=provider, top_k=3).bound_prefix_violation("q")

    assert result.upper_bound == pytest.approx(2 / 3)
    assert result.empirical_rate == pytest.approx(2 / 3)
    assert result.bound_gap == pytest.approx(0.0)


def test_experiment_1142_artifact_schema_can_be_written(tmp_path: Path) -> None:
    """The Exp 1142 artifact contains the required certificate fields.

    Spec: REQ-VERIFY-1142-5, SCENARIO-VERIFY-1142
    """

    bounder = BEAVERLiteBounder(provider=MockLogprobProvider(), top_k=50)
    result = bounder.bound_prefix_violation(
        "If Alice has 3 apples and Bob gives her 4, how many does she have?"
    )
    artifact = build_experiment_artifact(result)

    required = {
        "beaver_lite_bounder_written",
        "module_path",
        "n_sample_questions",
        "n_completions_sampled",
        "mock_logprobs_used",
        "unsafe_mass_bound",
        "empirical_violation_rate",
        "bound_gap",
        "bound_is_sound",
        "beaver_lite_bound_reported",
        "honest_verdict",
    }
    assert required <= set(artifact)
    assert artifact["beaver_lite_bounder_written"] is True
    assert artifact["module_path"] == "python/carnot/verify/beaver_lite.py"
    assert artifact["n_sample_questions"] == 1
    assert artifact["n_completions_sampled"] == 50
    assert artifact["mock_logprobs_used"] is True
    assert artifact["bound_is_sound"] is True
    assert artifact["honest_verdict"] == "sound_bound_mock_logprobs"

    script = _load_script_module()
    output_path = tmp_path / "experiment_1142_beaver_lite_certificate_tier.json"
    written = script.run_experiment(output_path=output_path, provider=MockLogprobProvider())

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == written


def test_artifact_verdicts_cover_blocked_bug_and_live_modes() -> None:
    """Artifact verdicts distinguish blocked, violated, and live-logprob cases.

    Spec: REQ-VERIFY-1142-5
    """

    empty = BEAVERLiteBounder(provider=MockLogprobProvider(candidates=[]), top_k=1)
    assert (
        build_experiment_artifact(empty.bound_prefix_violation("q"))["honest_verdict"]
        == "llm_access_blocked"
    )

    violated = BEAVERLiteResult(
        question="q",
        upper_bound=0.1,
        empirical_rate=0.5,
        bound_gap=-0.4,
        unsafe_logprob=math.log(0.1),
        n_completions=2,
        mock_logprobs_used=True,
        completions=(),
    )
    assert build_experiment_artifact(violated)["honest_verdict"] == "bound_violated_bug"

    live = BEAVERLiteResult(
        question="q",
        upper_bound=0.5,
        empirical_rate=0.5,
        bound_gap=0.0,
        unsafe_logprob=math.log(0.5),
        n_completions=2,
        mock_logprobs_used=False,
        completions=(
            ScoredCompletion(
                text="Final answer: 7",
                tokens=("Final answer: 7", "<eos>"),
                logprob=math.log(0.5),
                terminal=True,
                satisfies_constraint=True,
                violates_constraint=False,
            ),
        ),
    )
    assert build_experiment_artifact(live)["honest_verdict"] == "sound_bound_live_logprobs"
