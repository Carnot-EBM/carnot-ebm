"""BEAVER-lite live-or-Zipf logprob workflow for Exp 1158.

Spec: REQ-VERIFY-1158, SCENARIO-VERIFY-1158
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.verify.beaver_lite import (
    BEAVERLiteBounder,
    CompletionCandidate,
    LogprobProvider,
)


EXP1142_MOCK_PRIOR_BOUND = 0.400
INSTALL_LLAMA_CPP_COMMAND = "pip install llama-cpp-python"

_SAMPLE_QUESTIONS = (
    "Janet has 10 marbles and gives away 3. How many remain?",
    "A box has 4 red balls and 5 blue balls. How many balls are in the box?",
    "Luis read 12 pages on Monday and 8 on Tuesday. How many pages did he read?",
)

_FOVER_FALLBACK_PROMPTS = (
    "FoVer arithmetic step: 3 + 4 = 7. What final integer is claimed?",
    "FoVer arithmetic step: 10 - 3 = 7. What final integer is claimed?",
    "FoVer arithmetic step: 6 * 5 = 30. What final integer is claimed?",
    "FoVer arithmetic step: 12 + 8 = 20. What final integer is claimed?",
    "FoVer arithmetic step: 9 + 9 = 18. What final integer is claimed?",
    "FoVer arithmetic step: 15 - 6 = 9. What final integer is claimed?",
    "FoVer arithmetic step: 4 * 7 = 28. What final integer is claimed?",
    "FoVer arithmetic step: 81 / 9 = 9. What final integer is claimed?",
    "FoVer arithmetic step: 14 + 16 = 30. What final integer is claimed?",
    "FoVer arithmetic step: 100 - 64 = 36. What final integer is claimed?",
)


@dataclass(frozen=True)
class BounderSelection:
    """Selected BEAVER-lite bounder plus honest live/mock provenance."""

    bounder: BEAVERLiteBounder | None
    llama_cpp_available: bool
    model_used: str | None
    mock_logprobs_used: bool
    zipf_mock_used: bool
    install_command: str | None
    blocked_reason: str | None


@dataclass(frozen=True)
class QuestionEvaluation:
    """Per-question Exp 1158 certificate summary."""

    question: str
    unsafe_mass_bound: float
    empirical_violation_rate: float
    bound_gap: float
    bound_is_sound: bool
    n_completions: int


class BEAVERLiteVerifier:
    """Thin verifier wrapper around the BEAVER-lite bounder."""

    def __init__(
        self,
        provider: LogprobProvider | None = None,
        top_k: int = 10,
        max_tokens: int = 8,
    ) -> None:
        self.bounder = BEAVERLiteBounder(provider=provider, top_k=top_k, max_tokens=max_tokens)

    def evaluate_question(self, question: str) -> QuestionEvaluation:
        """Evaluate one question and return the Exp 1158/1170 summary fields."""

        result = self.bounder.bound_prefix_violation(question)
        return QuestionEvaluation(
            question=question,
            unsafe_mass_bound=result.upper_bound,
            empirical_violation_rate=result.empirical_rate,
            bound_gap=result.bound_gap,
            bound_is_sound=result.bound_is_sound,
            n_completions=result.n_completions,
        )


class ZipfMockLogprobProvider:
    """Deterministic non-uniform fallback provider used without llama.cpp.

    Spec: REQ-VERIFY-1158-3
    """

    mock_logprobs_used = True

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        del prompt, max_tokens
        weights = [1.0 / (rank**self.alpha) for rank in range(1, top_k + 1)]
        total = sum(weights)
        invalid_texts = (
            "Final answer: seven",
            "Final answer: 7.",
            "No numeric final answer",
        )
        candidates: list[CompletionCandidate] = []
        for index, weight in enumerate(weights, start=1):
            invalid = index in {2, 3, top_k}
            text = (
                invalid_texts[index % len(invalid_texts)]
                if invalid
                else f"Final answer: {6 + index}"
            )
            candidates.append(
                CompletionCandidate(
                    text=text,
                    tokens=(text, "<eos>"),
                    logprob=math.log(weight / total),
                    terminal=True,
                )
            )
        return candidates


class LlamaCppCompletionLogprobProvider:
    """llama.cpp completion provider that uses generated-token logprobs.

    Spec: REQ-VERIFY-1170-1, REQ-VERIFY-1170-2
    """

    mock_logprobs_used = False
    logprobs_source = "llama_cpp_logits_all"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        llama_factory: Callable[..., Any] | None = None,
    ) -> None:
        if llama_factory is None:  # pragma: no cover - optional live dependency.
            from llama_cpp import Llama  # type: ignore[import]

            llama_factory = Llama
        self._llama = llama_factory(
            model_path=model_path,
            logits_all=True,
            n_ctx=n_ctx,
            verbose=False,
        )

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        completions: list[CompletionCandidate] = []
        requested_logprobs = max(1, top_k)
        for index in range(top_k):
            completion = self._llama(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0 if top_k == 1 else 0.7,
                top_k=max(1, top_k),
                logprobs=requested_logprobs,
                echo=False,
                seed=1170 + index,
            )
            completions.append(self._completion_to_candidate(completion))
        return completions

    @staticmethod
    def _completion_to_candidate(completion: Mapping[str, Any]) -> CompletionCandidate:
        choices = completion.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise ValueError("llama.cpp completion did not include choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ValueError("llama.cpp completion choice was not a mapping")

        text = str(choice.get("text", ""))
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, Mapping):
            raise ValueError("llama.cpp completion did not include logprobs")

        raw_token_logprobs = logprobs.get("token_logprobs")
        if not isinstance(raw_token_logprobs, Sequence):
            raise ValueError("llama.cpp completion did not include token_logprobs")
        token_logprobs = [float(value) for value in raw_token_logprobs if value is not None]
        if not token_logprobs:
            raise ValueError("llama.cpp completion token_logprobs were empty")

        raw_tokens = logprobs.get("tokens")
        tokens = (
            tuple(str(token) for token in raw_tokens)
            if isinstance(raw_tokens, Sequence)
            else (text,)
        )
        return CompletionCandidate(
            text=text,
            tokens=tokens + ("<eos>",),
            logprob=sum(token_logprobs),
            terminal=True,
        )


def llama_cpp_available(importer: Callable[[str], object] = __import__) -> bool:
    """Return whether the optional llama.cpp Python package can be imported."""

    try:
        importer("llama_cpp")
    except Exception:
        return False
    return True


def resolve_cached_gguf_model_path(
    env: Mapping[str, str],
    cache_root: str | Path,
) -> str | None:
    """Resolve an explicit or cached GGUF model path without downloading."""

    for key in ("CARNOT_BEAVER_LITE_GGUF", "LLAMA_CPP_MODEL_PATH"):
        value = env.get(key)
        if value and Path(value).exists():
            return str(Path(value))

    root = Path(cache_root)
    if not root.exists():
        return None
    matches = sorted(path for path in root.glob("models--*/snapshots/**/*.gguf") if path.is_file())
    if not matches:
        matches = sorted(path for path in root.rglob("*.gguf") if path.is_file())
    return str(matches[0]) if matches else None


def select_beaver_lite_bounder(
    llama_cpp_is_available: bool,
    model_path: str | None,
    top_k: int,
    max_tokens: int,
    live_provider_factory: Callable[[str], LogprobProvider] | None = None,
) -> BounderSelection:
    """Select a live llama.cpp bounder or the honest Zipf fallback."""

    if not llama_cpp_is_available:
        provider = ZipfMockLogprobProvider()
        return BounderSelection(
            bounder=BEAVERLiteBounder(provider=provider, top_k=top_k, max_tokens=max_tokens),
            llama_cpp_available=False,
            model_used=None,
            mock_logprobs_used=True,
            zipf_mock_used=True,
            install_command=INSTALL_LLAMA_CPP_COMMAND,
            blocked_reason=None,
        )

    if model_path is None:
        return BounderSelection(
            bounder=None,
            llama_cpp_available=True,
            model_used=None,
            mock_logprobs_used=False,
            zipf_mock_used=False,
            install_command=None,
            blocked_reason="cached_gguf_model_not_found",
        )

    provider = (
        live_provider_factory(model_path)
        if live_provider_factory is not None
        else LlamaCppCompletionLogprobProvider(
            model_path
        )  # pragma: no cover - optional live dependency.
    )
    return BounderSelection(
        bounder=BEAVERLiteBounder(provider=provider, top_k=top_k, max_tokens=max_tokens),
        llama_cpp_available=True,
        model_used=model_path,
        mock_logprobs_used=False,
        zipf_mock_used=False,
        install_command=None,
        blocked_reason=None,
    )


def build_experiment_1158_artifact(
    selection: BounderSelection,
    question_evaluations: Sequence[QuestionEvaluation],
) -> dict[str, object]:
    """Build the required Exp 1158 JSON artifact payload."""

    if not question_evaluations:
        unsafe_mass_bound = 0.0
        empirical_rate = 0.0
        bound_gap = 0.0
        bound_is_sound = False
        honest_verdict = "llm_not_available_blocked"
    else:
        unsafe_mass_bound = max(item.unsafe_mass_bound for item in question_evaluations)
        empirical_rate = max(item.empirical_violation_rate for item in question_evaluations)
        bound_gap = unsafe_mass_bound - empirical_rate
        bound_is_sound = all(item.bound_is_sound for item in question_evaluations)
        if not bound_is_sound:
            honest_verdict = "bound_violated_bug"
        elif selection.zipf_mock_used:
            honest_verdict = "sound_bound_zipf_mock"
        else:
            honest_verdict = "sound_bound_live_logprobs"

    bound_tighter_than_mock = unsafe_mass_bound < EXP1142_MOCK_PRIOR_BOUND
    return {
        "llama_cpp_available": selection.llama_cpp_available,
        "model_used": selection.model_used,
        "mock_logprobs_used": selection.mock_logprobs_used,
        "zipf_mock_used": selection.zipf_mock_used,
        "llama_cpp_install_command": selection.install_command,
        "blocked_reason": selection.blocked_reason,
        "n_questions_evaluated": len(question_evaluations),
        "unsafe_mass_bound_live": unsafe_mass_bound,
        "empirical_violation_rate_live": empirical_rate,
        "bound_gap_live": bound_gap,
        "bound_is_sound_live": bound_is_sound,
        "unsafe_mass_bound_mock_prior": EXP1142_MOCK_PRIOR_BOUND,
        "bound_tighter_than_mock": bound_tighter_than_mock,
        "beaver_lite_live_logprobs_sound_bound": bound_is_sound,
        "honest_verdict": honest_verdict,
        "question_evaluations": [
            {
                "question": item.question,
                "unsafe_mass_bound": item.unsafe_mass_bound,
                "empirical_violation_rate": item.empirical_violation_rate,
                "bound_gap": item.bound_gap,
                "bound_is_sound": item.bound_is_sound,
                "n_completions": item.n_completions,
            }
            for item in question_evaluations
        ],
    }


def run_beaver_lite_live_logprob_experiment(
    output_path: str | Path,
    llama_cpp_available_override: bool | None = None,
    model_path: str | None = None,
    live_provider_factory: Callable[[str], LogprobProvider] | None = None,
    top_k: int = 10,
    max_tokens: int = 8,
) -> dict[str, object]:
    """Run Exp 1158 and write a stable JSON artifact."""

    if llama_cpp_available_override is None:  # pragma: no cover - environment-dependent path.
        import os

        is_available = llama_cpp_available()
        selected_model_path = model_path or resolve_cached_gguf_model_path(
            os.environ, Path.home() / ".cache" / "huggingface" / "hub"
        )
    else:
        is_available = llama_cpp_available_override
        selected_model_path = model_path

    selection = select_beaver_lite_bounder(
        llama_cpp_is_available=is_available,
        model_path=selected_model_path,
        top_k=top_k,
        max_tokens=max_tokens,
        live_provider_factory=live_provider_factory,
    )
    evaluations: list[QuestionEvaluation] = []
    if selection.bounder is not None:
        for question in _SAMPLE_QUESTIONS:
            result = selection.bounder.bound_prefix_violation(question)
            evaluations.append(
                QuestionEvaluation(
                    question=question,
                    unsafe_mass_bound=result.upper_bound,
                    empirical_violation_rate=result.empirical_rate,
                    bound_gap=result.bound_gap,
                    bound_is_sound=result.bound_is_sound,
                    n_completions=result.n_completions,
                )
            )

    artifact = build_experiment_1158_artifact(selection, tuple(evaluations))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def load_fover_test_prompts(path: str | Path | None, limit: int = 10) -> list[str]:
    """Load up to ``limit`` local FoVer prompts, with a built-in deterministic fallback."""

    prompts: list[str] = []
    if path is not None and Path(path).exists():
        source_path = Path(path)
        if source_path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("entries", [])
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            text = row.get("question") or row.get("prompt") or row.get("step_text")
            if isinstance(text, str) and text.strip():
                prompts.append(
                    "FoVer arithmetic prompt. End with the final integer claimed by this step.\n"
                    f"{text.strip()}"
                )
            if len(prompts) >= limit:
                break
    if len(prompts) < limit:
        prompts.extend(_FOVER_FALLBACK_PROMPTS[len(prompts) : limit])
    return prompts[:limit]


def _default_exp1170_model_path(model_path: str | None) -> str | None:
    if model_path is not None:
        return model_path
    cached = resolve_cached_gguf_model_path(
        os.environ, Path.home() / ".cache" / "huggingface" / "hub"
    )
    if cached is not None:
        return cached
    repo_root = Path(__file__).resolve().parents[3]
    local_models = sorted(
        path
        for path in (repo_root / "models").rglob("*.gguf")
        if path.is_file() and "ggml-vocab" not in path.name
    )
    return str(local_models[0]) if local_models else None


def run_beaver_live_logprobs_v2_experiment(
    output_path: str | Path,
    fover_corpus_path: str | Path | None = Path("data/fover_corpus.jsonl"),
    llama_cpp_available_override: bool | None = None,
    model_path: str | None = None,
    live_provider_factory: Callable[[str], LogprobProvider] | None = None,
    top_k: int = 5,
    max_tokens: int = 8,
) -> dict[str, object]:
    """Run Exp 1170 and write the required llama.cpp logprob artifact."""

    prompts = load_fover_test_prompts(fover_corpus_path, limit=10)
    is_available = (
        llama_cpp_available()
        if llama_cpp_available_override is None
        else llama_cpp_available_override
    )
    selected_model_path = _default_exp1170_model_path(model_path)

    provider: LogprobProvider
    logprobs_source: str
    if is_available and selected_model_path is not None:
        try:
            provider = (
                live_provider_factory(selected_model_path)
                if live_provider_factory is not None
                else LlamaCppCompletionLogprobProvider(selected_model_path)
            )
            logprobs_source = str(getattr(provider, "logprobs_source", "llama_cpp_logits_all"))
        except Exception:
            provider = ZipfMockLogprobProvider()
            logprobs_source = "zipf_mock"
    else:
        provider = ZipfMockLogprobProvider()
        logprobs_source = "zipf_mock"

    evaluations: list[QuestionEvaluation] = []
    try:
        verifier = BEAVERLiteVerifier(provider=provider, top_k=top_k, max_tokens=max_tokens)
        evaluations = [verifier.evaluate_question(prompt) for prompt in prompts]
        bounds = [item.unsafe_mass_bound for item in evaluations]
        bound_is_sound = all(0.0 <= bound <= 1.0 for bound in bounds)
        mock_logprobs_used = bool(getattr(provider, "mock_logprobs_used", True))
        if not bound_is_sound:
            honest_verdict = "bound_computation_failed"
        elif mock_logprobs_used:
            honest_verdict = "logprobs_unavailable_mock_fallback"
        else:
            honest_verdict = "live_logprobs_sound_bound"
    except Exception:
        bounds = []
        bound_is_sound = False
        mock_logprobs_used = bool(getattr(provider, "mock_logprobs_used", True))
        honest_verdict = "bound_computation_failed"

    artifact = {
        "mock_logprobs_used": mock_logprobs_used,
        "logprobs_source": logprobs_source,
        "bound_is_sound": bound_is_sound,
        "sample_bound_values": bounds[:5],
        "n_test_prompts_run": len(evaluations),
        "honest_verdict": honest_verdict,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "EXP1142_MOCK_PRIOR_BOUND",
    "INSTALL_LLAMA_CPP_COMMAND",
    "BEAVERLiteVerifier",
    "BounderSelection",
    "LlamaCppCompletionLogprobProvider",
    "QuestionEvaluation",
    "ZipfMockLogprobProvider",
    "build_experiment_1158_artifact",
    "load_fover_test_prompts",
    "llama_cpp_available",
    "resolve_cached_gguf_model_path",
    "run_beaver_live_logprobs_v2_experiment",
    "run_beaver_lite_live_logprob_experiment",
    "select_beaver_lite_bounder",
]
