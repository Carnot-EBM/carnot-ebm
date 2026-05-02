"""BEAVER-lite probability-mass certificates for arithmetic answer constraints.

This is a deliberately small certificate-tier prototype.  The constraint is
prefix-closed over terminal prefixes: before a completion reaches EOS (or the
finite search frontier) it is not yet invalid, but once terminal text fails to
end with an integer in [0, 9999], no continuation can repair it.

Spec: REQ-VERIFY-1142, SCENARIO-VERIFY-1142
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


_DEFAULT_QUESTION = "If Alice has 3 apples and Bob gives her 4, how many does she have?"
_EOS_TOKEN = "<eos>"


def logsumexp(log_values: list[float] | tuple[float, ...]) -> float:
    """Return log(sum(exp(values))) without losing small probability masses.

    Spec: REQ-VERIFY-1142-3
    """

    if not log_values:
        return -math.inf
    max_value = max(log_values)
    if max_value == -math.inf:
        return -math.inf
    return max_value + math.log(sum(math.exp(value - max_value) for value in log_values))


class FinalIntegerConstraint:
    """Terminal text must end with an integer in the inclusive range [0, 9999].

    Prefix-closed means the verifier only declares a violation after the
    terminal boundary is known.  Non-terminal prefixes remain repairable because
    a later token can still add a valid final integer.

    Spec: REQ-VERIFY-1142-1
    """

    _final_integer_re = re.compile(r"(?<![\d.\-])(?P<answer>\d{1,4})\s*$")

    def is_satisfied(self, text: str) -> bool:
        """Return True iff terminal response text ends with an allowed integer."""

        match = self._final_integer_re.search(text.strip())
        if match is None:
            return False
        value = int(match.group("answer"))
        return 0 <= value <= 9999

    def prefix_violates(self, text: str, terminal: bool) -> bool:
        """Return whether this prefix is already an unrecoverable violation."""

        return terminal and not self.is_satisfied(text)


@dataclass(frozen=True)
class CompletionCandidate:
    """A candidate completion prefix plus its cumulative log probability."""

    text: str
    tokens: tuple[str, ...]
    logprob: float
    terminal: bool = True


@dataclass(frozen=True)
class ScoredCompletion:
    """A completion annotated with final-integer constraint verdicts."""

    text: str
    tokens: tuple[str, ...]
    logprob: float
    terminal: bool
    satisfies_constraint: bool
    violates_constraint: bool


@dataclass(frozen=True)
class BEAVERLiteResult:
    """Result returned by the BEAVER-lite arithmetic certificate tier."""

    question: str
    upper_bound: float
    empirical_rate: float
    bound_gap: float
    unsafe_logprob: float
    n_completions: int
    mock_logprobs_used: bool
    completions: tuple[ScoredCompletion, ...]

    @property
    def bound_is_sound(self) -> bool:
        """Return True when the reported mass bound is not below empirical rate."""

        return self.bound_gap >= -1e-12


class LogprobProvider(Protocol):
    """Provider interface for deterministic top-K completion enumeration."""

    mock_logprobs_used: bool

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        """Return up to top_k terminal candidates with cumulative logprobs."""


class MockLogprobProvider:
    """Deterministic equal-mass completion provider used when llama.cpp is absent.

    The default frontier has exactly `top_k` terminal completions.  Three of
    every five end in valid final integers; two of every five violate the
    constraint.  Equal logprob mass makes the probability bound directly
    comparable to the requested count-based empirical violation rate.

    Spec: REQ-VERIFY-1142-4
    """

    mock_logprobs_used = True

    def __init__(self, candidates: list[CompletionCandidate] | None = None) -> None:
        self._candidates = list(candidates) if candidates is not None else None

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        del prompt, max_tokens
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if self._candidates is not None:
            return list(self._candidates[:top_k])

        logprob = -math.log(top_k)
        invalid_texts = (
            "Final answer: seven",
            "Final answer: 7.",
            "Final answer: -1",
            "Final answer: 10000",
            "No numeric final answer",
        )
        candidates: list[CompletionCandidate] = []
        for idx in range(top_k):
            if idx % 5 in (0, 1, 2):
                value = (7 + idx) % 10000
                text = f"Final answer: {value}"
            else:
                text = invalid_texts[idx % len(invalid_texts)]
            candidates.append(
                CompletionCandidate(
                    text=text,
                    tokens=(text, _EOS_TOKEN),
                    logprob=logprob,
                    terminal=True,
                )
            )
        return candidates


@dataclass(frozen=True)
class _Beam:  # pragma: no cover
    token_ids: tuple[int, ...]
    text: str
    logprob: float
    terminal: bool = False


class LlamaCppLogprobProvider:  # pragma: no cover
    """llama.cpp logits-backed top-K prefix enumerator.

    This provider is intentionally optional.  It is used only when
    llama-cpp-python is importable and a GGUF path is supplied via the
    constructor, `CARNOT_BEAVER_LITE_GGUF`, or `LLAMA_CPP_MODEL_PATH`.

    Spec: REQ-VERIFY-1142-2, REQ-VERIFY-1142-3
    """

    mock_logprobs_used = False

    def __init__(self, model_path: str, branch_k: int = 50, n_ctx: int = 512) -> None:
        from llama_cpp import Llama  # type: ignore[import]

        self._llama = Llama(
            model_path=model_path,
            logits_all=True,
            n_ctx=n_ctx,
            verbose=False,
        )
        self._branch_k = branch_k

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        prompt_tokens = tuple(self._llama.tokenize(prompt.encode("utf-8"), add_bos=True))
        eos_id = self._eos_token_id()
        beams = [_Beam(token_ids=(), text="", logprob=0.0)]
        finished: list[_Beam] = []

        for _ in range(max_tokens):
            expanded: list[_Beam] = []
            for beam in beams:
                if beam.terminal:
                    finished.append(beam)
                    continue
                for token_id, token_logprob in self._next_token_logprobs(
                    prompt_tokens + beam.token_ids,
                    min(self._branch_k, top_k),
                ):
                    terminal = token_id == eos_id
                    token_text = "" if terminal else self._decode_token(token_id)
                    expanded.append(
                        _Beam(
                            token_ids=beam.token_ids + (token_id,),
                            text=beam.text + token_text,
                            logprob=beam.logprob + token_logprob,
                            terminal=terminal,
                        )
                    )

            if not expanded:
                break
            expanded.sort(key=lambda item: item.logprob, reverse=True)
            beams = expanded[:top_k]
            finished.extend(beam for beam in beams if beam.terminal)
            if len(finished) >= top_k:
                break

        if len(finished) < top_k:
            finished.extend(_Beam(beam.token_ids, beam.text, beam.logprob, True) for beam in beams)
        finished.sort(key=lambda item: item.logprob, reverse=True)
        return [
            CompletionCandidate(
                text=beam.text,
                tokens=tuple(str(token_id) for token_id in beam.token_ids) + (_EOS_TOKEN,),
                logprob=beam.logprob,
                terminal=True,
            )
            for beam in finished[:top_k]
        ]

    def _eos_token_id(self) -> int:
        token_eos = getattr(self._llama, "token_eos", None)
        if callable(token_eos):
            return int(token_eos())
        if token_eos is not None:
            return int(token_eos)
        return 2

    def _decode_token(self, token_id: int) -> str:
        return self._llama.detokenize([token_id]).decode("utf-8", errors="ignore")

    def _next_token_logprobs(
        self,
        token_ids: tuple[int, ...],
        branch_k: int,
    ) -> list[tuple[int, float]]:
        import numpy as np

        self._llama.reset()
        self._llama.eval(list(token_ids))
        logits = np.asarray(self._llama.scores[-1], dtype=np.float64)
        max_logit = float(np.max(logits))
        log_denominator = max_logit + math.log(float(np.exp(logits - max_logit).sum()))
        logprobs = logits - log_denominator
        top_ids = np.argsort(logprobs)[-branch_k:][::-1]
        return [(int(token_id), float(logprobs[token_id])) for token_id in top_ids]


class BEAVERLiteBounder:
    """Enumerate prefixes and bound unsafe probability mass for final answers.

    Spec: REQ-VERIFY-1142, SCENARIO-VERIFY-1142
    """

    def __init__(
        self,
        provider: LogprobProvider | None = None,
        constraint: FinalIntegerConstraint | None = None,
        top_k: int = 50,
        max_tokens: int = 16,
        model_path: str | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.constraint = constraint or FinalIntegerConstraint()
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.provider = provider or self._default_provider(model_path)

    @property
    def mock_logprobs_used(self) -> bool:
        """Return whether the active provider is the deterministic mock."""

        return bool(self.provider.mock_logprobs_used)

    def enumerate_completions(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[CompletionCandidate]:
        """Enumerate top-K candidate terminal prefixes.

        Spec: REQ-VERIFY-1142-2
        """

        k = self.top_k if top_k is None else top_k
        prompt = self._build_prompt(question)
        try:
            return self.provider.enumerate_completions(prompt, k, self.max_tokens)
        except Exception:
            if self.provider.mock_logprobs_used:
                raise
            self.provider = MockLogprobProvider()
            return self.provider.enumerate_completions(prompt, k, self.max_tokens)

    def score_completion(self, completion: CompletionCandidate) -> ScoredCompletion:
        """Score one terminal prefix against the final-integer constraint."""

        satisfies = completion.terminal and self.constraint.is_satisfied(completion.text)
        violates = self.constraint.prefix_violates(completion.text, completion.terminal)
        return ScoredCompletion(
            text=completion.text,
            tokens=completion.tokens,
            logprob=completion.logprob,
            terminal=completion.terminal,
            satisfies_constraint=satisfies,
            violates_constraint=violates,
        )

    def bound_prefix_violation(
        self,
        question: str = _DEFAULT_QUESTION,
        top_k: int | None = None,
    ) -> BEAVERLiteResult:
        """Return empirical and BEAVER-lite probability-mass violation bounds.

        Spec: REQ-VERIFY-1142-2, REQ-VERIFY-1142-3, SCENARIO-VERIFY-1142
        """

        completions = tuple(
            self.score_completion(c) for c in self.enumerate_completions(question, top_k)
        )
        n = len(completions)
        if n == 0:
            unsafe_logprob = -math.inf
            upper_bound = 0.0
            empirical_rate = 0.0
        else:
            unsafe_logprob = logsumexp(
                [completion.logprob for completion in completions if completion.violates_constraint]
            )
            upper_bound = 0.0 if unsafe_logprob == -math.inf else math.exp(unsafe_logprob)
            upper_bound = max(0.0, min(1.0, upper_bound))
            empirical_rate = sum(c.violates_constraint for c in completions) / n
        bound_gap = upper_bound - empirical_rate
        if abs(bound_gap) <= 1e-12:
            upper_bound = empirical_rate
            bound_gap = 0.0
        return BEAVERLiteResult(
            question=question,
            upper_bound=upper_bound,
            empirical_rate=empirical_rate,
            bound_gap=bound_gap,
            unsafe_logprob=unsafe_logprob,
            n_completions=n,
            mock_logprobs_used=self.mock_logprobs_used,
            completions=completions,
        )

    def _default_provider(self, model_path: str | None) -> LogprobProvider:
        path = (
            model_path or os.getenv("CARNOT_BEAVER_LITE_GGUF") or os.getenv("LLAMA_CPP_MODEL_PATH")
        )
        if path:
            try:
                return LlamaCppLogprobProvider(path)
            except Exception:
                return MockLogprobProvider()
        try:
            import llama_cpp  # noqa: F401
        except Exception:
            return MockLogprobProvider()
        return MockLogprobProvider()  # pragma: no cover - depends on optional llama_cpp import.

    def _build_prompt(self, question: str) -> str:
        return (
            "Answer the arithmetic question. End the response with the final integer.\n"
            f"Question: {question}\n"
            "Answer:"
        )


def build_experiment_artifact(result: BEAVERLiteResult) -> dict[str, object]:
    """Build the required Exp 1142 JSON artifact payload.

    Spec: REQ-VERIFY-1142-5
    """

    if result.n_completions == 0:
        honest_verdict = "llm_access_blocked"
    elif not result.bound_is_sound:
        honest_verdict = "bound_violated_bug"
    elif result.mock_logprobs_used:
        honest_verdict = "sound_bound_mock_logprobs"
    else:
        honest_verdict = "sound_bound_live_logprobs"

    return {
        "beaver_lite_bounder_written": True,
        "module_path": "python/carnot/verify/beaver_lite.py",
        "n_sample_questions": 1,
        "n_completions_sampled": result.n_completions,
        "mock_logprobs_used": result.mock_logprobs_used,
        "unsafe_mass_bound": result.upper_bound,
        "empirical_violation_rate": result.empirical_rate,
        "bound_gap": result.bound_gap,
        "bound_is_sound": result.bound_is_sound,
        "beaver_lite_bound_reported": True,
        "honest_verdict": honest_verdict,
        "question": result.question,
        "unsafe_logprob": result.unsafe_logprob,
    }


def write_experiment_artifact(artifact: dict[str, object], path: str | Path) -> None:
    """Write the Exp 1142 artifact as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "BEAVERLiteBounder",
    "BEAVERLiteResult",
    "CompletionCandidate",
    "FinalIntegerConstraint",
    "LogprobProvider",
    "MockLogprobProvider",
    "ScoredCompletion",
    "build_experiment_artifact",
    "logsumexp",
    "write_experiment_artifact",
]
