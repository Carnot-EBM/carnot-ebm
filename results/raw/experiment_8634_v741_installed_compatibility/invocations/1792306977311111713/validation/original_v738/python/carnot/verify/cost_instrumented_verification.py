"""Cost instrumentation for cheap verifiers versus a live LLM judge.

The Exp 3905/3906 question is not whether a verifier can produce a score.  The
question is whether the cheap verifier earns its place by being materially
cheaper than asking a local SOTA model to judge every step.  This module keeps
that accounting explicit: one monotonic wall-clock timer wraps each verifier,
AUROC is computed from the same labeled fixture, the LLM path counts real GGUF
tokenizer tokens, and the energy path records an estimated CPU forward-pass
operation count for the local verifier ensemble.

Spec refs: REQ-VERIFY-3905, SCENARIO-VERIFY-3905.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.verify.reasoner_self_verification import (
    build_judge_prompt,
    build_positive_control_fixture,
    parse_self_verification_response,
)


VerifierItems = Sequence[Mapping[str, object]]
VerifierResult = Mapping[str, object]
Clock = Callable[[], float]
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def build_cost_fixture() -> tuple[dict[str, object], ...]:
    """Return the fixed ten-row cost fixture used by Exp 3905.

    The fixture is a prefix of the tested Exp 3894 positive control, preserving
    the same standalone arithmetic and logic shape while keeping the live model
    run bounded.  Five rows are correct and five contain injected errors.
    """

    return tuple(dict(item) for item in build_positive_control_fixture()[:10])


def _step_text(item: Mapping[str, object]) -> str:
    if "step" in item:
        return str(item["step"])
    if "step_text" in item:
        return str(item["step_text"])
    raise ValueError("each cost fixture item must include step text")  # pragma: no cover


def _gold_error(item: Mapping[str, object]) -> int:
    for key in ("gold_error", "gold_label", "is_error", "label"):
        if key not in item:
            continue
        value = item[key]
        if value in {1, "1", True, "incorrect", "error", "bad"}:
            return 1
        if value in {0, "0", False, "correct", "ok", "good"}:
            return 0
    raise ValueError("each cost fixture item must include a gold error label")  # pragma: no cover


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")  # pragma: no cover

    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _scores_from_result(result: VerifierResult, n_items: int) -> list[float]:
    scores = [float(score) for score in result.get("scores", [])]
    if len(scores) != n_items:
        raise ValueError("verifier result must contain one score per item")  # pragma: no cover
    return scores


def measure_verification_cost(
    verifier_fn: Callable[[tuple[dict[str, object], ...]], VerifierResult],
    items: VerifierItems,
    label: str,
    *,
    clock: Clock = time.perf_counter,
) -> dict[str, object]:
    """Measure AUROC and cost fields for one verifier function.

    ``label`` is intentionally just a human-readable measurement name.  Gold
    labels come from the fixture rows so both verifier functions are evaluated
    against the exact same target labels.
    """

    item_rows = tuple(dict(item) for item in items)
    if not item_rows:
        raise ValueError(f"{label} measurement requires at least one item")  # pragma: no cover

    start_s = clock()
    result = verifier_fn(item_rows)
    finish_s = clock()
    total_wall_s = finish_s - start_s

    scores = _scores_from_result(result, len(item_rows))
    labels = [_gold_error(item) for item in item_rows]
    return {
        "auroc": _auroc(labels, scores),
        "total_wall_s": total_wall_s,
        "per_item_wall_ms": (total_wall_s * 1000.0) / len(item_rows),
        "est_tokens": int(result.get("est_tokens", 0)),
        "est_flops": int(result.get("est_flops", 0)),
        "n_items": len(item_rows),
    }


def _text_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def run_energy_verifier(items: VerifierItems) -> dict[str, object]:
    """Run the local CPU verifier ensemble and estimate its forward-pass ops."""

    from carnot.verify.tier0r_curry_howard import Tier0rVerifier
    from carnot.verify.tier0s_halluguard import Tier0sVerifier
    from carnot.verify.tier0u_logical_consistency import Tier0uVerifier

    tier0r = Tier0rVerifier()
    tier0s = Tier0sVerifier()
    tier0u = Tier0uVerifier()
    scores: list[float] = []
    scanned_tokens = 0
    forward_ops = 0
    for item in items:
        text = _step_text(item)
        token_count = _text_token_count(text)
        char_count = len(text)
        r_score = float(tier0r.score(text))
        u_score = float(tier0u.score(text))
        s_score = float(tier0s.halluguard_ntk_score(text)) / 100.0
        scores.append((0.8 * r_score) + (0.1 * u_score) + (0.1 * s_score))
        scanned_tokens += token_count
        forward_ops += (3 * char_count) + (24 * token_count)
    return {"scores": scores, "est_tokens": scanned_tokens, "est_flops": forward_ops}


def model_params_for_path(model_path: str | Path) -> int:
    """Infer the parameter-count term used for the LLM FLOP estimate."""

    lowered = str(model_path).lower()
    if "qwen3.6-35b" in lowered or "qwen3_6_35b" in lowered:
        return 35_000_000_000
    if "gemma-4-26b" in lowered or "gemma_4_26b" in lowered:
        return 26_000_000_000
    return 35_000_000_000


def _llama_token_count(llm: Any, text: str, *, add_bos: bool) -> int:
    payload = text.encode("utf-8")
    try:
        return len(llm.tokenize(payload, add_bos=add_bos))
    except TypeError:  # pragma: no cover - compatibility for older llama.cpp builds.
        return len(llm.tokenize(payload))


def _extract_llama_text(result: Any) -> str:
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
            if isinstance(first, dict) and isinstance(first.get("message"), dict):
                return str(first["message"].get("content", ""))
    return str(result)


def run_llm_judge_verifier(
    items: VerifierItems,
    *,
    model_path: str | Path,
    llama_factory: Callable[..., Any] | None = None,
    model_params: int | None = None,
    max_tokens: int = 96,
    temperature: float = 0.0,
    n_gpu_layers: int = 0,
    n_ctx: int = 1024,
    n_batch: int = 64,
    offload_kqv: bool = False,
    random_seed: int = 3905,
) -> dict[str, object]:
    """Run a live or injected llama.cpp judge and account for its tokens."""

    if llama_factory is None:
        from llama_cpp import Llama  # pragma: no cover - exercised by live fixture subprocess.

        llama_factory = Llama  # pragma: no cover

    llm = llama_factory(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_batch,
        offload_kqv=offload_kqv,
        seed=random_seed,
        verbose=False,
    )
    scores: list[float] = []
    total_tokens = 0
    for item in items:
        prompt = build_judge_prompt(_step_text(item))
        prompt_tokens = _llama_token_count(llm, prompt, add_bos=True)
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nStep:", "\n\nStep:"],
        )
        response = _extract_llama_text(result).strip()
        completion_tokens = _llama_token_count(llm, response, add_bos=False)
        parsed = parse_self_verification_response(response)
        scores.append(parsed.score)
        total_tokens += prompt_tokens + completion_tokens

    params = model_params if model_params is not None else model_params_for_path(model_path)
    return {
        "scores": scores,
        "est_tokens": total_tokens,
        "est_flops": 2 * int(params) * total_tokens,
    }
