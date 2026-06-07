"""Competent process-style LLM judge for Exp 3925.

The 3917 efficiency comparison used the older self-verification prompt and
reported an LLM-judge AUROC below chance. This module keeps the useful part of
that harness, namely the shared GGUF generator interface, but changes the judge
contract to a process-verifier recipe: reason about this one step, then emit a
single explicit verdict line.

The returned score is intentionally named `verdict_prob` to match downstream
tasks, but it is always the probability that the step is incorrect. Correct
steps should therefore score near zero and incorrect steps near one.

Spec refs: REQ-VERIFY-3925, SCENARIO-VERIFY-3925.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from carnot.verify.gguf_inference import generate


COMPETENT_PREFER_ORDER: tuple[str, ...] = (
    "Qwen3.6-35B-A3B",
    "gemma-4-31B-it",
    "gemma-4-26B-A4B-it",
)
DEFAULT_MAX_TOKENS = 128
DEFAULT_N_CTX = 1024
DEFAULT_MAX_N_GPU_LAYERS = -1
ABSTAIN_PROB = 0.5


@dataclass(frozen=True)
class ParsedJudgeResponse:
    """Parsed final verdict from the model output.

    `verdict_prob` is an error probability. Unparsed responses are visible as
    `parsed=False` and receive the neutral abstention score 0.5, so parser
    failures cannot silently become all-correct or all-incorrect predictions.
    """

    verdict: str | None
    verdict_prob: float
    parsed: bool
    source: str


def build_separable_fixture() -> tuple[dict[str, object], ...]:
    """Return the fixed standalone positive-control fixture for Exp 3925."""

    return (
        {"id": "arith_add_correct", "step": "47 + 28 = 75.", "gold_error": 0},
        {"id": "arith_add_wrong", "step": "47 + 28 = 65.", "gold_error": 1},
        {"id": "arith_mul_correct", "step": "7 * 8 = 56.", "gold_error": 0},
        {"id": "arith_mul_wrong", "step": "7 * 8 = 65.", "gold_error": 1},
        {"id": "square_correct", "step": "12^2 = 144.", "gold_error": 0},
        {"id": "square_wrong", "step": "12^2 = 124.", "gold_error": 1},
        {
            "id": "rectangle_correct",
            "step": "A rectangle with width 5 and height 9 has area 45.",
            "gold_error": 0,
        },
        {
            "id": "rectangle_wrong",
            "step": "A rectangle with width 5 and height 9 has area 40.",
            "gold_error": 1,
        },
        {"id": "linear_correct", "step": "If x = 3, then 2x + 5 = 11.", "gold_error": 0},
        {"id": "linear_wrong", "step": "If x = 3, then 2x + 5 = 12.", "gold_error": 1},
        {
            "id": "logic_correct",
            "step": (
                "If all blickets are daxes and all wugs are blickets, "
                "then all wugs are daxes."
            ),
            "gold_error": 0,
        },
        {
            "id": "logic_wrong",
            "step": (
                "If all blickets are daxes and all wugs are daxes, "
                "then all daxes are wugs."
            ),
            "gold_error": 1,
        },
    )


def _step_text(item: dict[str, object] | str) -> str:
    if isinstance(item, str):
        return item
    for key in ("step", "step_text", "text", "candidate_text", "reasoning_step"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("judge item has no step text")


def _label_to_error(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "incorrect", "error", "wrong", "bad", "true"}:
        return 1
    if text in {"0", "correct", "valid", "right", "good", "false"}:
        return 0
    raise ValueError(f"unsupported gold_error label: {value!r}")


def build_judge_prompt(item: dict[str, object] | str) -> str:
    """Build a structured CoT-then-verdict prompt for exactly one step."""

    step = _step_text(item)
    return (
        "<|im_start|>system\n"
        "You are a careful process verifier. Examine only the step under "
        "review. A step is INCORRECT if an arithmetic equality is false, a "
        "calculation result is wrong, or a logical conclusion does not follow. "
        "Answer with a short reason and a final verdict.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Use exactly this format:\n"
        "REASON: <one short sentence checking the step>\n"
        "VERDICT: <CORRECT or INCORRECT>\n\n"
        "Example 1\n"
        "Step: 47 + 28 = 75.\n"
        "REASON: 47 + 28 equals 75.\n"
        "VERDICT: CORRECT\n\n"
        "Example 2\n"
        "Step: 47 + 28 = 65.\n"
        "REASON: 47 + 28 equals 75, not 65.\n"
        "VERDICT: INCORRECT\n\n"
        "Step under review:\n"
        f"{step}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "REASON:"
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_prob(value: float) -> float:
    return round(_clamp01(float(value)), 6)


def _json_candidates(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates


def _verdict_from_text(value: object) -> str | None:
    text = str(value).strip().lower()
    if re.search(r"\b(incorrect|wrong|invalid|false|error|mistake)\b", text):
        return "incorrect"
    if re.search(r"\b(correct|valid|true|right)\b", text):
        return "correct"
    return None


def _prob_from_json(payload: dict[str, Any], verdict: str) -> float:
    for key in ("p_incorrect", "probability_incorrect", "error_probability", "error_confidence"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return _round_prob(float(value))
    for key in ("p_correct", "probability_correct", "correct_probability", "correct_confidence"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return _round_prob(1.0 - float(value))
    value = payload.get("confidence")
    if isinstance(value, (int, float)):
        return _round_prob(float(value) if verdict == "incorrect" else 1.0 - float(value))
    return 0.9 if verdict == "incorrect" else 0.1


def _parse_json_response(response: str) -> ParsedJudgeResponse | None:
    for payload in _json_candidates(response):
        for key in ("verdict", "answer", "decision", "classification", "label"):
            if key not in payload:
                continue
            verdict = _verdict_from_text(payload[key])
            if verdict is None:
                continue
            return ParsedJudgeResponse(
                verdict=verdict,
                verdict_prob=_prob_from_json(payload, verdict),
                parsed=True,
                source="json",
            )
        for key in ("p_incorrect", "probability_incorrect", "error_probability", "error_confidence"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                probability = _round_prob(float(value))
                verdict = "incorrect" if probability > 0.5 else "correct"
                return ParsedJudgeResponse(
                    verdict=verdict,
                    verdict_prob=probability,
                    parsed=True,
                    source="json_probability",
                )
    return None


def _parse_text_response(response: str) -> ParsedJudgeResponse | None:
    normalized = re.sub(r"\s+", " ", response.strip())
    if not normalized:
        return None

    verdict_matches = re.findall(
        r"\bVERDICT\s*[:=\-]\s*(?:the\s+step\s+is\s+)?(INCORRECT|CORRECT)\b",
        response,
        flags=re.IGNORECASE,
    )
    if verdict_matches:
        verdict = verdict_matches[-1].lower()
        return ParsedJudgeResponse(
            verdict=verdict,
            verdict_prob=0.9 if verdict == "incorrect" else 0.1,
            parsed=True,
            source="verdict_line",
        )

    label_lines = re.findall(r"^\s*(INCORRECT|CORRECT)\s*\.?\s*$", response, flags=re.IGNORECASE | re.MULTILINE)
    if label_lines:
        verdict = label_lines[-1].lower()
        return ParsedJudgeResponse(
            verdict=verdict,
            verdict_prob=0.9 if verdict == "incorrect" else 0.1,
            parsed=True,
            source="label_line",
        )

    tail = normalized[-240:].lower()
    verdict = _verdict_from_text(tail)
    if verdict is None:
        return None
    return ParsedJudgeResponse(
        verdict=verdict,
        verdict_prob=0.9 if verdict == "incorrect" else 0.1,
        parsed=True,
        source="text_tail",
    )


def parse_judge_response(response: str) -> ParsedJudgeResponse:
    """Parse a judge response with neutral abstention for unparseable text."""

    raw = str(response).strip()
    parsed = _parse_json_response(raw) or _parse_text_response(raw)
    if parsed is not None:
        return parsed
    return ParsedJudgeResponse(
        verdict=None,
        verdict_prob=ABSTAIN_PROB,
        parsed=False,
        source="unparsed",
    )


def judge_step(
    item: dict[str, object] | str,
    generator: object,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    """Judge one reasoning step through the supplied robust GGUF generator."""

    prompt = build_judge_prompt(item)
    raw_text = generate(generator, prompt, max_tokens=max_tokens).strip()
    parsed = parse_judge_response(raw_text)
    return {
        "verdict_prob": parsed.verdict_prob,
        "raw_text": raw_text,
        "parsed": parsed.parsed,
        "verdict": parsed.verdict,
        "parse_source": parsed.source,
        "step_text": _step_text(item),
    }


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _prediction_constant(verdicts: Sequence[str | None]) -> bool:
    parsed = [verdict for verdict in verdicts if verdict is not None]
    return len(parsed) == 0 or len(set(parsed)) <= 1


def run_judge_fixture(
    fixture: Sequence[dict[str, object]],
    generator: object,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    """Run the competent judge over a labeled fixture and compute AUROC."""

    labels = [_label_to_error(item.get("gold_error", item.get("label"))) for item in fixture]
    rows = [judge_step(item, generator, max_tokens=max_tokens) for item in fixture]
    scores = [float(row["verdict_prob"]) for row in rows]
    parsed_flags = [bool(row["parsed"]) for row in rows]
    verdicts = [row["verdict"] if isinstance(row["verdict"], str) else None for row in rows]
    return {
        "fixture_auroc": _auroc(labels, scores),
        "verdicts_parse_rate": sum(parsed_flags) / len(parsed_flags) if parsed_flags else 0.0,
        "parser_constant_prediction": _prediction_constant(verdicts),
        "scores": scores,
        "labels": labels,
        "raw_texts": [str(row["raw_text"]) for row in rows],
        "parsed": parsed_flags,
        "verdicts": verdicts,
        "parse_sources": [str(row["parse_source"]) for row in rows],
    }


__all__ = [
    "ABSTAIN_PROB",
    "COMPETENT_PREFER_ORDER",
    "DEFAULT_MAX_N_GPU_LAYERS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_N_CTX",
    "ParsedJudgeResponse",
    "build_judge_prompt",
    "build_separable_fixture",
    "judge_step",
    "parse_judge_response",
    "run_judge_fixture",
]
