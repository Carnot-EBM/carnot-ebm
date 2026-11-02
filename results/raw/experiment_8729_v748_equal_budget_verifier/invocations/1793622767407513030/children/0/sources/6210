"""Live reasoner self-verification harness for Exp 3894.

This module is deliberately small and direct.  The failed Exp 3827/3885 path
asked a model for YES/NO and then treated every response that did not contain
the substring ``"no"`` as "correct".  Empty output, prose, JSON, or any
unparsed format could therefore collapse all predictions to "correct" and
produce the 0-caught degeneracy.  Here the parser accepts the common output
families explicitly and records unparsed responses as neutral evidence instead
of silently accepting the step.

Spec refs: REQ-VERIFY-3894, SCENARIO-VERIFY-3894.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are auditing one reasoning step. Decide whether the step is correct "
    "as written. A step is incorrect if an arithmetic equality is false, a "
    "logical conclusion does not follow, or the stated result contradicts the "
    "given quantities. Return one compact JSON object only."
)


@dataclass(frozen=True)
class ParsedSelfVerification:
    """Parsed model judgment for one reasoning step.

    ``pred`` is ``1`` when the model says the step contains an error, ``0``
    when it says the step is correct, and ``None`` when the response was not
    parseable.  ``score`` is always an error score in ``[0, 1]`` so AUROC can be
    computed without inventing a positive or negative label for unparsed text.
    """

    pred: int | None
    score: float
    parsed: bool
    source: str


def build_positive_control_fixture() -> tuple[dict[str, object], ...]:
    """Return the fixed Exp 3894 positive-control steps.

    The fixture uses standalone arithmetic and symbolic-logic claims so a
    strong local reasoner does not need hidden problem context.  Exactly half
    the rows contain injected errors; the harness must catch at least one and
    should rank the erroneous rows above the correct rows.
    """

    return (
        {"id": "arith_correct_add", "step": "47 + 28 = 75.", "gold_error": 0},
        {"id": "arith_bad_add", "step": "47 + 28 = 65.", "gold_error": 1},
        {"id": "arith_correct_mul", "step": "7 * 8 = 56.", "gold_error": 0},
        {"id": "arith_bad_mul", "step": "7 * 8 = 65.", "gold_error": 1},
        {"id": "arith_correct_square", "step": "12^2 = 144.", "gold_error": 0},
        {"id": "arith_bad_square", "step": "12^2 = 124.", "gold_error": 1},
        {
            "id": "area_correct",
            "step": "A rectangle with width 5 and height 9 has area 45.",
            "gold_error": 0,
        },
        {
            "id": "area_bad",
            "step": "A rectangle with width 5 and height 9 has area 40.",
            "gold_error": 1,
        },
        {
            "id": "logic_correct",
            "step": (
                "If all blickets are daxes and all wugs are blickets, "
                "then all wugs are daxes."
            ),
            "gold_error": 0,
        },
        {
            "id": "logic_bad",
            "step": (
                "If all blickets are daxes and all wugs are daxes, "
                "then all wugs are blickets."
            ),
            "gold_error": 1,
        },
        {"id": "linear_correct", "step": "If x = 3, then 2x + 5 = 11.", "gold_error": 0},
        {"id": "linear_bad", "step": "If x = 3, then 2x + 5 = 12.", "gold_error": 1},
    )


def build_judge_prompt(step: str) -> str:
    """Build the robust step-judge prompt used by the live harness."""

    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        "Return one compact JSON object with exactly these keys:\n"
        "- verdict: choose either correct or incorrect\n"
        "- error_confidence: choose a number from 0.0 to 1.0\n\n"
        "Set verdict to incorrect for any false arithmetic or invalid logic. "
        "Set error_confidence near 1.0 when the error is clear and near 0.0 "
        "when the step is clearly correct.\n\n"
        "Step:\n"
        f"{step}\n\n"
        "JSON:"
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round_score(value: float) -> float:
    return round(_clamp01(value), 6)


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


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def _pred_from_value(key: str, value: Any) -> int | None:
    normalized_key = key.lower()
    bool_value = _coerce_bool(value)
    if bool_value is not None:
        if normalized_key in {"is_correct", "correct"}:
            return 0 if bool_value else 1
        if normalized_key in {"is_error", "contains_error", "has_error", "incorrect", "error"}:
            return 1 if bool_value else 0

    text = str(value).strip().lower()
    if re.search(r"\b(incorrect|wrong|invalid|false|error|mistake)\b", text):
        return 1
    if re.search(r"\b(correct|valid|true|right|yes)\b", text):
        return 0
    if re.search(r"\bno\b", text):
        return 1
    return None


def _score_from_json(payload: dict[str, Any], pred: int) -> float:
    for key in ("error_confidence", "error_score", "probability_error", "p_error"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            score = _round_score(float(value))
            if (pred == 1 and score < 0.5) or (pred == 0 and score > 0.5):
                return 0.8 if pred == 1 else 0.2
            return score
    for key in ("correct_confidence", "correct_score", "probability_correct", "p_correct"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return _round_score(1.0 - float(value))
    value = payload.get("confidence")
    if isinstance(value, (int, float)):
        return _round_score(float(value) if pred == 1 else 1.0 - float(value))
    value = payload.get("score")
    if isinstance(value, (int, float)):
        return _round_score(float(value))
    return 0.8 if pred == 1 else 0.2


def _parse_json_response(response: str) -> ParsedSelfVerification | None:
    for payload in _json_candidates(response):
        for key in (
            "verdict",
            "answer",
            "label",
            "decision",
            "classification",
            "is_error",
            "contains_error",
            "has_error",
            "incorrect",
            "is_correct",
            "correct",
        ):
            if key not in payload:
                continue
            pred = _pred_from_value(key, payload[key])
            if pred is None:
                continue
            return ParsedSelfVerification(
                pred=pred,
                score=_score_from_json(payload, pred),
                parsed=True,
                source="json",
            )
    return None


def _parse_text_response(response: str) -> ParsedSelfVerification | None:
    normalized = re.sub(r"\s+", " ", response.strip().lower())
    if not normalized:
        return None
    first_word = re.match(r"^[^a-z0-9]*(yes|no)\b", normalized)
    if first_word:
        pred = 0 if first_word.group(1) == "yes" else 1
        return ParsedSelfVerification(pred=pred, score=0.8 if pred else 0.2, parsed=True, source="yes_no")
    if re.search(r"\b(no error|no mistake|not erroneous)\b", normalized):
        return ParsedSelfVerification(pred=0, score=0.2, parsed=True, source="text")
    if re.search(r"\b(incorrect|wrong|invalid|false|erroneous|mistake)\b", normalized):
        return ParsedSelfVerification(pred=1, score=0.8, parsed=True, source="text")
    if re.search(r"\b(correct|valid|true|right)\b", normalized):
        return ParsedSelfVerification(pred=0, score=0.2, parsed=True, source="text")
    return None


def parse_self_verification_response(response: str) -> ParsedSelfVerification:
    """Parse a judge response without the Exp 3827 all-correct default.

    JSON is preferred because the live prompt requests it, but local GGUF
    models often prepend prose or answer with terse "YES"/"NO" text.  Those
    forms are accepted explicitly.  If none match, the output is marked
    unparsed with a neutral error score of 0.5 so downstream code can detect
    parser failure instead of laundering it into a negative prediction.
    """

    raw = str(response).strip()
    parsed = _parse_json_response(raw) or _parse_text_response(raw)
    if parsed is not None:
        return parsed
    return ParsedSelfVerification(pred=None, score=0.5, parsed=False, source="unparsed")


def _extract_llama_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"])
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
    return str(result)


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _prediction_constant(preds: Sequence[int | None]) -> bool:
    parsed_preds = [pred for pred in preds if pred is not None]
    return len(parsed_preds) == 0 or len(set(parsed_preds)) <= 1


def reasoner_self_verify(
    steps: Sequence[str],
    model_path: str | Path,
    *,
    gold_labels: Sequence[int] | None = None,
    llama_factory: Callable[..., Any] | None = None,
    max_tokens: int = 96,
    temperature: float = 0.0,
    n_gpu_layers: int = -1,
    n_ctx: int = 1024,
    n_batch: int = 64,
    offload_kqv: bool = True,
    random_seed: int = 3894,
    prompt_builder: Callable[[str], str] = build_judge_prompt,
) -> dict[str, object]:
    """Run live or injected reasoner self-verification over ``steps``.

    ``gold_labels`` uses ``1`` for known erroneous steps and ``0`` for correct
    steps.  When labels are supplied, the function computes AUROC and the
    number of gold errors caught by the reasoner's own self-verification.  The
    return value intentionally uses plain JSON-friendly fields because Exp 3894
    writes it into a results artifact.
    """

    step_texts = [str(step) for step in steps]
    if gold_labels is not None and len(gold_labels) != len(step_texts):
        raise ValueError("gold_labels must align with steps")
    if llama_factory is None:
        from llama_cpp import Llama  # pragma: no cover

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

    raw_responses: list[str] = []
    parsed_rows: list[ParsedSelfVerification] = []
    for step in step_texts:
        result = llm(
            prompt_builder(step),
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nStep:", "\n\nStep:"],
        )
        text = _extract_llama_text(result).strip()
        raw_responses.append(text)
        parsed_rows.append(parse_self_verification_response(text))

    raw_preds = [row.pred for row in parsed_rows]
    output_preds = [pred if pred is not None else -1 for pred in raw_preds]
    scores = [row.score for row in parsed_rows]
    labels = [int(label) for label in gold_labels] if gold_labels is not None else None
    auroc = _auroc(labels, scores) if labels is not None else None
    n_caught = (
        sum(1 for label, pred in zip(labels, raw_preds, strict=True) if label == 1 and pred == 1)
        if labels is not None
        else 0
    )

    return {
        "per_step_pred": output_preds,
        "per_step_score": scores,
        "raw_responses": raw_responses,
        "parse_sources": [row.source for row in parsed_rows],
        "parsed_count": sum(1 for row in parsed_rows if row.parsed),
        "unparsed_count": sum(1 for row in parsed_rows if not row.parsed),
        "parser_constant_prediction": _prediction_constant(raw_preds),
        "auroc": auroc,
        "n_caught": n_caught,
    }
