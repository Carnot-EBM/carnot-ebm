"""Helpers for live extraction autopsy experiments.

Spec: REQ-VERIFY-008, SCENARIO-VERIFY-008
"""

from __future__ import annotations

import re
from typing import Any

from carnot.pipeline.extract import ArithmeticExtractor

_ARITHMETIC_PATTERN = re.compile(r"(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)")
_APPROXIMATION_TERMS = (
    "about",
    "approx",
    "approximately",
    "around",
    "roughly",
    "nearly",
)


def extract_final_number(text: str) -> int | None:
    """Extract a final numeric answer from a model response."""
    numeric_token = r"-?(?:\d[\d,]*)"
    gsm8k_match = re.search(rf"####\s*({numeric_token})", text)
    if gsm8k_match:
        return int(gsm8k_match.group(1).replace(",", ""))

    answer_match = re.search(rf"[Aa]nswer[:\s]+({numeric_token})", text)
    if answer_match:
        return int(answer_match.group(1).replace(",", ""))

    numbers = re.findall(numeric_token, text)
    if numbers:
        return int(numbers[-1].replace(",", ""))

    return None


def capture_arithmetic_matches(response: str) -> list[dict[str, Any]]:
    """Capture the exact regex spans and verification metadata for a response."""
    extractor = ArithmeticExtractor()
    results = extractor.extract(response, domain="arithmetic")
    raw_matches = list(_ARITHMETIC_PATTERN.finditer(response))

    captured: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        raw_match = raw_matches[idx] if idx < len(raw_matches) else None
        metadata = dict(result.metadata)
        captured.append(
            {
                "matched_text": (
                    raw_match.group(0)
                    if raw_match is not None
                    else result.description.split(" (correct:", 1)[0]
                ),
                "span_start": raw_match.start() if raw_match is not None else -1,
                "span_end": raw_match.end() if raw_match is not None else -1,
                "description": result.description,
                "satisfied": bool(metadata.get("satisfied", False)),
                "correct_result": metadata.get("correct_result"),
                "metadata": metadata,
            }
        )

    return captured


def diagnose_case(
    response: str,
    regex_matches: list[dict[str, Any]],
    extracted_answer: int | None,
    ground_truth: int,
    manual_override: dict[str, str] | None = None,
) -> dict[str, str]:
    """Diagnose why a response was or was not caught by regex arithmetic extraction."""
    if manual_override is not None:
        return dict(manual_override)

    if extracted_answer == ground_truth:
        if regex_matches:
            actual_error = "No reasoning error observed in the final answer."
        else:
            actual_error = (
                "No reasoning error observed in the final answer, but the model "
                "did not emit an explicit regex-readable equation."
            )
        return {
            "failure_category": "correct_answer",
            "actual_error": actual_error,
            "proposed_extraction": (
                "Keep the current regex checks for explicit equations, but add a "
                "natural-language step parser to improve coverage on non-equation text."
            ),
        }

    violated = [match for match in regex_matches if not match["satisfied"]]
    if violated:
        first = violated[0]
        return {
            "failure_category": "explicit_arithmetic_violation",
            "actual_error": (
                f"The response contains an explicit incorrect equation ({first['matched_text']})."
            ),
            "proposed_extraction": (
                "Surface the failing regex match directly to the verifier and repair loop."
            ),
        }

    lowered = response.lower()
    if extracted_answer is None:
        return {
            "failure_category": "missing_final_answer",
            "actual_error": (
                "The response never states a parsable final numeric answer, so the "
                "benchmark cannot align the reasoning trace to ground truth."
            ),
            "proposed_extraction": (
                "Use a structured answer extractor that can recover spelled-out or "
                "implicit final answers before arithmetic verification."
            ),
        }

    if any(term in lowered for term in _APPROXIMATION_TERMS):
        return {
            "failure_category": "rounding_or_approximation",
            "actual_error": (
                "The response relies on approximate arithmetic or rounding language "
                f"and lands on {extracted_answer} instead of {ground_truth}."
            ),
            "proposed_extraction": (
                "Use interval-aware or symbolic parsing that can model approximate quantities."
            ),
        }

    if not regex_matches:
        return {
            "failure_category": "format_mismatch",
            "actual_error": (
                f"The response reaches final answer {extracted_answer} instead of "
                f"{ground_truth} without emitting any `a +/- b = c` equation "
                "for the regex extractor."
            ),
            "proposed_extraction": (
                "Use a natural-language step parser or LLM/Z3 extractor that can "
                "normalize phrases like 'half of', 'left', 'twice', or implicit totals."
            ),
        }

    return {
        "failure_category": "subtle_logic_error",
        "actual_error": (
            f"All regex-visible equations are internally consistent, but the final "
            f"answer {extracted_answer} is still wrong for ground truth {ground_truth}."
        ),
        "proposed_extraction": (
            "Track variable bindings and semantic links across steps instead of "
            "verifying isolated equations only."
        ),
    }


def build_case_record(
    sample_position: int,
    dataset_idx: int,
    question: str,
    ground_truth: int,
    response: str,
    prompt: str | None = None,
    manual_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one serializable autopsy record."""
    extracted_answer = extract_final_number(response)
    regex_matches = capture_arithmetic_matches(response)
    diagnosis = diagnose_case(
        response=response,
        regex_matches=regex_matches,
        extracted_answer=extracted_answer,
        ground_truth=ground_truth,
        manual_override=manual_override,
    )
    correct = extracted_answer == ground_truth if extracted_answer is not None else False

    return {
        "sample_position": sample_position,
        "dataset_idx": dataset_idx,
        "question": question,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "response": response,
        "extracted_answer": extracted_answer,
        "correct": correct,
        "regex_matches": regex_matches,
        "n_regex_matches": len(regex_matches),
        "n_regex_violations": sum(1 for match in regex_matches if not match["satisfied"]),
        "failure_category": diagnosis["failure_category"],
        "actual_error": diagnosis["actual_error"],
        "proposed_extraction": diagnosis["proposed_extraction"],
    }


def select_showcase_cases(
    cases: list[dict[str, Any]],
    n_correct_examples: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Split the full run into wrong-answer autopsies and correct contrast cases."""
    wrong = [case for case in cases if not case["correct"]]
    correct = sorted(
        (case for case in cases if case["correct"]),
        key=lambda case: (-int(case["n_regex_matches"]), int(case["sample_position"])),
    )[:n_correct_examples]
    return {
        "wrong_answer_autopsies": wrong,
        "correct_answer_examples": correct,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize an autopsy run for reporting and JSON output."""
    failure_categories: dict[str, int] = {}
    for case in cases:
        category = str(case["failure_category"])
        failure_categories[category] = failure_categories.get(category, 0) + 1

    return {
        "n_questions": len(cases),
        "n_correct": sum(1 for case in cases if case["correct"]),
        "n_wrong": sum(1 for case in cases if not case["correct"]),
        "n_with_regex_matches": sum(1 for case in cases if case["n_regex_matches"] > 0),
        "n_with_regex_violations": sum(1 for case in cases if case["n_regex_violations"] > 0),
        "n_wrong_with_regex_violations": sum(
            1 for case in cases if (not case["correct"]) and case["n_regex_violations"] > 0
        ),
        "failure_categories": failure_categories,
    }
