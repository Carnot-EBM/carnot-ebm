"""VERGE-style Minimal Correction Subset repair for NSVIF arithmetic claims.

Spec: REQ-VERIFY-2353, SCENARIO-VERIFY-2353
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

import z3

from carnot.extraction.nsvif_extractor import NsvifExtractor, _claim_texts
from carnot.verify.z3_math_verifier import _eval_arithmetic, _has_binary_operator

EXPERIMENT_ID = 2353
RANDOM_SEED = 42
SPEC_REFS = ["REQ-VERIFY-2353", "SCENARIO-VERIFY-2353"]
DEFAULT_ARTIFACT_PATH = Path("results/experiment_2353_verge_repair.json")
_COMPARATOR_RE = re.compile(r"\s*(?P<left>.+?)\s*(?P<cmp><=|>=|==|!=|=|<|>)\s*(?P<right>.+?)\s*")


@dataclass(frozen=True)
class RepairScenario:
    """One deterministic arithmetic repair localization scenario."""

    case_id: str
    response: str
    actual_error_index: int
    expected_bad_claim: str
    expected_good_claim: str


class VergeRepairEngine:
    """Locate minimal claim relaxations and produce deterministic repair hints.

    The MCS search is intentionally small and exact: it tests subsets of the
    reported violated indices in increasing size order and returns the first
    subset whose removal makes the remaining constraints satisfiable.
    """

    def __init__(self, extractor: NsvifExtractor | None = None) -> None:
        self.extractor = extractor or NsvifExtractor()

    def find_mcs(self, constraints: list[z3.BoolRef], violated: list[int]) -> list[int]:
        """Return the minimal violated constraint indices to relax.

        Parameters
        ----------
        constraints:
            Full Z3 constraint set extracted from an LLM response.
        violated:
            Candidate claim indices reported as violated by NSVIF or an
            upstream verifier. Invalid and duplicate indices are ignored.
        """

        if not constraints or _is_satisfiable(constraints):
            return []

        candidates = _dedupe_valid_indices(violated, len(constraints))
        if not candidates:
            candidates = list(range(len(constraints)))

        for width in range(1, len(candidates) + 1):
            for subset in combinations(candidates, width):
                relaxed = set(subset)
                remaining = [
                    constraint for idx, constraint in enumerate(constraints) if idx not in relaxed
                ]
                if _is_satisfiable(remaining):
                    return list(subset)
        return candidates

    def suggest_repair(self, response: str, violations: list[str]) -> str:
        """Return a concrete edit suggestion for the first supported violation."""

        candidate_violations = list(violations)
        if not candidate_violations:
            verified = self.extractor.verify(response)
            candidate_violations = list(verified.get("violations", []))

        for violation in candidate_violations:
            suggestion = _suggestion_for_violation(violation)
            if suggestion is not None:
                return suggestion

        if candidate_violations:
            return f"Review violated claim '{candidate_violations[0]}'"
        return "No repair suggested: no NSVIF violation was provided"


def build_experiment_2353_scenarios() -> list[RepairScenario]:
    """Return the deterministic 10-scenario Exp 2353 arithmetic repair set."""

    return [
        RepairScenario(
            case_id="repair_01_addition",
            response="First compute 12 + 7 = 20. Therefore 19 + 3 = 22.",
            actual_error_index=0,
            expected_bad_claim="12 + 7 = 20",
            expected_good_claim="12 + 7 = 19",
        ),
        RepairScenario(
            case_id="repair_02_multiplication",
            response="We have 4 times 6 equals 25. Thus 24 - 5 = 19.",
            actual_error_index=0,
            expected_bad_claim="4 * 6 = 25",
            expected_good_claim="4 * 6 = 24",
        ),
        RepairScenario(
            case_id="repair_03_division",
            response="100 divided by 4 equals 26. Therefore 25 + 10 = 35.",
            actual_error_index=0,
            expected_bad_claim="100 / 4 = 26",
            expected_good_claim="100 / 4 = 25",
        ),
        RepairScenario(
            case_id="repair_04_subtraction",
            response="15 minus 8 equals 6. So 7 * 3 = 21.",
            actual_error_index=0,
            expected_bad_claim="15 - 8 = 6",
            expected_good_claim="15 - 8 = 7",
        ),
        RepairScenario(
            case_id="repair_05_percent",
            response="25 percent of 80 equals 22. Thus 20 + 5 = 25.",
            actual_error_index=0,
            expected_bad_claim="25 * 80 / 100 = 22",
            expected_good_claim="25 * 80 / 100 = 20",
        ),
        RepairScenario(
            case_id="repair_06_subtract_from",
            response="Subtracting 9 from 30 gives 20. Therefore 21 / 3 = 7.",
            actual_error_index=0,
            expected_bad_claim="30 - 9 = 20",
            expected_good_claim="30 - 9 = 21",
        ),
        RepairScenario(
            case_id="repair_07_decimal",
            response="2.5 + 1.5 = 5. Therefore 4 * 2 = 8.",
            actual_error_index=0,
            expected_bad_claim="2.5 + 1.5 = 5",
            expected_good_claim="2.5 + 1.5 = 4",
        ),
        RepairScenario(
            case_id="repair_08_square",
            response="9 times 9 gives 82. Thus 81 - 1 = 80.",
            actual_error_index=0,
            expected_bad_claim="9 * 9 = 82",
            expected_good_claim="9 * 9 = 81",
        ),
        RepairScenario(
            case_id="repair_09_second_step_addition",
            response="18 / 3 = 6. Therefore 6 + 4 = 11.",
            actual_error_index=1,
            expected_bad_claim="6 + 4 = 11",
            expected_good_claim="6 + 4 = 10",
        ),
        RepairScenario(
            case_id="repair_10_second_step_subtraction",
            response="Double 14 equals 28. Therefore 28 - 8 = 21.",
            actual_error_index=1,
            expected_bad_claim="28 - 8 = 21",
            expected_good_claim="28 - 8 = 20",
        ),
    ]


def evaluate_verge_repair_scenarios(
    scenarios: list[RepairScenario] | None = None,
) -> dict[str, Any]:
    """Run VERGE MCS localization over the deterministic repair scenarios."""

    engine = VergeRepairEngine()
    rows: list[dict[str, Any]] = []
    successes = 0
    cases = scenarios or build_experiment_2353_scenarios()

    for scenario in cases:
        assertions = engine.extractor._encode_assertions(  # noqa: SLF001 - test harness needs metadata.
            engine.extractor.extract_steps(scenario.response)
        )
        verification = engine.extractor.verify(scenario.response)
        violation_steps = set(verification["violations"])
        constraints = [assertion.formula for assertion in assertions]
        violated_indices = [
            idx for idx, assertion in enumerate(assertions) if assertion.step in violation_steps
        ]
        mcs = engine.find_mcs(constraints, violated_indices)
        suggestion = engine.suggest_repair(scenario.response, list(verification["violations"]))
        mcs_correct = mcs == [scenario.actual_error_index]
        suggestion_correct = (
            scenario.expected_bad_claim in suggestion and scenario.expected_good_claim in suggestion
        )
        success = mcs_correct and suggestion_correct
        successes += int(success)
        rows.append(
            {
                "case_id": scenario.case_id,
                "actual_error_index": scenario.actual_error_index,
                "violated_indices": violated_indices,
                "mcs": mcs,
                "mcs_correct": mcs_correct,
                "suggestion": suggestion,
                "suggestion_correct": suggestion_correct,
                "success": success,
                "violations": verification["violations"],
                "constraints": [assertion.formula_text for assertion in assertions],
            }
        )

    n_cases = len(cases)
    success_rate = successes / n_cases if n_cases else 0.0
    return {
        "mcs_repair_success_rate": success_rate,
        "n_repair_scenarios": n_cases,
        "case_results": rows,
    }


def run_experiment_2353(
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Write the Exp 2353 VERGE repair artifact."""

    started_at = _utc_now()
    t0 = time.perf_counter()
    metrics = evaluate_verge_repair_scenarios()
    validated = metrics["mcs_repair_success_rate"] >= 0.50
    status = "complete" if validated else "failed"
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "title": "VERGE SMT Minimal Correction Subset repair",
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(time.perf_counter() - t0, 3),
        "schema_version": "carnot.repair.verge_repair.exp2353.v1",
        "source_arxiv": "https://arxiv.org/abs/2601.20055",
        "spec_refs": list(SPEC_REFS),
        "repair_module": "carnot.repair.verge_repair",
        "nsvif_extractor_module": "carnot.extraction.nsvif_extractor",
        "verge_repair_validated": validated,
        "mcs_repair_success_rate": metrics["mcs_repair_success_rate"],
        "n_repair_scenarios": metrics["n_repair_scenarios"],
        "random_seed": RANDOM_SEED,
        "z3_version": z3.get_version_string(),
        "case_results": metrics["case_results"],
        "tests_run": [
            '.venv/bin/python -c "import z3; print(z3.get_version_string())"',
            '.venv/bin/python -c "from carnot.extraction.nsvif_extractor import NsvifExtractor; print(\'OK\')"',
            'PYTEST_ADDOPTS="" .venv/bin/python -m pytest tests/python/ -k "verge" -v --no-cov 2>&1 | tail -15',
        ],
        "field_principles": {
            "honest_verdict": "Terminal-prefix required.",
            "verge_repair_validated": "True if mcs_repair_success_rate >= 0.50.",
            "mcs_repair_success_rate": (
                "Fraction of violated scenarios where MCS correctly identifies the error location."
            ),
            "n_repair_scenarios": "Must be 10.",
            "random_seed": "Reproducibility. Must be 42.",
        },
        "honest_verdict": (
            f"complete: mcs_repair_success_rate={metrics['mcs_repair_success_rate']:.3f}"
            if validated
            else f"failed: mcs_repair_success_rate={metrics['mcs_repair_success_rate']:.3f}"
        ),
    }

    artifact_out = Path(artifact_path)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    artifact_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _suggestion_for_violation(violation: str) -> str | None:
    for claim in _claim_texts(violation):
        suggestion = _suggestion_for_claim(claim)
        if suggestion is not None:
            return suggestion
    return None


def _suggestion_for_claim(claim: str) -> str | None:
    match = _COMPARATOR_RE.fullmatch(claim)
    if match is None:
        return None

    left = match.group("left").strip()
    comparator = match.group("cmp")
    right = match.group("right").strip()

    try:
        left_value = _eval_arithmetic(left)
        right_value = _eval_arithmetic(right)
    except Exception:
        return None

    normalized_comparator = "==" if comparator == "=" else comparator
    if normalized_comparator in {"==", "!="}:
        replacement = _equality_replacement(left, left_value, right, right_value)
        if replacement is None:
            return None
        return f"Change '{left} = {right}' to '{replacement}'"

    corrected = _correct_comparator(left_value, right_value)
    if corrected is None or corrected == normalized_comparator:
        return None
    return f"Change '{left} {comparator} {right}' to '{left} {corrected} {right}'"


def _equality_replacement(
    left: str,
    left_value: Fraction,
    right: str,
    right_value: Fraction,
) -> str | None:
    if left_value == right_value:
        return None
    if _has_binary_operator(left) and not _has_binary_operator(right):
        return f"{left} = {_format_fraction(left_value)}"
    if _has_binary_operator(right) and not _has_binary_operator(left):
        return f"{_format_fraction(right_value)} = {right}"
    if _has_binary_operator(left):
        return f"{left} = {_format_fraction(left_value)}"
    return None


def _correct_comparator(left_value: Fraction, right_value: Fraction) -> str | None:
    if left_value < right_value:
        return "<"
    if left_value > right_value:
        return ">"
    if left_value == right_value:
        return "="
    return None


def _is_satisfiable(constraints: list[z3.BoolRef]) -> bool:
    solver = z3.Solver()
    solver.add(*constraints)
    return solver.check() == z3.sat


def _dedupe_valid_indices(indices: list[int], constraint_count: int) -> list[int]:
    seen: set[int] = set()
    valid: list[int] = []
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        if index < 0 or index >= constraint_count or index in seen:
            continue
        seen.add(index)
        valid.append(index)
    return valid


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.12g}"


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs.
    run_experiment_2353()
