"""NSVIF neuro-symbolic Z3 extraction for arithmetic reasoning traces.

This module provides the lightweight extractor surface requested by Exp 2352:
split chain-of-thought prose into assertion-like steps, encode supported
arithmetic claims as Z3 constraints, and classify traces by satisfiability.

The implementation is deliberately bounded. It accepts simple arithmetic
equalities and inequalities written either symbolically (``12 + 7 = 19``) or in
common instruction-tuned prose (``12 plus 7 equals 19``). Unsupported prose is
ignored instead of being treated as a violation, so an empty extraction cannot
create a false alarm.

Spec: REQ-VERIFY-1996, SCENARIO-VERIFY-1996
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3

EXPERIMENT_ID = 2352
RANDOM_SEED = 42
SPEC_REFS = ["REQ-VERIFY-1996", "SCENARIO-VERIFY-1996"]
DEFAULT_CORPUS_PATH = Path("results/experiment_2352_nsvif_corpus.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_2352_nsvif_extractor.json")

_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_IDENTIFIER = r"[A-Za-z_]\w*"
_TERM = rf"(?:{_NUMBER}|{_IDENTIFIER}|\([^()]+\))"
_EXPRESSION = rf"{_TERM}(?:\s*(?:\+|-|\*|/)\s*{_TERM})*"
_COMPARATOR = r"<=|>=|==|!=|=|<|>"
_CLAIM_RE = re.compile(
    rf"(?P<left>{_EXPRESSION})\s*(?P<cmp>{_COMPARATOR})\s*(?P<right>{_EXPRESSION})"
)
_STEP_SPLIT_RE = re.compile(r"(?:\n+|;|[.!?](?!\d))\s*")
_RESULT_CUE = (
    r"(?:=|==|equals?|is|are|gives?(?:\s+us)?|results?\s+in|"
    r"which\s+gives|which\s+is)"
)
_INFIX_RE = re.compile(
    rf"(?P<a>{_NUMBER})\s*"
    r"(?P<op>multiplied\s+by|divided\s+by|plus|minus|times|[+\-*/])\s*"
    rf"(?P<b>{_NUMBER})\s*(?:[,;:]?\s*){_RESULT_CUE}\s*"
    rf"(?P<claimed>{_NUMBER})",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    rf"(?P<pct>{_NUMBER})\s*(?:%|percent)\s+of\s+(?P<base>{_NUMBER})"
    rf"\s*(?:[,;:]?\s*){_RESULT_CUE}\s*(?P<claimed>{_NUMBER})",
    re.IGNORECASE,
)
_SUBTRACT_FROM_RE = re.compile(
    rf"subtract(?:ing)?\s+(?P<subtrahend>{_NUMBER})\s+from\s+(?P<base>{_NUMBER})"
    rf"\s*(?:[,;:]?\s*){_RESULT_CUE}\s*(?P<claimed>{_NUMBER})",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    rf"(?P<scale>half|double|twice|triple|three\s+times)\s+(?:of\s+)?"
    rf"(?P<operand>{_NUMBER})\s*(?:[,;:]?\s*){_RESULT_CUE}\s*"
    rf"(?P<claimed>{_NUMBER})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _EncodedAssertion:
    """One parsed assertion with source text retained for violation reporting."""

    step: str
    formula_text: str
    formula: z3.BoolRef


class NsvifExtractor:
    """Deterministic NSVIF-style extractor backed by Z3.

    The class exposes the three methods requested by Exp 2352:
    ``extract_steps`` finds assertion-like CoT sentences, ``encode_z3`` converts
    those assertions to Z3 ``BoolRef`` formulas, and ``verify`` runs a solver.
    The supported grammar is intentionally small because this is a feasibility
    extractor, not a full natural-language theorem prover.

    Spec: REQ-VERIFY-1996, SCENARIO-VERIFY-1996
    """

    def extract_steps(self, response: str) -> list[str]:
        """Return sentence-level arithmetic assertion candidates from a response."""

        normalized = _normalize_text(response)
        parts = [
            _strip_step_prefix(part).strip(" ,")
            for part in _STEP_SPLIT_RE.split(normalized)
            if part.strip(" ,")
        ]
        return [part for part in parts if _looks_like_assertion_step(part)]

    def encode_z3(self, steps: list[str]) -> list[z3.BoolRef]:
        """Encode supported arithmetic assertions as Z3 formulas."""

        return [assertion.formula for assertion in self._encode_assertions(steps)]

    def verify(self, response: str) -> dict[str, Any]:
        """Verify a reasoning trace and report satisfiability plus violations.

        ``verification_pass`` is true only when at least one constraint was
        extracted and the full constraint set is satisfiable. This prevents an
        empty extractor from accidentally passing the experiment gate.
        """

        steps = self.extract_steps(response)
        assertions = self._encode_assertions(steps)
        if not assertions:
            return {
                "satisfiable": True,
                "violations": [],
                "verification_pass": False,
                "solver_status": "no_constraints",
                "steps": steps,
                "constraints": [],
                "n_constraints": 0,
            }

        solver = z3.Solver()
        label_to_assertion: dict[str, _EncodedAssertion] = {}
        for idx, assertion in enumerate(assertions):
            label = f"nsvif_{idx}"
            label_to_assertion[label] = assertion
            solver.assert_and_track(assertion.formula, label)

        status = solver.check()
        satisfiable = status == z3.sat
        violations = _localize_violations(assertions, solver, label_to_assertion, satisfiable)

        return {
            "satisfiable": satisfiable,
            "violations": violations,
            "verification_pass": satisfiable,
            "solver_status": str(status),
            "steps": steps,
            "constraints": [assertion.formula_text for assertion in assertions],
            "n_constraints": len(assertions),
        }

    def _encode_assertions(self, steps: list[str]) -> list[_EncodedAssertion]:
        symbols: dict[str, z3.ArithRef] = {}
        assertions: list[_EncodedAssertion] = []
        seen: set[tuple[str, str]] = set()

        for step in steps:
            for claim_text in _claim_texts(step):
                encoded = _encode_claim_text(step, claim_text, symbols)
                if encoded is None:
                    continue
                dedupe_key = (encoded.step, encoded.formula_text)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                assertions.append(encoded)

        return assertions


def build_experiment_2352_corpus() -> list[dict[str, Any]]:
    """Return the deterministic 10-correct/10-incorrect Exp 2352 corpus."""

    correct = [
        ("correct_01", "First compute 12 + 7 = 19. Therefore 19 + 3 = 22."),
        ("correct_02", "We have 4 times 6 equals 24. Thus 24 - 5 = 19."),
        ("correct_03", "100 divided by 4 equals 25. Therefore 25 + 10 = 35."),
        ("correct_04", "15 minus 8 equals 7. So 7 * 3 = 21."),
        ("correct_05", "25 percent of 80 equals 20. Thus 20 + 5 = 25."),
        ("correct_06", "Subtracting 9 from 30 gives 21. Therefore 21 / 3 = 7."),
        ("correct_07", "2.5 + 1.5 = 4. Therefore 4 * 2 = 8."),
        ("correct_08", "9 times 9 gives 81. Thus 81 - 1 = 80."),
        ("correct_09", "14 + 6 equals 20. So 20 >= 19."),
        ("correct_10", "50 divided by 2 gives 25. Therefore 25 <= 30."),
    ]
    incorrect = [
        ("incorrect_01", "First compute 12 + 7 = 20. Therefore 20 + 3 = 23."),
        ("incorrect_02", "We have 4 times 6 equals 25. Thus 25 - 5 = 20."),
        ("incorrect_03", "100 divided by 4 equals 26. Therefore 26 + 10 = 36."),
        ("incorrect_04", "15 minus 8 equals 6. So 6 * 3 = 18."),
        ("incorrect_05", "25 percent of 80 equals 22. Thus 22 + 5 = 27."),
        ("incorrect_06", "Subtracting 9 from 30 gives 20. Therefore 20 / 3 = 7."),
        ("incorrect_07", "2.5 + 1.5 = 5. Therefore 5 * 2 = 10."),
        ("incorrect_08", "9 times 9 gives 82. Thus 82 - 1 = 81."),
        ("incorrect_09", "14 + 6 equals 20. So 20 < 19."),
        ("incorrect_10", "50 divided by 2 gives 25. Therefore 25 > 30."),
    ]

    rows: list[dict[str, Any]] = []
    for case_id, response in correct:
        rows.append({"case_id": case_id, "response": response, "expected_correct": True})
    for case_id, response in incorrect:
        rows.append({"case_id": case_id, "response": response, "expected_correct": False})
    return rows


def evaluate_nsvif_corpus(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Run ``NsvifExtractor`` on a corpus and compute Exp 2352 metrics."""

    extractor = NsvifExtractor()
    rows: list[dict[str, Any]] = []
    correct_classifications = 0
    covered = 0

    for case in corpus:
        result = extractor.verify(str(case["response"]))
        predicted_correct = bool(result["verification_pass"])
        expected_correct = bool(case["expected_correct"])
        if predicted_correct == expected_correct:
            correct_classifications += 1
        if int(result["n_constraints"]) > 0:
            covered += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "expected_correct": expected_correct,
                "predicted_correct": predicted_correct,
                "classification_correct": predicted_correct == expected_correct,
                "satisfiable": result["satisfiable"],
                "n_constraints": result["n_constraints"],
                "violations": result["violations"],
            }
        )

    n_cases = len(corpus)
    return {
        "verification_pass_rate": correct_classifications / n_cases if n_cases else 0.0,
        "extraction_coverage": covered / n_cases if n_cases else 0.0,
        "case_results": rows,
    }


def run_experiment_2352(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Write the Exp 2352 corpus and terminal artifact."""

    started_at = _utc_now()
    t0 = time.perf_counter()
    corpus = build_experiment_2352_corpus()
    corpus_out = Path(corpus_path)
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics = evaluate_nsvif_corpus(corpus)
    n_correct = sum(1 for case in corpus if bool(case["expected_correct"]))
    n_incorrect = sum(1 for case in corpus if not bool(case["expected_correct"]))
    validated = metrics["verification_pass_rate"] >= 0.60
    status = "complete" if validated else "failed"

    payload: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "title": "NSVIF neuro-symbolic Z3 constraint extractor",
        "status": status,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(time.perf_counter() - t0, 3),
        "schema_version": "carnot.extraction.nsvif_extractor.exp2352.v1",
        "source_arxiv": "https://arxiv.org/abs/2601.17789",
        "spec_refs": list(SPEC_REFS),
        "extractor_module": "carnot.extraction.nsvif_extractor",
        "corpus_path": str(corpus_out),
        "nsvif_extractor_validated": validated,
        "verification_pass_rate": metrics["verification_pass_rate"],
        "extraction_coverage": metrics["extraction_coverage"],
        "n_correct_examples": n_correct,
        "n_incorrect_examples": n_incorrect,
        "random_seed": RANDOM_SEED,
        "z3_version": z3.get_version_string(),
        "case_results": metrics["case_results"],
        "tests_run": [
            'PYTEST_ADDOPTS="" .venv/bin/python -m pytest tests/python/ -k "nsvif" -v --no-cov'
        ],
        "field_principles": {
            "honest_verdict": "Terminal-prefix required. complete: with pass rate.",
            "nsvif_extractor_validated": (
                "True if verification_pass_rate >= 0.60 on 20-example corpus."
            ),
            "verification_pass_rate": (
                "Primary metric. Expected >= 0.60 for basic feasibility."
            ),
            "extraction_coverage": (
                "Fraction of examples with >= 1 Z3 constraint found. Guards against empty-extractor."
            ),
            "n_correct_examples": "Must be 10. Validates correct path of NSVIF.",
            "n_incorrect_examples": "Must be 10. Validates violation detection.",
            "random_seed": "Reproducibility. Must be 42.",
            "z3_version": "Records which z3 version was used.",
        },
        "honest_verdict": (
            "complete: verification_pass_rate="
            f"{metrics['verification_pass_rate']:.3f}, extraction_coverage="
            f"{metrics['extraction_coverage']:.3f}"
            if validated
            else "failed: verification_pass_rate="
            f"{metrics['verification_pass_rate']:.3f}, extraction_coverage="
            f"{metrics['extraction_coverage']:.3f}"
        ),
    }

    artifact_out = Path(artifact_path)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    artifact_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("×", " * ").replace("÷", " / ")
    normalized = normalized.replace("≤", "<=").replace("≥", ">=")
    normalized = re.sub(r"</?think>", "\n", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def _strip_step_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[\):-]|\d+\.(?!\d))\s*", "", text).strip()


def _looks_like_assertion_step(step: str) -> bool:
    lowered = step.lower()
    has_number = bool(re.search(r"\d", step))
    has_cue = any(
        cue in lowered
        for cue in (
            "therefore",
            "thus",
            "hence",
            "so",
            "equals",
            "equal",
            "gives",
            "results in",
            "percent of",
        )
    )
    has_comparator = bool(re.search(r"(?:<=|>=|==|!=|=|<|>)", step))
    has_operator_words = bool(
        re.search(
            r"\b(?:plus|minus|times|multiplied by|divided by|subtracting)\b",
            lowered,
        )
    )
    return has_number and (has_cue or has_comparator or has_operator_words)


def _claim_texts(step: str) -> list[str]:
    claims: list[str] = []
    for match in _INFIX_RE.finditer(step):
        claims.append(
            f"{_clean_number(match.group('a'))} {_operator_symbol(match.group('op'))} "
            f"{_clean_number(match.group('b'))} = {_clean_number(match.group('claimed'))}"
        )

    for match in _PERCENT_RE.finditer(step):
        claims.append(
            f"{_clean_number(match.group('pct'))} * {_clean_number(match.group('base'))} "
            f"/ 100 = {_clean_number(match.group('claimed'))}"
        )

    for match in _SUBTRACT_FROM_RE.finditer(step):
        claims.append(
            f"{_clean_number(match.group('base'))} - {_clean_number(match.group('subtrahend'))} "
            f"= {_clean_number(match.group('claimed'))}"
        )

    for match in _SCALE_RE.finditer(step):
        multiplier = {
            "half": "0.5",
            "double": "2",
            "twice": "2",
            "triple": "3",
            "three times": "3",
        }[re.sub(r"\s+", " ", match.group("scale").lower())]
        claims.append(
            f"{multiplier} * {_clean_number(match.group('operand'))} = "
            f"{_clean_number(match.group('claimed'))}"
        )

    if claims:
        return _dedupe_preserve_order(claims)

    generic = _normalize_for_generic_claims(step)
    for match in _CLAIM_RE.finditer(generic):
        left = _clean_expression(match.group("left"))
        right = _clean_expression(match.group("right"))
        comparator = "==" if match.group("cmp") == "=" else match.group("cmp")
        if _valid_expression_shape(left) and _valid_expression_shape(right):
            claims.append(f"{left} {comparator} {right}")

    return _dedupe_preserve_order(claims)


def _normalize_for_generic_claims(step: str) -> str:
    normalized = step
    replacements = [
        (r"\bmultiplied\s+by\b", " * "),
        (r"\bdivided\s+by\b", " / "),
        (r"\bplus\b", " + "),
        (r"\bminus\b", " - "),
        (r"\btimes\b", " * "),
        (r"\bequals?\b", " = "),
        (r"\bgives?(?:\s+us)?\b", " = "),
        (r"\bresults?\s+in\b", " = "),
        (r"\b(?:therefore|thus|hence|so)\b", " "),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized.replace(",", " ")


def _encode_claim_text(
    step: str,
    claim_text: str,
    symbols: dict[str, z3.ArithRef],
) -> _EncodedAssertion | None:
    match = re.fullmatch(
        rf"\s*(?P<left>.+?)\s*(?P<cmp>{_COMPARATOR})\s*(?P<right>.+?)\s*",
        claim_text,
    )
    if match is None:
        return None

    left = _parse_arithmetic(match.group("left"), symbols)
    right = _parse_arithmetic(match.group("right"), symbols)
    if left is None or right is None:
        return None

    comparator = "==" if match.group("cmp") == "=" else match.group("cmp")
    formula = _compare(left, comparator, right)
    if formula is None:
        return None
    formula_text = (
        f"{_clean_expression(match.group('left'))} {comparator} "
        f"{_clean_expression(match.group('right'))}"
    )
    return _EncodedAssertion(step=step, formula_text=formula_text, formula=formula)


def _parse_arithmetic(expr: str, symbols: dict[str, z3.ArithRef]) -> z3.ArithRef | None:
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    try:
        return _ast_to_z3(parsed.body, symbols)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _ast_to_z3(node: ast.AST, symbols: dict[str, z3.ArithRef]) -> z3.ArithRef:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise TypeError("unsupported constant")
        return _z3_number(str(node.value))
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            symbols[node.id] = z3.Real(node.id)
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_to_z3(node.operand, symbols)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _ast_to_z3(node.operand, symbols)
    if isinstance(node, ast.BinOp):
        left = _ast_to_z3(node.left, symbols)
        right = _ast_to_z3(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise TypeError(f"unsupported arithmetic node {type(node).__name__}")


def _compare(left: z3.ArithRef, comparator: str, right: z3.ArithRef) -> z3.BoolRef | None:
    if comparator == "==":
        return left == right
    if comparator == "!=":
        return left != right
    if comparator == "<":
        return left < right
    if comparator == "<=":
        return left <= right
    if comparator == ">":
        return left > right
    if comparator == ">=":
        return left >= right
    return None


def _localize_violations(
    assertions: list[_EncodedAssertion],
    solver: z3.Solver,
    label_to_assertion: dict[str, _EncodedAssertion],
    satisfiable: bool,
) -> list[str]:
    violations: list[str] = []
    for assertion in assertions:
        local = z3.Solver()
        local.add(assertion.formula)
        if local.check() == z3.unsat:
            violations.append(assertion.step)

    if satisfiable or violations:
        return _dedupe_preserve_order(violations)

    for core_ref in solver.unsat_core():
        assertion = label_to_assertion.get(str(core_ref))
        if assertion is not None:
            violations.append(assertion.step)
    return _dedupe_preserve_order(violations)


def _z3_number(text: str) -> z3.ArithRef:
    cleaned = _clean_number(text)
    if "." in cleaned:
        return z3.RealVal(cleaned)
    return z3.IntVal(int(cleaned))


def _operator_symbol(operator: str) -> str:
    normalized = re.sub(r"\s+", " ", operator.lower().strip())
    return {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
    }.get(normalized, normalized)


def _clean_number(text: str) -> str:
    return text.replace(",", "")


def _clean_expression(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).strip()


def _valid_expression_shape(expr: str) -> bool:
    if not expr:
        return False
    return bool(re.search(r"\d", expr) or re.fullmatch(_IDENTIFIER, expr))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs.
    run_experiment_2352()
