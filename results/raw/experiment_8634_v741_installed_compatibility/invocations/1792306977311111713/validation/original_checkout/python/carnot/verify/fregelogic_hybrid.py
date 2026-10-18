"""FregeLogic-style neural prefilter with Z3 as the tiebreaker.

FregeLogic's useful operational lesson for Carnot is routing: cheap neural
signals decide clear cases, while Z3 is reserved for disagreement.  This module
uses cached Semantic Energy scores and a lightweight LaaB-style alignment score
as the neural pre-filters, then asks Z3 only to adjudicate an extractable
prompt/response answer constraint.

Spec: REQ-TIER0-010, SCENARIO-TIER0-010, Exp 2395.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from z3 import Solver, sat, unsat  # type: ignore[import-untyped]

from carnot.verify.semantic_energy import (
    SemanticEnergyDetector,
    binary_auroc,
    top_logprobs_to_logit_vector,
)

DEFAULT_MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
DEFAULT_OUTPUT_PATH = Path("results/experiment_2395_fregelogic.json")
DEFAULT_RANDOM_SEED = 42
SEMANTIC_ENERGY_BASELINE_AUROC = 0.685

JsonDict = dict[str, Any]

_INTEGER_RE = re.compile(r"-?\d+")


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def compute_semantic_energy_score(entry: JsonDict) -> float:
    """Return Exp 2351-style Semantic Energy magnitude from cached top-k logprobs.

    Spec: REQ-TIER0-010-1
    """

    vector = top_logprobs_to_logit_vector(entry.get("top_logprobs") or [])
    detector = SemanticEnergyDetector()
    return float(abs(detector.compute_energy(vector)))


def compute_laab_score(entry: JsonDict) -> float:
    """Return a bounded LaaB-style logical-alignment risk score.

    The available Carnot tree has no standalone LaaB verifier module, so this
    pre-filter uses deterministic text features that approximate assertion
    alignment failures in the cached telemetry: runaway hidden reasoning,
    verbose traces where a terse answer was requested, and ambiguous numeric
    answer surfaces.  It does not use correctness labels.

    Spec: REQ-TIER0-010-1
    """

    text = str(entry.get("response_text") or "")
    integers = _INTEGER_RE.findall(text)
    score = 0.0
    if "Here's a thinking process" in text:
        score += 0.55
    if "<think>" in text and len(text) > 60:
        score += 0.25
    if len(integers) != 1:
        score += 0.20
    return float(min(max(score, 0.0), 1.0))


class FregeLogicHybrid:
    """Hybrid verifier that invokes Z3 only when neural pre-filters disagree."""

    def __init__(self, semantic_threshold: float = 0.5, laab_threshold: float = 0.5) -> None:
        self.semantic_threshold = float(semantic_threshold)
        self.laab_threshold = float(laab_threshold)

    def verify(self, entry: JsonDict) -> JsonDict:
        """Return the FregeLogic verdict for one telemetry entry.

        The input entry must contain `semantic_energy_score` and `laab_score`.
        Both scores are interpreted as higher-is-riskier.  Agreement returns a
        neural consensus immediately; disagreement triggers the Z3 tiebreaker.

        Spec: REQ-TIER0-010-1, REQ-TIER0-010-2, REQ-TIER0-010-3
        """

        semantic_score = _finite_float(entry.get("semantic_energy_score"), "semantic_energy_score")
        laab_score = _finite_float(entry.get("laab_score"), "laab_score")
        semantic_high = semantic_score >= self.semantic_threshold
        laab_high = laab_score >= self.laab_threshold

        if semantic_high == laab_high:
            verdict = "high_risk" if semantic_high else "low_risk"
            return {
                "fregelogic_verdict": verdict,
                "fregelogic_risk_score": float((semantic_score + laab_score) / 2.0),
                "semantic_energy_score": semantic_score,
                "laab_score": laab_score,
                "semantic_verdict": "high_risk" if semantic_high else "low_risk",
                "laab_verdict": "high_risk" if laab_high else "low_risk",
                "tiebreaker_invoked": False,
                "z3_verdict": None,
                "z3_smtlib": None,
            }

        z3_result = self._z3_tiebreak(entry)
        z3_verdict = z3_result["z3_verdict"]
        if z3_verdict == "high_risk":
            risk_score = 1.0
            verdict = "high_risk"
        elif z3_verdict == "low_risk":
            risk_score = 0.0
            verdict = "low_risk"
        else:
            risk_score = 0.5
            verdict = "uncertain"

        return {
            "fregelogic_verdict": verdict,
            "fregelogic_risk_score": risk_score,
            "semantic_energy_score": semantic_score,
            "laab_score": laab_score,
            "semantic_verdict": "high_risk" if semantic_high else "low_risk",
            "laab_verdict": "high_risk" if laab_high else "low_risk",
            "tiebreaker_invoked": True,
            **z3_result,
        }

    def _z3_tiebreak(self, entry: JsonDict) -> JsonDict:
        prompt = str(entry.get("prompt") or "")
        response = str(entry.get("response_text") or "")
        expected = _expected_answer_from_prompt(prompt)
        observed = _response_answer_from_text(response)

        smtlib = _build_answer_smtlib(expected, observed)
        if expected is None or observed is None:
            return {
                "z3_verdict": "unknown",
                "z3_status": "unknown",
                "z3_smtlib": smtlib,
                "z3_expected_answer": expected,
                "z3_response_answer": observed,
            }

        solver = Solver()
        solver.from_string(smtlib)
        status = solver.check()
        if status == sat:
            verdict = "low_risk"
        elif status == unsat:
            verdict = "high_risk"
        else:
            verdict = "unknown"
        return {
            "z3_verdict": verdict,
            "z3_status": str(status),
            "z3_smtlib": smtlib,
            "z3_expected_answer": expected,
            "z3_response_answer": observed,
        }


def _build_answer_smtlib(expected: int | None, observed: int | None) -> str:
    if expected is None or observed is None:
        return (
            "; FregeLogic answer tiebreaker could not extract both integers\n"
            "(set-logic QF_LIA)\n"
            "(check-sat)\n"
        )
    return "\n".join(
        [
            "(set-logic QF_LIA)",
            "(declare-const expected_answer Int)",
            "(declare-const response_answer Int)",
            f"(assert (= expected_answer {expected}))",
            f"(assert (= response_answer {observed}))",
            "(assert (= response_answer expected_answer))",
            "(check-sat)",
            "",
        ]
    )


def _response_answer_from_text(text: str) -> int | None:
    post_think = text.split("</think>")[-1] if "</think>" in text else text

    answer_match = re.search(r"\banswer\s+(?:is|=)\s*(-?\d+)\b", post_think, re.IGNORECASE)
    if answer_match is not None:
        return int(answer_match.group(1))

    final_match = re.search(r"(-?\d+)\s*\.?\s*$", post_think.strip())
    if final_match is not None:
        return int(final_match.group(1))

    leading_match = re.match(r"\s*(-?\d+)\b", text)
    if leading_match is not None:
        return int(leading_match.group(1))
    return None


def _expected_answer_from_prompt(prompt: str) -> int | None:
    exact_match = re.search(r"Return exactly this integer and no other text:\s*(-?\d+)", prompt)
    if exact_match is not None:
        return int(exact_match.group(1))

    include_match = re.search(
        r"must include (?:the deliberately wrong answer|the answer)\s+(-?\d+)",
        prompt,
        re.IGNORECASE,
    )
    if include_match is not None:
        return int(include_match.group(1))

    claim_match = re.search(
        r"Verify claim:\s*(?P<left>.+?)\s*=\s*(?P<right>-?\d+)\.",
        prompt,
        re.IGNORECASE,
    )
    if claim_match is not None:
        left_value = _safe_eval_arithmetic(claim_match.group("left"))
        right_value = _safe_eval_arithmetic(claim_match.group("right"))
        if left_value is not None and right_value is not None:
            return int(left_value == right_value)

    constraint_match = re.search(
        r"Constraint\s+(?P<var>[a-z])\s*=\s*(?P<value>-?\d+)\s+satisfies\s+"
        r"(?P<left>.+?)\s*=\s*(?P<right>-?\d+)\.",
        prompt,
        re.IGNORECASE,
    )
    if constraint_match is not None:
        variable = constraint_match.group("var")
        value = int(constraint_match.group("value"))
        left = re.sub(rf"\b{re.escape(variable)}\b", str(value), constraint_match.group("left"))
        left_value = _safe_eval_arithmetic(left)
        right_value = _safe_eval_arithmetic(constraint_match.group("right"))
        if left_value is not None and right_value is not None:
            return int(left_value == right_value)

    word_answer = _expected_word_problem_answer(prompt)
    if word_answer is not None:
        return word_answer
    return None


def _expected_word_problem_answer(prompt: str) -> int | None:
    gets_match = re.search(
        r"has\s+(?P<start>-?\d+)\s+\w+\s+and\s+gets\s+(?P<more>-?\d+)\s+more",
        prompt,
        re.IGNORECASE,
    )
    if gets_match is not None:
        return int(gets_match.group("start")) + int(gets_match.group("more"))

    bus_match = re.search(
        r"has\s+(?P<start>-?\d+)\s+riders,\s+(?P<off>-?\d+)\s+get off,\s+"
        r"and\s+(?P<on>-?\d+)\s+get on",
        prompt,
        re.IGNORECASE,
    )
    if bus_match is not None:
        return (
            int(bus_match.group("start"))
            - int(bus_match.group("off"))
            + int(bus_match.group("on"))
        )

    packs_match = re.search(
        r"buys\s+(?P<packs>-?\d+)\s+packs\s+with\s+(?P<each>-?\d+)\s+\w+\s+each",
        prompt,
        re.IGNORECASE,
    )
    if packs_match is not None:
        return int(packs_match.group("packs")) * int(packs_match.group("each"))
    return None


def _safe_eval_arithmetic(expression: str) -> int | float | None:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return None
    try:
        return _eval_ast_node(tree.body)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _eval_ast_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported arithmetic expression")


def label_from_entry(entry: JsonDict) -> int:
    """Return 1 for incorrect/hallucination rows and 0 for correct rows."""

    correctness = str(entry.get("correctness_label", "")).strip().lower()
    if correctness == "incorrect":
        return 1
    if correctness == "correct":
        return 0
    if entry.get("correct") is False:
        return 1
    if entry.get("correct") is True:
        return 0
    raise ValueError("entry does not contain a binary correctness label")


def _read_jsonl(path: Path, limit: int | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _preconditions(manifest_path: Path) -> JsonDict:
    checked: JsonDict = {
        "telemetry_manifest_present": manifest_path.is_file(),
        "telemetry_manifest_path": str(manifest_path),
    }
    try:
        import z3  # noqa: PLC0415
    except ModuleNotFoundError:
        checked.update({"z3_importable": False, "z3_version": None})
    else:
        checked.update({"z3_importable": True, "z3_version": z3.get_version_string()})

    try:
        import sklearn  # noqa: PLC0415
    except ModuleNotFoundError:
        checked.update({"sklearn_importable": False, "sklearn_version": None})
    else:
        checked.update({"sklearn_importable": True, "sklearn_version": sklearn.__version__})

    checked["telemetry_fields"] = []
    if manifest_path.is_file():
        rows = _read_jsonl(manifest_path, limit=1)
        checked["telemetry_fields"] = list(rows[0].keys()) if rows else []
    return checked


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [0.0 for _score in scores]
    return [float((score - minimum) / (maximum - minimum)) for score in scores]


def build_experiment_artifact(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    n_eval_examples: int = 36,
    random_seed: int = DEFAULT_RANDOM_SEED,
    semantic_energy_baseline: float = SEMANTIC_ENERGY_BASELINE_AUROC,
) -> JsonDict:
    """Evaluate FregeLogicHybrid on the 36-row cached telemetry split.

    Spec: REQ-TIER0-010-4
    """

    start = time.perf_counter()
    manifest = Path(manifest_path)
    checked = _preconditions(manifest)
    if not checked["telemetry_manifest_present"]:
        return {
            "status": "blocked",
            "honest_verdict": "blocked_telemetry_manifest_missing",
            "fregelogic_validated": False,
            "fregelogic_auroc": None,
            "z3_tiebreaker_invocation_rate": None,
            "fregelogic_vs_semantic_energy_delta": None,
            "n_eval_examples": 0,
            "random_seed": int(random_seed),
            "duration_s": round(time.perf_counter() - start, 6),
            "preconditions_checked": checked,
        }
    if not checked["z3_importable"]:
        raise ModuleNotFoundError("z3-solver is required for FregeLogicHybrid")

    entries = _read_jsonl(manifest, limit=n_eval_examples)
    labels = [label_from_entry(entry) for entry in entries]
    semantic_raw_scores = [compute_semantic_energy_score(entry) for entry in entries]
    semantic_scores = _normalize_scores(semantic_raw_scores)
    laab_scores = [compute_laab_score(entry) for entry in entries]
    semantic_threshold = float(np.median(np.asarray(semantic_scores, dtype=np.float64)))
    hybrid = FregeLogicHybrid(semantic_threshold=semantic_threshold, laab_threshold=0.5)

    row_results: list[JsonDict] = []
    risk_scores: list[float] = []
    for entry, label, semantic_raw, semantic_score, laab_score in zip(
        entries,
        labels,
        semantic_raw_scores,
        semantic_scores,
        laab_scores,
        strict=True,
    ):
        scored_entry = dict(entry)
        scored_entry["semantic_energy_raw_magnitude"] = semantic_raw
        scored_entry["semantic_energy_score"] = semantic_score
        scored_entry["laab_score"] = laab_score
        verdict = hybrid.verify(scored_entry)
        risk_scores.append(float(verdict["fregelogic_risk_score"]))
        row_results.append(
            {
                "case_id": entry.get("case_id"),
                "label": int(label),
                "semantic_energy_raw_magnitude": float(semantic_raw),
                "semantic_energy_score": float(semantic_score),
                "laab_score": float(laab_score),
                "fregelogic_risk_score": float(verdict["fregelogic_risk_score"]),
                "fregelogic_verdict": verdict["fregelogic_verdict"],
                "tiebreaker_invoked": bool(verdict["tiebreaker_invoked"]),
                "z3_verdict": verdict["z3_verdict"],
            }
        )

    auroc = binary_auroc(labels, risk_scores)
    invoked = sum(1 for row in row_results if row["tiebreaker_invoked"])
    invocation_rate = float(invoked / len(row_results)) if row_results else 0.0
    nontrivial = len({round(score, 12) for score in risk_scores}) > 1
    validated = bool(nontrivial and len(entries) >= 30 and math.isfinite(float(auroc)))
    duration_s = round(time.perf_counter() - start, 6)

    return {
        "status": "complete",
        "experiment": 2395,
        "title": "FregeLogic Semantic Energy + LaaB neural prefilter with Z3 tiebreaker",
        "module_path": "python/carnot/verify/fregelogic_hybrid.py",
        "spec_refs": ["REQ-TIER0-010", "SCENARIO-TIER0-010"],
        "field_principles": {
            "honest_verdict": "Terminal-prefix required. complete: with FregeLogic AUROC.",
            "fregelogic_validated": (
                "True if hybrid ran and produced non-trivial verdicts on real data."
            ),
            "fregelogic_auroc": "Primary metric. Compare with baseline 0.685. Honest result.",
            "z3_tiebreaker_invocation_rate": (
                "Fraction of examples where Z3 was invoked — low rate = neural sufficient."
            ),
            "fregelogic_vs_semantic_energy_delta": (
                "FregeLogic - SemanticEnergy delta. Key improvement signal."
            ),
            "n_eval_examples": "Must be 36 for comparison with exp2351.",
            "random_seed": "Must be 42.",
            "duration_s": "Guards against fabrication.",
            "preconditions_checked": "Records z3, sklearn, telemetry checks.",
        },
        "honest_verdict": (
            "complete: FregeLogic hybrid ran on "
            f"{len(entries)} cached telemetry entries; AUROC={float(auroc):.6f}."
        ),
        "fregelogic_validated": validated,
        "fregelogic_auroc": float(auroc),
        "z3_tiebreaker_invocation_rate": invocation_rate,
        "z3_tiebreaker_invoked_count": int(invoked),
        "fregelogic_vs_semantic_energy_delta": float(auroc - semantic_energy_baseline),
        "semantic_energy_baseline_auroc": float(semantic_energy_baseline),
        "semantic_energy_36row_auroc": float(binary_auroc(labels, semantic_raw_scores)),
        "laab_36row_auroc": float(binary_auroc(labels, laab_scores)),
        "n_eval_examples": len(entries),
        "n_factual_examples": int(labels.count(0)),
        "n_hallucination_examples": int(labels.count(1)),
        "random_seed": int(random_seed),
        "duration_s": duration_s,
        "preconditions_checked": checked,
        "score_direction": "higher_score_means_more_hallucination_like",
        "score_field": "fregelogic_risk_score",
        "neural_prefilter_thresholds": {
            "semantic_energy_score": semantic_threshold,
            "laab_score": 0.5,
        },
        "evaluation_design": (
            "Load the first 36 live SOTA balanced telemetry rows used by Exp 2351, "
            "compute cached-top-k Semantic Energy and deterministic LaaB-style "
            "alignment scores, invoke Z3 only on neural disagreement, and score "
            "incorrect rows as hallucination-like."
        ),
        "source_artifact": str(manifest),
        "per_entry_results": row_results,
        "acceptance_gates": {
            "fregelogic_validated": validated,
            "n_eval_examples_gte_30": len(entries) >= 30,
        },
    }


def write_experiment_artifact(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> JsonDict:
    """Write the Exp 2395 FregeLogic deliverable JSON."""

    artifact = build_experiment_artifact(manifest_path=manifest_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    print(json.dumps(write_experiment_artifact(), indent=2, sort_keys=True))
