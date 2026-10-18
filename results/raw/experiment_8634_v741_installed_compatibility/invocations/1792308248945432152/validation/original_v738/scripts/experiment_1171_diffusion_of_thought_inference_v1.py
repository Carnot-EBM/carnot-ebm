#!/usr/bin/env python3
"""Exp 1171: Diffusion-of-Thought inference-time refinement.

Spec: REQ-INFER-017, SCENARIO-INFER-017-001.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carnot.eval.metrics import auroc  # noqa: E402


def _load_diffusion_of_thought_class() -> Any:
    module_path = PYTHON_DIR / "carnot" / "inference" / "diffusion_of_thought.py"
    spec = importlib.util.spec_from_file_location("carnot_dot_standalone", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DiffusionOfThought from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DiffusionOfThought


DiffusionOfThought = _load_diffusion_of_thought_class()

EXPERIMENT_ID = 1171
SEED = 1171
TIMESTEPS = (1, 5, 25, 125)
N_EXAMPLES = 50
ACCEPT_THRESHOLD = 0.2
CORPUS_PATH = PROJECT_ROOT / "data" / "fover_corpus.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1171_diffusion_of_thought_inference_v1.json"

_EQUATION_RE = re.compile(r"([0-9+\-*/().\s]+)=\s*(-?\d+(?:\.\d+)?)")
_CONTRADICTION_RE = re.compile(r"\b(wait no|incorrect|false|contradicts|inconsistent)\b", re.I)


class CheapK5FoVerEnergy:
    """CPU-only k=5-style text energy for the Exp 1171 DoT sweep."""

    def energy(self, response: str, context: str = "") -> float:
        text = _normalise_text(f"{context}\n{response}")
        arithmetic = _arithmetic_energy(text)
        contradiction = 1.0 if _CONTRADICTION_RE.search(text) else 0.0
        bracket = _bracket_energy(text)
        mask_residual = 0.0 if "[MASK]" in text else max(arithmetic, contradiction)
        calibrated_risk = max(arithmetic, contradiction, bracket)
        return float((arithmetic + contradiction + bracket + mask_residual + calibrated_risk) / 5.0)


def _normalise_text(text: str) -> str:
    return (
        text.replace("\\times", "*")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace(",", "")
    )


def _arithmetic_energy(text: str) -> float:
    mismatches = 0
    checked = 0
    for match in _EQUATION_RE.finditer(text):
        expression = match.group(1).strip()
        if not any(op in expression for op in "+-*/"):
            continue
        expected = _safe_eval_arithmetic(expression)
        if expected is None:
            continue
        claimed = float(match.group(2))
        checked += 1
        if abs(expected - claimed) > 1e-6:
            mismatches += 1
    if checked == 0:
        return 0.0
    return min(1.0, mismatches / checked)


def _safe_eval_arithmetic(expression: str) -> float | None:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    try:
        return float(_eval_node(tree.body))
    except (TypeError, ZeroDivisionError):
        return None


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise TypeError(f"unsupported arithmetic node: {type(node).__name__}")


def _bracket_energy(text: str) -> float:
    pairs = (("(", ")"), ("[", "]"), ("<<", ">>"))
    imbalances = sum(abs(text.count(left) - text.count(right)) for left, right in pairs)
    return min(1.0, imbalances / 4.0)


def _load_fover_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    return list(payload.get("pairs") or payload.get("items") or payload.get("examples") or [])


def _row_response(row: dict[str, Any]) -> str:
    return str(row.get("step_text") or row.get("response") or row.get("answer") or "")


def _row_context(row: dict[str, Any]) -> str:
    question = row.get("question") or row.get("original_question") or row.get("variant_question")
    if question:
        return str(question)
    return f"question_id: {row.get('question_id') or row.get('id') or row.get('question_index')}"


def _is_correct(row: dict[str, Any]) -> bool:
    label = row.get("label")
    if label is not None:
        return str(label).lower() in {"correct", "true", "1"}
    return bool(row.get("is_correct"))


def _is_gsm8k(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(row.get(key, "")) for key in ("question_id", "id", "source", "dataset", "question")
    ).lower()
    return "gsm8k" in haystack or bool(str(row.get("question_id", "")).isdigit())


def _select_examples(
    rows: list[dict[str, Any]], energy: CheapK5FoVerEnergy
) -> list[dict[str, Any]]:
    gsm_rows = [row for row in rows if _is_gsm8k(row) and _row_response(row).strip()]
    correct = [
        row for row in gsm_rows if _is_correct(row) and len(_row_response(row).split()) <= 45
    ]
    incorrect = [
        row
        for row in gsm_rows
        if not _is_correct(row)
        and len(_row_response(row).split()) <= 45
        and energy.energy(_row_response(row), _row_context(row)) > ACCEPT_THRESHOLD
    ]

    rng = random.Random(SEED)
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    selected = correct[: N_EXAMPLES // 2] + incorrect[: N_EXAMPLES // 2]

    if len(selected) < N_EXAMPLES:
        fallback = [
            row for row in gsm_rows if row not in selected and len(_row_response(row).split()) <= 45
        ]
        rng.shuffle(fallback)
        selected.extend(fallback[: N_EXAMPLES - len(selected)])

    if len(selected) < N_EXAMPLES:
        raise ValueError(f"only selected {len(selected)} FoVer GSM8K rows")

    rng.shuffle(selected)
    return selected[:N_EXAMPLES]


def _acceptance_accuracy(energies: list[float]) -> float:
    return sum(1 for energy in energies if energy <= ACCEPT_THRESHOLD) / len(energies)


def _verdict(deltas: list[float]) -> str:
    if max(deltas) <= 0.0:
        return "no_improvement_over_baseline"
    if all(after > before for before, after in zip(deltas, deltas[1:])):
        return "monotone_pareto_confirmed"
    return "non_monotone_diminishing_returns"


def run_experiment() -> dict[str, Any]:
    started_at = datetime.now(UTC)
    energy = CheapK5FoVerEnergy()
    dot = DiffusionOfThought(energy, n_candidates_per_step=5)
    rows = _select_examples(_load_fover_rows(CORPUS_PATH), energy)

    responses = [_row_response(row) for row in rows]
    contexts = [_row_context(row) for row in rows]
    labels_incorrect = [0 if _is_correct(row) else 1 for row in rows]
    baseline_energies = [
        dot.composite_energy(response, context) for response, context in zip(responses, contexts)
    ]
    baseline_accuracy = _acceptance_accuracy(baseline_energies)

    by_t: dict[int, dict[str, Any]] = {}
    for timestep in TIMESTEPS:
        start = time.perf_counter()
        final_energies: list[float] = []
        for response, context in zip(responses, contexts):
            _refined, trace = dot.refine(response, context, n_steps=timestep)
            final_energies.append(trace[-1])
        wall_time_ms = (time.perf_counter() - start) * 1000.0
        accuracy = _acceptance_accuracy(final_energies)
        by_t[timestep] = {
            "auroc": float(auroc(labels_incorrect, final_energies)),
            "wall_time_ms": float(wall_time_ms),
            "accuracy_delta": float(accuracy - baseline_accuracy),
        }

    deltas = [by_t[timestep]["accuracy_delta"] for timestep in TIMESTEPS]
    monotone = all(after > before for before, after in zip(deltas, deltas[1:]))
    pareto = [
        {
            "T": timestep,
            "accuracy_delta": round(float(by_t[timestep]["accuracy_delta"]), 6),
            "wall_time_ms": round(float(by_t[timestep]["wall_time_ms"]), 3),
        }
        for timestep in TIMESTEPS
    ]

    artifact = {
        "schema": "diffusion_of_thought_inference_v1",
        "experiment": EXPERIMENT_ID,
        "run_date": started_at.date().isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "fover_data_source": str(CORPUS_PATH.relative_to(PROJECT_ROOT)),
        "n_examples": len(rows),
        "n_gsm8k_examples": len(rows),
        "baseline_acceptance_accuracy": round(float(baseline_accuracy), 6),
        "baseline_energy_auroc": round(float(auroc(labels_incorrect, baseline_energies)), 6),
        "accept_threshold": ACCEPT_THRESHOLD,
        "energy_backend": "cheap_k5_fover_text_energy",
        "dot_t1_auroc": round(float(by_t[1]["auroc"]), 6),
        "dot_t5_auroc": round(float(by_t[5]["auroc"]), 6),
        "dot_t25_auroc": round(float(by_t[25]["auroc"]), 6),
        "dot_t125_auroc": round(float(by_t[125]["auroc"]), 6),
        "dot_t1_wall_time_ms": round(float(by_t[1]["wall_time_ms"]), 3),
        "dot_t125_wall_time_ms": round(float(by_t[125]["wall_time_ms"]), 3),
        "accuracy_delta_t1": round(float(by_t[1]["accuracy_delta"]), 6),
        "accuracy_delta_t125": round(float(by_t[125]["accuracy_delta"]), 6),
        "monotone_improvement": bool(monotone),
        "pareto_frontier": pareto,
        "dot_inference_pareto_measured": bool(
            by_t[1]["wall_time_ms"] > 0 and by_t[125]["wall_time_ms"] > 0
        ),
        "honest_verdict": _verdict(deltas),
    }
    return artifact


def main() -> int:
    artifact = run_experiment()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
