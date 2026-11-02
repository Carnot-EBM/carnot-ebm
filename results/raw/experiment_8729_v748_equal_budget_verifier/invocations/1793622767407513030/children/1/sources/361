"""Pure helpers for Exp 1169 FoVer SOTA expansion v6.

Spec: REQ-VERIFY-1169, SCENARIO-VERIFY-1169
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SOURCE_TARGETS: dict[str, int] = {
    "gsm8k": 200,
    "humaneval": 100,
    "arc_challenge": 200,
}

REQUIRED_ARTIFACT_FIELDS: set[str] = {
    "n_new_pairs",
    "fover_sota_pairs_v6_above_500",
    "n_coherent",
    "n_incoherent",
    "n_z3_labeled",
    "n_sc_energy_labeled",
    "label_breakdown",
    "total_corpus_size",
    "honest_verdict",
}

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class CaseSpec:
    """Normalized prompt metadata for one Exp 1169 source item."""

    case_id: str
    source: str
    question: str
    answer: str | None
    choices: list[str] = field(default_factory=list)
    canonical_solution: str | None = None


def build_source_plan() -> list[CaseSpec]:
    """Return the required 200/100/200 source plan.

    The live script replaces these placeholders with dataset-loaded cases, but
    this pure plan makes the cardinality requirement testable without network
    or dataset access.
    """
    cases: list[CaseSpec] = []
    for source, n_cases in SOURCE_TARGETS.items():
        for idx in range(n_cases):
            cases.append(
                CaseSpec(
                    case_id=f"{source}_{idx:04d}",
                    source=source,
                    question=f"{source} placeholder question {idx}",
                    answer="0",
                )
            )
    return cases


def build_prompt(case: CaseSpec) -> str:
    """Build a concise SOTA generation prompt with the gold target included."""
    choices = "\n".join(case.choices)
    choice_block = f"\nChoices:\n{choices}" if choices else ""
    solution_hint = ""
    if case.canonical_solution:
        solution_hint = f"\nReference implementation:\n{case.canonical_solution.strip()}"
    answer = "" if case.answer is None else str(case.answer)
    return (
        "You are generating labeled reasoning-chain data for FoVer v6.\n"
        f"Source: {case.source}\n"
        f"Question: {case.question}{choice_block}{solution_hint}\n"
        f"Verified answer: {answer}\n"
        "Write a compact correct standard response with exactly three numbered steps.\n"
        "Step 2 must contain at least one explicit arithmetic equation such as 1 + 1 = 2.\n"
        "End with `Final answer: <verified answer>`."
    )


def latest_fover_corpus_size(
    results_dir: Path,
    exclude_paths: set[Path] | None = None,
) -> tuple[Path, int]:
    """Return the valid FoVer JSON in ``results_dir`` with the largest pair count."""
    best: tuple[Path, int] | None = None
    excluded = {path.resolve() for path in exclude_paths or set()}
    for path in sorted(results_dir.glob("*.json")):
        if path.resolve() in excluded:
            continue
        name = path.name.lower()
        if "fover" not in name:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        n_pairs = _pair_count(data)
        if n_pairs is None:
            continue
        if best is None or n_pairs > best[1] or (n_pairs == best[1] and path.name > best[0].name):
            best = (path, n_pairs)
    if best is None:
        raise FileNotFoundError(f"no valid FoVer corpus JSON found in {results_dir}")
    return best


def _pair_count(data: Any) -> int | None:
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return None
    for key in (
        "total_corpus_size",
        "n_pairs_after",
        "n_total_pairs",
        "n_pairs",
        "n_total",
        "n_examples",
    ):
        value = data.get(key)
        if isinstance(value, int):
            return value
    for key in ("pairs", "items", "examples", "data", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def assign_sc_energy_label(
    z3_pass: bool | None,
    ast_valid: bool,
    no_semantic_contradiction: bool,
    answer_matches_expected: bool,
) -> str:
    """Map verifier booleans to the SC-Energy training label."""
    if z3_pass is True and ast_valid and no_semantic_contradiction and answer_matches_expected:
        return "coherent"
    return "incoherent"


def inject_adversarial_step(response: str) -> str:
    """Inject a step-2/step-3 arithmetic and semantic contradiction."""
    injected = "Step 2: 2 + 2 = 5. Contradiction: this conflicts with valid arithmetic."
    lines = response.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"\s*Step\s*[23]\b", line, flags=re.IGNORECASE):
            lines[idx] = injected
            return "\n".join(lines)
    base = response.strip()
    return f"{base}\n{injected}" if base else injected


def answer_matches(response: str, expected: str | None) -> bool:
    """Return whether ``response`` commits to ``expected``."""
    if expected is None or str(expected).strip() == "":
        return True
    expected_text = str(expected).strip()
    expected_num = _parse_number(expected_text)
    if expected_num is not None:
        numbers = [_parse_number(match) for match in _NUM_RE.findall(response)]
        numbers = [value for value in numbers if value is not None]
        return bool(numbers and abs(numbers[-1] - expected_num) < 1e-6)
    return expected_text.lower() in response.lower()


def _parse_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def build_labeled_rows(
    case: CaseSpec,
    standard_response: str,
    model_id: str,
    *,
    z3_verifier: Any,
    ast_verifier: Any,
    semantic_verifier: Any,
) -> list[dict[str, Any]]:
    """Create standard and adversarial FoVer v6 rows for one case."""
    standard = _label_one_response(
        case,
        standard_response,
        model_id,
        response_kind="standard",
        z3_verifier=z3_verifier,
        ast_verifier=ast_verifier,
        semantic_verifier=semantic_verifier,
    )
    adversarial_response = inject_adversarial_step(standard_response)
    adversarial = _label_one_response(
        case,
        adversarial_response,
        model_id,
        response_kind="adversarial",
        z3_verifier=z3_verifier,
        ast_verifier=ast_verifier,
        semantic_verifier=semantic_verifier,
    )
    return [standard, adversarial]


def _label_one_response(
    case: CaseSpec,
    response: str,
    model_id: str,
    *,
    response_kind: str,
    z3_verifier: Any,
    ast_verifier: Any,
    semantic_verifier: Any,
) -> dict[str, Any]:
    z3_score = float(z3_verifier.score(response))
    if z3_score < 0.3:
        z3_pass: bool | None = True
    elif z3_score > 0.7:
        z3_pass = False
    else:
        z3_pass = None

    ast_score = float(ast_verifier.score(response))
    semantic_score = float(semantic_verifier.score(response))
    ast_valid = ast_score < 0.3
    no_semantic_contradiction = semantic_score < 0.5
    expected_matches = answer_matches(response, case.answer)
    sc_label = assign_sc_energy_label(
        z3_pass,
        ast_valid,
        no_semantic_contradiction,
        expected_matches,
    )
    row_id = f"exp1169-{case.source}-{case.case_id}-{response_kind}"
    return {
        "row_id": row_id,
        "question_id": case.case_id,
        "source": case.source,
        "label_source": case.source,
        "question": case.question,
        "response": response,
        "step_text": response,
        "response_kind": response_kind,
        "model": model_id,
        "label": "correct" if sc_label == "coherent" else "incorrect",
        "sc_energy_label": sc_label,
        "coherence_label": sc_label,
        "z3_pass": z3_pass,
        "z3_score": round(z3_score, 6),
        "ast_valid": ast_valid,
        "ast_score": round(ast_score, 6),
        "no_semantic_contradiction": no_semantic_contradiction,
        "semantic_contradiction_detected": not no_semantic_contradiction,
        "semantic_score": round(semantic_score, 6),
        "answer_matches_expected": expected_matches,
        "source_experiment": 1169,
        "schema_version": "fover_sota_expansion_v6",
    }


def append_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append rows to a JSONL corpus without truncating existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def build_artifact(
    rows: list[dict[str, Any]],
    *,
    prior_n_pairs: int,
    current_corpus_size: int,
    latest_corpus_path: Path,
    models_used: list[str],
    models_unavailable: list[str],
    batch_log: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """Build the required Exp 1169 artifact dictionary."""
    n_new = len(rows)
    n_coherent = sum(1 for row in rows if row.get("sc_energy_label") == "coherent")
    n_incoherent = sum(1 for row in rows if row.get("sc_energy_label") == "incoherent")
    n_z3_labeled = sum(1 for row in rows if row.get("z3_pass") is not None)
    n_sc_labeled = sum(
        1 for row in rows if row.get("sc_energy_label") in {"coherent", "incoherent"}
    )
    breakdown = _label_breakdown(rows)
    if not models_used:
        verdict = "gguf_model_unavailable"
    elif n_new >= 500 and n_sc_labeled == n_new:
        verdict = "corpus_expanded_labels_complete"
    else:
        verdict = "partial_500_not_reached"

    now = datetime.now(tz=UTC)
    artifact: dict[str, Any] = {
        "experiment": 1169,
        "title": "FoVer SOTA expansion v6 for SC-Energy labels",
        "schema": "fover_sota_expansion_v6",
        "run_date": now.strftime("%Y-%m-%d"),
        "started_at": "",
        "finished_at": now.isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "success" if verdict == "corpus_expanded_labels_complete" else "partial",
        "n_pairs_before": prior_n_pairs,
        "n_new_pairs": n_new,
        "fover_sota_pairs_v6_above_500": n_new >= 500,
        "n_coherent": n_coherent,
        "n_incoherent": n_incoherent,
        "n_z3_labeled": n_z3_labeled,
        "n_sc_energy_labeled": n_sc_labeled,
        "label_breakdown": breakdown,
        "total_corpus_size": current_corpus_size,
        "latest_fover_corpus_json": str(latest_corpus_path),
        "models_used": models_used,
        "models_unavailable": models_unavailable,
        "batch_size": 8,
        "batch_log": batch_log,
        "honest_verdict": verdict,
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    return artifact


def _label_breakdown(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "coherent": 0,
            "incoherent": 0,
            "z3_labeled": 0,
            "sc_energy_labeled": 0,
        }
    )
    for row in rows:
        source = str(row.get("source", "unknown"))
        label = row.get("sc_energy_label")
        out[source]["total"] += 1
        if label in {"coherent", "incoherent"}:
            out[source][label] += 1
            out[source]["sc_energy_labeled"] += 1
        if row.get("z3_pass") is not None:
            out[source]["z3_labeled"] += 1
    return dict(out)
