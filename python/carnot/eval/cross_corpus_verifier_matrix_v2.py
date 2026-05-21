"""Exp 2832 real-data cross-corpus verifier matrix.

This analyzer is intentionally simpler than the retired Exp 2824 script. The
older artifact filled missing upstream inputs with mock verifier rows, which
made the matrix look complete when no benchmark had actually measured those
verifiers. Here, absent upstream AUROC stays absent. That keeps the downstream
paper table from silently turning blocked runs into synthetic evidence.

Spec traces: REQ-VERIFY-2832, SCENARIO-VERIFY-2832,
SCENARIO-VERIFY-2832-BLOCKED-UPSTREAM.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


CORPORA = ("FoVer", "MBPP", "HumanEval", "TruthfulQA")
ARTIFACT_FILES = {
    "FoVer": "results/experiment_2828_fover_memory_leakage_isolation.json",
    "MBPP": "results/experiment_2829_mbpp_ensemble_eval.json",
    "HumanEval": "results/experiment_2830_humaneval_full_ensemble_eval.json",
    "TruthfulQA": "results/experiment_2831_truthfulqa_ensemble_eval.json",
}
OUTPUT_FILENAME = "experiment_2832_cross_corpus_verifier_matrix_v2.json"

ARCHITECTURE_TRANSFER_FLOOR = 0.75
HIGH_SIGNAL_FLOOR = 0.75
LOW_SIGNAL_CEILING = 0.65
MEMORY_DELTA_FLOOR = 0.10
REQUIRED_ARCHITECTURE_TRANSFER_PER_NON_FOVER = 3

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix.",
    "verifier_corpus_dual_matrix": (
        "Full 3-D matrix; downstream paper-v6 section 5 dual-condition table."
    ),
    "architecture_transfer_verifiers": (
        "Verifiers whose architecture-only signal transfers across every corpus."
    ),
    "memory_augmented_verifiers": (
        "Verifiers whose production signal depends materially on FR-11 memory."
    ),
    "corpus_specific_verifiers": "High-signal verifiers that do not transfer globally.",
    "low_signal_verifiers": "Verifiers below useful AUROC on every measured cell.",
    "diversity_gap_on_non_fover": (
        "True if fewer than 3 architecture-transfer verifiers cover each non-FoVer corpus."
    ),
    "duration_s": "Pure artifact analysis wall time; never padded.",
}


def normalize_auroc(value: object) -> float | None:
    """Normalize an upstream AUROC field into one measured float.

    The source artifacts are not perfectly uniform: Exp 2828 can report scalar
    per-verifier values while Exp 2829-2831 summarize seed rows as lists. The
    matrix needs one number per verifier/corpus/condition cell, so lists are
    averaged and null or empty values stay missing rather than becoming chance
    defaults.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("unsupported AUROC value: bool")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if not value:
            return None
        return sum(float(item) for item in value) / len(value)
    raise TypeError(f"unsupported AUROC value: {type(value).__name__}")


def _per_verifier_map(artifact: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = artifact.get(field, {})
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field} must be a mapping")


def _condition_value(artifact: Mapping[str, object], field: str, verifier: str) -> float | None:
    values = _per_verifier_map(artifact, field)
    if verifier not in values:
        return None
    return normalize_auroc(values[verifier])


def build_dual_matrix(
    corpora_artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Build the verifier x corpus x condition AUROC matrix from measured rows."""

    verifiers: set[str] = set()
    for corpus in CORPORA:
        artifact = corpora_artifacts.get(corpus, {})
        verifiers.update(_per_verifier_map(artifact, "per_verifier_condition_a_auroc"))
        verifiers.update(_per_verifier_map(artifact, "per_verifier_condition_b_auroc"))

    matrix: dict[str, dict[str, dict[str, float | None]]] = {}
    for verifier in sorted(verifiers):
        matrix[verifier] = {}
        for corpus in CORPORA:
            artifact = corpora_artifacts.get(corpus, {})
            production = _condition_value(artifact, "per_verifier_condition_a_auroc", verifier)
            architecture_only = _condition_value(
                artifact, "per_verifier_condition_b_auroc", verifier
            )
            delta = (
                production - architecture_only
                if production is not None and architecture_only is not None
                else None
            )
            matrix[verifier][corpus] = {
                "production": production,
                "architecture_only": architecture_only,
                "delta": delta,
            }
    return matrix


def _measured_values(cells: Mapping[str, Mapping[str, float | None]]) -> list[float]:
    values: list[float] = []
    for corpus in CORPORA:
        cell = cells.get(corpus, {})
        for key in ("production", "architecture_only"):
            value = cell.get(key)
            if value is not None:
                values.append(value)
    return values


def classify_verifier(cells: Mapping[str, Mapping[str, float | None]]) -> str:
    """Classify one verifier into the four Exp 2832 buckets.

    The precedence is deliberate. A verifier with robust architecture-only
    signal across all corpora is architecture-transfer even when production
    adds more lift. A verifier with no high production cell remains low-signal
    even if a low baseline makes the delta numerically large.
    """

    architecture_values = [
        cells.get(corpus, {}).get("architecture_only") for corpus in CORPORA
    ]
    if all(
        value is not None and value >= ARCHITECTURE_TRANSFER_FLOOR
        for value in architecture_values
    ):
        return "ARCHITECTURE_TRANSFER"

    for corpus in CORPORA:
        cell = cells.get(corpus, {})
        production = cell.get("production")
        architecture_only = cell.get("architecture_only")
        delta = cell.get("delta")
        if production is None or architecture_only is None:
            continue
        memory_drop = architecture_only < LOW_SIGNAL_CEILING
        memory_delta = delta is not None and delta >= MEMORY_DELTA_FLOOR
        if production >= HIGH_SIGNAL_FLOOR and (memory_drop or memory_delta):
            return "MEMORY_AUGMENTED"

    measured = _measured_values(cells)
    if not measured or all(value < LOW_SIGNAL_CEILING for value in measured):
        return "LOW_SIGNAL"
    return "CORPUS_SPECIFIC"


def classify_matrix(
    matrix: Mapping[str, Mapping[str, Mapping[str, float | None]]],
) -> dict[str, list[str]]:
    """Return deterministic verifier lists for every Exp 2832 category."""

    categories = {
        "ARCHITECTURE_TRANSFER": [],
        "MEMORY_AUGMENTED": [],
        "CORPUS_SPECIFIC": [],
        "LOW_SIGNAL": [],
    }
    for verifier in sorted(matrix):
        categories[classify_verifier(matrix[verifier])].append(verifier)
    return categories


def has_diversity_gap(
    matrix: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    architecture_transfer_verifiers: Sequence[str],
) -> bool:
    """Check whether every non-FoVer corpus has at least three transfer verifiers."""

    for corpus in CORPORA:
        if corpus == "FoVer":
            continue
        cover_count = sum(
            1
            for verifier in architecture_transfer_verifiers
            if (
                matrix.get(verifier, {})
                .get(corpus, {})
                .get("architecture_only")
                is not None
            )
            and matrix[verifier][corpus]["architecture_only"]
            >= ARCHITECTURE_TRANSFER_FLOOR
        )
        if cover_count < REQUIRED_ARCHITECTURE_TRANSFER_PER_NON_FOVER:
            return True
    return False


def _upstream_status(
    corpora_artifacts: Mapping[str, Mapping[str, object]],
    upstream_paths: Mapping[str, str] | None,
) -> dict[str, dict[str, object]]:
    statuses: dict[str, dict[str, object]] = {}
    for corpus in CORPORA:
        artifact = corpora_artifacts.get(corpus, {})
        condition_a = _per_verifier_map(artifact, "per_verifier_condition_a_auroc")
        condition_b = _per_verifier_map(artifact, "per_verifier_condition_b_auroc")
        status: dict[str, object] = {
            "honest_verdict": artifact.get("honest_verdict"),
            "n_condition_a_verifiers": len(condition_a),
            "n_condition_b_verifiers": len(condition_b),
        }
        if upstream_paths is not None:
            status["path"] = upstream_paths[corpus]
        statuses[corpus] = status
    return statuses


def build_matrix_artifact(
    corpora_artifacts: Mapping[str, Mapping[str, object]],
    *,
    duration_s: float,
    upstream_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build the terminal Exp 2832 artifact from already-loaded JSON payloads."""

    matrix = build_dual_matrix(corpora_artifacts)
    categories = classify_matrix(matrix)
    architecture_transfer = categories["ARCHITECTURE_TRANSFER"]
    diversity_gap = has_diversity_gap(matrix, architecture_transfer)
    if matrix:
        verdict = "complete: real upstream per-verifier AUROC matrix built"
        methodology_note = (
            "Exp 2832 loaded only Exp 2828-2831 artifact fields and normalized "
            "scalar or per-seed AUROC values to means. Missing verifier/corpus "
            "cells remain null; no fallback AUROC values were imputed."
        )
    else:
        verdict = (
            "complete: upstream artifacts loaded but no measured per-verifier "
            "AUROC rows were present"
        )
        methodology_note = (
            "No synthetic verifier rows were inferred. The upstream artifacts "
            "contained empty per-verifier AUROC maps, so the real-data matrix is "
            "empty and the diversity gap remains open."
        )

    return {
        "honest_verdict": verdict,
        "verifier_corpus_dual_matrix": matrix,
        "architecture_transfer_verifiers": architecture_transfer,
        "memory_augmented_verifiers": categories["MEMORY_AUGMENTED"],
        "corpus_specific_verifiers": categories["CORPUS_SPECIFIC"],
        "low_signal_verifiers": categories["LOW_SIGNAL"],
        "diversity_gap_on_non_fover": diversity_gap,
        "duration_s": duration_s,
        "upstream_artifact_statuses": _upstream_status(corpora_artifacts, upstream_paths),
        "classification_thresholds": {
            "architecture_transfer_floor": ARCHITECTURE_TRANSFER_FLOOR,
            "high_signal_floor": HIGH_SIGNAL_FLOOR,
            "low_signal_ceiling": LOW_SIGNAL_CEILING,
            "memory_delta_floor": MEMORY_DELTA_FLOOR,
            "required_architecture_transfer_per_non_fover": (
                REQUIRED_ARCHITECTURE_TRANSFER_PER_NON_FOVER
            ),
        },
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": methodology_note,
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifact(repo_root: Path, artifact: Mapping[str, object]) -> None:
    """Write the Exp 2832 artifact in the repository results directory."""

    output_path = repo_root / "results" / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def run_analysis(
    repo_root: Path,
    *,
    write: bool = True,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Load Exp 2828-2831 from disk, analyze them, and optionally write JSON."""

    start = clock()
    root = Path(repo_root)
    corpora_artifacts = {
        corpus: _load_json(root / relative_path)
        for corpus, relative_path in ARTIFACT_FILES.items()
    }
    artifact = build_matrix_artifact(
        corpora_artifacts,
        duration_s=clock() - start,
        upstream_paths=ARTIFACT_FILES,
    )
    if write:
        write_artifact(root, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through run_analysis tests.
    run_analysis(Path(__file__).resolve().parents[3])


if __name__ == "__main__":  # pragma: no cover
    main()
