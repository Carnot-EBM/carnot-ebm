"""Exp 2840 real-data cross-corpus verifier matrix v3.

This analyzer is deliberately conservative: it only summarizes per-verifier
AUROC rows that already exist in upstream dual-condition artifacts. The .269
run left two nearby naming families in the repository, so the loader considers
both and selects the available artifact with the most measured verifier rows
for each corpus. Blocked corpora remain null in the matrix instead of becoming
synthetic chance-level rows.

Spec traces: REQ-VERIFY-MATRIX-2840,
SCENARIO-VERIFY-MATRIX-2840-REAL,
SCENARIO-VERIFY-MATRIX-2840-BLOCKED.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


CORPORA = ("FoVer", "MBPP", "HumanEval", "TruthfulQA")
ARTIFACT_CANDIDATES = {
    "FoVer": (
        "results/experiment_2836_fover_memory_leakage_isolation.json",
        "results/experiment_2837_fover_memory_leakage_v3.json",
    ),
    "MBPP": (
        "results/experiment_2837_mbpp_ensemble_eval.json",
        "results/experiment_2838_mbpp_dual_condition_v3.json",
    ),
    "HumanEval": (
        "results/experiment_2838_humaneval_full_ensemble_eval.json",
        "results/experiment_2839_humaneval_dual_condition_v3.json",
    ),
    "TruthfulQA": (
        "results/experiment_2839_truthfulqa_ensemble_eval.json",
        "results/experiment_2840_truthfulqa_dual_condition_v4.json",
    ),
}
OUTPUT_FILENAME = "experiment_2840_cross_corpus_verifier_matrix_v3.json"

ARCHITECTURE_TRANSFER_FLOOR = 0.75
HIGH_SIGNAL_FLOOR = 0.75
LOW_SIGNAL_CEILING = 0.65
MEMORY_DELTA_FLOOR = 0.10
REQUIRED_ARCHITECTURE_TRANSFER_PER_NON_FOVER = 3

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix.",
    "verifier_corpus_dual_matrix": (
        "Full verifier x corpus x condition matrix from measured upstream AUROC only."
    ),
    "architecture_transfer_verifiers": (
        "Architecture-only high signal on every measured corpus, including non-FoVer."
    ),
    "memory_augmented_verifiers": (
        "Production high signal whose architecture-only counterpart is absent or weak."
    ),
    "corpus_specific_verifiers": (
        "High-signal measured verifiers without enough architecture-only transfer evidence."
    ),
    "low_signal_verifiers": "Verifiers below useful AUROC on every measured cell.",
    "diversity_gap_on_non_fover": (
        "True if fewer than 3 architecture-transfer verifiers cover any non-FoVer corpus."
    ),
    "duration_s": "Pure artifact analysis wall time; never padded.",
}


def normalize_auroc(value: object) -> float | None:
    """Normalize one upstream AUROC field into a measured float.

    Upstream experiments sometimes store a scalar mean and sometimes store the
    five seed-level AUROCs. The matrix needs one value per condition cell, so
    seed lists are averaged. Empty and null values stay missing because turning
    a blocked measurement into 0.5 would create evidence that was never run.
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


def _measured_verifier_count(artifact: Mapping[str, object]) -> int:
    verifiers = set(_per_verifier_map(artifact, "per_verifier_condition_a_auroc"))
    verifiers.update(_per_verifier_map(artifact, "per_verifier_condition_b_auroc"))
    return len(verifiers)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_artifact_for_corpus(
    repo_root: Path,
    corpus: str,
    candidate_paths: Sequence[str],
) -> tuple[str, dict[str, object]]:
    """Select the available artifact with the most measured verifier rows."""

    root = Path(repo_root)
    selected_path = ""
    selected_payload: dict[str, object] = {}
    selected_count = -1
    for relative_path in candidate_paths:
        path = root / relative_path
        if not path.is_file():
            continue
        payload = _load_json(path)
        measured_count = _measured_verifier_count(payload)
        if measured_count > selected_count:
            selected_path = relative_path
            selected_payload = payload
            selected_count = measured_count
    if not selected_path:
        selected_payload = {
            "honest_verdict": f"blocked_missing_artifact_{corpus.lower()}",
            "per_verifier_condition_a_auroc": {},
            "per_verifier_condition_b_auroc": {},
        }
    return selected_path, selected_payload


def load_corpora_artifacts(
    repo_root: Path,
    artifact_candidates: Mapping[str, Sequence[str]] = ARTIFACT_CANDIDATES,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Load selected upstream artifacts and record path-selection evidence."""

    artifacts: dict[str, dict[str, object]] = {}
    source_artifacts: dict[str, dict[str, object]] = {}
    root = Path(repo_root)
    for corpus in CORPORA:
        candidates = tuple(artifact_candidates[corpus])
        candidate_counts = {}
        for relative_path in candidates:
            path = root / relative_path
            if path.is_file():
                candidate_counts[relative_path] = _measured_verifier_count(_load_json(path))
        selected_path, payload = select_artifact_for_corpus(root, corpus, candidates)
        artifacts[corpus] = payload
        condition_a = _per_verifier_map(payload, "per_verifier_condition_a_auroc")
        condition_b = _per_verifier_map(payload, "per_verifier_condition_b_auroc")
        source_artifacts[corpus] = {
            "candidate_paths": list(candidates),
            "candidate_measured_verifier_counts": candidate_counts,
            "selected_path": selected_path,
            "honest_verdict": payload.get("honest_verdict"),
            "n_condition_a_verifiers": len(condition_a),
            "n_condition_b_verifiers": len(condition_b),
            "n_measured_verifiers": _measured_verifier_count(payload),
        }
    return artifacts, source_artifacts


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
                artifact,
                "per_verifier_condition_b_auroc",
                verifier,
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


def _measured_corpora(cells: Mapping[str, Mapping[str, float | None]]) -> list[str]:
    measured = []
    for corpus in CORPORA:
        cell = cells.get(corpus, {})
        if cell.get("production") is not None or cell.get("architecture_only") is not None:
            measured.append(corpus)
    return measured


def classify_verifier(cells: Mapping[str, Mapping[str, float | None]]) -> str:
    """Classify one verifier into the four Exp 2840 matrix buckets."""

    measured_corpora = _measured_corpora(cells)
    if measured_corpora and any(corpus != "FoVer" for corpus in measured_corpora):
        architecture_values = [
            cells.get(corpus, {}).get("architecture_only") for corpus in measured_corpora
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
        if production is None or production < HIGH_SIGNAL_FLOOR:
            continue
        missing_architecture = architecture_only is None
        weak_architecture = architecture_only is not None and architecture_only < LOW_SIGNAL_CEILING
        memory_delta = delta is not None and delta >= MEMORY_DELTA_FLOOR
        if missing_architecture or weak_architecture or memory_delta:
            return "MEMORY_AUGMENTED"

    measured = _measured_values(cells)
    if not measured or all(value < LOW_SIGNAL_CEILING for value in measured):
        return "LOW_SIGNAL"
    return "CORPUS_SPECIFIC"


def classify_matrix(
    matrix: Mapping[str, Mapping[str, Mapping[str, float | None]]],
) -> dict[str, list[str]]:
    """Return deterministic verifier lists for every Exp 2840 category."""

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
            if (matrix.get(verifier, {}).get(corpus, {}).get("architecture_only") is not None)
            and matrix[verifier][corpus]["architecture_only"] >= ARCHITECTURE_TRANSFER_FLOOR
        )
        if cover_count < REQUIRED_ARCHITECTURE_TRANSFER_PER_NON_FOVER:
            return True
    return False


def build_matrix_artifact(
    corpora_artifacts: Mapping[str, Mapping[str, object]],
    *,
    duration_s: float,
    source_artifacts: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build the terminal Exp 2840 artifact from already-loaded payloads."""

    matrix = build_dual_matrix(corpora_artifacts)
    categories = classify_matrix(matrix)
    architecture_transfer = categories["ARCHITECTURE_TRANSFER"]
    diversity_gap = has_diversity_gap(matrix, architecture_transfer)
    if matrix:
        verdict = "complete: real upstream per-verifier AUROC matrix v3 built"
        methodology_note = (
            "Exp 2840 loaded measured per-verifier AUROC rows from the selected "
            "post-torch-fix/post-codex-flip artifacts. Missing verifier/corpus "
            "cells remain null; no fallback AUROC values were imputed."
        )
    else:
        verdict = (
            "complete: upstream artifacts loaded but no measured per-verifier "
            "AUROC rows were present"
        )
        methodology_note = (
            "No synthetic verifier rows were inferred. The selected upstream "
            "artifacts contained empty per-verifier AUROC maps, so the real-data "
            "matrix is empty and the diversity gap remains open."
        )

    return {
        "honest_verdict": verdict,
        "verifier_corpus_dual_matrix": matrix,
        "architecture_transfer_verifiers": architecture_transfer,
        "memory_augmented_verifiers": categories["MEMORY_AUGMENTED"],
        "corpus_specific_verifiers": categories["CORPUS_SPECIFIC"],
        "low_signal_verifiers": categories["LOW_SIGNAL"],
        "diversity_gap_on_non_fover": diversity_gap,
        "duration_s": float(duration_s),
        "source_artifacts": dict(source_artifacts or {}),
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


def write_artifact(repo_root: Path, artifact: Mapping[str, object]) -> None:
    """Write the Exp 2840 matrix artifact in the repository results directory."""

    output_path = Path(repo_root) / "results" / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def run_analysis(
    repo_root: Path,
    *,
    write: bool = True,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Load selected upstream artifacts, analyze them, and optionally write JSON."""

    start = clock()
    corpora_artifacts, source_artifacts = load_corpora_artifacts(Path(repo_root))
    artifact = build_matrix_artifact(
        corpora_artifacts,
        duration_s=clock() - start,
        source_artifacts=source_artifacts,
    )
    if write:
        write_artifact(Path(repo_root), artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through run_analysis tests.
    run_analysis(Path(__file__).resolve().parents[3])


if __name__ == "__main__":  # pragma: no cover
    main()
