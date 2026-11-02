"""Exp 4231 cross-candidate oracle-distinct ARC aggregator.

Spec refs: REQ-VERIFY-4231, SCENARIO-VERIFY-4231,
SCENARIO-VERIFY-4231-NO-GAIN, SCENARIO-VERIFY-4231-BLOCKED.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


RANDOM_SEED = 4231
BOOTSTRAP_N = 1000
BASELINE_AUROC_391 = 0.778980279
BASELINE_IMPROVEMENT_EPSILON = 0.001
SPARSE_POSITIVE_THRESHOLD = 30
WRONG_MAJORITY_TARGET = 30
OUTPUT_REL = Path("results/experiment_4231_oracle_distinct_arc_aggregator_build.json")
AGGREGATOR_REL = Path("results/experiment_4231_oracle_distinct_arc_aggregator_model.json")
SPEC_REFS = [
    "REQ-VERIFY-4231",
    "SCENARIO-VERIFY-4231",
    "SCENARIO-VERIFY-4231-NO-GAIN",
    "SCENARIO-VERIFY-4231-BLOCKED",
]


@dataclass(frozen=True)
class PoolSpec:
    source_id: str
    pool_rel: Path
    programs_rel: Path
    required: bool


DEFAULT_POOL_SPECS = (
    PoolSpec(
        "gap3_stage2",
        Path("results/arc3_gap3_stage2_eval_pool.json.gz"),
        Path("results/arc3_gap4_induced_programs.json"),
        True,
    ),
    PoolSpec(
        "gap4_arc2",
        Path("results/arc3_gap4_arc2_eval_pool.json.gz"),
        Path("results/arc3_gap4_arc2_induced_programs.json"),
        False,
    ),
)

BASE_FEATURE_NAMES = (
    "vote_weight",
    "self_consistency_margin",
    "vote_weight_rank_fraction",
    "cell_confidence_mean",
    "cell_confidence_margin",
    "cell_confidence_rank_fraction",
    "grid_height",
    "grid_width",
    "grid_cells",
    "grid_color_count",
    "grid_nonzero_frac",
    "grid_entropy",
    "program_length",
    "program_digit_fraction",
    "program_demo_fit",
    "program_n_calls",
)
CROSS_FEATURE_NAMES = (
    "set_candidate_count",
    "set_vote_mean",
    "set_vote_max",
    "set_vote_std",
    "set_confidence_mean",
    "set_confidence_max",
    "set_confidence_std",
    "set_entropy_mean",
    "set_entropy_max",
    "set_entropy_std",
    "set_cells_mean",
    "set_cells_max",
    "set_cells_std",
    "vote_weight_zscore",
    "cell_confidence_zscore",
    "grid_entropy_zscore",
    "grid_cells_zscore",
    "modal_cell_agreement_frac",
    "grid_duplicate_count",
    "grid_duplicate_frac",
    "shape_family_count",
    "shape_family_frac",
    "shape_vote_frac",
    "is_modal_shape",
    "palette_family_count",
    "palette_family_frac",
    "palette_vote_frac",
    "is_modal_palette",
    "same_shape_as_input",
    "area_delta_from_input_frac",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + CROSS_FEATURE_NAMES

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (complete:/success:/passed:/shipped:). A trained off-fold "
        "aggregator OR an honest 'no learnable gain over the .391 logistic baseline' "
        "is COMPLETE -- both feed A2."
    ),
    "aggregator_trained": (
        "BARE bool: A2's gate compares this raw value (gated-fields-must-be-bare); "
        "true iff a learned cross-candidate ARC aggregator artifact was persisted out-of-fold."
    ),
    "oracle_distinct_auroc": (
        "BARE float: off-fold detection AUROC of the AGGREGATOR vs is_correct -- the "
        "oracle-distinct discrimination; must improve on the .391 logistic 0.779 for "
        "a stronger A2 read, and >0.5 CI95-excl is the precondition for a beats-vote win."
    ),
    "held_out_task_n": (
        "BARE int: number of held-out tasks the A2 gate will score on -- target >=30 "
        "(CLT floor) so the A2 null/win is not under-powered like the .391 n=14."
    ),
    "wrong_majority_n": (
        "BARE int: count of stratified tasks where oracle@K > vote@1 -- the "
        "ARBITER/AggLM headroom the aggregator targets; A2 measures vote-beating ON these."
    ),
    "learned_verifier_path": (
        "The persisted aggregator artifact A2 loads to rerank held-out ARC candidates; the build deliverable."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the aggregator scores WITHOUT executing the demos "
        "(Circularity Discipline); this is what makes an A2 win headline/gate-eligible, "
        "unlike the circular execution verifier."
    ),
    "model_specs": (
        "The set-encoder/aggregator architecture + the cross-candidate feature set + "
        "the calibrated imbalance-aware loss; required methodology."
    ),
    "random_seed": (
        "Determinism precondition; the fold split + model init seeded so the AUROC is reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the ARC pools + fold split + features; catches silent pool/feature drift before A2 measures."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "aggregator_trained",
    "oracle_distinct_auroc",
    "oracle_distinct_auroc_ci95",
    "held_out_task_n",
    "wrong_majority_n",
    "learned_verifier_path",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ArcAggregatorRow:
    source_id: str
    task_id: str
    candidate_id: str
    candidate_index: int
    vote_weight: float
    correct: bool
    features: dict[str, float]
    raw_candidate_correct_flag: bool | None


@dataclass(frozen=True)
class ArcAggregatorCorpus:
    rows: list[ArcAggregatorRow]
    source_paths: list[Path]
    source_sha256: dict[str, str]
    held_out_task_n: int
    wrong_majority_n: int
    raw_candidate_n: int
    detector_row_n: int
    skipped_optional_pools: list[str]


@dataclass(frozen=True)
class OOFRow:
    task_id: str
    candidate_id: str
    correct: bool
    score: float
    fold: int
    train_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class OOFReport:
    oracle_distinct_auroc: float
    oracle_distinct_auroc_ci95: tuple[float, float]
    fold_task_ids: list[list[str]]
    oof_rows: list[OOFRow]
    final_aggregator: dict[str, Any]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _flatten_grid(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    flat: list[float] = []
    for row in value:
        if isinstance(row, list):
            flat.extend(_as_float(cell) for cell in row)
        else:
            flat.append(_as_float(row))
    return flat


def _grid_shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, list):
        return (0, 0)
    height = len(value)
    width = max((len(row) if isinstance(row, list) else 1 for row in value), default=0)
    return height, width


def _grid_equal(left: Any, right: Any) -> bool:
    left_h, left_w = _grid_shape(left)
    right_h, right_w = _grid_shape(right)
    return (left_h, left_w) == (right_h, right_w) and _flatten_grid(left) == _flatten_grid(right)


def _grid_stats(value: Any) -> dict[str, float]:
    height, width = _grid_shape(value)
    flat = _flatten_grid(value)
    cells = len(flat)
    if cells == 0:
        return {
            "grid_height": 0.0,
            "grid_width": 0.0,
            "grid_cells": 0.0,
            "grid_color_count": 0.0,
            "grid_nonzero_frac": 0.0,
            "grid_entropy": 0.0,
        }
    counts = Counter(flat)
    entropy = -sum((count / cells) * math.log2(count / cells) for count in counts.values())
    return {
        "grid_height": float(height),
        "grid_width": float(width),
        "grid_cells": float(cells),
        "grid_color_count": float(len(counts)),
        "grid_nonzero_frac": sum(1 for cell in flat if cell != 0.0) / float(cells),
        "grid_entropy": entropy,
    }


def _rank_fraction(value: float, values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    ordered = sorted(values, reverse=True)
    if value not in ordered:
        return 0.0
    rank = ordered.index(value)
    return 1.0 - rank / float(len(values) - 1)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values)))


def _program_stats(program: dict[str, Any]) -> dict[str, float]:
    code = program.get("code", "")
    if not isinstance(code, str):
        code = json.dumps(code, sort_keys=True)
    digit_count = sum(1 for char in code if char.isdigit())
    length = len(code)
    return {
        "program_length": float(length),
        "program_digit_fraction": digit_count / float(length) if length else 0.0,
        "program_demo_fit": _as_float(program.get("demo_fit")),
        "program_n_calls": _as_float(program.get("n_calls")),
    }


def _grid_signature(grid: Any) -> str:
    return json.dumps(grid if isinstance(grid, list) else [], sort_keys=True, separators=(",", ":"))


def _shape_key(grid: Any) -> str:
    height, width = _grid_shape(grid)
    return f"{height}x{width}"


def _palette_key(grid: Any) -> str:
    return ",".join(str(value) for value in sorted(set(_flatten_grid(grid))))


def _modal_grids_by_shape(grids: list[Any]) -> dict[str, list[list[float]]]:
    grouped: dict[str, list[Any]] = {}
    for grid in grids:
        grouped.setdefault(_shape_key(grid), []).append(grid)
    modal_by_shape: dict[str, list[list[float]]] = {}
    for shape, shape_grids in grouped.items():
        height, width = _grid_shape(shape_grids[0])
        modal: list[list[float]] = []
        for row_index in range(height):
            row: list[float] = []
            for col_index in range(width):
                values = []
                for grid in shape_grids:
                    try:
                        values.append(_as_float(grid[row_index][col_index]))
                    except (IndexError, TypeError):
                        values.append(0.0)
                counts = Counter(values)
                row.append(max(counts, key=lambda value: (counts[value], -value)))
            modal.append(row)
        modal_by_shape[shape] = modal
    return modal_by_shape


def _modal_cell_agreement(grid: Any, modal_grid: Any) -> float:
    flat = _flatten_grid(grid)
    modal = _flatten_grid(modal_grid)
    if not flat or len(flat) != len(modal):
        return 0.0
    return sum(1 for left, right in zip(flat, modal, strict=True) if left == right) / float(len(flat))


def _base_features(
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    vote_weight: float,
    vote_weights: list[float],
    confidence_values: list[float],
    program: dict[str, Any],
) -> dict[str, float]:
    q_mean = _as_float(candidate.get("q_mean"))
    max_other_vote = max(
        (weight for index, weight in enumerate(vote_weights) if index != candidate_index),
        default=0.0,
    )
    mean_confidence = _mean(confidence_values)
    features = {
        "vote_weight": vote_weight,
        "self_consistency_margin": vote_weight - max_other_vote,
        "vote_weight_rank_fraction": _rank_fraction(vote_weight, vote_weights),
        "cell_confidence_mean": q_mean,
        "cell_confidence_margin": q_mean - mean_confidence,
        "cell_confidence_rank_fraction": _rank_fraction(q_mean, confidence_values),
    }
    features.update(_grid_stats(candidate.get("grid")))
    features.update(_program_stats(program))
    return features


def _family_features(
    *,
    candidate: dict[str, Any],
    test_input: Any,
    vote_weight: float,
    candidate_count: int,
    signature_counts: Counter[str],
    shape_counts: Counter[str],
    palette_counts: Counter[str],
    shape_vote_weights: dict[str, float],
    palette_vote_weights: dict[str, float],
    modal_shapes: set[str],
    modal_palettes: set[str],
    modal_by_shape: dict[str, list[list[float]]],
) -> dict[str, float]:
    grid = candidate.get("grid")
    signature = _grid_signature(grid)
    shape = _shape_key(grid)
    palette = _palette_key(grid)
    height, width = _grid_shape(grid)
    input_height, input_width = _grid_shape(test_input)
    cells = max(1, height * width)
    input_cells = max(1, input_height * input_width)
    shape_weight = shape_vote_weights.get(shape, vote_weight)
    palette_weight = palette_vote_weights.get(palette, vote_weight)
    return {
        "modal_cell_agreement_frac": _modal_cell_agreement(grid, modal_by_shape.get(shape, [])),
        "grid_duplicate_count": float(signature_counts[signature]),
        "grid_duplicate_frac": signature_counts[signature] / float(candidate_count),
        "shape_family_count": float(shape_counts[shape]),
        "shape_family_frac": shape_counts[shape] / float(candidate_count),
        "shape_vote_frac": shape_weight,
        "is_modal_shape": 1.0 if shape in modal_shapes else 0.0,
        "palette_family_count": float(palette_counts[palette]),
        "palette_family_frac": palette_counts[palette] / float(candidate_count),
        "palette_vote_frac": palette_weight,
        "is_modal_palette": 1.0 if palette in modal_palettes else 0.0,
        "same_shape_as_input": 1.0 if (height, width) == (input_height, input_width) else 0.0,
        "area_delta_from_input_frac": (cells - input_cells) / float(input_cells),
    }


def _task_rows(
    *,
    source_id: str,
    entry_index: int,
    entry: dict[str, Any],
    program: dict[str, Any],
) -> list[ArcAggregatorRow]:
    raw_task_id = str(entry.get("task") or f"entry-{entry_index}")
    task_id = f"{source_id}:{raw_task_id}"
    candidates = entry.get("candidates")
    if not isinstance(candidates, list):
        return []
    valid_candidates = [
        (candidate_index, candidate)
        for candidate_index, candidate in enumerate(candidates)
        if isinstance(candidate, dict)
    ]
    if len(valid_candidates) < 2:
        return []

    votes = [_as_float(candidate.get("votes")) for _, candidate in valid_candidates]
    total_votes = sum(votes)
    vote_weights = [vote / total_votes if total_votes else 0.0 for vote in votes]
    confidence_values = [_as_float(candidate.get("q_mean")) for _, candidate in valid_candidates]
    grids = [candidate.get("grid") for _, candidate in valid_candidates]
    signature_counts = Counter(_grid_signature(grid) for grid in grids)
    shape_counts = Counter(_shape_key(grid) for grid in grids)
    palette_counts = Counter(_palette_key(grid) for grid in grids)
    shape_vote_weights: dict[str, float] = {}
    palette_vote_weights: dict[str, float] = {}
    for (_, candidate), vote_weight in zip(valid_candidates, vote_weights, strict=True):
        shape = _shape_key(candidate.get("grid"))
        palette = _palette_key(candidate.get("grid"))
        shape_vote_weights[shape] = shape_vote_weights.get(shape, 0.0) + vote_weight
        palette_vote_weights[palette] = palette_vote_weights.get(palette, 0.0) + vote_weight
    max_shape_count = max(shape_counts.values(), default=0)
    max_palette_count = max(palette_counts.values(), default=0)
    modal_shapes = {shape for shape, count in shape_counts.items() if count == max_shape_count}
    modal_palettes = {
        palette for palette, count in palette_counts.items() if count == max_palette_count
    }
    modal_by_shape = _modal_grids_by_shape(grids)
    pred_grid = program.get("pred_grid")
    base_rows = []
    for local_index, (candidate_index, candidate) in enumerate(valid_candidates):
        base = _base_features(
            candidate,
            candidate_index=local_index,
            vote_weight=vote_weights[local_index],
            vote_weights=vote_weights,
            confidence_values=confidence_values,
            program=program,
        )
        base_rows.append((candidate_index, candidate, vote_weights[local_index], base))

    vote_mean = _mean([base["vote_weight"] for _, _, _, base in base_rows])
    confidence_mean = _mean([base["cell_confidence_mean"] for _, _, _, base in base_rows])
    entropy_mean = _mean([base["grid_entropy"] for _, _, _, base in base_rows])
    cells_mean = _mean([base["grid_cells"] for _, _, _, base in base_rows])
    vote_std = _std([base["vote_weight"] for _, _, _, base in base_rows], vote_mean) or 1.0
    confidence_std = _std(
        [base["cell_confidence_mean"] for _, _, _, base in base_rows], confidence_mean
    ) or 1.0
    entropy_std = _std([base["grid_entropy"] for _, _, _, base in base_rows], entropy_mean) or 1.0
    cells_std = _std([base["grid_cells"] for _, _, _, base in base_rows], cells_mean) or 1.0
    candidate_count = len(base_rows)

    rows: list[ArcAggregatorRow] = []
    for candidate_index, candidate, vote_weight, base in base_rows:
        cross = {
            "set_candidate_count": float(candidate_count),
            "set_vote_mean": vote_mean,
            "set_vote_max": max(base["vote_weight"] for _, _, _, base in base_rows),
            "set_vote_std": _std([base["vote_weight"] for _, _, _, base in base_rows], vote_mean),
            "set_confidence_mean": confidence_mean,
            "set_confidence_max": max(base["cell_confidence_mean"] for _, _, _, base in base_rows),
            "set_confidence_std": _std(
                [base["cell_confidence_mean"] for _, _, _, base in base_rows], confidence_mean
            ),
            "set_entropy_mean": entropy_mean,
            "set_entropy_max": max(base["grid_entropy"] for _, _, _, base in base_rows),
            "set_entropy_std": _std(
                [base["grid_entropy"] for _, _, _, base in base_rows], entropy_mean
            ),
            "set_cells_mean": cells_mean,
            "set_cells_max": max(base["grid_cells"] for _, _, _, base in base_rows),
            "set_cells_std": _std([base["grid_cells"] for _, _, _, base in base_rows], cells_mean),
            "vote_weight_zscore": (base["vote_weight"] - vote_mean) / vote_std,
            "cell_confidence_zscore": (
                base["cell_confidence_mean"] - confidence_mean
            )
            / confidence_std,
            "grid_entropy_zscore": (base["grid_entropy"] - entropy_mean) / entropy_std,
            "grid_cells_zscore": (base["grid_cells"] - cells_mean) / cells_std,
        }
        cross.update(
            _family_features(
                candidate=candidate,
                test_input=entry.get("test_input"),
                vote_weight=vote_weight,
                candidate_count=candidate_count,
                signature_counts=signature_counts,
                shape_counts=shape_counts,
                palette_counts=palette_counts,
                shape_vote_weights=shape_vote_weights,
                palette_vote_weights=palette_vote_weights,
                modal_shapes=modal_shapes,
                modal_palettes=modal_palettes,
                modal_by_shape=modal_by_shape,
            )
        )
        features = {**base, **cross}
        raw_flag = candidate.get("correct")
        rows.append(
            ArcAggregatorRow(
                source_id=source_id,
                task_id=task_id,
                candidate_id=f"{task_id}::candidate{candidate_index}",
                candidate_index=candidate_index,
                vote_weight=vote_weight,
                correct=_grid_equal(candidate.get("grid"), pred_grid),
                features={name: float(features[name]) for name in FEATURE_NAMES},
                raw_candidate_correct_flag=raw_flag if isinstance(raw_flag, bool) else None,
            )
        )
    return rows


def _import_detector_module() -> Any:  # pragma: no cover - environment import is integration-level
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts import exp_verifier_detector_auroc as detector
    except Exception as exc:  # pragma: no cover
        raise BlockedRun("blocked_arc_gap4_pools_missing") from exc
    return detector


def _load_gap_payloads(
    repo_root: Path,
    spec: PoolSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int] | None:
    pool_path = repo_root / spec.pool_rel
    programs_path = repo_root / spec.programs_rel
    if not pool_path.exists() or not programs_path.exists():
        if spec.required:
            raise BlockedRun("blocked_arc_gap4_pools_missing")
        return None
    detector = _import_detector_module()
    try:
        detector_rows = detector.load_arc_rows(pool_path, programs_path)
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            pool = json.load(handle)
        programs_payload = json.loads(programs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if spec.required:
            raise BlockedRun("blocked_arc_gap4_pools_missing") from exc
        return None
    entries = pool.get("entries") if isinstance(pool, dict) else None
    programs = programs_payload.get("programs") if isinstance(programs_payload, dict) else None
    if not isinstance(entries, list) or not isinstance(programs, list):
        if spec.required:
            raise BlockedRun("blocked_arc_gap4_pools_missing")
        return None
    return entries, programs, len(detector_rows)


def load_labeled_arc_pool(
    repo_root: Path | str = Path("."), pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS
) -> ArcAggregatorCorpus:
    """SCENARIO-VERIFY-4231: build task-grouped labels from GAP-4 pred_grid equality."""

    root = Path(repo_root)
    rows: list[ArcAggregatorRow] = []
    source_paths: list[Path] = []
    source_sha256: dict[str, str] = {}
    skipped_optional_pools: list[str] = []
    raw_candidate_n = 0
    detector_row_n = 0
    for spec in pool_specs:
        loaded = _load_gap_payloads(root, spec)
        if loaded is None:
            skipped_optional_pools.append(spec.source_id)
            continue
        entries, programs, detector_count = loaded
        pool_path = (root / spec.pool_rel).resolve()
        programs_path = (root / spec.programs_rel).resolve()
        source_paths.extend([pool_path, programs_path])
        source_sha256[str(pool_path)] = _sha256_file(pool_path)
        source_sha256[str(programs_path)] = _sha256_file(programs_path)
        detector_row_n += detector_count
        by_entry = {
            int(program.get("entry_i", index)): program
            for index, program in enumerate(programs)
            if isinstance(program, dict)
        }
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            candidates = entry.get("candidates")
            if isinstance(candidates, list):
                raw_candidate_n += sum(1 for candidate in candidates if isinstance(candidate, dict))
            rows.extend(
                _task_rows(
                    source_id=spec.source_id,
                    entry_index=entry_index,
                    entry=entry,
                    program=by_entry.get(entry_index, {}),
                )
            )

    grouped: dict[str, list[ArcAggregatorRow]] = {}
    for row in rows:
        grouped.setdefault(row.task_id, []).append(row)
    wrong_majority_n = 0
    for task_rows in grouped.values():
        vote_winner = max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
        wrong_majority_n += int(any(row.correct for row in task_rows) and not vote_winner.correct)
    return ArcAggregatorCorpus(
        rows=rows,
        source_paths=source_paths,
        source_sha256=source_sha256,
        held_out_task_n=len(grouped),
        wrong_majority_n=wrong_majority_n,
        raw_candidate_n=raw_candidate_n,
        detector_row_n=detector_row_n,
        skipped_optional_pools=skipped_optional_pools,
    )


def accepted_rejected_counts(rows: list[ArcAggregatorRow]) -> dict[str, int]:
    accepted = sum(row.correct for row in rows)
    rejected = len(rows) - accepted
    return {"accepted": int(accepted), "rejected": int(rejected), "total": len(rows)}


def _feature_vector(row: ArcAggregatorRow) -> list[float]:
    return [float(row.features[name]) for name in FEATURE_NAMES]


def _split_task_folds(rows: list[ArcAggregatorRow], random_seed: int, n_folds: int) -> list[set[str]]:
    task_ids = sorted({row.task_id for row in rows})
    fold_count = max(2, min(int(n_folds), len(task_ids)))
    shuffled = task_ids[:]
    random.Random(random_seed).shuffle(shuffled)
    return [set(shuffled[index::fold_count]) for index in range(fold_count)]


def _standardizer(rows: list[ArcAggregatorRow]) -> tuple[list[float], list[float]]:
    vectors = [_feature_vector(row) for row in rows]
    means = [
        sum(vector[index] for vector in vectors) / float(len(vectors))
        for index in range(len(FEATURE_NAMES))
    ]
    scales: list[float] = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / float(len(vectors))
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _standardized_vector(features: dict[str, float], means: list[float], scales: list[float]) -> list[float]:
    return [
        (float(features.get(name, 0.0)) - means[index]) / scales[index]
        for index, name in enumerate(FEATURE_NAMES)
    ]


def _fit_isotonic(raw_scores: list[float], labels: list[bool]) -> dict[str, list[float]]:
    if len(set(raw_scores)) < 2 or len(set(labels)) < 2:
        base = sum(labels) / float(len(labels)) if labels else 0.0
        return {"x": [0.0, 1.0], "y": [base, base]}
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_scores, [int(label) for label in labels])
    return {
        "x": [float(value) for value in calibrator.X_thresholds_],
        "y": [float(value) for value in calibrator.y_thresholds_],
    }


def _apply_isotonic(value: float, calibration: dict[str, list[float]]) -> float:
    xs = [float(item) for item in calibration.get("x", [])]
    ys = [float(item) for item in calibration.get("y", [])]
    if not xs or not ys or len(xs) != len(ys):
        return value
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            left_x, right_x = xs[index - 1], xs[index]
            left_y, right_y = ys[index - 1], ys[index]
            if right_x == left_x:  # pragma: no cover - sorted thresholds make this unreachable
                return right_y
            frac = (value - left_x) / (right_x - left_x)
            return left_y + frac * (right_y - left_y)
    return ys[-1]  # pragma: no cover - value>=xs[-1] returns before the loop


def _constant_aggregator(rows: list[ArcAggregatorRow]) -> dict[str, Any]:
    counts = accepted_rejected_counts(rows)
    base_rate = counts["accepted"] / float(counts["total"]) if counts["total"] else 0.0
    return {
        "model_type": "constant_score",
        "feature_names": list(FEATURE_NAMES),
        "constant_score": float(base_rate),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _train_aggregator(rows: list[ArcAggregatorRow], random_seed: int) -> dict[str, Any]:
    means, scales = _standardizer(rows)
    model = LogisticRegression(
        random_state=random_seed,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
    )
    labels = [row.correct for row in rows]
    x_train = [_standardized_vector(row.features, means, scales) for row in rows]
    model.fit(x_train, [int(label) for label in labels])
    raw_scores = [float(value) for value in model.predict_proba(x_train)[:, 1]]
    return {
        "model_type": "standardized_logistic_regression_isotonic_calibrated",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
        "isotonic_calibration": _fit_isotonic(raw_scores, labels),
    }


def score_with_aggregator(aggregator: dict[str, Any], features: dict[str, float]) -> float:
    model_type = aggregator.get("model_type")
    if model_type == "constant_score":
        return float(aggregator.get("constant_score", 0.0))
    if model_type != "standardized_logistic_regression_isotonic_calibrated":
        raise ValueError("unknown aggregator model_type")
    means = [float(value) for value in aggregator["feature_means"]]
    scales = [float(value) for value in aggregator["feature_scales"]]
    values = _standardized_vector(features, means, scales)
    logit = float(aggregator["intercept"]) + sum(
        float(weight) * value for weight, value in zip(aggregator["coefficients"], values, strict=True)
    )
    raw_score = _sigmoid(logit)
    calibrated = _apply_isotonic(raw_score, aggregator.get("isotonic_calibration", {}))
    return 0.99 * calibrated + 0.01 * raw_score


def _auroc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / float(len(positives) * len(negatives))


def _bootstrap_auroc_ci95(
    labels: list[bool], scores: list[float], random_seed: int, bootstrap_n: int = BOOTSTRAP_N
) -> tuple[float, float]:
    if len(set(labels)) < 2 or not scores:
        return (0.0, 0.0)
    rng = random.Random(random_seed)
    samples: list[float] = []
    n = len(labels)
    for _ in range(bootstrap_n):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        sample_scores = [scores[index] for index in indices]
        samples.append(_auroc(sample_labels, sample_scores))
    if not samples:
        point = _auroc(labels, scores)
        return point, point
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return samples[low_index], samples[high_index]


def train_oof_aggregator(
    rows: list[ArcAggregatorRow],
    *,
    random_seed: int = RANDOM_SEED,
    n_folds: int = 5,
    bootstrap_n: int = BOOTSTRAP_N,
) -> OOFReport:
    """SCENARIO-VERIFY-4231: train on non-held-out ARC tasks and score OOF."""

    counts = accepted_rejected_counts(rows)
    task_ids = sorted({row.task_id for row in rows})
    labels = [row.correct for row in rows]
    if counts["accepted"] < 2 or counts["rejected"] < 2 or len(task_ids) < 2:
        score = counts["accepted"] / float(counts["total"]) if counts["total"] else 0.0
        scores = [score for _ in rows]
        return OOFReport(
            oracle_distinct_auroc=_auroc(labels, scores),
            oracle_distinct_auroc_ci95=_bootstrap_auroc_ci95(
                labels, scores, random_seed, bootstrap_n
            ),
            fold_task_ids=[task_ids],
            oof_rows=[
                OOFRow(row.task_id, row.candidate_id, row.correct, scores[index], 0, tuple())
                for index, row in enumerate(rows)
            ],
            final_aggregator=_constant_aggregator(rows),
        )

    folds = _split_task_folds(rows, random_seed, n_folds)
    oof_scores_by_id: dict[str, float] = {}
    oof_rows: list[OOFRow] = []
    for fold, heldout_task_ids in enumerate(folds):
        train_rows = [row for row in rows if row.task_id not in heldout_task_ids]
        test_rows = [row for row in rows if row.task_id in heldout_task_ids]
        train_counts = accepted_rejected_counts(train_rows)
        if train_counts["accepted"] < 1 or train_counts["rejected"] < 1:
            aggregator = _constant_aggregator(train_rows)
        else:
            aggregator = _train_aggregator(train_rows, random_seed + fold)
        train_task_ids = tuple(sorted({row.task_id for row in train_rows}))
        for row in test_rows:
            score = score_with_aggregator(aggregator, row.features)
            oof_scores_by_id[row.candidate_id] = score
            oof_rows.append(
                OOFRow(
                    task_id=row.task_id,
                    candidate_id=row.candidate_id,
                    correct=row.correct,
                    score=score,
                    fold=fold,
                    train_task_ids=train_task_ids,
                )
            )
    oof_scores = [oof_scores_by_id[row.candidate_id] for row in rows]
    final_aggregator = _train_aggregator(rows, random_seed)
    return OOFReport(
        oracle_distinct_auroc=_auroc(labels, oof_scores),
        oracle_distinct_auroc_ci95=_bootstrap_auroc_ci95(labels, oof_scores, random_seed, bootstrap_n),
        fold_task_ids=[sorted(fold) for fold in folds],
        oof_rows=oof_rows,
        final_aggregator=final_aggregator,
    )


def reproducibility_checksum(
    corpus: ArcAggregatorCorpus, report: OOFReport, *, random_seed: int = RANDOM_SEED
) -> str:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "fold_task_ids": report.fold_task_ids,
        "random_seed": random_seed,
        "source_sha256": corpus.source_sha256,
        "rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "features": row.features,
                "task_id": row.task_id,
            }
            for row in corpus.rows
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _model_specs(status: str) -> dict[str, Any]:
    return {
        "architecture": "cross_candidate_augmented_calibrated_logistic_aggregator",
        "feature_set": list(FEATURE_NAMES),
        "base_feature_set": list(BASE_FEATURE_NAMES),
        "cross_candidate_feature_set": list(CROSS_FEATURE_NAMES),
        "cross_candidate_conditioning": (
            "explicit per-task set statistics: set mean/max/std, within-set ranks, "
            "modal-grid agreement, duplicate counts, shape and palette family indicators"
        ),
        "imbalance_loss": "class_weight_balanced_logistic_loss",
        "calibration": "train_fold_isotonic_on_raw_probabilities",
        "training_recipe": "task_held_out_oof_candidate_detection",
        "baseline_auroc_391": BASELINE_AUROC_391,
        "status": status,
    }


def _no_gain_reason(
    counts: dict[str, int],
    corpus: ArcAggregatorCorpus,
    auroc: float,
    baseline_auroc: float,
) -> str | None:
    if counts["accepted"] < SPARSE_POSITIVE_THRESHOLD:
        return "too_few_positives_after_growth"
    if corpus.wrong_majority_n < WRONG_MAJORITY_TARGET:
        return "too_few_wrong_majority_tasks_after_growth"
    if auroc <= baseline_auroc + BASELINE_IMPROVEMENT_EPSILON:
        return "no_gain_over_391_logistic_baseline"
    return None


def persist_aggregator(
    path: Path,
    aggregator: dict[str, Any],
    *,
    checksum: str,
    counts: dict[str, int],
    corpus: ArcAggregatorCorpus,
    report: OOFReport,
    random_seed: int,
    no_learnable_gain_reason: str | None,
) -> None:
    payload = {
        **aggregator,
        "accepted_rejected_n": counts,
        "fold_task_ids": report.fold_task_ids,
        "held_out_task_n": corpus.held_out_task_n,
        "model_specs": _model_specs("trained"),
        "no_learnable_gain_reason": no_learnable_gain_reason,
        "oof_rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "fold": row.fold,
                "score": _round_metric(row.score),
                "task_id": row.task_id,
                "train_task_ids": list(row.train_task_ids),
            }
            for row in report.oof_rows
        ],
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "source_paths": [str(path) for path in corpus.source_paths],
        "spec_refs": SPEC_REFS,
        "verifier_is_oracle": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_aggregator(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("aggregator artifact must be a JSON object")
    return payload


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "aggregator_trained": False,
        "oracle_distinct_auroc": 0.0,
        "oracle_distinct_auroc_ci95": [0.0, 0.0],
        "held_out_task_n": 0,
        "wrong_majority_n": 0,
        "learned_verifier_path": "",
        "verifier_is_oracle": False,
        "model_specs": _model_specs("blocked"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": {"accepted": 0, "rejected": 0, "total": 0},
        "positive_candidate_n": 0,
        "positive_sparsity_flag": False,
        "duration_s": round(duration_s, 6),
    }


def _complete_artifact(
    corpus: ArcAggregatorCorpus,
    report: OOFReport,
    *,
    checksum: str,
    counts: dict[str, int],
    aggregator_path: Path,
    random_seed: int,
    duration_s: float,
    no_learnable_gain_reason: str | None,
) -> dict[str, Any]:
    auroc = _round_metric(report.oracle_distinct_auroc)
    ci95 = [_round_metric(value) for value in report.oracle_distinct_auroc_ci95]
    if no_learnable_gain_reason:
        verdict = f"complete_oracle_distinct_arc_aggregator_no_learnable_gain_auroc{auroc:.4f}"
    else:
        verdict = f"complete: oracle_distinct_arc_aggregator_trained_auroc_{auroc:.4f}"
    return {
        "experiment": "experiment_4231_oracle_distinct_arc_aggregator_build",
        "schema": "carnot.oracle_distinct_arc_aggregator_4231.v1",
        "honest_verdict": verdict,
        "aggregator_trained": True,
        "oracle_distinct_auroc": auroc,
        "oracle_distinct_auroc_ci95": ci95,
        "held_out_task_n": int(corpus.held_out_task_n),
        "wrong_majority_n": int(corpus.wrong_majority_n),
        "learned_verifier_path": str(aggregator_path),
        "verifier_is_oracle": False,
        "model_specs": _model_specs("trained"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": counts,
        "positive_candidate_n": counts["accepted"],
        "positive_sparsity_flag": counts["accepted"] < SPARSE_POSITIVE_THRESHOLD,
        "wrong_majority_target_met": corpus.wrong_majority_n >= WRONG_MAJORITY_TARGET,
        "baseline_auroc_391": BASELINE_AUROC_391,
        "no_learnable_gain_reason": no_learnable_gain_reason,
        "raw_candidate_n": corpus.raw_candidate_n,
        "detector_row_n": corpus.detector_row_n,
        "candidate_pool_sources": [str(path) for path in corpus.source_paths],
        "skipped_optional_pools": corpus.skipped_optional_pools,
        "feature_names": list(FEATURE_NAMES),
        "oof_folds": len(report.fold_task_ids),
        "label_source": "candidate_grid_equals_gap4_induced_pred_grid",
        "inference_substrate": "cached_gap_arc_pool_oof_cross_candidate_calibrated_aggregator",
        "duration_s": round(duration_s, 6),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    terminal_prefixes = ("complete:", "success:", "passed:", "shipped:", "complete_", "blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(terminal_prefixes):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact["aggregator_trained"], bool):
        raise ValueError("aggregator_trained must be a bare bool")
    if not isinstance(artifact["oracle_distinct_auroc"], float):
        raise ValueError("oracle_distinct_auroc must be a bare float")
    if not isinstance(artifact["held_out_task_n"], int):
        raise ValueError("held_out_task_n must be a bare int")
    if not isinstance(artifact["wrong_majority_n"], int):
        raise ValueError("wrong_majority_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4231")
    if artifact["aggregator_trained"] and not Path(artifact["learned_verifier_path"]).exists():
        raise ValueError("trained artifacts require a persisted aggregator")


def _blocked_checksum(repo_root: Path | str) -> str:
    root = Path(repo_root)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "sources": {
            str(root / spec.pool_rel): _sha256_file(root / spec.pool_rel)
            if (root / spec.pool_rel).exists()
            else ""
            for spec in DEFAULT_POOL_SPECS
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    n_folds: int = 5,
    bootstrap_n: int = BOOTSTRAP_N,
    baseline_auroc: float = BASELINE_AUROC_391,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        corpus = load_labeled_arc_pool(root)
        report = train_oof_aggregator(
            corpus.rows,
            random_seed=random_seed,
            n_folds=n_folds,
            bootstrap_n=bootstrap_n,
        )
        counts = accepted_rejected_counts(corpus.rows)
        checksum = reproducibility_checksum(corpus, report, random_seed=random_seed)
        no_gain_reason = _no_gain_reason(
            counts, corpus, report.oracle_distinct_auroc, baseline_auroc
        )
        aggregator_path = (root / AGGREGATOR_REL).resolve()
        persist_aggregator(
            aggregator_path,
            report.final_aggregator,
            checksum=checksum,
            counts=counts,
            corpus=corpus,
            report=report,
            random_seed=random_seed,
            no_learnable_gain_reason=no_gain_reason,
        )
        artifact = _complete_artifact(
            corpus,
            report,
            checksum=checksum,
            counts=counts,
            aggregator_path=aggregator_path,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
            no_learnable_gain_reason=no_gain_reason,
        )
    except BlockedRun as blocked:
        artifact = _blocked_artifact(
            blocked.reason,
            random_seed=random_seed,
            checksum=_blocked_checksum(root),
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
