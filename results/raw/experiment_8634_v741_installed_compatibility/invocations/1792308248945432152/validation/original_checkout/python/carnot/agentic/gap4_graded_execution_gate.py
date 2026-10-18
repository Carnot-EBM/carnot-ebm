"""Production-safe GAP-4 graded execution gate.

The gate is intentionally narrow: a saved program may promote a candidate only
when it is demo-perfect and its executed prediction is within a small normalized
Hamming radius of a pool candidate. A vote-aware guard blocks extreme
high-confidence vote leaders from being demoted by that promotion.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TAU = 0.005
DEFAULT_BAND_TAU = 0.02
DEFAULT_HIGH_VOTE_GUARD_THRESHOLD = 900


def _as_rows(grid: Any) -> list[list[Any]]:
    if grid is None:
        return []
    return [list(row) for row in grid]


def _grid_shape(grid: Any) -> tuple[int, int]:
    rows = _as_rows(grid)
    return len(rows), len(rows[0]) if rows else 0


def _grid_size(grid: Any) -> int:
    rows, cols = _grid_shape(grid)
    return rows * cols


def normalized_hamming(candidate: Any, prediction: Any) -> float:
    """Return normalized Hamming distance, with shape mismatches always worse.

    Same-shape grids are scored as the fraction of disagreeing cells. Shape
    mismatches return `1 + relative_size_delta`, so every same-shape candidate
    beats every shape-mismatched candidate under the graded gate.
    """
    c_rows = _as_rows(candidate)
    p_rows = _as_rows(prediction)
    c_shape = _grid_shape(c_rows)
    p_shape = _grid_shape(p_rows)
    if c_shape == p_shape:
        total = max(1, c_shape[0] * c_shape[1])
        diff = sum(
            1
            for r in range(c_shape[0])
            for col in range(c_shape[1])
            if c_rows[r][col] != p_rows[r][col]
        )
        return diff / total
    c_size = _grid_size(c_rows)
    p_size = _grid_size(p_rows)
    return 1.0 + abs(c_size - p_size) / max(1, c_size, p_size)


def select_guarded_graded_candidate(
    candidates: list[dict[str, Any]],
    *,
    prediction: Any,
    demo_fit: float | int | None,
    task_id: str = "",
    tau: float = DEFAULT_TAU,
    high_vote_guard_threshold: int | float | None = DEFAULT_HIGH_VOTE_GUARD_THRESHOLD,
    agreement_confidence_label: bool = False,
) -> dict[str, Any]:
    """Choose the guarded graded-gate candidate for one ARC pool entry.

    Selection uses only demo fit, executed prediction, candidate grids, and
    candidate vote counts. `correct` labels, when present in replay fixtures, are
    ignored by the selector and used only by callers for post-hoc scoring.
    """
    result = {
        "task_id": task_id,
        "gate_fired": False,
        "selected_index": None,
        "would_select_index": None,
        "min_hamming": None,
        "guard_blocked": False,
        "reason": "not_evaluated",
        "agreement_confidence_label": bool(agreement_confidence_label),
    }
    if float(demo_fit or 0.0) != 1.0:
        result["reason"] = "demo_fit_not_exact"
        return result
    if prediction is None:
        result["reason"] = "prediction_missing"
        return result
    if not candidates:
        result["reason"] = "candidate_pool_empty"
        return result

    distances = [normalized_hamming(c.get("grid"), prediction) for c in candidates]
    min_index = min(range(len(distances)), key=lambda i: distances[i])
    min_hamming = distances[min_index]
    result["would_select_index"] = min_index
    result["min_hamming"] = min_hamming
    if min_hamming > tau:
        result["reason"] = "outside_tau"
        return result

    top_vote_index = max(range(len(candidates)), key=lambda i: candidates[i].get("votes", 0))
    top_votes = candidates[top_vote_index].get("votes", 0)
    if (
        high_vote_guard_threshold is not None
        and min_index != top_vote_index
        and top_votes >= high_vote_guard_threshold
    ):
        result["guard_blocked"] = True
        result["reason"] = "vote_aware_guard_high_vote_leader"
        return result

    result["gate_fired"] = True
    result["selected_index"] = min_index
    result["reason"] = "promoted"
    return result


def vote_rank_indices(candidates: list[dict[str, Any]]) -> list[int]:
    """Return candidate indices in TRM vote order."""
    return sorted(range(len(candidates)), key=lambda i: (-candidates[i].get("votes", 0), i))


def gated_rank_indices(candidates: list[dict[str, Any]], selected_index: int | None) -> list[int]:
    """Return vote order with one guarded gate promotion, if any."""
    return sorted(
        range(len(candidates)),
        key=lambda i: (0 if selected_index is not None and i == selected_index else 1,
                       -candidates[i].get("votes", 0),
                       i),
    )


def pass_at_k(entries: list[dict[str, Any]], rankings: list[list[int]], k: int) -> float:
    """Score pass@k from candidate rankings using post-hoc `correct` labels."""
    hits = 0
    for entry, order in zip(entries, rankings, strict=True):
        candidates = entry["candidates"]
        hits += int(any(candidates[i].get("correct", False) for i in order[:k]))
    return round(hits / max(1, len(entries)), 4)


def hit_indices(entries: list[dict[str, Any]], rankings: list[list[int]], k: int) -> set[int]:
    """Return entry indices whose ranked top-k contains a gold candidate."""
    out: set[int] = set()
    for i, (entry, order) in enumerate(zip(entries, rankings, strict=True)):
        candidates = entry["candidates"]
        if any(candidates[j].get("correct", False) for j in order[:k]):
            out.add(i)
    return out


def non_exact_band_precision(
    entries: list[dict[str, Any]],
    programs: list[dict[str, Any]],
    *,
    band_tau: float = DEFAULT_BAND_TAU,
) -> dict[str, Any]:
    """Measure precision for non-exact near candidates with `0 < hamming <= band_tau`.

    This is diagnostic only. It intentionally excludes exact matches because the
    exact-match gate already established the ARC-1 positive; the band measures
    whether graded relaxation itself is trustworthy.
    """
    total = 0
    correct = 0
    for entry, program in zip(entries, programs, strict=True):
        if float(program.get("demo_fit", 0.0)) != 1.0 or program.get("pred_grid") is None:
            continue
        distances = [
            normalized_hamming(candidate.get("grid"), program["pred_grid"])
            for candidate in entry["candidates"]
        ]
        if not distances:
            continue
        min_distance = min(distances)
        if 0.0 < min_distance <= band_tau:
            total += 1
            min_index = distances.index(min_distance)
            correct += int(entry["candidates"][min_index].get("correct", False))
    return {
        "tau": band_tau,
        "definition": "demo-perfect entries whose closest candidate has 0 < min_hamming <= tau",
        "correct": correct,
        "total": total,
        "precision": round(correct / total, 4) if total else None,
    }
