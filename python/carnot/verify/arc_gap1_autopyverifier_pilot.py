"""Exp 5205: AutoPyVerifier-inspired GAP-1 set search for ARC orientation.

Spec refs: REQ-VERIFY-5205, SCENARIO-VERIFY-5205.

This module deliberately borrows only the cheap set-search idea from
AutoPyVerifier. It does not synthesize verifier code with an LLM. Instead, a
small deterministic library of spatial discriminators is searched over a cached
ARC square-transpose distractor pool. The scoring functions see only train
pairs plus a candidate grid, so gold labels are used only for offline evaluation
of the searched subset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time
from typing import Any


JsonDict = dict[str, Any]
Grid = Sequence[Sequence[int]]
FeatureCounter = Counter[tuple[Any, ...]]
FeatureFn = Callable[[Grid], FeatureCounter]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5205_autopyverifier_gap1_pilot_v476"
EXPERIMENT_ID = 5205
SCHEMA = "carnot.arc_gap1_autopyverifier_pilot_5205.v1"
RESULT_RELATIVE_PATH = "results/experiment_5205_autopyverifier_gap1_pilot_v476.json"
SOURCE_ARTIFACT_RELATIVE_PATH = "results/arc_grid_verifier_invariants_v2.json"
VERIFIER_GAPS_RELATIVE_PATH = "ops/verifier_gaps.md"
DEFAULT_ARC_ROOT = Path("/home/ianblenke/trm_src/kaggle/combined")
RANDOM_SEED = 5205
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
REFUTED_DIRECTIONAL = "directional_adjacency_refuted_20260609"
SPEC_REFS = ("REQ-VERIFY-5205", "SCENARIO-VERIFY-5205")

FIELD_PRINCIPLES: dict[str, str] = {
    "candidate_discriminators_authored": (
        "Names every hand-authored cheap spatial discriminator searched; this replaces LLM synthesis "
        "with deterministic candidate authoring."
    ),
    "best_subset_found": "The searched discriminator subset selected on grouped training rows before final evaluation.",
    "pass_at_2_baseline_always_on_only": (
        "Candidate-pool pass@2 using only object_count and palette_histogram_shape, the always-on "
        "transpose-invariant baseline."
    ),
    "pass_at_2_best_subset": "Candidate-pool pass@2 after adding the selected searched discriminator set.",
    "transpose_misvotes_captured": (
        "Human-readable count of dimension-preserving transpose cases where the selected set strictly "
        "prefers gold over the transposed candidate."
    ),
    "verifier_is_oracle": "Gold labels are used only for offline subset evaluation, never inside candidate scoring.",
    "random_seed": "Deterministic ARC distractor reconstruction, grouped split, subset tie-breaks, and checksum.",
    "reproducibility_checksum": "SHA-256 checksum over the terminal artifact with this field blanked.",
    "inference_substrate": (
        "This pilot evaluates cheap discriminators against a cached distractor pool -- no LLM call is made, "
        "unlike AutoPyVerifier's own LLM-synthesis step, which this pilot deliberately substitutes with "
        "hand-authored candidates for cost/determinism."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_, and must state plainly whether the set-search "
        "approach beats the always-on baseline and the already-refuted single-invariant approach, or whether "
        "GAP-1 remains open under this attempt too."
    ),
}
REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class CandidateGrid:
    candidate_id: str
    kind: str
    grid: Grid
    correct: bool


@dataclass(frozen=True)
class TaskPool:
    task_id: str
    train_pairs: tuple[Mapping[str, Grid], ...]
    test_input: Grid
    candidates: tuple[CandidateGrid, ...]


@dataclass(frozen=True)
class Discriminator:
    name: str
    description: str
    transpose_sensitive: bool
    feature_fn: FeatureFn


def _grid_tuple(grid: Grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def dims(grid: Grid) -> tuple[int, int]:
    g = _grid_tuple(grid)
    return (len(g), len(g[0]) if g else 0)


def transpose_grid(grid: Grid) -> list[list[int]]:
    h, w = dims(grid)
    g = _grid_tuple(grid)
    return [[g[r][c] for r in range(h)] for c in range(w)]


def rotate_180_grid(grid: Grid) -> list[list[int]]:
    return [list(reversed(row)) for row in reversed(_grid_tuple(grid))]


def _colors(grid: Grid) -> set[int]:
    return {cell for row in _grid_tuple(grid) for cell in row}


def _palette_counts(grid: Grid) -> Counter[int]:
    return Counter(cell for row in _grid_tuple(grid) for cell in row)


def _bg(grid: Grid) -> int:
    counts = _palette_counts(grid)
    return counts.most_common(1)[0][0] if counts else 0


def _copy_grid(grid: Grid) -> list[list[int]]:
    return [list(row) for row in _grid_tuple(grid)]


def _rand_grid(h: int, w: int, rng: random.Random, palette: set[int]) -> list[list[int]]:
    colors = sorted(palette) or [0]
    return [[rng.choice(colors) for _c in range(w)] for _r in range(h)]


def _perturb(grid: Grid, rng: random.Random, frac: float = 0.12) -> list[list[int]]:
    out = _copy_grid(grid)
    h, w = dims(out)
    colors = sorted(_colors(out)) or [0]
    for _ in range(max(1, int(frac * h * w))):
        out[rng.randrange(h)][rng.randrange(w)] = rng.choice(colors)
    return out


def _color_swap(grid: Grid, rng: random.Random) -> list[list[int]] | None:
    colors = sorted(_colors(grid))
    if len(colors) < 2:
        return None
    a, b = rng.sample(colors, 2)
    return [[b if cell == a else a if cell == b else cell for cell in row] for row in _grid_tuple(grid)]


def _wrong_dim(grid: Grid, rng: random.Random) -> list[list[int]]:
    h, w = dims(grid)
    nh = max(1, h + rng.choice([-1, 1, 2]))
    nw = max(1, w + rng.choice([-1, 1, 2]))
    source = _grid_tuple(grid)
    colors = sorted(_colors(grid)) or [0]
    return [[source[r][c] if r < h and c < w else colors[0] for c in range(nw)] for r in range(nh)]


def _distractors(
    gold: Grid,
    test_input: Grid,
    all_golds: Sequence[Grid],
    rng: random.Random,
) -> dict[str, Grid]:
    h, w = dims(gold)
    candidates: dict[str, Grid] = {
        "copy_input": _copy_grid(test_input),
        "wrong_task_gold": all_golds[rng.randrange(len(all_golds))],
        "random": _rand_grid(h, w, rng, set(range(10))),
        "blank": [[0] * w for _ in range(h)],
        "perturbed_gold": _perturb(gold, rng),
        "wrong_dim_gold": _wrong_dim(gold, rng),
    }
    swapped = _color_swap(gold, rng)
    if swapped is not None:
        candidates["color_swap_gold"] = swapped
    transposed = transpose_grid(gold)
    if transposed != _copy_grid(gold):
        candidates["transposed_gold"] = transposed
    return {kind: grid for kind, grid in candidates.items() if _grid_tuple(grid) != _grid_tuple(gold)}


def _objects(grid: Grid) -> list[tuple[int, int]]:
    g = _grid_tuple(grid)
    h, w = dims(g)
    background = _bg(g)
    seen = [[False] * w for _ in range(h)]
    out: list[tuple[int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0]:
                continue
            color = g[r0][c0]
            seen[r0][c0] = True
            if color == background:
                continue
            stack = [(r0, c0)]
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            out.append((color, size))
    return out


def _distribution(counter: Counter[Any]) -> dict[Any, float]:
    total = sum(counter.values())
    return {key: value / total for key, value in counter.items()} if total else {}


def _l1_half(left: Mapping[Any, float], right: Mapping[Any, float]) -> float:
    keys = set(left) | set(right)
    return sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2.0


def _avg_feature_distribution(train_pairs: Sequence[Mapping[str, Grid]], feature_fn: FeatureFn) -> dict[Any, float]:
    dists = [_distribution(feature_fn(pair["output"])) for pair in train_pairs]
    keys = set().union(*(dist.keys() for dist in dists)) if dists else set()
    return {key: sum(dist.get(key, 0.0) for dist in dists) / len(dists) for key in keys} if dists else {}


def _feature_score(grid: Grid, train_pairs: Sequence[Mapping[str, Grid]], feature_fn: FeatureFn) -> float:
    signature = _avg_feature_distribution(train_pairs, feature_fn)
    return _l1_half(_distribution(feature_fn(grid)), signature)


def _object_feature_counter(grid: Grid) -> FeatureCounter:
    counter: FeatureCounter = Counter()
    for color, size in _objects(grid):
        counter[("object", color, size)] += 1
    return counter


def _palette_feature_counter(grid: Grid) -> FeatureCounter:
    return Counter(("color", color) for color, count in _palette_counts(grid).items() for _ in range(count))


def always_on_score(grid: Grid, train_pairs: Sequence[Mapping[str, Grid]]) -> float:
    """Score the two always-on families that GAP-1 says are transpose-invariant."""

    return max(
        _feature_score(grid, train_pairs, _object_feature_counter),
        _feature_score(grid, train_pairs, _palette_feature_counter),
    )


def _bin(index: int, total: int, bins: int = 3) -> int:
    return min(bins - 1, int(bins * index / max(total, 1)))


def _directional_adjacency_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()
    for r in range(h):
        for c in range(w - 1):
            counter[("H", g[r][c], g[r][c + 1])] += 1
    for r in range(h - 1):
        for c in range(w):
            counter[("V", g[r][c], g[r + 1][c])] += 1
    return counter


def _row_ordered_edge_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()
    for r in range(h):
        band = _bin(r, h)
        for c in range(w - 1):
            counter[("row_edge", band, g[r][c], g[r][c + 1])] += 1
    return counter


def _column_ordered_edge_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()
    for c in range(w):
        band = _bin(c, w)
        for r in range(h - 1):
            counter[("column_edge", band, g[r][c], g[r + 1][c])] += 1
    return counter


def _diagonal_adjacency_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()
    for r in range(h - 1):
        for c in range(w - 1):
            counter[("down_right", g[r][c], g[r + 1][c + 1])] += 1
        for c in range(1, w):
            counter[("down_left", g[r][c], g[r + 1][c - 1])] += 1
    return counter


def _corner_quadrant_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()
    for r, row in enumerate(g):
        vertical = "top" if r < h / 2 else "bottom"
        for c, color in enumerate(row):
            horizontal = "left" if c < w / 2 else "right"
            counter[("quadrant", vertical, horizontal, color)] += 1
    return counter


def _border_ordered_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()
    for c in range(w):
        counter[("top", _bin(c, w), g[0][c])] += 1
        counter[("bottom", _bin(c, w), g[h - 1][c])] += 1
    for r in range(h):
        counter[("left", _bin(r, h), g[r][0])] += 1
        counter[("right", _bin(r, h), g[r][w - 1])] += 1
    return counter


def _centroid_orientation_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    positions: dict[int, list[tuple[int, int]]] = {}
    for r, row in enumerate(g):
        for c, color in enumerate(row):
            positions.setdefault(color, []).append((r, c))
    counter: FeatureCounter = Counter()
    for color, points in positions.items():
        mean_r = sum(r for r, _c in points) / len(points)
        mean_c = sum(c for _r, c in points) / len(points)
        counter[("row_centroid", color, _bin(int(round(mean_r)), h))] += 1
        counter[("col_centroid", color, _bin(int(round(mean_c)), w))] += 1
    return counter


def _run_profile_counter(grid: Grid) -> FeatureCounter:
    g = _grid_tuple(grid)
    h, w = dims(g)
    counter: FeatureCounter = Counter()

    def length_bin(length: int) -> int:
        return min(4, int(math.log2(max(1, length))))

    for row in g:
        start = 0
        for c in range(1, w + 1):
            if c == w or row[c] != row[start]:
                counter[("H_run", row[start], length_bin(c - start))] += 1
                start = c
    for c in range(w):
        start = 0
        for r in range(1, h + 1):
            if r == h or g[r][c] != g[start][c]:
                counter[("V_run", g[start][c], length_bin(r - start))] += 1
                start = r
    return counter


def default_discriminators() -> tuple[Discriminator, ...]:
    return (
        Discriminator(
            REFUTED_DIRECTIONAL,
            "Previously refuted H/V ordered color-adjacency signature against train outputs.",
            True,
            _directional_adjacency_counter,
        ),
        Discriminator(
            "row_ordered_edge_profile",
            "Horizontal edge color-pair distributions anchored to row bands.",
            True,
            _row_ordered_edge_counter,
        ),
        Discriminator(
            "column_ordered_edge_profile",
            "Vertical edge color-pair distributions anchored to column bands.",
            True,
            _column_ordered_edge_counter,
        ),
        Discriminator(
            "diagonal_adjacency_asymmetry",
            "Down-right versus down-left ordered diagonal adjacency distributions.",
            True,
            _diagonal_adjacency_counter,
        ),
        Discriminator(
            "corner_anchored_quadrant_histogram",
            "Color histogram anchored separately to top-left, top-right, bottom-left, and bottom-right.",
            True,
            _corner_quadrant_counter,
        ),
        Discriminator(
            "border_ordered_profile",
            "Top, bottom, left, and right edge color distributions with positional bins.",
            True,
            _border_ordered_counter,
        ),
        Discriminator(
            "color_centroid_orientation",
            "Per-color row and column centroid bands compared without transpose symmetrization.",
            True,
            _centroid_orientation_counter,
        ),
        Discriminator(
            "row_column_run_profile",
            "Horizontal versus vertical color-run length profile.",
            True,
            _run_profile_counter,
        ),
    )


def candidate_discriminator_metadata(
    discriminators: Sequence[Discriminator] | None = None,
) -> list[JsonDict]:
    rows = discriminators or default_discriminators()
    return [
        {
            "name": row.name,
            "description": row.description,
            "transpose_sensitive": bool(row.transpose_sensitive),
        }
        for row in rows
    ]


def score_candidate_discriminators(
    grid: Grid,
    train_pairs: Sequence[Mapping[str, Grid]],
    discriminators: Sequence[Discriminator] | None = None,
) -> dict[str, float]:
    rows = discriminators or default_discriminators()
    return {row.name: _feature_score(grid, train_pairs, row.feature_fn) for row in rows}


def _combined_score(
    grid: Grid,
    train_pairs: Sequence[Mapping[str, Grid]],
    subset: Sequence[str],
    discriminators_by_name: Mapping[str, Discriminator],
) -> float:
    scores = [always_on_score(grid, train_pairs)]
    for name in subset:
        scores.append(_feature_score(grid, train_pairs, discriminators_by_name[name].feature_fn))
    return max(scores)


def _score_table(
    pools: Sequence[TaskPool],
    discriminators_by_name: Mapping[str, Discriminator],
) -> dict[tuple[str, str], dict[str, float]]:
    table: dict[tuple[str, str], dict[str, float]] = {}
    for pool in pools:
        signatures = {
            name: _avg_feature_distribution(pool.train_pairs, discriminator.feature_fn)
            for name, discriminator in discriminators_by_name.items()
        }
        always_signatures = {
            "object_count": _avg_feature_distribution(pool.train_pairs, _object_feature_counter),
            "palette_histogram_shape": _avg_feature_distribution(pool.train_pairs, _palette_feature_counter),
        }
        for candidate in pool.candidates:
            row = {
                "object_count": _l1_half(
                    _distribution(_object_feature_counter(candidate.grid)),
                    always_signatures["object_count"],
                ),
                "palette_histogram_shape": _l1_half(
                    _distribution(_palette_feature_counter(candidate.grid)),
                    always_signatures["palette_histogram_shape"],
                ),
            }
            row["__always_on__"] = max(row["object_count"], row["palette_histogram_shape"])
            for name, discriminator in discriminators_by_name.items():
                row[name] = _l1_half(_distribution(discriminator.feature_fn(candidate.grid)), signatures[name])
            table[(pool.task_id, candidate.candidate_id)] = row
    return table


def _combined_score_cached(
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
    pool: TaskPool,
    candidate: CandidateGrid,
    subset: Sequence[str],
) -> float:
    row = score_table[(pool.task_id, candidate.candidate_id)]
    return max([row["__always_on__"], *(row[name] for name in subset)])


def _ranked_candidates(
    pool: TaskPool,
    subset: Sequence[str],
    discriminators_by_name: Mapping[str, Discriminator],
    score_table: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
) -> list[CandidateGrid]:
    return sorted(
        pool.candidates,
        key=lambda candidate: (
            _combined_score_cached(score_table, pool, candidate, subset)
            if score_table is not None
            else _combined_score(candidate.grid, pool.train_pairs, subset, discriminators_by_name),
            candidate.candidate_id,
        ),
    )


def _pass_at_2(
    pools: Sequence[TaskPool],
    subset: Sequence[str],
    discriminators_by_name: Mapping[str, Discriminator],
    score_table: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
) -> float:
    if not pools:
        return 0.0
    hits = 0
    for pool in pools:
        ranked = _ranked_candidates(pool, subset, discriminators_by_name, score_table)
        hits += int(any(candidate.correct for candidate in ranked[:2]))
    return round(hits / len(pools), 6)


def _transpose_capture(
    pools: Sequence[TaskPool],
    subset: Sequence[str],
    discriminators_by_name: Mapping[str, Discriminator],
    score_table: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
) -> tuple[int, int]:
    captured = 0
    total = 0
    for pool in pools:
        gold = next((candidate for candidate in pool.candidates if candidate.correct), None)
        transposed = next((candidate for candidate in pool.candidates if candidate.kind == "transposed_gold"), None)
        if gold is None or transposed is None:
            continue
        total += 1
        if score_table is None:
            gold_score = _combined_score(gold.grid, pool.train_pairs, subset, discriminators_by_name)
            transposed_score = _combined_score(transposed.grid, pool.train_pairs, subset, discriminators_by_name)
        else:
            gold_score = _combined_score_cached(score_table, pool, gold, subset)
            transposed_score = _combined_score_cached(score_table, pool, transposed, subset)
        if gold_score < transposed_score:
            captured += 1
    return captured, total


def _split_pools(
    pools: Sequence[TaskPool],
    *,
    seed: int,
    heldout_fraction: float = 0.34,
) -> tuple[list[TaskPool], list[TaskPool]]:
    ordered = list(pools)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < 2:
        return ordered, ordered
    heldout_n = max(1, min(len(ordered) - 1, round(len(ordered) * heldout_fraction)))
    return ordered[heldout_n:], ordered[:heldout_n]


def search_best_subset(
    pools: Sequence[TaskPool],
    *,
    discriminators: Sequence[Discriminator] | None = None,
    seed: int = RANDOM_SEED,
) -> JsonDict:
    rows = tuple(discriminators or default_discriminators())
    by_name = {row.name: row for row in rows}
    train, heldout = _split_pools(pools, seed=seed)
    table = _score_table(pools, by_name)
    names = [row.name for row in rows]
    candidates: list[tuple[str, ...]] = [()]
    for size in range(1, len(names) + 1):
        candidates.extend(itertools.combinations(names, size))

    scored: list[JsonDict] = []
    for subset in candidates:
        train_pass2 = _pass_at_2(train, subset, by_name, table)
        captured_train, _total_train = _transpose_capture(train, subset, by_name, table)
        scored.append(
            {
                "subset": list(subset),
                "train_pass@2": train_pass2,
                "train_transpose_captures": captured_train,
                "subset_size": len(subset),
            }
        )
    best = max(
        scored,
        key=lambda row: (
            row["train_pass@2"],
            row["train_transpose_captures"],
            -row["subset_size"],
            tuple(reversed(row["subset"])),
        ),
    )
    subset = tuple(best["subset"])
    return {
        "train_task_count": len(train),
        "heldout_task_count": len(heldout),
        "searched_subset_count": len(scored),
        "best_subset": list(subset),
        "train_pass@2": best["train_pass@2"],
        "heldout_pass@2": _pass_at_2(heldout, subset, by_name, table),
        "baseline_heldout_pass@2": _pass_at_2(heldout, (), by_name, table),
        "scored_subsets": sorted(
            scored,
            key=lambda row: (-row["train_pass@2"], -row["train_transpose_captures"], row["subset_size"]),
        )[:10],
    }


def load_square_transpose_subset(
    *,
    root: Path | str = REPO_ROOT,
    arc_root: Path | str = DEFAULT_ARC_ROOT,
    seed: int = RANDOM_SEED,
) -> list[TaskPool]:
    root_path = Path(root)
    source = root_path / SOURCE_ARTIFACT_RELATIVE_PATH
    if not source.exists():
        raise FileNotFoundError(f"missing source artifact: {SOURCE_ARTIFACT_RELATIVE_PATH}")

    arc_path = Path(arc_root)
    challenges = json.loads((arc_path / "arc-agi_training_challenges.json").read_text(encoding="utf-8"))
    solutions = json.loads((arc_path / "arc-agi_training_solutions.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    task_ids = list(challenges)
    all_golds = [solutions[task_id][0] for task_id in task_ids if solutions.get(task_id)]
    pools: list[TaskPool] = []

    for task_id in task_ids:
        task = challenges[task_id]
        for test_index, test in enumerate(task.get("test", [])):
            if not solutions.get(task_id) or test_index >= len(solutions[task_id]):
                continue
            gold = solutions[task_id][test_index]
            distractors = _distractors(gold, test["input"], all_golds, rng)
            transposed = distractors.get("transposed_gold")
            if transposed is None or dims(transposed) != dims(gold):
                continue
            candidates = [CandidateGrid("z_gold", "gold", gold, True)]
            for kind, grid in sorted(distractors.items()):
                candidates.append(CandidateGrid(f"{kind}", kind, grid, False))
            pools.append(
                TaskPool(
                    task_id=f"{task_id}:{test_index}",
                    train_pairs=tuple(task["train"]),
                    test_input=test["input"],
                    candidates=tuple(sorted(candidates, key=lambda row: row.candidate_id)),
                )
            )
    return pools


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _value(artifact: Mapping[str, Any], field: str) -> Any:
    raw = artifact.get(field)
    if isinstance(raw, Mapping) and "value" in raw:
        return raw["value"]
    return raw


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, Mapping):
        checksum = dict(checksum)
        checksum["value"] = ""
        payload["reproducibility_checksum"] = checksum
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _source_context(root: Path, pools: Sequence[TaskPool]) -> JsonDict:
    source = root / SOURCE_ARTIFACT_RELATIVE_PATH
    return {
        "source_artifact": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_artifact_exists": source.exists(),
        "source_artifact_sha256": (
            "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
        ),
        "square_transpose_subset_n": len(pools),
    }


def _candidate_effects(
    pools: Sequence[TaskPool],
    baseline_pass2: float,
    discriminators_by_name: Mapping[str, Discriminator],
    score_table: Mapping[tuple[str, str], Mapping[str, float]],
) -> list[JsonDict]:
    out = []
    for name in discriminators_by_name:
        pass2 = _pass_at_2(pools, (name,), discriminators_by_name, score_table)
        captured, total = _transpose_capture(pools, (name,), discriminators_by_name, score_table)
        if pass2 > baseline_pass2:
            effect = "helped"
        elif pass2 < baseline_pass2:
            effect = "hurt"
        else:
            effect = "neutral"
        out.append(
            {
                "name": name,
                "pass@2": pass2,
                "delta_vs_baseline": _round_float(pass2 - baseline_pass2),
                "transpose_captures": f"{captured} out of {total}",
                "effect": effect,
            }
        )
    return out


def _gap_update_block(artifact: Mapping[str, Any]) -> str:
    verdict = _value(artifact, "honest_verdict")
    best = _value(artifact, "best_subset_found")
    baseline = _value(artifact, "pass_at_2_baseline_always_on_only")
    pass2 = _value(artifact, "pass_at_2_best_subset")
    captured = _value(artifact, "transpose_misvotes_captured")
    effects = artifact.get("candidate_discriminator_effects", [])
    effect_text = ", ".join(f"{row['name']}={row['effect']}" for row in effects[:8])
    return (
        "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 start -->\n"
        "- experiment_5205 AutoPyVerifier-inspired deterministic set search (2026-07-03): "
        f"best_subset={best}, pass@2 baseline={baseline}, pass@2 best={pass2}, "
        f"transpose captures={captured}. Candidate singletons: {effect_text}. "
        f"Verdict: {verdict}\n"
        "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 end -->\n"
    )


def update_verifier_gap_doc(root: Path | str, artifact: Mapping[str, Any]) -> None:
    path = Path(root) / VERIFIER_GAPS_RELATIVE_PATH
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    start = "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 start -->"
    end = "<!-- experiment_5205_autopyverifier_gap1_pilot_v476 end -->"
    block = _gap_update_block(artifact)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        path.write_text(before + block + after, encoding="utf-8")
        return
    marker = "### GAP-2:"
    if marker in text:
        text = text.replace(marker, block + "\n" + marker, 1)
    else:
        text += "\n" + block
    path.write_text(text, encoding="utf-8")


def build_artifact(
    pools: Sequence[TaskPool],
    *,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
    duration_s: float = 0.0,
) -> JsonDict:
    discriminators = default_discriminators()
    by_name = {row.name: row for row in discriminators}
    score_table = _score_table(pools, by_name)
    search = search_best_subset(pools, discriminators=discriminators, seed=random_seed)
    best_subset = tuple(search["best_subset"])
    baseline_pass2 = _pass_at_2(pools, (), by_name, score_table)
    best_pass2 = _pass_at_2(pools, best_subset, by_name, score_table)
    refuted_pass2 = _pass_at_2(pools, (REFUTED_DIRECTIONAL,), by_name, score_table)
    captured, transpose_total = _transpose_capture(pools, best_subset, by_name, score_table)
    beats_baseline = best_pass2 > baseline_pass2
    beats_refuted = best_pass2 > refuted_pass2
    gap_status = "gap1_candidate_positive" if beats_baseline and captured > 0 else "gap1_remains_open"
    verdict = (
        f"complete: set_search_{'beats' if beats_baseline else 'does_not_beat'}_always_on_"
        f"{'beats' if beats_refuted else 'does_not_beat'}_single_refuted_"
        f"baseline_{baseline_pass2:.4f}_best_{best_pass2:.4f}_single_refuted_{refuted_pass2:.4f}_"
        f"captured_{captured}_of_{transpose_total}_{gap_status}"
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "source_context": _source_context(Path(root), pools),
        "split_summary": {
            "train_task_count": search["train_task_count"],
            "heldout_task_count": search["heldout_task_count"],
            "heldout_pass@2": search["heldout_pass@2"],
            "baseline_heldout_pass@2": search["baseline_heldout_pass@2"],
        },
        "search_summary": {
            "searched_subset_count": search["searched_subset_count"],
            "top_train_subsets": search["scored_subsets"],
            "selection_rule": "maximize train pass@2, then train transpose captures, then smaller subset",
        },
        "ranker_definition": (
            "Candidate-pool pass@2 uses the same top-2 selection unit as the TRM HYBRID rerank analyses. "
            "The v2 distractor artifact has no TRM vote counts, so all cached candidates are equal-vote "
            "and the verifier score supplies the ranking key."
        ),
        "candidate_discriminator_effects": _candidate_effects(pools, baseline_pass2, by_name, score_table),
        "single_refuted_directional_adjacency_pass@2": refuted_pass2,
        "transpose_distractor_count": transpose_total,
        "candidate_discriminators_authored": _wrap(
            "candidate_discriminators_authored",
            candidate_discriminator_metadata(discriminators),
        ),
        "best_subset_found": _wrap("best_subset_found", list(best_subset)),
        "pass_at_2_baseline_always_on_only": _wrap(
            "pass_at_2_baseline_always_on_only",
            _round_float(baseline_pass2),
        ),
        "pass_at_2_best_subset": _wrap("pass_at_2_best_subset", _round_float(best_pass2)),
        "transpose_misvotes_captured": _wrap(
            "transpose_misvotes_captured",
            f"{captured} out of {transpose_total}",
        ),
        "verifier_is_oracle": _wrap("verifier_is_oracle", False),
        "random_seed": _wrap("random_seed", int(random_seed)),
        "reproducibility_checksum": _wrap("reproducibility_checksum", ""),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "duration_s": round(float(duration_s), 3),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _wrap("reproducibility_checksum", payload_checksum(artifact))
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_PRINCIPLED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    for field in REQUIRED_PRINCIPLED_FIELDS:
        raw = artifact.get(field)
        if not isinstance(raw, Mapping) or "value" not in raw or "principle" not in raw:
            errors.append(f"{field} must be principle-wrapped")
            continue
        if raw.get("principle") != FIELD_PRINCIPLES[field]:
            errors.append(f"{field} principle mismatch")
    if _value(artifact, "verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if _value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be verifier_ensemble_against_cached_candidates")
    verdict = _value(artifact, "honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal complete/success prefix")
    authored = _value(artifact, "candidate_discriminators_authored")
    if not isinstance(authored, list) or not (5 <= len(authored) <= 10):
        errors.append("candidate_discriminators_authored must list 5-10 discriminators")
    elif REFUTED_DIRECTIONAL not in {str(row.get("name")) for row in authored if isinstance(row, Mapping)}:
        errors.append("candidate_discriminators_authored must include the refuted directional member")
    for field in ("pass_at_2_baseline_always_on_only", "pass_at_2_best_subset"):
        value = _value(artifact, field)
        if not isinstance(value, (float, int)) or not (0.0 <= float(value) <= 1.0):
            errors.append(f"{field} must be a float in [0,1]")
    checksum = _value(artifact, "reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    arc_root: Path | str = DEFAULT_ARC_ROOT,
    pools: Sequence[TaskPool] | None = None,
    result_path: Path | str | None = None,
    random_seed: int = RANDOM_SEED,
    duration_s: float | None = None,
    update_gap_doc: bool = True,
) -> JsonDict:
    started = time.time()
    root_path = Path(root)
    task_pools = list(pools) if pools is not None else load_square_transpose_subset(
        root=root_path,
        arc_root=arc_root,
        seed=random_seed,
    )
    elapsed = time.time() - started if duration_s is None else duration_s
    artifact = build_artifact(task_pools, root=root_path, random_seed=random_seed, duration_s=elapsed)
    output = Path(result_path) if result_path is not None else root_path / RESULT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if update_gap_doc:
        update_verifier_gap_doc(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by direct experiment invocation, not unit tests.
    artifact = run()
    print(artifact["honest_verdict"]["value"])
    print(f"wrote {RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
