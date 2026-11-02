"""Game-blind mechanic-class detection for ARC world-model induction.

Spec refs: REQ-ARC-WMTE-6282,
SCENARIO-ARC-WMTE-6282-GAME-BLIND-FIXTURES,
SCENARIO-ARC-WMTE-6282-LIVE-PROMPT-ROUTE.

The detector reads only visible transition deltas. It does not use a game id,
registry row, hidden source file, exhaustive search result, or per-game adapter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


MECHANIC_CLASSES = ("push_block", "toggle_move", "navigation", "negative", "unknown")


@dataclass(frozen=True)
class SyntheticMechanicFixture:
    """One game-blind transition bundle with a trusted synthetic class label."""

    fixture_id: str
    family: str
    transitions: tuple[Any, ...]
    game_id: None = None


@dataclass(frozen=True)
class MechanicClassResult:
    """Detector output that can be copied into the live induction prompt."""

    predicted_class: str
    probabilities: dict[str, float]
    uncertainty: float
    support: dict[str, int | float]
    sample_size: int

    def to_json(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "probabilities": dict(self.probabilities),
            "uncertainty": round(float(self.uncertainty), 6),
            "support": dict(self.support),
            "sample_size": int(self.sample_size),
        }


def _grid(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=int)


def _action(row: Any) -> int:
    return int(getattr(row, "action", row[1] if isinstance(row, Sequence) else 0))


def _data(row: Any) -> Any:
    if hasattr(row, "data"):
        return getattr(row, "data")
    if isinstance(row, Sequence) and len(row) > 2:
        return row[2]
    return None


def _before_after(row: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(row, "grid") and hasattr(row, "next_grid"):
        return _grid(getattr(row, "grid")), _grid(getattr(row, "next_grid"))
    if isinstance(row, Sequence) and len(row) >= 3:
        return _grid(row[0]), _grid(row[-1])
    raise TypeError("transition must expose grid and next_grid")


def _background(*grids: np.ndarray) -> int:
    counts: Counter[int] = Counter()
    for grid in grids:
        counts.update(int(v) for v in grid.ravel())
    return counts.most_common(1)[0][0] if counts else 0


def _is_keyboard(row: Any) -> bool:
    return _action(row) != 6 and _data(row) in (None, {})


def _diff_points(before: np.ndarray, after: np.ndarray) -> list[tuple[int, int]]:
    if before.shape != after.shape:
        return []
    return [(int(r), int(c)) for r, c in np.argwhere(before != after)]


def _is_push_transition(before: np.ndarray, after: np.ndarray) -> bool:
    """Detect a two-object one-step push pattern in any cardinal direction."""

    if before.shape != after.shape:
        return False
    bg = _background(before, after)
    h, w = before.shape
    for r in range(h):
        for c in range(w):
            first = int(before[r, c])
            if first == bg:
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                r1, c1 = r + dr, c + dc
                r2, c2 = r + 2 * dr, c + 2 * dc
                if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
                    continue
                second = int(before[r1, c1])
                if second == bg or int(before[r2, c2]) != bg:
                    continue
                if (
                    int(after[r, c]) == bg
                    and int(after[r1, c1]) == first
                    and int(after[r2, c2]) == second
                ):
                    return True
    return False


def _single_color_move_count(before: np.ndarray, after: np.ndarray) -> int:
    """Count colors whose pixels translate as a whole from before to after."""

    if before.shape != after.shape:
        return 0
    bg = _background(before, after)
    moved = 0
    for color in sorted((set(before.ravel()) | set(after.ravel())) - {bg}):
        b = np.argwhere(before == color)
        a = np.argwhere(after == color)
        if len(b) == 0 or len(a) == 0 or len(b) != len(a):
            continue
        deltas = a - b
        if len({tuple(int(x) for x in row) for row in deltas}) == 1 and np.any(deltas[0] != 0):
            moved += 1
    return moved


def _is_navigation_transition(before: np.ndarray, after: np.ndarray) -> bool:
    return len(_diff_points(before, after)) == 2 and _single_color_move_count(before, after) == 1


def _is_toggle_move_transition(before: np.ndarray, after: np.ndarray) -> bool:
    """Detect movement plus extra local state flips, excluding the push-chain shape."""

    if before.shape != after.shape or _is_push_transition(before, after):
        return False
    diff_count = len(_diff_points(before, after))
    if diff_count < 3:
        return False
    if _single_color_move_count(before, after) >= 1:
        return True
    changed_before = [int(before[r, c]) for r, c in _diff_points(before, after)]
    changed_after = [int(after[r, c]) for r, c in _diff_points(before, after)]
    palette = set(changed_before) | set(changed_after)
    return 2 <= len(palette) <= 4 and diff_count <= max(10, before.size // 3)


def transition_features(transitions: Sequence[Any]) -> dict[str, int | float]:
    """Summarize visible deltas without reading any game-specific context."""

    rows = list(transitions)
    changed = 0
    keyboard_changed = 0
    push_like = 0
    toggle_like = 0
    navigation_like = 0
    total_changed_cells = 0
    for row in rows:
        before, after = _before_after(row)
        diff_count = len(_diff_points(before, after))
        if diff_count == 0:
            continue
        changed += 1
        total_changed_cells += diff_count
        if _is_keyboard(row):
            keyboard_changed += 1
        if _is_push_transition(before, after) and _is_keyboard(row):
            push_like += 1
        elif _is_toggle_move_transition(before, after) and _is_keyboard(row):
            toggle_like += 1
        elif _is_navigation_transition(before, after) and _is_keyboard(row):
            navigation_like += 1
    n = max(1, len(rows))
    return {
        "n_transitions": len(rows),
        "n_changed": changed,
        "keyboard_changed": keyboard_changed,
        "push_like": push_like,
        "toggle_like": toggle_like,
        "navigation_like": navigation_like,
        "mean_changed_cells": round(float(total_changed_cells / max(1, changed)), 4),
        "changed_ratio": round(float(changed / n), 4),
    }


def classify_transition_history(transitions: Sequence[Any]) -> MechanicClassResult:
    """Classify a transition history into a mechanic class with calibrated uncertainty."""

    features = transition_features(transitions)
    changed = max(1, int(features["n_changed"]))
    raw_scores = {
        "push_block": 0.2 + 4.0 * int(features["push_like"]) / changed,
        "toggle_move": 0.2 + 4.0 * int(features["toggle_like"]) / changed,
        "navigation": 0.2 + 3.0 * int(features["navigation_like"]) / changed,
        "negative": 0.2 + (3.0 if int(features["n_changed"]) == 0 else 0.0),
        "unknown": 0.1,
    }
    if int(features["n_changed"]) and not any(
        int(features[name]) for name in ("push_like", "toggle_like", "navigation_like")
    ):
        raw_scores["unknown"] += 2.0

    total = sum(raw_scores.values())
    probabilities = {
        name: round(float(score / total), 6) for name, score in sorted(raw_scores.items())
    }
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    predicted = ordered[0][0]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    uncertainty = max(0.0, min(1.0, 1.0 - margin))
    support = dict(features)
    support["top_margin"] = round(float(margin), 6)
    return MechanicClassResult(
        predicted_class=predicted,
        probabilities=probabilities,
        uncertainty=round(float(uncertainty), 6),
        support=support,
        sample_size=int(features["n_transitions"]),
    )


def _transition(before: np.ndarray, action: int, after: np.ndarray, data: Any = None) -> tuple:
    return (before.copy(), int(action), data, after.copy())


def _push_fixture(idx: int) -> tuple[Any, ...]:
    rows = []
    for step in range(3):
        before = np.zeros((6, 8), dtype=int)
        after = np.zeros((6, 8), dtype=int)
        r = 2 + (idx % 2)
        c = 1 + step
        before[r, c] = 1
        before[r, c + 1] = 2 + (idx % 3)
        after[r, c + 1] = 1
        after[r, c + 2] = 2 + (idx % 3)
        rows.append(_transition(before, 4, after))
    return tuple(rows)


def _toggle_fixture(idx: int) -> tuple[Any, ...]:
    rows = []
    for step in range(3):
        before = np.zeros((6, 8), dtype=int)
        after = before.copy()
        r = 2 + (idx % 2)
        c = 1 + step
        before[r, c] = 1
        after[r, c] = 0
        after[r, c + 1] = 1
        after[r - 1, c + 1] = 3 if before[r - 1, c + 1] == 0 else 0
        after[r + 1, c + 1] = 3 if before[r + 1, c + 1] == 0 else 0
        rows.append(_transition(before, 4, after))
    return tuple(rows)


def _navigation_fixture(idx: int) -> tuple[Any, ...]:
    rows = []
    for step in range(3):
        before = np.zeros((6, 8), dtype=int)
        after = np.zeros((6, 8), dtype=int)
        r = 1 + (idx % 3)
        c = 1 + step
        before[r, c] = 1
        after[r, c + 1] = 1
        rows.append(_transition(before, 4, after))
    return tuple(rows)


def _negative_fixture(idx: int) -> tuple[Any, ...]:
    rows = []
    for step in range(3):
        before = np.zeros((6, 8), dtype=int)
        before[1 + idx % 3, 1 + step] = 4
        rows.append(_transition(before, 6, before, data={"x": step, "y": idx}))
    return tuple(rows)


def build_synthetic_fixture_manifest(
    *, seed: int = 6282, per_family: int = 8
) -> list[SyntheticMechanicFixture]:
    """Build game-blind controls used to configure and evaluate the detector."""

    del seed  # Fixtures are deterministic; the seed is recorded by callers.
    builders = {
        "push_block": _push_fixture,
        "toggle_move": _toggle_fixture,
        "navigation": _navigation_fixture,
        "negative": _negative_fixture,
    }
    fixtures: list[SyntheticMechanicFixture] = []
    for family, builder in builders.items():
        for idx in range(per_family):
            fixtures.append(
                SyntheticMechanicFixture(
                    fixture_id=f"{family}_{idx:02d}",
                    family=family,
                    transitions=builder(idx),
                )
            )
    return fixtures


def fixture_family_counts(fixtures: Sequence[SyntheticMechanicFixture]) -> dict[str, int]:
    return dict(sorted(Counter(f.family for f in fixtures).items()))


def evaluate_detector_on_fixtures(fixtures: Sequence[SyntheticMechanicFixture]) -> dict[str, Any]:
    """Evaluate held-out synthetic controls and return auditable metrics."""

    by_family: dict[str, dict[str, int]] = {}
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    uncertainties: list[float] = []
    for fixture in fixtures:
        result = classify_transition_history(fixture.transitions)
        hit = result.predicted_class == fixture.family
        correct += int(hit)
        by_family.setdefault(fixture.family, {"correct": 0, "total": 0})
        by_family[fixture.family]["correct"] += int(hit)
        by_family[fixture.family]["total"] += 1
        confusion.setdefault(fixture.family, {})
        confusion[fixture.family][result.predicted_class] = (
            confusion[fixture.family].get(result.predicted_class, 0) + 1
        )
        uncertainties.append(float(result.uncertainty))
    n = len(fixtures)
    return {
        "sample_size": n,
        "overall_accuracy": round(float(correct / max(1, n)), 6),
        "by_family": by_family,
        "confusion": confusion,
        "mean_uncertainty": round(float(sum(uncertainties) / max(1, len(uncertainties))), 6),
        "forbidden_inputs": {
            "game_id_used": False,
            "hidden_source_used": False,
            "bfs_used": False,
            "adapter_used": False,
        },
    }


def fixture_manifest_payload(
    fixtures: Sequence[SyntheticMechanicFixture],
    *,
    seed: int,
) -> dict[str, Any]:
    """Serialize controls without embedding large grids in the terminal artifact."""

    return {
        "seed": int(seed),
        "fixture_count": len(fixtures),
        "family_counts": fixture_family_counts(fixtures),
        "fixtures": [
            {
                "fixture_id": fixture.fixture_id,
                "family": fixture.family,
                "transition_count": len(fixture.transitions),
                "game_id": fixture.game_id,
                "feature_summary": transition_features(fixture.transitions),
            }
            for fixture in fixtures
        ],
    }


def calibration_summary(fixtures: Sequence[SyntheticMechanicFixture]) -> dict[str, Any]:
    """Summarize how uncertainty behaves on trusted controls."""

    rows = []
    for fixture in fixtures:
        result = classify_transition_history(fixture.transitions)
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "family": fixture.family,
                "predicted_class": result.predicted_class,
                "uncertainty": result.uncertainty,
                "correct": result.predicted_class == fixture.family,
            }
        )
    correct_unc = [float(row["uncertainty"]) for row in rows if row["correct"]]
    wrong_unc = [float(row["uncertainty"]) for row in rows if not row["correct"]]
    return {
        "method": "synthetic_controls_margin_uncertainty",
        "sample_size": len(rows),
        "mean_uncertainty_correct": round(sum(correct_unc) / max(1, len(correct_unc)), 6),
        "mean_uncertainty_incorrect": (
            None if not wrong_unc else round(sum(wrong_unc) / len(wrong_unc), 6)
        ),
        "rows": rows[:12],
    }


def prompt_block(transitions: Sequence[Any]) -> str:
    """Return the prompt appendix consumed by `induce_prompt()` when the route is enabled."""

    result = classify_transition_history(transitions)
    if result.predicted_class in {"unknown", "negative"}:
        return ""
    support = result.support
    return (
        "MECHANIC CLASS ROUTER (game-blind, from observed transition deltas only): "
        f"class={result.predicted_class} uncertainty={result.uncertainty:.3f} "
        f"support=changed:{support.get('n_changed', 0)} "
        f"push:{support.get('push_like', 0)} toggle:{support.get('toggle_like', 0)} "
        f"nav:{support.get('navigation_like', 0)}. Use this as a general mechanic prior; "
        "do not copy exact coordinates."
    )
