"""Pure analysis helpers for the Exp5215 PAW amortization gate.

The gate answers one question: can a compile-once, act-cheaply PAW-like step
pay for itself inside the remaining action budget of ARC-AGI-3 episodes? It
does not train a live adapter, alter the live agent, or claim any ARC level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SPEC_REFS = (
    "REQ-ARC-WMTE-5215",
    "SCENARIO-ARC-WMTE-5215-AMORTIZATION-GATE",
    "SCENARIO-ARC-WMTE-5215-NO-SOLVE-OR-REGISTRY-MUTATION",
)
RESULT_SCHEMA = "carnot.arc_paw_amortization_gate.v1"
INFERENCE_SUBSTRATE = "arc_log_analysis_plus_local_timing"
DEFAULT_MARGIN = 1.25


@dataclass(frozen=True)
class ArcEpisodeRecord:
    game: str
    total_actions: int
    reached_level: int | None
    solution_labels: tuple[str, ...]
    level_up_action_indices: tuple[int, ...] = ()
    source_path: str = ""

    def with_level_up_action_indices(self, indices: Sequence[int]) -> "ArcEpisodeRecord":
        return ArcEpisodeRecord(
            game=self.game,
            total_actions=self.total_actions,
            reached_level=self.reached_level,
            solution_labels=self.solution_labels,
            level_up_action_indices=tuple(int(value) for value in indices),
            source_path=self.source_path,
        )


@dataclass(frozen=True)
class RemainingActionDistribution:
    values: list[float]
    median: float
    p75: float
    missing: list[dict[str, str]]


@dataclass(frozen=True)
class TimingEstimate:
    compile_wall_clock_s: float
    current_step_wall_clock_s: float
    cheap_step_wall_clock_s: float
    evidence: dict[str, Any]


def load_arc_loop_records(results_dir: Path) -> tuple[ArcEpisodeRecord, ...]:
    """REQ-ARC-WMTE-5215: load existing public-game ARC loop-solve action logs."""

    records: list[ArcEpisodeRecord] = []
    for path in sorted(Path(results_dir).glob("arc_loop_solve_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        game = str(payload.get("game") or path.stem.removeprefix("arc_loop_solve_"))
        labels = tuple(str(label) for label in (payload.get("solution_labels") or ()))
        total_actions = int(payload.get("moves") or payload.get("total_actions") or len(labels))
        reached_raw = payload.get("reproduced_levels", payload.get("reached_level"))
        reached_level = None if reached_raw is None else int(reached_raw)
        records.append(
            ArcEpisodeRecord(
                game=game,
                total_actions=total_actions,
                reached_level=reached_level,
                solution_labels=labels,
                level_up_action_indices=_logged_level_up_indices(payload),
                source_path=str(path),
            )
        )
    return tuple(records)


def _logged_level_up_indices(payload: Mapping[str, Any]) -> tuple[int, ...]:
    raw = payload.get("level_up_action_indices", payload.get("level_up_actions", ()))
    return tuple(int(value) for value in raw or ())


def replay_level_up_action_indices(
    *,
    labels: Sequence[str],
    env: Any,
    apply: Callable[[Any, str, Any], Any],
    warmup_label: str | None = None,
    frame_level: Callable[[Any], int] | None = None,
) -> tuple[int, ...]:
    """SCENARIO-ARC-WMTE-5215-AMORTIZATION-GATE: recover level-up boundaries."""

    level_of = frame_level or (lambda frame: int(getattr(frame, "levels_completed", 0) or 0))
    frame = env.reset()
    if warmup_label is not None:
        frame = apply(env, warmup_label, frame)
    previous_level = level_of(frame)
    indices: list[int] = []
    for action_index, label in enumerate(labels, start=1):
        frame = apply(env, label, frame)
        level = level_of(frame)
        if level > previous_level:
            indices.extend([action_index] * (level - previous_level))
            previous_level = level
    return tuple(indices)


def remaining_action_distribution(
    records: Sequence[ArcEpisodeRecord],
) -> RemainingActionDistribution:
    values: list[float] = []
    missing: list[dict[str, str]] = []
    for record in records:
        if not record.level_up_action_indices:
            missing.append(
                {
                    "game": record.game,
                    "source_path": record.source_path,
                    "reason": "missing_level_up_checkpoint",
                }
            )
            continue
        remaining = float(record.total_actions - int(record.level_up_action_indices[0]))
        values.append(remaining)
    values.sort()
    return RemainingActionDistribution(
        values=values,
        median=round(_percentile(values, 50), 6),
        p75=round(_percentile(values, 75), 6),
        missing=missing,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:  # pragma: no cover - terminal artifact requires usable logs.
        return 0.0
    if len(values) == 1:  # pragma: no cover - covered by multi-log gate tests.
        return float(values[0])
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def break_even_remaining_actions(
    *,
    compile_wall_clock_s: float,
    current_step_wall_clock_s: float,
    cheap_step_wall_clock_s: float,
) -> float:
    saved_per_step = float(current_step_wall_clock_s) - float(cheap_step_wall_clock_s)
    if saved_per_step <= 0.0:
        return float("inf")
    return round(float(compile_wall_clock_s) / saved_per_step, 6)


def paw_amortization_viable(
    *,
    median_remaining_actions: float,
    p75_remaining_actions: float,
    break_even_remaining_actions: float,
    margin: float = DEFAULT_MARGIN,
) -> bool:
    required = float(break_even_remaining_actions) * float(margin)
    return bool(
        float(median_remaining_actions) >= required and float(p75_remaining_actions) >= required
    )


def build_artifact(
    *,
    records: Sequence[ArcEpisodeRecord],
    timing: TimingEstimate,
    duration_s: float,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    distribution = remaining_action_distribution(records)
    break_even = break_even_remaining_actions(
        compile_wall_clock_s=timing.compile_wall_clock_s,
        current_step_wall_clock_s=timing.current_step_wall_clock_s,
        cheap_step_wall_clock_s=timing.cheap_step_wall_clock_s,
    )
    viable = paw_amortization_viable(
        median_remaining_actions=distribution.median,
        p75_remaining_actions=distribution.p75,
        break_even_remaining_actions=break_even,
        margin=margin,
    )
    verdict = (
        "complete_paw_amortization_gate_viable_no_arc_solve_claim"
        if viable
        else "complete_paw_amortization_gate_not_viable_no_arc_solve_claim"
    )
    return {
        "experiment": "experiment_5215_arc_paw_amortization_gate_v477",
        "schema": RESULT_SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "paw_amortization_viable": _field(viable, "median and p75 must clear break-even with margin"),
        "median_remaining_actions": _field(distribution.median, "empirical first-level-up checkpoint"),
        "p75_remaining_actions": _field(distribution.p75, "empirical first-level-up checkpoint"),
        "compile_wall_clock_s": _field(
            round(float(timing.compile_wall_clock_s), 6),
            "conservative small-LoRA compile wall-clock estimate on local GPU",
        ),
        "current_step_wall_clock_s": _field(
            round(float(timing.current_step_wall_clock_s), 6),
            "Qwen3.5-9B-MTP action-step estimate from local ARC generator timing logs",
        ),
        "cheap_step_wall_clock_s": _field(
            round(float(timing.cheap_step_wall_clock_s), 6),
            "cheap interpreter action-step estimate from bounded local timing plus conservative floor",
        ),
        "break_even_remaining_actions": _field(break_even, "compile / per-step wall-clock savings"),
        "arc_registry_modified": _field(False, "pure analysis gate; registry must remain unchanged"),
        "inference_substrate": _field(INFERENCE_SUBSTRATE, "ARC log analysis plus bounded local timing"),
        "honest_verdict": {
            "value": verdict,
            "principle": (
                "Must start with complete:/complete_/success:/success_ or blocked_ and must not "
                "claim PAW solves ARC."
            ),
        },
        "checkpoint_analysis": {
            "first_level_up": {
                "status": "computed",
                "n": len(distribution.values),
                "remaining_actions": distribution.values,
                "missing": distribution.missing,
            },
            "stable_transition_model": {
                "status": "missing_data",
                "missing": "existing logs do not record a stable-transition-model checkpoint",
            },
            "no_new_transition_discovery": {
                "status": "missing_data",
                "missing": "existing logs do not record per-action transition-discovery events",
            },
        },
        "timing_evidence": timing.evidence,
        "source_artifacts_read": [record.source_path for record in records if record.source_path],
        "viability_margin": margin,
        "level_solve_claimed": False,
    }


def _field(value: Any, principle: str) -> dict[str, Any]:
    return {"value": value, "principle": principle}


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
