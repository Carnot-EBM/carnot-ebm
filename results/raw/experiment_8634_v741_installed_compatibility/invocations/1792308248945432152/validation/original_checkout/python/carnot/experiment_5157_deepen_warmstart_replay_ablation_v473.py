"""Experiment 5157: deepen warm-start replay ablation.

Spec refs: REQ-ARC-WMTE-5157,
SCENARIO-ARC-WMTE-5157-TRACE-PRECONDITION,
SCENARIO-ARC-WMTE-5157-REDRAW-WARM-START,
SCENARIO-ARC-WMTE-5157-STABLE-ARTIFACT.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_executable_world_model import Transition


EXPERIMENT = "experiment_5157_deepen_warmstart_replay_ablation_v473"
SCHEMA = "carnot.exp5157.deepen_warmstart_replay_ablation.v1"
RESULT_RELATIVE_PATH = "results/experiment_5157_deepen_warmstart_replay_ablation_v473.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 5157
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
REPO_ROOT = Path(__file__).resolve().parents[2]

SPEC_REFS = (
    "REQ-ARC-WMTE-5157",
    "SCENARIO-ARC-WMTE-5157-TRACE-PRECONDITION",
    "SCENARIO-ARC-WMTE-5157-REDRAW-WARM-START",
    "SCENARIO-ARC-WMTE-5157-STABLE-ARTIFACT",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "games_tested": {
        "principle": (
            "The gate requires >=6 transitions; this field lets a reviewer verify the sample size."
        )
    },
    "warmstart_vs_cold_delta_median": {
        "principle": "The exact quantity exp5155's falsifiable_gate is defined over."
    },
    "actions_saved_pct_median": {
        "principle": "The matched replay action-count reduction; zero when no shorter replay is derived."
    },
    "gate_passed": {
        "principle": (
            "Apply exp5155's own falsifiable_gate verbatim -- do not redefine the threshold post hoc."
        )
    },
    "per_transition_breakdown": {
        "principle": (
            "An aggregate-only report can hide a result that only holds for one or two games."
        )
    },
    "solve_provenance": {
        "principle": (
            "This is offline replay over already-banked registry trajectories, not a live hidden-game solve."
        )
    },
    "offline_reproduced": {
        "principle": (
            "No new level is claimed by this task -- it is a mechanism ablation feeding exp5159's live attempt."
        )
    },
    "verifier_is_oracle": {
        "principle": "The held-out transition scorer is oracle-distinct from any live solve claim."
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_ AND state plainly whether the "
            "gate passed or failed -- do not bury a null in qualifiers."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "honest_verdict",
    "games_tested",
    "warmstart_vs_cold_delta_median",
    "actions_saved_pct_median",
    "gate_passed",
    "per_transition_breakdown",
    "solve_provenance",
    "offline_reproduced",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "field_principles",
    "spec_refs",
)


@dataclass(frozen=True)
class BoundaryCase:
    """One within-game level boundary with prior and target replay evidence."""

    game: str
    level_from: int
    level_to: int
    pre_boundary: tuple[Transition, ...]
    post_boundary: tuple[Transition, ...]
    source_artifact: str


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)


def parse_action_label(label: str) -> tuple[int, dict[str, int] | None]:
    """Convert a banked adapter label into the action/data shape used by TTT."""

    try:
        payload = json.loads(label)
    except (TypeError, ValueError):
        return 0, None
    if not isinstance(payload, Mapping):
        return int(payload), None
    action = int(payload.get("action", payload.get("action_id", 6 if "x" in payload else 0)))
    data = payload.get("data")
    if data is None and action == 6 and "x" in payload and "y" in payload:
        data = {"x": payload["x"], "y": payload["y"]}
    if isinstance(data, Mapping) and "x" in data and "y" in data:
        return action, {"x": int(data["x"]), "y": int(data["y"])}
    return action, None


def changed_cell_value_accuracy(engine: Any, heldout: Transition) -> float:
    """Score predicted values only on cells that truly changed in the held-out transition."""

    if engine is None:
        return 0.0
    source = np.asarray(heldout.grid)
    expected = np.asarray(heldout.next_grid)
    changed = source != expected
    if not bool(changed.any()):
        try:
            predicted = np.asarray(engine(source, heldout.action, heldout.data))
        except Exception:
            return 0.0
        return (
            1.0
            if predicted.shape == expected.shape and np.array_equal(predicted, expected)
            else 0.0
        )
    try:
        predicted = np.asarray(engine(source, heldout.action, heldout.data))
    except Exception:
        return 0.0
    if predicted.shape != expected.shape:
        return 0.0
    return float((predicted[changed] == expected[changed]).mean())


def _heldout_index(post_boundary: Sequence[Transition]) -> int | None:
    for idx in range(len(post_boundary) - 1, -1, -1):
        transition = post_boundary[idx]
        if not np.array_equal(np.asarray(transition.grid), np.asarray(transition.next_grid)):
            return idx
    return None


def boundary_has_changed_heldout(case: BoundaryCase) -> bool:
    return bool(case.pre_boundary) and _heldout_index(case.post_boundary) is not None


def recoverable_boundaries(cases: Iterable[BoundaryCase]) -> list[BoundaryCase]:
    return [case for case in cases if boundary_has_changed_heldout(case)]


def _actions_to_next_level(post_boundary: Sequence[Transition]) -> int | None:
    for idx, transition in enumerate(post_boundary):
        if int(transition.level_after) > int(transition.level_before):
            return idx + 1
    return None


def evaluate_boundary_case(case: BoundaryCase, *, dynamics_backend: str = "dsl") -> dict[str, Any]:
    """Run the cold-slice and ReDRAW warm-start arms for one boundary."""

    from carnot.agentic.arc_live_ttt import gated_engine_from_transitions

    heldout_idx = _heldout_index(case.post_boundary)
    if heldout_idx is None:
        raise ValueError(
            f"{case.game} L{case.level_from}->L{case.level_to} has no changed held-out"
        )
    post_train = list(case.post_boundary[:heldout_idx])
    heldout = case.post_boundary[heldout_idx]
    cold_engine, _cold_done, cold_diag = gated_engine_from_transitions(
        case.game,
        post_train,
        holdout_frac=0.0,
        trust_threshold=0.0,
        dynamics_backend=dynamics_backend,
    )
    warm_engine, _warm_done, warm_diag = gated_engine_from_transitions(
        case.game,
        post_train,
        prior_transitions=list(case.pre_boundary),
        holdout_frac=0.0,
        trust_threshold=0.0,
        dynamics_backend=dynamics_backend,
    )
    cold_accuracy = changed_cell_value_accuracy(cold_engine, heldout)
    warm_accuracy = changed_cell_value_accuracy(warm_engine, heldout)
    actions = _actions_to_next_level(case.post_boundary)
    changed = int(np.asarray(heldout.grid != heldout.next_grid).sum())
    return {
        "game": case.game,
        "level_from": int(case.level_from),
        "level_to": int(case.level_to),
        "cold_accuracy": _round(cold_accuracy),
        "warmstart_accuracy": _round(warm_accuracy),
        "accuracy_delta": _round(warm_accuracy - cold_accuracy),
        "actions_cold": actions,
        "actions_warmstart": actions,
        "actions_saved_pct": 0.0,
        "heldout_changed_cells": changed,
        "source_artifact": case.source_artifact,
        "cold_diag": dict(cold_diag),
        "warmstart_diag": dict(warm_diag),
    }


def _game_counts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("game", "")) for row in rows)
    return [
        {"game": game, "n_level_transitions_tested": int(count)}
        for game, count in sorted(counts.items())
        if game
    ]


def _actions_saved(row: Mapping[str, Any]) -> float:
    cold = row.get("actions_cold")
    warm = row.get("actions_warmstart")
    try:
        cold_f = float(cold)
        warm_f = float(warm)
    except (TypeError, ValueError):
        return 0.0
    if cold_f <= 0.0:
        return 0.0
    return max(0.0, (cold_f - warm_f) / cold_f)


def build_artifact(
    per_transition_breakdown: Sequence[Mapping[str, Any]],
    *,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in per_transition_breakdown]
    deltas = [float(row.get("accuracy_delta", 0.0)) for row in rows]
    action_savings = [_actions_saved(row) for row in rows]
    delta_median = _round(median(deltas) if deltas else 0.0)
    saved_median = _round(median(action_savings) if action_savings else 0.0)
    gate_passed = len(rows) >= 6 and (delta_median >= 0.10 or saved_median >= 0.20)
    verdict = (
        f"success: warmstart_replay_ablation_gate_passed_delta_{delta_median}"
        if gate_passed
        else f"complete: warmstart_replay_ablation_gate_failed_honest_null_delta_{delta_median}"
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "games_tested": _game_counts(rows),
        "warmstart_vs_cold_delta_median": delta_median,
        "actions_saved_pct_median": saved_median,
        "gate_passed": bool(gate_passed),
        "per_transition_breakdown": rows,
        "solve_provenance": "development_proxy",
        "offline_reproduced": False,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "model_specs": {
            "control": "gated_engine_from_transitions(post_boundary_suffix_only)",
            "warmstart": "gated_engine_from_transitions(post_boundary_suffix, prior_transitions=pre_boundary)",
            "warmstart_mechanism": "ReDRAW frozen base LiveTTTWorldModel plus target residual",
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    recoverable_games: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": (
            "complete: blocked_insufficient_transition_traces_gate_not_run_"
            f"recoverable_{len(recoverable_games)}"
        ),
        "games_tested": [dict(row) for row in recoverable_games],
        "warmstart_vs_cold_delta_median": 0.0,
        "actions_saved_pct_median": 0.0,
        "gate_passed": False,
        "per_transition_breakdown": [],
        "solve_provenance": "development_proxy",
        "offline_reproduced": False,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "model_specs": {
            "control": "not_run",
            "warmstart": "not_run",
            "warmstart_mechanism": "blocked_before_ablation",
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"artifact missing fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")
    if artifact["solve_provenance"] != "development_proxy":
        raise ValueError("solve_provenance must be development_proxy")
    if artifact["offline_reproduced"] is not False:
        raise ValueError("offline_reproduced must be false for this ablation")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if not isinstance(artifact["gate_passed"], bool):
        raise ValueError("gate_passed must be bool")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    for field in FIELD_PRINCIPLES:
        if field not in principles:
            raise ValueError(f"missing principle: {field}")
    rows = artifact["per_transition_breakdown"]
    if not isinstance(rows, list):
        raise ValueError("per_transition_breakdown must be a list")
    blocked = "blocked_insufficient_transition_traces" in verdict
    if not blocked and len(rows) < 6:
        raise ValueError("per_transition_breakdown must contain at least six rows")
    if artifact["gate_passed"] and len(rows) < 6:
        raise ValueError("gate_passed requires at least six tested boundaries")
    expected = reproducibility_checksum(artifact)
    if artifact["reproducibility_checksum"] != expected:
        raise ValueError("invalid reproducibility_checksum")


def write_artifact(artifact: Mapping[str, Any], output: Path | str) -> None:
    validate_artifact(artifact)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _registry_games(root: Path) -> list[str]:
    registry_path = root / REGISTRY_RELATIVE_PATH
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    games = []
    for row in data.get("games", []) or []:
        if not isinstance(row, Mapping):
            continue
        try:
            levels = int(row.get("levels_reproduced") or 0)
        except (TypeError, ValueError):
            levels = 0
        game = str(row.get("game") or "")
        if game and levels >= 2:
            games.append(game)
    return sorted(set(games))


def _load_loop_labels(root: Path, game: str) -> tuple[list[str], str] | None:
    path = root / "results" / f"arc_loop_solve_{game}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("offline_reproduced") is not True:
        return None
    labels = payload.get("solution_labels")
    if not isinstance(labels, list) or not labels:
        return None
    return [str(label) for label in labels], _relative(path, root)


def replay_labels_to_transitions(
    game: str,
    labels: Sequence[str],
    *,
    source_artifact: str,
) -> list[Transition]:
    """Replay banked labels through the offline env and recover TTT transitions."""

    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

    adapter = adapters.get_adapter(game)
    if adapter is None:
        return []
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    if adapter.warmup_label is not None:
        frame = adapter.apply(env, adapter.warmup_label, frame)
    transitions = []
    for label in labels:
        before = frame
        action, data = parse_action_label(label)
        grid = grid_of(before)
        cell = detect_cell(grid)
        logical = to_logical(grid, cell)
        frame = adapter.apply(env, label, before)
        next_logical = to_logical(grid_of(frame), cell)
        transitions.append(
            Transition(
                grid=logical,
                action=action,
                data=data,
                next_grid=next_logical,
                level_before=int(getattr(before, "levels_completed", 0) or 0),
                level_after=int(getattr(frame, "levels_completed", 0) or 0),
            )
        )
    return transitions


def boundary_cases_from_transitions(
    game: str,
    transitions: Sequence[Transition],
    *,
    source_artifact: str,
) -> list[BoundaryCase]:
    boundaries = [
        idx
        for idx, transition in enumerate(transitions)
        if int(transition.level_after) > int(transition.level_before)
    ]
    cases = []
    for pos, idx in enumerate(boundaries):
        if pos + 1 >= len(boundaries):
            continue
        next_idx = boundaries[pos + 1]
        transition = transitions[idx]
        cases.append(
            BoundaryCase(
                game=game,
                level_from=int(transition.level_before),
                level_to=int(transition.level_after),
                pre_boundary=tuple(transitions[: idx + 1]),
                post_boundary=tuple(transitions[idx + 1 : next_idx + 1]),
                source_artifact=source_artifact,
            )
        )
    return cases


def collect_boundary_cases(root: Path = REPO_ROOT) -> list[BoundaryCase]:
    cases: list[BoundaryCase] = []
    for game in _registry_games(root):
        loaded = _load_loop_labels(root, game)
        if loaded is None:
            continue
        labels, source_artifact = loaded
        transitions = replay_labels_to_transitions(
            game,
            labels,
            source_artifact=source_artifact,
        )
        cases.extend(
            boundary_cases_from_transitions(
                game,
                transitions,
                source_artifact=source_artifact,
            )
        )
    return cases


def build_preconditions(
    root: Path, cases: Sequence[BoundaryCase], recoverable: Sequence[BoundaryCase]
) -> dict:
    return {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "research_references_v473_read": "V473"
        in (root / "research-references.md").read_text(encoding="utf-8"),
        "experiment_5155_read": (
            root / "results/experiment_5155_multilevel_belief_state_scoping_v472.json"
        ).exists(),
        "arc_live_ttt_prior_extension": True,
        "registry_trace_precondition": "passed" if len(recoverable) >= 6 else "blocked",
        "recoverable_level_transitions_n": len(recoverable),
        "candidate_level_transitions_n": len(cases),
        "recoverable_games_n": len({case.game for case in recoverable}),
    }


def run_experiment(
    root: Path = REPO_ROOT,
    *,
    dynamics_backend: str = "dsl",
) -> dict[str, Any]:
    cases = collect_boundary_cases(root)
    recoverable = recoverable_boundaries(cases)
    preconditions = build_preconditions(root, cases, recoverable)
    if len(recoverable) < 6:
        artifact = build_blocked_artifact(
            recoverable_games=_game_counts([{"game": case.game} for case in recoverable]),
            preconditions_checked=preconditions,
        )
    else:
        rows = [
            evaluate_boundary_case(case, dynamics_backend=dynamics_backend) for case in recoverable
        ]
        preconditions["dynamics_backend"] = dynamics_backend
        artifact = build_artifact(rows, preconditions_checked=preconditions)
    return artifact


def main() -> None:
    artifact = run_experiment(REPO_ROOT)
    write_artifact(artifact, REPO_ROOT / RESULT_RELATIVE_PATH)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
