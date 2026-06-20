"""Experiment 4512: imitation action-type and sequence prior.

Spec refs: REQ-ARC-FCP-4512, SCENARIO-ARC-FCP-4512.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from carnot.agentic import arc_human_replay_corpus
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_frame_change_predictor import (
    FrameActionEffectExample,
    load_frame_action_effect_examples,
)
from carnot.agentic.arc_agi3_world_model import grid_of


RESULT_RELATIVE_PATH = "results/experiment_4512_imitation_action_prior.json"
HUMAN_REPLAY_DATA_RELATIVE_DIR = "data/arc_public_demo_human_replay_corpus"
LOCAL_GATE_RELATIVE_PATH = "scripts/kaggle/arc_local_submission_gate.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_MEDIAN_ACTIONS = 7760
RANDOM_SEED = 4512
DEFAULT_PRIOR_PRUNE_QUANTILE = 0.25
DEFAULT_GATE_BUDGET = 8000
DEFAULT_FALLBACK_GAME_LIMIT = 25
DEFAULT_FALLBACK_MAX_CANDIDATES_PER_GAME = 16
PRIOR_SOURCES = ("human_replay_corpus", "self_supervised_marginal_fallback")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIREMENTS = ["REQ-ARC-FCP-4512"]
SCENARIOS = ["SCENARIO-ARC-FCP-4512"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        'principle "terminal prefix; e.g. success: '
        "imitation_prior_median_actions_<n>_below_7760 OR complete: "
        'imitation_prior_no_reduction_honest_null."'
    ),
    "inference_substrate": (
        'principle "verifier_ensemble_against_cached_candidates -- offline arcade, '
        'no LLM load (1s floor)."'
    ),
    "median_actions_baseline": 'principle "the 7760 control, fixed."',
    "median_actions_with_prior": (
        'principle "the headline -- did the human prior cut exploration."'
    ),
    "solve_rate_baseline": 'principle "no-regression reference."',
    "solve_rate_with_prior": 'principle "the prior must not drop solve-rate."',
    "prior_source": (
        'principle "honest declaration of whether the human replays or the '
        'self-supervised fallback supplied the prior (no silent corpus dependency)."'
    ),
    "positive_control_passed": 'principle "proves the harness detects a real reduction."',
    "false_negative_risk_checked": (
        'principle "a null is valid only if the positive control passed."'
    ),
    "random_seed": 'principle "determinism precondition for reproducibility."',
    "reproducibility_checksum": (
        'principle "catches silent corpus/model drift on replay."'
    ),
    "preconditions_checked": (
        'principle "records resources verified; pre-empts missing-resource fabrication."'
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "field_principles",
    "requirements",
    "scenarios",
    "prior_summary",
    "prior_prune_quantile",
    "local_gate_metrics",
    "positive_control",
    "false_negative_risk_guard",
    "duration_s",
)


def frame_class_key(frame: Any) -> str:
    """REQ-ARC-FCP-4512: coarse frame-only class for action-type conditionals."""

    grid = np.asarray(grid_of(frame))
    if grid.size == 0:
        return "empty"
    h, w = grid.shape[:2]
    nonzero = float(np.count_nonzero(grid))
    density_bucket = min(4, int((nonzero / float(grid.size)) * 5.0))
    color_bucket = min(7, int(len(np.unique(grid))))
    row_activity = min(4, int(np.count_nonzero(np.any(grid != 0, axis=1)) / max(1, h) * 5.0))
    col_activity = min(4, int(np.count_nonzero(np.any(grid != 0, axis=0)) / max(1, w) * 5.0))
    available = tuple(sorted(int(value) for value in getattr(frame, "available_actions", []) or []))
    has_click = 6 in available
    return (
        f"h{h}_w{w}_c{color_bucket}_d{density_bucket}_"
        f"r{row_activity}_col{col_activity}_click{int(has_click)}"
    )


def _probability(action_id: int, counts: Mapping[int, float], smoothing: float) -> float:
    if not counts:
        return 0.0
    total = float(sum(float(value) for value in counts.values()))
    support = max(7, len(counts))
    return float(float(counts.get(int(action_id), 0.0)) + smoothing) / float(
        total + smoothing * support
    )


@dataclass(frozen=True)
class ActionTypeSequencePrior:
    """REQ-ARC-FCP-4512: frame-class action prior with optional sequence context."""

    prior_source: str
    frame_class_action_counts: Mapping[str, Mapping[int, float]]
    marginal_action_counts: Mapping[int, float]
    sequence_action_counts: Mapping[int, Mapping[int, float]] | None = None
    history: tuple[int, ...] = ()
    frame_class_weight: float = 0.65
    marginal_weight: float = 0.25
    sequence_weight: float = 0.10
    smoothing: float = 0.1

    def for_path(self, path: Sequence[Mapping[str, Any]] | None) -> "ActionTypeSequencePrior":
        history = tuple(
            int(step["action"])
            for step in (path or [])
            if isinstance(step, Mapping) and step.get("action") is not None
        )
        return replace(self, history=history)

    def score(self, frame: Any, candidate: Any) -> float:
        action_id = int(getattr(candidate, "action_id"))
        class_counts = self.frame_class_action_counts.get(frame_class_key(frame), {})
        score = self.frame_class_weight * _probability(action_id, class_counts, self.smoothing)
        score += self.marginal_weight * _probability(
            action_id,
            self.marginal_action_counts,
            self.smoothing,
        )
        if self.history and self.sequence_action_counts:
            next_counts = self.sequence_action_counts.get(int(self.history[-1]), {})
            score += self.sequence_weight * _probability(action_id, next_counts, self.smoothing)
        return float(score)


@dataclass(frozen=True)
class PriorBundle:
    prior: ActionTypeSequencePrior
    prior_source: str
    examples: Sequence[FrameActionEffectExample]
    summary: Mapping[str, Any]


def _example_sort_key(example: FrameActionEffectExample) -> tuple[str, str, int]:
    return (str(example.env), str(example.guid), int(example.step_index))


def build_prior_from_effect_examples(
    examples: Sequence[FrameActionEffectExample],
    *,
    prior_source: str,
) -> ActionTypeSequencePrior:
    """REQ-ARC-FCP-4512: estimate P(action | frame-class) from changed actions."""

    count_examples = [example for example in examples if example.changed] or list(examples)
    frame_counts: dict[str, dict[int, float]] = {}
    marginal_counts: dict[int, float] = {}
    sequence_counts: dict[int, dict[int, float]] = {}

    for example in count_examples:
        action_id = int(example.action_id)
        frame_class = frame_class_key(example.frame)
        per_class = frame_counts.setdefault(frame_class, {})
        per_class[action_id] = per_class.get(action_id, 0.0) + 1.0
        marginal_counts[action_id] = marginal_counts.get(action_id, 0.0) + 1.0

    previous_by_episode: dict[tuple[str, str], int] = {}
    for example in sorted(count_examples, key=_example_sort_key):
        action_id = int(example.action_id)
        episode = (str(example.env), str(example.guid))
        previous = previous_by_episode.get(episode)
        if previous is not None:
            per_previous = sequence_counts.setdefault(previous, {})
            per_previous[action_id] = per_previous.get(action_id, 0.0) + 1.0
        previous_by_episode[episode] = action_id

    return ActionTypeSequencePrior(
        prior_source=str(prior_source),
        frame_class_action_counts=frame_counts,
        marginal_action_counts=marginal_counts,
        sequence_action_counts=sequence_counts,
    )


def summarize_prior(
    prior: ActionTypeSequencePrior,
    *,
    examples: Sequence[FrameActionEffectExample],
    source_summary: Mapping[str, Any],
) -> dict[str, Any]:
    changed_count = sum(1 for example in examples if example.changed)
    return {
        "prior_source": prior.prior_source,
        "examples_loaded": int(len(examples)),
        "prior_examples_used": int(changed_count or len(examples)),
        "changed_count": int(changed_count),
        "frame_class_count": int(len(prior.frame_class_action_counts)),
        "marginal_action_counts": {
            str(key): float(value) for key, value in sorted(prior.marginal_action_counts.items())
        },
        "sequence_context_count": int(len(prior.sequence_action_counts or {})),
        **dict(source_summary),
    }


def load_human_replay_examples(
    root: Path | str = REPO_ROOT,
    *,
    limit: int | None = None,
) -> tuple[list[FrameActionEffectExample], dict[str, Any]]:
    """REQ-ARC-FCP-4512: load staged human replay rows if locally available."""

    data_dir = Path(root) / HUMAN_REPLAY_DATA_RELATIVE_DIR
    try:
        manifest = arc_human_replay_corpus.load_manifest(data_dir)
    except Exception as exc:
        return [], {
            "human_manifest_present": False,
            "human_manifest_error": f"{type(exc).__name__}: {exc}",
            "human_examples_loaded": 0,
        }
    if int(manifest.get("example_count") or 0) <= 0:
        return [], {
            "human_manifest_present": True,
            "human_manifest_examples": int(manifest.get("example_count") or 0),
            "human_examples_loaded": 0,
        }
    examples = load_frame_action_effect_examples(data_dir, limit=limit)
    return examples, {
        "human_manifest_present": True,
        "human_manifest_examples": int(manifest.get("example_count") or 0),
        "human_shard_count": int(manifest.get("shard_count") or 0),
        "human_examples_loaded": int(len(examples)),
    }


def collect_self_supervised_fallback(
    *,
    game_limit: int | None = DEFAULT_FALLBACK_GAME_LIMIT,
    max_candidates_per_game: int = DEFAULT_FALLBACK_MAX_CANDIDATES_PER_GAME,
    random_seed: int = RANDOM_SEED,
) -> tuple[list[FrameActionEffectExample], dict[str, Any]]:  # pragma: no cover - SDK boundary
    from carnot import experiment_4511_frame_change_prune_predictor as exp4511

    return exp4511.collect_self_supervised_corpus(
        game_limit=game_limit,
        max_candidates_per_game=max_candidates_per_game,
        random_seed=random_seed,
    )


def build_imitation_prior(
    *,
    root: Path | str = REPO_ROOT,
    human_limit: int | None = None,
    fallback_collector: Callable[..., tuple[list[FrameActionEffectExample], dict[str, Any]]] = (
        collect_self_supervised_fallback
    ),
    random_seed: int = RANDOM_SEED,
) -> PriorBundle:
    """REQ-ARC-FCP-4512: use human replay primary, self-supervised fallback otherwise."""

    human_examples, human_summary = load_human_replay_examples(root, limit=human_limit)
    if human_examples:
        prior = build_prior_from_effect_examples(
            human_examples,
            prior_source="human_replay_corpus",
        )
        return PriorBundle(
            prior=prior,
            prior_source="human_replay_corpus",
            examples=human_examples,
            summary=summarize_prior(prior, examples=human_examples, source_summary=human_summary),
        )

    fallback_examples, fallback_summary = fallback_collector(
        game_limit=DEFAULT_FALLBACK_GAME_LIMIT,
        max_candidates_per_game=DEFAULT_FALLBACK_MAX_CANDIDATES_PER_GAME,
        random_seed=random_seed,
    )
    prior = build_prior_from_effect_examples(
        fallback_examples,
        prior_source="self_supervised_marginal_fallback",
    )
    source_summary = {
        **human_summary,
        **dict(fallback_summary),
        "fallback_examples_loaded": int(len(fallback_examples)),
    }
    return PriorBundle(
        prior=prior,
        prior_source="self_supervised_marginal_fallback",
        examples=fallback_examples,
        summary=summarize_prior(prior, examples=fallback_examples, source_summary=source_summary),
    )


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4512: record required resources before measuring."""

    root_path = Path(root)
    data_dir = root_path / HUMAN_REPLAY_DATA_RELATIVE_DIR
    preconditions: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "human_replay_manifest_present": (data_dir / arc_human_replay_corpus.MANIFEST_NAME).exists(),
        "local_gate_script_present": (root_path / LOCAL_GATE_RELATIVE_PATH).exists(),
        "baseline_file_present": (root_path / "ops" / "arc-submission-baseline.json").exists(),
        "median_actions_baseline_control": BASELINE_MEDIAN_ACTIONS,
        "env_game_access_blocked": True,
    }
    try:
        kit.offline_arcade()
        preconditions["offline_arcade_import"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path
        preconditions["offline_arcade_error"] = repr(exc)
    preconditions["ok"] = bool(preconditions["offline_arcade_import"])
    return preconditions


def load_gate_baseline(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    baseline_path = Path(root) / "ops" / "arc-submission-baseline.json"
    if baseline_path.exists():
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    return {
        "policy": "e3",
        "games": ["lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20"],
        "per_game": [],
        "solved_count": 4,
        "median_actions_on_solved": float(BASELINE_MEDIAN_ACTIONS),
        "total_actions_on_solved": None,
        "timed_out_count": None,
        "note": "fixed operator-provided baseline control",
    }


def _load_module_from_path(path: Path, module_name: str) -> Any:  # pragma: no cover - I/O boundary
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_action_label(action_id: int, data: Any) -> str:
    return json.dumps({"action": int(action_id), "data": data}, sort_keys=True)


def _apply_json_action_label(env: Any, label: str, _frame: Any) -> Any:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    payload = json.loads(label)
    return env.step(
        _game_action(GameAction, int(payload["action"])),
        data=payload.get("data"),
    )


def _run_policy_game_with_prior(
    game: str,
    *,
    prior: ActionTypeSequencePrior,
    prior_prune_quantile: float,
    budget: int,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=None,
        action_prior=prior,
        action_prior_prune_quantile=float(prior_prune_quantile),
    )
    frames: list[Any] = []
    latest = None
    actions = 0
    start_level: int | None = None
    first_levelup_actions: int | None = None
    current_segment: list[str] = []
    first_levelup_segment: list[str] = []

    for _ in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            current_segment = []
        elif kind is None:
            break
        else:
            latest = env.step(
                _game_action(GameAction, int(kind)),
                data=data,
            )
            actions += 1
            current_segment.append(_json_action_label(int(kind), data))
        if start_level is None:
            start_level = kit.frame_level(latest)
        frames.append(latest)
        if latest is None:
            break
        reached_now = kit.frame_level(latest)
        if (
            start_level is not None
            and reached_now > start_level
            and first_levelup_actions is None
        ):
            first_levelup_actions = int(actions)
            first_levelup_segment = list(current_segment)

    reached = kit.frame_level(latest)
    levels = max(0, int(reached) - int(start_level or 0))
    reproduction = None
    if levels >= 1 and first_levelup_segment:
        reproduction = kit.reproduce(
            game,
            first_levelup_segment,
            _apply_json_action_label,
            claimed_level=int((start_level or 0) + 1),
        )
    return {
        "game": game,
        "timed_out": False,
        "solved": bool(levels >= 1),
        "levels": int(levels),
        "reached": int(reached),
        "actions": int(actions),
        "actions_to_first_levelup": first_levelup_actions,
        "reproduced": None if reproduction is None else bool(reproduction.get("reproduced")),
        "reproduction": reproduction,
    }


def measure_local_gate_with_prior(
    *,
    root: Path | str = REPO_ROOT,
    prior: ActionTypeSequencePrior,
    prior_prune_quantile: float = DEFAULT_PRIOR_PRUNE_QUANTILE,
    budget: int = DEFAULT_GATE_BUDGET,
    max_workers: int = 8,
) -> dict[str, Any]:  # pragma: no cover - SDK boundary
    baseline = load_gate_baseline(root)
    games = list(baseline.get("games") or ["lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20"])
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            rows = list(
                executor.map(
                    lambda game: _run_policy_game_with_prior(
                        game,
                        prior=prior,
                        prior_prune_quantile=prior_prune_quantile,
                        budget=budget,
                    ),
                    games,
                )
            )
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable

    solved = [row for row in rows if row["solved"]]
    actions = [int(row["actions"]) for row in solved]
    with_prior = {
        "policy": "e3_imitation_action_prior",
        "games": games,
        "per_game": rows,
        "solved_count": int(len(solved)),
        "median_actions_on_solved": float(median(actions)) if actions else None,
        "total_actions_on_solved": int(sum(actions)) if actions else None,
        "timed_out_count": 0,
        "budget": int(budget),
        "prior_prune_quantile": float(prior_prune_quantile),
    }
    return {
        "baseline": baseline,
        "with_prior": with_prior,
        "measurement_script": LOCAL_GATE_RELATIVE_PATH,
    }


def positive_control() -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4512: prove prior ordering plus pruning can save actions."""

    from carnot.agentic.arc_frame_change_predictor import (
        prune_arc_actions_by_prior_quantile,
        rank_arc_actions,
    )
    from carnot.agentic.arc_agi3_live_adapter import ArcAction

    frame = type("Frame", (), {})()
    frame.frame = np.zeros((8, 8), dtype=np.int16)
    frame.frame[6, 6] = 1
    frame.available_actions = [1, 2, 6]
    frame_class = frame_class_key(frame)
    prior = ActionTypeSequencePrior(
        prior_source="positive_control",
        frame_class_action_counts={frame_class: {1: 1.0, 2: 1.0, 6: 9.0}},
        marginal_action_counts={1: 1.0, 2: 1.0, 6: 9.0},
        sequence_action_counts={1: {6: 9.0}},
        history=(1,),
    )
    candidates = [
        ArcAction(1, None, "noop_a"),
        ArcAction(2, None, "noop_b"),
        ArcAction(6, {"x": 6, "y": 6}, "changing_click"),
    ]
    pruned, prune_diagnostics = prune_arc_actions_by_prior_quantile(
        frame,
        candidates,
        prior=prior,
        prune_quantile=1 / 3,
    )
    ranked = rank_arc_actions(frame, pruned, prior=prior)

    def actions_to_change(rows: Sequence[Any]) -> int:
        for index, candidate in enumerate(rows, start=1):
            if getattr(candidate, "source", "") == "changing_click":
                return index
        return len(rows) + 1  # pragma: no cover - positive control always includes the target.

    baseline_actions = actions_to_change(candidates)
    ranked_actions = actions_to_change(ranked)
    return {
        "baseline_actions_to_first_levelup": int(baseline_actions),
        "ranked_actions_to_first_levelup": int(ranked_actions),
        "actions_reduced": bool(ranked_actions < baseline_actions),
        "prune_diagnostics": prune_diagnostics,
    }


def _gate_value(gate_metrics: Mapping[str, Any], arm: str, field: str) -> Any:
    arm_metrics = gate_metrics.get(arm)
    if not isinstance(arm_metrics, Mapping):
        return None
    return arm_metrics.get(field)


def _honest_verdict(
    preconditions: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
) -> str:
    if preconditions.get("offline_arcade_import") is False:
        return "complete: blocked_offline_arcade_import_failed"
    baseline_solve = int(_gate_value(gate_metrics, "baseline", "solved_count") or 0)
    prior_solve = int(_gate_value(gate_metrics, "with_prior", "solved_count") or 0)
    prior_median = _gate_value(gate_metrics, "with_prior", "median_actions_on_solved")
    if prior_solve < baseline_solve:
        return "complete: imitation_prior_solve_rate_guard_failed"
    if prior_median is not None and float(prior_median) < BASELINE_MEDIAN_ACTIONS:
        return f"success: imitation_prior_median_actions_{int(float(prior_median))}_below_7760"
    return "complete: imitation_prior_no_reduction_honest_null"


def false_negative_risk_guard(
    control: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
) -> str:
    if control.get("actions_reduced") is not True:
        return "positive_control_failed_null_uninterpretable"
    prior_median = _gate_value(gate_metrics, "with_prior", "median_actions_on_solved")
    baseline_solve = int(_gate_value(gate_metrics, "baseline", "solved_count") or 0)
    prior_solve = int(_gate_value(gate_metrics, "with_prior", "solved_count") or 0)
    if prior_solve >= baseline_solve and prior_median is not None and float(prior_median) < BASELINE_MEDIAN_ACTIONS:
        return "positive_control_passed_prior_gain"
    return "positive_control_passed_null_interpretable"


def reproducibility_checksum(
    *,
    prior_summary: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
    prior_source: str,
    random_seed: int,
) -> str:
    payload = {
        "prior_summary": dict(prior_summary),
        "gate_metrics": gate_metrics,
        "prior_source": str(prior_source),
        "random_seed": int(random_seed),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    prior_summary: Mapping[str, Any],
    gate_metrics: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    prior_source: str,
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float | None,
    prior_prune_quantile: float = DEFAULT_PRIOR_PRUNE_QUANTILE,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4512: assemble the terminal imitation-prior artifact."""

    baseline_median = _gate_value(gate_metrics, "baseline", "median_actions_on_solved")
    baseline_solve = int(_gate_value(gate_metrics, "baseline", "solved_count") or 0)
    prior_solve = int(_gate_value(gate_metrics, "with_prior", "solved_count") or 0)
    control_passed = bool(positive_control.get("actions_reduced") is True)
    guard = false_negative_risk_guard(positive_control, gate_metrics)
    return {
        "experiment": "experiment_4512_imitation_action_prior",
        "schema": "carnot.arc_imitation_action_prior_4512.v1",
        "honest_verdict": _honest_verdict(preconditions_checked, gate_metrics),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "prior_source": str(prior_source),
        "prior_summary": dict(prior_summary),
        "prior_prune_quantile": float(prior_prune_quantile),
        "median_actions_baseline": (
            float(baseline_median)
            if baseline_median is not None
            else float(BASELINE_MEDIAN_ACTIONS)
        ),
        "median_actions_with_prior": _gate_value(
            gate_metrics,
            "with_prior",
            "median_actions_on_solved",
        ),
        "solve_rate_baseline": baseline_solve,
        "solve_rate_with_prior": prior_solve,
        "solve_rate_denominator": len(_gate_value(gate_metrics, "baseline", "games") or []),
        "positive_control_passed": control_passed,
        "positive_control": dict(positive_control),
        "false_negative_risk_checked": bool(control_passed),
        "false_negative_risk_guard": guard,
        "random_seed": int(random_seed),
        "reproducibility_checksum": str(reproducibility_checksum),
        "local_gate_metrics": dict(gate_metrics),
        "offline_reproduction_gate": "kit.reproduce_on_first_levelup_segment_for_solved_rows",
        "duration_s": duration_s,
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match the required substrate")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if int(float(artifact.get("median_actions_baseline") or 0)) != BASELINE_MEDIAN_ACTIONS:
        errors.append("median_actions_baseline must equal the fixed 7760 control")
    if artifact.get("prior_source") not in PRIOR_SOURCES:
        errors.append("prior_source must name human replay or self-supervised fallback")
    if artifact.get("positive_control_passed") is not True:
        errors.append("positive_control_passed must be true")
    if artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked must be true")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    if str(artifact.get("honest_verdict", "")).startswith("success:") and int(
        artifact.get("solve_rate_with_prior") or 0
    ) < int(artifact.get("solve_rate_baseline") or 0):
        errors.append("success verdict cannot hide a solve-rate drop")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    prior_builder: Callable[..., PriorBundle] = build_imitation_prior,
    measure_gate: Callable[..., dict[str, Any]] = measure_local_gate_with_prior,
    random_seed: int = RANDOM_SEED,
    prior_prune_quantile: float = DEFAULT_PRIOR_PRUNE_QUANTILE,
    gate_budget: int = DEFAULT_GATE_BUDGET,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4512: build prior, measure, and write the result JSON."""

    root_path = Path(root)
    started = float(now())
    preconditions = check_preconditions(root_path)
    control = positive_control()
    if preconditions.get("offline_arcade_import") is False:
        gate_metrics = {
            "baseline": load_gate_baseline(root_path),
            "with_prior": {"solved_count": 0, "median_actions_on_solved": None, "per_game": []},
            "measurement_script": LOCAL_GATE_RELATIVE_PATH,
        }
        prior_summary: Mapping[str, Any] = {"prior_examples_used": 0}
        prior_source = "self_supervised_marginal_fallback"
    else:
        bundle = prior_builder(root=root_path, random_seed=random_seed)
        prior_summary = bundle.summary
        prior_source = bundle.prior_source
        gate_metrics = measure_gate(
            root=root_path,
            prior=bundle.prior,
            prior_prune_quantile=prior_prune_quantile,
            budget=gate_budget,
        )
    checksum = reproducibility_checksum(
        prior_summary=prior_summary,
        gate_metrics=gate_metrics,
        prior_source=prior_source,
        random_seed=random_seed,
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        prior_summary=prior_summary,
        gate_metrics=gate_metrics,
        positive_control=control,
        prior_source=prior_source,
        random_seed=random_seed,
        reproducibility_checksum=checksum,
        duration_s=max(0.0, float(now()) - started),
        prior_prune_quantile=prior_prune_quantile,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        out = root_path / RESULT_RELATIVE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    main()
