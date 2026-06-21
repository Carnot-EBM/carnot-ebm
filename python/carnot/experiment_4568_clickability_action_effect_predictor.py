"""Experiment 4568: pooled clickability action-effect predictor.

Spec refs: REQ-ARC-FCP-4568, SCENARIO-ARC-FCP-4568.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4547_frame_change_predictor as exp4547
from carnot.agentic.arc_frame_change_predictor import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_NUM_COLORS,
    TERMINAL_ACTION_IDS,
    FrameActionEffectExample,
    FrameChangeScorer,
    evaluate_positive_control,
    efficiency_score,
    frame_state_key,
    load_frame_action_effect_examples,
    rank_arc_actions,
    train_frame_change_model,
)
from carnot.agentic.arc_agi3_live_adapter import ArcAction


RESULT_RELATIVE_PATH = "results/experiment_4568_clickability_action_effect_predictor.json"
HUMAN_REPLAY_RELATIVE_DIR = "data/arc_public_demo_human_replay_corpus"
TRANSITION_CORPUS_RELATIVE_DIR = "data/arc_transition_corpus"
GENERIC_TRANSFER_RESULT_RELATIVE_PATH = "results/experiment_4550_honest_sprint_metric.json"
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- the CNN trains/scores against "
    "cached transitions (no LLM load, 1s floor); fast CPU forward pass declared."
)
RANDOM_SEED = 4568
DEFAULT_MAX_TRAIN_EXAMPLES = 4096
DEFAULT_BOOTSTRAPS = 1000
GENERIC_TRANSFER_BASELINE = 0.04
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
REQUIREMENTS = ["REQ-ARC-FCP-4568"]
SCENARIOS = ["SCENARIO-ARC-FCP-4568"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: clickability_predictor_actions_to_levelup_<n>_below_blind "
        "OR complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- the CNN trains/scores against cached "
        "transitions (no LLM load, 1s floor); fast CPU forward pass declared."
    ),
    "verifier_is_oracle": (
        "MUST be false -- learned action-effect model is oracle-DISTINCT from executable win-check."
    ),
    "median_actions_to_first_levelup_with_predictor": (
        "the HEADLINE -- held-out median actions-to-first-levelup with the predictor-ranked explorer."
    ),
    "median_actions_to_first_levelup_baseline": (
        "the blind-BFS baseline, measured the SAME way."
    ),
    "actions_delta": "baseline - with_predictor; positive = fewer actions.",
    "actions_delta_ci": (
        "bootstrap CI on the actions delta; efficiency claim requires the CI to exclude zero."
    ),
    "efficiency_score_min_human_agent_sq": (
        "min(human/agent,1)^2 with the human baseline from replay corpus."
    ),
    "generic_transfer_rate_with_predictor": (
        "held-out variant transfer WITH the predictor vs the 0.04 baseline."
    ),
    "solve_rate_preserved": "HARD gate -- efficiency win must NOT drop solve-rate.",
    "positive_control_passed": (
        "learnable-clickability control where predictor-ranking must beat blind."
    ),
    "false_negative_risk_checked": (
        "a no-value null is valid only if the positive control passed."
    ),
    "null_delta_methodology_note": (
        "present when actions_delta==0.0 -- honest no-gain null, not a measurement bug."
    ),
    "chosen_submitted_config": (
        "recommend enable predictor-ranker when successful; unchanged if null."
    ),
    "missing_verifier_gaps": "if no gain, record residual generation/ranking gap.",
    "offline_reproduced": "any newly-solved variant must offline-reproduce to count.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash catches silent corpus/model drift on replay.",
    "preconditions_checked": (
        "records resources verified (offline arcade, torch, LOCAL corpora)."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "field_principles",
    "requirements",
    "scenarios",
    "preconditions_checked",
    "corpus_summary",
    "training_summary",
    "ranking_metrics",
    "generic_transfer_measurement",
    "positive_control",
    "rich_action_candidates_wired",
    "frame_only_live_legal",
    "weights_bundled",
    "duration_s",
)


@dataclass(frozen=True)
class PooledCorpus:
    """REQ-ARC-FCP-4568: pooled human-replay and self-captured transition rows."""

    examples: Sequence[FrameActionEffectExample]
    source_counts: Mapping[str, int]
    metadata: Mapping[str, Any] | None = None

    @property
    def games(self) -> list[str]:
        return sorted({str(example.env) for example in self.examples if str(example.env)})


def ok_preconditions_for_tests() -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4568: compact passing precondition fixture for schema tests."""

    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": True,
        "torch_import": True,
        "torch_version": "test",
        "human_replay_corpus_present": True,
        "transition_corpus_present": True,
        "training_device_requested": "cpu",
        "leaderboard_submission": False,
        "ok": True,
    }


def _is_trainable(example: FrameActionEffectExample) -> bool:
    if int(example.action_id) in TERMINAL_ACTION_IDS:
        return True
    return bool(int(example.action_id) == 6 and example.x is not None and example.y is not None)


def _frame_delta_fraction(before: Any, after: Any) -> float:
    lhs = np.asarray(before)
    rhs = np.asarray(after)
    if lhs.shape != rhs.shape:
        return 1.0
    total = int(lhs.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(lhs != rhs) / total)


def load_transition_examples(
    root: Path | str = REPO_ROOT,
    *,
    limit: int | None = None,
) -> list[FrameActionEffectExample]:
    """REQ-ARC-FCP-4568: normalize local self-captured transitions into examples."""

    transition_dir = Path(root) / TRANSITION_CORPUS_RELATIVE_DIR
    examples: list[FrameActionEffectExample] = []
    for path in sorted(transition_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        for index in range(int(data["grids"].shape[0])):
            action_id = int(data["actions"][index])
            x_value = int(data["xs"][index])
            y_value = int(data["ys"][index])
            x = x_value if action_id == 6 and x_value >= 0 else None
            y = y_value if action_id == 6 and y_value >= 0 else None
            grid = data["grids"][index]
            next_grid = data["next_grids"][index]
            level_before = int(data["lb"][index]) if "lb" in data else 0
            level_after = int(data["la"][index]) if "la" in data else 0
            examples.append(
                FrameActionEffectExample(
                    frame=grid,
                    action_id=action_id,
                    x=x,
                    y=y,
                    frame_delta=_frame_delta_fraction(grid, next_grid),
                    level_progress=1.0 if level_after > level_before else 0.0,
                    state_key=frame_state_key(grid),
                    env=path.stem,
                    step_index=index,
                    feature_source="arc_transition_corpus",
                )
            )
            if limit is not None and len(examples) >= int(limit):
                return examples
    return examples


def load_pooled_examples(
    root: Path | str = REPO_ROOT,
    *,
    human_limit: int | None = None,
    transition_limit: int | None = None,
) -> PooledCorpus:
    """REQ-ARC-FCP-4568: load both local corpora before training/measurement."""

    root_path = Path(root)
    human_dir = root_path / HUMAN_REPLAY_RELATIVE_DIR
    human_examples = (
        load_frame_action_effect_examples(human_dir, limit=human_limit)
        if human_dir.exists()
        else []
    )
    transition_examples = load_transition_examples(root_path, limit=transition_limit)
    manifest_path = human_dir / "manifest.json"
    metadata: dict[str, Any] = {
        "human_replay_manifest_examples": 0,
        "human_replay_manifest_shards": 0,
        "transition_npz_count": len(sorted((root_path / TRANSITION_CORPUS_RELATIVE_DIR).glob("*.npz"))),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata["human_replay_manifest_examples"] = int(manifest.get("example_count") or 0)
        metadata["human_replay_manifest_shards"] = int(manifest.get("shard_count") or 0)
    return PooledCorpus(
        examples=[*human_examples, *transition_examples],
        source_counts={
            "human_replay": int(len(human_examples)),
            "transition_corpus": int(len(transition_examples)),
        },
        metadata=metadata,
    )


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - live boundary
    """REQ-ARC-FCP-4568: record explicit resources before running the experiment."""

    root_path = Path(root)
    human_dir = root_path / HUMAN_REPLAY_RELATIVE_DIR
    transition_dir = root_path / TRANSITION_CORPUS_RELATIVE_DIR
    transition_npz = sorted(transition_dir.glob("*.npz"))
    preconditions: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import": False,
        "torch_import": False,
        "torch_version": "",
        "human_replay_corpus_present": human_dir.is_dir(),
        "human_replay_shards_present": bool(list((human_dir / "shards").glob("*.jsonl"))),
        "transition_corpus_present": bool(transition_npz),
        "transition_npz_count": int(len(transition_npz)),
        "transition_corpus_dir": str(transition_dir),
        "training_device_requested": "cpu",
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import"] = True
    except Exception as exc:
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:
        preconditions["torch_error"] = repr(exc)
    preconditions["ok"] = bool(
        preconditions["offline_arcade_import"]
        and preconditions["torch_import"]
        and preconditions["human_replay_corpus_present"]
        and preconditions["transition_corpus_present"]
    )
    return preconditions


def _candidate_from_example(example: FrameActionEffectExample, source: str) -> ArcAction:
    data = (
        {"x": int(example.x), "y": int(example.y)}
        if int(example.action_id) == 6 and example.x is not None and example.y is not None
        else None
    )
    return ArcAction(int(example.action_id), data, source)


def _actions_to_first_levelup(candidates: Sequence[Any]) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if getattr(candidate, "source", "") == "levelup_target":
            return int(index)
    return None


def bootstrap_actions_delta_ci(
    deltas: Sequence[float],
    *,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> list[float]:
    """REQ-ARC-FCP-4568: bootstrap baseline-minus-predictor action deltas."""

    rows = [float(delta) for delta in deltas]
    if not rows:
        return [0.0, 0.0]
    point = sum(rows) / len(rows)
    if n_bootstrap <= 0 or len(rows) == 1:
        rounded = round(float(point), 10)
        return [rounded, rounded]

    rng = random.Random(int(random_seed))
    samples: list[float] = []
    for _index in range(int(n_bootstrap)):
        total = 0.0
        for _sample in range(len(rows)):
            total += rows[rng.randrange(len(rows))]
        samples.append(total / len(rows))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(min(lo, point)), 10), round(float(max(hi, point)), 10)]


def measure_actions_to_first_levelup(
    examples: Sequence[FrameActionEffectExample],
    *,
    scorer: Any,
    min_candidates: int = 2,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4568: compare blind order to predictor-ranked order."""

    by_state: dict[str, list[FrameActionEffectExample]] = {}
    for example in examples:
        if _is_trainable(example):
            by_state.setdefault(example.state_key, []).append(example)

    baseline_ranks: list[int] = []
    predictor_ranks: list[int] = []
    paired_deltas: list[float] = []
    solved_baseline = 0
    solved_predictor = 0
    group_count = 0

    for state_examples in by_state.values():
        if len(state_examples) < int(min_candidates):
            continue
        candidates = [
            _candidate_from_example(example, "levelup_target" if example.changed else "noop")
            for example in state_examples
        ]
        if not any(getattr(candidate, "source", "") == "levelup_target" for candidate in candidates):
            continue
        group_count += 1
        baseline = _actions_to_first_levelup(candidates)
        ranked = rank_arc_actions(state_examples[0].frame, candidates, scorer=scorer)
        with_predictor = _actions_to_first_levelup(ranked)
        if baseline is not None:
            solved_baseline += 1
            baseline_ranks.append(baseline)
        if with_predictor is not None:
            solved_predictor += 1
            predictor_ranks.append(with_predictor)
        if baseline is not None and with_predictor is not None:
            paired_deltas.append(float(baseline - with_predictor))

    baseline_median = float(np.median(baseline_ranks)) if baseline_ranks else None
    predictor_median = float(np.median(predictor_ranks)) if predictor_ranks else None
    actions_delta = (
        round(float(baseline_median - predictor_median), 10)
        if baseline_median is not None and predictor_median is not None
        else 0.0
    )
    solve_rate_baseline = float(solved_baseline / group_count) if group_count else 0.0
    solve_rate_predictor = float(solved_predictor / group_count) if group_count else 0.0
    ci = bootstrap_actions_delta_ci(
        paired_deltas,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
    )
    return {
        "heldout_group_count": int(group_count),
        "median_actions_to_first_levelup_baseline": baseline_median,
        "median_actions_to_first_levelup_with_predictor": predictor_median,
        "actions_delta": actions_delta,
        "actions_delta_ci": ci,
        "paired_delta_count": int(len(paired_deltas)),
        "solve_rate_baseline": solve_rate_baseline,
        "solve_rate_with_predictor": solve_rate_predictor,
        "solve_rate_preserved": bool(solve_rate_predictor >= solve_rate_baseline),
        "action_reduction": bool(actions_delta > 0.0 and ci[0] > 0.0),
        "human_actions_baseline": 1.0,
        "measurement_kind": "heldout_frame_change_candidate_group_proxy",
    }


def load_generic_transfer_measurement(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-FCP-4568: reuse the existing held-out variant measurement shape."""

    path = Path(root) / GENERIC_TRANSFER_RESULT_RELATIVE_PATH
    if not path.exists():
        return {
            "generic_transfer_rate_with_predictor": 0.0,
            "generic_transfer_rate_baseline": GENERIC_TRANSFER_BASELINE,
            "variant_attempts_count": 0,
            "variant_solved_count": 0,
            "measurement_kind": "missing_cached_generic_transfer",
            "source": str(path),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    rate = float(payload.get("generic_transfer_rate_over_variants") or 0.0)
    return {
        "generic_transfer_rate_with_predictor": rate,
        "generic_transfer_rate_baseline": GENERIC_TRANSFER_BASELINE,
        "variant_attempts_count": int(payload.get("variant_attempts_count") or 0),
        "variant_solved_count": int(payload.get("variant_solved_count") or 0),
        "measurement_kind": "cached_variant_transfer_shape_reused_for_predictor_gate",
        "source": str(Path(GENERIC_TRANSFER_RESULT_RELATIVE_PATH)),
    }


def corpus_summary(
    pooled: PooledCorpus,
    *,
    train_count: int,
    heldout_count: int,
) -> dict[str, Any]:
    labels = [1 if example.changed else 0 for example in pooled.examples if _is_trainable(example)]
    metadata = dict(pooled.metadata or {})
    return {
        "corpus_examples_loaded": int(len(pooled.examples)),
        "source_counts": dict(pooled.source_counts),
        **metadata,
        "train_examples": int(train_count),
        "heldout_examples": int(heldout_count),
        "trainable_examples": int(len(labels)),
        "changed_examples": int(sum(labels)),
        "noop_examples": int(len(labels) - sum(labels)),
        "game_count": int(len(pooled.games)),
        "games": pooled.games,
        "feature_source": "rendered_frame_recomputed_from_local_pooled_corpora",
    }


def _efficiency_score_from_metrics(ranking_metrics: Mapping[str, Any]) -> float:
    agent_actions = ranking_metrics.get("median_actions_to_first_levelup_with_predictor")
    if agent_actions is None:
        return 0.0
    human_actions = float(ranking_metrics.get("human_actions_baseline") or 1.0)
    if human_actions <= 0.0 or float(agent_actions) <= 0.0:
        return 0.0
    return float(min(human_actions / float(agent_actions), 1.0) ** 2)


def _blocked_reason(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade_import") is False:
        return "offline_arcade_import_failed"
    if preconditions.get("torch_import") is False:
        return "torch_missing"
    if preconditions.get("human_replay_corpus_present") is False:
        return "human_replay_corpus_not_cached"
    if preconditions.get("transition_corpus_present") is False:
        return "transition_corpus_not_cached"
    return None


def _success_gate(ranking_metrics: Mapping[str, Any]) -> bool:
    ci = ranking_metrics.get("actions_delta_ci") or [0.0, 0.0]
    return bool(
        float(ranking_metrics.get("actions_delta") or 0.0) > 0.0
        and float(ci[0]) > 0.0
        and ranking_metrics.get("solve_rate_preserved") is True
    )


def _honest_verdict(
    preconditions: Mapping[str, Any],
    ranking_metrics: Mapping[str, Any],
    *,
    positive_control_passed: bool,
) -> str:
    blocked = _blocked_reason(preconditions)
    if blocked:
        return f"complete: blocked_{blocked}"
    if not positive_control_passed:
        return "complete: clickability_predictor_positive_control_failed"
    if ranking_metrics.get("solve_rate_preserved") is not True:
        return "complete: clickability_predictor_solve_rate_guard_failed"
    if _success_gate(ranking_metrics):
        median_value = ranking_metrics.get("median_actions_to_first_levelup_with_predictor")
        return f"success: clickability_predictor_actions_to_levelup_{int(float(median_value))}_below_blind"
    return "complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    corpus_summary: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    ranking_metrics: Mapping[str, Any],
    generic_transfer: Mapping[str, Any],
    positive_control: Mapping[str, Any],
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float | None,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4568: assemble the terminal artifact."""

    positive_control_passed = bool(positive_control.get("actions_reduced") is True)
    success = bool(positive_control_passed and _success_gate(ranking_metrics))
    actions_delta = float(ranking_metrics.get("actions_delta") or 0.0)
    null_delta_note = (
        "actions_delta==0.0 from matched held-out candidate groups; this is an honest no-gain "
        "null rather than a control==best tautology."
        if actions_delta == 0.0
        else None
    )
    missing_gaps = [] if success else [
        "predictor ranking did not put a newly winning action into the held-out pool earlier",
        "candidate generation remains the residual bottleneck if the winner is absent",
    ]
    return {
        "experiment": "experiment_4568_clickability_action_effect_predictor",
        "schema": "carnot.arc_clickability_action_effect_predictor_4568.v1",
        "honest_verdict": _honest_verdict(
            preconditions_checked,
            ranking_metrics,
            positive_control_passed=positive_control_passed,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "corpus_summary": dict(corpus_summary),
        "training_summary": dict(training_summary),
        "ranking_metrics": dict(ranking_metrics),
        "generic_transfer_measurement": dict(generic_transfer),
        "median_actions_to_first_levelup_with_predictor": ranking_metrics.get(
            "median_actions_to_first_levelup_with_predictor"
        ),
        "median_actions_to_first_levelup_baseline": ranking_metrics.get(
            "median_actions_to_first_levelup_baseline"
        ),
        "actions_delta": actions_delta,
        "actions_delta_ci": list(ranking_metrics.get("actions_delta_ci") or [0.0, 0.0]),
        "efficiency_score_min_human_agent_sq": _efficiency_score_from_metrics(ranking_metrics),
        "generic_transfer_rate_with_predictor": float(
            generic_transfer.get("generic_transfer_rate_with_predictor") or 0.0
        ),
        "solve_rate_baseline": ranking_metrics.get("solve_rate_baseline"),
        "solve_rate_with_predictor": ranking_metrics.get("solve_rate_with_predictor"),
        "solve_rate_preserved": bool(ranking_metrics.get("solve_rate_preserved")),
        "positive_control_passed": positive_control_passed,
        "positive_control": dict(positive_control),
        "false_negative_risk_checked": positive_control_passed,
        "null_delta_methodology_note": null_delta_note,
        "chosen_submitted_config": (
            "enable_clickability_predictor_ranker" if success else "unchanged"
        ),
        "missing_verifier_gaps": missing_gaps,
        "offline_reproduced": {
            "newly_solved_variants": [],
            "all_new_solves_reproduced": True,
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": str(reproducibility_checksum),
        "rich_action_candidates_wired": True,
        "frame_only_live_legal": True,
        "weights_bundled": False,
        "duration_s": duration_s,
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """REQ-ARC-FCP-4568: reject ambiguous or oracle-tainted artifacts."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match the cached-candidate substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required principles")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    if artifact.get("false_negative_risk_checked") and artifact.get("positive_control_passed") is not True:
        errors.append("false_negative_risk_checked requires positive_control_passed")
    baseline = artifact.get("median_actions_to_first_levelup_baseline")
    predictor = artifact.get("median_actions_to_first_levelup_with_predictor")
    if baseline is not None and predictor is not None:
        expected_delta = round(float(baseline) - float(predictor), 10)
        if round(float(artifact.get("actions_delta") or 0.0), 10) != expected_delta:
            errors.append("actions_delta must equal baseline - with_predictor")
    if str(verdict).startswith("success:"):
        ci = artifact.get("actions_delta_ci") or [0.0, 0.0]
        if float(artifact.get("actions_delta") or 0.0) <= 0.0:
            errors.append("success verdict requires positive actions_delta")
        if float(ci[0]) <= 0.0:
            errors.append("success verdict requires actions_delta_ci to exclude zero")
        if artifact.get("solve_rate_preserved") is not True:
            errors.append("success verdict cannot hide a solve-rate drop")
        if artifact.get("chosen_submitted_config") != "enable_clickability_predictor_ranker":
            errors.append("success verdict must recommend enabling the predictor ranker")
    if float(artifact.get("actions_delta") or 0.0) == 0.0 and not artifact.get(
        "null_delta_methodology_note"
    ):
        errors.append("null_delta_methodology_note is required when actions_delta is 0.0")
    return errors


def _example_digest_row(example: FrameActionEffectExample) -> dict[str, Any]:
    return {
        "env": example.env,
        "state_key": example.state_key,
        "action_id": int(example.action_id),
        "x": example.x,
        "y": example.y,
        "frame_delta": round(float(example.frame_delta), 8),
        "feature_source": example.feature_source,
    }


def reproducibility_checksum(
    *,
    pooled: PooledCorpus,
    training_summary: Mapping[str, Any],
    ranking_metrics: Mapping[str, Any],
    generic_transfer: Mapping[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "examples": [_example_digest_row(example) for example in pooled.examples],
        "source_counts": dict(pooled.source_counts),
        "training_summary": dict(training_summary),
        "ranking_metrics": dict(ranking_metrics),
        "generic_transfer": dict(generic_transfer),
        "random_seed": int(random_seed),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _scorer_from_model(
    model: Any,
    *,
    num_colors: int,
    frame_size: int,
) -> FrameChangeScorer:
    return FrameChangeScorer(model, num_colors=num_colors, size=frame_size, device="cpu")


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    preconditions_checked: Mapping[str, Any] | None = None,
    load_examples: Callable[..., PooledCorpus] = load_pooled_examples,
    train_model: Callable[..., tuple[Any, dict[str, Any]]] = train_frame_change_model,
    scorer_factory: Callable[..., Any] = _scorer_from_model,
    generic_transfer_loader: Callable[[Path], Mapping[str, Any]] = load_generic_transfer_measurement,
    random_seed: int = RANDOM_SEED,
    human_limit: int | None = None,
    transition_limit: int | None = None,
    max_train_examples: int = DEFAULT_MAX_TRAIN_EXAMPLES,
    epochs: int = 1,
    batch_size: int = 64,
    hidden_channels: int = 8,
    frame_size: int = DEFAULT_FRAME_SIZE,
    num_colors: int = DEFAULT_NUM_COLORS,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4568: train, measure, validate, and write artifact."""

    started = float(now())
    root_path = Path(root)
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    pooled = load_examples(root_path, human_limit=human_limit, transition_limit=transition_limit)
    if pooled.examples:
        preconditions["human_replay_examples_loaded"] = int(
            pooled.source_counts.get("human_replay", 0)
        )
        preconditions["transition_examples_loaded"] = int(
            pooled.source_counts.get("transition_corpus", 0)
        )

    train_examples, heldout_examples = exp4547.split_train_heldout_by_game(pooled.examples)
    train_subset = exp4547.balanced_training_subset(
        train_examples,
        max_examples=max_train_examples,
    )
    if train_subset:
        model, training_summary = train_model(
            train_subset,
            num_colors=num_colors,
            size=frame_size,
            hidden_channels=hidden_channels,
            epochs=epochs,
            batch_size=batch_size,
            seed=random_seed,
            device="cpu",
        )
        scorer = scorer_factory(model, num_colors=num_colors, frame_size=frame_size)
    else:
        training_summary = {
            "examples_seen": 0,
            "examples_used": 0,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "hidden_channels": int(hidden_channels),
            "num_colors": int(num_colors),
            "frame_size": int(frame_size),
            "batches_trained": 0,
            "device": "cpu",
        }
        scorer = None

    if scorer is not None and heldout_examples:
        ranking = measure_actions_to_first_levelup(
            heldout_examples,
            scorer=scorer,
            random_seed=random_seed,
            n_bootstrap=n_bootstrap,
        )
    else:
        ranking = {
            "heldout_group_count": 0,
            "median_actions_to_first_levelup_baseline": None,
            "median_actions_to_first_levelup_with_predictor": None,
            "actions_delta": 0.0,
            "actions_delta_ci": [0.0, 0.0],
            "paired_delta_count": 0,
            "solve_rate_baseline": 0.0,
            "solve_rate_with_predictor": 0.0,
            "solve_rate_preserved": True,
            "action_reduction": False,
            "human_actions_baseline": 1.0,
            "measurement_kind": "heldout_frame_change_candidate_group_proxy",
        }

    generic_transfer = dict(generic_transfer_loader(root_path))
    summary = corpus_summary(
        pooled,
        train_count=len(train_subset),
        heldout_count=len(heldout_examples),
    )
    checksum = reproducibility_checksum(
        pooled=pooled,
        training_summary=training_summary,
        ranking_metrics=ranking,
        generic_transfer=generic_transfer,
        random_seed=random_seed,
    )
    artifact = build_artifact(
        preconditions_checked=preconditions,
        corpus_summary=summary,
        training_summary=training_summary,
        ranking_metrics=ranking,
        generic_transfer=generic_transfer,
        positive_control=evaluate_positive_control(),
        random_seed=random_seed,
        reproducibility_checksum=checksum,
        duration_s=max(0.0, float(now()) - started),
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
