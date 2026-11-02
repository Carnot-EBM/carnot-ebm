"""Exp 4851: measure L1 first-contact generation coverage.

The diagnostic runs the live generic StepwiseExplorer proposal path cold, records
the candidate pools it actually enumerates, and only then compares those pools to
banked executable L1 prefixes. The banked prefix is classification ground truth,
not proposer input.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - script execution path
    sys.path.insert(0, str(REPO_ROOT / "python"))

EXPERIMENT_ID = 4851
RESULT_RELATIVE_PATH = "results/experiment_4851_generation_coverage_diagnostic.json"
CHECKPOINT_RELATIVE_DIR = "results/experiment_4851_generation_coverage_diagnostic_checkpoints"
SPEC_REFS = [
    "REQ-ARC-WMTE-4851",
    "SCENARIO-ARC-WMTE-4851-COVERAGE-BUCKETS",
    "SCENARIO-ARC-WMTE-4851-POSITIVE-CONTROL",
    "SCENARIO-ARC-WMTE-4851-BLOCKED-PRECONDITION",
]
BUCKETS = ("COVERED", "ENUMERATED_BUT_LOST", "NEVER_ENUMERATED")
DEFAULT_ACTION_BUDGET = 160
DEFAULT_MAX_DEPTH = 45
DEFAULT_POSITIVE_CONTROL_GAME = "tu93"
RANDOM_SEED = 20260627
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a measured decomposition is "
            "complete_generation_wall_<dominant_bucket>_dominant "
            "(e.g. complete_generation_wall_never_enumerated_dominant)."
        )
    },
    "per_game_coverage": {
        "principle": (
            "per-game mapping game -> {bucket in "
            "COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, winning_prefix_len, "
            "pool_size, reached_l1_win, budget_actions} -- the quantitative measurement."
        )
    },
    "dominant_bucket": {
        "principle": (
            "the bucket the majority of held-out games fall in -- the headline that "
            "redirects .448 (never_enumerated -> generation expressibility; "
            "enumerated_but_lost -> budget/pruning; covered -> ranking)."
        )
    },
    "positive_control_game": {
        "principle": (
            "the adaptered game used as positive control; it MUST be COVERED or the "
            "measurement is a harness artifact (a Phase-Prototype positive control)."
        )
    },
    "positive_control_covered": {
        "principle": (
            "true iff the positive control came out COVERED -- the load-bearing "
            "not-a-harness-artifact check."
        )
    },
    "proposer_blind_to_banked_answer": {
        "principle": (
            "true -- the banked winning prefix was NOT injected into the proposer "
            "(the tautology trap B1 audits)."
        )
    },
    "n_games_measured": {"principle": ">=3 held-out games for a non-degenerate distribution."},
    "verifier_is_oracle": {
        "principle": (
            "true -- the reproduction gate defining the winner is the executable "
            "oracle (circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the instrumentation hooks the live proposer (arc_orphan_solver_lint "
            "passes) -- a diagnostic the live agent cannot reach is wasted effort."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- an offline coverage measurement, NOT a live "
            "first-win; declared honestly."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference if the proposer invokes the LLM (60s floor), else "
            "verifier_ensemble_against_cached_candidates -- declare what actually ran."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/generator/ground-truth checks; a missing resource emits "
            "blocked_, never a fabricated bucket."
        )
    },
    "random_seed": {"principle": "determinism for the proposer's stochastic search."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, proposer config, budget) so a replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]


class DiagnosticError(RuntimeError):
    """Raised when the artifact builder would otherwise write invalid results."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalise_data(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, Mapping):
        out: dict[str, Any] = {}
        for key, value in sorted(data.items()):
            if value is None:
                out[str(key)] = None
            elif isinstance(value, bool):
                out[str(key)] = bool(value)
            elif isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
                out[str(key)] = int(value)
            else:
                out[str(key)] = value
        return out
    return data


def normalize_action(action: Any) -> JsonDict | None:
    """Return the canonical action shape used by StepwiseExplorer and banked traces."""

    if action is None:
        return None
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except json.JSONDecodeError:
            return None
    if not isinstance(action, Mapping):
        return None
    raw_action = action.get("action")
    if raw_action is None:
        raw_action = action.get("kind")
    if raw_action is None:
        return None
    try:
        action_id = int(raw_action)
    except (TypeError, ValueError):
        return None
    data = action.get("data")
    if data is None and {"x", "y"} <= set(action):
        data = {"x": action.get("x"), "y": action.get("y")}
    return {"action": action_id, "data": _normalise_data(data)}


def normalize_sequence(actions: Sequence[Any] | None) -> list[JsonDict]:
    out: list[JsonDict] = []
    for action in actions or []:
        normalised = normalize_action(action)
        if normalised is not None:
            out.append(normalised)
    return out


def action_key(action: Any) -> str:
    normalised = normalize_action(action)
    return _json_dumps(normalised) if normalised is not None else "<invalid>"


def prefix_key(prefix: Sequence[Any] | None) -> str:
    return _json_dumps(normalize_sequence(prefix))


def _unique_pool_size(records: Sequence[Mapping[str, Any]]) -> int:
    keys: set[tuple[str, str]] = set()
    for record in records:
        pkey = prefix_key(record.get("prefix") or [])
        for candidate in record.get("candidates") or []:
            ckey = action_key(candidate)
            if ckey != "<invalid>":
                keys.add((pkey, ckey))
    return len(keys)


def classify_game_coverage(
    game: str,
    winning_prefix: Sequence[Any],
    proposal_records: Sequence[Mapping[str, Any]],
    *,
    reached_l1_win: bool,
    budget_actions: int,
) -> JsonDict:
    """Classify one game's recorded proposal pools against its banked L1 prefix."""

    winner = normalize_sequence(winning_prefix)
    by_prefix: dict[str, set[str]] = {}
    for record in proposal_records:
        pkey = prefix_key(record.get("prefix") or [])
        by_prefix.setdefault(pkey, set())
        for candidate in record.get("candidates") or []:
            ckey = action_key(candidate)
            if ckey != "<invalid>":
                by_prefix[pkey].add(ckey)

    matched = 0
    for index, step in enumerate(winner):
        candidates = by_prefix.get(prefix_key(winner[:index]), set())
        if action_key(step) not in candidates:
            break
        matched += 1

    if reached_l1_win:
        bucket = "COVERED"
    elif winner and matched == len(winner):
        bucket = "ENUMERATED_BUT_LOST"
    else:
        bucket = "NEVER_ENUMERATED"

    return {
        "game": str(game),
        "bucket": bucket,
        "winning_prefix_len": len(winner),
        "pool_size": _unique_pool_size(proposal_records),
        "reached_l1_win": bool(reached_l1_win),
        "budget_actions": int(budget_actions),
        "matched_winning_prefix_len": int(matched),
    }


def compute_dominant_bucket(per_game_coverage: Mapping[str, Any]) -> str | None:
    buckets = [
        str(row.get("bucket")) for row in per_game_coverage.values() if isinstance(row, Mapping)
    ]
    counts = Counter(bucket for bucket in buckets if bucket in BUCKETS)
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], BUCKETS.index(item[0])))[0][0]


def _terminal_verdict_for(dominant: str | None, positive_control_covered: bool) -> str:
    if not positive_control_covered:
        return "complete_generation_coverage_diagnostic_retired_positive_control_failed"
    if dominant is None:
        return "complete_generation_coverage_diagnostic_retired_no_distribution"
    return f"complete_generation_wall_{dominant.lower()}_dominant"


def _proposer_config(action_budget: int, max_depth: int, max_games: int | None) -> JsonDict:
    return {
        "proposer": "StepwiseExplorer._candidates",
        "search_mode": "depth_first_ride",
        "action_budget": int(action_budget),
        "max_depth": int(max_depth),
        "max_games": max_games,
        "llm_invoked": False,
        "game_adapter_for_heldout": False,
        "positive_control_adaptered": True,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "games": sorted((artifact.get("per_game_coverage") or {}).keys()),
        "positive_control_game": artifact.get("positive_control_game"),
        "proposer_config": artifact.get("proposer_config") or {},
        "random_seed": artifact.get("random_seed"),
        "spec_refs": artifact.get("spec_refs") or SPEC_REFS,
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool = False,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_games: int | None = None,
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": str(verdict),
        "per_game_coverage": {},
        "dominant_bucket": None,
        "positive_control_game": DEFAULT_POSITIVE_CONTROL_GAME,
        "positive_control_covered": False,
        "positive_control_coverage": None,
        "proposer_blind_to_banked_answer": True,
        "n_games_measured": 0,
        "verifier_is_oracle": True,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "proposer_config": _proposer_config(action_budget, max_depth, max_games),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_artifact(
    *,
    per_game_coverage: Mapping[str, Mapping[str, Any]],
    positive_control_game: str,
    positive_control_coverage: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    action_budget: int,
    max_depth: int,
    max_games: int | None,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    dominant = compute_dominant_bucket(per_game_coverage)
    positive_control_covered = positive_control_coverage.get("bucket") == "COVERED"
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _terminal_verdict_for(dominant, positive_control_covered),
        "per_game_coverage": {str(k): dict(v) for k, v in per_game_coverage.items()},
        "dominant_bucket": dominant,
        "positive_control_game": str(positive_control_game),
        "positive_control_covered": bool(positive_control_covered),
        "positive_control_coverage": dict(positive_control_coverage),
        "proposer_blind_to_banked_answer": True,
        "n_games_measured": len(per_game_coverage),
        "verifier_is_oracle": True,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "proposer_config": _proposer_config(action_budget, max_depth, max_games),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(FIELD_PRINCIPLES) | {
        "schema_version",
        "experiment_id",
        "spec_refs",
        "positive_control_coverage",
        "proposer_config",
        "retire_if_same_verdict",
        "duration_s",
        "field_principles",
    }
    for field in sorted(required):
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    terminal = verdict.startswith(("complete_", "blocked_", "success_"))
    if not terminal:
        errors.append("honest_verdict_terminal_prefix")
    blocked = verdict.startswith("blocked_")
    retired = "retired" in verdict

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")

    per_game = artifact.get("per_game_coverage")
    if not isinstance(per_game, Mapping):
        errors.append("per_game_coverage")
        per_game = {}
    for game, row in per_game.items():
        if not isinstance(row, Mapping):
            errors.append(f"per_game_coverage.{game}")
            continue
        if row.get("bucket") not in BUCKETS:
            errors.append(f"per_game_coverage.{game}.bucket")
        for key in ("winning_prefix_len", "pool_size", "budget_actions"):
            try:
                if int(row.get(key)) < (
                    1 if key in {"winning_prefix_len", "budget_actions"} else 0
                ):
                    errors.append(f"per_game_coverage.{game}.{key}")
            except (TypeError, ValueError):
                errors.append(f"per_game_coverage.{game}.{key}")
        if not isinstance(row.get("reached_l1_win"), bool):
            errors.append(f"per_game_coverage.{game}.reached_l1_win")

    if not blocked:
        dominant = compute_dominant_bucket(per_game)
        if artifact.get("dominant_bucket") != dominant:
            errors.append("dominant_bucket")
        try:
            n_games = int(artifact.get("n_games_measured"))
        except (TypeError, ValueError):
            n_games = -1
            errors.append("n_games_measured")
        if n_games != len(per_game):
            errors.append("n_games_measured")
        if n_games < 3:
            errors.append("n_games_measured_minimum")
        if artifact.get("positive_control_covered") is not True and not retired:
            errors.append("positive_control_covered")
        if artifact.get("live_path_reachable") is not True and not retired:
            errors.append("live_path_reachable")
    elif artifact.get("per_game_coverage") != {}:
        errors.append("blocked_artifact_has_coverage")

    if artifact.get("proposer_blind_to_banked_answer") is not True:
        errors.append("proposer_blind_to_banked_answer")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("retire_if_same_verdict") is not True:
        errors.append("retire_if_same_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_checkpoint(game: str, row: Mapping[str, Any], *, root: Path | str) -> Path:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _registry_games(
    root: Path | str = REPO_ROOT,
) -> set[str]:  # pragma: no cover - file precondition
    path = Path(root) / "ops" / "arc_solve_registry.yaml"
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"(?m)^- game:\s*([A-Za-z0-9_]+)\s*$", text))


def offline_arcade_available() -> bool:  # pragma: no cover - live SDK boundary
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
    except Exception:
        return False
    return True


def _step_env(env: Any, action: Mapping[str, Any], *, policy: str) -> Any:  # pragma: no cover
    from arcengine import GameAction

    action_id = int(action["action"])
    return env.step(
        getattr(GameAction, f"ACTION{action_id}"),
        data=action.get("data"),
        reasoning={"policy": policy},
    )


def extract_l1_prefix(
    game: str,
    actions: Sequence[Mapping[str, Any]],
    *,
    arcade: Any | None = None,
) -> list[JsonDict] | None:  # pragma: no cover - live SDK boundary
    from carnot.agentic import arc_solver_kit as kit

    arc = arcade or kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    start_level = kit.frame_level(frame)
    normalised = normalize_sequence(actions)
    if not normalised:
        return None

    prefix: list[JsonDict] = []
    if game == "sc25":
        warmup = normalised[0]
        frame = _step_env(env, warmup, policy="exp4851_warmup_for_banked_prefix")
        prefix.append(warmup)
        if frame is not None and kit.frame_level(frame) > start_level:
            return prefix

    for step in normalised:
        frame = _step_env(env, step, policy="exp4851_banked_prefix_oracle")
        prefix.append(step)
        if frame is None:
            return None
        if kit.frame_level(frame) > start_level:
            return list(prefix)
    return None


def load_banked_l1_prefixes(
    root: Path | str = REPO_ROOT,
) -> dict[str, list[JsonDict]]:  # pragma: no cover - live artifact/offline boundary
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import load_solutions

    registry = _registry_games(root)
    arcade = kit.offline_arcade()
    prefixes: dict[str, list[JsonDict]] = {}
    for game, actions in load_solutions().items():
        if registry and game not in registry:
            continue
        prefix = extract_l1_prefix(game, actions, arcade=arcade)
        if prefix:
            prefixes[game] = prefix
    return prefixes


def measure_game_with_stepwise_explorer(
    *,
    game: str,
    winning_prefix: Sequence[Mapping[str, Any]],
    action_budget: int,
    max_depth: int = DEFAULT_MAX_DEPTH,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:  # pragma: no cover - live simulator boundary
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import StepwiseExplorer

    class RecordingStepwiseExplorer(StepwiseExplorer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.proposal_records: list[JsonDict] = []
            super().__init__(*args, **kwargs)

        def _candidates(
            self, frame: Any, path: Sequence[dict] | None = None, previous_frame: Any | None = None
        ) -> list[dict]:  # type: ignore[override]
            rows = super()._candidates(frame, path=path, previous_frame=previous_frame)
            self.proposal_records.append(
                {"prefix": normalize_sequence(path or []), "candidates": normalize_sequence(rows)}
            )
            return rows

    _ = root, random_seed
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    explorer = RecordingStepwiseExplorer(
        target_levels=1,
        max_depth=max_depth,
        value_head=None,
        value_weight=0.0,
        search_mode="depth_first_ride",
        online_discriminative=False,
        frame_change_scorer=None,
        action_effect_expansion_prior=None,
        action_prior=None,
        candidate_router=None,
        dense_curiosity=False,
        goal_bias=None,
        goal_candidate_guidance=None,
        qd_generator=None,
        controllable_novelty=False,
        object_centric_proposal=False,
        program_synthesis_filter=None,
        amortized_first_contact_prior=False,
        go_explore_archive=False,
    )

    frames: list[Any] = []
    latest: Any = None
    start_level: int | None = None
    reached_l1_win = False
    actions_executed = 0
    turns = 0
    max_turns = int(action_budget) * 4 + 10
    while actions_executed < int(action_budget) and turns < max_turns:
        turns += 1
        if explorer.is_done(frames, latest):
            break
        action_id, data = explorer.next_move(frames, latest)
        if action_id == "RESET":
            latest = env.reset()
            frames.append(latest)
            if start_level is None:
                start_level = kit.frame_level(latest)
            continue
        if action_id is None:
            break
        latest = env.step(
            getattr(GameAction, f"ACTION{int(action_id)}"),
            data=data,
            reasoning={"policy": "exp4851_stepwise_generation_coverage"},
        )
        actions_executed += 1
        if latest is None:
            break
        frames.append(latest)
        if start_level is None:
            start_level = kit.frame_level(latest)
        if kit.frame_level(latest) > int(start_level):
            reached_l1_win = True
            break

    row = classify_game_coverage(
        game,
        winning_prefix,
        explorer.proposal_records,
        reached_l1_win=reached_l1_win,
        budget_actions=action_budget,
    )
    row["actions_executed"] = actions_executed
    row["proposal_record_count"] = len(explorer.proposal_records)
    return row


def measure_adapter_positive_control(
    *,
    game: str = DEFAULT_POSITIVE_CONTROL_GAME,
    winning_prefix: Sequence[Mapping[str, Any]] | None = None,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:  # pragma: no cover - live adapter boundary
    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic import arc_solver_kit as kit

    _ = root, random_seed
    adapter = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        game,
        adapter.action_labels,
        adapter.apply,
        adapter.state_key,
        warmup_label=getattr(adapter, "warmup_label", None),
        verifier=getattr(adapter, "hand_verifier", None),
        branch_mode=getattr(adapter, "branch_mode", "replay"),
        max_nodes=max(2000, int(action_budget) * 100),
    )
    start_frame = solver._replay(env, [])
    start_level = kit.frame_level(start_frame)
    depth_cap = min(max_depth, max(1, len(winning_prefix or []), 60))
    path, states = solver.solve_level(env, start_level, [], depth_cap)
    reached = path is not None
    prefix = normalize_sequence(winning_prefix or path or [])
    records = []
    if path:
        for index, label in enumerate(path):
            records.append({"prefix": prefix[:index], "candidates": [normalize_action(label)]})
    row = classify_game_coverage(
        game,
        prefix,
        records,
        reached_l1_win=reached,
        budget_actions=action_budget,
    )
    row["states_expanded"] = int(states)
    row["adaptered"] = True
    if reached:
        row["bucket"] = "COVERED"
        row["reached_l1_win"] = True
    return row


def run_orphan_lint(root: Path | str = REPO_ROOT) -> bool:  # pragma: no cover - subprocess boundary
    result = subprocess.run(
        [sys.executable, "scripts/arc_orphan_solver_lint.py"],
        cwd=Path(root),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    return result.returncode == 0


def _validate_or_raise(artifact: JsonDict) -> JsonDict:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise DiagnosticError(";".join(errors))
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    offline_arcade_checker: Callable[[], bool] = offline_arcade_available,
    ground_truth_loader: Callable[[Path], Mapping[str, Sequence[Mapping[str, Any]]]] = (
        load_banked_l1_prefixes
    ),
    coverage_measurer: Callable[..., Mapping[str, Any]] = measure_game_with_stepwise_explorer,
    positive_control_runner: Callable[..., Mapping[str, Any]] = measure_adapter_positive_control,
    live_path_checker: Callable[[Path], bool] = run_orphan_lint,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_games: int | None = None,
    positive_control_game: str = DEFAULT_POSITIVE_CONTROL_GAME,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions: JsonDict = {
        "offline_arcade": {"ok": False},
        "generator": {
            "required": False,
            "available": True,
            "detail": "not_required_stepwise_explorer_no_llm",
        },
        "ground_truth": {"ok": False, "available_games": []},
        "live_path": {"ok": False},
        "proposer_invokes_llm": False,
    }

    if not offline_arcade_checker():
        preconditions["offline_arcade"] = {"ok": False, "detail": "offline_arcade_import_failed"}
        artifact = build_blocked_artifact(
            "blocked_offline_arcade_missing",
            preconditions_checked=preconditions,
            action_budget=action_budget,
            max_depth=max_depth,
            max_games=max_games,
            duration_s=now() - start,
            random_seed=random_seed,
        )
        if write:
            write_artifact(artifact, root=root_path)
        return _validate_or_raise(artifact)

    preconditions["offline_arcade"] = {"ok": True}
    ground_truth = {
        str(game): normalize_sequence(prefix)
        for game, prefix in ground_truth_loader(root_path).items()
        if normalize_sequence(prefix)
    }
    heldout_games = [game for game in sorted(ground_truth) if game != positive_control_game]
    if max_games is not None:
        heldout_games = heldout_games[: int(max_games)]
    preconditions["ground_truth"] = {
        "ok": len(heldout_games) >= 3,
        "available_games": list(heldout_games),
        "n_available": len(heldout_games),
        "positive_control_game_present": positive_control_game in ground_truth,
    }
    if len(heldout_games) < 3:
        artifact = build_blocked_artifact(
            "blocked_no_banked_ground_truth",
            preconditions_checked=preconditions,
            action_budget=action_budget,
            max_depth=max_depth,
            max_games=max_games,
            duration_s=now() - start,
            random_seed=random_seed,
        )
        if write:
            write_artifact(artifact, root=root_path)
        return _validate_or_raise(artifact)

    live_path_reachable = bool(live_path_checker(root_path))
    preconditions["live_path"] = {"ok": live_path_reachable}
    if not live_path_reachable:
        artifact = build_blocked_artifact(
            "blocked_live_path_unreachable",
            preconditions_checked=preconditions,
            live_path_reachable=False,
            action_budget=action_budget,
            max_depth=max_depth,
            max_games=max_games,
            duration_s=now() - start,
            random_seed=random_seed,
        )
        if write:
            write_artifact(artifact, root=root_path)
        return _validate_or_raise(artifact)

    per_game: dict[str, JsonDict] = {}
    for game in heldout_games:
        row = dict(
            coverage_measurer(
                game=game,
                winning_prefix=ground_truth[game],
                action_budget=action_budget,
                max_depth=max_depth,
                root=root_path,
                random_seed=random_seed,
            )
        )
        per_game[game] = row
        if write_checkpoints:
            _write_checkpoint(game, row, root=root_path)

    positive_prefix = ground_truth.get(positive_control_game)
    positive_control = dict(
        positive_control_runner(
            game=positive_control_game,
            winning_prefix=positive_prefix,
            action_budget=action_budget,
            max_depth=max_depth,
            root=root_path,
            random_seed=random_seed,
        )
    )
    preconditions["positive_control"] = {
        "game": positive_control_game,
        "covered": positive_control.get("bucket") == "COVERED",
    }
    artifact = build_artifact(
        per_game_coverage=per_game,
        positive_control_game=positive_control_game,
        positive_control_coverage=positive_control,
        preconditions_checked=preconditions,
        live_path_reachable=live_path_reachable,
        action_budget=action_budget,
        max_depth=max_depth,
        max_games=max_games,
        duration_s=now() - start,
        random_seed=random_seed,
    )
    if write:
        write_artifact(artifact, root=root_path)
    return _validate_or_raise(artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    _ = argv
    artifact = run(root=REPO_ROOT, write=True, write_checkpoints=True)
    print(
        json.dumps({"artifact": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
