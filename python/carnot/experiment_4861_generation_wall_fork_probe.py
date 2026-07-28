"""Exp 4861: fork the Exp 4851 generation wall with live induce->plan.

The probe keeps the planner blind to banked winning prefixes. It uses cold
agent-collected transitions to invoke the live E3 induction method, reloads the
induced executable world model through the live loader, plans in that model, and
only then compares the planned pool with banked L1 prefixes for classification.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - script execution path
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4851_generation_coverage_diagnostic import (  # noqa: E402
    action_key,
    load_banked_l1_prefixes,
    normalize_sequence,
    offline_arcade_available,
    run_orphan_lint,
)


EXPERIMENT_ID = 4861
RESULT_RELATIVE_PATH = "results/experiment_4861_generation_wall_fork_probe.json"
CHECKPOINT_RELATIVE_DIR = "results/experiment_4861_generation_wall_fork_probe_checkpoints"
SPEC_REFS = [
    "REQ-ARC-WMTE-4861",
    "SCENARIO-ARC-WMTE-4861-BLOCKED-PRECONDITION",
    "SCENARIO-ARC-WMTE-4861-JOINT-FORK",
    "SCENARIO-ARC-WMTE-4861-PARTIAL-CHECKPOINT",
]
HELDOUT_GAMES = ("cd82", "cn04", "ls20", "m0r0", "r11l", "sk48", "sp80", "su15", "wa30")
DEFAULT_POSITIVE_CONTROL_GAME = "tu93"
BUCKETS = ("COVERED", "ENUMERATED_BUT_LOST", "NEVER_ENUMERATED")
FORK_VERDICTS = ("GUIDANCE_WALL", "PLANNER_GAP", "INDUCER_CEILING")
HIGH_ACCURACY_THRESHOLD = 0.5
DEFAULT_TRANSITION_BUDGET = 32
DEFAULT_HELDOUT_TRANSITIONS = 24
DEFAULT_PLAN_MAX_NODES = 20000
DEFAULT_PLAN_MAX_DEPTH = 40
DEFAULT_SOFT_ELAPSED_BUDGET_S = 4200.0
RANDOM_SEED = 20260627
INFERENCE_SUBSTRATE = "live_llm_inference"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a measured fork is complete_generation_wall_<fork>_<detail> "
            "(e.g. complete_generation_wall_inducer_ceiling_low_accuracy_no_migration)."
        )
    },
    "fork_verdict": {
        "principle": (
            "one of GUIDANCE_WALL | PLANNER_GAP | INDUCER_CEILING -- the headline "
            "that redirects .449."
        )
    },
    "per_game_fork": {
        "principle": (
            "per-game mapping game -> {engine_heldout_accuracy, planned_bucket in "
            "COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, migrated (bare NEVER->planned "
            "COVERED), winning_prefix_len} -- the quantitative joint measurement."
        )
    },
    "coverage_migration_count": {
        "principle": (
            "how many NEVER_ENUMERATED games migrated to COVERED under induce->plan -- "
            "the GUIDANCE signal."
        )
    },
    "median_engine_heldout_accuracy": {
        "principle": (
            "the induced world-model quality across games -- distinguishes INDUCER "
            "ceiling (low) from planner gap (high)."
        )
    },
    "positive_control_game": {
        "principle": (
            "tu93 -- MUST be HIGH accuracy + COVERED or the measurement is a harness artifact."
        )
    },
    "positive_control_migrated": {
        "principle": (
            "true iff tu93 came out HIGH accuracy + COVERED -- the load-bearing "
            "not-a-harness-artifact check."
        )
    },
    "planner_blind_to_banked_answer": {
        "principle": (
            "true -- the banked winning prefix was NOT injected into induction or planning "
            "(the tautology trap B1 audits)."
        )
    },
    "n_games_measured": {
        "principle": ">=3 NEVER_ENUMERATED held-out games for a non-degenerate joint table."
    },
    "verifier_is_oracle": {
        "principle": (
            "true -- the reproduction gate defining the winner is the executable oracle "
            "(circularity discipline)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the probe calls the live e3.load_engine/plan_in_model path "
            "(arc_orphan_solver_lint passes) -- a diagnostic the live agent cannot reach "
            "is wasted effort."
        )
    },
    "solve_provenance": {
        "principle": (
            "development_proxy -- an offline fork measurement, NOT a live first-win; "
            "declared honestly."
        )
    },
    "checkpoint_emitted": {
        "principle": (
            "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); "
            "per-game checkpointing."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (60s floor) -- induce->plan invokes the LLM."
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/generator/held-out-games checks; a missing resource emits "
            "blocked_, never a fabricated fork."
        )
    },
    "random_seed": {"principle": "determinism for induction + planning stochastic search."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, induce/plan config, budget) so a replication catches drift."
        )
    },
}


JsonDict = dict[str, Any]
Clock = Callable[[], float]


class DiagnosticError(RuntimeError):
    """Raised when the artifact builder would otherwise write invalid results."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalise_generator_result(result: Any) -> JsonDict:
    if isinstance(result, Mapping):
        out = dict(result)
        out["ok"] = bool(out.get("ok"))
        return out
    return {"ok": bool(result)}


def _induce_plan_config(
    *,
    transition_budget: int,
    heldout_transitions: int,
    plan_max_nodes: int,
    plan_max_depth: int,
    soft_elapsed_budget_s: float,
    heldout_games: Sequence[str],
) -> JsonDict:
    return {
        "live_path": "E3AgentPolicy._induce_and_plan -> e3.load_engine -> e3.plan_in_model",
        "llm_model": "Qwen3.5-9B-MTP",
        "igpu_only": True,
        "high_accuracy_threshold": HIGH_ACCURACY_THRESHOLD,
        "transition_budget": int(transition_budget),
        "heldout_transitions": int(heldout_transitions),
        "plan_max_nodes": int(plan_max_nodes),
        "plan_max_depth": int(plan_max_depth),
        "soft_elapsed_budget_s": float(soft_elapsed_budget_s),
        "heldout_games": list(heldout_games),
        "game_adapter_for_heldout": False,
        "planner_blind_to_banked_answer": True,
    }


def classify_planned_pool(
    game: str,
    winning_prefix: Sequence[Any],
    planned_pool: Sequence[Any] | None,
    *,
    planner_reached_l1_win: bool,
) -> JsonDict:
    """Classify the post-planning pool against the banked prefix after planning."""

    winner = normalize_sequence(winning_prefix)
    planned = normalize_sequence(planned_pool or [])
    matched = 0
    for expected, actual in zip(winner, planned):
        if action_key(expected) != action_key(actual):
            break
        matched += 1

    if planner_reached_l1_win:
        bucket = "COVERED"
    elif winner and matched == len(winner):
        bucket = "ENUMERATED_BUT_LOST"
    else:
        bucket = "NEVER_ENUMERATED"

    return {
        "game": str(game),
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED",
        "winning_prefix_len": len(winner),
        "planned_prefix_len": int(matched),
        "planned_pool_size": 1 if planned else 0,
        "plan_length": len(planned),
        "planner_reached_l1_win": bool(planner_reached_l1_win),
    }


def _median_accuracy(per_game_fork: Mapping[str, Mapping[str, Any]]) -> float | None:
    values: list[float] = []
    for row in per_game_fork.values():
        if not isinstance(row, Mapping):
            continue
        try:
            accuracy = float(row.get("engine_heldout_accuracy"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= accuracy <= 1.0:
            values.append(accuracy)
    return float(median(values)) if values else None


def _positive_control_passed(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    try:
        accuracy = float(row.get("engine_heldout_accuracy"))
    except (TypeError, ValueError):
        accuracy = 0.0
    return row.get("planned_bucket") == "COVERED" and accuracy >= HIGH_ACCURACY_THRESHOLD


def _coverage_migration_count(per_game_fork: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in per_game_fork.values()
        if isinstance(row, Mapping) and bool(row.get("migrated"))
    )


def compute_fork_verdict(
    per_game_fork: Mapping[str, Mapping[str, Any]],
    *,
    positive_control_row: Mapping[str, Any] | None,
) -> str | None:
    if len(per_game_fork) < 3 or not _positive_control_passed(positive_control_row):
        return None
    median_accuracy = _median_accuracy(per_game_fork)
    high_accuracy = (
        median_accuracy is not None and float(median_accuracy) >= HIGH_ACCURACY_THRESHOLD
    )
    migrations = _coverage_migration_count(per_game_fork)
    if high_accuracy and migrations >= 1:
        return "GUIDANCE_WALL"
    if high_accuracy:
        return "PLANNER_GAP"
    return "INDUCER_CEILING"


def _terminal_verdict(
    *,
    fork_verdict: str | None,
    positive_control_row: Mapping[str, Any] | None,
    n_games: int,
    partial: bool,
) -> str:
    if partial:
        return "complete_generation_wall_fork_probe_partial_budget_stop"
    if not _positive_control_passed(positive_control_row):
        return "complete_generation_wall_fork_probe_retired_positive_control_failed"
    if n_games < 3 or fork_verdict is None:
        return "complete_generation_wall_fork_probe_retired_no_joint_table"
    if fork_verdict == "GUIDANCE_WALL":
        return "complete_generation_wall_guidance_wall_high_accuracy_migration"
    if fork_verdict == "PLANNER_GAP":
        return "complete_generation_wall_planner_gap_high_accuracy_no_migration"
    return "complete_generation_wall_inducer_ceiling_low_accuracy_no_migration"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "games": sorted((artifact.get("per_game_fork") or {}).keys()),
        "positive_control_game": artifact.get("positive_control_game"),
        "induce_plan_config": artifact.get("induce_plan_config") or {},
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
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
    transition_budget: int = DEFAULT_TRANSITION_BUDGET,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
) -> JsonDict:
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": str(verdict),
        "fork_verdict": None,
        "per_game_fork": {},
        "coverage_migration_count": 0,
        "median_engine_heldout_accuracy": None,
        "positive_control_game": DEFAULT_POSITIVE_CONTROL_GAME,
        "positive_control_migrated": False,
        "positive_control_fork": None,
        "planner_blind_to_banked_answer": True,
        "n_games_measured": 0,
        "verifier_is_oracle": True,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": False,
        "partial": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "induce_plan_config": _induce_plan_config(
            transition_budget=transition_budget,
            heldout_transitions=heldout_transitions,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
        ),
        "retire_if_same_verdict": True,
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_artifact(
    *,
    per_game_fork: Mapping[str, Mapping[str, Any]],
    positive_control_game: str,
    positive_control_row: Mapping[str, Any] | None,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
    partial: bool,
    checkpoint_emitted: bool = True,
    random_seed: int = RANDOM_SEED,
    transition_budget: int = DEFAULT_TRANSITION_BUDGET,
    heldout_transitions: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
) -> JsonDict:
    rows = {str(k): dict(v) for k, v in per_game_fork.items()}
    control = dict(positive_control_row) if isinstance(positive_control_row, Mapping) else None
    fork = compute_fork_verdict(rows, positive_control_row=control)
    med = _median_accuracy(rows)
    artifact: JsonDict = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _terminal_verdict(
            fork_verdict=fork,
            positive_control_row=control,
            n_games=len(rows),
            partial=partial,
        ),
        "fork_verdict": fork,
        "per_game_fork": rows,
        "coverage_migration_count": _coverage_migration_count(rows),
        "median_engine_heldout_accuracy": med,
        "positive_control_game": str(positive_control_game),
        "positive_control_migrated": _positive_control_passed(control),
        "positive_control_fork": control,
        "planner_blind_to_banked_answer": True,
        "n_games_measured": len(rows),
        "verifier_is_oracle": True,
        "live_path_reachable": bool(live_path_reachable),
        "solve_provenance": "development_proxy",
        "checkpoint_emitted": bool(checkpoint_emitted),
        "partial": bool(partial),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": int(random_seed),
        "induce_plan_config": _induce_plan_config(
            transition_budget=transition_budget,
            heldout_transitions=heldout_transitions,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
        ),
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
        "positive_control_fork",
        "induce_plan_config",
        "retire_if_same_verdict",
        "duration_s",
        "partial",
        "field_principles",
    }
    for field in sorted(required):
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict_terminal_prefix")
    blocked = verdict.startswith("blocked_")
    partial = artifact.get("partial") is True
    retired = "retired" in verdict

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")

    per_game = artifact.get("per_game_fork")
    if not isinstance(per_game, Mapping):
        errors.append("per_game_fork")
        per_game = {}
    for game, row in per_game.items():
        if not isinstance(row, Mapping):
            errors.append(f"per_game_fork.{game}")
            continue
        if row.get("planned_bucket") not in BUCKETS:
            errors.append(f"per_game_fork.{game}.planned_bucket")
        try:
            accuracy = float(row.get("engine_heldout_accuracy"))
            if not 0.0 <= accuracy <= 1.0:
                errors.append(f"per_game_fork.{game}.engine_heldout_accuracy")
        except (TypeError, ValueError):
            errors.append(f"per_game_fork.{game}.engine_heldout_accuracy")
        for key in ("winning_prefix_len", "planned_pool_size", "heldout_transition_count"):
            try:
                if int(row.get(key)) < (1 if key == "winning_prefix_len" else 0):
                    errors.append(f"per_game_fork.{game}.{key}")
            except (TypeError, ValueError):
                errors.append(f"per_game_fork.{game}.{key}")
        if not isinstance(row.get("migrated"), bool):
            errors.append(f"per_game_fork.{game}.migrated")

    if blocked:
        if per_game:
            errors.append("blocked_artifact_has_fork_rows")
    else:
        fork = artifact.get("fork_verdict")
        if fork is not None and fork not in FORK_VERDICTS:
            errors.append("fork_verdict")
        try:
            n_games = int(artifact.get("n_games_measured"))
        except (TypeError, ValueError):
            n_games = -1
            errors.append("n_games_measured")
        if n_games != len(per_game):
            errors.append("n_games_measured")
        if not partial and not retired:
            if n_games < 3:
                errors.append("n_games_measured_minimum")
            if fork not in FORK_VERDICTS:
                errors.append("fork_verdict")
            if artifact.get("positive_control_migrated") is not True:
                errors.append("positive_control_migrated")
            if artifact.get("live_path_reachable") is not True:
                errors.append("live_path_reachable")
        if partial and n_games < 1:
            errors.append("partial_without_rows")

    if artifact.get("coverage_migration_count") != _coverage_migration_count(per_game):
        errors.append("coverage_migration_count")
    if artifact.get("median_engine_heldout_accuracy") != _median_accuracy(per_game):
        errors.append("median_engine_heldout_accuracy")
    if artifact.get("positive_control_migrated") != _positive_control_passed(
        artifact.get("positive_control_fork")
    ):
        errors.append("positive_control_migrated")
    if artifact.get("planner_blind_to_banked_answer") is not True:
        errors.append("planner_blind_to_banked_answer")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance")
    if not isinstance(artifact.get("checkpoint_emitted"), bool):
        errors.append("checkpoint_emitted")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("retire_if_same_verdict") is not True:
        errors.append("retire_if_same_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _validate_or_raise(artifact: JsonDict) -> JsonDict:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise DiagnosticError(";".join(errors))
    return artifact


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_checkpoint(game: str, row: Mapping[str, Any], *, root: Path | str) -> Path:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _load_checkpoint(game: str, *, root: Path | str) -> JsonDict | None:
    path = Path(root) / CHECKPOINT_RELATIVE_DIR / f"{game}.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return dict(row) if isinstance(row, Mapping) else None


def _environment_games(_arcade: Any = None) -> set[str]:  # pragma: no cover - ARC runtime
    from carnot.agentic import arc_solver_kit as kit

    arc = _arcade or kit.offline_arcade()
    return {str(getattr(env, "game_id", "")).split("-", 1)[0] for env in arc.get_environments()}


def _gid(arcade: Any, short: str) -> str:  # pragma: no cover - ARC runtime
    for env in arcade.get_environments():
        game_id = str(getattr(env, "game_id", ""))
        if game_id.split("-", 1)[0] == short:
            return game_id
    raise RuntimeError(f"{short} unavailable")


def make_live_qwen_proposer() -> Any:  # pragma: no cover - llama boundary
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
        # mtp is DELIBERATELY NOT PASSED. This line used to read
        # `mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0")` -- a literal "1" that is NOT the
        # project's canonical local default (`ARC_LIVE_GENERATOR_MTP_DEFAULT` is "0"). With
        # CARNOT_ARC_MTP unset that handed the proposer mtp=True, which at the shipped n_ctx 81920
        # needs ~14 offloaded FFN layers on a 24 GB card -- past the auto-fit cap, so the VRAM guard
        # declines CUDA, the generator falls back to the ~2 tok/s iGPU, every induce times out, and
        # the run proceeds LLM-OFF while still reporting itself LLM-on. Omitting the argument lets
        # `LocalGGUFProposer.mtp`'s own default factory (`_mtp_default_on()`) answer, which reads
        # the SAME env var against the canonical constant -- identical override behaviour, correct
        # default, and one place to change it.
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=int(os.environ.get("CARNOT_ARC_4861_MAX_TOKENS", "2560")),
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
        timeout=int(os.environ.get("CARNOT_ARC_4861_LLM_TIMEOUT", "300")),
        tries=int(os.environ.get("CARNOT_ARC_4861_LLM_TRIES", "1")),
    )


def generator_available(*, proposer: Any | None = None) -> JsonDict:  # pragma: no cover
    from carnot.agentic import arc_executable_world_model as e3

    server, launch_env = e3._generator_server_and_env()
    detail: JsonDict = {
        "server": str(server),
        "launch_env_cuda_visible_devices": (
            None if launch_env is None else launch_env.get("CUDA_VISIBLE_DEVICES")
        ),
        "model": "Qwen3.5-9B-MTP",
        "igpu_required": True,
    }
    if launch_env is not None and launch_env.get("CUDA_VISIBLE_DEVICES"):
        return {**detail, "ok": False, "detail": "cuda_3090_generator_disallowed"}
    if "build-hip" not in str(server) and not os.environ.get("CARNOT_LLAMA_SERVER"):
        return {**detail, "ok": False, "detail": "igpu_hip_llama_server_not_selected"}
    prop = proposer or make_live_qwen_proposer()
    ensure = getattr(prop, "_ensure_server", None)
    if not callable(ensure):
        return {**detail, "ok": False, "detail": "generator_missing_ensure_server"}
    ok = bool(ensure())
    return {**detail, "ok": ok, "detail": "ok" if ok else "qwen_llama_server_unhealthy"}


def _collect_cold_policy_transitions(
    *,
    game: str,
    proposer: Any,
    transition_budget: int,
    action_budget: int,
) -> JsonDict:  # pragma: no cover - ARC runtime
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

    arc = kit.offline_arcade()
    gid = _gid(arc, game)
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        gid,
        proposer=proposer,
        target_levels=1,
        explore_budget=max(int(action_budget) + 100, int(transition_budget) + 100),
        value_head=None,
        candidate_router=None,
        frame_change_scorer=None,
        goal_bias=None,
        goal_candidate_guidance=None,
    )
    frames: list[Any] = []
    latest: Any = None
    root_grid = None
    actions = 0
    turns = 0
    max_turns = int(action_budget) * 4 + 20
    while len(policy.transitions) < int(transition_budget) and turns < max_turns:
        turns += 1
        move, data = policy.next_move(frames, latest)
        if move == "RESET":
            latest = env.reset()
        elif move is None:
            break
        else:
            latest = env.step(
                getattr(GameAction, f"ACTION{int(move)}"),
                data=data,
                reasoning={"policy": "exp4861_cold_transition_collection"},
            )
            actions += 1
        if latest is None:
            break
        if root_grid is None:
            policy.cell = detect_cell(grid_of(latest))
            root_grid = to_logical(grid_of(latest), policy.cell)
            policy.root_grid = root_grid
        frames.append(latest)
        if actions >= int(action_budget):
            break
    if latest is not None and len(policy.transitions) < int(transition_budget):
        try:
            policy.next_move(frames, latest)
        except Exception:
            pass
    if policy.root_grid is None and root_grid is not None:
        policy.root_grid = root_grid
    return {
        "policy": policy,
        "transitions": list(policy.transitions),
        "cell": int(policy.cell),
        "root_grid": policy.root_grid,
        "actions": int(actions),
        "turns": int(turns),
    }


def _heldout_transitions(
    *,
    game: str,
    n: int,
    seed: int,
) -> list[Any]:  # pragma: no cover - ARC runtime
    from carnot.agentic import arc_executable_world_model as e3

    transitions, _cell = e3.collect_transitions(game, n=int(n), warmup=False, seed=int(seed))
    return list(transitions)


def _execute_plan_reaches_l1(
    game: str,
    plan: Sequence[Any],
) -> bool:  # pragma: no cover - ARC runtime
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _level_of

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    start_level = _level_of(frame)
    for step in normalize_sequence(plan):
        frame = env.step(
            getattr(GameAction, f"ACTION{int(step['action'])}"),
            data=step.get("data"),
            reasoning={"policy": "exp4861_planned_pool_oracle"},
        )
        if frame is None:
            return False
        if _level_of(frame) > start_level:
            return True
    return False


def measure_game_with_live_induce_plan(
    *,
    game: str,
    winning_prefix: Sequence[Mapping[str, Any]],
    proposer: Any,
    transition_budget: int = DEFAULT_TRANSITION_BUDGET,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:  # pragma: no cover - live ARC/LLM boundary
    from carnot.agentic import arc_executable_world_model as e3

    _ = root
    cold = _collect_cold_policy_transitions(
        game=game,
        proposer=proposer,
        transition_budget=transition_budget,
        action_budget=max(transition_budget * 2, 40),
    )
    policy = cold["policy"]
    policy.transitions = list(cold["transitions"])
    policy.root_grid = cold["root_grid"]
    policy.cell = int(cold["cell"])
    policy._pending_induction_reason = "exp4861_generation_wall_fork_probe"
    policy.induced = False
    live_methods = ["E3AgentPolicy._induce_and_plan"]
    if policy.transitions and policy.root_grid is not None:
        policy._induce_and_plan()

    engine = None
    is_done = None
    load_error = ""
    try:
        engine, is_done = e3.load_engine(game)
        live_methods.append("arc_executable_world_model.load_engine")
    except Exception as exc:
        load_error = repr(exc)[:160]

    heldout = _heldout_transitions(
        game=game,
        n=heldout_transition_budget,
        seed=random_seed + sum(ord(ch) for ch in game),
    )
    accuracy = 0.0
    cell_recall = 0.0
    if engine is not None and heldout:
        score = e3.WorldModelVerifier(heldout).score(engine)
        accuracy = float(score.accuracy)
        cell_recall = float(score.cell_recall)

    plan = None
    plan_error = ""
    if engine is not None and is_done is not None and policy.root_grid is not None:
        try:
            plan = e3.plan_in_model(
                engine,
                is_done,
                policy.root_grid,
                max_nodes=int(plan_max_nodes),
                max_depth=int(plan_max_depth),
            )
            live_methods.append("arc_executable_world_model.plan_in_model")
        except Exception as exc:
            plan_error = repr(exc)[:160]
    planned = normalize_sequence(plan or policy.plan or [])
    reached = _execute_plan_reaches_l1(game, planned) if planned else False
    row = classify_planned_pool(
        game,
        winning_prefix,
        planned,
        planner_reached_l1_win=reached,
    )
    row.update(
        {
            "engine_heldout_accuracy": round(float(accuracy), 6),
            "engine_heldout_cell_recall": round(float(cell_recall), 6),
            "heldout_transition_count": len(heldout),
            "cold_transition_count": len(policy.transitions),
            "cold_actions": int(cold["actions"]),
            "induction_attempts": list(getattr(policy, "induction_attempts", [])),
            "live_path_methods_called": live_methods,
            "load_engine_error": load_error,
            "plan_error": plan_error,
        }
    )
    return row


def run(
    *,
    root: Path | str = REPO_ROOT,
    offline_arcade_checker: Callable[[], bool] = offline_arcade_available,
    generator_checker: Callable[[], Any] | None = None,
    ground_truth_loader: Callable[[Path], Mapping[str, Sequence[Mapping[str, Any]]]] = (
        load_banked_l1_prefixes
    ),
    environment_games_loader: Callable[[Any], set[str]] = _environment_games,
    game_measurer: Callable[..., Mapping[str, Any]] = measure_game_with_live_induce_plan,
    positive_control_runner: Callable[..., Mapping[str, Any]] = measure_game_with_live_induce_plan,
    live_path_checker: Callable[[Path], bool] = run_orphan_lint,
    now: Clock = time.time,
    write: bool = True,
    write_checkpoints: bool = True,
    transition_budget: int = DEFAULT_TRANSITION_BUDGET,
    heldout_transition_budget: int = DEFAULT_HELDOUT_TRANSITIONS,
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES,
    plan_max_depth: int = DEFAULT_PLAN_MAX_DEPTH,
    soft_elapsed_budget_s: float = DEFAULT_SOFT_ELAPSED_BUDGET_S,
    heldout_games: Sequence[str] = HELDOUT_GAMES,
    positive_control_game: str = DEFAULT_POSITIVE_CONTROL_GAME,
    random_seed: int = RANDOM_SEED,
    proposer: Any | None = None,
) -> JsonDict:
    root_path = Path(root)
    start = now()
    preconditions: JsonDict = {
        "offline_arcade": {"ok": False},
        "generator": {"ok": False, "model": "Qwen3.5-9B-MTP", "igpu_required": True},
        "heldout_games": {"ok": False, "available_games": []},
        "live_path": {"ok": False},
        "planner_blind_to_banked_answer": True,
    }

    def _blocked(verdict: str, *, live_path_reachable: bool = False) -> JsonDict:
        artifact = build_blocked_artifact(
            verdict,
            preconditions_checked=preconditions,
            live_path_reachable=live_path_reachable,
            duration_s=now() - start,
            random_seed=random_seed,
            transition_budget=transition_budget,
            heldout_transitions=heldout_transition_budget,
            plan_max_nodes=plan_max_nodes,
            plan_max_depth=plan_max_depth,
            soft_elapsed_budget_s=soft_elapsed_budget_s,
            heldout_games=heldout_games,
        )
        if write:
            write_artifact(artifact, root=root_path)
        return _validate_or_raise(artifact)

    if not offline_arcade_checker():
        preconditions["offline_arcade"] = {"ok": False, "detail": "offline_arcade_import_failed"}
        return _blocked("blocked_offline_arcade_missing")
    preconditions["offline_arcade"] = {"ok": True}

    if generator_checker is None:
        proposer = proposer or make_live_qwen_proposer()
        generator_result = generator_available(proposer=proposer)
    else:
        generator_result = _normalise_generator_result(generator_checker())
    preconditions["generator"] = generator_result
    if not generator_result.get("ok"):
        return _blocked("blocked_generator_unavailable")

    ground_truth = {
        str(game): normalize_sequence(prefix)
        for game, prefix in ground_truth_loader(root_path).items()
        if normalize_sequence(prefix)
    }
    env_games = set(environment_games_loader(None))
    available_heldout = [
        game
        for game in heldout_games
        if game in ground_truth and game in env_games and game != positive_control_game
    ]
    preconditions["heldout_games"] = {
        "ok": len(available_heldout) >= 3 and positive_control_game in ground_truth,
        "requested_games": list(heldout_games),
        "available_games": list(available_heldout),
        "n_available": len(available_heldout),
        "positive_control_game_present": positive_control_game in ground_truth,
        "positive_control_game": positive_control_game,
    }
    if len(available_heldout) < 3 or positive_control_game not in ground_truth:
        return _blocked("blocked_no_heldout_games")

    live_path_reachable = bool(live_path_checker(root_path))
    preconditions["live_path"] = {"ok": live_path_reachable}
    if not live_path_reachable:
        return _blocked("blocked_live_path_unreachable", live_path_reachable=False)

    per_game: dict[str, JsonDict] = {}
    checkpoint_emitted = False
    partial = False
    for game in available_heldout:
        existing = _load_checkpoint(game, root=root_path)
        if existing is not None:
            per_game[game] = existing
            checkpoint_emitted = True
            continue
        print(f"exp4861 measuring {game}", flush=True)
        row = dict(
            game_measurer(
                game=game,
                winning_prefix=ground_truth[game],
                proposer=proposer,
                transition_budget=transition_budget,
                heldout_transition_budget=heldout_transition_budget,
                plan_max_nodes=plan_max_nodes,
                plan_max_depth=plan_max_depth,
                root=root_path,
                random_seed=random_seed,
            )
        )
        per_game[game] = row
        if write_checkpoints:
            _write_checkpoint(game, row, root=root_path)
            checkpoint_emitted = True
        print(
            "exp4861 measured "
            f"{game}: acc={row.get('engine_heldout_accuracy')} "
            f"bucket={row.get('planned_bucket')}",
            flush=True,
        )
        if now() - start >= float(soft_elapsed_budget_s) and len(per_game) < len(available_heldout):
            partial = True
            break

    positive_control: JsonDict | None = None
    if not partial:
        print(f"exp4861 measuring positive control {positive_control_game}", flush=True)
        positive_control = dict(
            positive_control_runner(
                game=positive_control_game,
                winning_prefix=ground_truth[positive_control_game],
                proposer=proposer,
                transition_budget=transition_budget,
                heldout_transition_budget=heldout_transition_budget,
                plan_max_nodes=plan_max_nodes,
                plan_max_depth=plan_max_depth,
                root=root_path,
                random_seed=random_seed,
            )
        )
        preconditions["positive_control"] = {
            "game": positive_control_game,
            "passed": _positive_control_passed(positive_control),
        }

    artifact = build_artifact(
        per_game_fork=per_game,
        positive_control_game=positive_control_game,
        positive_control_row=positive_control,
        preconditions_checked=preconditions,
        live_path_reachable=live_path_reachable,
        duration_s=now() - start,
        partial=partial,
        checkpoint_emitted=checkpoint_emitted,
        random_seed=random_seed,
        transition_budget=transition_budget,
        heldout_transitions=heldout_transition_budget,
        plan_max_nodes=plan_max_nodes,
        plan_max_depth=plan_max_depth,
        soft_elapsed_budget_s=soft_elapsed_budget_s,
        heldout_games=heldout_games,
    )
    if write:
        write_artifact(artifact, root=root_path)
    return _validate_or_raise(artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    _ = argv
    artifact = run(root=REPO_ROOT, write=True, write_checkpoints=True)
    print(
        json.dumps(
            {
                "artifact": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
                "fork_verdict": artifact["fork_verdict"],
                "partial": artifact["partial"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
