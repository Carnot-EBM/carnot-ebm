"""Exp5621 ARC live self-discovery attempt artifact builder.

The module records the V507 standing-floor ARC attempt.  It keeps Exp5620 as an
advisory input only: a blocked or unsafe branch A/B cannot skip the live attempt,
and a successful branch would have to provide an exact promoted configuration
before anything changes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.artifact_gate_annotations import checksum_core

from carnot import experiment_5610_arc_live_self_discovery_levelup_v506 as v506


ARC_LIVE_AGENT_NO_LLM_SUBSTRATE = v506.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE

EXPERIMENT_ID = 5621
EXPERIMENT = "experiment_5621_arc_live_self_discovery_levelup_v507"
MILESTONE = "2026.07.507"
RESULT_RELATIVE_PATH = f"results/{EXPERIMENT}.json"
TRACE_RELATIVE_PATH = f"results/{EXPERIMENT}_trace.json"
REGISTRY_RELATIVE_PATH = v506.REGISTRY_RELATIVE_PATH
EXP5585_RELATIVE_PATH = v506.EXP5585_RELATIVE_PATH
EXP5610_RELATIVE_PATH = v506.RESULT_RELATIVE_PATH
EXP5620_RELATIVE_PATH = "results/experiment_5620_arc_cycle_guarded_live_update_ab.json"

SPEC_REQUIREMENT = "REQ-ARC-FCP-5621"
SOLVE_PROVENANCE = "live_agent_self_discovery"
INFERENCE_SUBSTRATE = ARC_LIVE_AGENT_NO_LLM_SUBSTRATE
RANDOM_SEED = 5621
RANDOM_SEEDS = [RANDOM_SEED]
ACTION_BUDGET = v506.ACTION_BUDGET
STOPPING_RULE = "fixed_action_budget_or_target_level_reached_no_llm_induction_disabled_v507"
FROZEN_GENERATOR_CHOICE = "unchanged_current_live_agent_generator_not_invoked_no_llm"
NO_LLM_BASELINE_CONFIGURATION = {
    "name": "unchanged_current_no_new_llm_live_baseline",
    "llm_invoked": False,
    "new_llm_calls": False,
    "transition_cycle_verifier": None,
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "field_principles": {
        "principle": "principle annotations are carried in the artifact so every required 5621 field is auditable.",
    },
    "registry_precheck": {
        "principle": "duplicate levels receive no credit; all public games, registry depths, arc_loop_solve depths, prior artifact targets, Exp5610's attempted target, and same-v507 attempts are checked before target selection.",
    },
    "target_selection_receipt": {
        "principle": "rotation and authenticated public-game headroom are explicit, so the selected next level is not a duplicate.",
    },
    "live_attempt_executed": {
        "principle": "bare bool true proves the ARC standing floor was a real runtime attempt, not an advisory precheck.",
    },
    "live_branch_configuration": {
        "principle": "Exp5620 promotion use is auditable; blocked or unsafe Exp5620 receipts leave the no-new-LLM baseline unchanged and cannot skip the attempt.",
    },
    "action_budget": {
        "principle": "search cost is bounded before runtime begins.",
    },
    "attempt_trace_path": {
        "principle": "discovery evidence is replayable from a durable trace.",
    },
    "levels_before": {
        "principle": "authoritative registry total before the attempt; the north-star delta is exact.",
    },
    "levels_after": {
        "principle": "authoritative registry total after accepted banking; unchanged on honest nulls.",
    },
    "new_reproducible_levels": {
        "principle": "only newly reproduced levels beyond the precheck depth count.",
    },
    "offline_reproduced": {
        "principle": "a live reach needs independent replay; duplicate or unreplayed reaches do not bank.",
    },
    "registry_updated": {
        "principle": "successful evidence becomes durable, while null attempts leave the registry unchanged.",
    },
    "solve_provenance": {
        "principle": "must equal live_agent_self_discovery for any credited path.",
    },
    "source_files_read": {
        "principle": "must be false; outer-loop source reverse engineering is excluded.",
    },
    "per_game_adapter_used": {
        "principle": "must be false; hidden per-game solvers are not smuggled into live self-discovery credit.",
    },
    "model_specs": {
        "principle": "empty only when no LLM is invoked; otherwise it contains a mandated cached V507 model with invocation receipt.",
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm when no LLM fires; otherwise the authenticated local GGUF substrate.",
    },
    "random_seeds": {
        "principle": "deterministic seeds make the attempt replayable and auditable.",
    },
    "reproducibility_checksum": {
        "principle": "content-addressed artifact checksum catches silent target, trace, or branch-configuration drift.",
    },
    "honest_verdict": {
        "principle": "no-new-level is terminal; a blocked or negative Exp5620 A/B is not permission to skip the attempt.",
    },
}
REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)

read_json = v506.read_json
read_yaml = v506.read_yaml
write_json = v506.write_json
action_trace_sha256 = v506.action_trace_sha256
load_public_env_metadata = v506.load_public_env_metadata
load_arc_loop_depths = v506.load_arc_loop_depths


def _int(value: Any, default: int = 0) -> int:
    return v506._int(value, default)


def _stable_json(value: Any) -> str:
    return v506._stable_json(value)


def _sha256(value: Any) -> str:
    return v506._sha256(value)


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return v506._registry_rows(registry)


def _registry_depth(row: Mapping[str, Any] | None) -> int:
    return v506._registry_depth(row)


def _registry_total(registry: Mapping[str, Any]) -> int:
    return v506._registry_total(registry)


def _public_game_items(
    public_games: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    return v506._public_game_items(public_games)


def _headroom(meta: Mapping[str, Any]) -> int:
    return v506._headroom(meta)


def _level_number(value: Any) -> int:
    if isinstance(value, str):
        return _int(value.strip().lstrip("Ll"))
    return _int(value)


def _target_entries(artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, int], dict[str, Any]] = {}
    source = str(artifact.get("experiment") or artifact.get("experiment_id") or "")

    def add(game: Any, level: Any) -> None:
        game_text = str(game or "")
        level_number = _level_number(level)
        if not game_text or level_number <= 0:
            return
        pairs[(game_text, level_number)] = {
            "game": game_text,
            "target_level": level_number,
            "source_experiment": source,
        }

    add(
        artifact.get("game_targeted")
        or artifact.get("selected_game")
        or artifact.get("target_game"),
        artifact.get("target_level") or artifact.get("selected_level"),
    )
    for key in ("target_selection", "target_selection_receipt"):
        selection = artifact.get(key)
        if isinstance(selection, Mapping):
            add(
                selection.get("selected_game")
                or selection.get("game")
                or selection.get("target_game"),
                selection.get("target_level") or selection.get("selected_level"),
            )
    return list(pairs.values())


def _entries_by_pair(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for artifact in artifacts:
        for entry in _target_entries(artifact):
            entries[(entry["game"], entry["target_level"])] = entry
    return entries


def registry_precheck(
    registry: Mapping[str, Any],
    public_envs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    arc_loop_depths: Mapping[str, int] | None,
    prior_artifacts: Sequence[Mapping[str, Any]] = (),
    current_v507_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the REQ-ARC-FCP-5621 duplicate/headroom precheck receipt."""

    registry_rows = _registry_rows(registry)
    loop_depths = {str(game): _int(depth) for game, depth in (arc_loop_depths or {}).items()}
    prior_targets = _entries_by_pair(prior_artifacts)
    current_targets = _entries_by_pair(current_v507_artifacts)
    exp5610_targets = {
        pair: entry
        for pair, entry in prior_targets.items()
        if "5610" in str(entry.get("source_experiment", ""))
    }
    candidate_rows: list[dict[str, Any]] = []

    for game, meta in _public_game_items(public_envs):
        registry_depth = _registry_depth(registry_rows.get(game))
        arc_loop_depth = loop_depths.get(game, 0)
        authenticated_headroom = _headroom(meta)
        target_level = registry_depth + 1
        pair = (game, target_level)
        reasons: list[str] = []
        if target_level <= registry_depth:  # pragma: no cover - defensive invariant
            reasons.append("already_in_registry")
        if target_level <= arc_loop_depth:
            reasons.append("already_present_in_arc_loop_solve")
        if authenticated_headroom and target_level > authenticated_headroom:
            reasons.append("no_authenticated_headroom")
        if pair in exp5610_targets:
            reasons.append("exp5610_attempted_target")
        elif pair in prior_targets:
            reasons.append("prior_artifact_target")
        if pair in current_targets:
            reasons.append("current_v507_duplicate_target")
        candidate_rows.append(
            {
                "game": game,
                "registry_depth": registry_depth,
                "arc_loop_depth": arc_loop_depth,
                "authenticated_headroom": authenticated_headroom,
                "target_level": target_level,
                "target_label": f"L{target_level}",
                "excluded": bool(reasons),
                "exclude_reasons": reasons,
                "has_authenticated_headroom": target_level <= authenticated_headroom
                if authenticated_headroom
                else False,
            }
        )

    eligible = [row for row in candidate_rows if not row["excluded"]]
    return {
        "spec_ref": SPEC_REQUIREMENT,
        "public_games_checked": len(candidate_rows),
        "registry_games_checked": len(registry_rows),
        "levels_before": _registry_total(registry),
        "registry_total_from_file": registry.get("reproducible_total_levels"),
        "registry_total_from_rows": sum(_registry_depth(row) for row in registry_rows.values()),
        "arc_loop_targets_considered": len(loop_depths),
        "prior_artifact_targets_excluded": sorted(
            prior_targets.values(), key=lambda row: (row["game"], row["target_level"])
        ),
        "exp5610_targets_excluded": sorted(
            exp5610_targets.values(), key=lambda row: (row["game"], row["target_level"])
        ),
        "current_v507_targets_excluded": sorted(
            current_targets.values(), key=lambda row: (row["game"], row["target_level"])
        ),
        "candidate_rows": candidate_rows,
        "eligible_candidates": [
            {
                "game": row["game"],
                "target_level": row["target_level"],
                "target_label": row["target_label"],
                "authenticated_headroom": row["authenticated_headroom"],
                "registry_depth": row["registry_depth"],
            }
            for row in eligible
        ],
        "duplicate_credit_policy": "levels already present in registry, arc_loop_solve, prior artifacts, Exp5610, or same-v507 attempts receive no credit",
        "ok": bool(eligible),
    }


def select_target_from_precheck(precheck: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in precheck.get("candidate_rows", [])
        if isinstance(row, Mapping) and not row.get("excluded")
    ]
    rows = sorted(rows, key=lambda row: (_int(row.get("target_level")), str(row.get("game"))))
    if not rows:
        return {
            "blocked": True,
            "selected_game": None,
            "selected_level": None,
            "target_level": None,
            "rotation_reason": "no_non_duplicate_authenticated_headroom_candidate_v507",
            "selection_reason": "no_non_duplicate_authenticated_headroom_candidate_v507",
            "rotation_order": [],
            "duplicate_targets_rejected": [
                row
                for row in precheck.get("candidate_rows", [])
                if isinstance(row, Mapping) and row.get("excluded")
            ],
        }

    selected = rows[0]
    return {
        "blocked": False,
        "selected_game": selected["game"],
        "selected_level": selected["target_label"],
        "target_level": selected["target_level"],
        "prior_levels_reproduced": selected["registry_depth"],
        "authenticated_headroom": selected["authenticated_headroom"],
        "arc_loop_depth": selected["arc_loop_depth"],
        "rotation_reason": "lowest_next_level_with_authenticated_headroom_after_exp5610_and_v507_exclusions",
        "selection_reason": "rotated_non_duplicate_authenticated_headroom_v507",
        "rotation_order": [
            {
                "game": row["game"],
                "target_level": row["target_level"],
                "target_label": row["target_label"],
            }
            for row in rows
        ],
        "duplicate_targets_rejected": [
            row
            for row in precheck.get("candidate_rows", [])
            if isinstance(row, Mapping) and row.get("excluded")
        ],
    }


def _safety_regression(payload: Mapping[str, Any]) -> bool:
    if payload.get("safety_regression") is True:
        return True
    if _int(payload.get("unsafe_transition_accept_count")) > 0:
        return True
    for gate in payload.get("gates_evaluated", []) or []:
        if not isinstance(gate, Mapping):
            continue
        if (
            str(gate.get("artifact_field")) == "unsafe_transition_accept_count"
            and _int(gate.get("actual")) > 0
        ):
            return True
    return False


def live_branch_configuration_from_exp5620(exp5620: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = exp5620 or {}
    status = str(payload.get("status") or "")
    score_raw = payload.get("live_branch_promotion_score")
    try:
        score = None if score_raw is None else float(score_raw)
    except (TypeError, ValueError):
        score = None
    safety_regression = _safety_regression(payload)
    exact_config = payload.get("live_branch_configuration") or payload.get("promoted_configuration")
    promoted = (
        score == 1.0
        and not safety_regression
        and status != "blocked"
        and isinstance(exact_config, Mapping)
    )
    return {
        "source_artifact": EXP5620_RELATIVE_PATH,
        "source_status": status or None,
        "source_honest_verdict": payload.get("honest_verdict"),
        "live_branch_promotion_score": score,
        "safety_regression": safety_regression,
        "attempt_gated_by_exp5620": False,
        "enabled": bool(promoted),
        "enabled_configuration": dict(exact_config) if promoted else NO_LLM_BASELINE_CONFIGURATION,
        "baseline_unchanged": not promoted,
        "reason": "exact_non_regressing_exp5620_promotion_enabled"
        if promoted
        else "exp5620_blocked_negative_or_unsafe_baseline_unchanged",
    }


def _attempt_filter_configuration(_branch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_artifact": EXP5620_RELATIVE_PATH,
        "attempt_gated_by_exp5620": False,
        "enabled_filters": [],
        "inert_click_pruner": False,
        "object_history_salience": False,
        "baseline_unchanged": True,
        "reason": "v507_no_llm_baseline_filter_surface_unchanged",
    }


def run_live_self_discovery_attempt(
    target_selection_receipt: Mapping[str, Any],
    live_branch_configuration: Mapping[str, Any],
    action_budget: int = ACTION_BUDGET,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:  # pragma: no cover - live ARC SDK boundary
    attempt = v506.run_live_self_discovery_attempt(
        target_selection_receipt=target_selection_receipt,
        filter_configuration=_attempt_filter_configuration(live_branch_configuration),
        action_budget=action_budget,
        random_seed=random_seed,
    )
    attempt["random_seed"] = random_seed
    attempt["random_seeds"] = [random_seed]
    attempt["stopping_rule"] = STOPPING_RULE
    attempt["model_specs"] = (
        [] if not attempt.get("llm_invoked") else attempt.get("model_specs", [])
    )
    attempt["live_branch_configuration"] = dict(live_branch_configuration)
    return attempt


def _accepted_new_levels(
    target_selection_receipt: Mapping[str, Any], attempt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    prior = _int(target_selection_receipt.get("prior_levels_reproduced"))
    target = _int(target_selection_receipt.get("target_level"))
    post = _int(attempt.get("post_levels_reproduced"))
    if not attempt.get("offline_reproduced"):
        return []
    if (
        attempt.get("source_files_read")
        or attempt.get("per_game_adapter_used")
        or attempt.get("offline_bfs_used")
    ):
        return []
    if attempt.get("action_trace_sha256") != attempt.get("trace_replay_checksum"):
        return []
    if post < target or post <= prior:
        return []
    return [
        {
            "game": target_selection_receipt.get("selected_game"),
            "level": level,
        }
        for level in range(prior + 1, post + 1)
    ]


def compute_artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the MEASURED record, excluding post-hoc review annotations.

    Excluding the fabrication gate's ``flagged_adversarial`` / ``corrigendum_*`` stamp is
    load-bearing, not cosmetic. This artifact was stamped by ``adversarial_verify.py`` AFTER it
    landed; hashing that stamp made the artifact's own recorded checksum fail to reproduce, so
    ``validate_artifact`` rejected the committed record -- the mandated review process was
    invalidating the artifact it reviewed. Recomputing with only the gate keys removed reproduces
    the checksum recorded at authoring time EXACTLY, which is what proves the measured record is
    untouched. Every measurement, seed, duration, verdict and substrate declaration is still
    hashed, so real tampering is still caught. See ``carnot.artifact_gate_annotations``.
    """
    return _sha256(checksum_core(artifact))


def build_artifact(
    registry_precheck: Mapping[str, Any],
    target_selection_receipt: Mapping[str, Any],
    live_branch_configuration: Mapping[str, Any],
    attempt: Mapping[str, Any],
    attempt_trace_path: str = TRACE_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    new_levels = _accepted_new_levels(target_selection_receipt, attempt)
    levels_before = _int(registry_precheck.get("levels_before"))
    levels_after = levels_before + len(new_levels)
    selected_game = target_selection_receipt.get("selected_game")
    selected_level = target_selection_receipt.get("selected_level")
    banked = bool(new_levels)
    if banked:
        verdict = f"complete: banked_{selected_game}_{selected_level}_via_live_self_discovery_v507"
    else:
        verdict = f"complete: no_new_arc_level_banked_{selected_game}_{selected_level}_bounded_live_attempt_v507"

    llm_invoked = bool(attempt.get("llm_invoked", False))
    model_specs = list(attempt.get("model_specs") or [])
    artifact: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "schema": "arc_live_self_discovery_levelup_attempt.v2",
        "spec_refs": [SPEC_REQUIREMENT],
        "field_principles": FIELD_PRINCIPLES,
        "result_path": RESULT_RELATIVE_PATH,
        "registry_precheck": registry_precheck,
        "target_selection_receipt": target_selection_receipt,
        "live_attempt_executed": bool(attempt.get("live_attempt_executed")),
        "live_branch_configuration": live_branch_configuration,
        "action_budget": attempt.get("action_budget", ACTION_BUDGET),
        "attempt_trace_path": attempt_trace_path,
        "levels_before": levels_before,
        "levels_after": levels_after,
        "new_reproducible_levels": new_levels,
        "offline_reproduced": banked,
        "registry_updated": banked,
        "solve_provenance": SOLVE_PROVENANCE,
        "source_files_read": bool(attempt.get("source_files_read", False)),
        "per_game_adapter_used": bool(attempt.get("per_game_adapter_used", False)),
        "model_specs": model_specs,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": attempt.get("random_seed", RANDOM_SEED),
        "random_seeds": list(attempt.get("random_seeds") or RANDOM_SEEDS),
        "honest_verdict": verdict,
        "stopping_rule": attempt.get("stopping_rule", STOPPING_RULE),
        "frozen_generator_choice": FROZEN_GENERATOR_CHOICE,
        "llm_invoked": llm_invoked,
        "no_model_specs_required": not llm_invoked,
        "target_reached_live": _int(attempt.get("max_level_reached"))
        >= _int(target_selection_receipt.get("target_level")),
        "max_level_reached": attempt.get("max_level_reached"),
        "post_levels_reproduced": attempt.get("post_levels_reproduced"),
        "action_trace_sha256": attempt.get("action_trace_sha256"),
        "trace_replay_checksum": attempt.get("trace_replay_checksum"),
        "reproduction_gate": attempt.get("reproduction_gate", {}),
        "terminal_reason": attempt.get("terminal_reason"),
        "runtime_reverse_engineering": attempt.get("runtime_reverse_engineering", {}),
        "duration_s": round(float(duration_s or 0.0), 3),
        "tests_run": list(tests_run or []),
    }
    checksum = compute_artifact_checksum(artifact)
    artifact["artifact_checksum"] = checksum
    artifact["reproducibility_checksum"] = checksum
    return artifact


def build_attempt_trace(
    target_selection_receipt: Mapping[str, Any],
    attempt: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": "arc_live_self_discovery_attempt_trace.v2",
        "spec_refs": [SPEC_REQUIREMENT],
        "selected_game": target_selection_receipt.get("selected_game"),
        "selected_level": target_selection_receipt.get("selected_level"),
        "target_selection_receipt": target_selection_receipt,
        "live_branch_configuration": artifact.get("live_branch_configuration"),
        "executed_actions": attempt.get("action_rows", []),
        "observations": attempt.get("observations", []),
        "level_counter_changes": attempt.get("level_counter_changes", []),
        "runtime_reverse_engineering": attempt.get("runtime_reverse_engineering", {}),
        "reproduction_gate": attempt.get("reproduction_gate", {}),
        "action_trace_sha256": attempt.get("action_trace_sha256"),
        "trace_replay_checksum": attempt.get("trace_replay_checksum"),
        "artifact_checksum": artifact.get("artifact_checksum"),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance must be live_agent_self_discovery")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("source_files_read") is not False:
        errors.append("source_files_read must be false")
    if artifact.get("per_game_adapter_used") is not False:
        errors.append("per_game_adapter_used must be false")
    if artifact.get("live_attempt_executed") is not True:
        errors.append("live_attempt_executed must be true")
    branch = artifact.get("live_branch_configuration")
    if not isinstance(branch, Mapping) or branch.get("attempt_gated_by_exp5620") is not False:
        errors.append("live_branch_configuration must be advisory and non-gating")
    if artifact.get("registry_updated") and not artifact.get("new_reproducible_levels"):
        errors.append("registry_updated requires new_reproducible_levels")
    if artifact.get("new_reproducible_levels") and artifact.get("offline_reproduced") is not True:
        errors.append("new_reproducible_levels require offline_reproduced=true")
    if artifact.get("levels_after") != _int(artifact.get("levels_before")) + len(
        artifact.get("new_reproducible_levels") or []
    ):
        errors.append("levels_after must equal levels_before plus new_reproducible_levels")
    if artifact.get("action_trace_sha256") != artifact.get("trace_replay_checksum"):
        errors.append("action trace checksum and replay checksum must match exactly")
    if artifact.get("llm_invoked"):
        if not artifact.get("model_specs"):
            errors.append("llm_invoked requires model_specs")
    elif artifact.get("model_specs") != []:
        errors.append("no-LLM attempts require model_specs=[]")
    if artifact.get("random_seeds") != RANDOM_SEEDS:
        errors.append("random_seeds mismatch")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != compute_artifact_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def _read_existing(
    paths: Sequence[Path],
) -> list[dict[str, Any]]:  # pragma: no cover - filesystem wrapper
    return [payload for payload in (read_json(path) for path in paths) if payload]


def load_current_v507_artifacts(
    root: Path,
) -> list[dict[str, Any]]:  # pragma: no cover - filesystem wrapper
    artifacts: list[dict[str, Any]] = []
    for path in sorted((root / "results").glob("experiment_56*.json")):
        if path.as_posix().endswith(RESULT_RELATIVE_PATH):
            continue
        payload = read_json(path)
        if payload.get("milestone") == MILESTONE:
            artifacts.append(payload)
    return artifacts


def main() -> int:  # pragma: no cover - command wrapper
    root = Path(__file__).resolve().parents[2]
    started = time.time()
    registry = read_yaml(root / REGISTRY_RELATIVE_PATH)
    public_games = load_public_env_metadata()
    loop_depths = load_arc_loop_depths(root)
    prior = _read_existing([root / EXP5585_RELATIVE_PATH, root / EXP5610_RELATIVE_PATH])
    current_v507 = load_current_v507_artifacts(root)
    precheck = registry_precheck(registry, public_games, loop_depths, prior, current_v507)
    target = select_target_from_precheck(precheck)
    branch = live_branch_configuration_from_exp5620(read_json(root / EXP5620_RELATIVE_PATH))
    if target.get("blocked"):
        print(f"{EXPERIMENT}: no non-duplicate authenticated target available")
        return 1

    attempt = run_live_self_discovery_attempt(target, branch)
    artifact = build_artifact(
        precheck,
        target,
        branch,
        attempt,
        attempt_trace_path=TRACE_RELATIVE_PATH,
        duration_s=time.time() - started,
    )
    try:
        validate_artifact(artifact)
    except ValueError as exc:
        print(f"validation error: {exc}")
        return 1
    trace = build_attempt_trace(target, attempt, artifact)
    write_json(root / TRACE_RELATIVE_PATH, trace)
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    print(
        f"{EXPERIMENT}: {artifact['honest_verdict']} "
        f"levels_before={artifact['levels_before']} levels_after={artifact['levels_after']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
