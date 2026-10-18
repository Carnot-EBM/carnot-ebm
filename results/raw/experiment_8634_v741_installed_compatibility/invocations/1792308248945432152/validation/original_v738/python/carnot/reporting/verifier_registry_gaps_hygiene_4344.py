"""Exp 4344 registry/gaps/manifest hygiene for .401 verifier outcomes.

Spec refs: REQ-VERIFY-4344, SCENARIO-VERIFY-4344.

This runner continues the Exp 4333 ledger hygiene line. It uses the robust
aggregate-available helper so a missing .401 artifact is only a gap for that
axis, then records the available .401 truth and preserves the standing GAP-4
execution win.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4310 as exp4310
from carnot.reporting import verifier_registry_gaps_hygiene_4333 as exp4333


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4344
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4344_ARTIFACT_PATH = "results/experiment_4344_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = exp4333.EXCLUSION_MANIFEST_PATH
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4333.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4333.ARC1_PROGRAMS_PATH

EXP4333_PATH = exp4333.EXP4333_ARTIFACT_PATH
EXP4338_PATH = "results/experiment_4338_in_generation_moat_replicate_leak_robust.json"
EXP4339_PATH = "results/experiment_4339_e3_explore_verify_plan_ar25.json"
EXP4340_PATH = "results/experiment_4340_e3_explore_verify_plan_ka59.json"
EXP4341_PATH = "results/experiment_4341_e3_sc25_reproduction.json"
EXP4342_PATH = "results/experiment_4342_self_learning_action_role_cross_game_encoder.json"

OUTCOME_ARTIFACT_PATHS = [
    EXP4338_PATH,
    EXP4339_PATH,
    EXP4340_PATH,
    EXP4341_PATH,
    EXP4342_PATH,
]
REQUIRED_COPY_PATHS = list(
    dict.fromkeys(
        [
            EXP4333_PATH,
            *exp4333.REQUIRED_COPY_PATHS,
            *OUTCOME_ARTIFACT_PATHS,
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
        ]
    )
)

GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER = (
    exp4333.GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER
)
GAP_E3_WORLD_MODEL_RULE_AR25_4339 = "GAP-E3-WORLD-MODEL-RULE-AR25-4339"
GAP_E3_WORLD_MODEL_RULE_KA59_4340 = "GAP-E3-WORLD-MODEL-RULE-KA59-4340"
GAP_ACTION_ROLE_TRANSFER_4342 = "GAP-4342"
CROSS_GAME_TRANSFER_RETIREMENT_ID = "cross_game_value_transfer_retired_exp4342_v401"
IN_GENERATION_MOAT_RETIREMENT_ID = "in_generation_moat_retired_exp4338_v401"

V401_ROLE_ID = "oracle_distinct_v401_registry_gaps_hygiene_4344"
V401_STATE = (
    "in_generation_moat_replicated__e3_ar25_sc25_reproduced__"
    "ka59_hidden_step_counter_gap__action_role_transfer_third_null_retired"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "registry_reconciled",
    "manifest_reconciled",
    "gaps_logged",
    "reproducibility_checksum",
    "v401_outcomes",
    "availability_report",
    "random_seed",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records registry/gaps/manifest reconciled to .401 truth "
        "+ the GAP-4 regression-guard result."
    ),
    "regression_guard_passed": (
        "BARE bool: the GAP-4 ARC execution win did not regress "
        "(the execution-grounded baseline is preserved)."
    ),
    "registry_reconciled": (
        "BARE bool: ops/verifier_registry.yaml reflects the .401 verifier state "
        "(filled gaps moved to status: filled)."
    ),
    "manifest_reconciled": (
        "BARE bool: ops/exclusion_manifest.yaml reflects the .401 retirements "
        "(the in-generation moat if exp4338 retired it; the cross-game-transfer "
        "direction if exp4342 is a 3rd null)."
    ),
    "gaps_logged": (
        "BARE int: the count of .401 missing-verifier gaps appended to "
        "ops/verifier_gaps.md (the E3 residual-mismatch gaps + any new failure "
        "modes) -- the build backlog for future verifiers."
    ),
    "reproducibility_checksum": (
        "Hash of the reconciled registry/gaps/manifest state; lets a third party "
        "verify the reconciliation."
    ),
}

ARTIFACT_KEYS = {
    "4338_in_generation_moat": EXP4338_PATH,
    "4339_e3_ar25": EXP4339_PATH,
    "4340_e3_ka59": EXP4340_PATH,
    "4341_e3_sc25": EXP4341_PATH,
    "4342_action_role_transfer": EXP4342_PATH,
}
ARTIFACT_EXPERIMENT_IDS = {
    "4338_in_generation_moat": 4338,
    "4339_e3_ar25": 4339,
    "4340_e3_ka59": 4340,
    "4341_e3_sc25": 4341,
    "4342_action_role_transfer": 4342,
}

check_preconditions = exp4333.check_preconditions
ledger_checksum = exp4333.ledger_checksum
_load_optional_json = exp4333._load_optional_json
_load_manifest = exp4333._load_manifest


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4344: compare cached replay with the .400 recorded GAP-4 numbers."""
    prior_artifact = base._load_json(repo_root / EXP4333_PATH)
    recorded = exp4310._recorded_arc1_rule_exec(prior_artifact)
    replay_guard = exp4333.run_gap4_regression_guard(repo_root)
    replayed = dict(replay_guard.get("replayed_arc1_rule_exec", {}))
    passed = (
        replayed.get("n") == recorded.get("n")
        and replayed.get("vote_pass2") == recorded.get("vote_pass2")
        and replayed.get("gated_pass2", 0.0) >= recorded.get("gated_pass2", 0.0)
        and replayed.get("headroom_recovered", 0) >= recorded.get("headroom_recovered", 0)
        and replayed.get("vote_wins_lost", 999999) <= recorded.get("vote_wins_lost", 999999)
    )
    return {
        "regression_guard_passed": bool(passed),
        "prior_artifact_path": EXP4333_PATH,
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "exp4333_guard": replay_guard,
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="in_generation_moat",
            required_keys=("4338_in_generation_moat",),
            verdict_fn=lambda present: (
                present["4338_in_generation_moat"].get("in_generation_moat_replicates")
                is True
                and present["4338_in_generation_moat"].get("controls_differentiated")
                is True
                and present["4338_in_generation_moat"].get("scorer_leak_recheck_passed")
                is True
            ),
        ),
        aggregate.AxisSpec(
            name="e3_ar25",
            required_keys=("4339_e3_ar25",),
            verdict_fn=lambda present: (
                present["4339_e3_ar25"].get("offline_reproduced") is True
                and int(present["4339_e3_ar25"].get("reproduced_levels", 0)) > 0
            ),
        ),
        aggregate.AxisSpec(
            name="e3_ka59",
            required_keys=("4340_e3_ka59",),
            verdict_fn=lambda present: (
                present["4340_e3_ka59"].get("offline_reproduced") is True
                and int(present["4340_e3_ka59"].get("reproduced_levels", 0)) > 0
            ),
        ),
        aggregate.AxisSpec(
            name="e3_sc25",
            required_keys=("4341_e3_sc25",),
            verdict_fn=lambda present: (
                present["4341_e3_sc25"].get("offline_reproduced") is True
                and int(present["4341_e3_sc25"].get("reproduced_levels", 0)) > 0
            ),
        ),
        aggregate.AxisSpec(
            name="cross_game_transfer",
            required_keys=("4342_action_role_transfer",),
            verdict_fn=lambda present: present["4342_action_role_transfer"].get(
                "learned_encoder_transfer_helps"
            )
            is True,
        ),
    ]


def load_v401_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4344: read available .401 outcomes through robust availability."""
    raw_artifacts: dict[str, Any] = {}
    artifact_errors: dict[str, str] = {}
    for key, rel_path in ARTIFACT_KEYS.items():
        payload, error = _load_optional_json(repo_root, rel_path)
        raw_artifacts[key] = payload
        if error:
            artifact_errors[key] = error

    availability_report = aggregate.aggregate_available_report_gaps(
        raw_artifacts,
        _axis_specs(),
        artifact_experiment_ids=ARTIFACT_EXPERIMENT_IDS,
    )
    exp4333_payload, exp4333_error = _load_optional_json(repo_root, EXP4333_PATH)
    return {
        "v401_outcomes": {
            "in_generation_moat": _read_in_generation(
                raw_artifacts.get("4338_in_generation_moat")
            ),
            "e3": _read_e3(raw_artifacts),
            "cross_game_transfer": _read_cross_game_transfer(
                raw_artifacts.get("4342_action_role_transfer")
            ),
            "exp4333_baseline": _read_exp4333_baseline(exp4333_payload, exp4333_error),
        },
        "availability_report": availability_report,
        "artifact_errors": artifact_errors,
    }


def _read_in_generation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):  # pragma: no cover - exercised through availability.
        return {"artifact_path": EXP4338_PATH, "available": False}
    return {
        "artifact_path": EXP4338_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "acceptance_gate": payload.get("acceptance_gate") is True,
        "in_generation_moat_replicates": payload.get("in_generation_moat_replicates")
        is True,
        "controls_differentiated": payload.get("controls_differentiated") is True,
        "scorer_leak_recheck_passed": payload.get("scorer_leak_recheck_passed") is True,
        "benchmark_n": payload.get("benchmark_n"),
        "benchmark_n_per_seed": payload.get("benchmark_n_per_seed"),
        "carnot_minus_best_control_delta": payload.get("carnot_minus_best_control_delta"),
        "carnot_minus_self_reward_smc_delta": payload.get(
            "carnot_minus_self_reward_smc_delta"
        ),
        "carnot_minus_unguided_delta": payload.get("carnot_minus_unguided_delta"),
        "replication_ci95": payload.get("replication_ci95"),
        "independent_leak_recheck": dict(payload.get("independent_leak_recheck", {})),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_e3(raw_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    games = {
        "ar25": _read_e3_single(raw_artifacts.get("4339_e3_ar25"), EXP4339_PATH, "ar25"),
        "ka59": _read_e3_single(raw_artifacts.get("4340_e3_ka59"), EXP4340_PATH, "ka59"),
        "sc25": _read_e3_single(raw_artifacts.get("4341_e3_sc25"), EXP4341_PATH, "sc25"),
    }
    total = sum(
        int(game.get("reproduced_levels") or 0)
        for game in games.values()
        if game.get("offline_reproduced") is True
    )
    return {
        "artifact_paths": [EXP4339_PATH, EXP4340_PATH, EXP4341_PATH],
        "available": any(game.get("available") is True for game in games.values()),
        "offline_reproduced_any": any(
            game.get("offline_reproduced") is True for game in games.values()
        ),
        "reproduced_levels_total": total,
        "games": games,
    }


def _read_e3_single(payload: Any, artifact_path: str, game: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": artifact_path, "game": game, "available": False}
    return {
        "artifact_path": artifact_path,
        "game": str(payload.get("game", game)),
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "offline_reproduced": payload.get("offline_reproduced") is True,
        "plan_executed": payload.get("plan_executed") is True,
        "reproduced_levels": payload.get("reproduced_levels"),
        "residual_mismatch_class": str(payload.get("residual_mismatch_class", "")),
        "verifier_best_accuracy": payload.get("verifier_best_accuracy"),
        "verifier_accuracy_per_round": list(payload.get("verifier_accuracy_per_round", [])),
        "world_model_path": str(payload.get("world_model_path", "")),
        "world_model_sha256": str(payload.get("world_model_sha256", "")),
        "reproducibility_checksum": str(payload.get("reproducibility_checksum", "")),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_cross_game_transfer(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):  # pragma: no cover - exercised through availability.
        return {
            "artifact_path": EXP4342_PATH,
            "available": False,
            "missing_verifier_gaps": [],
        }
    return {
        "artifact_path": EXP4342_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "acceptance_gate_passed": payload.get("acceptance_gate_passed") is True,
        "learned_encoder_transfer_helps": payload.get("learned_encoder_transfer_helps")
        is True,
        "positive_control_passed": payload.get("positive_control_passed") is True,
        "cross_game_state_reduction": payload.get("cross_game_state_reduction"),
        "cross_game_state_reduction_ci95": payload.get("cross_game_state_reduction_ci95"),
        "n_held_out_games": payload.get("n_held_out_games"),
        "n_held_out_levels": payload.get("n_held_out_levels"),
        "missing_verifier_gaps": list(payload.get("missing_verifier_gaps", [])),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_exp4333_baseline(payload: Any, error: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):  # pragma: no cover - copied precondition covers normal path.
        return {"artifact_path": EXP4333_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4333_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "regression_guard_passed": payload.get("regression_guard_passed") is True,
        "registry_reconciled": payload.get("registry_reconciled") is True,
        "manifest_reconciled": payload.get("manifest_reconciled") is True,
        "gaps_logged": len(payload.get("gaps_logged", [])),
        "reproducibility_checksum": str(payload.get("reproducibility_checksum", "")),
    }


def build_gap_entries(outcome_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4344: collect the .401 missing-verifier gaps to append."""
    outcomes = outcome_bundle["v401_outcomes"]
    games = outcomes["e3"]["games"]
    gaps: list[dict[str, Any]] = []

    ar25 = games.get("ar25", {})
    if _has_residual_gap(ar25):
        gaps.append(_e3_world_model_gap("ar25", ar25))
    ka59 = games.get("ka59", {})
    if _has_residual_gap(ka59):
        gaps.append(_e3_world_model_gap("ka59", ka59))

    cross_game = outcomes["cross_game_transfer"]
    if (
        cross_game.get("available") is True
        and cross_game.get("learned_encoder_transfer_helps") is not True
    ):
        gaps.extend(_upstream_or_fallback(cross_game.get("missing_verifier_gaps", [])))

    deduped: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        deduped[gap["gap_id"]] = gap
    return list(deduped.values())


def _has_residual_gap(row: Mapping[str, Any]) -> bool:
    residual = str(row.get("residual_mismatch_class", ""))
    return row.get("available") is True and bool(residual) and residual != "none"


def _e3_world_model_gap(game: str, row: Mapping[str, Any]) -> dict[str, Any]:
    gap_id = {
        "ar25": GAP_E3_WORLD_MODEL_RULE_AR25_4339,
        "ka59": GAP_E3_WORLD_MODEL_RULE_KA59_4340,
    }[game]
    status = "open_residual_after_l1_reproduction" if row.get("offline_reproduced") else "open"
    return {
        "gap_id": gap_id,
        "status": status,
        "evidence": (
            f"{row.get('artifact_path')}; game={game}; offline_reproduced="
            f"{row.get('offline_reproduced')}; reproduced_levels="
            f"{row.get('reproduced_levels')}; verifier_best_accuracy="
            f"{row.get('verifier_best_accuracy')}; residual_mismatch_class="
            f"{row.get('residual_mismatch_class')}"
        ),
        "failure_mode": (
            f"E3 induced world model for {game} still exposes residual mismatch "
            f"{row.get('residual_mismatch_class')} after the .401 run"
        ),
        "missing_discriminator": (
            f"{game} executable world-model rule coverage for "
            f"{row.get('residual_mismatch_class')}"
        ),
        "candidate_design": (
            "mine the divergent transition traces, add the missing action/rule cases "
            "to the executable model, and keep halt-on-divergence plus reproduce() as the gate"
        ),
        "priority": "high",
    }


def _upstream_or_fallback(upstream_gaps: Any) -> list[dict[str, Any]]:
    valid = [
        _normalize_upstream_gap(upstream)
        for upstream in upstream_gaps
        if isinstance(upstream, Mapping)
    ]
    return valid or [_action_role_transfer_gap()]


def _normalize_upstream_gap(upstream: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": str(upstream.get("gap_id", GAP_ACTION_ROLE_TRANSFER_4342)),
        "status": str(upstream.get("status", "open_third_null_retired_direction")),
        "evidence": f"{EXP4342_PATH}; upstream_missing_verifier_gap=true",
        "failure_mode": str(upstream.get("failure_mode", "")),
        "missing_discriminator": str(upstream.get("missing_discriminator", "")),
        "candidate_design": str(upstream.get("candidate_design", "")),
        "priority": str(upstream.get("priority", "high")),
    }


def _action_role_transfer_gap() -> dict[str, Any]:  # pragma: no cover - upstream supplies gap.
    return {
        "gap_id": GAP_ACTION_ROLE_TRANSFER_4342,
        "status": "open_third_null_retired_direction",
        "evidence": f"{EXP4342_PATH}; learned_encoder_transfer_helps=False",
        "failure_mode": (
            "game-agnostic action-role interaction value head did not produce a "
            "decision-grade held-out OfflineSolver state reduction"
        ),
        "missing_discriminator": "transferable object-interaction value representation",
        "candidate_design": (
            "larger interaction encoder, richer affordance discovery, or more "
            "reproduced traces before reopening cross-game value transfer"
        ),
        "priority": "high",
    }


def ensure_ledgers_record_v401(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .401 truth represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcome_bundle, gap_entries)
    _ensure_v401_role(updated_registry, outcome_bundle, gap_entries)

    updated_gaps = _mark_prior_in_generation_gap_filled(gaps_text, outcome_bundle)
    for gap in gap_entries:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4344-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    updated_manifest = deepcopy(exclusion_manifest)
    _ensure_cross_game_transfer_retirement(updated_manifest, outcome_bundle)
    gap_ids = [gap["gap_id"] for gap in gap_entries]
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_reconciled": registry_contains_v401(updated_registry),
            "manifest_reconciled": manifest_contains_cross_game_transfer_retirement(
                updated_manifest
            )
            and not manifest_contains_in_generation_moat_retirement(updated_manifest),
            "gaps_logged_ids": [gap_id for gap_id in gap_ids if gap_id in updated_gaps],
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    outcomes = outcome_bundle["v401_outcomes"]
    availability = outcome_bundle["availability_report"]
    arc1 = regression_guard.get("replayed_arc1_rule_exec", {})
    in_generation = outcomes["in_generation_moat"]
    e3 = outcomes["e3"]
    games = e3["games"]
    cross_game = outcomes["cross_game_transfer"]
    eval_update = {
        "eval_exp_4344": EXP4344_ARTIFACT_PATH,
        "exp4344_regression_guard_passed": bool(
            regression_guard.get("regression_guard_passed")
        ),
        "exp4344_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
        "exp4344_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
        "exp4344_arc1_headroom_recovered": arc1.get("headroom_recovered"),
        "exp4344_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        "exp4344_v401_state": V401_STATE,
        "exp4344_available_artifact_keys": list(
            availability.get("available_artifact_keys", [])
        ),
        "exp4344_missing_upstream_artifacts": list(
            availability.get("missing_upstream_artifacts", [])
        ),
        "exp4344_flagged_artifacts_excluded": list(
            availability.get("flagged_artifacts_excluded", [])
        ),
        "exp4344_filled_gaps": [GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER],
        "exp4344_in_generation_artifact": EXP4338_PATH,
        "exp4344_in_generation_moat_replicates": in_generation.get(
            "in_generation_moat_replicates"
        ),
        "exp4344_in_generation_controls_differentiated": in_generation.get(
            "controls_differentiated"
        ),
        "exp4344_scorer_leak_recheck_passed": in_generation.get(
            "scorer_leak_recheck_passed"
        ),
        "exp4344_in_generation_benchmark_n": in_generation.get("benchmark_n"),
        "exp4344_in_generation_carnot_minus_best_control_delta": in_generation.get(
            "carnot_minus_best_control_delta"
        ),
        "exp4344_in_generation_replication_ci95": in_generation.get("replication_ci95"),
        "exp4344_e3_reproduced_levels_total": e3.get("reproduced_levels_total"),
        "exp4344_e3_offline_reproduced_any": e3.get("offline_reproduced_any"),
        "exp4344_learned_encoder_artifact": EXP4342_PATH,
        "exp4344_learned_encoder_transfer_helps": cross_game.get(
            "learned_encoder_transfer_helps"
        ),
        "exp4344_cross_game_state_reduction": cross_game.get(
            "cross_game_state_reduction"
        ),
        "exp4344_cross_game_state_reduction_ci95": cross_game.get(
            "cross_game_state_reduction_ci95"
        ),
        "exp4344_cross_game_n_held_out_levels": cross_game.get("n_held_out_levels"),
        "exp4344_cross_game_third_null_retired": True,
        "exp4344_gaps_logged": [gap["gap_id"] for gap in gap_entries],
    }
    for game, row in games.items():
        eval_update[f"exp4344_e3_{game}_offline_reproduced"] = row.get(
            "offline_reproduced"
        )
        eval_update[f"exp4344_e3_{game}_reproduced_levels"] = row.get(
            "reproduced_levels"
        )
        eval_update[f"exp4344_e3_{game}_residual_mismatch_class"] = row.get(
            "residual_mismatch_class"
        )
        eval_update[f"exp4344_e3_{game}_verifier_best_accuracy"] = row.get(
            "verifier_best_accuracy"
        )
    entry.setdefault("eval", {}).update(eval_update)


def _ensure_v401_role(
    registry: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - _ensure_gap4_eval creates it first.
        return
    outcomes = outcome_bundle["v401_outcomes"]
    role = {
        "role_id": V401_ROLE_ID,
        "experiment": EXP4344_ARTIFACT_PATH,
        "role": "registry_gap_manifest_hygiene_v401",
        "status": "v401_outcomes_recorded_with_robust_availability",
        "v401_state": V401_STATE,
        "in_generation_moat_replicates": outcomes["in_generation_moat"].get(
            "in_generation_moat_replicates"
        ),
        "e3_reproduced_levels_total": outcomes["e3"].get("reproduced_levels_total"),
        "learned_encoder_transfer_helps": outcomes["cross_game_transfer"].get(
            "learned_encoder_transfer_helps"
        ),
        "cross_game_state_reduction": outcomes["cross_game_transfer"].get(
            "cross_game_state_reduction"
        ),
        "gap_ids_logged": [gap["gap_id"] for gap in gap_entries],
        "filled_gap_ids": [GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER],
        "eval_exp_4344": EXP4344_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V401_ROLE_ID
    ] + [role]


def _mark_prior_in_generation_gap_filled(
    gaps_text: str,
    outcome_bundle: dict[str, Any],
) -> str:
    in_generation = outcome_bundle["v401_outcomes"]["in_generation_moat"]
    if in_generation.get("in_generation_moat_replicates") is not True:
        return gaps_text
    block = (
        "### GAP-DIFFUSIONGEMMA-SECOND-CORPUS-LEAK-FREE-SCORER-4325: "
        "Exp 4344 .401 filled verifier gap update\n"
        "- status: filled (leak_robust_in_generation_partial_state_scorer_exp4338)\n"
        f"- evidence: {EXP4338_PATH}; in_generation_moat_replicates=True; "
        "controls_differentiated=True; scorer_leak_recheck_passed=True; "
        f"benchmark_n={in_generation.get('benchmark_n')}; "
        f"replication_ci95={in_generation.get('replication_ci95')}.\n"
        "- failure mode: the prior second-corpus leak-free scorer gap is filled by "
        "the leak-robust .401 replication.\n"
        "- missing discriminator: none; the scorer now passes the answer-masked "
        "held-out leak recheck for this replication scope.\n"
        "- candidate design: preserve the exp4338 leak-robust scorer protocol as "
        "the in-generation moat gate.\n"
        "- priority: high\n"
    )
    return base._replace_marked_block(
        gaps_text,
        f"exp4333-{GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER.lower()}",
        block,
    )


def _ensure_cross_game_transfer_retirement(
    manifest: dict[str, Any],
    outcome_bundle: dict[str, Any],
) -> None:
    cross_game = outcome_bundle["v401_outcomes"]["cross_game_transfer"]
    if cross_game.get("learned_encoder_transfer_helps") is True:
        return
    if manifest_contains_cross_game_transfer_retirement(manifest):
        return
    manifest.setdefault("retired_extras", []).append(
        {
            "id": CROSS_GAME_TRANSFER_RETIREMENT_ID,
            "experiment_scope": (
                "ARC cross-game learned value-transfer direction after raw-frame, "
                "small-frame, and action-role encoder nulls"
            ),
            "reason": (
                "retire_if_same_verdict: Exp 4342 is the third cross-game transfer "
                "null after Exp 4318 and Exp 4331; future cross-game value-transfer "
                "reruns need a new discriminator, more reproduced traces, and "
                "operator authorization."
            ),
            "experiment_ids": ["exp4318", "exp4331", "exp4342"],
            "retired_milestone": "2026.06.401",
            "retired_by_artifact": EXP4342_PATH,
            "recorded_by_artifact": EXP4344_ARTIFACT_PATH,
            "operator_reopen_required": True,
            "retire_if_same_verdict": True,
            "blocked_patterns": [
                "cross-game value transfer",
                "cross_game_state_reduction rerun",
                "action-role cross-game encoder rerun",
            ],
        }
    )


def manifest_contains_cross_game_transfer_retirement(manifest: Mapping[str, Any]) -> bool:
    for entry in manifest.get("retired_extras", []):
        if isinstance(entry, Mapping) and entry.get("id") == CROSS_GAME_TRANSFER_RETIREMENT_ID:
            return True
    return False


def manifest_contains_in_generation_moat_retirement(manifest: Mapping[str, Any]) -> bool:
    for entry in manifest.get("retired_extras", []):
        if isinstance(entry, Mapping) and entry.get("id") == IN_GENERATION_MOAT_RETIREMENT_ID:
            return True
    return False


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4344 .401 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def registry_contains_v401(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4344") == EXP4344_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4344_v401_state") == V401_STATE
        and any(role.get("role_id") == V401_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def model_specs_for_reconciliation() -> dict[str, Any]:
    return {
        "method": "cached_v401_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "prior_hygiene_artifact": EXP4333_PATH,
        "upstream_artifacts": list(OUTCOME_ARTIFACT_PATHS),
        "robust_aggregator_helper": "carnot.reporting.capstone_aggregate_available",
        "codex_calls": 0,
        "live_model_inference": False,
        "gguf_inference": False,
        "gpu_inference": False,
        "trm_training_touched": False,
        "stable_checkpoint_write": False,
    }


def build_artifact(
    *,
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gap_entries: list[dict[str, Any]],
    registry_reconciled: bool,
    manifest_reconciled: bool,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4344 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = guard_ok and registry_reconciled and manifest_reconciled
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4344_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4344_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_manifest_reconciled_to_v401_truth_"
            f"regression_guard_passed_{guard_ok}_gaps_logged_{len(gap_entries)}"
        ),
        "regression_guard_passed": guard_ok,
        "registry_reconciled": bool(registry_reconciled),
        "manifest_reconciled": bool(manifest_reconciled),
        "gaps_logged": len(gap_entries),
        "gap_entries": list(gap_entries),
        "reproducibility_checksum": reproducibility_checksum,
        "v401_outcomes": outcome_bundle["v401_outcomes"],
        "availability_report": outcome_bundle["availability_report"],
        "artifact_errors": outcome_bundle.get("artifact_errors", {}),
        "random_seed": RANDOM_SEED,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4344", "SCENARIO-VERIFY-4344"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "regression_guard": regression_guard,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "exclusion_manifest_path": EXCLUSION_MANIFEST_PATH,
        "cited_upstream_artifacts": list(OUTCOME_ARTIFACT_PATHS + [EXP4333_PATH]),
    }
    validate_artifact(artifact)
    return artifact


def _blocked_ledgers_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "ledger")
    artifact = {
        "experiment": "experiment_4344_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4344_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}_unparseable",
        "regression_guard_passed": False,
        "registry_reconciled": False,
        "manifest_reconciled": False,
        "gaps_logged": 0,
        "reproducibility_checksum": f"blocked:{blocked}_unparseable",
        "v401_outcomes": {},
        "availability_report": {},
        "random_seed": RANDOM_SEED,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4344", "SCENARIO-VERIFY-4344"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4344 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if type(artifact["regression_guard_passed"]) is not bool:
        raise ValueError("regression_guard_passed must be a BARE bool")
    if type(artifact["registry_reconciled"]) is not bool:
        raise ValueError("registry_reconciled must be a bare bool")
    if type(artifact["manifest_reconciled"]) is not bool:
        raise ValueError("manifest_reconciled must be a bare bool")
    if isinstance(artifact["gaps_logged"], bool) or not isinstance(artifact["gaps_logged"], int):
        raise ValueError("gaps_logged must be a bare int")
    if not isinstance(artifact["v401_outcomes"], dict):
        raise ValueError("v401_outcomes must be an object")
    if not isinstance(artifact["availability_report"], dict):
        raise ValueError("availability_report must be an object")
    if isinstance(artifact["random_seed"], bool) or not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be a bare int")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if not isinstance(artifact["model_specs"], dict) or not artifact["model_specs"]:
        raise ValueError("model_specs must be a non-empty object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4344 principles")
    if artifact["spec_refs"] != ["REQ-VERIFY-4344", "SCENARIO-VERIFY-4344"]:
        raise ValueError("spec_refs must cite REQ-VERIFY-4344 and SCENARIO-VERIFY-4344")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4344 and write the terminal artifact plus reconciled ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4344_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_ledgers_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    manifest_path = repo_root / EXCLUSION_MANIFEST_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    manifest = _load_manifest(manifest_path)
    regression_guard = run_gap4_regression_guard(repo_root)
    outcome_bundle = load_v401_outcomes(repo_root)
    gap_entries = build_gap_entries(outcome_bundle)
    registry, gaps_text, manifest, ledger_summary = ensure_ledgers_record_v401(
        registry,
        gaps_text,
        manifest,
        regression_guard,
        outcome_bundle,
        gap_entries,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    checksum = ledger_checksum(registry_path, gaps_path, manifest_path)
    artifact = build_artifact(
        regression_guard=regression_guard,
        outcome_bundle=outcome_bundle,
        gap_entries=gap_entries,
        registry_reconciled=bool(ledger_summary["registry_reconciled"]),
        manifest_reconciled=bool(ledger_summary["manifest_reconciled"]),
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through results entrypoint.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4344_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
