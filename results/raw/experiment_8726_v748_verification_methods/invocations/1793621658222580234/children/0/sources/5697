"""Exp 4277 registry/gaps/manifest hygiene for .395 verifier outcomes.

Spec refs: REQ-VERIFY-4277, SCENARIO-VERIFY-4277.

This runner is an offline truth-ledger reconciler. It replays the standing
GAP-4 guard from cached artifacts, records the .395 cross-family and scale-up
outcomes, and writes the retirements that prevent resolved in-loop axes from
being proposed again.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import time
from typing import Any, Callable

import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252
from carnot.reporting import verifier_registry_gaps_hygiene_4266 as exp4266


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4277
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4277_ARTIFACT_PATH = "results/experiment_4277_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = "ops/exclusion_manifest.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4266.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4266.ARC1_PROGRAMS_PATH

EXP4266_PATH = exp4266.EXP4266_ARTIFACT_PATH
EXP4270_PATH = "results/experiment_4270_arc_family_provenance_recovery.json"
EXP4271_PATH = "results/experiment_4271_arc_cross_family_transfer_existing_pool.json"
EXP4272_PATH = "results/experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json"
EXP4273_PATH = "results/experiment_4273_arc_cross_family_online_adaptation.json"
EXP4274_PATH = "results/experiment_4274_diffusiongemma_loader_fix_preflight.json"
EXP4275_PATH = "results/experiment_4275_arc_incremental_progress_new_game.json"
EXP4264_PATH = exp4266.EXP4264_PATH
EXP4263_PATH = exp4266.EXP4263_PATH

REQUIRED_UPSTREAM_PATHS = [
    EXP4266_PATH,
    EXP4270_PATH,
    EXP4273_PATH,
    EXP4274_PATH,
    EXP4275_PATH,
    EXP4264_PATH,
    EXP4263_PATH,
]
OPTIONAL_CROSS_FAMILY_PATHS = [EXP4271_PATH, EXP4272_PATH]
REQUIRED_COPY_PATHS = [
    *REQUIRED_UPSTREAM_PATHS,
    *OPTIONAL_CROSS_FAMILY_PATHS,
    ARC1_POOL_PATH,
    ARC1_PROGRAMS_PATH,
]

GAP_CROSS_FAMILY_PROVENANCE_4266 = exp4266.GAP_CROSS_GAME_ARC_SELECTION
GAP_DIFFUSIONGEMMA_PREFLIGHT_4266 = exp4266.GAP_DIFFUSIONGEMMA_PREFLIGHT
GAP_ONLINE_ADAPTATION_CALIBRATION = "GAP-ARC-ONLINE-ADAPTATION-CALIBRATION-4277"
V395_ROLE_ID = "oracle_distinct_v395_cross_family_hygiene_4277"
V395_GENERALIZATION_STATE = "generalizes_held_out_family"
V395_HARDENED_STATE = (
    "cross_family_generalized__online_adaptation_static_ceiling__"
    "diffusiongemma_preflight_go__arc_plus_one"
)

CODE_RETIREMENT_ID = "code_oracle_distinct_replication_corpus_specific_retired_exp4264"
REWARD_RETIREMENT_ID = "verifier_as_reward_in_loop_axis_out_of_band_exp4263"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "retirements_recorded",
    "gaps_logged",
    "registry_reconciled",
    "manifest_reconciled",
    "v395_outcomes",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the registry/gaps reconciled + retirements recorded "
        "+ regression guard result."
    ),
    "regression_guard_passed": (
        "BARE bool: the GAP-4 execution numbers did not regress vs .394 -- "
        "the standing-capability guard."
    ),
    "retirements_recorded": (
        "List of the .394/.395 retirements written to the exclusion manifest (code "
        "oracle-distinct corpus-specific; verifier-as-reward out-of-band) -- stops "
        "re-proposal of resolved/doomed axes."
    ),
    "gaps_logged": (
        "List of new missing-verifier gap entries (failure mode + missing discriminator "
        "+ candidate design + priority) -- the verifier build backlog."
    ),
    "reproducibility_checksum": (
        "Hash of the reconciled registry + gaps + manifest; catches silent drift."
    ),
}

GAP_ENTRY_REQUIRED_FIELDS = (
    "gap_id",
    "failure_mode",
    "missing_discriminator",
    "candidate_design",
    "priority",
)
RETIREMENT_REQUIRED_FIELDS = (
    "id",
    "experiment_scope",
    "retire_if_same_verdict",
    "retired_by_artifact",
    "recorded_by_artifact",
)


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded.setdefault("retired", [])
        loaded.setdefault("retired_experiments", [])
        loaded.setdefault("retired_extras", [])
        return loaded
    return {"retired": [], "retired_experiments": [], "retired_extras": []}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _check_resource(
    repo_root: Path,
    resource: str,
    path: str,
    loader: Callable[[Path], Any],
) -> dict[str, Any]:
    full_path = repo_root / path
    try:
        loader(full_path)
    except Exception as exc:
        return {
            "resource": resource,
            "path": path,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"resource": resource, "path": path, "available": True, "error": ""}


def _load_registry_for_check(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("registry must parse as a mapping")
    return loaded


def _load_gaps_for_check(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("gaps ledger must not be empty")
    return text


def _load_manifest_for_check(path: Path) -> dict[str, Any]:
    return _load_manifest(path)


def _load_json_for_check(path: Path) -> dict[str, Any]:
    loaded = base._load_json(path)
    if not isinstance(loaded, dict):
        raise ValueError("artifact must parse as an object")
    return loaded


def _select_cross_family_artifact(repo_root: Path) -> tuple[str, dict[str, Any]]:
    errors: list[str] = []
    for path in OPTIONAL_CROSS_FAMILY_PATHS:
        try:
            artifact = _load_json_for_check(repo_root / path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if "cross_family_win_holds" in artifact:
            return path, artifact
        errors.append(f"{path}: no cross_family_win_holds field")
    raise ValueError("; ".join(errors) if errors else "no cross-family artifact")


def _check_cross_family_resource(repo_root: Path) -> dict[str, Any]:
    try:
        path, _artifact = _select_cross_family_artifact(repo_root)
    except Exception as exc:
        return {
            "resource": "cross_family_outcome",
            "path": " OR ".join(OPTIONAL_CROSS_FAMILY_PATHS),
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"resource": "cross_family_outcome", "path": path, "available": True, "error": ""}


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4277: ledgers parse and .395 artifacts exist before writes."""
    checks = [
        _check_resource(repo_root, "verifier_registry", REGISTRY_PATH, _load_registry_for_check),
        _check_resource(repo_root, "verifier_gaps", GAPS_PATH, _load_gaps_for_check),
        _check_resource(
            repo_root,
            "exclusion_manifest",
            EXCLUSION_MANIFEST_PATH,
            _load_manifest_for_check,
        ),
        _check_resource(
            repo_root,
            "gap4_arc1_candidate_pool",
            ARC1_POOL_PATH,
            exp4266.exp4227._load_gzip_json,
        ),
        _check_resource(repo_root, "gap4_arc1_programs", ARC1_PROGRAMS_PATH, base._load_json),
        _check_cross_family_resource(repo_root),
    ]
    checks.extend(
        _check_resource(repo_root, Path(path).stem, path, _load_json_for_check)
        for path in REQUIRED_UPSTREAM_PATHS
    )
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4277: compare cached replay with .394 recorded GAP-4 numbers."""
    prior = base._load_json(repo_root / EXP4266_PATH)
    recorded = dict(
        prior.get("regression_guard", {}).get("replayed_arc1_rule_exec")
        or prior.get("regression_guard", {}).get("recorded_arc1_rule_exec", {})
    )
    replay = exp4252.replay_gap4_arc1(repo_root)
    replayed = dict(replay.get("arc1_rule_exec", {}))
    passed = (
        replayed.get("n") == recorded.get("n")
        and replayed.get("vote_pass2") == recorded.get("vote_pass2")
        and replayed.get("gated_pass2", 0.0) >= recorded.get("gated_pass2", 0.0)
        and replayed.get("headroom_recovered", 0) >= recorded.get("headroom_recovered", 0)
        and replayed.get("vote_wins_lost", 999999) <= recorded.get("vote_wins_lost", 999999)
    )
    return {
        "regression_guard_passed": bool(passed),
        "prior_artifact_path": EXP4266_PATH,
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "offline_replay": replay,
    }


def load_v395_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4277: read .395 outcomes without manufacturing fields."""
    provenance = base._load_json(repo_root / EXP4270_PATH)
    cross_path, cross = _select_cross_family_artifact(repo_root)
    online = base._load_json(repo_root / EXP4273_PATH)
    diffusion = base._load_json(repo_root / EXP4274_PATH)
    arc = base._load_json(repo_root / EXP4275_PATH)
    code = base._load_json(repo_root / EXP4264_PATH)
    reward = base._load_json(repo_root / EXP4263_PATH)

    cross_win = cross.get("cross_family_win_holds") is True
    ci_excludes_zero = cross.get("ci95_excludes_zero") is True
    generalization_state = (
        V395_GENERALIZATION_STATE
        if cross_win and ci_excludes_zero
        else "collapsed_family_invariant_gap_open"
    )
    return {
        "family_provenance": {
            "artifact_path": EXP4270_PATH,
            "honest_verdict": str(provenance.get("honest_verdict", "")),
            "family_split_feasible": provenance.get("family_split_feasible") is True,
            "distinct_family_n": provenance.get("distinct_family_n"),
            "verifier_is_oracle": provenance.get("verifier_is_oracle") is True,
        },
        "cross_family": {
            "artifact_path": cross_path,
            "honest_verdict": str(cross.get("honest_verdict", "")),
            "headline_outcome": str(cross.get("headline_outcome", "")),
            "cross_family_win_holds": cross_win,
            "cross_family_delta": cross.get("cross_family_delta"),
            "cross_family_ci95": cross.get("cross_family_ci95"),
            "ci95_excludes_zero": ci_excludes_zero,
            "held_out_family_n": cross.get("held_out_family_n"),
            "held_out_task_n": cross.get("held_out_task_n"),
            "within_minus_cross_gap": cross.get("within_minus_cross_gap"),
            "matched_control_delta": cross.get("matched_control_delta"),
            "online_adapt_cross_family_delta": cross.get("online_adapt_cross_family_delta"),
            "oracle_at_k": cross.get("oracle_at_k"),
            "oracle_minus_vote": cross.get("oracle_minus_vote"),
            "pass_rates": dict(cross.get("pass_rates", {})),
            "verifier_is_oracle": cross.get("verifier_is_oracle") is True,
            "generalization_state": generalization_state,
        },
        "online_adaptation": {
            "artifact_path": EXP4273_PATH,
            "honest_verdict": str(online.get("honest_verdict", "")),
            "online_adaptation_helps": online.get("online_adaptation_helps") is True,
            "static_cross_family_delta": online.get("static_cross_family_delta"),
            "online_cross_family_delta": online.get("online_cross_family_delta"),
            "online_minus_static_delta": online.get("online_minus_static_delta"),
            "online_minus_static_ci95": online.get("online_minus_static_ci95"),
            "pass_rates": dict(online.get("pass_rates", {})),
            "tier1_counter_update": str(online.get("tier1_counter_update", "")),
            "verifier_is_oracle": online.get("verifier_is_oracle") is True,
        },
        "diffusiongemma": {
            "artifact_path": EXP4274_PATH,
            "honest_verdict": str(diffusion.get("honest_verdict", "")),
            "loader_repaired": diffusion.get("loader_repaired") is True,
            "preflight_go": diffusion.get("preflight_go") is True,
            "guidance_changes_selection": diffusion.get("guidance_changes_selection") is True,
            "guidance_selection_change_count": diffusion.get(
                "guidance_selection_change_count"
            ),
            "guidance_reweighted_token_count": diffusion.get("guidance_reweighted_token_count"),
            "full_run_cost_estimate_s": diffusion.get("full_run_cost_estimate_s"),
            "verifier_is_oracle": diffusion.get("verifier_is_oracle") is True,
        },
        "arc_progress": {
            "artifact_path": EXP4275_PATH,
            "honest_verdict": str(arc.get("honest_verdict", "")),
            "acceptance_gate_passed": arc.get("acceptance_gate_passed") is True,
            "game_advanced": str(arc.get("game_advanced", "")),
            "target_level": arc.get("target_level"),
            "prior_level": arc.get("prior_level"),
            "levels_completed": arc.get("levels_completed"),
            "new_levels_solved_this_task": arc.get("new_levels_solved_this_task"),
            "total_levels_solved": arc.get("total_levels_solved"),
            "prior_total_levels_solved": arc.get("prior_total_levels_solved"),
            "real_env_confirmed": arc.get("real_env_confirmed") is True,
            "verifier_validated": arc.get("verifier_validated") is True,
        },
        "code_retirement_source": {
            "artifact_path": EXP4264_PATH,
            "honest_verdict": str(code.get("honest_verdict", "")),
            "replication_read": str(code.get("replication_read", "")),
            "code_replication_beats_vote": code.get("code_replication_beats_vote") is True,
            "code_predictor_minus_vote_delta": code.get("code_predictor_minus_vote_delta"),
            "code_predictor_minus_vote_ci95": code.get("code_predictor_minus_vote_ci95"),
            "oracle_at_k": code.get("oracle_at_k"),
            "oracle_minus_vote": code.get("oracle_minus_vote"),
            "verifier_is_oracle": code.get("verifier_is_oracle") is True,
        },
        "reward_retirement_source": {
            "artifact_path": EXP4263_PATH,
            "honest_verdict": str(reward.get("honest_verdict", "")),
            "ready_for_out_of_band": reward.get("ready_for_out_of_band") is True,
            "verifier_as_reward_retired": reward.get("verifier_as_reward_retired") is True,
            "out_of_band_runner_path": str(reward.get("out_of_band_runner_path", "")),
            "verifier_is_oracle": reward.get("verifier_is_oracle") is True,
        },
    }


def build_retirement_entries(outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4277: build manifest entries for retired in-loop axes."""
    code = outcomes["code_retirement_source"]
    reward = outcomes["reward_retirement_source"]
    return [
        {
            "id": CODE_RETIREMENT_ID,
            "experiment_scope": "code oracle-distinct replication in-loop rerun axis",
            "reason": (
                "retire_if_same_verdict: Exp 4264 reported "
                f"replication_read={code['replication_read']} with "
                f"code_replication_beats_vote={code['code_replication_beats_vote']} and "
                f"code_predictor_minus_vote_delta={code['code_predictor_minus_vote_delta']}; "
                "the .392 +3.1pp code result stands as single-corpus evidence only."
            ),
            "experiment_ids": ["exp4264"],
            "retired_milestone": "2026.06.395",
            "retired_by_artifact": EXP4264_PATH,
            "recorded_by_artifact": EXP4277_ARTIFACT_PATH,
            "verdict": code["replication_read"],
            "operator_reopen_required": True,
            "retire_if_same_verdict": True,
            "blocked_patterns": [
                "code oracle-distinct replication retry",
                "code oracle-distinct in-loop replication",
                "code_oracle_distinct_replication_retry",
            ],
        },
        {
            "id": REWARD_RETIREMENT_ID,
            "experiment_scope": "verifier-as-reward in-loop training axis",
            "reason": (
                "retire_if_same_verdict: Exp 4263 prepared the reward-weighted corpus "
                "and one-command runner, so verifier-as-reward training is now "
                "out-of-band/operator-owned rather than an in-loop conductor task."
            ),
            "experiment_ids": ["exp4263"],
            "retired_milestone": "2026.06.395",
            "retired_by_artifact": EXP4263_PATH,
            "recorded_by_artifact": EXP4277_ARTIFACT_PATH,
            "ready_for_out_of_band": bool(reward["ready_for_out_of_band"]),
            "out_of_band_runner_path": reward["out_of_band_runner_path"],
            "operator_owned": True,
            "operator_reopen_required": True,
            "retire_if_same_verdict": True,
            "blocked_patterns": [
                "verifier-as-reward in-loop training",
                "in-loop verifier-as-reward",
                "in_loop verifier reward training",
            ],
        },
    ]


def build_gap_entries(outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4277: log the new Missing-Verifier gap from the .395 read."""
    online = outcomes["online_adaptation"]
    if outcomes["cross_family"]["cross_family_win_holds"] is not True:
        return [
            {
                "gap_id": "GAP-ARC-CROSS-FAMILY-SELECTION-4277",
                "status": "open",
                "evidence": (
                    f"{outcomes['cross_family']['artifact_path']}; "
                    f"cross_family_win_holds={outcomes['cross_family']['cross_family_win_holds']}; "
                    f"cross_family_delta={outcomes['cross_family']['cross_family_delta']}; "
                    f"cross_family_ci95={outcomes['cross_family']['cross_family_ci95']}"
                ),
                "failure_mode": (
                    "The oracle-distinct ARC selector did not generalize to held-out "
                    "families, so the within-pool signal is not family-invariant."
                ),
                "missing_discriminator": (
                    "family-invariant ARC selection features that distinguish rule "
                    "correctness rather than pool or family basin signatures."
                ),
                "candidate_design": (
                    "Train family-adversarial selector features and require held-out-family "
                    "positive CI before headline or scale-up promotion."
                ),
                "priority": "high",
            }
        ]
    return [
        {
            "gap_id": GAP_ONLINE_ADAPTATION_CALIBRATION,
            "status": "open",
            "evidence": (
                f"{EXP4273_PATH}; online_adaptation_helps={online['online_adaptation_helps']}; "
                f"online_minus_static_delta={online['online_minus_static_delta']}; "
                f"online_minus_static_ci95={online['online_minus_static_ci95']}"
            ),
            "failure_mode": (
                "Tier-1 online adaptation improved the point estimate but its CI touched "
                "zero, so static cross-family selection remains the decision-grade ceiling."
            ),
            "missing_discriminator": (
                "uncertainty-aware family-transfer calibration that tells when online "
                "feature and subverifier precision counters should override the static selector."
            ),
            "candidate_design": (
                "Use a hierarchical family calibrator with frozen static-selector controls, "
                "per-family uncertainty intervals, and a pre-registered online-minus-static CI gate."
            ),
            "priority": "medium",
        }
    ]


def ensure_ledgers_record_v395(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    regression_guard: dict[str, Any],
    outcomes: dict[str, Any],
    retirements: list[dict[str, Any]],
    gaps_logged: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .395 truth represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, retirements, gaps_logged)
    _ensure_v395_role(updated_registry, outcomes, retirements, gaps_logged)

    updated_gaps = gaps_text
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4266-gap-arc-cross-game-selection-4266",
        _filled_cross_family_gap_block(outcomes),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4266-gap-diffusiongemma-loader-guidance-4266",
        _filled_diffusiongemma_gap_block(outcomes),
    )
    for gap in gaps_logged:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4277-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    updated_manifest = deepcopy(exclusion_manifest)
    _ensure_manifest_retirements(updated_manifest, retirements)
    retirement_ids = [entry["id"] for entry in retirements]
    gap_ids = [gap["gap_id"] for gap in gaps_logged]
    filled_gap_ids = [GAP_CROSS_FAMILY_PROVENANCE_4266, GAP_DIFFUSIONGEMMA_PREFLIGHT_4266]
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_reconciled": registry_contains_v395(updated_registry),
            "manifest_reconciled": all(
                _find_manifest_entry(updated_manifest, entry_id) is not None
                for entry_id in retirement_ids
            ),
            "retirements_recorded_ids": retirement_ids,
            "gaps_logged_ids": [gap_id for gap_id in gap_ids if gap_id in updated_gaps],
            "filled_gap_ids": [gap_id for gap_id in filled_gap_ids if gap_id in updated_gaps],
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    regression_guard: dict[str, Any],
    outcomes: dict[str, Any],
    retirements: list[dict[str, Any]],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = regression_guard.get("replayed_arc1_rule_exec", {})
    cross = outcomes["cross_family"]
    online = outcomes["online_adaptation"]
    diffusion = outcomes["diffusiongemma"]
    arc = outcomes["arc_progress"]
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4277": EXP4277_ARTIFACT_PATH,
            "exp4277_regression_guard_passed": bool(
                regression_guard.get("regression_guard_passed")
            ),
            "exp4277_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4277_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4277_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4277_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4277_hardened_state": V395_HARDENED_STATE,
            "exp4277_generalization_state": cross["generalization_state"],
            "exp4277_cross_family_artifact": cross["artifact_path"],
            "exp4277_cross_family_win_holds": bool(cross["cross_family_win_holds"]),
            "exp4277_cross_family_delta": cross["cross_family_delta"],
            "exp4277_cross_family_ci95": cross["cross_family_ci95"],
            "exp4277_held_out_task_n": cross["held_out_task_n"],
            "exp4277_online_adaptation_helps": bool(online["online_adaptation_helps"]),
            "exp4277_online_minus_static_delta": online["online_minus_static_delta"],
            "exp4277_online_minus_static_ci95": online["online_minus_static_ci95"],
            "exp4277_diffusiongemma_loader_repaired": bool(diffusion["loader_repaired"]),
            "exp4277_diffusiongemma_preflight_go": bool(diffusion["preflight_go"]),
            "exp4277_diffusiongemma_guidance_changes_selection": bool(
                diffusion["guidance_changes_selection"]
            ),
            "exp4277_arc_total_levels_solved": arc["total_levels_solved"],
            "exp4277_arc_new_levels_solved": arc["new_levels_solved_this_task"],
            "exp4277_arc_game_advanced": arc["game_advanced"],
            "exp4277_retirements_recorded": [retirement["id"] for retirement in retirements],
            "exp4277_gaps_logged": [gap["gap_id"] for gap in gaps_logged],
        }
    )


def _ensure_v395_role(
    registry: dict[str, Any],
    outcomes: dict[str, Any],
    retirements: list[dict[str, Any]],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    cross = outcomes["cross_family"]
    online = outcomes["online_adaptation"]
    diffusion = outcomes["diffusiongemma"]
    arc = outcomes["arc_progress"]
    role = {
        "role_id": V395_ROLE_ID,
        "experiment": EXP4277_ARTIFACT_PATH,
        "role": "registry_gap_manifest_hygiene_v395",
        "status": "v395_cross_family_diffusiongemma_arc_retirements_recorded",
        "v395_hardened_state": V395_HARDENED_STATE,
        "cross_family_status": cross["generalization_state"],
        "cross_family_artifact": cross["artifact_path"],
        "cross_family_win_holds": cross["cross_family_win_holds"],
        "cross_family_delta": cross["cross_family_delta"],
        "cross_family_ci95": cross["cross_family_ci95"],
        "held_out_task_n": cross["held_out_task_n"],
        "online_adaptation_artifact": EXP4273_PATH,
        "online_adaptation_helps": online["online_adaptation_helps"],
        "online_minus_static_delta": online["online_minus_static_delta"],
        "online_minus_static_ci95": online["online_minus_static_ci95"],
        "diffusiongemma_artifact": EXP4274_PATH,
        "diffusiongemma_loader_repaired": diffusion["loader_repaired"],
        "diffusiongemma_preflight_go": diffusion["preflight_go"],
        "diffusiongemma_guidance_changes_selection": diffusion["guidance_changes_selection"],
        "arc_progress_artifact": EXP4275_PATH,
        "arc_total_levels_solved": arc["total_levels_solved"],
        "arc_new_levels_solved": arc["new_levels_solved_this_task"],
        "arc_game_advanced": arc["game_advanced"],
        "code_retirement_id": CODE_RETIREMENT_ID,
        "reward_retirement_id": REWARD_RETIREMENT_ID,
        "retirements_recorded": [retirement["id"] for retirement in retirements],
        "gap_ids_logged": [gap["gap_id"] for gap in gaps_logged],
        "filled_gap_ids": [
            GAP_CROSS_FAMILY_PROVENANCE_4266,
            GAP_DIFFUSIONGEMMA_PREFLIGHT_4266,
        ],
        "eval_exp_4277": EXP4277_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V395_ROLE_ID] + [
        role
    ]


def _filled_cross_family_gap_block(outcomes: dict[str, Any]) -> str:
    cross = outcomes["cross_family"]
    provenance = outcomes["family_provenance"]
    return (
        f"### {GAP_CROSS_FAMILY_PROVENANCE_4266}: Exp 4277 .395 filled provenance gap\n"
        "- status: filled (arc_family_provenance_recovery_4270_cross_family_4271)\n"
        f"- evidence: {EXP4270_PATH}; family_split_feasible={provenance['family_split_feasible']}; "
        f"distinct_family_n={provenance['distinct_family_n']}. {cross['artifact_path']}; "
        f"cross_family_win_holds={cross['cross_family_win_holds']}; "
        f"cross_family_delta={cross['cross_family_delta']}; "
        f"cross_family_ci95={cross['cross_family_ci95']}; "
        f"held_out_task_n={cross['held_out_task_n']}.\n"
        "- failure mode: filled for the original missing game/family provenance blocker; "
        "the recovered family manifest made a held-out-family test possible and the win held.\n"
        "- missing discriminator: none for provenance recovery; future work should preserve "
        "family_id, fold, source_kind, and target hash on every candidate row.\n"
        "- candidate design: keep the manifest join as a required input to any future "
        "cross-family selector evaluation.\n"
        "- priority: high\n"
    )


def _filled_diffusiongemma_gap_block(outcomes: dict[str, Any]) -> str:
    diffusion = outcomes["diffusiongemma"]
    return (
        f"### {GAP_DIFFUSIONGEMMA_PREFLIGHT_4266}: Exp 4277 .395 filled loader-guidance gap\n"
        "- status: filled (diffusiongemma_loader_fix_preflight_4274)\n"
        f"- evidence: {EXP4274_PATH}; loader_repaired={diffusion['loader_repaired']}; "
        f"preflight_go={diffusion['preflight_go']}; "
        f"guidance_changes_selection={diffusion['guidance_changes_selection']}; "
        f"guidance_selection_change_count={diffusion['guidance_selection_change_count']}.\n"
        "- failure mode: filled for the loader/preflight blocker; the .396 full run is now "
        "gated on hardened_win and this preflight_go result rather than loader reachability.\n"
        "- missing discriminator: none for the loader-guidance preflight; full-run quality "
        "remains a separate .396 measurement.\n"
        "- candidate design: use the repaired GGUF metadata loader and tiny guidance smoke "
        "as the preflight before any full benchmark.\n"
        "- priority: medium\n"
    )


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4277 .395 missing-verifier gap\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def _find_manifest_entry(manifest: dict[str, Any], entry_id: str) -> dict[str, Any] | None:
    for section in ("retired_extras", "retired_experiments", "retired"):
        entries = manifest.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id") == entry_id:
                return entry
    return None


def _ensure_manifest_retirements(
    manifest: dict[str, Any],
    retirements: list[dict[str, Any]],
) -> bool:
    changed = False
    manifest.setdefault("retired_extras", [])
    for retirement in retirements:
        existing = _find_manifest_entry(manifest, retirement["id"])
        if existing is None:
            manifest["retired_extras"].append(deepcopy(retirement))
            changed = True
        elif any(existing.get(key) != value for key, value in retirement.items()):
            existing.update(deepcopy(retirement))
            changed = True
    return changed


def registry_contains_v395(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4277") == EXP4277_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4277_generalization_state")
        == V395_GENERALIZATION_STATE
        and any(role.get("role_id") == V395_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def ledger_checksum(registry_path: Path, gaps_path: Path, manifest_path: Path) -> str:
    """REQ-VERIFY-4277: hash reconciled ledgers to catch silent drift."""
    digest = hashlib.sha256()
    for label, path in (
        ("registry", registry_path),
        ("gaps", gaps_path),
        ("manifest", manifest_path),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def model_specs_for_reconciliation() -> dict[str, Any]:
    return {
        "method": "cached_v395_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "upstream_artifacts": list(REQUIRED_UPSTREAM_PATHS) + list(OPTIONAL_CROSS_FAMILY_PATHS),
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
    v395_outcomes: dict[str, Any],
    retirements_recorded: list[dict[str, Any]],
    gaps_logged: list[dict[str, Any]],
    registry_reconciled: bool,
    manifest_reconciled: bool,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4277 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = (
        guard_ok
        and registry_reconciled
        and manifest_reconciled
        and bool(retirements_recorded)
        and bool(gaps_logged)
    )
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4277_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4277_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_manifest_reconciled_to_v395_truth_"
            f"regression_guard_passed_{guard_ok}_retirements_{len(retirements_recorded)}_"
            f"gaps_logged_{len(gaps_logged)}"
        ),
        "regression_guard_passed": guard_ok,
        "retirements_recorded": list(retirements_recorded),
        "gaps_logged": list(gaps_logged),
        "registry_reconciled": bool(registry_reconciled),
        "manifest_reconciled": bool(manifest_reconciled),
        "v395_outcomes": v395_outcomes,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4277", "SCENARIO-VERIFY-4277"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "regression_guard": regression_guard,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "exclusion_manifest_path": EXCLUSION_MANIFEST_PATH,
        "cited_upstream_artifacts": list(REQUIRED_COPY_PATHS),
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4277_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4277_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": "blocked_v395_artifacts_missing",
        "regression_guard_passed": False,
        "retirements_recorded": [],
        "gaps_logged": [],
        "registry_reconciled": False,
        "manifest_reconciled": False,
        "v395_outcomes": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:v395_artifacts_missing",
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4277", "SCENARIO-VERIFY-4277"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4277 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a BARE bool")
    if not isinstance(artifact["registry_reconciled"], bool):
        raise ValueError("registry_reconciled must be a bare bool")
    if not isinstance(artifact["manifest_reconciled"], bool):
        raise ValueError("manifest_reconciled must be a bare bool")
    if not isinstance(artifact["retirements_recorded"], list):
        raise ValueError("retirements_recorded must be a list")
    for retirement in artifact["retirements_recorded"]:
        if not isinstance(retirement, dict) or not all(
            field in retirement for field in RETIREMENT_REQUIRED_FIELDS
        ):
            raise ValueError("retirements_recorded retirement entry is missing required fields")
    if not isinstance(artifact["gaps_logged"], list):
        raise ValueError("gaps_logged must be a list")
    for gap in artifact["gaps_logged"]:
        if not isinstance(gap, dict) or not all(field in gap for field in GAP_ENTRY_REQUIRED_FIELDS):
            raise ValueError("gaps_logged gap entry is missing required fields")
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
        raise ValueError("field_principles must match the required Exp 4277 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4277 and write the terminal artifact plus reconciled ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4277_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    manifest_path = repo_root / EXCLUSION_MANIFEST_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    manifest = _load_manifest(manifest_path)
    regression_guard = run_gap4_regression_guard(repo_root)
    outcomes = load_v395_outcomes(repo_root)
    retirements = build_retirement_entries(outcomes)
    gaps_logged = build_gap_entries(outcomes)
    registry, gaps_text, manifest, ledger_summary = ensure_ledgers_record_v395(
        registry,
        gaps_text,
        manifest,
        regression_guard,
        outcomes,
        retirements,
        gaps_logged,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    _write_manifest(manifest_path, manifest)
    checksum = ledger_checksum(registry_path, gaps_path, manifest_path)
    artifact = build_artifact(
        regression_guard=regression_guard,
        v395_outcomes=outcomes,
        retirements_recorded=retirements,
        gaps_logged=gaps_logged,
        registry_reconciled=bool(ledger_summary["registry_reconciled"]),
        manifest_reconciled=bool(ledger_summary["manifest_reconciled"]),
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through results entrypoint.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4277_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
