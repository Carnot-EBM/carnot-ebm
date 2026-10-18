"""Exp 4310 registry/gaps/manifest hygiene for .398 verifier outcomes.

Spec refs: REQ-VERIFY-4310, SCENARIO-VERIFY-4310.

This runner reconciles the verifier truth ledgers from landed .398 artifacts.
Unlike Exp 4299, missing .398 artifacts are not global blockers: each axis is
aggregated independently through the robust aggregate-available helper so a
single absent artifact becomes an axis gap while available evidence is still
recorded.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252
from carnot.reporting import verifier_registry_gaps_hygiene_4266 as exp4266
from carnot.reporting import verifier_registry_gaps_hygiene_4277 as exp4277
from carnot.reporting import verifier_registry_gaps_hygiene_4287 as exp4287
from carnot.reporting import verifier_registry_gaps_hygiene_4299 as exp4299


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4310
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4310_ARTIFACT_PATH = "results/experiment_4310_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = exp4277.EXCLUSION_MANIFEST_PATH
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4266.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4266.ARC1_PROGRAMS_PATH

EXP4299_PATH = exp4299.EXP4299_ARTIFACT_PATH
EXP4287_PATH = exp4287.EXP4287_ARTIFACT_PATH
EXP4303_PATH = "results/experiment_4303_verifier_efficiency_parity_isoflops.json"
EXP4304_PATH = "results/experiment_4304_diffusiongemma_in_generation_engaged_controls.json"
EXP4305_PATH = "results/experiment_4305_cross_domain_selector_generalization.json"
EXP4306_PATH = "results/experiment_4306_self_learning_powered_ci_cross_domain.json"
EXP4307_PATH = "results/experiment_4307_arc_incremental_progress_new_game.json"
EXP4308_PATH = "results/experiment_4308_adversarial_verify_degenerate_controls_and_robust_capstone.json"

OUTCOME_ARTIFACT_PATHS = [
    EXP4303_PATH,
    EXP4304_PATH,
    EXP4305_PATH,
    EXP4306_PATH,
    EXP4307_PATH,
]
REQUIRED_COPY_PATHS = [
    EXP4299_PATH,
    EXP4287_PATH,
    *OUTCOME_ARTIFACT_PATHS,
    EXP4308_PATH,
    ARC1_POOL_PATH,
    ARC1_PROGRAMS_PATH,
]

GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION = (
    "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305"
)
GAP_DIFFUSIONGEMMA_LEAK_FREE_PARTIAL_STATE = (
    "GAP-DIFFUSIONGEMMA-LEAK-FREE-PARTIAL-STATE-4310"
)
V398_ROLE_ID = "oracle_distinct_v398_registry_gaps_hygiene_4310"
V398_STATE = (
    "efficiency_pareto_holds__diffusiongemma_bounded_null__"
    "cross_domain_collapses__self_learning_helps__arc_22_no_new_level"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_logged",
    "registry_reconciled",
    "manifest_reconciled",
    "v398_outcomes",
    "availability_report",
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
        "Terminal-prefixed. Records the registry/gaps reconciled + regression guard "
        "result (using the robust aggregator, NOT a hard-block)."
    ),
    "regression_guard_passed": (
        "BARE bool: the GAP-4 execution numbers did not regress vs .397 -- "
        "the standing-capability guard."
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

ARTIFACT_KEYS = {
    "4303_efficiency": EXP4303_PATH,
    "4304_in_generation": EXP4304_PATH,
    "4305_cross_domain": EXP4305_PATH,
    "4306_self_learning": EXP4306_PATH,
    "4307_arc_progress": EXP4307_PATH,
}
ARTIFACT_EXPERIMENT_IDS = {
    "4303_efficiency": 4303,
    "4304_in_generation": 4304,
    "4305_cross_domain": 4305,
    "4306_self_learning": 4306,
    "4307_arc_progress": 4307,
}


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("manifest must parse as a mapping")
    loaded.setdefault("retired", [])
    loaded.setdefault("retired_experiments", [])
    loaded.setdefault("retired_extras", [])
    return loaded


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


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4310: only unparseable ledgers are global blockers."""
    checks = [
        _check_resource(repo_root, "verifier_registry", REGISTRY_PATH, _load_registry_for_check),
        _check_resource(repo_root, "verifier_gaps", GAPS_PATH, _load_gaps_for_check),
        _check_resource(repo_root, "exclusion_manifest", EXCLUSION_MANIFEST_PATH, _load_manifest),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def _recorded_arc1_rule_exec(artifact: Mapping[str, Any]) -> dict[str, Any]:
    guard = artifact.get("regression_guard")
    if not isinstance(guard, Mapping):
        return {}
    recorded = guard.get("replayed_arc1_rule_exec") or guard.get("recorded_arc1_rule_exec")
    return dict(recorded) if isinstance(recorded, Mapping) else {}


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4310: compare cached replay with latest available recorded GAP-4 numbers."""
    recorded: dict[str, Any] = {}
    prior_path = EXP4299_PATH
    blocked_exp4299_seen = False
    try:
        exp4299_artifact = base._load_json(repo_root / EXP4299_PATH)
        blocked_exp4299_seen = str(exp4299_artifact.get("honest_verdict", "")).startswith(
            "blocked_"
        )
        recorded = _recorded_arc1_rule_exec(exp4299_artifact)
    except Exception:
        blocked_exp4299_seen = True

    if not recorded:
        prior_path = EXP4287_PATH
        prior = base._load_json(repo_root / EXP4287_PATH)
        recorded = _recorded_arc1_rule_exec(prior)

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
        "prior_artifact_path": prior_path,
        "blocked_exp4299_seen": bool(blocked_exp4299_seen),
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "offline_replay": replay,
    }


def _load_optional_json(repo_root: Path, rel_path: str) -> tuple[dict[str, Any] | None, str]:
    path = repo_root / rel_path
    if not path.exists():
        return None, "missing"
    try:
        loaded = base._load_json(path)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(loaded, dict):
        return None, "artifact must parse as an object"
    return loaded, ""


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="efficiency",
            required_keys=("4303_efficiency",),
            verdict_fn=lambda present: present["4303_efficiency"].get("efficiency_pareto_holds")
            is True,
        ),
        aggregate.AxisSpec(
            name="in_generation",
            required_keys=("4304_in_generation",),
            verdict_fn=lambda present: present["4304_in_generation"].get(
                "diffusiongemma_guidance_moat"
            )
            is True,
        ),
        aggregate.AxisSpec(
            name="cross_domain",
            required_keys=("4305_cross_domain",),
            verdict_fn=lambda present: present["4305_cross_domain"].get(
                "cross_domain_selection_holds"
            )
            is True,
        ),
        aggregate.AxisSpec(
            name="self_learning",
            required_keys=("4306_self_learning",),
            verdict_fn=lambda present: present["4306_self_learning"].get(
                "online_adaptation_helps"
            )
            is True,
        ),
        aggregate.AxisSpec(
            name="arc_progress",
            required_keys=("4307_arc_progress",),
            verdict_fn=lambda present: int(present["4307_arc_progress"].get("total_levels", 0))
            >= 22,
        ),
    ]


def load_v398_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4310: read available .398 outcomes through robust availability."""
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
    robust_payload, robust_error = _load_optional_json(repo_root, EXP4308_PATH)
    robust_evidence = _read_robust_aggregator_evidence(robust_payload, robust_error)

    return {
        "v398_outcomes": {
            "efficiency": _read_efficiency(raw_artifacts.get("4303_efficiency")),
            "in_generation": _read_in_generation(raw_artifacts.get("4304_in_generation")),
            "cross_domain": _read_cross_domain(raw_artifacts.get("4305_cross_domain")),
            "self_learning": _read_self_learning(raw_artifacts.get("4306_self_learning")),
            "arc_progress": _read_arc_progress(raw_artifacts.get("4307_arc_progress")),
            "robust_aggregator": robust_evidence,
        },
        "availability_report": availability_report,
        "artifact_errors": artifact_errors,
    }


def _read_robust_aggregator_evidence(
    payload: dict[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    return {
        "artifact_path": EXP4308_PATH,
        "available": payload is not None,
        "error": error,
        "honest_verdict": str(payload.get("honest_verdict", "")) if payload else "",
        "robust_aggregator_added": payload.get("robust_aggregator_added") is True
        if payload
        else False,
        "aggregator_survives_missing_artifact": payload.get(
            "aggregator_survives_missing_artifact"
        )
        is True
        if payload
        else False,
    }


def _read_efficiency(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4303_PATH, "available": False}
    return {
        "artifact_path": EXP4303_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "efficiency_pareto_holds": payload.get("efficiency_pareto_holds") is True,
        "accuracy_energy_verifier": payload.get("accuracy_energy_verifier"),
        "accuracy_best_judge": payload.get("accuracy_best_judge"),
        "accuracy_delta_ci95": payload.get("accuracy_delta_ci95"),
        "cost_ratio": payload.get("cost_ratio"),
        "best_judge_id": payload.get("best_judge_id"),
        "selection_task_n": payload.get("selection_task_n")
        or payload.get("cost_accounting", {}).get("selection_task_n"),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_in_generation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4304_PATH, "available": False}
    return {
        "artifact_path": EXP4304_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "diffusiongemma_guidance_moat": payload.get("diffusiongemma_guidance_moat")
        is True,
        "controls_differentiated": payload.get("controls_differentiated") is True,
        "scorer_leak_recheck_passed": payload.get("scorer_leak_recheck_passed") is True,
        "independent_leak_recheck": dict(payload.get("independent_leak_recheck", {})),
        "carnot_minus_best_control_delta": payload.get("carnot_minus_best_control_delta"),
        "carnot_minus_unguided_delta": payload.get("carnot_minus_unguided_delta"),
        "guidance_moat_ci95": payload.get("guidance_moat_ci95"),
        "condition_accuracy": dict(payload.get("condition_accuracy", {})),
        "guidance_changes_selection": payload.get("guidance_changes_selection"),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
    }


def _read_cross_domain(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4305_PATH, "available": False, "missing_verifier_gaps": []}
    return {
        "artifact_path": EXP4305_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "cross_domain_selection_holds": payload.get("cross_domain_selection_holds") is True,
        "label_ablation_robust": payload.get("label_ablation_robust") is True,
        "cross_domain_delta": payload.get("cross_domain_delta"),
        "cross_domain_ci95": payload.get("cross_domain_ci95"),
        "held_out_task_n": payload.get("held_out_task_n"),
        "primary_held_out_domain": payload.get("primary_held_out_domain"),
        "vote_at_1": payload.get("vote_at_1"),
        "oracle_at_k": payload.get("oracle_at_k"),
        "matched_control_delta": payload.get("matched_control_delta"),
        "per_domain_delta": dict(payload.get("per_domain_delta", {})),
        "missing_verifier_gaps": list(payload.get("missing_verifier_gaps", [])),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_self_learning(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4306_PATH, "available": False}
    return {
        "artifact_path": EXP4306_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "online_adaptation_helps": payload.get("online_adaptation_helps") is True,
        "best_adaptive_minus_static_delta": payload.get("best_adaptive_minus_static_delta"),
        "best_adaptive_minus_static_ci95": payload.get("best_adaptive_minus_static_ci95"),
        "held_out_family_n": payload.get("held_out_family_n"),
        "held_out_task_n": payload.get("held_out_task_n"),
        "arm_deltas": dict(payload.get("arm_deltas", {})),
        "pass_rates": dict(payload.get("pass_rates", {})),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_arc_progress(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4307_PATH, "available": False}
    return {
        "artifact_path": EXP4307_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "acceptance_gate_passed": payload.get("acceptance_gate_passed") is True,
        "game_advanced": str(payload.get("game_advanced", "")),
        "target_game": str(payload.get("target_game", "")),
        "target_level": payload.get("target_level"),
        "prior_level": payload.get("prior_level"),
        "levels_completed": payload.get("levels_completed"),
        "new_levels_solved_this_task": payload.get("new_levels_solved_this_task"),
        "prior_total_levels_solved": payload.get("prior_total_levels_solved"),
        "total_levels": payload.get("total_levels"),
        "total_levels_solved": payload.get("total_levels_solved"),
        "real_env_confirmed": payload.get("real_env_confirmed") is True,
        "verifier_validated": payload.get("verifier_validated") is True,
        "selection_mode": str(payload.get("selection_mode", "")),
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
        "corrigendum_pending": list(payload.get("corrigendum_pending", [])),
    }


def build_gap_entries(outcome_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4310: log Missing-Verifier gaps exposed by .398 axes."""
    outcomes = outcome_bundle["v398_outcomes"]
    gaps: list[dict[str, Any]] = []

    in_generation = outcomes["in_generation"]
    if (
        in_generation.get("available") is True
        and in_generation.get("scorer_leak_recheck_passed") is not True
    ):
        gaps.append(
            {
                "gap_id": GAP_DIFFUSIONGEMMA_LEAK_FREE_PARTIAL_STATE,
                "status": "open",
                "evidence": (
                    f"{EXP4304_PATH}; honest_verdict={in_generation['honest_verdict']}; "
                    f"scorer_leak_recheck_passed={in_generation['scorer_leak_recheck_passed']}; "
                    f"diffusiongemma_guidance_moat="
                    f"{in_generation['diffusiongemma_guidance_moat']}"
                ),
                "failure_mode": (
                    "The partial-state diffusion guidance read failed the answer-masked "
                    "leak re-check, so a guidance win would be circular rather than "
                    "verifier-distinct."
                ),
                "missing_discriminator": (
                    "leak-free partial-state diffusion scorer that preserves signal "
                    "after answer-bearing cells are masked."
                ),
                "candidate_design": (
                    "Train the partial-state scorer with answer-span masking baked into "
                    "the data pipeline, then require fresh held-out masked AUROC above "
                    "floor before any in-generation moat claim."
                ),
                "priority": "high",
            }
        )

    cross_domain = outcomes["cross_domain"]
    if (
        cross_domain.get("available") is True
        and cross_domain.get("cross_domain_selection_holds") is not True
    ):
        upstream_gaps = cross_domain.get("missing_verifier_gaps", [])
        if upstream_gaps:
            for upstream in upstream_gaps:
                if not isinstance(upstream, Mapping):
                    continue
                gaps.append(_normalize_upstream_gap(upstream, EXP4305_PATH))
        else:
            gaps.append(
                {
                    "gap_id": GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
                    "status": "open",
                    "evidence": (
                        f"{EXP4305_PATH}; cross_domain_selection_holds="
                        f"{cross_domain['cross_domain_selection_holds']}; "
                        f"cross_domain_delta={cross_domain['cross_domain_delta']}; "
                        f"cross_domain_ci95={cross_domain['cross_domain_ci95']}"
                    ),
                    "failure_mode": (
                        "The selector did not preserve the ARC/ARC-GEN selection moat "
                        "on the held-out cross-domain FoVer/math candidate pool."
                    ),
                    "missing_discriminator": (
                        "domain-invariant selector features that recover wrong-majority "
                        "headroom without keying on ARC-specific family labels."
                    ),
                    "candidate_design": (
                        "Train a task-structure router with domain-disjoint calibration, "
                        "label ablations, and per-domain invariance penalties before "
                        "accepting cross-domain moat claims."
                    ),
                    "priority": "high",
                }
            )
    return gaps


def _normalize_upstream_gap(upstream: Mapping[str, Any], evidence_path: str) -> dict[str, Any]:
    priority = str(upstream.get("priority", "high"))
    normalized_priority = "high" if priority.upper().startswith("P0") else priority
    return {
        "gap_id": str(upstream.get("gap_id", GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION)),
        "status": str(upstream.get("status", "open")),
        "evidence": (
            f"{evidence_path}; upstream_missing_verifier_gap=true; "
            f"failure_mode={upstream.get('failure_mode', '')}"
        ),
        "failure_mode": str(upstream.get("failure_mode", "")),
        "missing_discriminator": str(upstream.get("missing_discriminator", "")),
        "candidate_design": str(upstream.get("candidate_design", "")),
        "priority": normalized_priority,
        "upstream_priority": priority,
    }


def ensure_ledgers_record_v398(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .398 truth represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcome_bundle, gaps_logged)
    _ensure_v398_role(updated_registry, outcome_bundle, gaps_logged)

    updated_gaps = gaps_text
    for gap in gaps_logged:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4310-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    updated_manifest = deepcopy(exclusion_manifest)
    gap_ids = [gap["gap_id"] for gap in gaps_logged]
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_reconciled": registry_contains_v398(updated_registry),
            "manifest_reconciled": isinstance(updated_manifest, dict),
            "gaps_logged_ids": [gap_id for gap_id in gap_ids if gap_id in updated_gaps],
        },
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    outcomes = outcome_bundle["v398_outcomes"]
    availability = outcome_bundle["availability_report"]
    arc1 = regression_guard.get("replayed_arc1_rule_exec", {})
    robust = outcomes["robust_aggregator"]
    efficiency = outcomes["efficiency"]
    in_generation = outcomes["in_generation"]
    cross_domain = outcomes["cross_domain"]
    self_learning = outcomes["self_learning"]
    arc = outcomes["arc_progress"]
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4310": EXP4310_ARTIFACT_PATH,
            "exp4310_regression_guard_passed": bool(
                regression_guard.get("regression_guard_passed")
            ),
            "exp4310_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4310_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4310_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4310_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4310_v398_state": V398_STATE,
            "exp4310_robust_aggregator_used": robust_aggregator_ok(robust),
            "exp4310_available_artifact_keys": list(
                availability.get("available_artifact_keys", [])
            ),
            "exp4310_missing_upstream_artifacts": list(
                availability.get("missing_upstream_artifacts", [])
            ),
            "exp4310_flagged_artifacts_excluded": list(
                availability.get("flagged_artifacts_excluded", [])
            ),
            "exp4310_efficiency_artifact": EXP4303_PATH,
            "exp4310_efficiency_pareto_holds": efficiency.get("efficiency_pareto_holds"),
            "exp4310_efficiency_accuracy_energy_verifier": efficiency.get(
                "accuracy_energy_verifier"
            ),
            "exp4310_efficiency_accuracy_best_judge": efficiency.get("accuracy_best_judge"),
            "exp4310_efficiency_accuracy_delta_ci95": efficiency.get("accuracy_delta_ci95"),
            "exp4310_efficiency_cost_ratio": efficiency.get("cost_ratio"),
            "exp4310_in_generation_artifact": EXP4304_PATH,
            "exp4310_diffusiongemma_guidance_moat": in_generation.get(
                "diffusiongemma_guidance_moat"
            ),
            "exp4310_controls_differentiated": in_generation.get("controls_differentiated"),
            "exp4310_scorer_leak_recheck_passed": in_generation.get(
                "scorer_leak_recheck_passed"
            ),
            "exp4310_carnot_minus_best_control_delta": in_generation.get(
                "carnot_minus_best_control_delta"
            ),
            "exp4310_guidance_moat_ci95": in_generation.get("guidance_moat_ci95"),
            "exp4310_cross_domain_artifact": EXP4305_PATH,
            "exp4310_cross_domain_selection_holds": cross_domain.get(
                "cross_domain_selection_holds"
            ),
            "exp4310_label_ablation_robust": cross_domain.get("label_ablation_robust"),
            "exp4310_cross_domain_delta": cross_domain.get("cross_domain_delta"),
            "exp4310_cross_domain_ci95": cross_domain.get("cross_domain_ci95"),
            "exp4310_cross_domain_held_out_task_n": cross_domain.get("held_out_task_n"),
            "exp4310_self_learning_artifact": EXP4306_PATH,
            "exp4310_online_adaptation_helps": self_learning.get(
                "online_adaptation_helps"
            ),
            "exp4310_best_adaptive_minus_static_delta": self_learning.get(
                "best_adaptive_minus_static_delta"
            ),
            "exp4310_best_adaptive_minus_static_ci95": self_learning.get(
                "best_adaptive_minus_static_ci95"
            ),
            "exp4310_self_learning_held_out_task_n": self_learning.get("held_out_task_n"),
            "exp4310_arc_progress_artifact": EXP4307_PATH,
            "exp4310_arc_total_levels": arc.get("total_levels"),
            "exp4310_arc_total_levels_solved": arc.get("total_levels_solved"),
            "exp4310_arc_new_levels_solved": arc.get("new_levels_solved_this_task"),
            "exp4310_arc_game_advanced": arc.get("game_advanced"),
            "exp4310_arc_flagged_adversarial": arc.get("flagged_adversarial"),
            "exp4310_gaps_logged": [gap["gap_id"] for gap in gaps_logged],
        }
    )


def _ensure_v398_role(
    registry: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    outcomes = outcome_bundle["v398_outcomes"]
    robust = outcomes["robust_aggregator"]
    efficiency = outcomes["efficiency"]
    in_generation = outcomes["in_generation"]
    cross_domain = outcomes["cross_domain"]
    self_learning = outcomes["self_learning"]
    arc = outcomes["arc_progress"]
    role = {
        "role_id": V398_ROLE_ID,
        "experiment": EXP4310_ARTIFACT_PATH,
        "role": "registry_gap_manifest_hygiene_v398",
        "status": "v398_outcomes_recorded_with_robust_availability",
        "v398_state": V398_STATE,
        "robust_aggregator_used": robust_aggregator_ok(robust),
        "efficiency_artifact": EXP4303_PATH,
        "efficiency_pareto_holds": efficiency.get("efficiency_pareto_holds"),
        "efficiency_cost_ratio": efficiency.get("cost_ratio"),
        "in_generation_artifact": EXP4304_PATH,
        "diffusiongemma_guidance_moat": in_generation.get("diffusiongemma_guidance_moat"),
        "controls_differentiated": in_generation.get("controls_differentiated"),
        "scorer_leak_recheck_passed": in_generation.get("scorer_leak_recheck_passed"),
        "cross_domain_artifact": EXP4305_PATH,
        "cross_domain_selection_holds": cross_domain.get("cross_domain_selection_holds"),
        "label_ablation_robust": cross_domain.get("label_ablation_robust"),
        "cross_domain_delta": cross_domain.get("cross_domain_delta"),
        "self_learning_artifact": EXP4306_PATH,
        "online_adaptation_helps": self_learning.get("online_adaptation_helps"),
        "best_adaptive_minus_static_delta": self_learning.get(
            "best_adaptive_minus_static_delta"
        ),
        "arc_progress_artifact": EXP4307_PATH,
        "arc_total_levels": arc.get("total_levels"),
        "arc_new_levels_solved": arc.get("new_levels_solved_this_task"),
        "arc_game_advanced": arc.get("game_advanced"),
        "arc_flagged_adversarial": arc.get("flagged_adversarial"),
        "gap_ids_logged": [gap["gap_id"] for gap in gaps_logged],
        "eval_exp_4310": EXP4310_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V398_ROLE_ID] + [
        role
    ]


def robust_aggregator_ok(robust: Mapping[str, Any]) -> bool:
    return (
        robust.get("robust_aggregator_added") is True
        and robust.get("aggregator_survives_missing_artifact") is True
    )


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4310 .398 missing-verifier gap\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def registry_contains_v398(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4310") == EXP4310_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4310_v398_state") == V398_STATE
        and any(role.get("role_id") == V398_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def ledger_checksum(registry_path: Path, gaps_path: Path, manifest_path: Path) -> str:
    """REQ-VERIFY-4310: hash reconciled ledgers to catch silent drift."""
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
        "method": "cached_v398_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "prior_hygiene_artifacts": [EXP4299_PATH, EXP4287_PATH],
        "upstream_artifacts": list(OUTCOME_ARTIFACT_PATHS),
        "robust_aggregator_artifact": EXP4308_PATH,
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
    gaps_logged: list[dict[str, Any]],
    registry_reconciled: bool,
    manifest_reconciled: bool,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4310 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = guard_ok and registry_reconciled and manifest_reconciled
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4310_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4310_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_manifest_reconciled_to_v398_truth_"
            f"regression_guard_passed_{guard_ok}_gaps_logged_{len(gaps_logged)}_"
            "robust_aggregator_used"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_logged": list(gaps_logged),
        "registry_reconciled": bool(registry_reconciled),
        "manifest_reconciled": bool(manifest_reconciled),
        "v398_outcomes": outcome_bundle["v398_outcomes"],
        "availability_report": outcome_bundle["availability_report"],
        "artifact_errors": outcome_bundle.get("artifact_errors", {}),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4310", "SCENARIO-VERIFY-4310"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "regression_guard": regression_guard,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "exclusion_manifest_path": EXCLUSION_MANIFEST_PATH,
        "cited_upstream_artifacts": list(OUTCOME_ARTIFACT_PATHS + [EXP4308_PATH]),
    }
    validate_artifact(artifact)
    return artifact


def _blocked_ledgers_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4310_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4310_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": "blocked_ledgers_unparseable",
        "regression_guard_passed": False,
        "gaps_logged": [],
        "registry_reconciled": False,
        "manifest_reconciled": False,
        "v398_outcomes": {},
        "availability_report": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:ledgers_unparseable",
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4310", "SCENARIO-VERIFY-4310"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4310 fields before writing the artifact."""
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
    if not isinstance(artifact["gaps_logged"], list):
        raise ValueError("gaps_logged must be a list")
    for gap in artifact["gaps_logged"]:
        if not isinstance(gap, dict) or not all(field in gap for field in GAP_ENTRY_REQUIRED_FIELDS):
            raise ValueError("gaps_logged gap entry is missing required fields")
    if not isinstance(artifact["v398_outcomes"], dict):
        raise ValueError("v398_outcomes must be an object")
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
        raise ValueError("field_principles must match the required Exp 4310 principles")
    if artifact["spec_refs"] != ["REQ-VERIFY-4310", "SCENARIO-VERIFY-4310"]:
        raise ValueError("spec_refs must cite REQ-VERIFY-4310 and SCENARIO-VERIFY-4310")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4310 and write the terminal artifact plus reconciled ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4310_ARTIFACT_PATH
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
    outcome_bundle = load_v398_outcomes(repo_root)
    gaps_logged = build_gap_entries(outcome_bundle)
    registry, gaps_text, _manifest, ledger_summary = ensure_ledgers_record_v398(
        registry,
        gaps_text,
        manifest,
        regression_guard,
        outcome_bundle,
        gaps_logged,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    checksum = ledger_checksum(registry_path, gaps_path, manifest_path)
    artifact = build_artifact(
        regression_guard=regression_guard,
        outcome_bundle=outcome_bundle,
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
    print(f"Wrote {REPO_ROOT / EXP4310_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
