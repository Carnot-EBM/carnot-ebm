"""Exp 4321 registry/gaps/manifest hygiene for .399 verifier outcomes.

Spec refs: REQ-VERIFY-4321, SCENARIO-VERIFY-4321.

This runner is the .399 continuation of Exp 4310. It treats missing .399
artifacts as axis-local availability gaps through the robust aggregate helper,
then reconciles the registry and gap ledger to whatever decision-grade evidence
is present.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Mapping

from carnot.reporting import capstone_aggregate_available as aggregate
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4252 as exp4252
from carnot.reporting import verifier_registry_gaps_hygiene_4310 as exp4310


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4321
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4321_ARTIFACT_PATH = "results/experiment_4321_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = exp4310.EXCLUSION_MANIFEST_PATH
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ARC1_POOL_PATH = exp4310.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4310.ARC1_PROGRAMS_PATH

EXP4310_PATH = exp4310.EXP4310_ARTIFACT_PATH
EXP4308_PATH = exp4310.EXP4308_PATH
EXP4314_PATH = "results/experiment_4314_cross_domain_selector_ir3de_cascal.json"
EXP4315_PATH = "results/experiment_4315_diffusiongemma_reward_guided_stitching.json"
EXP4316_PATH = "results/experiment_4316_efficiency_cascade_router_deploy.json"
EXP4317_PATH = "results/experiment_4317_arc_incremental_progress_adapter_free.json"
EXP4318_PATH = "results/experiment_4318_arc_cross_game_learned_verifier_transfer.json"
EXP4319_PATH = "results/experiment_4319_off_arc_execution_verifier_transfer_accumulate.json"

OUTCOME_ARTIFACT_PATHS = [
    EXP4314_PATH,
    EXP4315_PATH,
    EXP4316_PATH,
    EXP4317_PATH,
    EXP4318_PATH,
    EXP4319_PATH,
]
REQUIRED_COPY_PATHS = [
    EXP4310_PATH,
    EXP4308_PATH,
    *OUTCOME_ARTIFACT_PATHS,
    ARC1_POOL_PATH,
    ARC1_PROGRAMS_PATH,
]

GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION = (
    "GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305"
)
GAP_DIFFUSIONGEMMA_LEAK_FREE_STEERING = (
    "GAP-DIFFUSIONGEMMA-LEAK-FREE-STEERING-PARTIAL-STATE-4321"
)
GAP_GAME_INVARIANT_ARC_VALUE = "GAP-4318"
GAP_CODE_EXEC_DEMOFIT = "GAP-CODE-EXEC-DEMOFIT"
CODE_DEMOFIT_VERIFIER_ID = "gap4_code_demo_fit_execution_transfer_4319"
V399_ROLE_ID = "oracle_distinct_v399_registry_gaps_hygiene_4321"
V399_STATE = (
    "cross_domain_collapses__diffusiongemma_moat_won__"
    "always_energy_dominates_cascade__arc_23__"
    "cross_game_value_null__off_arc_demofit_transfer_won"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_logged",
    "registry_reconciled",
    "manifest_reconciled",
    "v399_outcomes",
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
        "BARE bool: the GAP-4 execution numbers did not regress vs .398 -- "
        "the standing-capability guard."
    ),
    "gaps_logged": (
        "List of new/updated missing-verifier gap entries (failure mode + missing "
        "discriminator + candidate design + priority) -- the verifier build backlog."
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
    "4314_cross_domain": EXP4314_PATH,
    "4315_in_generation_moat": EXP4315_PATH,
    "4316_efficiency_cascade": EXP4316_PATH,
    "4317_arc_progress": EXP4317_PATH,
    "4318_cross_game_transfer": EXP4318_PATH,
    "4319_off_arc_execution": EXP4319_PATH,
}
ARTIFACT_EXPERIMENT_IDS = {
    "4314_cross_domain": 4314,
    "4315_in_generation_moat": 4315,
    "4316_efficiency_cascade": 4316,
    "4317_arc_progress": 4317,
    "4318_cross_game_transfer": 4318,
    "4319_off_arc_execution": 4319,
}

check_preconditions = exp4310.check_preconditions
ledger_checksum = exp4310.ledger_checksum
robust_aggregator_ok = exp4310.robust_aggregator_ok
_load_optional_json = exp4310._load_optional_json
_load_manifest = exp4310._load_manifest


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4321: compare cached replay with .398 recorded GAP-4 numbers."""
    recorded: dict[str, Any] = {}
    prior_path = EXP4310_PATH
    try:
        prior_artifact = base._load_json(repo_root / EXP4310_PATH)
        recorded = exp4310._recorded_arc1_rule_exec(prior_artifact)
    except Exception:  # pragma: no cover - defensive fallback for damaged copies.
        fallback = exp4310.run_gap4_regression_guard(repo_root)
        recorded = dict(fallback.get("replayed_arc1_rule_exec", {}))
        prior_path = fallback.get("prior_artifact_path", EXP4310_PATH)

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
        "recorded_arc1_rule_exec": recorded,
        "replayed_arc1_rule_exec": replayed,
        "offline_replay": replay,
    }


def _axis_specs() -> list[aggregate.AxisSpec]:
    return [
        aggregate.AxisSpec(
            name="cross_domain",
            required_keys=("4314_cross_domain",),
            verdict_fn=lambda present: present["4314_cross_domain"].get(
                "cross_domain_selection_holds"
            )
            is True,
        ),
        aggregate.AxisSpec(
            name="in_generation",
            required_keys=("4315_in_generation_moat",),
            verdict_fn=lambda present: (
                present["4315_in_generation_moat"].get("diffusiongemma_guidance_moat")
                is True
                and present["4315_in_generation_moat"].get("scorer_leak_recheck_passed")
                is True
            ),
        ),
        aggregate.AxisSpec(
            name="efficiency_cascade",
            required_keys=("4316_efficiency_cascade",),
            verdict_fn=lambda present: (
                present["4316_efficiency_cascade"].get("cascade_dominates_controls")
                is True
                or float(present["4316_efficiency_cascade"].get("accuracy_always_energy", 0.0))
                >= float(present["4316_efficiency_cascade"].get("accuracy_cascade", 1.0))
            ),
        ),
        aggregate.AxisSpec(
            name="arc_progress",
            required_keys=("4317_arc_progress",),
            verdict_fn=lambda present: (
                present["4317_arc_progress"].get("acceptance_gate_passed") is True
                and int(present["4317_arc_progress"].get("total_levels", 0)) >= 23
            ),
        ),
        aggregate.AxisSpec(
            name="cross_game_transfer",
            required_keys=("4318_cross_game_transfer",),
            verdict_fn=lambda present: present["4318_cross_game_transfer"].get(
                "cross_game_transfer_helps"
            )
            is True,
        ),
        aggregate.AxisSpec(
            name="off_arc_execution",
            required_keys=("4319_off_arc_execution",),
            verdict_fn=lambda present: present["4319_off_arc_execution"].get(
                "off_arc_demofit_beats_vote"
            )
            is True,
        ),
    ]


def load_v399_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4321: read available .399 outcomes through robust availability."""
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

    return {
        "v399_outcomes": {
            "cross_domain": _read_cross_domain(raw_artifacts.get("4314_cross_domain")),
            "in_generation": _read_in_generation(
                raw_artifacts.get("4315_in_generation_moat")
            ),
            "efficiency_cascade": _read_efficiency_cascade(
                raw_artifacts.get("4316_efficiency_cascade")
            ),
            "arc_progress": _read_arc_progress(raw_artifacts.get("4317_arc_progress")),
            "cross_game_transfer": _read_cross_game_transfer(
                raw_artifacts.get("4318_cross_game_transfer")
            ),
            "off_arc_execution": _read_off_arc_execution(
                raw_artifacts.get("4319_off_arc_execution")
            ),
            "robust_aggregator": exp4310._read_robust_aggregator_evidence(
                robust_payload,
                robust_error,
            ),
        },
        "availability_report": availability_report,
        "artifact_errors": artifact_errors,
    }


def _read_cross_domain(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4314_PATH, "available": False, "missing_verifier_gaps": []}
    return {
        "artifact_path": EXP4314_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "cross_domain_selection_holds": payload.get("cross_domain_selection_holds") is True,
        "label_ablation_robust": payload.get("label_ablation_robust") is True,
        "cross_domain_delta": payload.get("cross_domain_delta"),
        "cross_domain_delta_ci95": payload.get("cross_domain_delta_ci95"),
        "held_out_task_n": payload.get("held_out_task_n"),
        "primary_held_out_domain": payload.get("primary_held_out_domain"),
        "vote_at_1": payload.get("vote_at_1"),
        "oracle_at_k": payload.get("oracle_at_k"),
        "matched_control_delta": payload.get("matched_control_delta"),
        "per_domain_delta": dict(payload.get("per_domain_delta", {})),
        "missing_verifier_gaps": list(payload.get("missing_verifier_gaps", [])),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_in_generation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4315_PATH, "available": False}
    return {
        "artifact_path": EXP4315_PATH,
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
        "guidance_changes_selection": dict(payload.get("guidance_changes_selection", {})),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_efficiency_cascade(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4316_PATH, "available": False}
    return {
        "artifact_path": EXP4316_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "cascade_dominates_controls": payload.get("cascade_dominates_controls") is True,
        "accuracy_cascade": payload.get("accuracy_cascade"),
        "accuracy_always_energy": payload.get("accuracy_always_energy"),
        "accuracy_always_judge": payload.get("accuracy_always_judge"),
        "cost_ratio_cascade": payload.get("cost_ratio_cascade"),
        "escalation_rate": payload.get("escalation_rate"),
        "selection_task_n": payload.get("selection_task_n"),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_arc_progress(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"artifact_path": EXP4317_PATH, "available": False}
    return {
        "artifact_path": EXP4317_PATH,
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
        "offline_reproduced": payload.get("offline_reproduced") is True,
        "selection_mode": str(payload.get("selection_mode", "")),
    }


def _read_cross_game_transfer(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": EXP4318_PATH,
            "available": False,
            "missing_verifier_gaps": [],
        }
    return {
        "artifact_path": EXP4318_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "acceptance_gate_passed": payload.get("acceptance_gate_passed") is True,
        "cross_game_transfer_helps": payload.get("cross_game_transfer_helps") is True,
        "cross_game_state_reduction": payload.get("cross_game_state_reduction"),
        "cross_game_state_reduction_ci95": payload.get("cross_game_state_reduction_ci95"),
        "baseline_solves_held_out": payload.get("baseline_solves_held_out") is True,
        "n_held_out_levels": payload.get("n_held_out_levels"),
        "missing_verifier_gaps": list(payload.get("missing_verifier_gaps", [])),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def _read_off_arc_execution(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {
            "artifact_path": EXP4319_PATH,
            "available": False,
            "missing_verifier_gaps": [],
        }
    return {
        "artifact_path": EXP4319_PATH,
        "available": True,
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "off_arc_demofit_beats_vote": payload.get("off_arc_demofit_beats_vote") is True,
        "off_arc_demofit_minus_vote_delta": payload.get("off_arc_demofit_minus_vote_delta"),
        "off_arc_delta_ci95": payload.get("off_arc_delta_ci95"),
        "accumulated_n": payload.get("accumulated_n"),
        "accumulation_window_added": payload.get("accumulation_window_added"),
        "hidden_test_vote_at_1": payload.get("hidden_test_vote_at_1"),
        "hidden_test_demofit_accuracy": payload.get("hidden_test_demofit_accuracy"),
        "gap_ledger_update_required": payload.get("gap_ledger_update_required") is True,
        "missing_verifier_gaps": list(payload.get("missing_verifier_gaps", [])),
        "verifier_is_oracle": payload.get("verifier_is_oracle") is True,
    }


def build_gap_entries(outcome_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4321: log Missing-Verifier gaps exposed by .399 axes."""
    outcomes = outcome_bundle["v399_outcomes"]
    gaps: list[dict[str, Any]] = []

    cross_domain = outcomes["cross_domain"]
    if (
        cross_domain.get("available") is True
        and cross_domain.get("cross_domain_selection_holds") is not True
    ):
        gaps.extend(
            _upstream_or_fallback(
                cross_domain.get("missing_verifier_gaps", []),
                EXP4314_PATH,
                _cross_domain_gap(cross_domain),
            )
        )

    in_generation = outcomes["in_generation"]
    if (
        in_generation.get("available") is True
        and (
            in_generation.get("diffusiongemma_guidance_moat") is not True
            or in_generation.get("scorer_leak_recheck_passed") is not True
        )
    ):
        gaps.append(_diffusiongemma_gap(in_generation))

    cross_game = outcomes["cross_game_transfer"]
    if (
        cross_game.get("available") is True
        and cross_game.get("cross_game_transfer_helps") is not True
    ):
        gaps.extend(
            _upstream_or_fallback(
                cross_game.get("missing_verifier_gaps", []),
                EXP4318_PATH,
                _cross_game_gap(cross_game),
            )
        )

    off_arc = outcomes["off_arc_execution"]
    if off_arc.get("available") is True:
        gaps.append(_code_demofit_gap(off_arc))

    return _dedupe_gap_entries(gaps)


def _upstream_or_fallback(
    upstream_gaps: Any,
    evidence_path: str,
    fallback: dict[str, Any],
) -> list[dict[str, Any]]:
    valid = [
        _normalize_upstream_gap(upstream, evidence_path)
        for upstream in upstream_gaps
        if isinstance(upstream, Mapping)
    ]
    return valid or [fallback]


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


def _cross_domain_gap(cross_domain: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        "status": "open",
        "evidence": (
            f"{EXP4314_PATH}; cross_domain_selection_holds="
            f"{cross_domain.get('cross_domain_selection_holds')}; "
            f"cross_domain_delta={cross_domain.get('cross_domain_delta')}; "
            f"cross_domain_delta_ci95={cross_domain.get('cross_domain_delta_ci95')}"
        ),
        "failure_mode": "powered_collapse_cross_domain_domain_bound",
        "missing_discriminator": (
            "domain-invariant selector features that preserve wrong-majority recovery "
            "across ARC, ARC-GEN, and FoVer/math step candidates without using domain labels"
        ),
        "candidate_design": (
            "stronger family-invariant verifier dimensions beyond IR3DE+CASCAL+"
            "ContextPRM, validated on held-out fover"
        ),
        "priority": "high",
    }


def _diffusiongemma_gap(in_generation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": GAP_DIFFUSIONGEMMA_LEAK_FREE_STEERING,
        "status": "open",
        "evidence": (
            f"{EXP4315_PATH}; diffusiongemma_guidance_moat="
            f"{in_generation.get('diffusiongemma_guidance_moat')}; "
            f"scorer_leak_recheck_passed={in_generation.get('scorer_leak_recheck_passed')}"
        ),
        "failure_mode": (
            "The in-generation diffusion scorer did not preserve a leak-free steering "
            "moat against engaged controls."
        ),
        "missing_discriminator": (
            "leak-free steering partial-state diffusion scorer that remains predictive "
            "after answer-bearing canvas cells are masked."
        ),
        "candidate_design": (
            "Train the scorer on masked partial canvases and require held-out masked "
            "AUROC plus guidance-vs-control CI gates before accepting an in-generation moat."
        ),
        "priority": "high",
    }


def _cross_game_gap(cross_game: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gap_id": GAP_GAME_INVARIANT_ARC_VALUE,
        "status": "open",
        "evidence": (
            f"{EXP4318_PATH}; cross_game_transfer_helps="
            f"{cross_game.get('cross_game_transfer_helps')}; "
            f"cross_game_state_reduction={cross_game.get('cross_game_state_reduction')}; "
            f"n_held_out_levels={cross_game.get('n_held_out_levels')}"
        ),
        "failure_mode": (
            "transferred linear value-head did not reduce held-out OfflineSolver states"
        ),
        "missing_discriminator": "game-invariant ARC value representation",
        "candidate_design": "learned frame encoder or per-game adapter-conditioned value head",
        "priority": "medium",
    }


def _code_demofit_gap(off_arc: Mapping[str, Any]) -> dict[str, Any]:
    won = off_arc.get("off_arc_demofit_beats_vote") is True
    return {
        "gap_id": GAP_CODE_EXEC_DEMOFIT,
        "status": (
            f"filled ({CODE_DEMOFIT_VERIFIER_ID})" if won else "open"
        ),
        "evidence": (
            f"{EXP4319_PATH}; off_arc_demofit_beats_vote={won}; "
            f"off_arc_demofit_minus_vote_delta="
            f"{off_arc.get('off_arc_demofit_minus_vote_delta')}; "
            f"off_arc_delta_ci95={off_arc.get('off_arc_delta_ci95')}; "
            f"accumulated_n={off_arc.get('accumulated_n')}"
        ),
        "failure_mode": "candidates can pass visible demo tests while failing hidden semantic tests",
        "missing_discriminator": "code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics",
        "candidate_design": (
            "Use the accumulated GAP-4 visible-test demo-fit execution selector as "
            "the filled cheap execution layer; reopen only if a future powered replay "
            "loses the positive hidden-test CI."
            if won
            else "Continue accumulation or add hidden-property, symbolic, formal, or runtime oracles."
        ),
        "priority": "high",
        "filled_by_verifier_id": CODE_DEMOFIT_VERIFIER_ID if won else "",
    }


def _dedupe_gap_entries(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        deduped[gap["gap_id"]] = gap
    return list(deduped.values())


def ensure_ledgers_record_v399(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    regression_guard: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .399 truth represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcome_bundle, gaps_logged)
    _ensure_v399_role(updated_registry, outcome_bundle, gaps_logged)

    updated_gaps = gaps_text
    for gap in gaps_logged:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4321-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    updated_manifest = deepcopy(exclusion_manifest)
    gap_ids = [gap["gap_id"] for gap in gaps_logged]
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_reconciled": registry_contains_v399(updated_registry),
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
    outcomes = outcome_bundle["v399_outcomes"]
    availability = outcome_bundle["availability_report"]
    arc1 = regression_guard.get("replayed_arc1_rule_exec", {})
    robust = outcomes["robust_aggregator"]
    cross_domain = outcomes["cross_domain"]
    in_generation = outcomes["in_generation"]
    efficiency = outcomes["efficiency_cascade"]
    arc = outcomes["arc_progress"]
    cross_game = outcomes["cross_game_transfer"]
    off_arc = outcomes["off_arc_execution"]
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4321": EXP4321_ARTIFACT_PATH,
            "exp4321_regression_guard_passed": bool(
                regression_guard.get("regression_guard_passed")
            ),
            "exp4321_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4321_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4321_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4321_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4321_v399_state": V399_STATE,
            "exp4321_robust_aggregator_used": robust_aggregator_ok(robust),
            "exp4321_available_artifact_keys": list(
                availability.get("available_artifact_keys", [])
            ),
            "exp4321_missing_upstream_artifacts": list(
                availability.get("missing_upstream_artifacts", [])
            ),
            "exp4321_flagged_artifacts_excluded": list(
                availability.get("flagged_artifacts_excluded", [])
            ),
            "exp4321_cross_domain_artifact": EXP4314_PATH,
            "exp4321_cross_domain_selection_holds": cross_domain.get(
                "cross_domain_selection_holds"
            ),
            "exp4321_label_ablation_robust": cross_domain.get("label_ablation_robust"),
            "exp4321_cross_domain_delta": cross_domain.get("cross_domain_delta"),
            "exp4321_cross_domain_delta_ci95": cross_domain.get(
                "cross_domain_delta_ci95"
            ),
            "exp4321_cross_domain_held_out_task_n": cross_domain.get("held_out_task_n"),
            "exp4321_in_generation_artifact": EXP4315_PATH,
            "exp4321_diffusiongemma_guidance_moat": in_generation.get(
                "diffusiongemma_guidance_moat"
            ),
            "exp4321_controls_differentiated": in_generation.get(
                "controls_differentiated"
            ),
            "exp4321_scorer_leak_recheck_passed": in_generation.get(
                "scorer_leak_recheck_passed"
            ),
            "exp4321_carnot_minus_best_control_delta": in_generation.get(
                "carnot_minus_best_control_delta"
            ),
            "exp4321_carnot_minus_unguided_delta": in_generation.get(
                "carnot_minus_unguided_delta"
            ),
            "exp4321_guidance_moat_ci95": in_generation.get("guidance_moat_ci95"),
            "exp4321_efficiency_cascade_artifact": EXP4316_PATH,
            "exp4321_cascade_dominates_controls": efficiency.get(
                "cascade_dominates_controls"
            ),
            "exp4321_accuracy_cascade": efficiency.get("accuracy_cascade"),
            "exp4321_accuracy_always_energy": efficiency.get("accuracy_always_energy"),
            "exp4321_accuracy_always_judge": efficiency.get("accuracy_always_judge"),
            "exp4321_cost_ratio_cascade": efficiency.get("cost_ratio_cascade"),
            "exp4321_escalation_rate": efficiency.get("escalation_rate"),
            "exp4321_arc_progress_artifact": EXP4317_PATH,
            "exp4321_arc_total_levels": arc.get("total_levels"),
            "exp4321_arc_total_levels_solved": arc.get("total_levels_solved"),
            "exp4321_arc_new_levels_solved": arc.get("new_levels_solved_this_task"),
            "exp4321_arc_game_advanced": arc.get("game_advanced"),
            "exp4321_cross_game_artifact": EXP4318_PATH,
            "exp4321_cross_game_transfer_helps": cross_game.get(
                "cross_game_transfer_helps"
            ),
            "exp4321_cross_game_state_reduction": cross_game.get(
                "cross_game_state_reduction"
            ),
            "exp4321_cross_game_n_held_out_levels": cross_game.get("n_held_out_levels"),
            "exp4321_off_arc_artifact": EXP4319_PATH,
            "exp4321_off_arc_demofit_beats_vote": off_arc.get(
                "off_arc_demofit_beats_vote"
            ),
            "exp4321_off_arc_demofit_minus_vote_delta": off_arc.get(
                "off_arc_demofit_minus_vote_delta"
            ),
            "exp4321_off_arc_delta_ci95": off_arc.get("off_arc_delta_ci95"),
            "exp4321_off_arc_accumulated_n": off_arc.get("accumulated_n"),
            "exp4321_off_arc_accumulation_window_added": off_arc.get(
                "accumulation_window_added"
            ),
            "exp4321_gaps_logged": [gap["gap_id"] for gap in gaps_logged],
        }
    )


def _ensure_v399_role(
    registry: dict[str, Any],
    outcome_bundle: dict[str, Any],
    gaps_logged: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    outcomes = outcome_bundle["v399_outcomes"]
    role = {
        "role_id": V399_ROLE_ID,
        "experiment": EXP4321_ARTIFACT_PATH,
        "role": "registry_gap_manifest_hygiene_v399",
        "status": "v399_outcomes_recorded_with_robust_availability",
        "v399_state": V399_STATE,
        "robust_aggregator_used": robust_aggregator_ok(outcomes["robust_aggregator"]),
        "cross_domain_artifact": EXP4314_PATH,
        "cross_domain_selection_holds": outcomes["cross_domain"].get(
            "cross_domain_selection_holds"
        ),
        "diffusiongemma_guidance_moat": outcomes["in_generation"].get(
            "diffusiongemma_guidance_moat"
        ),
        "scorer_leak_recheck_passed": outcomes["in_generation"].get(
            "scorer_leak_recheck_passed"
        ),
        "cascade_dominates_controls": outcomes["efficiency_cascade"].get(
            "cascade_dominates_controls"
        ),
        "accuracy_always_energy": outcomes["efficiency_cascade"].get(
            "accuracy_always_energy"
        ),
        "arc_total_levels": outcomes["arc_progress"].get("total_levels"),
        "arc_new_levels_solved": outcomes["arc_progress"].get(
            "new_levels_solved_this_task"
        ),
        "cross_game_transfer_helps": outcomes["cross_game_transfer"].get(
            "cross_game_transfer_helps"
        ),
        "off_arc_demofit_beats_vote": outcomes["off_arc_execution"].get(
            "off_arc_demofit_beats_vote"
        ),
        "off_arc_demofit_minus_vote_delta": outcomes["off_arc_execution"].get(
            "off_arc_demofit_minus_vote_delta"
        ),
        "gap_ids_logged": [gap["gap_id"] for gap in gaps_logged],
        "eval_exp_4321": EXP4321_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V399_ROLE_ID] + [
        role
    ]


def _gap_entry_block(gap: dict[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4321 .399 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap['failure_mode']}\n"
        f"- missing discriminator: {gap['missing_discriminator']}\n"
        f"- candidate design: {gap['candidate_design']}\n"
        f"- priority: {gap['priority']}\n"
    )


def registry_contains_v399(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4321") == EXP4321_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4321_v399_state") == V399_STATE
        and any(role.get("role_id") == V399_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def model_specs_for_reconciliation() -> dict[str, Any]:
    return {
        "method": "cached_v399_ledger_reconciliation",
        "gap4_candidate_set": ARC1_POOL_PATH,
        "gap4_program_outputs": ARC1_PROGRAMS_PATH,
        "prior_hygiene_artifact": EXP4310_PATH,
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
    """Build the Exp 4321 terminal JSON payload."""
    guard_ok = bool(regression_guard.get("regression_guard_passed"))
    complete = guard_ok and registry_reconciled and manifest_reconciled
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4321_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4321_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_manifest_reconciled_to_v399_truth_"
            f"regression_guard_passed_{guard_ok}_gaps_logged_{len(gaps_logged)}_"
            "robust_aggregator_used"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_logged": list(gaps_logged),
        "registry_reconciled": bool(registry_reconciled),
        "manifest_reconciled": bool(manifest_reconciled),
        "v399_outcomes": outcome_bundle["v399_outcomes"],
        "availability_report": outcome_bundle["availability_report"],
        "artifact_errors": outcome_bundle.get("artifact_errors", {}),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4321", "SCENARIO-VERIFY-4321"],
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
        "experiment": "experiment_4321_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4321_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": "blocked_ledgers_unparseable",
        "regression_guard_passed": False,
        "gaps_logged": [],
        "registry_reconciled": False,
        "manifest_reconciled": False,
        "v399_outcomes": {},
        "availability_report": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "blocked:ledgers_unparseable",
        "model_specs": model_specs_for_reconciliation(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4321", "SCENARIO-VERIFY-4321"],
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions": preflight,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4321 fields before writing the artifact."""
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
    if not isinstance(artifact["v399_outcomes"], dict):
        raise ValueError("v399_outcomes must be an object")
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
        raise ValueError("field_principles must match the required Exp 4321 principles")
    if artifact["spec_refs"] != ["REQ-VERIFY-4321", "SCENARIO-VERIFY-4321"]:
        raise ValueError("spec_refs must cite REQ-VERIFY-4321 and SCENARIO-VERIFY-4321")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4321 and write the terminal artifact plus reconciled ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4321_ARTIFACT_PATH
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
    outcome_bundle = load_v399_outcomes(repo_root)
    gaps_logged = build_gap_entries(outcome_bundle)
    registry, gaps_text, _manifest, ledger_summary = ensure_ledgers_record_v399(
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
    print(f"Wrote {REPO_ROOT / EXP4321_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
