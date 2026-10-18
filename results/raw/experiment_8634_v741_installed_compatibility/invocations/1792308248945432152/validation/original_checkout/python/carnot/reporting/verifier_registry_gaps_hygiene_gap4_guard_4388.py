"""Exp 4388 registry/gaps hygiene, GAP-4 guard, and capstone stamp durability.

Spec refs: REQ-VERIFY-4388, SCENARIO-VERIFY-4388.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available
from carnot.reporting import capstone_v404_4379
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4377 as exp4377


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4388
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_capstone_stamp_audit"

EXP4388_ARTIFACT_PATH = "results/experiment_4388_registry_gaps_hygiene_gap4_guard.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
FOVER_VERIFIER_ID = exp4377.FOVER_VERIFIER_ID

CAPSTONE_V404_PATH = "results/experiment_4379_capstone_v404.json"
EXP4381_PATH = "results/experiment_4381_biprm_detector_localization_abstention.json"
EXP4382_PATH = "results/experiment_4382_detector_localization_skeptic_proof.json"
EXP4383_PATH = "results/experiment_4383_e3_deeper_high_headroom_lookahead.json"
EXP4384_PATH = "results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"
EXP4385_PATH = "results/experiment_4385_detector_self_learning_compounds.json"
EXP4386_PATH = "results/experiment_4386_cross_domain_detection_generalization.json"

V405_ROLE_ID = "oracle_distinct_v405_registry_gaps_hygiene_4388"
V405_STATE = (
    "detector_actionable_null__detector_compounds__cross_domain_generalizes__"
    "arc_total_34_no_new_levels"
)

GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED = "GAP-FOVER-BIPRM-LOCALIZATION-untyped"
GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383 = "GAP-E3-WORLD-MODEL-RULE-LP85-L6-4383"
GAP_E3_WORLD_MODEL_RULE_TU93_L5_4383 = "GAP-E3-WORLD-MODEL-RULE-TU93-L5-4383"
GAP_E3_WORLD_MODEL_RULE_TN36_L8_4383 = "GAP-E3-WORLD-MODEL-RULE-TN36-L8-4383"
GAP_E3_WORLD_MODEL_RULE_TR87_L7_4383 = "GAP-E3-WORLD-MODEL-RULE-TR87-L7-4383"
GAP_E3_WORLD_MODEL_RULE_AR25_L2_4384 = "GAP-E3-WORLD-MODEL-RULE-AR25-L2-4384"
GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384 = "GAP-E3-WORLD-MODEL-RULE-KA59-L2-4384"
GAP_E3_WORLD_MODEL_RULE_FT09_L2_4384 = "GAP-E3-WORLD-MODEL-RULE-FT09-L2-4384"

SPEC_REFS = ["REQ-VERIFY-4388", "SCENARIO-VERIFY-4388"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_regression_guard_passed",
    "capstone_stamp_fix_durable",
    "registries_reconciled",
    "preconditions_checked",
    "reproducibility_checksum",
    "v405_outcomes",
    "registry_reconciliation",
    "gap4_regression_guard",
    "capstone_stamp_fix",
    "random_seed",
    "model_specs",
    "field_principles",
    "spec_refs",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records hygiene reconciled + guard run + stamp fix durable."
    ),
    "gap4_regression_guard_passed": (
        "BARE bool: the ARC oracle-distinct verifier-beats-vote result has not "
        "silently regressed."
    ),
    "capstone_stamp_fix_durable": (
        "BARE bool: the capstone aggregation still propagates verifier_is_oracle "
        "(false for an oracle-distinct moat) -> adversarial_verify.py does NOT "
        "fire CIRCULAR_MOAT_OVERCLAIM on a correct capstone."
    ),
    "registries_reconciled": (
        "BARE bool: verifier_registry.yaml + verifier_gaps.md + "
        "arc_solve_registry.yaml updated with the .405 outcomes (never-prune)."
    ),
    "preconditions_checked": (
        "Records the registry/gaps file readability; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
}

Gap4GuardRunner = Callable[[Path], dict[str, Any]]
CapstoneStampRunner = Callable[[Path], dict[str, Any]]


_json_hash = exp4377._json_hash
_load_optional_json = exp4377._load_optional_json
_bool = exp4377._bool
_int = exp4377._int
_float = exp4377._float
_str = exp4377._str
_list = exp4377._list
_flags_from_report = exp4377._flags_from_report


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4388: read all ledgers before mutating any of them."""
    return exp4377.check_preconditions(repo_root)


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4388: reuse the durable GAP-4 regression guard."""
    return exp4377.run_gap4_regression_guard(repo_root)


def _scorecard_map(rows: list[Any], residual_field: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game", ""))
        if not game:
            continue
        mapped[game] = {
            "game": game,
            "offline_reproduced": row.get("offline_reproduced") is True,
            "new_reproduced_level": row.get("new_reproduced_level"),
            "target_level": row.get("target_level"),
            "prior_best_level": row.get("prior_best_level"),
            "residual_gap_class": str(row.get(residual_field, "")),
            "verifier_accuracy": row.get("verifier_accuracy"),
            "lookahead_fidelity": row.get("lookahead_fidelity"),
            "world_model_path": str(row.get("world_model_path", "")),
            "mind_studio_skill_file": str(row.get("mind_studio_skill_file", "")),
            "mechanic_checks_passed": row.get("mechanic_checks_passed"),
        }
    return mapped


def _read_actionable_detector(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4381_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4381_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "detector_localization_actionable": _bool(payload, "detector_localization_actionable") is True,
        "localization_delta_ci95": _list(payload, "localization_delta_ci95"),
        "localization_f1_by_direction": dict(payload.get("localization_f1_by_direction", {})),
        "abstention_curve": dict(payload.get("abstention_curve", {})),
        "n_traces": _int(payload, "n_traces"),
        "n_error_traces": _int(payload, "n_error_traces"),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_gate_check(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4382_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4382_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "status": _str(payload, "status"),
        "gate_check_summary": _str(payload, "gate_check_summary"),
        "gates_evaluated": _list(payload, "gates_evaluated"),
    }


def _read_e3_partial(
    payload: dict[str, Any] | None,
    error: str,
    artifact_path: str,
    rows_key: str,
    residual_key: str,
) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": artifact_path, "available": False, "error": error, "rows": {}}
    return {
        "artifact_path": artifact_path,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
        "rows": _scorecard_map(_list(payload, rows_key), residual_key),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_detector_compounds(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4385_PATH, "available": False, "error": error}
    learning_curve = _list(payload, "learning_curve")
    final_point = learning_curve[-1] if learning_curve and isinstance(learning_curve[-1], Mapping) else {}
    return {
        "artifact_path": EXP4385_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "detector_compounds": _bool(payload, "detector_compounds") is True,
        "positive_control_passed": _bool(payload, "positive_control_passed") is True,
        "compounding_delta_ci95": _list(payload, "compounding_delta_ci95"),
        "no_learning_baseline": _float(payload, "no_learning_baseline"),
        "learning_curve": learning_curve,
        "final_held_out_localization_f1": final_point.get("held_out_localization_f1"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_cross_domain_detector(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4386_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4386_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "detector_generalizes_cross_domain": _bool(payload, "detector_generalizes_cross_domain") is True,
        "detection_by_domain": _list(payload, "detection_by_domain"),
        "domains_at_chance": _list(payload, "domains_at_chance"),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def load_v405_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4388: read .405 outcomes without fabricating missing artifacts."""
    actionable_payload, actionable_error = _load_optional_json(repo_root, EXP4381_PATH)
    gate_payload, gate_error = _load_optional_json(repo_root, EXP4382_PATH)
    deeper_payload, deeper_error = _load_optional_json(repo_root, EXP4383_PATH)
    blocked_payload, blocked_error = _load_optional_json(repo_root, EXP4384_PATH)
    compounds_payload, compounds_error = _load_optional_json(repo_root, EXP4385_PATH)
    cross_payload, cross_error = _load_optional_json(repo_root, EXP4386_PATH)
    return {
        "actionable_detector": _read_actionable_detector(actionable_payload, actionable_error),
        "localization_skeptic_proof": _read_gate_check(gate_payload, gate_error),
        "arc_e3": {
            "deeper_lookahead": _read_e3_partial(
                deeper_payload,
                deeper_error,
                EXP4383_PATH,
                "per_target_scorecard",
                "residual_win_mechanic_gap_class",
            ),
            "blocked_mechanics": _read_e3_partial(
                blocked_payload,
                blocked_error,
                EXP4384_PATH,
                "per_game_scorecard",
                "residual_gap_class",
            ),
        },
        "detector_self_learning": _read_detector_compounds(compounds_payload, compounds_error),
        "cross_domain_detector": _read_cross_domain_detector(cross_payload, cross_error),
    }


def _gap_entry(
    gap_id: str,
    *,
    status: str,
    evidence: str,
    failure_mode: str,
    missing_discriminator: str,
    candidate_design: str,
    priority: str = "high",
) -> dict[str, Any]:
    return {
        "gap_id": gap_id,
        "status": status,
        "evidence": evidence,
        "failure_mode": failure_mode,
        "missing_discriminator": missing_discriminator,
        "candidate_design": candidate_design,
        "priority": priority,
    }


def build_gap_entries(outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4388: collect .405 residual missing-verifier gaps."""
    entries: dict[str, dict[str, Any]] = {}
    actionable = outcomes["actionable_detector"]
    for gap in actionable.get("missing_verifier_gaps", []):
        if not isinstance(gap, Mapping):
            continue
        gap_id = str(gap.get("gap_id", ""))
        if not gap_id:
            continue
        entries[gap_id] = _gap_entry(
            gap_id,
            status=str(gap.get("status", "open")),
            evidence=(
                f"{EXP4381_PATH}; detector_localization_actionable="
                f"{actionable.get('detector_localization_actionable')}; "
                f"localization_delta_ci95={actionable.get('localization_delta_ci95')}; "
                f"missed_first_error_traces={gap.get('missed_first_error_traces')}"
            ),
            failure_mode=(
                "FoVer detects trace-level error risk but misses the earliest causal "
                f"step for error_class={gap.get('error_class', 'unknown')}"
            ),
            missing_discriminator=str(gap.get("missing_discriminator", "")),
            candidate_design=(
                "add typed step-error labels plus a contrastive earliest-error objective; "
                "report causal L2R separately from offline R2L"
            ),
            priority="medium",
        )

    cross = outcomes["cross_domain_detector"]
    for row in cross.get("domains_at_chance", []):
        if not isinstance(row, Mapping):
            continue
        domain = str(row.get("domain", "unknown"))
        gap_id = f"GAP-DETECTOR-CROSS-DOMAIN-{domain}-4386"
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=(
                f"{EXP4386_PATH}; domain={domain}; auroc={row.get('detection_auroc')}; "
                f"ci95={row.get('auroc_ci95')}; n={row.get('n')}"
            ),
            failure_mode=f"{domain} detector AUROC remains statistically at chance.",
            missing_discriminator=f"domain-specific correctness signal for {domain}",
            candidate_design=(
                "collect a cached scored pool with labels, add domain features, and require "
                "CI95 lower bound above 0.5 before claiming detector generalization"
            ),
            priority="medium",
        )

    deeper = outcomes["arc_e3"]["deeper_lookahead"]["rows"]
    blocked = outcomes["arc_e3"]["blocked_mechanics"]["rows"]
    gap_specs = [
        (deeper, "lp85", "L6", GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383, EXP4383_PATH),
        (deeper, "tu93", "L5", GAP_E3_WORLD_MODEL_RULE_TU93_L5_4383, EXP4383_PATH),
        (deeper, "tn36", "L8", GAP_E3_WORLD_MODEL_RULE_TN36_L8_4383, EXP4383_PATH),
        (deeper, "tr87", "L7", GAP_E3_WORLD_MODEL_RULE_TR87_L7_4383, EXP4383_PATH),
        (blocked, "ar25", "L2", GAP_E3_WORLD_MODEL_RULE_AR25_L2_4384, EXP4384_PATH),
        (blocked, "ka59", "L2", GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384, EXP4384_PATH),
        (blocked, "ft09", "L2", GAP_E3_WORLD_MODEL_RULE_FT09_L2_4384, EXP4384_PATH),
    ]
    for rows, game, level, gap_id, artifact_path in gap_specs:
        row = rows.get(game, {})
        residual = str(row.get("residual_gap_class", ""))
        if not row or row.get("offline_reproduced") is True or residual in ("", "none"):
            continue
        entries[gap_id] = _gap_entry(
            gap_id,
            status="open",
            evidence=(
                f"{artifact_path}; game={game}; offline_reproduced=False; "
                f"prior_best_level={row.get('prior_best_level')}; "
                f"new_reproduced_level={row.get('new_reproduced_level')}; "
                f"target_level={row.get('target_level')}; verifier_accuracy="
                f"{row.get('verifier_accuracy')}; lookahead_fidelity="
                f"{row.get('lookahead_fidelity')}; residual={residual}"
            ),
            failure_mode=f"{game} {level} remains unreproduced due to {residual}",
            missing_discriminator=f"{game} executable world-model rule coverage for {residual}",
            candidate_design=(
                "mine divergent active traces for the named residual, add transition tests, "
                "and count progress only through the offline reproduce() gate"
            ),
        )
    return list(entries.values())


def _gap_entry_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4388 .405 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
    )


def _replace_or_append_gap(text: str, marker: str, gap: Mapping[str, Any]) -> str:
    if f"<!-- {marker}:start -->" in text or str(gap["gap_id"]) not in text:
        return base._replace_marked_block(text, marker, _gap_entry_block(gap))
    return text


def _domain_row(cross_domain: Mapping[str, Any], domain: str) -> dict[str, Any]:
    for row in cross_domain.get("detection_by_domain", []):
        if isinstance(row, Mapping) and row.get("domain") == domain:
            return dict(row)
    return {}


def _arc_new_levels(outcomes: Mapping[str, Any]) -> int:
    return int(outcomes["arc_e3"]["deeper_lookahead"].get("new_levels_reproduced") or 0) + int(
        outcomes["arc_e3"]["blocked_mechanics"].get("new_levels_reproduced") or 0
    )


def _arc_total(outcomes: Mapping[str, Any]) -> int:
    return max(
        int(outcomes["arc_e3"]["deeper_lookahead"].get("reproducible_total_levels") or 0),
        int(outcomes["arc_e3"]["blocked_mechanics"].get("reproducible_total_levels") or 0),
    )


def _ensure_gap4_eval(
    registry: dict[str, Any],
    guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    replay = guard.get("replayed_arc1_rule_exec", {})
    actionable = outcomes["actionable_detector"]
    compounds = outcomes["detector_self_learning"]
    cross = outcomes["cross_domain_detector"]
    gap4_arc = _domain_row(cross, "gap4_arc")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4388": EXP4388_ARTIFACT_PATH,
            "exp4388_gap4_regression_guard_passed": bool(
                guard.get("regression_guard_passed")
            ),
            "exp4388_arc1_rule_exec_vote_pass2": replay.get("vote_pass2"),
            "exp4388_arc1_rule_exec_gated_pass2": replay.get("gated_pass2"),
            "exp4388_arc1_headroom_recovered": replay.get("headroom_recovered"),
            "exp4388_arc1_vote_wins_lost": replay.get("vote_wins_lost"),
            "exp4388_v405_state": V405_STATE,
            "exp4388_detector_localization_actionable": actionable.get(
                "detector_localization_actionable"
            ),
            "exp4388_detector_compounds": compounds.get("detector_compounds"),
            "exp4388_compounding_delta_ci95": compounds.get("compounding_delta_ci95"),
            "exp4388_detector_generalizes_cross_domain": cross.get(
                "detector_generalizes_cross_domain"
            ),
            "exp4388_cross_domain_gap4_arc_auroc": gap4_arc.get("detection_auroc"),
            "exp4388_cross_domain_gap4_arc_ci95": gap4_arc.get("auroc_ci95"),
            "exp4388_cross_domain_gap4_arc_selection_headroom": gap4_arc.get(
                "selection_headroom"
            ),
            "exp4388_arc_reproducible_total_levels": _arc_total(outcomes),
            "exp4388_new_levels_reproduced": _arc_new_levels(outcomes),
            "exp4388_filled_gaps": [],
            "exp4388_gaps_logged": [gap["gap_id"] for gap in gap_entries],
        }
    )


def _ensure_v405_role(
    registry: dict[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V405_ROLE_ID,
        "experiment": EXP4388_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v405",
        "status": "v405_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "v405_state": V405_STATE,
        "detector_localization_actionable": outcomes["actionable_detector"].get(
            "detector_localization_actionable"
        ),
        "detector_compounds": outcomes["detector_self_learning"].get("detector_compounds"),
        "detector_generalizes_cross_domain": outcomes["cross_domain_detector"].get(
            "detector_generalizes_cross_domain"
        ),
        "arc_reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "gap_ids_logged": [gap["gap_id"] for gap in gap_entries],
        "filled_gap_ids": [],
        "eval_exp_4388": EXP4388_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V405_ROLE_ID
    ] + [role]


def _ensure_fover_detector(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    entry = base._find_verifier(registry, FOVER_VERIFIER_ID)
    if entry is None:
        entry = {
            "verifier_id": FOVER_VERIFIER_ID,
            "domain": "math_reasoning",
            "version": 4,
            "kind": "ensemble",
            "eval": {},
            "status": "active",
        }
        registry.setdefault("verifiers", []).append(entry)
    actionable = outcomes["actionable_detector"]
    compounds = outcomes["detector_self_learning"]
    cross = outcomes["cross_domain_detector"]
    gap4_arc = _domain_row(cross, "gap4_arc")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4381": EXP4381_PATH,
            "eval_exp_4385": EXP4385_PATH,
            "eval_exp_4386": EXP4386_PATH,
            "eval_exp_4388": EXP4388_ARTIFACT_PATH,
            "exp4388_detector_localization_actionable": actionable.get(
                "detector_localization_actionable"
            ),
            "exp4388_localization_delta_ci95": actionable.get("localization_delta_ci95"),
            "exp4388_n_error_traces": actionable.get("n_error_traces"),
            "exp4388_detector_compounds": compounds.get("detector_compounds"),
            "exp4388_positive_control_passed": compounds.get("positive_control_passed"),
            "exp4388_compounding_delta_ci95": compounds.get("compounding_delta_ci95"),
            "exp4388_final_held_out_localization_f1": compounds.get(
                "final_held_out_localization_f1"
            ),
            "exp4388_detector_generalizes_cross_domain": cross.get(
                "detector_generalizes_cross_domain"
            ),
            "exp4388_gap4_arc_detection_auroc": gap4_arc.get("detection_auroc"),
            "exp4388_gap4_arc_detection_ci95": gap4_arc.get("auroc_ci95"),
            "exp4388_domains_at_chance": cross.get("domains_at_chance"),
            "exp4388_verifier_is_oracle": cross.get("verifier_is_oracle"),
        }
    )


def _ensure_arc_registry(arc_registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    arc_registry["updated"] = "2026-06-18"
    arc_registry["reproducible_total_levels"] = max(
        int(arc_registry.get("reproducible_total_levels") or 0),
        _arc_total(outcomes),
    )
    arc_registry["latest_hygiene_4388"] = {
        "artifact": EXP4388_ARTIFACT_PATH,
        "reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "exp4383_new_levels_reproduced": outcomes["arc_e3"]["deeper_lookahead"].get(
            "new_levels_reproduced"
        ),
        "exp4384_new_levels_reproduced": outcomes["arc_e3"]["blocked_mechanics"].get(
            "new_levels_reproduced"
        ),
        "note": ".405 E3 passes did not add reproduced ARC levels; residual gaps remain open.",
    }


def registry_contains_v405(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    fover = base._find_verifier(registry, FOVER_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4388") == EXP4388_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4388_v405_state") == V405_STATE
        and any(role.get("role_id") == V405_ROLE_ID for role in gap4.get("registry_roles", []))
        and gap4.get("eval", {}).get("exp4388_detector_compounds") is True
        and gap4.get("eval", {}).get("exp4388_detector_generalizes_cross_domain") is True
        and fover
        and fover.get("eval", {}).get("eval_exp_4388") == EXP4388_ARTIFACT_PATH
        and fover.get("eval", {}).get("exp4388_detector_localization_actionable") is False
        and fover.get("eval", {}).get("exp4388_detector_compounds") is True
    )


def arc_registry_contains_v405(arc_registry: dict[str, Any]) -> bool:
    latest = arc_registry.get("latest_hygiene_4388", {})
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 34
        and isinstance(latest, Mapping)
        and latest.get("artifact") == EXP4388_ARTIFACT_PATH
        and latest.get("new_levels_reproduced") == 0
    )


def gaps_contain_v405(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return all(gap["gap_id"] in gaps_text for gap in gap_entries)


def ensure_ledgers_record_v405(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .405 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_v405_role(updated_registry, outcomes, gap_entries)
    _ensure_fover_detector(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)

    marker_by_gap_id = {gap["gap_id"]: f"exp4388-{gap['gap_id'].lower()}" for gap in gap_entries}
    for gap in gap_entries:
        gaps_text = _replace_or_append_gap(gaps_text, marker_by_gap_id[gap["gap_id"]], gap)

    registry_ok = registry_contains_v405(updated_registry)
    arc_ok = arc_registry_contains_v405(updated_arc)
    gaps_ok = gaps_contain_v405(gaps_text, gap_entries)
    return (
        updated_registry,
        gaps_text,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "registries_reconciled": registry_ok and arc_ok and gaps_ok,
            "filled_gap_ids": [],
            "gaps_logged_ids": [gap["gap_id"] for gap in gap_entries],
        },
    )


def _capstone_aggregation_propagates_oracle_stamp() -> bool:
    return (
        "verifier_is_oracle" in capstone_v404_4379.REQUIRED_ARTIFACT_FIELDS
        and "verifier_is_oracle" in capstone_v404_4379.FIELD_PRINCIPLES
    )


def _capstone_aggregation_uses_available_helper() -> bool:
    return (
        capstone_v404_4379.aggregate.aggregate_available_report_gaps
        is capstone_aggregate_available.aggregate_available_report_gaps
    )


def verify_capstone_stamp_fix_durable(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4388: re-run adversarial_verify.py on the .404 capstone."""
    capstone_path = repo_root / CAPSTONE_V404_PATH
    propagates = _capstone_aggregation_propagates_oracle_stamp()
    uses_helper = _capstone_aggregation_uses_available_helper()
    try:
        capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "capstone_stamp_fix_durable": False,
            "capstone_path": CAPSTONE_V404_PATH,
            "error": f"{type(exc).__name__}: {exc}",
            "capstone_verifier_is_oracle": None,
            "capstone_aggregation_propagates_oracle_stamp": propagates,
            "capstone_aggregation_uses_available_helper": uses_helper,
            "circular_moat_overclaim_fired": False,
            "flag_count": 0,
            "flags": [],
            "returncode": None,
        }
    command = [
        sys.executable,
        str(repo_root / "scripts" / "adversarial_verify.py"),
        "--json",
        str(capstone_path),
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"reports": [], "parse_error": completed.stdout[-500:]}
    flags = _flags_from_report(parsed)
    circular = [flag for flag in flags if flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM"]
    durable = (
        capstone.get("verifier_is_oracle") is False
        and capstone.get("verifier_is_oracle_honored") is True
        and propagates
        and uses_helper
        and not circular
        and not flags
        and completed.returncode == 0
    )
    return {
        "capstone_stamp_fix_durable": durable,
        "capstone_path": CAPSTONE_V404_PATH,
        "capstone_verifier_is_oracle": capstone.get("verifier_is_oracle"),
        "capstone_verifier_is_oracle_honored": capstone.get("verifier_is_oracle_honored"),
        "capstone_aggregation_propagates_oracle_stamp": propagates,
        "capstone_aggregation_uses_available_helper": uses_helper,
        "circular_moat_overclaim_fired": bool(circular),
        "flag_count": len(flags),
        "flags": flags,
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def model_specs() -> dict[str, Any]:
    return {
        "method": "cached_v405_ledger_reconciliation_gap4_guard_and_stamp_audit",
        "upstream_artifacts": [
            EXP4381_PATH,
            EXP4382_PATH,
            EXP4383_PATH,
            EXP4384_PATH,
            EXP4385_PATH,
            EXP4386_PATH,
            CAPSTONE_V404_PATH,
        ],
        "gap4_guard_source": "carnot.reporting.verifier_registry_gaps_hygiene_gap4_guard_4377",
        "capstone_stamp_source": CAPSTONE_V404_PATH,
        "codex_calls": 0,
        "live_model_inference": False,
        "gguf_inference": False,
        "gpu_inference": False,
    }


def build_artifact(
    *,
    preconditions_checked: dict[str, Any],
    gap4_regression_guard: dict[str, Any],
    capstone_stamp_fix: dict[str, Any],
    v405_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = bool(gap4_regression_guard.get("regression_guard_passed"))
    stamp_ok = bool(capstone_stamp_fix.get("capstone_stamp_fix_durable"))
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    complete = guard_ok and stamp_ok and reconciled
    artifact = {
        "experiment": "experiment_4388_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4388_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v405_truth_"
            f"gap4_guard_passed_{guard_ok}_capstone_stamp_fix_durable_{stamp_ok}"
            if complete
            else "blocked_v405_hygiene_incomplete"
        ),
        "gap4_regression_guard_passed": guard_ok,
        "capstone_stamp_fix_durable": stamp_ok,
        "registries_reconciled": reconciled,
        "preconditions_checked": preconditions_checked,
        "reproducibility_checksum": reproducibility_checksum,
        "v405_outcomes": v405_outcomes,
        "registry_reconciliation": registry_reconciliation,
        "gap4_regression_guard": gap4_regression_guard,
        "capstone_stamp_fix": capstone_stamp_fix,
        "random_seed": RANDOM_SEED,
        "model_specs": model_specs(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "arc_registry_path": ARC_REGISTRY_PATH,
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_file") or "registry")
    artifact = {
        "experiment": "experiment_4388_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4388_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": f"blocked_{blocked}_unreadable",
        "gap4_regression_guard_passed": False,
        "capstone_stamp_fix_durable": False,
        "registries_reconciled": False,
        "preconditions_checked": preflight,
        "reproducibility_checksum": f"blocked:{blocked}_unreadable",
        "v405_outcomes": {},
        "registry_reconciliation": {},
        "gap4_regression_guard": {},
        "capstone_stamp_fix": {},
        "random_seed": RANDOM_SEED,
        "model_specs": model_specs(),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4388 terminal artifact before writing."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in (
        "gap4_regression_guard_passed",
        "capstone_stamp_fix_durable",
        "registries_reconciled",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a BARE bool")
    for field in (
        "preconditions_checked",
        "v405_outcomes",
        "registry_reconciliation",
        "gap4_regression_guard",
        "capstone_stamp_fix",
    ):
        if not isinstance(artifact[field], dict):
            raise ValueError(f"{field} must be an object")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if isinstance(artifact["random_seed"], bool) or not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4388 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4388 and SCENARIO-VERIFY-4388")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def _patch_arc_registry_text(text: str, outcomes: Mapping[str, Any]) -> str:
    if "latest_hygiene_4388:" in text:
        return text
    block = (
        "latest_hygiene_4388:\n"
        f"  artifact: {EXP4388_ARTIFACT_PATH}\n"
        f"  reproducible_total_levels: {_arc_total(outcomes)}\n"
        f"  new_levels_reproduced: {_arc_new_levels(outcomes)}\n"
        "  exp4383_new_levels_reproduced: "
        f"{outcomes['arc_e3']['deeper_lookahead'].get('new_levels_reproduced')}\n"
        "  exp4384_new_levels_reproduced: "
        f"{outcomes['arc_e3']['blocked_mechanics'].get('new_levels_reproduced')}\n"
        '  note: ".405 E3 passes did not add reproduced ARC levels; residual gaps remain open."\n'
    )
    return text.rstrip() + "\n\n" + block


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4388 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix_durable
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4388_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    arc_path = repo_root / ARC_REGISTRY_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    arc_registry = yaml.safe_load(arc_path.read_text(encoding="utf-8"))
    if not isinstance(arc_registry, dict):  # pragma: no cover - preconditions gate this.
        arc_registry = {}

    guard = gap4_guard_runner(repo_root)
    stamp = capstone_stamp_runner(repo_root)
    outcomes = load_v405_outcomes(repo_root)
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v405(
        registry,
        gaps_text,
        arc_registry,
        guard,
        outcomes,
        gap_entries,
    )

    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    original_arc_text = arc_path.read_text(encoding="utf-8")
    patched_arc_text = _patch_arc_registry_text(original_arc_text, outcomes)
    if patched_arc_text != original_arc_text:
        arc_path.write_text(patched_arc_text, encoding="utf-8")
    elif not arc_registry_contains_v405(yaml.safe_load(original_arc_text) or {}):
        arc_path.write_text(yaml.safe_dump(arc_registry, sort_keys=False), encoding="utf-8")

    checksum = _json_hash(
        {
            "registry": registry,
            "gaps_text_sha256": hashlib.sha256(gaps_text.encode("utf-8")).hexdigest(),
            "arc_registry": arc_registry,
        }
    )
    artifact = build_artifact(
        preconditions_checked=preflight,
        gap4_regression_guard=guard,
        capstone_stamp_fix=stamp,
        v405_outcomes=outcomes,
        registry_reconciliation=summary,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised by results entrypoint tests.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4388_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
