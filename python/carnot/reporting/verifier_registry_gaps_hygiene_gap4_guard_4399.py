"""Exp 4399 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4399, SCENARIO-VERIFY-4399.
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
from carnot.reporting import capstone_v405_4390
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4388 as exp4388


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4399
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_capstone_stamp_audit"

EXP4399_ARTIFACT_PATH = "results/experiment_4399_registry_gaps_hygiene_gap4_guard.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
FOVER_VERIFIER_ID = exp4388.FOVER_VERIFIER_ID

CAPSTONE_V405_PATH = "results/experiment_4390_capstone_v405.json"
EXP4392_PATH = "results/experiment_4392_verifiable_process_data_localizer.json"
EXP4393_PATH = "results/experiment_4393_localizer_skeptic_proof.json"
EXP4394_PATH = "results/experiment_4394_e3_deeper_fidelity_gate.json"
EXP4395_PATH = "results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json"
EXP4396_PATH = "results/experiment_4396_localizer_self_learning_compounds.json"
EXP4397_PATH = "results/experiment_4397_cross_domain_detection_calibration.json"

V406_ROLE_ID = "oracle_distinct_v406_registry_gaps_hygiene_4399"
V406_STATE = (
    "localizer_quarantined__localizer_saturated_null__"
    "calibration_contract_false__arc_total_34_no_new_levels"
)

GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED = exp4388.GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED
GAP_4392_FIRST_ERROR_GAP4_ARC = "GAP-4392-FIRST-ERROR-GAP-4-ARC-arc_candidate_process_proxy"
GAP_4393_LOCALIZER_POSITION_TEMPLATE_CONFOUND = (
    "GAP-4393-LOCALIZER-POSITION-OR-TEMPLATE-CONFOUND"
)
GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383 = exp4388.GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383
GAP_E3_WORLD_MODEL_RULE_TU93_L5_4383 = exp4388.GAP_E3_WORLD_MODEL_RULE_TU93_L5_4383
GAP_E3_WORLD_MODEL_RULE_TN36_L8_4383 = exp4388.GAP_E3_WORLD_MODEL_RULE_TN36_L8_4383
GAP_E3_WORLD_MODEL_RULE_TR87_L7_4383 = exp4388.GAP_E3_WORLD_MODEL_RULE_TR87_L7_4383
GAP_E3_WORLD_MODEL_RULE_AR25_L2_4384 = exp4388.GAP_E3_WORLD_MODEL_RULE_AR25_L2_4384
GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384 = exp4388.GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384
GAP_E3_WORLD_MODEL_RULE_FT09_L2_4384 = exp4388.GAP_E3_WORLD_MODEL_RULE_FT09_L2_4384

SPEC_REFS = ["REQ-VERIFY-4399", "SCENARIO-VERIFY-4399"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_regression_guard_passed",
    "capstone_stamp_fix_durable",
    "registries_reconciled",
    "preconditions_checked",
    "reproducibility_checksum",
    "v406_outcomes",
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
        "BARE bool: verifier_registry.yaml + verifier_gaps.md (esp. "
        "GAP-FOVER-BIPRM-LOCALIZATION) + arc_solve_registry.yaml updated with "
        "the .406 outcomes (never-prune)."
    ),
    "preconditions_checked": (
        "Records the registry/gaps file readability; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
}

Gap4GuardRunner = Callable[[Path], dict[str, Any]]
CapstoneStampRunner = Callable[[Path], dict[str, Any]]

_json_hash = exp4388._json_hash
_load_optional_json = exp4388._load_optional_json
_bool = exp4388._bool
_int = exp4388._int
_float = exp4388._float
_str = exp4388._str
_list = exp4388._list
_flags_from_report = exp4388._flags_from_report
_scorecard_map = exp4388._scorecard_map


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4399: read all ledgers before mutating any of them."""
    return exp4388.check_preconditions(repo_root)


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4399: reuse the durable GAP-4 regression guard."""
    return exp4388.run_gap4_regression_guard(repo_root)


def _read_localizer(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4392_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4392_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "localizer_beats_ensemble_baseline": (
            _bool(payload, "localizer_beats_ensemble_baseline") is True
        ),
        "localization_f1_by_domain": dict(payload.get("localization_f1_by_domain", {})),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "n_traces": _int(payload, "n_traces"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_skeptic(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4393_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4393_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "localizer_win_is_genuine": _bool(payload, "localizer_win_is_genuine") is True,
        "beats_position_only_baseline": _bool(payload, "beats_position_only_baseline"),
        "template_ablation_drop": _float(payload, "template_ablation_drop"),
        "held_out_real_localization_delta_ci95": _list(
            payload, "held_out_real_localization_delta_ci95"
        ),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
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


def _read_compounds(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4396_PATH, "available": False, "error": error}
    learning_curve = _list(payload, "learning_curve")
    final = learning_curve[-1] if learning_curve and isinstance(learning_curve[-1], Mapping) else {}
    return {
        "artifact_path": EXP4396_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "localizer_compounds": _bool(payload, "localizer_compounds") is True,
        "learning_curve": learning_curve,
        "no_learning_baseline": _float(payload, "no_learning_baseline"),
        "positive_control_passed": _bool(payload, "positive_control_passed") is True,
        "compounding_delta_ci95": _list(payload, "compounding_delta_ci95"),
        "fallback_to_ensemble": _bool(payload, "fallback_to_ensemble") is True,
        "final_held_out_localization_f1": final.get("held_out_localization_f1"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_calibration(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4397_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4397_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "detection_calibrated_multi_domain": (
            _bool(payload, "detection_calibrated_multi_domain") is True
        ),
        "detection_by_domain": _list(payload, "detection_by_domain"),
        "domains_at_chance": _list(payload, "domains_at_chance"),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def load_v406_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4399: read .406 outcomes without fabricating missing artifacts."""
    localizer_payload, localizer_error = _load_optional_json(repo_root, EXP4392_PATH)
    skeptic_payload, skeptic_error = _load_optional_json(repo_root, EXP4393_PATH)
    deeper_payload, deeper_error = _load_optional_json(repo_root, EXP4394_PATH)
    blocked_payload, blocked_error = _load_optional_json(repo_root, EXP4395_PATH)
    compounds_payload, compounds_error = _load_optional_json(repo_root, EXP4396_PATH)
    calibration_payload, calibration_error = _load_optional_json(repo_root, EXP4397_PATH)
    return {
        "localizer": _read_localizer(localizer_payload, localizer_error),
        "localizer_skeptic_proof": _read_skeptic(skeptic_payload, skeptic_error),
        "arc_e3": {
            "deeper_fidelity": _read_e3_partial(
                deeper_payload,
                deeper_error,
                EXP4394_PATH,
                "per_target_scorecard",
                "residual_win_mechanic_gap_class",
            ),
            "blocked_mechanics": _read_e3_partial(
                blocked_payload,
                blocked_error,
                EXP4395_PATH,
                "per_game_scorecard",
                "residual_gap_class",
            ),
        },
        "localizer_self_learning": _read_compounds(compounds_payload, compounds_error),
        "cross_domain_calibration": _read_calibration(calibration_payload, calibration_error),
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


def _localizer_status(skeptic: Mapping[str, Any]) -> str:
    if skeptic.get("localizer_win_is_genuine") is True:
        return "filled (exp4392_first_error_localizer_after_4393_controls)"
    return "open"


def build_gap_entries(outcomes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-VERIFY-4399: collect .406 residual missing-verifier gaps."""
    entries: dict[str, dict[str, Any]] = {}
    localizer = outcomes["localizer"]
    skeptic = outcomes["localizer_skeptic_proof"]
    compounds = outcomes["localizer_self_learning"]
    status = _localizer_status(skeptic)
    entries[GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED] = _gap_entry(
        GAP_FOVER_BIPRM_LOCALIZATION_UNTYPED,
        status=status,
        evidence=(
            f"{EXP4392_PATH}; localizer_beats_ensemble_baseline="
            f"{localizer.get('localizer_beats_ensemble_baseline')}; {EXP4393_PATH}; "
            f"localizer_win_is_genuine={skeptic.get('localizer_win_is_genuine')}; "
            f"{EXP4396_PATH}; localizer_compounds={compounds.get('localizer_compounds')}; "
            f"compounding_delta_ci95={compounds.get('compounding_delta_ci95')}"
        ),
        failure_mode=(
            "Exp 4392 solved the original FoVer split, but Exp 4393 quarantined "
            "the A1 headline unless position/template controls pass."
            if status == "open"
            else "Exp 4393 controls graduated the synthetic first-error localizer."
        ),
        missing_discriminator=(
            "held-out real first-error labels with non-degenerate position/template variation"
        ),
        candidate_design=(
            "collect varied real first-error traces, type the residual error classes, "
            "and retrain with template-family holdouts before marking the gap filled"
        ),
        priority="high" if status == "open" else "medium",
    )
    for gap in localizer.get("missing_verifier_gaps", []):
        if not isinstance(gap, Mapping):
            continue
        gap_id = str(gap.get("gap_id", ""))
        if not gap_id:
            continue
        entries[gap_id] = _gap_entry(
            gap_id,
            status=str(gap.get("status", "open")),
            evidence=(
                f"{EXP4392_PATH}; domain={gap.get('domain')}; "
                f"error_class={gap.get('error_class')}; "
                f"missed_first_error_traces={gap.get('missed_first_error_traces')}"
            ),
            failure_mode=(
                "synthetic-trained earliest-error localizer still ranks a later "
                "inheritor or ARC proxy artifact ahead of the first break"
            ),
            missing_discriminator=str(gap.get("missing_discriminator", "")),
            candidate_design=str(gap.get("candidate_design", "")),
            priority=str(gap.get("priority", "medium")),
        )
    if skeptic.get("localizer_win_is_genuine") is not True:
        entries[GAP_4393_LOCALIZER_POSITION_TEMPLATE_CONFOUND] = _gap_entry(
            GAP_4393_LOCALIZER_POSITION_TEMPLATE_CONFOUND,
            status="open",
            evidence=(
                f"{EXP4393_PATH}; beats_position_only_baseline="
                f"{skeptic.get('beats_position_only_baseline')}; "
                f"template_ablation_drop={skeptic.get('template_ablation_drop')}; "
                f"delta_ci95={skeptic.get('held_out_real_localization_delta_ci95')}"
            ),
            failure_mode="position/template controls quarantine the A1 localizer headline",
            missing_discriminator="real held-out split with varied first-error positions",
            candidate_design=(
                "construct real first-error traces with template-family holdouts and "
                "require a positive A1-vs-position and A1-vs-ablation CI"
            ),
            priority="high",
        )
    calibration = outcomes["cross_domain_calibration"]
    for row in calibration.get("domains_at_chance", []):
        if not isinstance(row, Mapping):
            continue
        domain = str(row.get("domain", "unknown"))
        entries[f"GAP-DETECTOR-CROSS-DOMAIN-{domain}-4397"] = _gap_entry(
            f"GAP-DETECTOR-CROSS-DOMAIN-{domain}-4397",
            status="open",
            evidence=(
                f"{EXP4397_PATH}; domain={domain}; auroc={row.get('detection_auroc')}; "
                f"ci95={row.get('auroc_ci95')}; n={row.get('n')}"
            ),
            failure_mode=f"{domain} calibrated detector AUROC remains statistically at chance.",
            missing_discriminator=f"domain-specific correctness signal for {domain}",
            candidate_design=(
                "build a cached labeled pool with domain features and require "
                "CI95 lower bound above 0.5 before claiming calibration transfer"
            ),
            priority="medium",
        )
    deeper = outcomes["arc_e3"]["deeper_fidelity"]["rows"]
    blocked = outcomes["arc_e3"]["blocked_mechanics"]["rows"]
    gap_specs = [
        (deeper, "lp85", "L6", GAP_E3_WORLD_MODEL_RULE_LP85_L6_4383, EXP4394_PATH),
        (deeper, "tu93", "L5", GAP_E3_WORLD_MODEL_RULE_TU93_L5_4383, EXP4394_PATH),
        (deeper, "tn36", "L8", GAP_E3_WORLD_MODEL_RULE_TN36_L8_4383, EXP4394_PATH),
        (deeper, "tr87", "L7", GAP_E3_WORLD_MODEL_RULE_TR87_L7_4383, EXP4394_PATH),
        (blocked, "ar25", "L2", GAP_E3_WORLD_MODEL_RULE_AR25_L2_4384, EXP4395_PATH),
        (blocked, "ka59", "L2", GAP_E3_WORLD_MODEL_RULE_KA59_L2_4384, EXP4395_PATH),
        (blocked, "ft09", "L2", GAP_E3_WORLD_MODEL_RULE_FT09_L2_4384, EXP4395_PATH),
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
                f"target_level={row.get('target_level')}; "
                f"new_reproduced_level={row.get('new_reproduced_level')}; "
                f"verifier_accuracy={row.get('verifier_accuracy')}; "
                f"lookahead_fidelity={row.get('lookahead_fidelity')}; residual={residual}"
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
        f"### {gap['gap_id']}: Exp 4399 .406 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
    )


def _domain_row(cross_domain: Mapping[str, Any], domain: str) -> dict[str, Any]:
    for row in cross_domain.get("detection_by_domain", []):
        if isinstance(row, Mapping) and row.get("domain") == domain:
            return dict(row)
    return {}


def _arc_new_levels(outcomes: Mapping[str, Any]) -> int:
    return int(outcomes["arc_e3"]["deeper_fidelity"].get("new_levels_reproduced") or 0) + int(
        outcomes["arc_e3"]["blocked_mechanics"].get("new_levels_reproduced") or 0
    )


def _arc_total(outcomes: Mapping[str, Any]) -> int:
    return max(
        int(outcomes["arc_e3"]["deeper_fidelity"].get("reproducible_total_levels") or 0),
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
    localizer = outcomes["localizer"]
    skeptic = outcomes["localizer_skeptic_proof"]
    compounds = outcomes["localizer_self_learning"]
    calibration = outcomes["cross_domain_calibration"]
    gap4_arc = _domain_row(calibration, "gap4_arc")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4399": EXP4399_ARTIFACT_PATH,
            "exp4399_gap4_regression_guard_passed": bool(
                guard.get("regression_guard_passed")
            ),
            "exp4399_arc1_rule_exec_vote_pass2": replay.get("vote_pass2"),
            "exp4399_arc1_rule_exec_gated_pass2": replay.get("gated_pass2"),
            "exp4399_arc1_headroom_recovered": replay.get("headroom_recovered"),
            "exp4399_arc1_vote_wins_lost": replay.get("vote_wins_lost"),
            "exp4399_v406_state": V406_STATE,
            "exp4399_localizer_beats_ensemble_baseline": localizer.get(
                "localizer_beats_ensemble_baseline"
            ),
            "exp4399_localizer_win_is_genuine": skeptic.get("localizer_win_is_genuine"),
            "exp4399_localizer_compounds": compounds.get("localizer_compounds"),
            "exp4399_detection_calibrated_multi_domain": calibration.get(
                "detection_calibrated_multi_domain"
            ),
            "exp4399_cross_domain_gap4_arc_auroc": gap4_arc.get("detection_auroc"),
            "exp4399_cross_domain_gap4_arc_ci95": gap4_arc.get("auroc_ci95"),
            "exp4399_cross_domain_gap4_arc_selection_headroom": gap4_arc.get(
                "selection_headroom"
            ),
            "exp4399_arc_reproducible_total_levels": _arc_total(outcomes),
            "exp4399_new_levels_reproduced": _arc_new_levels(outcomes),
            "exp4399_filled_gaps": [
                gap["gap_id"]
                for gap in gap_entries
                if str(gap.get("status", "")).startswith("filled")
            ],
            "exp4399_gaps_logged": [gap["gap_id"] for gap in gap_entries],
        }
    )


def _ensure_v406_role(
    registry: dict[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    filled = [
        gap["gap_id"] for gap in gap_entries if str(gap.get("status", "")).startswith("filled")
    ]
    role = {
        "role_id": V406_ROLE_ID,
        "experiment": EXP4399_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v406",
        "status": "v406_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "v406_state": V406_STATE,
        "localizer_beats_ensemble_baseline": outcomes["localizer"].get(
            "localizer_beats_ensemble_baseline"
        ),
        "localizer_win_is_genuine": outcomes["localizer_skeptic_proof"].get(
            "localizer_win_is_genuine"
        ),
        "localizer_compounds": outcomes["localizer_self_learning"].get(
            "localizer_compounds"
        ),
        "detection_calibrated_multi_domain": outcomes["cross_domain_calibration"].get(
            "detection_calibrated_multi_domain"
        ),
        "arc_reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "gap_ids_logged": [gap["gap_id"] for gap in gap_entries],
        "filled_gap_ids": filled,
        "eval_exp_4399": EXP4399_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V406_ROLE_ID
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
    localizer = outcomes["localizer"]
    skeptic = outcomes["localizer_skeptic_proof"]
    compounds = outcomes["localizer_self_learning"]
    calibration = outcomes["cross_domain_calibration"]
    fover = _domain_row(calibration, "fover")
    gap4_arc = _domain_row(calibration, "gap4_arc")
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4392": EXP4392_PATH,
            "eval_exp_4393": EXP4393_PATH,
            "eval_exp_4396": EXP4396_PATH,
            "eval_exp_4397": EXP4397_PATH,
            "eval_exp_4399": EXP4399_ARTIFACT_PATH,
            "exp4399_localizer_beats_ensemble_baseline": localizer.get(
                "localizer_beats_ensemble_baseline"
            ),
            "exp4399_localization_f1_by_domain": localizer.get(
                "localization_f1_by_domain"
            ),
            "exp4399_localizer_win_is_genuine": skeptic.get("localizer_win_is_genuine"),
            "exp4399_beats_position_only_baseline": skeptic.get(
                "beats_position_only_baseline"
            ),
            "exp4399_template_ablation_drop": skeptic.get("template_ablation_drop"),
            "exp4399_localizer_compounds": compounds.get("localizer_compounds"),
            "exp4399_positive_control_passed": compounds.get("positive_control_passed"),
            "exp4399_compounding_delta_ci95": compounds.get("compounding_delta_ci95"),
            "exp4399_final_held_out_localization_f1": compounds.get(
                "final_held_out_localization_f1"
            ),
            "exp4399_detection_calibrated_multi_domain": calibration.get(
                "detection_calibrated_multi_domain"
            ),
            "exp4399_fover_detection_auroc": fover.get("detection_auroc"),
            "exp4399_fover_detection_ci95": fover.get("auroc_ci95"),
            "exp4399_gap4_arc_detection_auroc": gap4_arc.get("detection_auroc"),
            "exp4399_gap4_arc_detection_ci95": gap4_arc.get("auroc_ci95"),
            "exp4399_domains_at_chance": calibration.get("domains_at_chance"),
            "exp4399_verifier_is_oracle": calibration.get("verifier_is_oracle"),
        }
    )


def _ensure_arc_registry(arc_registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    arc_registry["updated"] = "2026-06-18"
    arc_registry["reproducible_total_levels"] = max(
        int(arc_registry.get("reproducible_total_levels") or 0),
        _arc_total(outcomes),
    )
    arc_registry["latest_hygiene_4399"] = {
        "artifact": EXP4399_ARTIFACT_PATH,
        "reproducible_total_levels": _arc_total(outcomes),
        "new_levels_reproduced": _arc_new_levels(outcomes),
        "exp4394_new_levels_reproduced": outcomes["arc_e3"]["deeper_fidelity"].get(
            "new_levels_reproduced"
        ),
        "exp4395_new_levels_reproduced": outcomes["arc_e3"]["blocked_mechanics"].get(
            "new_levels_reproduced"
        ),
        "note": ".406 E3 passes did not add reproduced ARC levels; residual gaps remain open.",
    }


def registry_contains_v406(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    fover = base._find_verifier(registry, FOVER_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4399") == EXP4399_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4399_v406_state") == V406_STATE
        and any(role.get("role_id") == V406_ROLE_ID for role in gap4.get("registry_roles", []))
        and gap4.get("eval", {}).get("exp4399_localizer_beats_ensemble_baseline") is True
        and gap4.get("eval", {}).get("exp4399_localizer_win_is_genuine") is False
        and fover
        and fover.get("eval", {}).get("eval_exp_4399") == EXP4399_ARTIFACT_PATH
        and fover.get("eval", {}).get("exp4399_localizer_beats_ensemble_baseline") is True
        and fover.get("eval", {}).get("exp4399_localizer_win_is_genuine") is False
    )


def arc_registry_contains_v406(arc_registry: dict[str, Any]) -> bool:
    latest = arc_registry.get("latest_hygiene_4399", {})
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 34
        and isinstance(latest, Mapping)
        and latest.get("artifact") == EXP4399_ARTIFACT_PATH
        and latest.get("new_levels_reproduced") == 0
    )


def gaps_contain_v406(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return all(f"<!-- exp4399-{gap['gap_id'].lower()}:start -->" in gaps_text for gap in gap_entries)


def ensure_ledgers_record_v406(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .406 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_v406_role(updated_registry, outcomes, gap_entries)
    _ensure_fover_detector(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)

    for gap in gap_entries:
        marker = f"exp4399-{gap['gap_id'].lower()}"
        gaps_text = base._replace_marked_block(gaps_text, marker, _gap_entry_block(gap))

    registry_ok = registry_contains_v406(updated_registry)
    arc_ok = arc_registry_contains_v406(updated_arc)
    gaps_ok = gaps_contain_v406(gaps_text, gap_entries)
    filled = [gap["gap_id"] for gap in gap_entries if str(gap.get("status", "")).startswith("filled")]
    return (
        updated_registry,
        gaps_text,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "registries_reconciled": registry_ok and arc_ok and gaps_ok,
            "filled_gap_ids": filled,
            "gaps_logged_ids": [gap["gap_id"] for gap in gap_entries],
        },
    )


def _capstone_aggregation_propagates_oracle_stamp() -> bool:
    return (
        "verifier_is_oracle" in capstone_v405_4390.REQUIRED_ARTIFACT_FIELDS
        and "verifier_is_oracle" in capstone_v405_4390.FIELD_PRINCIPLES
    )


def _capstone_aggregation_uses_available_helper() -> bool:
    return (
        capstone_v405_4390.aggregate.aggregate_available_report_gaps
        is capstone_aggregate_available.aggregate_available_report_gaps
    )


def verify_capstone_stamp_fix_durable(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4399: scan the .405 capstone and inspect the capstone helper."""
    capstone_path = repo_root / CAPSTONE_V405_PATH
    propagates = _capstone_aggregation_propagates_oracle_stamp()
    uses_helper = _capstone_aggregation_uses_available_helper()
    try:
        capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "capstone_stamp_fix_durable": False,
            "capstone_path": CAPSTONE_V405_PATH,
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
        "capstone_path": CAPSTONE_V405_PATH,
        "capstone_verifier_is_oracle": capstone.get("verifier_is_oracle"),
        "capstone_verifier_is_oracle_honored": capstone.get("verifier_is_oracle_honored"),
        "capstone_aggregation_propagates_oracle_stamp": propagates,
        "capstone_aggregation_uses_available_helper": uses_helper,
        "capstone_aggregation_source": "carnot.reporting.capstone_v405_4390",
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
        "method": "cached_v406_ledger_reconciliation_gap4_guard_and_stamp_audit",
        "upstream_artifacts": [
            EXP4392_PATH,
            EXP4393_PATH,
            EXP4394_PATH,
            EXP4395_PATH,
            EXP4396_PATH,
            EXP4397_PATH,
            CAPSTONE_V405_PATH,
        ],
        "gap4_guard_source": "carnot.reporting.verifier_registry_gaps_hygiene_gap4_guard_4388",
        "capstone_stamp_source": CAPSTONE_V405_PATH,
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
    v406_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = bool(gap4_regression_guard.get("regression_guard_passed"))
    stamp_ok = bool(capstone_stamp_fix.get("capstone_stamp_fix_durable"))
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    complete = guard_ok and stamp_ok and reconciled
    artifact = {
        "experiment": "experiment_4399_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4399_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v406_truth_"
            f"gap4_guard_passed_{guard_ok}_capstone_stamp_fix_durable_{stamp_ok}"
            if complete
            else "blocked_v406_hygiene_incomplete"
        ),
        "gap4_regression_guard_passed": guard_ok,
        "capstone_stamp_fix_durable": stamp_ok,
        "registries_reconciled": reconciled,
        "preconditions_checked": preconditions_checked,
        "reproducibility_checksum": reproducibility_checksum,
        "v406_outcomes": v406_outcomes,
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
        "experiment": "experiment_4399_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4399_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": f"blocked_{blocked}_unreadable",
        "gap4_regression_guard_passed": False,
        "capstone_stamp_fix_durable": False,
        "registries_reconciled": False,
        "preconditions_checked": preflight,
        "reproducibility_checksum": f"blocked:{blocked}_unreadable",
        "v406_outcomes": {},
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
    """Validate the Exp 4399 terminal artifact before writing."""
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
        "v406_outcomes",
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
        raise ValueError("field_principles must match the required Exp 4399 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4399 and SCENARIO-VERIFY-4399")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def _patch_arc_registry_text(text: str, outcomes: Mapping[str, Any]) -> str:
    if "latest_hygiene_4399:" in text:
        return text
    block = (
        "latest_hygiene_4399:\n"
        f"  artifact: {EXP4399_ARTIFACT_PATH}\n"
        f"  reproducible_total_levels: {_arc_total(outcomes)}\n"
        f"  new_levels_reproduced: {_arc_new_levels(outcomes)}\n"
        "  exp4394_new_levels_reproduced: "
        f"{outcomes['arc_e3']['deeper_fidelity'].get('new_levels_reproduced')}\n"
        "  exp4395_new_levels_reproduced: "
        f"{outcomes['arc_e3']['blocked_mechanics'].get('new_levels_reproduced')}\n"
        '  note: ".406 E3 passes did not add reproduced ARC levels; residual gaps remain open."\n'
    )
    return text.rstrip() + "\n\n" + block


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4399 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix_durable
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4399_ARTIFACT_PATH
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
    outcomes = load_v406_outcomes(repo_root)
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v406(
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
    elif not arc_registry_contains_v406(yaml.safe_load(original_arc_text) or {}):
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
        v406_outcomes=outcomes,
        registry_reconciliation=summary,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:
    artifact = run_hygiene(REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
