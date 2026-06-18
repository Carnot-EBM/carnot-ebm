"""Exp 4377 registry/gaps hygiene, GAP-4 guard, and capstone stamp durability.

Spec refs: REQ-VERIFY-4377, SCENARIO-VERIFY-4377.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_aggregate_available
from carnot.reporting import capstone_v403_4368
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4366 as exp4366


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4377
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_capstone_stamp_audit"

EXP4377_ARTIFACT_PATH = "results/experiment_4377_registry_gaps_hygiene_gap4_guard.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ACTION_COST_VERIFIER_ID = exp4366.ACTION_COST_VERIFIER_ID
FOVER_VERIFIER_ID = "fover_production_ensemble"

EXP4366_PATH = exp4366.EXP4366_ARTIFACT_PATH
CAPSTONE_V403_PATH = "results/experiment_4368_capstone_v403.json"
EXP4370_PATH = "results/experiment_4370_llm_generated_action_cost_heuristics.json"
EXP4371_PATH = "results/experiment_4371_llm_heuristic_contamination_skeptic_proof.json"
EXP4372_PATH = "results/experiment_4372_e3_deeper_high_headroom_games.json"
EXP4373_PATH = "results/experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json"
EXP4374_PATH = "results/experiment_4374_diffusiongemma_scorer_repair_or_retire.json"
EXP4375_PATH = "results/experiment_4375_verifier_as_detector_measurement.json"

V404_ROLE_ID = "oracle_distinct_v404_registry_gaps_hygiene_4377"
V404_STATE = (
    "llm_heuristic_clean_null__arc_total_34_lp85_l5_reproduced__"
    "diffusiongemma_retired__fover_detector_positive"
)

GAP_4370 = "GAP-4370"
GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361 = exp4366.GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361
GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361 = exp4366.GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361
GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361 = exp4366.GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361
GAP_E3_WORLD_MODEL_RULE_AR25_L2_4362 = exp4366.GAP_E3_WORLD_MODEL_RULE_AR25_L2_4362
GAP_E3_WORLD_MODEL_RULE_KA59_L2_4362 = exp4366.GAP_E3_WORLD_MODEL_RULE_KA59_L2_4362
GAP_E3_WORLD_MODEL_RULE_FT09_L2_4373 = "GAP-E3-WORLD-MODEL-RULE-FT09-L2-4373"

SPEC_REFS = ["REQ-VERIFY-4377", "SCENARIO-VERIFY-4377"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_regression_guard_passed",
    "capstone_stamp_fix_durable",
    "registries_reconciled",
    "preconditions_checked",
    "reproducibility_checksum",
    "v404_outcomes",
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
        "arc_solve_registry.yaml updated with the .404 outcomes (never-prune)."
    ),
    "preconditions_checked": (
        "Records the registry/gaps file readability; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
}

Gap4GuardRunner = Callable[[Path], dict[str, Any]]
CapstoneStampRunner = Callable[[Path], dict[str, Any]]


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_optional_json(repo_root: Path, rel_path: str) -> tuple[dict[str, Any] | None, str]:
    path = repo_root / rel_path
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(loaded, dict):
        return None, "top-level JSON is not an object"
    return loaded, ""


def _bool(payload: Mapping[str, Any] | None, key: str) -> bool | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _int(payload: Mapping[str, Any] | None, key: str) -> int:
    if not isinstance(payload, Mapping):
        return 0
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _float(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _str(payload: Mapping[str, Any] | None, key: str) -> str:
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _list(payload: Mapping[str, Any] | None, key: str) -> list[Any]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    return list(value) if isinstance(value, list) else []


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
            "reproduced_levels": row.get("reproduced_levels"),
            "prior_best_level": row.get("prior_best_level"),
            "residual_gap_class": str(row.get(residual_field, "")),
            "verifier_accuracy": row.get("verifier_accuracy"),
            "world_model_path": str(row.get("world_model_path", "")),
            "world_model_sha256": str(row.get("world_model_sha256", "")),
            "mechanic_checks_passed": row.get("mechanic_checks_passed"),
            "plan_action_count": row.get("plan_action_count"),
        }
    return mapped


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4377: read all ledgers before mutating any of them."""
    checks: dict[str, dict[str, Any]] = {}
    blocked_file: str | None = None

    for key, rel_path in (
        ("verifier_registry", REGISTRY_PATH),
        ("arc_solve_registry", ARC_REGISTRY_PATH),
    ):
        path = repo_root / rel_path
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            ok = isinstance(loaded, dict)
            error = "" if ok else "top-level YAML is not a mapping"
        except (OSError, yaml.YAMLError) as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        checks[key] = {"path": rel_path, "readable": ok, "error": error}
        if not ok and blocked_file is None:
            blocked_file = key

    path = repo_root / GAPS_PATH
    try:
        path.read_text(encoding="utf-8")
        ok = True
        error = ""
    except OSError as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    checks["verifier_gaps"] = {"path": GAPS_PATH, "readable": ok, "error": error}
    if not ok and blocked_file is None:
        blocked_file = "verifier_gaps"

    return {"ok": blocked_file is None, "blocked_file": blocked_file, "files": checks}


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4377: reuse the durable GAP-4 regression guard."""
    return exp4366.run_gap4_regression_guard(repo_root)


def _read_llm_heuristic(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4370_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4370_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "acceptance_gate_passed": _bool(payload, "acceptance_gate_passed") is True,
        "llm_heuristic_beats_linear": _bool(payload, "llm_heuristic_beats_linear") is True,
        "static_leakage_clean": _bool(payload, "static_leakage_clean") is True,
        "reproduction_gated": _bool(payload, "reproduction_gated") is True,
        "n_held_out_levels": _int(payload, "n_held_out_levels"),
        "held_out_actions_by_heuristic": dict(payload.get("held_out_actions_by_heuristic", {})),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_skeptic(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4371_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4371_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "status": _str(payload, "status"),
        "gate_check_summary": _str(payload, "gate_check_summary"),
        "gates_evaluated": _list(payload, "gates_evaluated"),
    }


def _read_deeper(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4372_PATH, "available": False, "error": error, "targets": {}}
    return {
        "artifact_path": EXP4372_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
        "targets": _scorecard_map(
            _list(payload, "per_target_scorecard"),
            "residual_win_mechanic_gap_class",
        ),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_blocked_mechanics(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4373_PATH, "available": False, "error": error, "games": {}}
    return {
        "artifact_path": EXP4373_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
        "games": _scorecard_map(_list(payload, "per_game_scorecard"), "residual_gap_class"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_diffusiongemma(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4374_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4374_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "acceptance_gate": _bool(payload, "acceptance_gate"),
        "s3_guided_beats_control": _bool(payload, "s3_guided_beats_control"),
        "controls_differentiated": _bool(payload, "controls_differentiated"),
        "codila_control_differentiates": _bool(payload, "codila_control_differentiates"),
        "scorer_requalified_leak_clean": _bool(payload, "scorer_requalified_leak_clean"),
        "s3_minus_best_of_n_delta": _float(payload, "s3_minus_best_of_n_delta"),
        "s3_gain_ci95": _list(payload, "s3_gain_ci95"),
        "benchmark_n": _int(payload, "benchmark_n"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_detector(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4375_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4375_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "detector_beats_chance": _bool(payload, "detector_beats_chance") is True,
        "detector_auroc": _float(payload, "detector_auroc"),
        "detector_auroc_ci95": _list(payload, "detector_auroc_ci95"),
        "n_candidates": _int(payload, "n_candidates"),
        "selection_headroom": dict(payload.get("selection_headroom", {})),
        "per_verifier_auroc": dict(payload.get("per_verifier_auroc", {})),
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def load_v404_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4377: read .404 outcomes without fabricating missing artifacts."""
    llm_payload, llm_error = _load_optional_json(repo_root, EXP4370_PATH)
    skeptic_payload, skeptic_error = _load_optional_json(repo_root, EXP4371_PATH)
    deeper_payload, deeper_error = _load_optional_json(repo_root, EXP4372_PATH)
    blocked_payload, blocked_error = _load_optional_json(repo_root, EXP4373_PATH)
    diffusion_payload, diffusion_error = _load_optional_json(repo_root, EXP4374_PATH)
    detector_payload, detector_error = _load_optional_json(repo_root, EXP4375_PATH)
    return {
        "llm_generated_action_cost": _read_llm_heuristic(llm_payload, llm_error),
        "llm_heuristic_skeptic_proof": _read_skeptic(skeptic_payload, skeptic_error),
        "arc_e3": {
            "deeper_high_headroom": _read_deeper(deeper_payload, deeper_error),
            "blocked_mechanics": _read_blocked_mechanics(blocked_payload, blocked_error),
        },
        "diffusiongemma_repair_or_retire": _read_diffusiongemma(
            diffusion_payload,
            diffusion_error,
        ),
        "verifier_as_detector": _read_detector(detector_payload, detector_error),
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
    """REQ-VERIFY-4377: collect .404 residual missing-verifier gaps."""
    entries: dict[str, dict[str, Any]] = {}
    llm = outcomes["llm_generated_action_cost"]
    for gap in llm.get("missing_verifier_gaps", []):
        if not isinstance(gap, Mapping):
            continue
        gap_id = str(gap.get("gap_id", ""))
        if gap_id:
            entries[gap_id] = _gap_entry(
                gap_id,
                status="open",
                evidence=(
                    f"{EXP4370_PATH}; llm_heuristic_beats_linear="
                    f"{llm.get('llm_heuristic_beats_linear')}; "
                    f"held_out_actions_by_heuristic={llm.get('held_out_actions_by_heuristic')}"
                ),
                failure_mode=str(gap.get("failure_mode", "")),
                missing_discriminator=str(gap.get("missing_discriminator", "")),
                candidate_design=str(gap.get("candidate_design", "")),
                priority=str(gap.get("priority", "medium")),
            )

    deeper = outcomes["arc_e3"]["deeper_high_headroom"]["targets"]
    blocked = outcomes["arc_e3"]["blocked_mechanics"]["games"]
    gap_specs = [
        (deeper, "sc25", GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361, EXP4372_PATH, "L2"),
        (deeper, "tn36", GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361, EXP4372_PATH, "L8"),
        (blocked, "ar25", GAP_E3_WORLD_MODEL_RULE_AR25_L2_4362, EXP4373_PATH, "L2"),
        (blocked, "ka59", GAP_E3_WORLD_MODEL_RULE_KA59_L2_4362, EXP4373_PATH, "L2"),
        (blocked, "ft09", GAP_E3_WORLD_MODEL_RULE_FT09_L2_4373, EXP4373_PATH, "L2"),
    ]
    for rows, game, gap_id, artifact_path, level in gap_specs:
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
                f"verifier_accuracy={row.get('verifier_accuracy')}; residual={residual}"
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
        f"### {gap['gap_id']}: Exp 4377 .404 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
    )


def _filled_lp85_gap_block(row: Mapping[str, Any]) -> str:
    return (
        f"### {GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361}: Exp 4377 .404 filled verifier gap update\n"
        "- status: filled (exp4372_lp85_l5_world_model)\n"
        f"- evidence: {EXP4372_PATH}; game=lp85; offline_reproduced="
        f"{row.get('offline_reproduced')}; new_reproduced_level="
        f"{row.get('new_reproduced_level')}; verifier_accuracy="
        f"{row.get('verifier_accuracy')}; world_model_path={row.get('world_model_path')}.\n"
        "- failure mode: the prior lp85 L5 reset-replay blocker no longer prevents "
        "an offline reproduced L5 gate.\n"
        "- missing discriminator: none for the reproduced L5 plan; deeper future "
        "mechanics remain separate gaps if exposed.\n"
        "- candidate design: preserve the Exp 4372 reproduce() gate for lp85 L5.\n"
        "- priority: high\n"
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
    llm = outcomes["llm_generated_action_cost"]
    deeper = outcomes["arc_e3"]["deeper_high_headroom"]
    blocked = outcomes["arc_e3"]["blocked_mechanics"]
    diffusion = outcomes["diffusiongemma_repair_or_retire"]
    detector = outcomes["verifier_as_detector"]
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4377": EXP4377_ARTIFACT_PATH,
            "exp4377_gap4_regression_guard_passed": bool(
                guard.get("regression_guard_passed")
            ),
            "exp4377_arc1_rule_exec_vote_pass2": replay.get("vote_pass2"),
            "exp4377_arc1_rule_exec_gated_pass2": replay.get("gated_pass2"),
            "exp4377_arc1_headroom_recovered": replay.get("headroom_recovered"),
            "exp4377_arc1_vote_wins_lost": replay.get("vote_wins_lost"),
            "exp4377_v404_state": V404_STATE,
            "exp4377_llm_heuristic_beats_linear": llm.get("llm_heuristic_beats_linear"),
            "exp4377_arc_reproducible_total_levels": deeper.get(
                "reproducible_total_levels"
            ),
            "exp4377_lp85_reproduced_level": deeper["targets"].get("lp85", {}).get(
                "new_reproduced_level"
            ),
            "exp4377_blocked_mechanics_new_levels": blocked.get("new_levels_reproduced"),
            "exp4377_diffusiongemma_status": diffusion.get("honest_verdict"),
            "exp4377_diffusiongemma_guided_beats_control": diffusion.get(
                "s3_guided_beats_control"
            ),
            "exp4377_detector_beats_chance": detector.get("detector_beats_chance"),
            "exp4377_detector_auroc": detector.get("detector_auroc"),
            "exp4377_detector_n_candidates": detector.get("n_candidates"),
            "exp4377_filled_gaps": [GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361],
            "exp4377_gaps_logged": [gap["gap_id"] for gap in gap_entries],
        }
    )


def _ensure_v404_role(
    registry: dict[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V404_ROLE_ID,
        "experiment": EXP4377_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v404",
        "status": "v404_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "v404_state": V404_STATE,
        "llm_heuristic_beats_linear": outcomes["llm_generated_action_cost"].get(
            "llm_heuristic_beats_linear"
        ),
        "arc_reproducible_total_levels": outcomes["arc_e3"][
            "deeper_high_headroom"
        ].get("reproducible_total_levels"),
        "diffusiongemma_status": outcomes["diffusiongemma_repair_or_retire"].get(
            "honest_verdict"
        ),
        "detector_beats_chance": outcomes["verifier_as_detector"].get(
            "detector_beats_chance"
        ),
        "gap_ids_logged": [gap["gap_id"] for gap in gap_entries],
        "filled_gap_ids": [GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361],
        "eval_exp_4377": EXP4377_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V404_ROLE_ID
    ] + [role]


def _ensure_action_cost_verifier(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    llm = outcomes["llm_generated_action_cost"]
    entry = base._find_verifier(registry, ACTION_COST_VERIFIER_ID)
    if entry is None:
        entry = {
            "verifier_id": ACTION_COST_VERIFIER_ID,
            "domain": "arc_agi3_interactive",
            "version": 1,
            "kind": "search_heuristic",
            "code_commit": "HEAD",
            "code_path": "python/carnot/experiment_4353_learned_action_cost_heuristic_efficiency.py",
            "label_source": "offline_reproduced_solve_traces",
            "eval": {},
            "status": "candidate",
            "notes": "Oracle-distinct learned action-cost heuristic.",
        }
        registry.setdefault("verifiers", []).append(entry)
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4370": EXP4370_PATH,
            "eval_exp_4377": EXP4377_ARTIFACT_PATH,
            "exp4377_llm_heuristic_beats_linear": llm.get("llm_heuristic_beats_linear"),
            "exp4377_static_leakage_clean": llm.get("static_leakage_clean"),
            "exp4377_reproduction_gated": llm.get("reproduction_gated"),
            "exp4377_n_held_out_levels": llm.get("n_held_out_levels"),
            "exp4377_held_out_actions_by_heuristic": llm.get(
                "held_out_actions_by_heuristic"
            ),
            "exp4377_verifier_is_oracle": llm.get("verifier_is_oracle"),
            "exp4377_gap_id": GAP_4370,
        }
    )


def _ensure_fover_detector(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    detector = outcomes["verifier_as_detector"]
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
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4375": EXP4375_PATH,
            "eval_exp_4377": EXP4377_ARTIFACT_PATH,
            "exp4377_detector_beats_chance": detector.get("detector_beats_chance"),
            "exp4377_detector_auroc": detector.get("detector_auroc"),
            "exp4377_detector_auroc_ci95": detector.get("detector_auroc_ci95"),
            "exp4377_detector_n_candidates": detector.get("n_candidates"),
            "exp4377_selection_headroom": detector.get("selection_headroom"),
            "exp4377_per_verifier_auroc": detector.get("per_verifier_auroc"),
            "exp4377_verifier_is_oracle": detector.get("verifier_is_oracle"),
        }
    )


def _find_game(arc_registry: dict[str, Any], game: str) -> dict[str, Any] | None:
    for row in arc_registry.get("games", []):
        if isinstance(row, dict) and row.get("game") == game:
            return row
    return None


def _ensure_game(arc_registry: dict[str, Any], game: str) -> dict[str, Any]:
    row = _find_game(arc_registry, game)
    if row is None:
        row = {"game": game}
        arc_registry.setdefault("games", []).append(row)
    return row


def _ensure_arc_registry(arc_registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    deeper = outcomes["arc_e3"]["deeper_high_headroom"]
    total = deeper.get("reproducible_total_levels")
    if isinstance(total, int) and not isinstance(total, bool):
        arc_registry["reproducible_total_levels"] = max(
            int(arc_registry.get("reproducible_total_levels") or 0),
            total,
        )
    arc_registry["reproducible_total_games"] = max(
        int(arc_registry.get("reproducible_total_games") or 0),
        17,
    )
    arc_registry["updated"] = "2026-06-18"
    lp85 = deeper["targets"].get("lp85", {})
    if lp85.get("offline_reproduced") is True:
        row = _ensure_game(arc_registry, "lp85")
        row.update(
            {
                "reproducibility": "reproduced",
                "levels_reproduced": max(
                    int(row.get("levels_reproduced") or 0),
                    int(lp85.get("new_reproduced_level") or 0),
                ),
                "solver": f"{EXP4372_PATH} + {lp85.get('world_model_path')}",
                "reproduce": (
                    "Exp4372 per_target_scorecard lp85 offline_reproduced=True, "
                    "new_reproduced_level=5; reproduction-gated."
                ),
            }
        )


def _patch_arc_registry_text(text: str, outcomes: Mapping[str, Any]) -> str:
    """Update the comment-rich ARC registry text without YAML reformatting it."""
    deeper = outcomes["arc_e3"]["deeper_high_headroom"]
    lp85 = deeper["targets"].get("lp85", {})
    updated = text.replace('updated: "2026-06-17"', 'updated: "2026-06-18"', 1)
    total = deeper.get("reproducible_total_levels")
    if isinstance(total, int) and not isinstance(total, bool):
        updated = re.sub(
            r"reproducible_total_levels: 33\s+#.*",
            (
                "reproducible_total_levels: 34   # ... -> +tu93 L4 via Exp4361, "
                "+ft09 L1 via Exp4363, and +lp85 L5 via Exp4372"
            ),
            updated,
            count=1,
        )
        updated = updated.replace("lp85 4 +", "lp85 5 +", 1)
        updated = updated.replace("= 33 across 17 games", "= 34 across 17 games", 1)

    if lp85.get("offline_reproduced") is not True:
        return updated

    def replace_lp85_block(match: re.Match[str]) -> str:
        block = match.group(1)
        suffix = match.group(2)
        block = re.sub(
            r"    levels_reproduced: \d+.*",
            "    levels_reproduced: 5   # 4 -> 5 (2026-06-18: Exp4372 offline_reproduced=True, reproduction-gated)",
            block,
            count=1,
        )
        block = re.sub(
            r'    solver: ".*"\n',
            (
                '    solver: "scripts/arc3_lp85_offline_solver.py + arc_loop_solve '
                'adaptered path; Exp4372 extends the offline reproduced chain through '
                'L5 via python/carnot/agentic/arc_game_adapters.py."\n'
            ),
            block,
            count=1,
        )
        block = re.sub(
            r'    reproduce: ".*"\n',
            (
                '    reproduce: "Exp4372 per_target_scorecard lp85 '
                'offline_reproduced=True, new_reproduced_level=5; reproduction-gated. '
                'Prior results/arc_loop_solve_lp85.json remains the L4 gate."\n'
            ),
            block,
            count=1,
        )
        return block + suffix

    return re.sub(
        r"(?ms)(  - game: lp85\n.*?)(\n\n  - game: sc25)",
        replace_lp85_block,
        updated,
        count=1,
    )


def registry_contains_v404(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    action = base._find_verifier(registry, ACTION_COST_VERIFIER_ID)
    fover = base._find_verifier(registry, FOVER_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4377") == EXP4377_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4377_v404_state") == V404_STATE
        and any(role.get("role_id") == V404_ROLE_ID for role in gap4.get("registry_roles", []))
        and action
        and action.get("eval", {}).get("eval_exp_4377") == EXP4377_ARTIFACT_PATH
        and action.get("eval", {}).get("exp4377_llm_heuristic_beats_linear") is False
        and fover
        and fover.get("eval", {}).get("eval_exp_4377") == EXP4377_ARTIFACT_PATH
        and fover.get("eval", {}).get("exp4377_detector_beats_chance") is True
    )


def arc_registry_contains_v404(arc_registry: dict[str, Any]) -> bool:
    lp85 = _find_game(arc_registry, "lp85") or {}
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 34
        and int(arc_registry.get("reproducible_total_games") or 0) >= 17
        and lp85.get("reproducibility") == "reproduced"
        and int(lp85.get("levels_reproduced") or 0) >= 5
    )


def gaps_contain_v404(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return (
        "status: filled (exp4372_lp85_l5_world_model)" in gaps_text
        and GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361 in gaps_text
        and all(gap["gap_id"] in gaps_text for gap in gap_entries)
    )


def _replace_or_append_gap(text: str, marker: str, gap: Mapping[str, Any]) -> str:
    if f"<!-- {marker}:start -->" in text or str(gap["gap_id"]) not in text:
        return base._replace_marked_block(text, marker, _gap_entry_block(gap))
    return text


def ensure_ledgers_record_v404(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .404 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_v404_role(updated_registry, outcomes, gap_entries)
    _ensure_action_cost_verifier(updated_registry, outcomes)
    _ensure_fover_detector(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)

    lp85 = outcomes["arc_e3"]["deeper_high_headroom"]["targets"].get("lp85", {})
    if lp85.get("offline_reproduced") is True:
        gaps_text = base._replace_marked_block(
            gaps_text,
            "exp4366-gap-e3-world-model-rule-lp85-l5-4361",
            _filled_lp85_gap_block(lp85),
        )

    marker_by_gap_id = {
        GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361: "exp4366-gap-e3-world-model-rule-sc25-l2-4361",
        GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361: "exp4366-gap-e3-world-model-rule-tn36-l8-4361",
        GAP_E3_WORLD_MODEL_RULE_AR25_L2_4362: "exp4366-gap-e3-world-model-rule-ar25-l2-4362",
        GAP_E3_WORLD_MODEL_RULE_KA59_L2_4362: "exp4366-gap-e3-world-model-rule-ka59-l2-4362",
        GAP_E3_WORLD_MODEL_RULE_FT09_L2_4373: "exp4377-gap-e3-world-model-rule-ft09-l2-4373",
        GAP_4370: "exp4377-gap-4370",
    }
    for gap in gap_entries:
        marker = marker_by_gap_id.get(gap["gap_id"], f"exp4377-{gap['gap_id'].lower()}")
        gaps_text = _replace_or_append_gap(gaps_text, marker, gap)

    registry_ok = registry_contains_v404(updated_registry)
    arc_ok = arc_registry_contains_v404(updated_arc)
    gaps_ok = gaps_contain_v404(gaps_text, gap_entries)
    return (
        updated_registry,
        gaps_text,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "registries_reconciled": registry_ok and arc_ok and gaps_ok,
            "filled_gap_ids": [GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361],
            "gaps_logged_ids": [gap["gap_id"] for gap in gap_entries],
        },
    )


def _flags_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = report.get("reports")
    if isinstance(reports, list) and reports:
        first = reports[0]
        if isinstance(first, Mapping):
            return [dict(flag) for flag in first.get("flags", []) if isinstance(flag, Mapping)]
    return []


def _capstone_aggregation_propagates_oracle_stamp() -> bool:
    return (
        "verifier_is_oracle" in capstone_v403_4368.REQUIRED_ARTIFACT_FIELDS
        and "verifier_is_oracle" in capstone_v403_4368.FIELD_PRINCIPLES
    )


def _capstone_aggregation_uses_available_helper() -> bool:
    return (
        capstone_v403_4368.aggregate.aggregate_available_report_gaps
        is capstone_aggregate_available.aggregate_available_report_gaps
    )


def verify_capstone_stamp_fix_durable(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4377: re-run adversarial_verify.py on the .403 capstone."""
    capstone_path = repo_root / CAPSTONE_V403_PATH
    propagates = _capstone_aggregation_propagates_oracle_stamp()
    uses_helper = _capstone_aggregation_uses_available_helper()
    try:
        capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "capstone_stamp_fix_durable": False,
            "capstone_path": CAPSTONE_V403_PATH,
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
        "capstone_path": CAPSTONE_V403_PATH,
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
        "method": "cached_v404_ledger_reconciliation_gap4_guard_and_stamp_audit",
        "upstream_artifacts": [
            EXP4370_PATH,
            EXP4371_PATH,
            EXP4372_PATH,
            EXP4373_PATH,
            EXP4374_PATH,
            EXP4375_PATH,
            CAPSTONE_V403_PATH,
        ],
        "gap4_guard_source": "carnot.reporting.verifier_registry_gaps_hygiene_gap4_guard_4366",
        "capstone_stamp_source": CAPSTONE_V403_PATH,
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
    v404_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = bool(gap4_regression_guard.get("regression_guard_passed"))
    stamp_ok = bool(capstone_stamp_fix.get("capstone_stamp_fix_durable"))
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    complete = guard_ok and stamp_ok and reconciled
    artifact = {
        "experiment": "experiment_4377_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4377_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v404_truth_"
            f"gap4_guard_passed_{guard_ok}_capstone_stamp_fix_durable_{stamp_ok}"
            if complete
            else "blocked_v404_hygiene_incomplete"
        ),
        "gap4_regression_guard_passed": guard_ok,
        "capstone_stamp_fix_durable": stamp_ok,
        "registries_reconciled": reconciled,
        "preconditions_checked": preconditions_checked,
        "reproducibility_checksum": reproducibility_checksum,
        "v404_outcomes": v404_outcomes,
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
        "experiment": "experiment_4377_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4377_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": f"blocked_{blocked}_unreadable",
        "gap4_regression_guard_passed": False,
        "capstone_stamp_fix_durable": False,
        "registries_reconciled": False,
        "preconditions_checked": preflight,
        "reproducibility_checksum": f"blocked:{blocked}_unreadable",
        "v404_outcomes": {},
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
    """Validate the Exp 4377 terminal artifact before writing."""
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
        "v404_outcomes",
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
    if isinstance(artifact["random_seed"], bool) or not isinstance(
        artifact["random_seed"], int
    ):
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4377 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4377 and SCENARIO-VERIFY-4377")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4377 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix_durable
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4377_ARTIFACT_PATH
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
    outcomes = load_v404_outcomes(repo_root)
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v404(
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
    elif not arc_registry_contains_v404(yaml.safe_load(original_arc_text) or {}):
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
        v404_outcomes=outcomes,
        registry_reconciliation=summary,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised by results entrypoint tests.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4377_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
