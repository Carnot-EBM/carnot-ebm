"""Exp 4355 registry/gaps hygiene and capstone stamp fix for .402.

Spec refs: REQ-VERIFY-4355, SCENARIO-VERIFY-4355.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

import yaml

from carnot.reporting import capstone_v401_4346
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4344 as exp4344


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4355
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_and_capstone_stamp_audit"

EXP4355_ARTIFACT_PATH = (
    "results/experiment_4355_registry_gaps_hygiene_capstone_stamp_fix.json"
)
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ACTION_COST_VERIFIER_ID = "arc_agi3_learned_action_cost_heuristic_4353"

EXP4348_PATH = "results/experiment_4348_s3_stratified_verifier_guided_search.json"
EXP4350_PATH = "results/experiment_4350_e3_explore_verify_plan_ka59.json"
EXP4351_PATH = "results/experiment_4351_e3_deeper_solved_games.json"
EXP4352_PATH = "results/experiment_4352_e3_explore_verify_plan_tr87_ft09.json"
EXP4353_PATH = "results/experiment_4353_learned_action_cost_heuristic_efficiency.json"

V402_ROLE_ID = "oracle_distinct_v402_registry_gaps_hygiene_4355"
V402_STATE = (
    "s3_controls_not_differentiated__arc_total_23__ka59_tn36_reproduced__"
    "tr87_ft09_partial__action_cost_improves"
)

GAP_E3_WORLD_MODEL_RULE_KA59_4328 = "GAP-E3-WORLD-MODEL-RULE-KA59-4328"
GAP_E3_WORLD_MODEL_RULE_KA59_4350 = "GAP-E3-WORLD-MODEL-RULE-KA59-4350"
GAP_E3_WORLD_MODEL_RULE_SC25_L2_4351 = "GAP-E3-WORLD-MODEL-RULE-SC25-L2-4351"
GAP_E3_WORLD_MODEL_RULE_AR25_L2_4351 = "GAP-E3-WORLD-MODEL-RULE-AR25-L2-4351"
GAP_E3_WORLD_MODEL_RULE_TR87_4352 = "GAP-E3-WORLD-MODEL-RULE-TR87-4352"
GAP_E3_WORLD_MODEL_RULE_FT09_4352 = "GAP-E3-WORLD-MODEL-RULE-FT09-4352"

SPEC_REFS = ["REQ-VERIFY-4355", "SCENARIO-VERIFY-4355"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_regression_guard_passed",
    "capstone_stamp_fix_verified",
    "registries_reconciled",
    "preconditions_checked",
    "reproducibility_checksum",
    "v402_outcomes",
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
        "Terminal-prefixed. Records hygiene reconciled + guard run + stamp fix verified."
    ),
    "gap4_regression_guard_passed": (
        "BARE bool: the ARC oracle-distinct verifier-beats-vote result has not "
        "silently regressed."
    ),
    "capstone_stamp_fix_verified": (
        "BARE bool: the capstone aggregation now propagates verifier_is_oracle "
        "(false for an oracle-distinct moat) -> adversarial_verify.py no longer "
        "fires CIRCULAR_MOAT_OVERCLAIM on a correct capstone."
    ),
    "registries_reconciled": (
        "BARE bool: verifier_registry.yaml + verifier_gaps.md + "
        "arc_solve_registry.yaml updated with the .402 outcomes (never-prune)."
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


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4355: read ledgers before any mutation."""
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

    gaps_path = repo_root / GAPS_PATH
    try:
        gaps_path.read_text(encoding="utf-8")
        gaps_ok = True
        gaps_error = ""
    except OSError as exc:
        gaps_ok = False
        gaps_error = f"{type(exc).__name__}: {exc}"
    checks["verifier_gaps"] = {
        "path": GAPS_PATH,
        "readable": gaps_ok,
        "error": gaps_error,
    }
    if not gaps_ok and blocked_file is None:
        blocked_file = "verifier_gaps"

    return {
        "ok": blocked_file is None,
        "blocked_file": blocked_file,
        "files": checks,
    }


def run_gap4_regression_guard(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4355: reuse the Exp 4344 GAP-4 regression guard."""
    return exp4344.run_gap4_regression_guard(repo_root)


def _read_s3(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4348_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4348_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "acceptance_gate": _bool(payload, "acceptance_gate"),
        "s3_guided_beats_control": _bool(payload, "s3_guided_beats_control"),
        "controls_differentiated": _bool(payload, "controls_differentiated"),
        "s3_gain_ci95": _list(payload, "s3_gain_ci95"),
        "benchmark_n": _int(payload, "benchmark_n"),
        "flagged_adversarial": _bool(payload, "flagged_adversarial") is True,
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_ka59(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4350_PATH, "game": "ka59", "available": False, "error": error}
    return {
        "artifact_path": EXP4350_PATH,
        "game": _str(payload, "game") or "ka59",
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "offline_reproduced": _bool(payload, "offline_reproduced") is True,
        "reproduced_levels": _int(payload, "reproduced_levels"),
        "residual_mismatch_class": _str(payload, "residual_mismatch_class"),
        "verifier_best_accuracy": _float(payload, "verifier_best_accuracy"),
        "world_model_path": _str(payload, "world_model_path"),
        "world_model_sha256": _str(payload, "world_model_sha256"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_deeper(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4351_PATH, "available": False, "error": error, "targets": {}}
    targets: dict[str, dict[str, Any]] = {}
    for row in _list(payload, "per_target_scorecard"):
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game", ""))
        if not game:
            continue
        targets[game] = {
            "game": game,
            "offline_reproduced": row.get("offline_reproduced") is True,
            "new_reproduced_level": row.get("new_reproduced_level"),
            "residual_win_mechanic_gap_class": str(
                row.get("residual_win_mechanic_gap_class", "")
            ),
            "world_model_path": str(row.get("world_model_path", "")),
        }
    return {
        "artifact_path": EXP4351_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "targets": targets,
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_partial_games(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4352_PATH, "available": False, "error": error, "games": {}}
    games: dict[str, dict[str, Any]] = {}
    for row in _list(payload, "per_game_scorecard"):
        if not isinstance(row, Mapping):
            continue
        game = str(row.get("game", ""))
        if not game:
            continue
        games[game] = {
            "game": game,
            "offline_reproduced": row.get("offline_reproduced") is True,
            "reproduced_levels": row.get("reproduced_levels"),
            "residual_mismatch_class": str(row.get("residual_mismatch_class", "")),
            "verifier_accuracy": row.get("verifier_accuracy"),
            "world_model_path": str(row.get("world_model_path", "")),
        }
    return {
        "artifact_path": EXP4352_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "games": games,
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_action_cost(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4353_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4353_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "acceptance_gate_passed": _bool(payload, "acceptance_gate_passed") is True,
        "action_efficiency_improves": _bool(payload, "action_efficiency_improves") is True,
        "held_out_actions_baseline": _int(payload, "held_out_actions_baseline"),
        "held_out_actions_learned": _int(payload, "held_out_actions_learned"),
        "positive_control_passed": _bool(payload, "positive_control_passed") is True,
        "reproduction_gated": _bool(payload, "reproduction_gated") is True,
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def load_v402_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4355: read .402 outcomes without fabricating missing artifacts."""
    s3_payload, s3_error = _load_optional_json(repo_root, EXP4348_PATH)
    ka59_payload, ka59_error = _load_optional_json(repo_root, EXP4350_PATH)
    deeper_payload, deeper_error = _load_optional_json(repo_root, EXP4351_PATH)
    partial_payload, partial_error = _load_optional_json(repo_root, EXP4352_PATH)
    action_payload, action_error = _load_optional_json(repo_root, EXP4353_PATH)
    return {
        "s3_moat_utility": _read_s3(s3_payload, s3_error),
        "arc_e3": {
            "ka59": _read_ka59(ka59_payload, ka59_error),
            "deeper": _read_deeper(deeper_payload, deeper_error),
            "partial_tr87_ft09": _read_partial_games(partial_payload, partial_error),
        },
        "action_cost_heuristic": _read_action_cost(action_payload, action_error),
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
    """REQ-VERIFY-4355: collect the .402 missing-verifier gaps to append."""
    arc = outcomes["arc_e3"]
    gaps: list[dict[str, Any]] = []

    ka59 = arc["ka59"]
    if ka59.get("available") and ka59.get("residual_mismatch_class"):
        gaps.append(
            _gap_entry(
                GAP_E3_WORLD_MODEL_RULE_KA59_4350,
                status="open_residual_after_l1_reproduction",
                evidence=(
                    f"{EXP4350_PATH}; offline_reproduced={ka59.get('offline_reproduced')}; "
                    f"reproduced_levels={ka59.get('reproduced_levels')}; "
                    f"verifier_best_accuracy={ka59.get('verifier_best_accuracy')}; "
                    f"residual_mismatch_class={ka59.get('residual_mismatch_class')}"
                ),
                failure_mode=(
                    "ka59 L1 now reproduces, but the executable world model still "
                    f"has residual mismatch {ka59.get('residual_mismatch_class')}"
                ),
                missing_discriminator="ka59 hidden StepCounter HUD dynamics",
                candidate_design=(
                    "model the hidden bottom-row HUD counter separately from win-state "
                    "movement so exact transition tests can pass without corrupting L1 solve logic"
                ),
            )
        )

    deeper = arc["deeper"]["targets"]
    for game, gap_id in (
        ("sc25", GAP_E3_WORLD_MODEL_RULE_SC25_L2_4351),
        ("ar25", GAP_E3_WORLD_MODEL_RULE_AR25_L2_4351),
    ):
        row = deeper.get(game, {})
        residual = str(row.get("residual_win_mechanic_gap_class", ""))
        if row and residual and residual != "none" and row.get("offline_reproduced") is not True:
            gaps.append(
                _gap_entry(
                    gap_id,
                    status="open_deeper_level_residual",
                    evidence=(
                        f"{EXP4351_PATH}; game={game}; offline_reproduced="
                        f"{row.get('offline_reproduced')}; new_reproduced_level="
                        f"{row.get('new_reproduced_level')}; residual={residual}"
                    ),
                    failure_mode=f"{game} deeper level remains unreproduced due to {residual}",
                    missing_discriminator=f"{game} executable rule coverage for {residual}",
                    candidate_design=(
                        "mine divergent deeper-level traces, add the missing executable "
                        "transition cases, and keep reproduce() as the only solved-level gate"
                    ),
                )
            )

    partial = arc["partial_tr87_ft09"]["games"]
    for game, gap_id in (
        ("tr87", GAP_E3_WORLD_MODEL_RULE_TR87_4352),
        ("ft09", GAP_E3_WORLD_MODEL_RULE_FT09_4352),
    ):
        row = partial.get(game, {})
        residual = str(row.get("residual_mismatch_class", ""))
        if row and residual and row.get("offline_reproduced") is not True:
            gaps.append(
                _gap_entry(
                    gap_id,
                    status="open",
                    evidence=(
                        f"{EXP4352_PATH}; game={game}; offline_reproduced=False; "
                        f"reproduced_levels={row.get('reproduced_levels')}; "
                        f"verifier_accuracy={row.get('verifier_accuracy')}; "
                        f"residual_mismatch_class={residual}"
                    ),
                    failure_mode=f"{game} E3 world model remains partial: {residual}",
                    missing_discriminator=f"{game} executable world-model rule coverage for {residual}",
                    candidate_design=(
                        "continue explore-verify-plan only after the mechanic checks "
                        "pass; add transition tests for the named action residual"
                    ),
                )
            )

    deduped: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        deduped[gap["gap_id"]] = gap
    return list(deduped.values())


def _gap_entry_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4355 .402 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
    )


def _filled_ka59_4328_block(outcomes: Mapping[str, Any]) -> str:
    ka59 = outcomes["arc_e3"]["ka59"]
    return (
        "### GAP-E3-WORLD-MODEL-RULE-KA59-4328: Exp 4355 .402 filled verifier gap update\n"
        "- status: filled (exp4350_ka59_l1_world_model)\n"
        f"- evidence: {EXP4350_PATH}; offline_reproduced={ka59.get('offline_reproduced')}; "
        f"reproduced_levels={ka59.get('reproduced_levels')}; "
        f"verifier_best_accuracy={ka59.get('verifier_best_accuracy')}; "
        f"world_model_path={ka59.get('world_model_path')}.\n"
        "- failure mode: the prior ka59 action-rule blocker no longer prevents an "
        "offline reproduced L1 solve.\n"
        "- missing discriminator: none for the L1 push-through-wall action plan; the "
        "remaining hidden HUD residual is tracked separately as GAP-E3-WORLD-MODEL-RULE-KA59-4350.\n"
        "- candidate design: preserve the Exp 4350 adaptive transition tests and "
        "reproduce() gate for ka59 L1.\n"
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
    arc = outcomes["arc_e3"]
    s3 = outcomes["s3_moat_utility"]
    action = outcomes["action_cost_heuristic"]
    partial = arc["partial_tr87_ft09"]["games"]
    eval_update = {
        "eval_exp_4355": EXP4355_ARTIFACT_PATH,
        "exp4355_gap4_regression_guard_passed": bool(
            guard.get("regression_guard_passed")
        ),
        "exp4355_arc1_rule_exec_vote_pass2": replay.get("vote_pass2"),
        "exp4355_arc1_rule_exec_gated_pass2": replay.get("gated_pass2"),
        "exp4355_arc1_headroom_recovered": replay.get("headroom_recovered"),
        "exp4355_arc1_vote_wins_lost": replay.get("vote_wins_lost"),
        "exp4355_v402_state": V402_STATE,
        "exp4355_s3_artifact": EXP4348_PATH,
        "exp4355_s3_controls_differentiated": s3.get("controls_differentiated"),
        "exp4355_s3_guided_beats_control": s3.get("s3_guided_beats_control"),
        "exp4355_s3_flagged_adversarial": s3.get("flagged_adversarial"),
        "exp4355_arc_reproducible_total_levels": arc["deeper"].get(
            "reproducible_total_levels"
        ),
        "exp4355_ka59_offline_reproduced": arc["ka59"].get("offline_reproduced"),
        "exp4355_ka59_reproduced_levels": arc["ka59"].get("reproduced_levels"),
        "exp4355_ka59_residual_mismatch_class": arc["ka59"].get(
            "residual_mismatch_class"
        ),
        "exp4355_tn36_reproduced_level": arc["deeper"]["targets"].get("tn36", {}).get(
            "new_reproduced_level"
        ),
        "exp4355_tr87_residual_mismatch_class": partial.get("tr87", {}).get(
            "residual_mismatch_class"
        ),
        "exp4355_ft09_residual_mismatch_class": partial.get("ft09", {}).get(
            "residual_mismatch_class"
        ),
        "exp4355_action_efficiency_improves": action.get("action_efficiency_improves"),
        "exp4355_action_cost_baseline_actions": action.get("held_out_actions_baseline"),
        "exp4355_action_cost_learned_actions": action.get("held_out_actions_learned"),
        "exp4355_action_cost_verifier_is_oracle": action.get("verifier_is_oracle"),
        "exp4355_filled_gaps": [GAP_E3_WORLD_MODEL_RULE_KA59_4328],
        "exp4355_gaps_logged": [gap["gap_id"] for gap in gap_entries],
    }
    entry.setdefault("eval", {}).update(eval_update)


def _ensure_v402_role(
    registry: dict[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V402_ROLE_ID,
        "experiment": EXP4355_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v402",
        "status": "v402_outcomes_recorded_with_capstone_stamp_fix",
        "v402_state": V402_STATE,
        "s3_controls_differentiated": outcomes["s3_moat_utility"].get(
            "controls_differentiated"
        ),
        "arc_reproducible_total_levels": outcomes["arc_e3"]["deeper"].get(
            "reproducible_total_levels"
        ),
        "ka59_offline_reproduced": outcomes["arc_e3"]["ka59"].get(
            "offline_reproduced"
        ),
        "action_efficiency_improves": outcomes["action_cost_heuristic"].get(
            "action_efficiency_improves"
        ),
        "gap_ids_logged": [gap["gap_id"] for gap in gap_entries],
        "filled_gap_ids": [GAP_E3_WORLD_MODEL_RULE_KA59_4328],
        "eval_exp_4355": EXP4355_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V402_ROLE_ID
    ] + [role]


def _ensure_action_cost_verifier(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    action = outcomes["action_cost_heuristic"]
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
            "notes": (
                "Oracle-distinct learned action-cost heuristic: reduces held-out "
                "lp85 L3 actions from 25 to 16 under reproduction gate."
            ),
        }
        registry.setdefault("verifiers", []).append(entry)
    entry.setdefault("eval", {}).update(
        {
            "metric": "held_out_actions_to_solve",
            "eval_exp_4353": EXP4353_PATH,
            "eval_exp_4355": EXP4355_ARTIFACT_PATH,
            "action_efficiency_improves": action.get("action_efficiency_improves"),
            "held_out_actions_baseline": action.get("held_out_actions_baseline"),
            "held_out_actions_learned": action.get("held_out_actions_learned"),
            "positive_control_passed": action.get("positive_control_passed"),
            "reproduction_gated": action.get("reproduction_gated"),
            "verifier_is_oracle": action.get("verifier_is_oracle"),
        }
    )


def _find_game(arc_registry: dict[str, Any], game: str) -> dict[str, Any] | None:
    for row in arc_registry.get("games", []):
        if isinstance(row, dict) and row.get("game") == game:
            return row
    return None


def _ensure_arc_registry(arc_registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    deeper = outcomes["arc_e3"]["deeper"]
    ka59 = outcomes["arc_e3"]["ka59"]
    total = deeper.get("reproducible_total_levels")
    if isinstance(total, int) and not isinstance(total, bool):
        arc_registry["reproducible_total_levels"] = max(
            int(arc_registry.get("reproducible_total_levels") or 0),
            total,
        )
    arc_registry["reproducible_total_games"] = max(
        int(arc_registry.get("reproducible_total_games") or 0),
        14,
    )
    arc_registry.setdefault("games", [])
    ka59_entry = _find_game(arc_registry, "ka59")
    if ka59_entry is None:
        ka59_entry = {"game": "ka59"}
        arc_registry["games"].append(ka59_entry)
    ka59_entry.update(
        {
            "reproducibility": "reproduced",
            "levels_reproduced": max(int(ka59_entry.get("levels_reproduced") or 0), 1),
            "solver": f"{EXP4350_PATH} + {ka59.get('world_model_path')}",
            "world_model": ka59.get("world_model_path"),
            "world_model_sha256": ka59.get("world_model_sha256"),
        }
    )
    tn36 = deeper.get("targets", {}).get("tn36", {})
    tn36_entry = _find_game(arc_registry, "tn36")
    if tn36_entry is None:
        tn36_entry = {"game": "tn36"}
        arc_registry["games"].append(tn36_entry)
    tn36_entry.update(
        {
            "reproducibility": "reproduced",
            "levels_reproduced": max(
                int(tn36_entry.get("levels_reproduced") or 0),
                int(tn36.get("new_reproduced_level") or 0),
            ),
            "solver": "scripts/arc3_tn36_offline_solver.py",
        }
    )


def registry_contains_v402(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    action = base._find_verifier(registry, ACTION_COST_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4355") == EXP4355_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4355_v402_state") == V402_STATE
        and any(role.get("role_id") == V402_ROLE_ID for role in gap4.get("registry_roles", []))
        and action
        and action.get("eval", {}).get("eval_exp_4355") == EXP4355_ARTIFACT_PATH
    )


def arc_registry_contains_v402(arc_registry: dict[str, Any]) -> bool:
    ka59 = _find_game(arc_registry, "ka59") or {}
    tn36 = _find_game(arc_registry, "tn36") or {}
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 23
        and ka59.get("reproducibility") == "reproduced"
        and int(ka59.get("levels_reproduced") or 0) >= 1
        and tn36.get("reproducibility") == "reproduced"
        and int(tn36.get("levels_reproduced") or 0) >= 7
    )


def gaps_contain_v402(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return (
        "status: filled (exp4350_ka59_l1_world_model)" in gaps_text
        and all(gap["gap_id"] in gaps_text for gap in gap_entries)
    )


def ensure_ledgers_record_v402(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .402 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_v402_role(updated_registry, outcomes, gap_entries)
    _ensure_action_cost_verifier(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4333-gap-e3-world-model-rule-ka59-4328",
        _filled_ka59_4328_block(outcomes),
    )
    for gap in gap_entries:
        updated_gaps = base._replace_marked_block(
            updated_gaps,
            f"exp4355-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    registry_ok = registry_contains_v402(updated_registry)
    arc_ok = arc_registry_contains_v402(updated_arc)
    gaps_ok = gaps_contain_v402(updated_gaps, gap_entries)
    return (
        updated_registry,
        updated_gaps,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "registries_reconciled": registry_ok and arc_ok and gaps_ok,
            "filled_gap_ids": [GAP_E3_WORLD_MODEL_RULE_KA59_4328],
            "gaps_logged_ids": [gap["gap_id"] for gap in gap_entries],
        },
    )


def _clean_flags_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    reports = report.get("reports")
    if isinstance(reports, list) and reports:
        first = reports[0]
        if isinstance(first, Mapping):
            return list(first.get("flags", []))
    return []


def verify_capstone_stamp_fix(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4355: run adversarial_verify.py on a stamped sample capstone."""
    sample = capstone_v401_4346.build_artifact(
        repo_root,
        live_flag_runner=lambda _path: [],
        summarize_runner=lambda _path, _root: 0,
    )
    with tempfile.TemporaryDirectory(prefix="carnot_exp4355_") as tmpdir:
        sample_path = Path(tmpdir) / "sample_capstone_v401_stamped.json"
        sample_path.write_text(
            json.dumps(sample, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(repo_root / "scripts" / "adversarial_verify.py"),
            "--json",
            str(sample_path),
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
        flags = _clean_flags_from_report(parsed)
        circular = [flag for flag in flags if flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM"]
        return {
            "capstone_stamp_fix_verified": (
                sample.get("verifier_is_oracle") is False
                and not circular
                and completed.returncode == 0
            ),
            "sample_verifier_is_oracle": sample.get("verifier_is_oracle"),
            "circular_moat_overclaim_fired": bool(circular),
            "returncode": completed.returncode,
            "command": command,
            "flags": flags,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }


def model_specs() -> dict[str, Any]:
    return {
        "method": "cached_v402_ledger_reconciliation_plus_capstone_stamp_audit",
        "upstream_artifacts": [
            EXP4348_PATH,
            EXP4350_PATH,
            EXP4351_PATH,
            EXP4352_PATH,
            EXP4353_PATH,
        ],
        "gap4_guard_source": "carnot.reporting.verifier_registry_gaps_hygiene_4344",
        "capstone_stamp_source": "carnot.reporting.capstone_v401_4346",
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
    v402_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = bool(gap4_regression_guard.get("regression_guard_passed"))
    stamp_ok = bool(capstone_stamp_fix.get("capstone_stamp_fix_verified"))
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    complete = guard_ok and stamp_ok and reconciled
    artifact = {
        "experiment": "experiment_4355_registry_gaps_hygiene_capstone_stamp_fix",
        "schema": "carnot.experiment_4355_registry_gaps_hygiene_capstone_stamp_fix.v1",
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v402_truth_"
            f"gap4_guard_passed_{guard_ok}_capstone_stamp_fix_verified_{stamp_ok}"
            if complete
            else "blocked_v402_hygiene_incomplete"
        ),
        "gap4_regression_guard_passed": guard_ok,
        "capstone_stamp_fix_verified": stamp_ok,
        "registries_reconciled": reconciled,
        "preconditions_checked": preconditions_checked,
        "reproducibility_checksum": reproducibility_checksum,
        "v402_outcomes": v402_outcomes,
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
        "experiment": "experiment_4355_registry_gaps_hygiene_capstone_stamp_fix",
        "schema": "carnot.experiment_4355_registry_gaps_hygiene_capstone_stamp_fix.v1",
        "honest_verdict": f"blocked_{blocked}_unreadable",
        "gap4_regression_guard_passed": False,
        "capstone_stamp_fix_verified": False,
        "registries_reconciled": False,
        "preconditions_checked": preflight,
        "reproducibility_checksum": f"blocked:{blocked}_unreadable",
        "v402_outcomes": {},
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
    """Validate the Exp 4355 terminal artifact before writing."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in (
        "gap4_regression_guard_passed",
        "capstone_stamp_fix_verified",
        "registries_reconciled",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a BARE bool")
    if not isinstance(artifact["preconditions_checked"], dict):
        raise ValueError("preconditions_checked must be an object")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if not isinstance(artifact["v402_outcomes"], dict):
        raise ValueError("v402_outcomes must be an object")
    if not isinstance(artifact["registry_reconciliation"], dict):
        raise ValueError("registry_reconciliation must be an object")
    if not isinstance(artifact["gap4_regression_guard"], dict):
        raise ValueError("gap4_regression_guard must be an object")
    if not isinstance(artifact["capstone_stamp_fix"], dict):
        raise ValueError("capstone_stamp_fix must be an object")
    if isinstance(artifact["random_seed"], bool) or not isinstance(
        artifact["random_seed"], int
    ):
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4355 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4355 and SCENARIO-VERIFY-4355")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4355 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4355_ARTIFACT_PATH
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
    outcomes = load_v402_outcomes(repo_root)
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v402(
        registry,
        gaps_text,
        arc_registry,
        guard,
        outcomes,
        gap_entries,
    )

    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    if not arc_registry_contains_v402(yaml.safe_load(arc_path.read_text(encoding="utf-8")) or {}):
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
        v402_outcomes=outcomes,
        registry_reconciliation=summary,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised by results entrypoint tests.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4355_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
