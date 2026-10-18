"""Exp 4366 registry/gaps hygiene, GAP-4 guard, and capstone stamp durability.

Spec refs: REQ-VERIFY-4366, SCENARIO-VERIFY-4366.
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

from carnot.reporting import capstone_v402_4357
from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_capstone_stamp_fix_4355 as exp4355


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4366
INFERENCE_SUBSTRATE = "cached_registry_reconciliation_gap4_guard_and_capstone_stamp_audit"

EXP4366_ARTIFACT_PATH = "results/experiment_4366_registry_gaps_hygiene_gap4_guard.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
ARC_REGISTRY_PATH = "ops/arc_solve_registry.yaml"
GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
ACTION_COST_VERIFIER_ID = exp4355.ACTION_COST_VERIFIER_ID

CAPSTONE_V402_PATH = "results/experiment_4357_capstone_v402.json"
EXP4359_PATH = "results/experiment_4359_prism_hardened_verifier_guided_search.json"
EXP4361_PATH = "results/experiment_4361_e3_deeper_high_headroom_games.json"
EXP4362_PATH = "results/experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json"
EXP4363_PATH = "results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json"
EXP4364_PATH = "results/experiment_4364_self_learning_action_cost_compounds.json"

V403_ROLE_ID = "oracle_distinct_v403_registry_gaps_hygiene_4366"
V403_STATE = (
    "prism_scorer_leaky__arc_total_33_games_17__tu93_tr87_ft09_reproduced__"
    "action_cost_compounds"
)

GAP_E3_WORLD_MODEL_RULE_TR87_4329 = "GAP-E3-WORLD-MODEL-RULE-TR87-4329"
GAP_E3_WORLD_MODEL_RULE_FT09_4329 = "GAP-E3-WORLD-MODEL-RULE-FT09-4329"
GAP_E3_WORLD_MODEL_RULE_TR87_4352 = exp4355.GAP_E3_WORLD_MODEL_RULE_TR87_4352
GAP_E3_WORLD_MODEL_RULE_FT09_4352 = exp4355.GAP_E3_WORLD_MODEL_RULE_FT09_4352
GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361 = "GAP-E3-WORLD-MODEL-RULE-SC25-L2-4361"
GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361 = "GAP-E3-WORLD-MODEL-RULE-TN36-L8-4361"
GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361 = "GAP-E3-WORLD-MODEL-RULE-LP85-L5-4361"
GAP_E3_WORLD_MODEL_RULE_AR25_L2_4362 = "GAP-E3-WORLD-MODEL-RULE-AR25-L2-4362"
GAP_E3_WORLD_MODEL_RULE_KA59_L2_4362 = "GAP-E3-WORLD-MODEL-RULE-KA59-L2-4362"

SPEC_REFS = ["REQ-VERIFY-4366", "SCENARIO-VERIFY-4366"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_regression_guard_passed",
    "capstone_stamp_fix_durable",
    "registries_reconciled",
    "preconditions_checked",
    "reproducibility_checksum",
    "v403_outcomes",
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
        "arc_solve_registry.yaml updated with the .403 outcomes (never-prune)."
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
    """REQ-VERIFY-4366: read all ledgers before mutating any of them."""
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
    """REQ-VERIFY-4366: reuse the durable GAP-4 regression guard."""
    return exp4355.run_gap4_regression_guard(repo_root)


def _read_prism(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4359_PATH, "available": False, "error": error}
    return {
        "artifact_path": EXP4359_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "acceptance_gate": _bool(payload, "acceptance_gate"),
        "controls_differentiated": _bool(payload, "controls_differentiated"),
        "s3_guided_beats_control": _bool(payload, "s3_guided_beats_control"),
        "scorer_leak_recheck_passed": _bool(payload, "scorer_leak_recheck_passed"),
        "benchmark_n": _int(payload, "benchmark_n"),
        "s3_gain_ci95": _list(payload, "s3_gain_ci95"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


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


def _read_deeper(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4361_PATH, "available": False, "error": error, "targets": {}}
    return {
        "artifact_path": EXP4361_PATH,
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
        return {"artifact_path": EXP4362_PATH, "available": False, "error": error, "games": {}}
    return {
        "artifact_path": EXP4362_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "reproducible_total_levels": _int(payload, "reproducible_total_levels"),
        "games": _scorecard_map(_list(payload, "per_game_scorecard"), "residual_gap_class"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _read_tail_games(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4363_PATH, "available": False, "error": error, "games": {}}
    return {
        "artifact_path": EXP4363_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "new_levels_reproduced": _int(payload, "new_levels_reproduced"),
        "games": _scorecard_map(_list(payload, "per_game_scorecard"), "residual_mismatch_class"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def _curve_endpoint(curve: list[Any], default: int) -> tuple[int, int]:
    rows = [row for row in curve if isinstance(row, Mapping)]
    if not rows:
        return default, default
    first = rows[0].get("held_out_actions_to_solve", default)
    last = rows[-1].get("held_out_actions_to_solve", default)
    return int(first), int(last)


def _read_action_compounding(payload: dict[str, Any] | None, error: str) -> dict[str, Any]:
    if payload is None:
        return {"artifact_path": EXP4364_PATH, "available": False, "error": error}
    curve = _list(payload, "compounding_curve")
    baseline, learned = _curve_endpoint(curve, 0)
    return {
        "artifact_path": EXP4364_PATH,
        "available": True,
        "honest_verdict": _str(payload, "honest_verdict"),
        "acceptance_gate_passed": _bool(payload, "acceptance_gate_passed") is True,
        "action_efficiency_compounds": _bool(payload, "action_efficiency_compounds") is True,
        "deployed_into_solver_kit": _bool(payload, "deployed_into_solver_kit") is True,
        "held_out_actions_baseline": baseline,
        "held_out_actions_learned": learned,
        "positive_control_passed": _bool(payload, "positive_control_passed") is True,
        "reproduction_gated": _bool(payload, "reproduction_gated") is True,
        "compounding_curve": curve,
        "missing_verifier_gaps": _list(payload, "missing_verifier_gaps"),
        "verifier_is_oracle": _bool(payload, "verifier_is_oracle"),
        "reproducibility_checksum": _str(payload, "reproducibility_checksum"),
    }


def load_v403_outcomes(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4366: read .403 outcomes without fabricating missing artifacts."""
    prism_payload, prism_error = _load_optional_json(repo_root, EXP4359_PATH)
    deeper_payload, deeper_error = _load_optional_json(repo_root, EXP4361_PATH)
    blocked_payload, blocked_error = _load_optional_json(repo_root, EXP4362_PATH)
    tail_payload, tail_error = _load_optional_json(repo_root, EXP4363_PATH)
    action_payload, action_error = _load_optional_json(repo_root, EXP4364_PATH)
    return {
        "prism_hardened_moat_utility": _read_prism(prism_payload, prism_error),
        "arc_e3": {
            "deeper_high_headroom": _read_deeper(deeper_payload, deeper_error),
            "blocked_mechanics": _read_blocked_mechanics(blocked_payload, blocked_error),
            "tail_games": _read_tail_games(tail_payload, tail_error),
        },
        "action_cost_compounding": _read_action_compounding(action_payload, action_error),
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
    """REQ-VERIFY-4366: collect the .403 residual missing-verifier gaps."""
    entries: dict[str, dict[str, Any]] = {}
    deeper = outcomes["arc_e3"]["deeper_high_headroom"]["targets"]
    blocked = outcomes["arc_e3"]["blocked_mechanics"]["games"]
    gap_specs = [
        (deeper, "sc25", GAP_E3_WORLD_MODEL_RULE_SC25_L2_4361, EXP4361_PATH, "L2"),
        (deeper, "tn36", GAP_E3_WORLD_MODEL_RULE_TN36_L8_4361, EXP4361_PATH, "L8"),
        (deeper, "lp85", GAP_E3_WORLD_MODEL_RULE_LP85_L5_4361, EXP4361_PATH, "L5"),
        (blocked, "ar25", GAP_E3_WORLD_MODEL_RULE_AR25_L2_4362, EXP4362_PATH, "L2"),
        (blocked, "ka59", GAP_E3_WORLD_MODEL_RULE_KA59_L2_4362, EXP4362_PATH, "L2"),
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
                "mine divergent traces for the named residual, add transition tests, "
                "and count progress only through the offline reproduce() gate"
            ),
        )
    return list(entries.values())


def _gap_entry_block(gap: Mapping[str, Any]) -> str:
    return (
        f"### {gap['gap_id']}: Exp 4366 .403 verifier gap update\n"
        f"- status: {gap.get('status', 'open')}\n"
        f"- evidence: {gap.get('evidence', '')}.\n"
        f"- failure mode: {gap.get('failure_mode', '')}\n"
        f"- missing discriminator: {gap.get('missing_discriminator', '')}\n"
        f"- candidate design: {gap.get('candidate_design', '')}\n"
        f"- priority: {gap.get('priority', 'high')}\n"
    )


def _filled_tail_gap_block(gap_id: str, game: str, row: Mapping[str, Any]) -> str:
    return (
        f"### {gap_id}: Exp 4366 .403 filled verifier gap update\n"
        "- status: filled (exp4363_tr87_ft09_world_models)\n"
        f"- evidence: {EXP4363_PATH}; game={game}; offline_reproduced="
        f"{row.get('offline_reproduced')}; reproduced_levels={row.get('reproduced_levels')}; "
        f"verifier_accuracy={row.get('verifier_accuracy')}; "
        f"residual_mismatch_class={row.get('residual_gap_class')}; "
        f"world_model_path={row.get('world_model_path')}.\n"
        f"- failure mode: the prior {game} partial world-model rule blocker no longer "
        "prevents an offline reproduced L1 gate.\n"
        "- missing discriminator: none for the reproduced L1 tail-game plan; deeper "
        "future mechanics remain separate gaps if exposed.\n"
        "- candidate design: preserve the Exp 4363 mechanic checks and reproduce() "
        "gate for this game.\n"
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
    prism = outcomes["prism_hardened_moat_utility"]
    arc = outcomes["arc_e3"]
    deeper = arc["deeper_high_headroom"]
    tails = arc["tail_games"]["games"]
    action = outcomes["action_cost_compounding"]
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4366": EXP4366_ARTIFACT_PATH,
            "exp4366_gap4_regression_guard_passed": bool(
                guard.get("regression_guard_passed")
            ),
            "exp4366_arc1_rule_exec_vote_pass2": replay.get("vote_pass2"),
            "exp4366_arc1_rule_exec_gated_pass2": replay.get("gated_pass2"),
            "exp4366_arc1_headroom_recovered": replay.get("headroom_recovered"),
            "exp4366_arc1_vote_wins_lost": replay.get("vote_wins_lost"),
            "exp4366_v403_state": V403_STATE,
            "exp4366_prism_artifact": EXP4359_PATH,
            "exp4366_prism_scorer_leak_recheck_passed": prism.get(
                "scorer_leak_recheck_passed"
            ),
            "exp4366_prism_controls_differentiated": prism.get(
                "controls_differentiated"
            ),
            "exp4366_prism_guided_beats_control": prism.get(
                "s3_guided_beats_control"
            ),
            "exp4366_arc_reproducible_total_levels": deeper.get(
                "reproducible_total_levels"
            ),
            "exp4366_tu93_reproduced_level": deeper["targets"].get("tu93", {}).get(
                "new_reproduced_level"
            ),
            "exp4366_tr87_offline_reproduced": tails.get("tr87", {}).get(
                "offline_reproduced"
            ),
            "exp4366_ft09_offline_reproduced": tails.get("ft09", {}).get(
                "offline_reproduced"
            ),
            "exp4366_action_efficiency_compounds": action.get(
                "action_efficiency_compounds"
            ),
            "exp4366_action_cost_deployed_into_solver_kit": action.get(
                "deployed_into_solver_kit"
            ),
            "exp4366_action_cost_baseline_actions": action.get(
                "held_out_actions_baseline"
            ),
            "exp4366_action_cost_learned_actions": action.get(
                "held_out_actions_learned"
            ),
            "exp4366_filled_gaps": [
                GAP_E3_WORLD_MODEL_RULE_TR87_4352,
                GAP_E3_WORLD_MODEL_RULE_FT09_4352,
            ],
            "exp4366_gaps_logged": [gap["gap_id"] for gap in gap_entries],
        }
    )


def _ensure_v403_role(
    registry: dict[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V403_ROLE_ID,
        "experiment": EXP4366_ARTIFACT_PATH,
        "role": "registry_gaps_arc_hygiene_v403",
        "status": "v403_outcomes_recorded_with_gap4_guard_and_stamp_durability",
        "v403_state": V403_STATE,
        "prism_scorer_leak_recheck_passed": outcomes["prism_hardened_moat_utility"].get(
            "scorer_leak_recheck_passed"
        ),
        "arc_reproducible_total_levels": outcomes["arc_e3"][
            "deeper_high_headroom"
        ].get("reproducible_total_levels"),
        "action_efficiency_compounds": outcomes["action_cost_compounding"].get(
            "action_efficiency_compounds"
        ),
        "gap_ids_logged": [gap["gap_id"] for gap in gap_entries],
        "filled_gap_ids": [
            GAP_E3_WORLD_MODEL_RULE_TR87_4352,
            GAP_E3_WORLD_MODEL_RULE_FT09_4352,
        ],
        "eval_exp_4366": EXP4366_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [
        old for old in old_roles if old.get("role_id") != V403_ROLE_ID
    ] + [role]


def _ensure_action_cost_verifier(registry: dict[str, Any], outcomes: Mapping[str, Any]) -> None:
    action = outcomes["action_cost_compounding"]
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
                "Oracle-distinct learned action-cost heuristic: Exp 4364 confirms "
                "the held-out action curve compounds to 16 actions and is deployed "
                "as the solver-kit default."
            ),
        }
        registry.setdefault("verifiers", []).append(entry)
    entry.setdefault("eval", {}).update(
        {
            "metric": "held_out_actions_to_solve",
            "eval_exp_4364": EXP4364_PATH,
            "eval_exp_4366": EXP4366_ARTIFACT_PATH,
            "action_efficiency_compounds": action.get("action_efficiency_compounds"),
            "deployed_into_solver_kit": action.get("deployed_into_solver_kit"),
            "held_out_actions_baseline": action.get("held_out_actions_baseline"),
            "held_out_actions_learned": action.get("held_out_actions_learned"),
            "positive_control_passed": action.get("positive_control_passed"),
            "reproduction_gated": action.get("reproduction_gated"),
            "compounding_curve": action.get("compounding_curve"),
            "verifier_is_oracle": action.get("verifier_is_oracle"),
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
    tails = outcomes["arc_e3"]["tail_games"]["games"]
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
    arc_registry.setdefault("games", [])
    tu93 = deeper["targets"].get("tu93", {})
    if tu93.get("offline_reproduced") is True:
        row = _ensure_game(arc_registry, "tu93")
        row.update(
            {
                "reproducibility": "reproduced",
                "levels_reproduced": max(
                    int(row.get("levels_reproduced") or 0),
                    int(tu93.get("new_reproduced_level") or 0),
                ),
                "solver": f"{EXP4361_PATH} + {tu93.get('world_model_path')}",
            }
        )
    for game in ("tr87", "ft09"):
        tail = tails.get(game, {})
        if tail.get("offline_reproduced") is True:
            row = _ensure_game(arc_registry, game)
            row.update(
                {
                    "reproducibility": "reproduced",
                    "levels_reproduced": max(
                        int(row.get("levels_reproduced") or 0),
                        int(tail.get("reproduced_levels") or 0),
                    ),
                    "solver": f"{EXP4363_PATH} + {tail.get('world_model_path')}",
                    "world_model": tail.get("world_model_path"),
                    "world_model_sha256": tail.get("world_model_sha256"),
                }
            )


def registry_contains_v403(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    action = base._find_verifier(registry, ACTION_COST_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4366") == EXP4366_ARTIFACT_PATH
        and gap4.get("eval", {}).get("exp4366_v403_state") == V403_STATE
        and any(role.get("role_id") == V403_ROLE_ID for role in gap4.get("registry_roles", []))
        and action
        and action.get("eval", {}).get("eval_exp_4366") == EXP4366_ARTIFACT_PATH
        and action.get("eval", {}).get("action_efficiency_compounds") is True
    )


def arc_registry_contains_v403(arc_registry: dict[str, Any]) -> bool:
    tu93 = _find_game(arc_registry, "tu93") or {}
    tr87 = _find_game(arc_registry, "tr87") or {}
    ft09 = _find_game(arc_registry, "ft09") or {}
    return bool(
        int(arc_registry.get("reproducible_total_levels") or 0) >= 33
        and int(arc_registry.get("reproducible_total_games") or 0) >= 17
        and tu93.get("reproducibility") == "reproduced"
        and int(tu93.get("levels_reproduced") or 0) >= 4
        and tr87.get("reproducibility") == "reproduced"
        and int(tr87.get("levels_reproduced") or 0) >= 1
        and ft09.get("reproducibility") == "reproduced"
        and int(ft09.get("levels_reproduced") or 0) >= 1
    )


def gaps_contain_v403(gaps_text: str, gap_entries: list[dict[str, Any]]) -> bool:
    return (
        "status: filled (exp4363_tr87_ft09_world_models)" in gaps_text
        and GAP_E3_WORLD_MODEL_RULE_TR87_4352 in gaps_text
        and GAP_E3_WORLD_MODEL_RULE_FT09_4352 in gaps_text
        and all(gap["gap_id"] in gaps_text for gap in gap_entries)
    )


def ensure_ledgers_record_v403(
    registry: dict[str, Any],
    gaps_text: str,
    arc_registry: dict[str, Any],
    regression_guard: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    gap_entries: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return ledgers with .403 truth represented idempotently."""
    updated_registry = deepcopy(registry)
    updated_arc = deepcopy(arc_registry)
    _ensure_gap4_eval(updated_registry, regression_guard, outcomes, gap_entries)
    _ensure_v403_role(updated_registry, outcomes, gap_entries)
    _ensure_action_cost_verifier(updated_registry, outcomes)
    _ensure_arc_registry(updated_arc, outcomes)

    tail_games = outcomes["arc_e3"]["tail_games"]["games"]
    for marker, gap_id, game in (
        ("exp4333-gap-e3-world-model-rule-tr87-4329", GAP_E3_WORLD_MODEL_RULE_TR87_4329, "tr87"),
        ("exp4333-gap-e3-world-model-rule-ft09-4329", GAP_E3_WORLD_MODEL_RULE_FT09_4329, "ft09"),
        ("exp4355-gap-e3-world-model-rule-tr87-4352", GAP_E3_WORLD_MODEL_RULE_TR87_4352, "tr87"),
        ("exp4355-gap-e3-world-model-rule-ft09-4352", GAP_E3_WORLD_MODEL_RULE_FT09_4352, "ft09"),
    ):
        row = tail_games.get(game, {})
        if row.get("offline_reproduced") is True:
            gaps_text = base._replace_marked_block(
                gaps_text,
                marker,
                _filled_tail_gap_block(gap_id, game, row),
            )
    for gap in gap_entries:
        gaps_text = base._replace_marked_block(
            gaps_text,
            f"exp4366-{gap['gap_id'].lower()}",
            _gap_entry_block(gap),
        )

    registry_ok = registry_contains_v403(updated_registry)
    arc_ok = arc_registry_contains_v403(updated_arc)
    gaps_ok = gaps_contain_v403(gaps_text, gap_entries)
    return (
        updated_registry,
        gaps_text,
        updated_arc,
        {
            "verifier_registry_reconciled": registry_ok,
            "arc_solve_registry_reconciled": arc_ok,
            "verifier_gaps_reconciled": gaps_ok,
            "registries_reconciled": registry_ok and arc_ok and gaps_ok,
            "filled_gap_ids": [
                GAP_E3_WORLD_MODEL_RULE_TR87_4352,
                GAP_E3_WORLD_MODEL_RULE_FT09_4352,
            ],
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
        "verifier_is_oracle" in capstone_v402_4357.REQUIRED_ARTIFACT_FIELDS
        and "verifier_is_oracle" in capstone_v402_4357.FIELD_PRINCIPLES
    )


def verify_capstone_stamp_fix_durable(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4366: re-run adversarial_verify.py on the .402 capstone."""
    capstone_path = repo_root / CAPSTONE_V402_PATH
    try:
        capstone = json.loads(capstone_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "capstone_stamp_fix_durable": False,
            "capstone_path": CAPSTONE_V402_PATH,
            "error": f"{type(exc).__name__}: {exc}",
            "capstone_verifier_is_oracle": None,
            "capstone_aggregation_propagates_oracle_stamp": _capstone_aggregation_propagates_oracle_stamp(),
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
    propagates = _capstone_aggregation_propagates_oracle_stamp()
    durable = (
        capstone.get("verifier_is_oracle") is False
        and propagates
        and not circular
        and not flags
        and completed.returncode == 0
    )
    return {
        "capstone_stamp_fix_durable": durable,
        "capstone_path": CAPSTONE_V402_PATH,
        "capstone_verifier_is_oracle": capstone.get("verifier_is_oracle"),
        "capstone_aggregation_propagates_oracle_stamp": propagates,
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
        "method": "cached_v403_ledger_reconciliation_gap4_guard_and_stamp_audit",
        "upstream_artifacts": [
            EXP4359_PATH,
            EXP4361_PATH,
            EXP4362_PATH,
            EXP4363_PATH,
            EXP4364_PATH,
            CAPSTONE_V402_PATH,
        ],
        "gap4_guard_source": "carnot.reporting.verifier_registry_gaps_hygiene_capstone_stamp_fix_4355",
        "capstone_stamp_source": "results/experiment_4357_capstone_v402.json",
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
    v403_outcomes: dict[str, Any],
    registry_reconciliation: dict[str, Any],
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    guard_ok = bool(gap4_regression_guard.get("regression_guard_passed"))
    stamp_ok = bool(capstone_stamp_fix.get("capstone_stamp_fix_durable"))
    reconciled = bool(registry_reconciliation.get("registries_reconciled"))
    complete = guard_ok and stamp_ok and reconciled
    artifact = {
        "experiment": "experiment_4366_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4366_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": (
            "complete: registry_gaps_arc_reconciled_to_v403_truth_"
            f"gap4_guard_passed_{guard_ok}_capstone_stamp_fix_durable_{stamp_ok}"
            if complete
            else "blocked_v403_hygiene_incomplete"
        ),
        "gap4_regression_guard_passed": guard_ok,
        "capstone_stamp_fix_durable": stamp_ok,
        "registries_reconciled": reconciled,
        "preconditions_checked": preconditions_checked,
        "reproducibility_checksum": reproducibility_checksum,
        "v403_outcomes": v403_outcomes,
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
        "experiment": "experiment_4366_registry_gaps_hygiene_gap4_guard",
        "schema": "carnot.experiment_4366_registry_gaps_hygiene_gap4_guard.v1",
        "honest_verdict": f"blocked_{blocked}_unreadable",
        "gap4_regression_guard_passed": False,
        "capstone_stamp_fix_durable": False,
        "registries_reconciled": False,
        "preconditions_checked": preflight,
        "reproducibility_checksum": f"blocked:{blocked}_unreadable",
        "v403_outcomes": {},
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
    """Validate the Exp 4366 terminal artifact before writing."""
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
        "v403_outcomes",
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
        raise ValueError("field_principles must match the required Exp 4366 principles")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs must cite REQ-VERIFY-4366 and SCENARIO-VERIFY-4366")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    gap4_guard_runner: Gap4GuardRunner | None = None,
    capstone_stamp_runner: CapstoneStampRunner | None = None,
) -> dict[str, Any]:
    """Run Exp 4366 and write the terminal artifact plus reconciled ledgers."""
    if gap4_guard_runner is None:
        gap4_guard_runner = run_gap4_regression_guard
    if capstone_stamp_runner is None:
        capstone_stamp_runner = verify_capstone_stamp_fix_durable
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4366_ARTIFACT_PATH
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
    outcomes = load_v403_outcomes(repo_root)
    gap_entries = build_gap_entries(outcomes)
    registry, gaps_text, arc_registry, summary = ensure_ledgers_record_v403(
        registry,
        gaps_text,
        arc_registry,
        guard,
        outcomes,
        gap_entries,
    )

    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    if not arc_registry_contains_v403(yaml.safe_load(arc_path.read_text(encoding="utf-8")) or {}):
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
        v403_outcomes=outcomes,
        registry_reconciliation=summary,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised by results entrypoint tests.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4366_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
