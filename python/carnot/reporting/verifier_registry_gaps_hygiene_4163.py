"""Exp 4163 registry/gaps hygiene for .385 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4163, SCENARIO-VERIFY-4163.

This runner is a ledger reconciler. It does not run a generator, a judge, Codex,
or a GGUF model; it replays the frozen GAP-4 ARC-1 guard from cached artifacts
and records the .385 Sudoku baseline, rerank, and graft state exactly as the
upstream artifacts reported it.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4153 as exp4153


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "cached_gap4_replay_and_ledger_reconciliation"

EXP4163_ARTIFACT_PATH = "results/experiment_4163_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4153.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4153.ARC1_PROGRAMS_PATH
EXP4157_PATH = "results/experiment_4157_baseline_harvest_contiguous_continue.json"
EXP4158_PATH = "results/experiment_4158_verifier_rerank_recovery_moat.json"
EXP4159_PATH = "results/experiment_4159_decisive_verifier_reward_graft.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
SUDOKU_BASELINE_GAP_ID = "GAP-SUDOKU-BASELINE-REPRODUCTION-4157"
SUDOKU_RERANK_GAP_ID = "GAP-SUDOKU-RERANK-RECOVERY-MOAT-4158"
SUDOKU_GRAFT_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4159"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4159"
SUDOKU_RERANK_ROLE_ID = "sudoku_executable_verifier_rerank_time_4158"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "sudoku_baseline",
    "sudoku_rerank_moat",
    "sudoku_decisive_graft",
    "diffusiongemma_gate_state",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .385 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched so the gap backlog stays the honest complement "
        "of the registry."
    ),
}


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _check_json_resource(repo_root: Path, resource: str, rel_path: str) -> dict[str, Any]:
    path = repo_root / rel_path
    if not path.exists():
        return {"resource": resource, "available": False, "detail": f"missing: {rel_path}"}
    try:
        loaded = base._load_json(path)
    except Exception as exc:  # pragma: no cover - exact JSON errors depend on parser version.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    if not isinstance(loaded, dict):
        return {"resource": resource, "available": False, "detail": "not_json_object"}
    return {"resource": resource, "available": True, "detail": rel_path}


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4163: verify cached fixtures, upstream artifacts, and ledgers."""
    base_preflight = exp4153.check_preconditions(repo_root)
    checks = list(base_preflight["checks"]) + [
        _check_json_resource(repo_root, "exp4157_baseline", EXP4157_PATH),
        _check_json_resource(repo_root, "exp4158_rerank_moat", EXP4158_PATH),
        _check_json_resource(repo_root, "exp4159_decisive_graft", EXP4159_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4163: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4153.replay_gap4_arc1(repo_root)


def _trajectory_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("val_trajectory", [])
    if not isinstance(rows, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        val = _numeric_or_none(row.get("val_exact_accuracy"))
        cleaned.append(
            {
                "index": index,
                "csv_version": row.get("csv_version"),
                "epoch": row.get("epoch"),
                "step": row.get("step"),
                "delta_vs_previous": _numeric_or_none(row.get("delta_vs_previous")),
                "val_exact_accuracy": val,
                "val_exact_accuracy_rounded": _round4(val),
            }
        )
    return cleaned


def _append_flag(status: str, flagged: bool) -> str:
    return f"{status}_flagged" if flagged else status


def classify_sudoku_baseline(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4163: summarize the Exp 4157 baseline trajectory honestly."""
    artifact = base._load_json(repo_root / EXP4157_PATH)
    rows = _trajectory_rows(artifact)
    current_val = _numeric_or_none(artifact.get("current_val"))
    if current_val is None and rows:
        current_val = rows[-1]["val_exact_accuracy"]
    max_val = _numeric_or_none(artifact.get("max_val"))
    if max_val is None and rows:
        measured = [row["val_exact_accuracy"] for row in rows if row["val_exact_accuracy"] is not None]
        max_val = max(measured) if measured else None

    faithful = artifact.get("baseline_faithful") is True
    flagged = artifact.get("flagged_adversarial") is True
    honest_verdict = str(artifact.get("honest_verdict", ""))
    val_text = f"{current_val:.4f}" if current_val is not None else "unknown"
    if faithful:
        status = f"reproduced_val_{val_text}"
    elif honest_verdict.startswith("blocked_noop_step_unchanged"):
        status = f"open_baseline_blocked_noop_step_unchanged_val_{val_text}"
    else:
        status = f"open_baseline_not_faithful_val_{val_text}"

    return {
        "gap_id": SUDOKU_BASELINE_GAP_ID,
        "status": _append_flag(status, flagged and not faithful),
        "artifact_path": EXP4157_PATH,
        "source_artifacts": [EXP4157_PATH],
        "honest_verdict": honest_verdict,
        "baseline_faithful": faithful,
        "current_val": current_val,
        "current_val_rounded": _round4(current_val),
        "max_val": max_val,
        "max_val_rounded": _round4(max_val),
        "run_alive": artifact.get("run_alive") is True,
        "manual_lr_step": artifact.get("manual_lr_step"),
        "native_trainer_launched": artifact.get("native_trainer_launched") is True,
        "blocked_cause": str(artifact.get("blocked_cause", "")),
        "estimated_passes_to_085": dict(artifact.get("estimated_passes_to_085", {})),
        "val_trajectory_385": rows,
        "val_trajectory_385_rounded": [row["val_exact_accuracy_rounded"] for row in rows],
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "flagged_adversarial": flagged,
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "missing_discriminator": (
            "faithful_sudoku_baseline_candidate_source_before_diffusiongemma_scaleup"
        ),
    }


def _compact_metric(metric: Any) -> dict[str, Any]:
    if not isinstance(metric, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in metric.items():
        if key == "per_puzzle" and isinstance(value, list):
            compact["per_puzzle_count"] = len(value)
        elif key != "per_puzzle":
            compact[key] = value
    return compact


def _ci_excludes_zero_positive(metric: dict[str, Any]) -> bool:
    delta = _numeric_or_none(metric.get("delta"))
    ci = metric.get("ci95")
    if delta is None or not isinstance(ci, list) or len(ci) != 2:
        return False
    lower = _numeric_or_none(ci[0])
    upper = _numeric_or_none(ci[1])
    return lower is not None and upper is not None and lower > 0.0 and delta > 0.0


def classify_sudoku_rerank_moat(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4163: summarize Exp 4158 rerank recovery without laundering nulls."""
    artifact = base._load_json(repo_root / EXP4158_PATH)
    rerank_lift = _compact_metric(artifact.get("rerank_lift_vs_vote", {}))
    flagged = artifact.get("flagged_adversarial") is True
    headroom_present = artifact.get("headroom_present") is True
    ci_positive = _ci_excludes_zero_positive(rerank_lift)
    if ci_positive:
        status = "filled_rerank_recovery_moat"
    elif not headroom_present:
        status = "open_rerank_uninformative_no_headroom"
    else:
        status = "open_honest_null_ci_includes_zero"

    return {
        "gap_id": SUDOKU_RERANK_GAP_ID,
        "status": _append_flag(status, flagged and not ci_positive),
        "artifact_path": EXP4158_PATH,
        "source_artifacts": [EXP4158_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "headroom_present": headroom_present,
        "ci_excludes_zero_positive": ci_positive,
        "verifier_recovers_outvoted": artifact.get("verifier_recovers_outvoted"),
        "vote_at_1": _numeric_or_none(artifact.get("vote_at_1")),
        "oracle_at_k": _numeric_or_none(artifact.get("oracle_at_k")),
        "n_candidate_pools": artifact.get("n_candidate_pools"),
        "k_candidates": artifact.get("k_candidates"),
        "rerank_lift_vs_vote": rerank_lift,
        "cost_ratio_vs_llm_judge": dict(artifact.get("cost_ratio_vs_llm_judge", {})),
        "baseline_status": dict(artifact.get("baseline_status", {})),
        "candidate_source": str(artifact.get("candidate_source", "")),
        "snapshot_checkpoint_path": str(artifact.get("snapshot_checkpoint_path", "")),
        "flagged_adversarial": flagged,
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "missing_discriminator": (
            "decision_grade_executable_sudoku_rerank_signal_with_selectable_headroom"
        ),
    }


def classify_sudoku_decisive_graft(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4163: summarize Exp 4159 graft value-added or deferral."""
    artifact = base._load_json(repo_root / EXP4159_PATH)
    flagged = artifact.get("flagged_adversarial") is True
    graft_deferred = artifact.get("graft_deferred") is True
    verifier_value_added = artifact.get("verifier_value_added") is True
    if verifier_value_added:
        status = "filled_training_time_verifier_value_added"
    elif graft_deferred:
        status = "open_graft_deferred_baseline_below_0.85"
    else:
        status = "open_honest_null_no_transferable_value_added"

    return {
        "gap_id": SUDOKU_GRAFT_GAP_ID,
        "status": _append_flag(status, flagged and not verifier_value_added),
        "artifact_path": EXP4159_PATH,
        "source_artifacts": [EXP4159_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "graft_deferred": graft_deferred,
        "verifier_value_added": verifier_value_added,
        "current_val": _numeric_or_none(artifact.get("current_val")),
        "current_val_rounded": _round4(_numeric_or_none(artifact.get("current_val"))),
        "baseline_status": dict(artifact.get("baseline_status", {})),
        "candidate_source": str(artifact.get("candidate_source", "")),
        "n_candidate_pools": artifact.get("n_candidate_pools"),
        "phase0_precision": dict(artifact.get("phase0_precision", {})),
        "rft_vs_ablation_delta": _compact_metric(artifact.get("rft_vs_ablation_delta", {})),
        "estimated_passes_to_converge_for_386": dict(
            artifact.get("estimated_passes_to_converge_for_386", {})
        ),
        "rft_training_mode": str(artifact.get("rft_training_mode", "")),
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "snapshot_checkpoint_path": artifact.get("snapshot_checkpoint_path"),
        "flagged_adversarial": flagged,
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "missing_discriminator": (
            "transferable_training_time_value_from_non_oracle_sudoku_verifier_labels"
        ),
    }


def classify_diffusiongemma_gate(
    sudoku_rerank: dict[str, Any],
    sudoku_graft: dict[str, Any],
) -> dict[str, Any]:
    """Return the DiffusionGemma scale-up gate from rerank/RFT value only."""
    rerank_positive = bool(sudoku_rerank.get("ci_excludes_zero_positive"))
    value_added = bool(sudoku_graft.get("verifier_value_added"))
    graft_deferred = bool(sudoku_graft.get("graft_deferred"))
    if value_added:
        return {
            "state": "unlocked_by_training_time_value_added",
            "reason": "training_time_verifier_value_added",
            "moved_by_rerank_signal": False,
            "rerank_ci_excludes_zero_positive": rerank_positive,
            "verifier_value_added": True,
            "graft_deferred": graft_deferred,
            "uses_executable_oracle_upper_bound": False,
            "basis": "exp4158_rerank_lift_vs_vote_or_exp4159_verifier_value_added",
        }
    if rerank_positive:
        return {
            "state": "unlocked_by_rerank_discrimination",
            "reason": "rerank_lift_ci_excludes_zero",
            "moved_by_rerank_signal": True,
            "rerank_ci_excludes_zero_positive": True,
            "verifier_value_added": False,
            "graft_deferred": graft_deferred,
            "uses_executable_oracle_upper_bound": False,
            "basis": "exp4158_rerank_lift_vs_vote_or_exp4159_verifier_value_added",
        }
    return {
        "state": "kept_gated",
        "reason": "no_positive_rerank_signal_and_no_training_time_value_added",
        "moved_by_rerank_signal": False,
        "rerank_ci_excludes_zero_positive": False,
        "verifier_value_added": False,
        "graft_deferred": graft_deferred,
        "uses_executable_oracle_upper_bound": False,
        "basis": "exp4158_rerank_lift_vs_vote_or_exp4159_verifier_value_added",
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_rerank: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .385 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_sudoku_roles(
        updated_registry,
        sudoku_baseline,
        sudoku_rerank,
        sudoku_graft,
        diffusiongemma_gate_state,
    )

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4163-sudoku-baseline-reproduction",
        _sudoku_baseline_gap_block(sudoku_baseline),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4163-sudoku-rerank-moat",
        _sudoku_rerank_gap_block(sudoku_rerank, diffusiongemma_gate_state),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4163-sudoku-decisive-graft",
        _sudoku_graft_gap_block(sudoku_baseline, sudoku_graft, diffusiongemma_gate_state),
    )
    touched = [
        gap_id
        for gap_id in (SUDOKU_BASELINE_GAP_ID, SUDOKU_RERANK_GAP_ID, SUDOKU_GRAFT_GAP_ID)
        if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "sudoku_baseline_recorded": SUDOKU_BASELINE_GAP_ID in touched,
            "sudoku_rerank_recorded": SUDOKU_RERANK_GAP_ID in touched,
            "sudoku_graft_recorded": SUDOKU_GRAFT_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4163": EXP4163_ARTIFACT_PATH,
            "exp4163_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4163_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4163_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4163_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4163_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _rerank_role_status(sudoku_rerank: dict[str, Any]) -> str:
    flagged = bool(sudoku_rerank.get("flagged_adversarial"))
    if bool(sudoku_rerank.get("ci_excludes_zero_positive")):
        return "candidate_rerank_signal"
    if not bool(sudoku_rerank.get("headroom_present")):
        return _append_flag("uninformative_no_headroom", flagged)
    return _append_flag("honest_null_ci_includes_zero", flagged)


def _training_role_status(sudoku_graft: dict[str, Any]) -> str:
    flagged = bool(sudoku_graft.get("flagged_adversarial"))
    if bool(sudoku_graft.get("verifier_value_added")):
        return "value_added_diffusiongemma_unlocked"
    if bool(sudoku_graft.get("graft_deferred")):
        return _append_flag("graft_deferred_baseline_below_0.85", flagged)
    return _append_flag("honest_null_no_transferable_value_added", flagged)


def _ensure_sudoku_roles(
    registry: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_rerank: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    rerank_status = _rerank_role_status(sudoku_rerank)
    training_status = _training_role_status(sudoku_graft)
    if rerank_status == "candidate_rerank_signal":
        combined_status = rerank_status
    elif training_status == "value_added_diffusiongemma_unlocked":
        combined_status = "candidate_training_time_value_added"
    else:
        combined_status = rerank_status

    entry["role_sudoku_executable"] = {
        "status": combined_status,
        "training_time_status": training_status,
        "rerank_time_status": rerank_status,
        "promoted_toward_candidate": combined_status.startswith("candidate"),
        "eval_exp_4163": EXP4163_ARTIFACT_PATH,
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
    }

    training_role = {
        "role_id": SUDOKU_TRAINING_ROLE_ID,
        "experiment": EXP4159_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": training_status,
        "outcome": sudoku_graft.get("status"),
        "honest_verdict": sudoku_graft.get("honest_verdict", ""),
        "baseline_artifact": EXP4157_PATH,
        "baseline_current_val": sudoku_baseline.get("current_val"),
        "baseline_current_val_rounded": sudoku_baseline.get("current_val_rounded"),
        "baseline_faithful": bool(sudoku_baseline.get("baseline_faithful")),
        "baseline_val_trajectory_385_rounded": sudoku_baseline.get(
            "val_trajectory_385_rounded"
        ),
        "graft_deferred": bool(sudoku_graft.get("graft_deferred")),
        "verifier_value_added": bool(sudoku_graft.get("verifier_value_added")),
        "phase0_precision": sudoku_graft.get("phase0_precision", {}),
        "rft_vs_ablation_delta": sudoku_graft.get("rft_vs_ablation_delta", {}),
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
        "flagged_adversarial": bool(sudoku_graft.get("flagged_adversarial")),
        "eval_exp_4163": EXP4163_ARTIFACT_PATH,
    }
    old_training = list(entry.get("training_time_roles", []))
    entry["training_time_roles"] = [
        role for role in old_training if role.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [training_role]

    rerank_role = {
        "role_id": SUDOKU_RERANK_ROLE_ID,
        "experiment": EXP4158_PATH,
        "role": "candidate_trm_rerank_time_executable_sudoku_verifier",
        "status": rerank_status,
        "outcome": sudoku_rerank.get("status"),
        "honest_verdict": sudoku_rerank.get("honest_verdict", ""),
        "baseline_artifact": EXP4157_PATH,
        "headroom_present": bool(sudoku_rerank.get("headroom_present")),
        "ci_excludes_zero_positive": bool(sudoku_rerank.get("ci_excludes_zero_positive")),
        "verifier_recovers_outvoted": sudoku_rerank.get("verifier_recovers_outvoted"),
        "vote_at_1": sudoku_rerank.get("vote_at_1"),
        "oracle_at_k": sudoku_rerank.get("oracle_at_k"),
        "rerank_lift_vs_vote": sudoku_rerank.get("rerank_lift_vs_vote", {}),
        "cost_ratio_vs_llm_judge": sudoku_rerank.get("cost_ratio_vs_llm_judge", {}),
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
        "flagged_adversarial": bool(sudoku_rerank.get("flagged_adversarial")),
        "eval_exp_4163": EXP4163_ARTIFACT_PATH,
    }
    old_rerank = list(entry.get("rerank_time_roles", []))
    entry["rerank_time_roles"] = [
        role for role in old_rerank if role.get("role_id") != SUDOKU_RERANK_ROLE_ID
    ] + [rerank_role]


def _sudoku_baseline_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SUDOKU_BASELINE_GAP_ID}: Exp 4163 .385 Sudoku baseline trajectory status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4157_PATH}`; "
        f"honest_verdict={outcome.get('honest_verdict')}; "
        f"current_val={outcome.get('current_val_rounded')}; "
        f"max_val={outcome.get('max_val_rounded')}; "
        f"baseline_faithful={str(bool(outcome.get('baseline_faithful'))).lower()}; "
        f"val_trajectory_385={outcome.get('val_trajectory_385_rounded')}; "
        f"native_trainer_launched={str(bool(outcome.get('native_trainer_launched'))).lower()}; "
        f"flagged_adversarial={str(bool(outcome.get('flagged_adversarial'))).lower()}.\n"
        "- failure mode: the .385 continuation advanced the visible validation "
        "trajectory to about 0.5010 but still did not reach the faithful 0.85 gate, "
        "and the source artifact carries flagged caveats plus a blocked/no-op/OOM "
        "verdict.\n"
        "- missing discriminator: faithful Sudoku baseline candidate source before "
        "DiffusionGemma scale-up or verifier-as-reward claims.\n"
        "- candidate design: continue or relaunch the baseline under a clean resource "
        "envelope, then rerun rerank/graft only after the candidate source has "
        "faithful accuracy and selectable headroom.\n"
        "- priority: high\n"
    )


def _sudoku_rerank_gap_block(
    sudoku_rerank: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> str:
    rerank = sudoku_rerank.get("rerank_lift_vs_vote", {})
    return (
        f"### {SUDOKU_RERANK_GAP_ID}: Exp 4163 .385 Sudoku executable-verifier rerank moat status\n"
        f"- status: {sudoku_rerank['status']}\n"
        f"- evidence: `{EXP4158_PATH}`; "
        f"headroom_present={str(bool(sudoku_rerank.get('headroom_present'))).lower()}; "
        f"oracle_at_k={sudoku_rerank.get('oracle_at_k')}; "
        f"vote_at_1={sudoku_rerank.get('vote_at_1')}; "
        f"verifier_recovers_outvoted={sudoku_rerank.get('verifier_recovers_outvoted')}; "
        f"rerank_lift_vs_vote_delta={rerank.get('delta')}; "
        f"rerank_lift_vs_vote_ci95={rerank.get('ci95')}; "
        f"ci_excludes_zero_positive={str(bool(sudoku_rerank.get('ci_excludes_zero_positive'))).lower()}; "
        f"flagged_adversarial={str(bool(sudoku_rerank.get('flagged_adversarial'))).lower()}; "
        f"diffusiongemma_gate_state={diffusiongemma_gate_state.get('state')}.\n"
        "- failure mode: Exp 4158 reported no selectable rerank headroom and zero "
        "outvoted recoveries, so the executable checker did not produce a "
        "decision-grade rerank moat signal.\n"
        "- missing discriminator: decision-grade executable Sudoku rerank signal with "
        "selectable headroom and a positive CI excluding zero.\n"
        "- candidate design: rerun on a faithful checkpoint/pool where oracle@K exceeds "
        "vote@1, then promote only if rerank lift has a positive confidence interval.\n"
        "- priority: high\n"
    )


def _sudoku_graft_gap_block(
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> str:
    rft = sudoku_graft.get("rft_vs_ablation_delta", {})
    return (
        f"### {SUDOKU_GRAFT_GAP_ID}: Exp 4163 .385 Sudoku decisive executable-verifier graft status\n"
        f"- status: {sudoku_graft['status']}\n"
        f"- evidence: `{EXP4159_PATH}` with baseline `{EXP4157_PATH}`; "
        f"baseline_current_val={sudoku_baseline.get('current_val_rounded')}; "
        f"baseline_faithful={str(bool(sudoku_baseline.get('baseline_faithful'))).lower()}; "
        f"graft_deferred={str(bool(sudoku_graft.get('graft_deferred'))).lower()}; "
        f"candidate_source={sudoku_graft.get('candidate_source')}; "
        f"n_candidate_pools={sudoku_graft.get('n_candidate_pools')}; "
        f"rft_vs_ablation_delta={rft.get('delta')}; "
        f"rft_vs_ablation_delta_status={rft.get('status')}; "
        f"verifier_value_added={str(bool(sudoku_graft.get('verifier_value_added'))).lower()}; "
        f"flagged_adversarial={str(bool(sudoku_graft.get('flagged_adversarial'))).lower()}; "
        f"diffusiongemma_gate_state={diffusiongemma_gate_state.get('state')}.\n"
        "- failure mode: Exp 4159 deferred the training-time graft because the baseline "
        "remained below the faithful 0.85 threshold; no verifier-as-reward value-added "
        "claim is available.\n"
        "- missing discriminator: transferable training-time value from non-oracle "
        "Sudoku verifier labels beyond vote labels.\n"
        "- candidate design: keep DiffusionGemma gated until rerank or RFT A-vs-B label "
        "contrast shows value on a faithful baseline.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4163") == EXP4163_ARTIFACT_PATH
        and gap4.get("role_sudoku_executable", {}).get("eval_exp_4163")
        == EXP4163_ARTIFACT_PATH
        and any(
            role.get("role_id") == SUDOKU_TRAINING_ROLE_ID
            for role in gap4.get("training_time_roles", [])
        )
        and any(
            role.get("role_id") == SUDOKU_RERANK_ROLE_ID
            for role in gap4.get("rerank_time_roles", [])
        )
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_rerank_moat: dict[str, Any],
    sudoku_decisive_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4163 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {SUDOKU_BASELINE_GAP_ID, SUDOKU_RERANK_GAP_ID, SUDOKU_GRAFT_GAP_ID}
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4163_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4163_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v385_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"baseline_{sudoku_baseline['status']}_"
            f"rerank_{sudoku_rerank_moat['status']}_"
            f"graft_{sudoku_decisive_graft['status']}_"
            f"diffusiongemma_{diffusiongemma_gate_state.get('state')}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "sudoku_baseline": sudoku_baseline,
        "sudoku_rerank_moat": sudoku_rerank_moat,
        "sudoku_decisive_graft": sudoku_decisive_graft,
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4157_PATH,
            EXP4158_PATH,
            EXP4159_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4163", "SCENARIO-VERIFY-4163"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4163_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4163_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "sudoku_baseline": {"status": "blocked_precondition", "gap_id": SUDOKU_BASELINE_GAP_ID},
        "sudoku_rerank_moat": {
            "status": "blocked_precondition",
            "gap_id": SUDOKU_RERANK_GAP_ID,
        },
        "sudoku_decisive_graft": {
            "status": "blocked_precondition",
            "gap_id": SUDOKU_GRAFT_GAP_ID,
        },
        "diffusiongemma_gate_state": {
            "state": "blocked",
            "reason": blocked,
            "moved_by_rerank_signal": False,
            "rerank_ci_excludes_zero_positive": False,
            "verifier_value_added": False,
            "graft_deferred": False,
            "uses_executable_oracle_upper_bound": False,
            "basis": "precondition_failed_before_replay",
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "preconditions": preflight,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4157_PATH,
            EXP4158_PATH,
            EXP4159_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4163", "SCENARIO-VERIFY-4163"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4163 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a bare bool")
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")
    if not isinstance(artifact["gaps_updated"], list):
        raise ValueError("gaps_updated must be a list")
    gate = artifact["diffusiongemma_gate_state"]
    if not isinstance(gate, dict) or not gate.get("state"):
        raise ValueError("diffusiongemma_gate_state must include a state")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4163 principles")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4163 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4163_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    sudoku_baseline = classify_sudoku_baseline(repo_root)
    sudoku_rerank = classify_sudoku_rerank_moat(repo_root)
    sudoku_graft = classify_sudoku_decisive_graft(repo_root)
    diffusiongemma_gate_state = classify_diffusiongemma_gate(sudoku_rerank, sudoku_graft)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        sudoku_baseline,
        sudoku_rerank,
        sudoku_graft,
        diffusiongemma_gate_state,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        sudoku_baseline=sudoku_baseline,
        sudoku_rerank_moat=sudoku_rerank,
        sudoku_decisive_graft=sudoku_graft,
        diffusiongemma_gate_state=diffusiongemma_gate_state,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4163_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
