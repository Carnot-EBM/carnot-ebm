"""Exp 4171 registry/gaps hygiene for .386 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4171, SCENARIO-VERIFY-4171.

This runner is a ledger reconciler. It replays the frozen GAP-4 ARC-1 guard
from cached artifacts and records the .386 outer-loop baseline status plus the
Exp 4168 defensive graft deferral. It does not run Codex, load GGUF models,
launch training, stop training, copy the stable checkpoint, or write the stable
checkpoint.
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

EXP4171_ARTIFACT_PATH = "results/experiment_4171_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4153.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4153.ARC1_PROGRAMS_PATH
EXP4167_PATH = "results/experiment_4167_outerloop_training_monitor.json"
EXP4168_PATH = "results/experiment_4168_decisive_verifier_graft_defensive.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
SUDOKU_BASELINE_GAP_ID = "GAP-SUDOKU-BASELINE-REPRODUCTION-4167"
SUDOKU_GRAFT_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4168"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4168"
FAITHFUL_GRAFT_THRESHOLD = 0.85

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "sudoku_baseline",
    "sudoku_decisive_graft",
    "diffusiongemma_gate_state",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .386 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": "Lists the verifier_gaps entries touched.",
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
    except Exception as exc:  # pragma: no cover - JSON parser details are host/version-specific.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    if not isinstance(loaded, dict):  # pragma: no cover - checked defensively for corrupted artifacts.
        return {"resource": resource, "available": False, "detail": "not_json_object"}
    return {"resource": resource, "available": True, "detail": rel_path}


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4171: verify cached fixtures, upstream artifacts, and ledgers."""
    base_preflight = exp4153.check_preconditions(repo_root)
    checks = list(base_preflight["checks"]) + [
        _check_json_resource(repo_root, "exp4167_outerloop_monitor", EXP4167_PATH),
        _check_json_resource(repo_root, "exp4168_decisive_graft", EXP4168_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4171: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4153.replay_gap4_arc1(repo_root)


def _trajectory_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("val_trajectory", [])
    if not isinstance(rows, list):  # pragma: no cover - corrupted upstream artifact guard.
        return []
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):  # pragma: no cover - corrupted upstream artifact guard.
            continue
        val = _numeric_or_none(row.get("val_exact_accuracy"))
        cleaned.append(
            {
                "index": index,
                "csv_version": row.get("csv_version"),
                "epoch": row.get("epoch"),
                "step": row.get("step"),
                "row_number": row.get("row_number"),
                "metrics_path": str(row.get("metrics_path", "")),
                "delta_vs_previous": _numeric_or_none(row.get("delta_vs_previous")),
                "val_exact_accuracy": val,
                "val_exact_accuracy_rounded": _round4(val),
            }
        )
    return cleaned


def _baseline_status(artifact: dict[str, Any], current_val: float | None) -> str:
    val_text = "unknown" if current_val is None else f"{current_val:.4f}"
    if artifact.get("baseline_faithful") is True:  # pragma: no cover - not the .386 observed state.
        return f"faithful_stable_val_{val_text}"
    if artifact.get("outerloop_train_alive") is True:
        return f"open_outerloop_training_alive_val_{val_text}"
    return f"open_baseline_below_0.85_val_{val_text}"  # pragma: no cover - not the .386 observed state.


def classify_sudoku_baseline(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4171: summarize the Exp 4167 outer-loop trajectory honestly."""
    artifact = base._load_json(repo_root / EXP4167_PATH)
    rows = _trajectory_rows(artifact)
    current_val = _numeric_or_none(artifact.get("current_val_exact_accuracy"))
    if current_val is None and rows:  # pragma: no cover - fallback for future schema drift.
        current_val = rows[-1]["val_exact_accuracy"]
    measured = [row["val_exact_accuracy"] for row in rows if row["val_exact_accuracy"] is not None]
    max_val = max(measured) if measured else None

    read_only = artifact.get("read_only_actions", {})
    return {
        "gap_id": SUDOKU_BASELINE_GAP_ID,
        "status": _baseline_status(artifact, current_val),
        "artifact_path": EXP4167_PATH,
        "source_artifacts": [EXP4167_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "baseline_faithful": artifact.get("baseline_faithful") is True,
        "faithful_threshold": _numeric_or_none(artifact.get("faithful_threshold")),
        "outerloop_train_alive": artifact.get("outerloop_train_alive") is True,
        "outerloop_pid": artifact.get("outerloop_pid"),
        "outerloop_pid_etime": artifact.get("outerloop_pid_etime"),
        "current_val_exact_accuracy": current_val,
        "current_val_exact_accuracy_rounded": _round4(current_val),
        "max_val_exact_accuracy": max_val,
        "max_val_exact_accuracy_rounded": _round4(max_val),
        "checkpoint_epoch": artifact.get("checkpoint_epoch"),
        "checkpoint_mtime": artifact.get("checkpoint_mtime"),
        "checkpoint_path": str(artifact.get("checkpoint_path", "")),
        "latest_metrics_path": str(artifact.get("latest_metrics_path", "")),
        "read_only_actions": dict(read_only) if isinstance(read_only, dict) else {},
        "val_trajectory_386": rows,
        "val_trajectory_386_rounded": [row["val_exact_accuracy_rounded"] for row in rows],
        "preconditions_checked": [
            "exp4167_json_read_only",
            "no_trm_training_launched_by_exp4171",
            "no_stable_checkpoint_write_by_exp4171",
        ],
        "missing_discriminator": (
            "faithful_outerloop_sudoku_baseline_candidate_source_before_diffusiongemma_scaleup"
        ),
    }


def _compact_metric(metric: Any) -> dict[str, Any]:
    if not isinstance(metric, dict):  # pragma: no cover - corrupted upstream artifact guard.
        return {}
    compact: dict[str, Any] = {}
    for key, value in metric.items():
        if key == "per_puzzle" and isinstance(value, list):  # pragma: no cover - not present in Exp 4168.
            compact["per_puzzle_count"] = len(value)
        elif key != "per_puzzle":
            compact[key] = value
    return compact


def _normalized_baseline_status(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict):  # pragma: no cover - corrupted upstream artifact guard.
        return {}
    normalized = dict(status)
    current_val = _numeric_or_none(normalized.get("current_val_exact_accuracy"))
    normalized["current_val_exact_accuracy_rounded"] = _round4(current_val)
    return normalized


def _graft_status(artifact: dict[str, Any], baseline_status: dict[str, Any]) -> str:
    current_val = _numeric_or_none(baseline_status.get("current_val_exact_accuracy"))
    val_text = "unknown" if current_val is None else f"{current_val:.4f}"
    if artifact.get("verifier_value_added") is True:  # pragma: no cover - not the .386 observed state.
        return f"filled_training_time_verifier_value_added_val_{val_text}"
    if artifact.get("graft_deferred") is True:
        return f"open_graft_deferred_outerloop_training_val_{val_text}"
    return f"open_honest_null_no_transferable_value_added_val_{val_text}"  # pragma: no cover


def classify_sudoku_decisive_graft(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4171: summarize Exp 4168 graft value-added or deferral."""
    artifact = base._load_json(repo_root / EXP4168_PATH)
    baseline_status = _normalized_baseline_status(artifact.get("baseline_status", {}))
    read_only = artifact.get("read_only_actions", {})
    return {
        "gap_id": SUDOKU_GRAFT_GAP_ID,
        "status": _graft_status(artifact, baseline_status),
        "artifact_path": EXP4168_PATH,
        "source_artifacts": [EXP4168_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "acceptance_gate_passed": artifact.get("acceptance_gate_passed") is True,
        "graft_deferred": artifact.get("graft_deferred") is True,
        "verifier_value_added": artifact.get("verifier_value_added") is True,
        "baseline_status": baseline_status,
        "candidate_source": str(artifact.get("candidate_source", "")),
        "n_candidate_pools": artifact.get("n_candidate_pools"),
        "checkpoint_copy_path": artifact.get("checkpoint_copy_path"),
        "checkpoint_copy_performed": artifact.get("checkpoint_copy_performed") is True,
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "read_only_actions": dict(read_only) if isinstance(read_only, dict) else {},
        "rerank_lift_vs_vote": _compact_metric(artifact.get("rerank_lift_vs_vote", {})),
        "rft_vs_ablation_delta": _compact_metric(artifact.get("rft_vs_ablation_delta", {})),
        "corpus_summary": dict(artifact.get("corpus_summary", {})),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "missing_discriminator": (
            "transferable_training_time_value_from_non_oracle_sudoku_verifier_labels"
        ),
    }


def _ci_excludes_zero_positive(metric: dict[str, Any]) -> bool:
    delta = _numeric_or_none(metric.get("delta"))
    ci = metric.get("ci95")
    if delta is None or not isinstance(ci, list) or len(ci) != 2:
        return False
    lower = _numeric_or_none(ci[0])
    return lower is not None and lower > 0.0 and delta > 0.0


def classify_diffusiongemma_gate(sudoku_graft: dict[str, Any]) -> dict[str, Any]:
    """Return the DiffusionGemma scale-up gate from Exp 4168 value only."""
    rerank_positive = _ci_excludes_zero_positive(
        sudoku_graft.get("rerank_lift_vs_vote", {})
        if isinstance(sudoku_graft.get("rerank_lift_vs_vote"), dict)
        else {}
    )
    value_added = bool(sudoku_graft.get("verifier_value_added"))
    graft_deferred = bool(sudoku_graft.get("graft_deferred"))
    if value_added:
        return {
            "state": "unlocked_by_training_time_value_added",
            "reason": "training_time_verifier_value_added",
            "rerank_ci_excludes_zero_positive": rerank_positive,
            "verifier_value_added": True,
            "graft_deferred": graft_deferred,
            "uses_executable_oracle_upper_bound": False,
            "basis": "exp4168_rerank_lift_vs_vote_or_verifier_value_added",
        }
    if rerank_positive:
        return {
            "state": "unlocked_by_rerank_discrimination",
            "reason": "rerank_lift_ci_excludes_zero",
            "rerank_ci_excludes_zero_positive": True,
            "verifier_value_added": False,
            "graft_deferred": graft_deferred,
            "uses_executable_oracle_upper_bound": False,
            "basis": "exp4168_rerank_lift_vs_vote_or_verifier_value_added",
        }
    return {
        "state": "kept_gated",
        "reason": "no_positive_rerank_signal_and_no_training_time_value_added",
        "rerank_ci_excludes_zero_positive": False,
        "verifier_value_added": False,
        "graft_deferred": graft_deferred,
        "uses_executable_oracle_upper_bound": False,
        "basis": "exp4168_rerank_lift_vs_vote_or_verifier_value_added",
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .386 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_sudoku_role(
        updated_registry,
        sudoku_baseline,
        sudoku_graft,
        diffusiongemma_gate_state,
    )

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4171-sudoku-baseline-reproduction",
        _sudoku_baseline_gap_block(sudoku_baseline),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4171-sudoku-decisive-graft",
        _sudoku_graft_gap_block(sudoku_baseline, sudoku_graft, diffusiongemma_gate_state),
    )
    touched = [
        gap_id
        for gap_id in (SUDOKU_BASELINE_GAP_ID, SUDOKU_GRAFT_GAP_ID)
        if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "sudoku_baseline_recorded": SUDOKU_BASELINE_GAP_ID in touched,
            "sudoku_graft_recorded": SUDOKU_GRAFT_GAP_ID in touched,
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - real/minimal registries include the GAP-4 entry.
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4171": EXP4171_ARTIFACT_PATH,
            "exp4171_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4171_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4171_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4171_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4171_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _training_role_status(sudoku_graft: dict[str, Any]) -> str:
    if bool(sudoku_graft.get("verifier_value_added")):
        return "value_added_diffusiongemma_unlocked"
    if bool(sudoku_graft.get("graft_deferred")):
        return "graft_deferred_outerloop_training"
    return "honest_null_no_transferable_value_added"


def _ensure_sudoku_role(
    registry: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - guarded by _ensure_gap4_eval.
        return
    training_status = _training_role_status(sudoku_graft)

    entry["role_sudoku_executable"] = {
        "status": training_status,
        "training_time_status": training_status,
        "promoted_toward_candidate": training_status.startswith("value_added"),
        "eval_exp_4171": EXP4171_ARTIFACT_PATH,
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
    }

    training_role = {
        "role_id": SUDOKU_TRAINING_ROLE_ID,
        "experiment": EXP4168_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": training_status,
        "outcome": sudoku_graft.get("status"),
        "honest_verdict": sudoku_graft.get("honest_verdict", ""),
        "baseline_artifact": EXP4167_PATH,
        "baseline_current_val": sudoku_baseline.get("current_val_exact_accuracy"),
        "baseline_current_val_rounded": sudoku_baseline.get(
            "current_val_exact_accuracy_rounded"
        ),
        "baseline_faithful": bool(sudoku_baseline.get("baseline_faithful")),
        "outerloop_train_alive": bool(sudoku_baseline.get("outerloop_train_alive")),
        "graft_baseline_current_val": sudoku_graft.get("baseline_status", {}).get(
            "current_val_exact_accuracy"
        ),
        "graft_baseline_current_val_rounded": sudoku_graft.get("baseline_status", {}).get(
            "current_val_exact_accuracy_rounded"
        ),
        "graft_deferred": bool(sudoku_graft.get("graft_deferred")),
        "verifier_value_added": bool(sudoku_graft.get("verifier_value_added")),
        "checkpoint_copy_performed": bool(sudoku_graft.get("checkpoint_copy_performed")),
        "rerank_lift_vs_vote": sudoku_graft.get("rerank_lift_vs_vote", {}),
        "rft_vs_ablation_delta": sudoku_graft.get("rft_vs_ablation_delta", {}),
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
        "eval_exp_4171": EXP4171_ARTIFACT_PATH,
    }
    old_training = list(entry.get("training_time_roles", []))
    entry["training_time_roles"] = [
        role for role in old_training if role.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [training_role]


def _sudoku_baseline_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SUDOKU_BASELINE_GAP_ID}: Exp 4171 .386 outer-loop Sudoku baseline trajectory status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4167_PATH}`; "
        f"honest_verdict={outcome.get('honest_verdict')}; "
        f"current_val={outcome.get('current_val_exact_accuracy_rounded')}; "
        f"max_val={outcome.get('max_val_exact_accuracy_rounded')}; "
        f"baseline_faithful={str(bool(outcome.get('baseline_faithful'))).lower()}; "
        f"outerloop_train_alive={str(bool(outcome.get('outerloop_train_alive'))).lower()}; "
        f"checkpoint_mtime={outcome.get('checkpoint_mtime')}; "
        f"val_trajectory_386_rounded={outcome.get('val_trajectory_386_rounded')}; "
        f"training_launched={str(bool(outcome.get('read_only_actions', {}).get('training_launched'))).lower()}; "
        f"train_process_stop_attempted={str(bool(outcome.get('read_only_actions', {}).get('train_process_stop_attempted'))).lower()}; "
        f"stable_checkpoint_written={str(bool(outcome.get('read_only_actions', {}).get('stable_checkpoint_written'))).lower()}.\n"
        "- failure mode: the outer-loop baseline is still below the faithful 0.85 "
        "gate, so verifier-graft claims remain deferred even though validation "
        "progress continued into the .386 window.\n"
        "- missing discriminator: faithful outer-loop Sudoku baseline candidate "
        "source before DiffusionGemma scale-up or verifier-as-reward claims.\n"
        "- candidate design: keep the outer-loop run owner authoritative, continue "
        "monitoring read-only status, and rerun graft only after the checkpoint is "
        "faithful and stable.\n"
        "- priority: high\n"
    )


def _sudoku_graft_gap_block(
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> str:
    rft = sudoku_graft.get("rft_vs_ablation_delta", {})
    rerank = sudoku_graft.get("rerank_lift_vs_vote", {})
    graft_baseline = sudoku_graft.get("baseline_status", {})
    return (
        f"### {SUDOKU_GRAFT_GAP_ID}: Exp 4171 .386 defensive executable-verifier graft status\n"
        f"- status: {sudoku_graft['status']}\n"
        f"- evidence: `{EXP4168_PATH}` with monitor `{EXP4167_PATH}`; "
        f"outerloop_monitor_current_val={sudoku_baseline.get('current_val_exact_accuracy_rounded')}; "
        f"baseline_current_val={graft_baseline.get('current_val_exact_accuracy_rounded')}; "
        f"baseline_faithful={str(bool(graft_baseline.get('baseline_faithful'))).lower()}; "
        f"faithful_stable={str(bool(graft_baseline.get('faithful_stable'))).lower()}; "
        f"graft_deferred={str(bool(sudoku_graft.get('graft_deferred'))).lower()}; "
        f"checkpoint_copy_performed={str(bool(sudoku_graft.get('checkpoint_copy_performed'))).lower()}; "
        f"candidate_source={sudoku_graft.get('candidate_source')}; "
        f"n_candidate_pools={sudoku_graft.get('n_candidate_pools')}; "
        f"rerank_lift_vs_vote_status={rerank.get('status')}; "
        f"rft_vs_ablation_delta={rft.get('delta')}; "
        f"rft_vs_ablation_delta_status={rft.get('status')}; "
        f"verifier_value_added={str(bool(sudoku_graft.get('verifier_value_added'))).lower()}; "
        f"diffusiongemma_gate_state={diffusiongemma_gate_state.get('state')}.\n"
        "- failure mode: Exp 4168 deferred before checkpoint copy, candidate "
        "sampling, or training because the baseline was not faithful and stable; "
        "no verifier-as-reward value-added claim is available.\n"
        "- missing discriminator: transferable training-time value from non-oracle "
        "Sudoku verifier labels beyond vote labels.\n"
        "- candidate design: keep DiffusionGemma gated until rerank or RFT A-vs-B "
        "label contrast shows value on a faithful stable baseline copy.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4171") == EXP4171_ARTIFACT_PATH
        and gap4.get("role_sudoku_executable", {}).get("eval_exp_4171")
        == EXP4171_ARTIFACT_PATH
        and any(
            role.get("role_id") == SUDOKU_TRAINING_ROLE_ID
            for role in gap4.get("training_time_roles", [])
        )
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_decisive_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4171 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {SUDOKU_BASELINE_GAP_ID, SUDOKU_GRAFT_GAP_ID}
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4171_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4171_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v386_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"baseline_{sudoku_baseline['status']}_"
            f"graft_{sudoku_decisive_graft['status']}_"
            f"diffusiongemma_{diffusiongemma_gate_state.get('state')}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "sudoku_baseline": sudoku_baseline,
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
            EXP4167_PATH,
            EXP4168_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4171", "SCENARIO-VERIFY-4171"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:  # pragma: no cover
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4171_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4171_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "sudoku_baseline": {"status": "blocked_precondition", "gap_id": SUDOKU_BASELINE_GAP_ID},
        "sudoku_decisive_graft": {
            "status": "blocked_precondition",
            "gap_id": SUDOKU_GRAFT_GAP_ID,
        },
        "diffusiongemma_gate_state": {
            "state": "blocked",
            "reason": blocked,
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
            EXP4167_PATH,
            EXP4168_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4171", "SCENARIO-VERIFY-4171"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4171 fields before writing the artifact."""
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
        raise ValueError("field_principles must match the required Exp 4171 principles")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4171 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4171_ARTIFACT_PATH
    if not preflight["ok"]:  # pragma: no cover - success path is the required artifact path.
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    sudoku_baseline = classify_sudoku_baseline(repo_root)
    sudoku_graft = classify_sudoku_decisive_graft(repo_root)
    diffusiongemma_gate_state = classify_diffusiongemma_gate(sudoku_graft)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        sudoku_baseline,
        sudoku_graft,
        diffusiongemma_gate_state,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        sudoku_baseline=sudoku_baseline,
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
    print(f"Wrote {REPO_ROOT / EXP4171_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
