"""Exp 4131 registry/gaps hygiene for .382 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4131, SCENARIO-VERIFY-4131.

This module reconciles ledgers; it does not launch training or model inference.
The cached GAP-4 replay protects the ARC verifier headline, while the Sudoku
records keep the training-time TRM story honest: the LR-resume bug is fixed and
the baseline improved quickly, but the baseline still has not reproduced the
published Sudoku-Extreme target, so the verifier graft remains deferred.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4122 as exp4122


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "cached_gap4_replay_and_ledger_reconciliation"

EXP4131_ARTIFACT_PATH = "results/experiment_4131_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

EXP4126_PATH = "results/experiment_4126_lr_resume_correctness_fix.json"
EXP4127_PATH = "results/experiment_4127_sudoku_extreme_accumulate_fixed.json"
EXP4128_PATH = "results/experiment_4128_carnot_verifier_graft_sudoku.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
LR_RESUME_GAP_ID = "GAP-SUDOKU-LR-RESUME-FIX-4126"
SUDOKU_BASELINE_GAP_ID = "GAP-SUDOKU-BASELINE-REPRODUCTION-4127"
SUDOKU_GRAFT_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4128"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4128"
PUBLISHED_SUDOKU_TARGET = 0.87

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "lr_resume_fix",
    "sudoku_baseline",
    "sudoku_graft",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .382 truth.",
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


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4131: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4122.replay_gap4_arc1(repo_root)


def classify_lr_resume_fix(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4131: summarize whether Exp 4126 fixed LR resume continuity."""
    artifact = base._load_json(repo_root / EXP4126_PATH)
    lr_points = artifact.get("train_lr_points", [])
    first_point = lr_points[0] if isinstance(lr_points, list) and lr_points else {}
    first_lr = _numeric_or_none(artifact.get("validation_first_lr"))
    if first_lr is None and isinstance(first_point, dict):
        first_lr = _numeric_or_none(first_point.get("value"))
    fresh_lr = _numeric_or_none(artifact.get("fresh_warmup_lr"))
    continuous_flag = artifact.get("lr_continuous_across_resume") is True
    differs_from_warmup = first_lr is not None and fresh_lr is not None and first_lr != fresh_lr
    continuous = continuous_flag and differs_from_warmup
    status = "fixed_lr_resume_continuous" if continuous else "open_lr_resume_not_continuous"
    return {
        "gap_id": LR_RESUME_GAP_ID,
        "status": status,
        "artifact_path": EXP4126_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "lr_continuous_across_resume": continuous,
        "artifact_lr_continuous_across_resume": continuous_flag,
        "validation_first_lr": first_lr,
        "validation_first_lr_rounded": _round4(first_lr),
        "fresh_warmup_lr": fresh_lr,
        "prior_pass_last_lr": _numeric_or_none(artifact.get("prior_pass_last_lr")),
        "manual_lr_step_restored": artifact.get("manual_lr_step_restored"),
        "lr_rewarm_root_cause": str(artifact.get("lr_rewarm_root_cause", "")),
        "full_batch_validation_attempt": artifact.get("full_batch_validation_attempt"),
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "duration_s": _numeric_or_none(artifact.get("duration_s")),
        "missing_discriminator": (
            "faithful_lr_schedule_resume_before_training_time_verifier_claims"
        ),
    }


def classify_sudoku_baseline(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4131: summarize the fixed-LR Sudoku baseline trajectory."""
    artifact = base._load_json(repo_root / EXP4127_PATH)
    trajectory_rows = artifact.get("val_trajectory", [])
    trajectory: list[float] = []
    passes: list[dict[str, Any]] = []
    if isinstance(trajectory_rows, list):
        for index, row in enumerate(trajectory_rows):
            if not isinstance(row, dict):
                continue
            val = _numeric_or_none(row.get("val_exact_accuracy"))
            if val is not None:
                trajectory.append(val)
            passes.append(
                {
                    "pass_index": row.get("pass_index", index),
                    "kind": str(row.get("kind", "")),
                    "source": row.get("source"),
                    "val_exact_accuracy": val,
                    "val_exact_accuracy_rounded": _round4(val),
                    "delta_vs_previous": _numeric_or_none(row.get("delta_vs_previous")),
                    "duration_s": _numeric_or_none(row.get("duration_s")),
                    "checkpoint_reload_ok": row.get("checkpoint_reload_ok"),
                }
            )

    final_val = _numeric_or_none(artifact.get("baseline", {}).get("val_exact_accuracy")) if isinstance(artifact.get("baseline"), dict) else None
    if final_val is None and trajectory:
        final_val = trajectory[-1]
    matches_published = artifact.get("matches_published_087") is True
    status = (
        f"reproduced_val_{final_val:.4f}"
        if matches_published and final_val is not None
        else f"open_baseline_not_reproduced_val_{final_val:.4f}"
    )
    per_pass_delta = artifact.get("per_pass_delta_vs_v381", {})
    if not isinstance(per_pass_delta, dict):
        per_pass_delta = {}
    return {
        "gap_id": SUDOKU_BASELINE_GAP_ID,
        "status": status,
        "artifact_path": EXP4127_PATH,
        "source_artifacts": [EXP4126_PATH, EXP4127_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "published_target_val_exact_accuracy": PUBLISHED_SUDOKU_TARGET,
        "matches_published_087": matches_published,
        "final_val_exact_accuracy": final_val,
        "final_val_exact_accuracy_rounded": _round4(final_val),
        "val_trajectory": trajectory,
        "val_trajectory_rounded": [_round4(value) for value in trajectory],
        "passes": passes,
        "per_pass_delta_vs_v381": per_pass_delta,
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "acceptance_gate_passed": bool(artifact.get("acceptance_gate_passed")),
        "baseline_reproduction_status": ("reproduced" if matches_published else "not_reproduced"),
        "contiguous_run_recommendation": artifact.get("contiguous_run_recommendation"),
        "total_duration_s": _numeric_or_none(artifact.get("total_duration_s")),
        "missing_discriminator": (
            "faithful_fixed_lr_trm_sudoku_candidate_source_before_training_time_verifier_claims"
        ),
    }


def classify_sudoku_graft(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4131: summarize Exp 4128 as a deferral unless it really grafted."""
    artifact = base._load_json(repo_root / EXP4128_PATH)
    graft_deferred = artifact.get("graft_deferred") is True
    verifier_value_added = artifact.get("verifier_value_added") is True
    if graft_deferred:
        status = "open_graft_deferred_verifier_value_added_false"
    elif verifier_value_added:
        status = "filled_verifier_value_added"
    else:
        status = "open_honest_null_no_value_added"
    return {
        "gap_id": SUDOKU_GRAFT_GAP_ID,
        "status": status,
        "artifact_path": EXP4128_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "graft_deferred": graft_deferred,
        "verifier_value_added": verifier_value_added,
        "verifier_value_added_meaningful": artifact.get("verifier_value_added_meaningful") is True,
        "acceptance_gate_passed": bool(artifact.get("acceptance_gate_passed")),
        "baseline_val_exact_accuracy": _numeric_or_none(artifact.get("baseline_val_exact_accuracy")),
        "baseline_matches_published_087": artifact.get("baseline_matches_published_087") is True,
        "estimated_passes_to_converge_for_383": artifact.get("estimated_passes_to_converge_for_383"),
        "rerank_lift_vs_vote": artifact.get("rerank_lift_vs_vote"),
        "rft_vs_ablation_delta": artifact.get("rft_vs_ablation_delta"),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "missing_discriminator": (
            "decision_grade_training_time_value_from_executable_verifier_labels_over_vote_labels"
        ),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    lr_resume_fix: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .382 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_sudoku_training_role(
        updated_registry,
        lr_resume_fix,
        sudoku_baseline,
        sudoku_graft,
    )

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4131-lr-resume-fix",
        _lr_resume_gap_block(lr_resume_fix),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4131-sudoku-baseline-reproduction",
        _sudoku_baseline_gap_block(lr_resume_fix, sudoku_baseline),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4131-sudoku-verifier-graft",
        _sudoku_graft_gap_block(sudoku_baseline, sudoku_graft),
    )
    touched = [
        gap_id
        for gap_id in (LR_RESUME_GAP_ID, SUDOKU_BASELINE_GAP_ID, SUDOKU_GRAFT_GAP_ID)
        if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "lr_resume_recorded": LR_RESUME_GAP_ID in touched,
            "sudoku_baseline_recorded": SUDOKU_BASELINE_GAP_ID in touched,
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
            "eval_exp_4131": EXP4131_ARTIFACT_PATH,
            "exp4131_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4131_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4131_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4131_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4131_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_sudoku_training_role(
    registry: dict[str, Any],
    lr_resume_fix: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    old_roles = list(entry.get("training_time_roles", []))
    if bool(sudoku_graft.get("graft_deferred")):
        status = "graft_deferred_baseline_not_reproduced"
    elif bool(sudoku_graft.get("verifier_value_added")):
        status = "value_added"
    else:
        status = "honest_null_no_value_added"
    role = {
        "role_id": SUDOKU_TRAINING_ROLE_ID,
        "experiment": EXP4128_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": status,
        "outcome": sudoku_graft.get("status"),
        "honest_verdict": sudoku_graft.get("honest_verdict", ""),
        "lr_fix_artifact": lr_resume_fix.get("artifact_path"),
        "lr_continuous_across_resume": bool(lr_resume_fix.get("lr_continuous_across_resume")),
        "validation_first_lr": lr_resume_fix.get("validation_first_lr"),
        "baseline_artifact": sudoku_baseline.get("artifact_path"),
        "baseline_reproduction_status": sudoku_baseline.get("baseline_reproduction_status"),
        "baseline_final_val_exact_accuracy": sudoku_baseline.get("final_val_exact_accuracy"),
        "baseline_val_trajectory_rounded": sudoku_baseline.get("val_trajectory_rounded"),
        "matches_published_087": bool(sudoku_baseline.get("matches_published_087")),
        "graft_deferred": bool(sudoku_graft.get("graft_deferred")),
        "verifier_value_added": bool(sudoku_graft.get("verifier_value_added")),
        "verifier_value_added_meaningful": bool(sudoku_graft.get("verifier_value_added_meaningful")),
        "flagged_adversarial": bool(sudoku_graft.get("flagged_adversarial")),
        "estimated_passes_to_converge_for_383": sudoku_graft.get("estimated_passes_to_converge_for_383"),
        "rerank_lift_vs_vote": sudoku_graft.get("rerank_lift_vs_vote"),
        "rft_vs_ablation_delta": sudoku_graft.get("rft_vs_ablation_delta"),
        "eval_exp_4131": EXP4131_ARTIFACT_PATH,
    }
    entry["training_time_roles"] = [
        old for old in old_roles if old.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [role]


def _lr_resume_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {LR_RESUME_GAP_ID}: Exp 4131 .382 LR resume correctness status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4126_PATH}`; "
        f"lr_continuous_across_resume="
        f"{str(bool(outcome.get('lr_continuous_across_resume'))).lower()}; "
        f"validation_first_lr={outcome.get('validation_first_lr')}; "
        f"fresh_warmup_lr={outcome.get('fresh_warmup_lr')}; "
        f"manual_lr_step_restored={outcome.get('manual_lr_step_restored')}; "
        f"full_batch_validation_attempt={outcome.get('full_batch_validation_attempt')}.\n"
        "- failure mode: previous bounded Sudoku resumes rewarmed the manual LR schedule, "
        "making verifier training-time claims underpowered and hard to interpret.\n"
        "- missing discriminator: faithful LR-schedule continuity before treating resumed "
        "TRM candidate pools as reward-training evidence.\n"
        "- candidate design: build on the fixed stable checkpoint lineage and keep "
        "measuring validation until the baseline is faithful enough for grafting.\n"
        "- priority: high\n"
    )


def _sudoku_baseline_gap_block(
    lr_resume_fix: dict[str, Any],
    outcome: dict[str, Any],
) -> str:
    delta = outcome.get("per_pass_delta_vs_v381", {})
    return (
        f"### {SUDOKU_BASELINE_GAP_ID}: Exp 4131 .382 Sudoku fixed-LR baseline status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4127_PATH}` with LR fix `{EXP4126_PATH}`; "
        f"val_trajectory={outcome.get('val_trajectory_rounded')}; "
        f"final_val={outcome.get('final_val_exact_accuracy_rounded')}; "
        f"matches_published_087={str(bool(outcome.get('matches_published_087'))).lower()}; "
        f"published_target={outcome.get('published_target_val_exact_accuracy')}; "
        f"lr_continuous_across_resume="
        f"{str(bool(lr_resume_fix.get('lr_continuous_across_resume'))).lower()}; "
        f"per_pass_delta_vs_v381={delta}.\n"
        "- failure mode: the fixed-LR nano-TRM Sudoku checkpoint improved much faster "
        "than the .381 rewarm runs but remains far below the published baseline, so "
        "verifier training-time claims over this pool are still underpowered.\n"
        "- missing discriminator: faithful fixed-LR TRM Sudoku candidate source before "
        "training-time verifier claims.\n"
        "- candidate design: continue the fixed checkpoint lineage or use verifier-guided "
        "candidate expansion before treating executable constraints as a reward-training win.\n"
        "- priority: high\n"
    )


def _sudoku_graft_gap_block(
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
) -> str:
    estimate = sudoku_graft.get("estimated_passes_to_converge_for_383")
    return (
        f"### {SUDOKU_GRAFT_GAP_ID}: Exp 4131 .382 Sudoku executable-verifier graft status\n"
        f"- status: {sudoku_graft['status']}\n"
        f"- evidence: `{EXP4128_PATH}`; "
        f"graft_deferred={str(bool(sudoku_graft.get('graft_deferred'))).lower()}; "
        f"verifier_value_added={str(bool(sudoku_graft.get('verifier_value_added'))).lower()}; "
        f"verifier_value_added_meaningful="
        f"{str(bool(sudoku_graft.get('verifier_value_added_meaningful'))).lower()}; "
        f"flagged_adversarial={str(bool(sudoku_graft.get('flagged_adversarial'))).lower()}; "
        f"baseline_final_val={sudoku_baseline.get('final_val_exact_accuracy_rounded')}; "
        f"baseline_trajectory={sudoku_baseline.get('val_trajectory_rounded')}; "
        f"estimated_passes_to_converge_for_383={estimate}.\n"
        "- failure mode: Exp 4128 did not run a meaningful graft because the .382 "
        "baseline was still not reproduced; the executable verifier therefore has no "
        "decision-grade training-time TRM value-added result.\n"
        "- missing discriminator: decision-grade training-time value from executable "
        "verifier labels beyond vote labels on held-out TRM Sudoku induction.\n"
        "- candidate design: rerun the graft only after baseline reproduction or after "
        "verifier-guided candidate expansion creates a candidate pool with measurable "
        "oracle headroom.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4131") == EXP4131_ARTIFACT_PATH
        and any(
            role.get("role_id") == SUDOKU_TRAINING_ROLE_ID
            for role in gap4.get("training_time_roles", [])
        )
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    lr_resume_fix: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4131 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    lr_recorded = LR_RESUME_GAP_ID in gaps_updated
    baseline_recorded = SUDOKU_BASELINE_GAP_ID in gaps_updated
    graft_recorded = SUDOKU_GRAFT_GAP_ID in gaps_updated
    prefix = "complete:" if guard_ok and lr_recorded and baseline_recorded and graft_recorded else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4131_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4131_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v382_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"lr_resume_{lr_resume_fix['status']}_"
            f"sudoku_baseline_{sudoku_baseline['status']}_"
            f"graft_deferred_{bool(sudoku_graft.get('graft_deferred'))}_"
            f"verifier_value_added_{bool(sudoku_graft.get('verifier_value_added'))}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "lr_resume_fix": lr_resume_fix,
        "sudoku_baseline": sudoku_baseline,
        "sudoku_graft": sudoku_graft,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            exp4122.exp4112.exp4103.exp4095.ARC1_POOL_PATH,
            exp4122.exp4112.exp4103.exp4095.ARC1_PROGRAMS_PATH,
            EXP4126_PATH,
            EXP4127_PATH,
            EXP4128_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4131 fields before writing the artifact."""
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
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4131 principles")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4131 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    lr_resume_fix = classify_lr_resume_fix(repo_root)
    sudoku_baseline = classify_sudoku_baseline(repo_root)
    sudoku_graft = classify_sudoku_graft(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        lr_resume_fix,
        sudoku_baseline,
        sudoku_graft,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        lr_resume_fix=lr_resume_fix,
        sudoku_baseline=sudoku_baseline,
        sudoku_graft=sudoku_graft,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4131_ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4131_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
