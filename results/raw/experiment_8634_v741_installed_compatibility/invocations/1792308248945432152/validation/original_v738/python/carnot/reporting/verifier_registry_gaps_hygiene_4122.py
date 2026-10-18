"""Exp 4122 registry/gaps hygiene for .381 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4122, SCENARIO-VERIFY-4122.

This module is deliberately a ledger reconciler, not a training runner. The
GAP-4 guard proves the existing ARC verifier still reproduces its cached
headline, while the Sudoku fields keep a weaker fact separate: the resumed TRM
baseline improved but did not reproduce the published Sudoku result, so the
Exp 4119 verifier graft was deferred instead of becoming a reward-training win.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4112 as exp4112


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4122_ARTIFACT_PATH = "results/experiment_4122_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

EXP4116_PATH = "results/experiment_4116_sudoku_extreme_resume_pass1.json"
EXP4117_PATH = "results/experiment_4117_sudoku_extreme_resume_pass2.json"
EXP4118_PATH = "results/experiment_4118_sudoku_extreme_resume_pass3.json"
EXP4119_PATH = "results/experiment_4119_carnot_verifier_graft_sudoku.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
SUDOKU_BASELINE_GAP_ID = "GAP-SUDOKU-BASELINE-REPRODUCTION-4118"
SUDOKU_GRAFT_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4119"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4119"
PUBLISHED_SUDOKU_TARGET = 0.87

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "sudoku_baseline",
    "sudoku_graft",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .381 truth.",
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
    """Replay the cached GAP-4 ARC-1 guard without fresh Codex or GGUF inference."""
    return exp4112.replay_gap4_arc1(repo_root)


def classify_sudoku_baseline(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize the .381 Sudoku baseline trajectory before any graft claim.

    Exp 4116's own artifact is flagged, but Exp 4117 links the first pass metric
    from the stable Hydra CSV. We record that provenance instead of pretending
    the first pass artifact itself was clean.
    """
    exp4116 = base._load_json(repo_root / EXP4116_PATH)
    exp4117 = base._load_json(repo_root / EXP4117_PATH)
    exp4118 = base._load_json(repo_root / EXP4118_PATH)

    pass1_link = exp4117.get("pass1", {})
    if not isinstance(pass1_link, dict):
        pass1_link = {}
    pass2_link = exp4118.get("pass2", {})
    if not isinstance(pass2_link, dict):
        pass2_link = {}

    pass1_val = _numeric_or_none(pass1_link.get("val_exact_accuracy"))
    pass2_val = _numeric_or_none(pass2_link.get("val_exact_accuracy"))
    pass3_val = _numeric_or_none(exp4118.get("val_exact_accuracy"))
    if pass2_val is None:
        pass2_val = _numeric_or_none(exp4117.get("val_exact_accuracy"))

    passes = [
        _baseline_pass(
            "pass1",
            EXP4116_PATH,
            exp4116,
            pass1_val,
            str(pass1_link.get("val_source", "")),
        ),
        _baseline_pass(
            "pass2",
            EXP4117_PATH,
            exp4117,
            pass2_val,
            str(pass2_link.get("val_source", exp4117.get("exact_accuracy_metrics_path", ""))),
        ),
        _baseline_pass(
            "pass3",
            EXP4118_PATH,
            exp4118,
            pass3_val,
            str(exp4118.get("exact_accuracy_metrics_path", "")),
        ),
    ]
    trajectory = [value for value in (pass1_val, pass2_val, pass3_val) if value is not None]
    final_val = pass3_val
    matches_published = exp4118.get("matches_published_087") is True
    status = (
        f"reproduced_val_{final_val:.4f}"
        if matches_published and final_val is not None
        else f"open_baseline_not_reproduced_val_{final_val:.4f}"
    )
    return {
        "gap_id": SUDOKU_BASELINE_GAP_ID,
        "status": status,
        "artifact_path": EXP4118_PATH,
        "source_artifacts": [EXP4116_PATH, EXP4117_PATH, EXP4118_PATH],
        "honest_verdict": str(exp4118.get("honest_verdict", "")),
        "published_target_val_exact_accuracy": PUBLISHED_SUDOKU_TARGET,
        "matches_published_087": matches_published,
        "final_val_exact_accuracy": final_val,
        "final_val_exact_accuracy_rounded": _round4(final_val),
        "val_trajectory": trajectory,
        "val_trajectory_rounded": [_round4(value) for value in trajectory],
        "passes": passes,
        "total_cumulative_epochs": exp4118.get("total_cumulative_epochs"),
        "stable_checkpoint_path": str(exp4118.get("stable_checkpoint_path", "")),
        "acceptance_gate_passed": bool(exp4118.get("acceptance_gate_passed")),
        "baseline_reproduction_status": ("reproduced" if matches_published else "not_reproduced"),
        "missing_discriminator": (
            "faithful_resumed_trm_sudoku_candidate_source_before_training_time_verifier_claims"
        ),
    }


def _baseline_pass(
    pass_id: str,
    artifact_path: str,
    artifact: dict[str, Any],
    val_exact_accuracy: float | None,
    val_source: str,
) -> dict[str, Any]:
    return {
        "pass_id": pass_id,
        "artifact_path": artifact_path,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "artifact_flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "artifact_acceptance_gate_passed": bool(artifact.get("acceptance_gate_passed")),
        "val_exact_accuracy": val_exact_accuracy,
        "val_exact_accuracy_rounded": _round4(val_exact_accuracy),
        "val_source": val_source,
        "run_dir": str(artifact.get("run_dir", "")),
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
    }


def classify_sudoku_graft(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize Exp 4119 as a deferred graft, not a verifier reward win."""
    artifact = base._load_json(repo_root / EXP4119_PATH)
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
        "artifact_path": EXP4119_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "graft_deferred": graft_deferred,
        "verifier_value_added": verifier_value_added,
        "acceptance_gate_passed": bool(artifact.get("acceptance_gate_passed")),
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
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .381 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_sudoku_training_role(updated_registry, sudoku_baseline, sudoku_graft)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4122-sudoku-baseline-reproduction",
        _sudoku_baseline_gap_block(sudoku_baseline),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4122-sudoku-verifier-graft",
        _sudoku_graft_gap_block(sudoku_baseline, sudoku_graft),
    )
    touched = [
        gap_id for gap_id in (SUDOKU_BASELINE_GAP_ID, SUDOKU_GRAFT_GAP_ID) if gap_id in updated_gaps
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
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4122": EXP4122_ARTIFACT_PATH,
            "exp4122_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4122_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4122_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4122_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4122_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_sudoku_training_role(
    registry: dict[str, Any],
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
        "experiment": EXP4119_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": status,
        "outcome": sudoku_graft.get("status"),
        "honest_verdict": sudoku_graft.get("honest_verdict", ""),
        "baseline_artifact": sudoku_baseline.get("artifact_path"),
        "baseline_reproduction_status": sudoku_baseline.get("baseline_reproduction_status"),
        "baseline_final_val_exact_accuracy": sudoku_baseline.get("final_val_exact_accuracy"),
        "baseline_val_trajectory_rounded": sudoku_baseline.get("val_trajectory_rounded"),
        "matches_published_087": bool(sudoku_baseline.get("matches_published_087")),
        "total_cumulative_epochs": sudoku_baseline.get("total_cumulative_epochs"),
        "graft_deferred": bool(sudoku_graft.get("graft_deferred")),
        "verifier_value_added": bool(sudoku_graft.get("verifier_value_added")),
        "flagged_adversarial": bool(sudoku_graft.get("flagged_adversarial")),
        "rerank_lift_vs_vote": sudoku_graft.get("rerank_lift_vs_vote"),
        "rft_vs_ablation_delta": sudoku_graft.get("rft_vs_ablation_delta"),
        "eval_exp_4122": EXP4122_ARTIFACT_PATH,
    }
    entry["training_time_roles"] = [
        old for old in old_roles if old.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [role]


def _sudoku_baseline_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SUDOKU_BASELINE_GAP_ID}: Exp 4122 .381 Sudoku baseline reproduction status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4116_PATH}`, `{EXP4117_PATH}`, `{EXP4118_PATH}`; "
        f"val_trajectory={outcome.get('val_trajectory_rounded')}; "
        f"final_val={outcome.get('final_val_exact_accuracy_rounded')}; "
        f"matches_published_087={str(bool(outcome.get('matches_published_087'))).lower()}; "
        f"published_target={outcome.get('published_target_val_exact_accuracy')}; "
        f"total_cumulative_epochs={outcome.get('total_cumulative_epochs')}.\n"
        "- failure mode: the nano-TRM Sudoku checkpoint resumed and improved, but the "
        "validation exact accuracy remains far below the published baseline, so verifier "
        "training-time claims over this pool are underpowered.\n"
        "- missing discriminator: faithful resumed TRM Sudoku candidate source before "
        "training-time verifier claims.\n"
        "- candidate design: continue the stable baseline reproduction or move the "
        "executable verifier into adaptive candidate expansion before treating it as a "
        "reward-training signal.\n"
        "- priority: high\n"
    )


def _sudoku_graft_gap_block(
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
) -> str:
    return (
        f"### {SUDOKU_GRAFT_GAP_ID}: Exp 4122 .381 Sudoku executable-verifier graft status\n"
        f"- status: {sudoku_graft['status']}\n"
        f"- evidence: `{EXP4119_PATH}`; "
        f"graft_deferred={str(bool(sudoku_graft.get('graft_deferred'))).lower()}; "
        f"verifier_value_added={str(bool(sudoku_graft.get('verifier_value_added'))).lower()}; "
        f"flagged_adversarial={str(bool(sudoku_graft.get('flagged_adversarial'))).lower()}; "
        f"baseline_final_val={sudoku_baseline.get('final_val_exact_accuracy_rounded')}; "
        f"baseline_trajectory={sudoku_baseline.get('val_trajectory_rounded')}.\n"
        "- failure mode: Exp 4119 did not run a meaningful graft because the .381 "
        "baseline was not reproduced; the executable verifier therefore has no "
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
        and gap4.get("eval", {}).get("eval_exp_4122") == EXP4122_ARTIFACT_PATH
        and any(
            role.get("role_id") == SUDOKU_TRAINING_ROLE_ID
            for role in gap4.get("training_time_roles", [])
        )
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4122 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    baseline_recorded = SUDOKU_BASELINE_GAP_ID in gaps_updated
    graft_recorded = SUDOKU_GRAFT_GAP_ID in gaps_updated
    prefix = "complete:" if guard_ok and baseline_recorded and graft_recorded else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4122_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4122_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v381_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"sudoku_baseline_{sudoku_baseline['status']}_"
            f"graft_deferred_{bool(sudoku_graft.get('graft_deferred'))}_"
            f"verifier_value_added_{bool(sudoku_graft.get('verifier_value_added'))}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "sudoku_baseline": sudoku_baseline,
        "sudoku_graft": sudoku_graft,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            exp4112.exp4103.exp4095.ARC1_POOL_PATH,
            exp4112.exp4103.exp4095.ARC1_PROGRAMS_PATH,
            EXP4116_PATH,
            EXP4117_PATH,
            EXP4118_PATH,
            EXP4119_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4122 fields before writing the artifact."""
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
        raise ValueError("field_principles must match the required Exp 4122 principles")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4122 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    sudoku_baseline = classify_sudoku_baseline(repo_root)
    sudoku_graft = classify_sudoku_graft(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        sudoku_baseline,
        sudoku_graft,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        sudoku_baseline=sudoku_baseline,
        sudoku_graft=sudoku_graft,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4122_ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4122_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
