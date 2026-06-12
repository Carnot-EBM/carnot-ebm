"""Exp 4103 registry/gaps hygiene for TRM-grid discrimination.

Spec refs: REQ-VERIFY-4103, SCENARIO-VERIFY-4103.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4095 as exp4095


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4103_ARTIFACT_PATH = "results/experiment_4103_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

EXP4099_PATH = "results/experiment_4099_trm_pool_verifier_discrimination_probe.json"
EXP4100_PATH = "results/experiment_4100_trm_verifier_rft_conditional.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
TRM_GRID_GAP_ID = "GAP-TRM-GRID-DISCRIMINATION"
TRM_RFT_GAP_ID = "GAP-TRM-VERIFIER-RFT-4100"
TRM_TRAINING_ROLE_ID = "trm_grid_discriminator_training_time_4100"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "exp4099_gap",
    "exp4100_outcome",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the registry/gaps were reconciled to the .379 truth."
    ),
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact from cached "
        "candidates; catches a silent verifier regression independent of the new work."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (GAP-TRM-GRID-DISCRIMINATION et al.) "
        "so the gap backlog stays the honest complement of the registry."
    ),
}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay the GAP-4 ARC-1 regression guard from cached candidates only."""
    replay = exp4095.replay_gap4_arc1(repo_root)
    return {
        **replay,
        "regression_guard_passed": bool(replay["gap4_arc1_reproduced"]),
    }


def classify_exp4099_gap(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Classify Exp 4099 as a filled or open TRM-grid discrimination gap."""
    artifact = base._load_json(repo_root / EXP4099_PATH)
    best = str(artifact.get("best_reranker", "unknown"))
    per_reranker = artifact.get("per_reranker", {})
    best_row = per_reranker.get(best, {}) if isinstance(per_reranker, dict) else {}
    captured = float(best_row.get("captured_pp", artifact.get("captured_pp_directional", 0.0)))
    captured_ci = list(best_row.get("captured_pp_ci95", [0.0, 0.0]))
    beats_vote = bool(artifact.get("verifier_beats_trm_vote"))
    status = (
        f"filled_by_{best}_captured_pp_{captured:.4f}"
        if beats_vote
        else f"open_captured_pp_{captured:.4f}"
    )
    return {
        "gap_id": TRM_GRID_GAP_ID,
        "status": status,
        "artifact_path": EXP4099_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_beats_trm_vote": beats_vote,
        "best_reranker": best,
        "captured_pp": round(captured, 4),
        "captured_pp_ci95": captured_ci,
        "trm_vote_pass2": artifact.get("trm_vote_pass2"),
        "pool_n_tasks": artifact.get("pool_n_tasks"),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "missing_discriminator": (
            "signal_separating_correct_trm_grid_from_confident_wrong_trm_grid_on_pool"
        ),
    }


def classify_exp4100_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize Exp 4100 without converting smoke into a training win."""
    artifact = base._load_json(repo_root / EXP4100_PATH)
    branch = str(artifact.get("branch_taken", "unknown"))
    delta = dict(artifact.get("rft_vs_ablation_delta", {}))
    verifier_gap = artifact.get("verifier_gap", {})
    bottleneck = (
        verifier_gap.get("bottleneck", "")
        if isinstance(verifier_gap, dict)
        else ""
    )
    status = "rft_complete" if branch == "rft" else f"{branch}_checkpoint_ok"
    return {
        "artifact_path": EXP4100_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "branch_taken": branch,
        "status": status,
        "trm_native_trainer_checkpoint_ok": bool(
            artifact.get("trm_native_trainer_checkpoint_ok")
        ),
        "rft_vs_ablation_delta": delta,
        "bottleneck": bottleneck,
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    exp4099_gap: dict[str, Any],
    exp4100_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with Exp 4103 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_trm_training_role(updated_registry, exp4099_gap, exp4100_outcome)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4103-trm-grid-discrimination",
        _trm_grid_gap_block(exp4099_gap),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4103-trm-rft-outcome",
        _trm_rft_outcome_block(exp4100_outcome),
    )
    touched = [
        gap_id
        for gap_id in (TRM_GRID_GAP_ID, TRM_RFT_GAP_ID)
        if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "trm_grid_gap_recorded": TRM_GRID_GAP_ID in touched,
            "exp4100_outcome_recorded": TRM_RFT_GAP_ID in touched,
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
            "eval_exp_4103": EXP4103_ARTIFACT_PATH,
            "exp4103_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4103_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4103_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4103_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4103_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_trm_training_role(
    registry: dict[str, Any],
    exp4099_gap: dict[str, Any],
    exp4100_outcome: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    old_roles = list(entry.get("training_time_roles", []))
    role = {
        "role_id": TRM_TRAINING_ROLE_ID,
        "experiment": EXP4100_PATH,
        "role": "candidate_trm_training_time_reward_signal",
        "status": exp4100_outcome.get("branch_taken"),
        "outcome": exp4100_outcome.get("status"),
        "honest_verdict": exp4100_outcome.get("honest_verdict", ""),
        "rft_vs_ablation_delta": exp4100_outcome.get("rft_vs_ablation_delta", {}),
        "bottleneck": exp4100_outcome.get("bottleneck", ""),
        "exp4099_captured_pp": exp4099_gap.get("captured_pp"),
        "eval_exp_4103": EXP4103_ARTIFACT_PATH,
    }
    entry["training_time_roles"] = [
        old for old in old_roles if old.get("role_id") != TRM_TRAINING_ROLE_ID
    ] + [role]


def _trm_grid_gap_block(outcome: dict[str, Any]) -> str:
    captured = float(outcome.get("captured_pp", 0.0))
    return (
        f"### {TRM_GRID_GAP_ID}: TRM-grid rerank discrimination\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{outcome.get('artifact_path', EXP4099_PATH)}`; "
        f"best_reranker={outcome.get('best_reranker')}; captured_pp={captured}; "
        f"captured_pp_ci95={outcome.get('captured_pp_ci95')}; "
        f"verifier_beats_trm_vote={str(bool(outcome.get('verifier_beats_trm_vote'))).lower()}; "
        f"pool_n_tasks={outcome.get('pool_n_tasks')}.\n"
        "- failure mode: a correct grid can be present in the TRM candidate pool but remain "
        "unselectable because every recorded reranker ties TRM vote or ranks confident-wrong "
        "grids ahead of it.\n"
        "- missing discriminator: signal separating a correct TRM grid from a confident-wrong "
        "TRM grid on the candidate pool.\n"
        "- candidate design: train or mine a TRM-grid discriminator only after a held-out pool "
        "shows it beats TRM vote; until then keep the gap open and treat Exp 4100 as smoke.\n"
        "- priority: high\n"
    )


def _trm_rft_outcome_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {TRM_RFT_GAP_ID}: Exp 4100 TRM verifier-RFT outcome\n"
        f"- status: {outcome.get('status')}\n"
        f"- evidence: `{outcome.get('artifact_path', EXP4100_PATH)}`; "
        f"branch_taken={outcome.get('branch_taken')}; "
        f"trm_native_trainer_checkpoint_ok="
        f"{str(bool(outcome.get('trm_native_trainer_checkpoint_ok'))).lower()}; "
        f"rft_vs_ablation_delta={outcome.get('rft_vs_ablation_delta')}; "
        f"bottleneck={outcome.get('bottleneck')}.\n"
        "- failure mode: verifier-as-reward RFT cannot be promoted when the upstream grid "
        "reranker captured 0.0pp; the mechanism smoke proves checkpoint plumbing, not a reward win.\n"
        "- missing discriminator: decision-grade evidence that verifier-certified TRM training "
        "beats the vote-label ablation on held-out grid induction.\n"
        "- candidate design: rerun full RFT only after a non-TRM grid reranker clears the Exp 4099 "
        "discrimination gate.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if gap4 is None:
        return False
    eval_ok = gap4.get("eval", {}).get("eval_exp_4103") == EXP4103_ARTIFACT_PATH
    role_ok = any(
        role.get("role_id") == TRM_TRAINING_ROLE_ID
        for role in gap4.get("training_time_roles", [])
    )
    return eval_ok and role_ok


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    exp4099_gap: dict[str, Any],
    exp4100_outcome: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4103 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    grid_gap_recorded = TRM_GRID_GAP_ID in gaps_updated
    prefix = "complete:" if guard_ok and grid_gap_recorded else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4103_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4103_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v379_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"trm_grid_gap_{exp4099_gap['status']}_"
            f"exp4100_{exp4100_outcome['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "exp4099_gap": exp4099_gap,
        "exp4100_outcome": exp4100_outcome,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            exp4095.ARC1_POOL_PATH,
            exp4095.ARC1_PROGRAMS_PATH,
            EXP4099_PATH,
            EXP4100_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4103 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a bare bool")  # pragma: no cover
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")  # pragma: no cover
    if not isinstance(artifact["gaps_updated"], list):
        raise ValueError("gaps_updated must be a list")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")  # pragma: no cover


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4103 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    exp4099_gap = classify_exp4099_gap(repo_root)
    exp4100_outcome = classify_exp4100_outcome(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        exp4099_gap,
        exp4100_outcome,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        exp4099_gap=exp4099_gap,
        exp4100_outcome=exp4100_outcome,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4103_ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4103_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
