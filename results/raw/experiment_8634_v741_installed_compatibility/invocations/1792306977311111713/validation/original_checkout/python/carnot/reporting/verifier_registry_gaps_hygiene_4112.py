"""Exp 4112 registry/gaps hygiene for .380 verifier outcomes.

Spec refs: REQ-VERIFY-4112, SCENARIO-VERIFY-4112.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4103 as exp4103


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4112_ARTIFACT_PATH = "results/experiment_4112_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

EXP4099_PATH = exp4103.EXP4099_PATH
EXP4109_PATH = "results/experiment_4109_carnot_verifier_graft_sudoku.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
TRM_GRID_GAP_ID = "GAP-TRM-GRID-DISCRIMINATION"
SUDOKU_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4109"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4109"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "trm_grid_discrimination",
    "sudoku_verifier",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .380 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched so the gap backlog stays the honest complement "
        "of the registry."
    ),
}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay the cached GAP-4 ARC-1 guard without live model or code generation."""
    return exp4103.replay_gap4_arc1(repo_root)


def classify_trm_grid_discrimination(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize the .379 TRM-grid anti-discrimination measurement."""
    artifact = base._load_json(repo_root / EXP4099_PATH)
    per_reranker = artifact["per_reranker"]
    best = str(artifact["best_reranker"])
    best_row = per_reranker[best]
    anti_rows = {
        name: row
        for name, row in per_reranker.items()
        if name != "TRM_VOTE" and float(row.get("captured_pp", 0.0)) < 0.0
    }
    worst_name, worst_row = sorted(
        anti_rows.items(),
        key=lambda item: (float(item[1].get("captured_pp", 0.0)), item[0]),
    )[0]
    captured = round(float(worst_row["captured_pp"]), 4)
    return {
        "gap_id": TRM_GRID_GAP_ID,
        "status": f"open_anti_discrimination_captured_pp_{captured:.4f}",
        "artifact_path": EXP4099_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_beats_trm_vote": bool(artifact.get("verifier_beats_trm_vote")),
        "best_reranker": best,
        "best_captured_pp": round(float(best_row.get("captured_pp", 0.0)), 4),
        "best_captured_pp_ci95": list(best_row.get("captured_pp_ci95", [0.0, 0.0])),
        "anti_discrimination_reranker": worst_name,
        "anti_discriminating_rerankers": sorted(
            name
            for name, row in anti_rows.items()
            if round(float(row.get("captured_pp", 0.0)), 4) == captured
        ),
        "anti_discrimination_captured_pp": captured,
        "anti_discrimination_captured_pp_rounded": round(captured, 2),
        "anti_discrimination_captured_pp_ci95": list(worst_row.get("captured_pp_ci95", [0.0, 0.0])),
        "trm_vote_pass2": artifact.get("trm_vote_pass2"),
        "pool_n_tasks": artifact.get("pool_n_tasks"),
        "missing_discriminator": (
            "signal_separating_correct_trm_grid_from_confident_wrong_trm_grid_on_pool"
        ),
    }


def classify_sudoku_verifier(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize the .380 executable Sudoku verifier value-added result."""
    artifact = base._load_json(repo_root / EXP4109_PATH)
    value_added = bool(artifact.get("verifier_value_added"))
    status = "filled_value_added" if value_added else "open_honest_null_no_value_added"
    return {
        "gap_id": SUDOKU_GAP_ID,
        "status": status,
        "artifact_path": EXP4109_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_value_added": value_added,
        "native_training_launched": bool(artifact.get("native_training_launched")),
        "rerank_lift_vs_vote": dict(artifact.get("rerank_lift_vs_vote", {})),
        "rft_vs_ablation_delta": dict(artifact.get("rft_vs_ablation_delta", {})),
        "a_vs_cold_lift": dict(artifact.get("a_vs_cold_lift", {})),
        "corpus_summary": dict(artifact.get("corpus_summary", {})),
        "baseline_limitation": artifact.get("baseline_limitation"),
        "native_training_limitation": artifact.get("native_training_limitation"),
        "acceptance_gate_passed": bool(artifact.get("acceptance_gate_passed")),
        "candidate_source": str(artifact.get("candidate_source", "")),
        "reproducibility_checksum": str(artifact.get("reproducibility_checksum", "")),
        "missing_discriminator": (
            "decision_grade_training_time_value_from_executable_verifier_labels_over_vote_labels"
        ),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    trm_grid_discrimination: dict[str, Any],
    sudoku_verifier: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with Exp 4112 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_sudoku_training_role(updated_registry, sudoku_verifier)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4103-trm-grid-discrimination",
        _trm_grid_gap_block(trm_grid_discrimination),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4112-sudoku-executable-verifier",
        _sudoku_gap_block(sudoku_verifier),
    )
    touched = [
        gap_id
        for gap_id in (TRM_GRID_GAP_ID, SUDOKU_GAP_ID)
        if gap_id in updated_gaps
    ]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "trm_grid_gap_recorded": TRM_GRID_GAP_ID in touched,
            "sudoku_gap_recorded": SUDOKU_GAP_ID in touched,
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
            "eval_exp_4112": EXP4112_ARTIFACT_PATH,
            "exp4112_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4112_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4112_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4112_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4112_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_sudoku_training_role(registry: dict[str, Any], sudoku_verifier: dict[str, Any]) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    assert entry is not None
    old_roles = list(entry.get("training_time_roles", []))
    status = (
        "value_added"
        if bool(sudoku_verifier.get("verifier_value_added"))
        else "honest_null_no_value_added"
    )
    role = {
        "role_id": SUDOKU_TRAINING_ROLE_ID,
        "experiment": EXP4109_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": status,
        "outcome": sudoku_verifier.get("status"),
        "honest_verdict": sudoku_verifier.get("honest_verdict", ""),
        "verifier_value_added": bool(sudoku_verifier.get("verifier_value_added")),
        "native_training_launched": bool(sudoku_verifier.get("native_training_launched")),
        "rft_vs_ablation_delta": sudoku_verifier.get("rft_vs_ablation_delta", {}),
        "rerank_lift_vs_vote": sudoku_verifier.get("rerank_lift_vs_vote", {}),
        "eval_exp_4112": EXP4112_ARTIFACT_PATH,
    }
    entry["training_time_roles"] = [
        old for old in old_roles if old.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [role]


def _trm_grid_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {TRM_GRID_GAP_ID}: Exp 4112 .379 TRM-grid anti-discrimination update\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{outcome.get('artifact_path', EXP4099_PATH)}`; "
        f"best_reranker={outcome.get('best_reranker')}; "
        f"best_captured_pp={outcome.get('best_captured_pp')}; "
        f"captured_pp={outcome.get('anti_discrimination_captured_pp')}; "
        f"captured_pp_rounded={outcome.get('anti_discrimination_captured_pp_rounded')}; "
        f"captured_pp_ci95={outcome.get('anti_discrimination_captured_pp_ci95')}; "
        f"anti_discriminating_rerankers={outcome.get('anti_discriminating_rerankers')}; "
        f"verifier_beats_trm_vote={str(bool(outcome.get('verifier_beats_trm_vote'))).lower()}; "
        f"pool_n_tasks={outcome.get('pool_n_tasks')}.\n"
        "- failure mode: the correct TRM grid can be present in the candidate pool but remain "
        "unselectable; the measured verifier rerankers either tie vote or actively anti-rank "
        "against it.\n"
        "- missing discriminator: signal separating a correct TRM grid from a confident-wrong "
        "TRM grid on the candidate pool.\n"
        "- candidate design: treat the anti-discrimination as a missing-verifier spec until a "
        "held-out discriminator beats TRM vote without relying on neutral vote fallback.\n"
        "- priority: high\n"
    )


def _sudoku_gap_block(outcome: dict[str, Any]) -> str:
    delta = outcome.get("rft_vs_ablation_delta", {})
    rerank = outcome.get("rerank_lift_vs_vote", {})
    return (
        f"### {SUDOKU_GAP_ID}: Exp 4112 .380 Sudoku executable-verifier update\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{outcome.get('artifact_path', EXP4109_PATH)}`; "
        f"verifier_value_added={str(bool(outcome.get('verifier_value_added'))).lower()}; "
        f"native_training_launched={str(bool(outcome.get('native_training_launched'))).lower()}; "
        f"rft_vs_ablation_delta={delta}; rerank_delta={rerank.get('delta')}; "
        f"n_matched={outcome.get('corpus_summary', {}).get('n_matched')}.\n"
        "- failure mode: executable Sudoku constraints can score candidate validity, but Exp 4109 "
        "did not show value over the vote-label ablation on the matched executable-domain corpus.\n"
        "- missing discriminator: decision-grade evidence that executable verifier labels add "
        "training-time value beyond vote labels on held-out TRM Sudoku induction.\n"
        "- candidate design: rerun only with native training launched or a corpus where the vote "
        "baseline leaves measurable headroom; keep the .380 result as an honest null meanwhile.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4112") == EXP4112_ARTIFACT_PATH
        and any(
            role.get("role_id") == SUDOKU_TRAINING_ROLE_ID
            for role in gap4.get("training_time_roles", [])
        )
    )


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    trm_grid_discrimination: dict[str, Any],
    sudoku_verifier: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4112 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    trm_recorded = TRM_GRID_GAP_ID in gaps_updated
    sudoku_recorded = SUDOKU_GAP_ID in gaps_updated
    prefix = "complete:" if guard_ok and trm_recorded and sudoku_recorded else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4112_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4112_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v380_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"trm_grid_{trm_grid_discrimination['status']}_"
            f"sudoku_{sudoku_verifier['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "trm_grid_discrimination": trm_grid_discrimination,
        "sudoku_verifier": sudoku_verifier,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 3),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            exp4103.exp4095.ARC1_POOL_PATH,
            exp4103.exp4095.ARC1_PROGRAMS_PATH,
            EXP4099_PATH,
            EXP4109_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4112 fields before writing the artifact."""
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


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4112 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    trm_grid_discrimination = classify_trm_grid_discrimination(repo_root)
    sudoku_verifier = classify_sudoku_verifier(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        trm_grid_discrimination,
        sudoku_verifier,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        trm_grid_discrimination=trm_grid_discrimination,
        sudoku_verifier=sudoku_verifier,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4112_ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4112_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
