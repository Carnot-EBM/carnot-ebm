"""Exp 4153 registry/gaps hygiene for .384 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4153, SCENARIO-VERIFY-4153.

The runner is a ledger reconciler, not a model experiment. It replays the
canonical GAP-4 ARC-1 guard from cached artifacts and records the .384 Sudoku
truth from Exp 4149/4150: the baseline stayed at 0.2782, the decisive graft was
deferred, and DiffusionGemma stays gated because no transferable verifier value
was measured.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4142 as exp4142


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "cached_gap4_replay_and_ledger_reconciliation"

EXP4153_ARTIFACT_PATH = "results/experiment_4153_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4142.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4142.ARC1_PROGRAMS_PATH
EXP4149_PATH = "results/experiment_4149_sudoku_accumulate_pass4_convergence.json"
EXP4150_PATH = "results/experiment_4150_decisive_verifier_graft_sudoku.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
SUDOKU_BASELINE_GAP_ID = "GAP-SUDOKU-BASELINE-REPRODUCTION-4149"
SUDOKU_GRAFT_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4150"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4150"
PUBLISHED_SUDOKU_TARGET = 0.87
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
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .384 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched so the gap backlog stays the honest complement "
        "of the registry."
    ),
    "diffusiongemma_gate_state": (
        "Records whether the .384 graft's verifier_value_added (on the transferable rerank + "
        "RFT, NOT the oracle) unlocked or kept-gated the DiffusionGemma scale-up -- the "
        "forward-planning signal."
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


def _rel_or_str(repo_root: Path, value: str) -> str:
    path = Path(value)
    try:
        return str(path.relative_to(repo_root)) if path.is_absolute() else value
    except ValueError:
        return value


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4153: verify cached fixtures and ledgers before replay."""
    return exp4142.check_preconditions(repo_root)


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4153: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4142.replay_gap4_arc1(repo_root)


def _trajectory_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("val_trajectory_v384", [])
    if not isinstance(rows, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        raw_val = _numeric_or_none(row.get("val_exact_accuracy"))
        effective_val = _numeric_or_none(row.get("effective_val_exact_accuracy"))
        cleaned.append(
            {
                "pass_index": index,
                "pass_label": str(row.get("pass_label", "")),
                "experiment": str(row.get("experiment", "")),
                "artifact_path": _rel_or_str(REPO_ROOT, str(row.get("artifact_path", ""))),
                "honest_verdict": str(row.get("honest_verdict", "")),
                "post_epoch": row.get("post_epoch"),
                "duration_s": _numeric_or_none(row.get("duration_s")),
                "val_exact_accuracy": raw_val,
                "val_exact_accuracy_rounded": _round4(raw_val),
                "effective_val_exact_accuracy": effective_val,
                "effective_val_exact_accuracy_rounded": _round4(effective_val),
            }
        )
    return cleaned


def classify_sudoku_baseline(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4153: summarize the Exp 4149 .384 baseline trajectory."""
    artifact = base._load_json(repo_root / EXP4149_PATH)
    rows = _trajectory_rows(artifact)
    raw_values = [
        row["val_exact_accuracy"]
        for row in rows
        if row["val_exact_accuracy"] is not None
    ]
    effective_values = [
        row["effective_val_exact_accuracy"]
        for row in rows
        if row["effective_val_exact_accuracy"] is not None
    ]
    final_val = _numeric_or_none(artifact.get("val_exact_accuracy"))
    if final_val is None and effective_values:
        final_val = effective_values[-1]

    honest_verdict = str(artifact.get("honest_verdict", ""))
    baseline_status = honest_verdict.replace("complete: ", "").replace("success: ", "")
    final_text = f"{final_val:.4f}" if final_val is not None else "unknown"
    matches_published = artifact.get("matches_published_087") is True
    faithful = final_val is not None and final_val >= FAITHFUL_GRAFT_THRESHOLD
    if matches_published:
        status = f"reproduced_val_{final_text}"
    elif honest_verdict.startswith("blocked_pass3_noop_unresolved"):
        status = f"open_baseline_blocked_pass3_noop_val_{final_text}"
    else:
        status = f"open_baseline_not_reproduced_val_{final_text}"

    return {
        "gap_id": SUDOKU_BASELINE_GAP_ID,
        "status": status,
        "artifact_path": EXP4149_PATH,
        "source_artifacts": [EXP4149_PATH],
        "honest_verdict": honest_verdict,
        "baseline_status": baseline_status,
        "blocked_cause": str(artifact.get("blocked_cause", "")),
        "published_target_val_exact_accuracy": PUBLISHED_SUDOKU_TARGET,
        "faithful_graft_threshold": FAITHFUL_GRAFT_THRESHOLD,
        "matches_published_087": matches_published,
        "faithful_for_graft_085": faithful,
        "native_trainer_launched": artifact.get("native_trainer_launched") is True,
        "final_val_exact_accuracy": final_val,
        "final_val_exact_accuracy_rounded": _round4(final_val),
        "raw_val_trajectory_v384": rows,
        "raw_val_trajectory_v384_rounded": [
            row["val_exact_accuracy_rounded"] for row in rows
        ],
        "effective_val_trajectory_v384_rounded": [
            _round4(value) for value in effective_values
        ],
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "missing_discriminator": (
            "faithful_sudoku_baseline_candidate_source_before_diffusiongemma_scaleup"
        ),
    }


def classify_sudoku_decisive_graft(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4153: summarize Exp 4150's decisive graft result or deferral."""
    artifact = base._load_json(repo_root / EXP4150_PATH)
    graft_deferred = artifact.get("graft_deferred") is True
    verifier_value_added = artifact.get("verifier_value_added") is True
    if verifier_value_added:
        status = "filled_transferable_verifier_value_added"
    elif graft_deferred:
        status = "open_graft_deferred_baseline_below_0.85"
    else:
        status = "open_honest_null_no_transferable_value_added"

    baseline_status = artifact.get("baseline_status", {})
    baseline_val = (
        _numeric_or_none(baseline_status.get("val_exact_accuracy"))
        if isinstance(baseline_status, dict)
        else None
    )
    return {
        "gap_id": SUDOKU_GRAFT_GAP_ID,
        "status": status,
        "artifact_path": EXP4150_PATH,
        "baseline_artifact_path": _rel_or_str(
            repo_root,
            str(artifact.get("baseline_artifact_path", EXP4149_PATH)),
        ),
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "acceptance_gate_passed": artifact.get("acceptance_gate_passed") is True,
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "graft_deferred": graft_deferred,
        "verifier_value_added": verifier_value_added,
        "baseline_status": baseline_status,
        "baseline_val_exact_accuracy": baseline_val,
        "baseline_faithful_085": bool(
            isinstance(baseline_status, dict) and baseline_status.get("faithful") is True
        ),
        "candidate_source": str(artifact.get("candidate_source", "")),
        "k_candidates_per_puzzle": artifact.get("k_candidates_per_puzzle"),
        "n_candidate_pools": artifact.get("n_candidate_pools"),
        "rerank_lift_vs_vote": dict(artifact.get("rerank_lift_vs_vote", {})),
        "rft_vs_ablation_delta": dict(artifact.get("rft_vs_ablation_delta", {})),
        "estimated_passes_to_converge_for_385": dict(
            artifact.get("estimated_passes_to_converge_for_385", {})
        ),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "corrigendum_pending": list(artifact.get("corrigendum_pending", [])),
        "missing_discriminator": (
            "transferable_training_time_value_from_non_oracle_sudoku_verifier_labels"
        ),
    }


def classify_diffusiongemma_gate(sudoku_graft: dict[str, Any]) -> dict[str, Any]:
    """Return the DiffusionGemma scale-up gate from transferable value only."""
    value_added = bool(sudoku_graft.get("verifier_value_added"))
    graft_deferred = bool(sudoku_graft.get("graft_deferred"))
    if value_added:
        return {
            "state": "unlocked",
            "reason": "transferable_verifier_value_added",
            "verifier_value_added": True,
            "graft_deferred": graft_deferred,
            "uses_executable_oracle_upper_bound": False,
            "basis": "rerank_lift_vs_vote_or_rft_vs_ablation_delta",
        }
    return {
        "state": "kept_gated",
        "reason": "no_transferable_verifier_value_added",
        "verifier_value_added": False,
        "graft_deferred": graft_deferred,
        "uses_executable_oracle_upper_bound": False,
        "basis": "rerank_lift_vs_vote_or_rft_vs_ablation_delta",
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .384 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_sudoku_training_role(
        updated_registry,
        sudoku_baseline,
        sudoku_graft,
        diffusiongemma_gate_state,
    )

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4153-sudoku-baseline-reproduction",
        _sudoku_baseline_gap_block(sudoku_baseline),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4153-sudoku-decisive-graft",
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
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4153": EXP4153_ARTIFACT_PATH,
            "exp4153_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4153_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4153_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4153_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4153_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_sudoku_training_role(
    registry: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    if bool(sudoku_graft.get("verifier_value_added")):
        status = "value_added_diffusiongemma_unlocked"
    elif bool(sudoku_graft.get("graft_deferred")):
        status = "graft_deferred_baseline_below_0.85"
    else:
        status = "honest_null_no_transferable_value_added"

    role = {
        "role_id": SUDOKU_TRAINING_ROLE_ID,
        "experiment": EXP4150_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": status,
        "outcome": sudoku_graft.get("status"),
        "honest_verdict": sudoku_graft.get("honest_verdict", ""),
        "baseline_artifact": sudoku_baseline.get("artifact_path"),
        "baseline_status": sudoku_baseline.get("baseline_status"),
        "baseline_final_val_exact_accuracy": sudoku_baseline.get(
            "final_val_exact_accuracy"
        ),
        "baseline_raw_val_trajectory_v384_rounded": sudoku_baseline.get(
            "raw_val_trajectory_v384_rounded"
        ),
        "baseline_effective_val_trajectory_v384_rounded": sudoku_baseline.get(
            "effective_val_trajectory_v384_rounded"
        ),
        "matches_published_087": bool(sudoku_baseline.get("matches_published_087")),
        "faithful_for_graft_085": bool(sudoku_baseline.get("faithful_for_graft_085")),
        "graft_deferred": bool(sudoku_graft.get("graft_deferred")),
        "verifier_value_added": bool(sudoku_graft.get("verifier_value_added")),
        "executable_sudoku_verifier_as_reward_status": (
            "deferred_baseline_below_0.85_no_transferable_value_added"
            if not bool(sudoku_graft.get("verifier_value_added"))
            else "transferable_value_added"
        ),
        "rerank_lift_vs_vote": sudoku_graft.get("rerank_lift_vs_vote", {}),
        "rft_vs_ablation_delta": sudoku_graft.get("rft_vs_ablation_delta", {}),
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
        "eval_exp_4153": EXP4153_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("training_time_roles", []))
    entry["training_time_roles"] = [
        old for old in old_roles if old.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [role]


def _sudoku_baseline_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SUDOKU_BASELINE_GAP_ID}: Exp 4153 .384 Sudoku baseline trajectory status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4149_PATH}`; "
        f"baseline_status={outcome.get('baseline_status')}; "
        f"raw_val_trajectory_v384={outcome.get('raw_val_trajectory_v384_rounded')}; "
        f"effective_val_trajectory_v384={outcome.get('effective_val_trajectory_v384_rounded')}; "
        f"final_val={outcome.get('final_val_exact_accuracy_rounded')}; "
        f"matches_published_087={str(bool(outcome.get('matches_published_087'))).lower()}; "
        f"faithful_for_graft_085={str(bool(outcome.get('faithful_for_graft_085'))).lower()}; "
        f"published_target={outcome.get('published_target_val_exact_accuracy')}.\n"
        "- failure mode: the .384 continuation did not produce real training progress; "
        "the pass1/pass2/pass3 no-op lineage carried forward and pass4 preserved the "
        "0.2782 baseline rather than approaching the published target.\n"
        "- missing discriminator: faithful Sudoku baseline candidate source before "
        "DiffusionGemma scale-up or verifier-as-reward claims.\n"
        "- candidate design: resolve the timer/no-op checkpoint lineage or create a clean "
        "contiguous baseline before rerunning the graft.\n"
        "- priority: high\n"
    )


def _sudoku_graft_gap_block(
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> str:
    rerank = sudoku_graft.get("rerank_lift_vs_vote", {})
    rft = sudoku_graft.get("rft_vs_ablation_delta", {})
    return (
        f"### {SUDOKU_GRAFT_GAP_ID}: Exp 4153 .384 Sudoku decisive executable-verifier graft status\n"
        f"- status: {sudoku_graft['status']}\n"
        f"- evidence: `{EXP4150_PATH}` with baseline `{EXP4149_PATH}`; "
        f"baseline_final_val={sudoku_baseline.get('final_val_exact_accuracy_rounded')}; "
        f"baseline_faithful_085={str(bool(sudoku_graft.get('baseline_faithful_085'))).lower()}; "
        f"graft_deferred={str(bool(sudoku_graft.get('graft_deferred'))).lower()}; "
        f"candidate_source={sudoku_graft.get('candidate_source')}; "
        f"n_candidate_pools={sudoku_graft.get('n_candidate_pools')}; "
        f"rerank_lift_vs_vote_delta={rerank.get('delta')}; "
        f"rerank_lift_vs_vote_status={rerank.get('status')}; "
        f"rft_vs_ablation_delta={rft.get('delta')}; "
        f"rft_vs_ablation_delta_status={rft.get('status')}; "
        f"verifier_value_added={str(bool(sudoku_graft.get('verifier_value_added'))).lower()}; "
        f"diffusiongemma_gate_state={diffusiongemma_gate_state.get('state')}.\n"
        "- failure mode: Exp 4150 correctly deferred the graft because the baseline was "
        "below the faithful 0.85 gate; no rerank or RFT candidate source was created, "
        "so there is no transferable verifier-value-added evidence.\n"
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
        and gap4.get("eval", {}).get("eval_exp_4153") == EXP4153_ARTIFACT_PATH
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
    """Build the Exp 4153 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    baseline_recorded = SUDOKU_BASELINE_GAP_ID in gaps_updated
    graft_recorded = SUDOKU_GRAFT_GAP_ID in gaps_updated
    prefix = "complete:" if guard_ok and baseline_recorded and graft_recorded else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4153_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4153_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v384_truth_"
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
            EXP4149_PATH,
            EXP4150_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4153", "SCENARIO-VERIFY-4153"],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4153_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4153_verifier_registry_gaps_hygiene.v1",
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
            EXP4149_PATH,
            EXP4150_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4153", "SCENARIO-VERIFY-4153"],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4153 fields before writing the artifact."""
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
        raise ValueError("field_principles must match the required Exp 4153 principles")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4153 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4153_ARTIFACT_PATH
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
    print(f"Wrote {REPO_ROOT / EXP4153_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
