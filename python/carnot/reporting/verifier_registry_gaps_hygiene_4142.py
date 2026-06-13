"""Exp 4142 registry/gaps hygiene for .383 verifier-relevant outcomes.

Spec refs: REQ-VERIFY-4142, SCENARIO-VERIFY-4142.

This reconciler is intentionally offline. It first proves the cached GAP-4
ARC-1 regression guard still reproduces the canonical vote-to-gated lift, then
records the Sudoku outcome without laundering an oracle upper bound into a
transferable verifier-as-reward win. That distinction matters because an
executable Sudoku validity check is an oracle on unique-solution puzzles; it
can show headroom, but the DiffusionGemma scale-up gate needs value from the
transferable ensemble rerank or the RFT label contrast.
"""

from __future__ import annotations

import gzip
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4131 as exp4131


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "cached_gap4_replay_and_ledger_reconciliation"

EXP4142_ARTIFACT_PATH = "results/experiment_4142_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = "results/arc3_gap3_stage2_eval_pool.json.gz"
ARC1_PROGRAMS_PATH = "results/arc3_gap4_induced_programs.json"
EXP4138_PATH = "results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json"
EXP4139_PATH = "results/experiment_4139_decisive_verifier_graft_sudoku.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
SUDOKU_BASELINE_GAP_ID = "GAP-SUDOKU-BASELINE-REPRODUCTION-4138"
SUDOKU_GRAFT_GAP_ID = "GAP-SUDOKU-EXECUTABLE-VERIFIER-4139"
SUDOKU_TRAINING_ROLE_ID = "sudoku_executable_verifier_training_time_4139"
PUBLISHED_SUDOKU_TARGET = 0.87

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
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .383 truth.",
    "regression_guard_passed": (
        "Bare bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched so the gap backlog stays the honest complement "
        "of the registry."
    ),
    "diffusiongemma_gate_state": (
        "Records whether the .383 graft's verifier_value_added (on the transferable ensemble + "
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


def _load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("expected JSON object")
    return loaded


def _check_resource(
    repo_root: Path,
    resource: str,
    rel_paths: list[str],
    loader: Any,
) -> dict[str, Any]:
    paths = [repo_root / rel_path for rel_path in rel_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        return {"resource": resource, "available": False, "detail": f"missing: {missing}"}
    try:
        for path in paths:
            loader(path)
    except Exception as exc:  # pragma: no cover - exact parse exceptions vary by parser.
        return {"resource": resource, "available": False, "detail": f"parse_error: {exc}"}
    return {"resource": resource, "available": True, "detail": ", ".join(rel_paths)}


def _load_registry_for_check(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("registry is not a mapping")
    if "verifiers" not in loaded:
        raise ValueError("registry missing verifiers")
    return loaded


def _load_gaps_for_check(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("gaps markdown is empty")
    return text


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4142: verify cached fixtures and ledgers before replay."""
    checks = [
        _check_resource(
            repo_root,
            "gap4_arc1_candidate_fixtures",
            [ARC1_POOL_PATH, ARC1_PROGRAMS_PATH],
            lambda path: _load_gzip_json(path)
            if path.suffix == ".gz"
            else base._load_json(path),
        ),
        _check_resource(repo_root, "verifier_registry", [REGISTRY_PATH], _load_registry_for_check),
        _check_resource(repo_root, "verifier_gaps", [GAPS_PATH], _load_gaps_for_check),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-VERIFY-4142: replay GAP-4 ARC-1 from cached artifacts only."""
    return exp4131.replay_gap4_arc1(repo_root)


def _trajectory_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = artifact.get("val_trajectory_383", [])
    if not isinstance(rows, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        val = _numeric_or_none(row.get("val_exact_accuracy"))
        cleaned.append(
            {
                "pass_index": row.get("pass_index", index),
                "label": str(row.get("label", "")),
                "experiment": str(row.get("experiment", "")),
                "status": str(row.get("status", "")),
                "val_exact_accuracy": val,
                "val_exact_accuracy_rounded": _round4(val),
                "delta_vs_previous": _numeric_or_none(row.get("delta_vs_previous")),
            }
        )
    return cleaned


def classify_sudoku_baseline(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4142: summarize the Exp 4138 baseline trajectory honestly."""
    artifact = base._load_json(repo_root / EXP4138_PATH)
    rows = _trajectory_rows(artifact)
    measured = [
        row["val_exact_accuracy"]
        for row in rows
        if row["val_exact_accuracy"] is not None
    ]
    final_val = _numeric_or_none(artifact.get("val_exact_accuracy"))
    if final_val is None:
        final_val = _numeric_or_none(
            artifact.get("baseline", {}).get("val_exact_accuracy")
            if isinstance(artifact.get("baseline"), dict)
            else None
        )
    if final_val is None and measured:
        final_val = measured[-1]

    baseline_status = str(artifact.get("baseline_status", "unknown"))
    matches_published = artifact.get("matches_published_087") is True
    near_faithful = artifact.get("near_faithful_080") is True
    final_text = f"{final_val:.4f}" if final_val is not None else "unknown"
    if matches_published:
        status = f"reproduced_val_{final_text}"
    elif baseline_status == "config-blocked":
        status = f"open_baseline_config_blocked_val_{final_text}"
    else:
        status = f"open_baseline_not_reproduced_val_{final_text}"

    return {
        "gap_id": SUDOKU_BASELINE_GAP_ID,
        "status": status,
        "artifact_path": EXP4138_PATH,
        "source_artifacts": [EXP4138_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "published_target_val_exact_accuracy": PUBLISHED_SUDOKU_TARGET,
        "matches_published_087": matches_published,
        "near_faithful_080": near_faithful,
        "baseline_status": baseline_status,
        "config_blocked": baseline_status == "config-blocked",
        "final_val_exact_accuracy": final_val,
        "final_val_exact_accuracy_rounded": _round4(final_val),
        "val_trajectory_383": rows,
        "val_trajectory_383_rounded": [
            row["val_exact_accuracy_rounded"] for row in rows
        ],
        "measured_val_trajectory": measured,
        "measured_val_trajectory_rounded": [_round4(value) for value in measured],
        "estimated_passes_to_converge": artifact.get("estimated_passes_to_converge"),
        "stable_checkpoint_path": str(artifact.get("stable_checkpoint_path", "")),
        "suspected_cause": str(artifact.get("suspected_cause", "")),
        "corrected_config_recommendation": str(
            artifact.get("corrected_config_recommendation", "")
        ),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "missing_discriminator": (
            "faithful_sudoku_baseline_candidate_source_before_diffusiongemma_scaleup"
        ),
    }


def classify_sudoku_decisive_graft(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4142: summarize Exp 4139 without counting the oracle as value."""
    artifact = base._load_json(repo_root / EXP4139_PATH)
    graft_deferred = artifact.get("graft_deferred") is True
    verifier_value_added = artifact.get("verifier_value_added") is True
    headroom_present = artifact.get("headroom_present") is True
    if verifier_value_added:
        status = "filled_transferable_verifier_value_added"
    elif graft_deferred:
        status = "open_graft_deferred_no_transferable_value_added"
    elif not headroom_present:
        status = "open_uninformative_no_headroom_no_transferable_value_added"
    else:
        status = "open_honest_null_no_transferable_value_added"

    return {
        "gap_id": SUDOKU_GRAFT_GAP_ID,
        "status": status,
        "artifact_path": EXP4139_PATH,
        "baseline_artifact_path": _rel_or_str(
            repo_root,
            str(artifact.get("baseline_artifact_path", EXP4138_PATH)),
        ),
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "flagged_adversarial": artifact.get("flagged_adversarial") is True,
        "headroom_present": headroom_present,
        "false_negative_risk": artifact.get("false_negative_risk") is True,
        "graft_deferred": graft_deferred,
        "verifier_value_added": verifier_value_added,
        "verifier_value_added_basis": list(artifact.get("verifier_value_added_basis", [])),
        "baseline_val_exact_accuracy": _numeric_or_none(
            artifact.get("baseline_val_exact_accuracy")
        ),
        "baseline_matches_published_087": artifact.get("baseline_matches_published_087")
        is True,
        "baseline_near_faithful_080": artifact.get("baseline_near_faithful_080") is True,
        "oracle_pass_at_k": _numeric_or_none(artifact.get("oracle_pass_at_k")),
        "vote_pass_at_1": _numeric_or_none(artifact.get("vote_pass_at_1")),
        "oracle_vs_vote_gap": _numeric_or_none(artifact.get("oracle_vs_vote_gap")),
        "executable_verifier_is_oracle": artifact.get("executable_verifier_is_oracle")
        is True,
        "executable_oracle_upper_bound": dict(
            artifact.get("executable_oracle_upper_bound", {})
        ),
        "ensemble_rerank_lift_vs_vote": dict(
            artifact.get("ensemble_rerank_lift_vs_vote", {})
        ),
        "rft_vs_ablation_delta": dict(artifact.get("rft_vs_ablation_delta", {})),
        "estimated_passes_to_converge_for_384": artifact.get(
            "estimated_passes_to_converge_for_384"
        ),
        "preconditions_checked": list(artifact.get("preconditions_checked", [])),
        "candidate_source": str(artifact.get("candidate_source", "")),
        "n_candidate_pools": artifact.get("n_candidate_pools"),
        "missing_discriminator": (
            "transferable_training_time_value_from_non_oracle_sudoku_verifier_labels"
        ),
    }


def _rel_or_str(repo_root: Path, value: str) -> str:
    path = Path(value)
    try:
        return str(path.relative_to(repo_root)) if path.is_absolute() else value
    except ValueError:
        return value


def classify_diffusiongemma_gate(sudoku_graft: dict[str, Any]) -> dict[str, Any]:
    """Return the DiffusionGemma scale-up gate from transferable value only."""
    value_added = bool(sudoku_graft.get("verifier_value_added"))
    headroom_present = bool(sudoku_graft.get("headroom_present"))
    if value_added:
        return {
            "state": "unlocked",
            "reason": "transferable_verifier_value_added",
            "verifier_value_added": True,
            "headroom_present": headroom_present,
            "uses_executable_oracle_upper_bound": False,
            "basis": "ensemble_rerank_lift_vs_vote_or_rft_vs_ablation_delta_not_oracle",
        }
    return {
        "state": "kept_gated",
        "reason": "no_transferable_verifier_value_added",
        "verifier_value_added": False,
        "headroom_present": headroom_present,
        "uses_executable_oracle_upper_bound": False,
        "basis": "ensemble_rerank_lift_vs_vote_or_rft_vs_ablation_delta_not_oracle",
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gaps text with the .383 outcomes represented."""
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
        "exp4142-sudoku-baseline-reproduction",
        _sudoku_baseline_gap_block(sudoku_baseline),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4142-sudoku-decisive-graft",
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
            "eval_exp_4142": EXP4142_ARTIFACT_PATH,
            "exp4142_regression_guard_passed": bool(
                offline_replay.get("regression_guard_passed")
            ),
            "exp4142_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4142_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4142_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4142_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
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
    old_roles = list(entry.get("training_time_roles", []))
    if bool(sudoku_graft.get("verifier_value_added")):
        status = "value_added_diffusiongemma_unlocked"
    elif bool(sudoku_graft.get("graft_deferred")):
        status = "graft_deferred_no_headroom"
    else:
        status = "honest_null_no_transferable_value_added"

    role = {
        "role_id": SUDOKU_TRAINING_ROLE_ID,
        "experiment": EXP4139_PATH,
        "role": "candidate_trm_training_time_reward_signal_executable_domain",
        "status": status,
        "outcome": sudoku_graft.get("status"),
        "honest_verdict": sudoku_graft.get("honest_verdict", ""),
        "baseline_artifact": sudoku_baseline.get("artifact_path"),
        "baseline_status": sudoku_baseline.get("baseline_status"),
        "baseline_final_val_exact_accuracy": sudoku_baseline.get("final_val_exact_accuracy"),
        "baseline_val_trajectory_383_rounded": sudoku_baseline.get(
            "val_trajectory_383_rounded"
        ),
        "matches_published_087": bool(sudoku_baseline.get("matches_published_087")),
        "near_faithful_080": bool(sudoku_baseline.get("near_faithful_080")),
        "headroom_present": bool(sudoku_graft.get("headroom_present")),
        "graft_deferred": bool(sudoku_graft.get("graft_deferred")),
        "verifier_value_added": bool(sudoku_graft.get("verifier_value_added")),
        "executable_sudoku_verifier_as_reward_status": (
            "deferred_oracle_upper_bound_only_no_transferable_value_added"
            if not bool(sudoku_graft.get("verifier_value_added"))
            else "transferable_value_added"
        ),
        "executable_oracle_upper_bound": sudoku_graft.get(
            "executable_oracle_upper_bound", {}
        ),
        "ensemble_rerank_lift_vs_vote": sudoku_graft.get(
            "ensemble_rerank_lift_vs_vote", {}
        ),
        "rft_vs_ablation_delta": sudoku_graft.get("rft_vs_ablation_delta", {}),
        "diffusiongemma_gate_state": diffusiongemma_gate_state,
        "eval_exp_4142": EXP4142_ARTIFACT_PATH,
    }
    entry["training_time_roles"] = [
        old for old in old_roles if old.get("role_id") != SUDOKU_TRAINING_ROLE_ID
    ] + [role]


def _sudoku_baseline_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {SUDOKU_BASELINE_GAP_ID}: Exp 4142 .383 Sudoku baseline trajectory status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4138_PATH}`; "
        f"baseline_status={outcome.get('baseline_status')}; "
        f"val_trajectory_383={outcome.get('val_trajectory_383_rounded')}; "
        f"measured_val_trajectory={outcome.get('measured_val_trajectory_rounded')}; "
        f"final_val={outcome.get('final_val_exact_accuracy_rounded')}; "
        f"matches_published_087={str(bool(outcome.get('matches_published_087'))).lower()}; "
        f"near_faithful_080={str(bool(outcome.get('near_faithful_080'))).lower()}; "
        f"published_target={outcome.get('published_target_val_exact_accuracy')}; "
        f"estimated_passes_to_converge={outcome.get('estimated_passes_to_converge')}.\n"
        "- failure mode: the .383 continuation did not produce new validation progress "
        "because the baseline lineage was config-blocked before pass4, so the "
        "Sudoku candidate source remains far below the published 0.87 target.\n"
        "- missing discriminator: faithful Sudoku baseline candidate source before "
        "DiffusionGemma scale-up or verifier-as-reward claims.\n"
        "- candidate design: fix the Timer/config-blocked resume path or run a clean "
        "contiguous baseline before rerunning the graft.\n"
        "- priority: high\n"
    )


def _sudoku_graft_gap_block(
    sudoku_baseline: dict[str, Any],
    sudoku_graft: dict[str, Any],
    diffusiongemma_gate_state: dict[str, Any],
) -> str:
    oracle = sudoku_graft.get("executable_oracle_upper_bound", {})
    ensemble = sudoku_graft.get("ensemble_rerank_lift_vs_vote", {})
    rft = sudoku_graft.get("rft_vs_ablation_delta", {})
    return (
        f"### {SUDOKU_GRAFT_GAP_ID}: Exp 4142 .383 Sudoku decisive executable-verifier graft status\n"
        f"- status: {sudoku_graft['status']}\n"
        f"- evidence: `{EXP4139_PATH}` with baseline `{EXP4138_PATH}`; "
        f"baseline_final_val={sudoku_baseline.get('final_val_exact_accuracy_rounded')}; "
        f"headroom_present={str(bool(sudoku_graft.get('headroom_present'))).lower()}; "
        f"graft_deferred={str(bool(sudoku_graft.get('graft_deferred'))).lower()}; "
        f"executable_verifier_is_oracle="
        f"{str(bool(sudoku_graft.get('executable_verifier_is_oracle'))).lower()}; "
        f"executable_oracle_upper_bound_delta={oracle.get('delta')}; "
        f"ensemble_rerank_lift_vs_vote_delta={ensemble.get('delta')}; "
        f"ensemble_rerank_lift_vs_vote_status={ensemble.get('status')}; "
        f"rft_vs_ablation_delta={rft.get('delta')}; "
        f"rft_vs_ablation_delta_status={rft.get('status')}; "
        f"verifier_value_added={str(bool(sudoku_graft.get('verifier_value_added'))).lower()}; "
        f"diffusiongemma_gate_state={diffusiongemma_gate_state.get('state')}.\n"
        "- failure mode: executable Sudoku validity is an oracle upper bound on "
        "unique-solution Sudoku, not a transferable verifier reward. The non-oracle "
        "ensemble rerank added no measured lift, and the RFT label contrast was "
        "deferred because the baseline was below the near-faithful gate.\n"
        "- missing discriminator: transferable training-time value from non-oracle "
        "Sudoku verifier labels beyond vote labels.\n"
        "- candidate design: keep DiffusionGemma gated until the transferable ensemble "
        "rerank or RFT A-vs-B label contrast shows value with selectable headroom.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4142") == EXP4142_ARTIFACT_PATH
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
    """Build the Exp 4142 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    baseline_recorded = SUDOKU_BASELINE_GAP_ID in gaps_updated
    graft_recorded = SUDOKU_GRAFT_GAP_ID in gaps_updated
    prefix = "complete:" if guard_ok and baseline_recorded and graft_recorded else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4142_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4142_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v383_truth_"
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
            EXP4138_PATH,
            EXP4139_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(
    preflight: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    artifact = {
        "experiment": "experiment_4142_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4142_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "sudoku_baseline": {"status": "blocked_precondition", "gap_id": SUDOKU_BASELINE_GAP_ID},
        "sudoku_decisive_graft": {"status": "blocked_precondition", "gap_id": SUDOKU_GRAFT_GAP_ID},
        "diffusiongemma_gate_state": {
            "state": "blocked",
            "reason": blocked,
            "verifier_value_added": False,
            "headroom_present": False,
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
            EXP4138_PATH,
            EXP4139_PATH,
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4142 fields before writing the artifact."""
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
        raise ValueError("field_principles must match the required Exp 4142 principles")


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4142 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4142_ARTIFACT_PATH
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
    print(f"Wrote {REPO_ROOT / EXP4142_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
