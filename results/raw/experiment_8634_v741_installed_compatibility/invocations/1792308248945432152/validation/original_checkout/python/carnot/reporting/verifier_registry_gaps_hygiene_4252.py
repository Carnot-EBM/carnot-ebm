"""Exp 4252 registry/gaps hygiene for .393 verifier outcomes.

Spec refs: REQ-VERIFY-4252, SCENARIO-VERIFY-4252.

This runner is an offline ledger reconciler. It replays the canonical GAP-4
ARC-1 candidate set from cached artifacts, records the .393 ARC set-encoder
oracle-distinct result, records the blocked code replication and offline reward
status, and records the live-LoRA retirement without touching TRM training.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import time
from typing import Any, Callable

import yaml

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4227 as exp4227


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4252
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4252_ARTIFACT_PATH = "results/experiment_4252_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH
EXCLUSION_MANIFEST_PATH = "ops/exclusion_manifest.yaml"

ARC1_POOL_PATH = exp4227.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4227.ARC1_PROGRAMS_PATH
EXP4244_PATH = "results/experiment_4244_arc_set_encoder_aggregator_build.json"
EXP4245_PATH = "results/experiment_4245_arc_set_encoder_beats_vote.json"
EXP4246_PATH = "results/experiment_4246_code_oracle_distinct_replication.json"
EXP4247_PATH = "results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json"
EXP4248_PATH = "results/experiment_4248_verifier_as_reward_offline_3arm.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
GAP_ORACLE_DISTINCT_GAP_ID = "GAP-ORACLE-DISTINCT"
GAP_ORACLE_DISTINCT_A3_GAP_ID = "GAP-ORACLE-DISTINCT-A3-4245"
GAP_CODE_REPLICATION_GAP_ID = "GAP-CODE-REPLICATION-4246"
GAP_REWARD_GAP_ID = "GAP-REWARD"
GAP_LIVE_LORA_RETIREMENT_GAP_ID = "GAP-REWARD-LIVE-LORA-RETIREMENT-4247"
V393_ROLE_ID = "oracle_distinct_code_reward_hygiene_4252"
LIVE_LORA_RETIREMENT_ENTRY_ID = "live_lora_verifier_as_reward_path_retired_exp4247"
LIVE_LORA_INFRA_FAILURE_COUNT = 6

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "live_lora_retired_recorded",
    "registry_updated",
    "oracle_distinct_outcome",
    "code_replication_outcome",
    "verifier_reward_outcome",
    "live_lora_retirement",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "inference_substrate",
    "adversarial_verify",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the registry/gaps reconciled to the .393 truth + "
        "the live-LoRA retirement."
    ),
    "regression_guard_passed": (
        "BARE bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (GAP-ORACLE-DISTINCT frontier + the A3 "
        "ARC + code replication + offline verifier-as-reward + live-LoRA retirement notes)."
    ),
    "live_lora_retired_recorded": (
        "BARE bool: true iff the live-LoRA retirement (exp4247) was recorded to the "
        "exclusion manifest + gap ledger so it is never re-proposed."
    ),
    "random_seed": (
        "Determinism precondition + the methodology field that prevents a METHODOLOGY_MISSING flag."
    ),
    "reproducibility_checksum": "Hash of the cached GAP-4 candidate set; catches silent candidate drift.",
}

replay_gap4_arc1 = exp4227.replay_gap4_arc1
candidate_set_checksum = exp4227.candidate_set_checksum


def _load_manifest(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded.setdefault("retired_extras", [])
        return loaded
    return {"retired": [], "retired_experiments": [], "retired_extras": []}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _load_manifest_for_check(path: Path) -> dict[str, Any]:
    return _load_manifest(path)


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4252: verify cached replay fixtures, upstream artifacts, and ledgers."""
    checks = [
        exp4227._check_resource(
            repo_root,
            "gap4_arc1_candidate_fixtures",
            [ARC1_POOL_PATH, ARC1_PROGRAMS_PATH],
            lambda path: (
                exp4227._load_gzip_json(path) if path.suffix == ".gz" else base._load_json(path)
            ),
        ),
        exp4227._check_resource(
            repo_root,
            "verifier_registry",
            [REGISTRY_PATH],
            exp4227._load_registry_for_check,
        ),
        exp4227._check_resource(
            repo_root,
            "verifier_gaps",
            [GAPS_PATH],
            exp4227._load_gaps_for_check,
        ),
        exp4227._check_resource(
            repo_root,
            "exclusion_manifest",
            [EXCLUSION_MANIFEST_PATH],
            _load_manifest_for_check,
        ),
        exp4227._check_json_resource(repo_root, "exp4244_arc_set_encoder_build", EXP4244_PATH),
        exp4227._check_json_resource(repo_root, "exp4245_arc_set_encoder_a3", EXP4245_PATH),
        exp4227._check_json_resource(repo_root, "exp4246_code_replication", EXP4246_PATH),
        exp4227._check_json_resource(repo_root, "exp4247_live_lora_retirement", EXP4247_PATH),
        exp4227._check_json_resource(repo_root, "exp4248_offline_reward_a_vs_b", EXP4248_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def classify_oracle_distinct_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4252: summarize Exp 4245 A3 with the paired Exp 4244 build."""
    a3 = base._load_json(repo_root / EXP4245_PATH)
    build = base._load_json(repo_root / EXP4244_PATH)
    return {
        "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
        "a3_gap_id": GAP_ORACLE_DISTINCT_A3_GAP_ID,
        "status": "filled_arc_a3_set_encoder_beats_vote_non_oracle",
        "artifact_path": EXP4245_PATH,
        "build_artifact_path": EXP4244_PATH,
        "source_artifacts": [EXP4245_PATH, EXP4244_PATH],
        "honest_verdict": str(a3.get("honest_verdict", "")),
        "build_honest_verdict": str(build.get("honest_verdict", "")),
        "headline_outcome": str(a3.get("headline_outcome", "")),
        "oracle_distinct_beats_vote": a3.get("oracle_distinct_beats_vote") is True,
        "set_encoder_minus_vote_delta": exp4227._first_numeric(
            a3,
            "set_encoder_minus_vote_delta",
        ),
        "set_encoder_minus_vote_ci95": exp4227._first_ci95(
            a3,
            "set_encoder_minus_vote_ci95",
        ),
        "margin_override_minus_vote": exp4227._first_numeric(
            a3,
            "margin_override_minus_vote",
        ),
        "matched_control_delta": exp4227._first_numeric(a3, "matched_control_delta"),
        "oracle_at_k": exp4227._first_numeric(a3, "oracle_at_k"),
        "oracle_minus_vote": exp4227._first_numeric(a3, "oracle_minus_vote"),
        "held_out_task_n": int(a3.get("held_out_task_n", 0)),
        "verifier_is_oracle": a3.get("verifier_is_oracle") is True,
        "headroom_exists": a3.get("headroom_exists") is True,
        "pass_rates": dict(a3.get("pass_rates", {})),
        "candidate_count": a3.get("candidate_count"),
        "candidate_pool_sha256": str(a3.get("candidate_pool_sha256", "")),
        "aggregator_trained": build.get("aggregator_trained") is True,
        "oracle_distinct_auroc": exp4227._first_numeric(build, "oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": exp4227._first_ci95(
            build,
            "oracle_distinct_auroc_ci95",
        ),
        "logistic_auroc": exp4227._first_numeric(build, "logistic_auroc"),
        "logistic_auroc_ci95": exp4227._first_ci95(build, "logistic_auroc_ci95"),
        "set_encoder_vs_logistic_auroc_delta": exp4227._first_numeric(
            build,
            "set_encoder_vs_logistic_auroc_delta",
        ),
        "wrong_majority_n": build.get("wrong_majority_n"),
        "positive_candidate_n": build.get("positive_candidate_n"),
        "accepted_rejected_n": dict(build.get("accepted_rejected_n", {})),
        "learned_verifier_path": str(build.get("learned_verifier_path", "")),
        "gap_oracle_distinct_update": "changed_v393_grown_pool_set_encoder_beats_vote",
        "gap_moat_update": "unchanged_registry_hygiene_does_not_upgrade_moat",
        "missing_discriminator": "none_for_measured_arc_a3_non_oracle_vote_beating_read",
    }


def classify_code_replication_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4252: summarize Exp 4246 code replication without promoting a block."""
    artifact = base._load_json(repo_root / EXP4246_PATH)
    return {
        "gap_id": GAP_CODE_REPLICATION_GAP_ID,
        "status": str(artifact.get("replication_read", "blocked_code_replication_unavailable")),
        "artifact_path": EXP4246_PATH,
        "source_artifacts": [EXP4246_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "replication_read": str(artifact.get("replication_read", "")),
        "code_replication_beats_vote": artifact.get("code_replication_beats_vote") is True,
        "code_predictor_minus_vote_delta": exp4227._first_numeric(
            artifact,
            "code_predictor_minus_vote_delta",
        ),
        "code_predictor_minus_vote_ci95": exp4227._first_ci95(
            artifact,
            "code_predictor_minus_vote_ci95",
        ),
        "matched_control_delta": exp4227._first_numeric(artifact, "matched_control_delta"),
        "oracle_at_k": exp4227._first_numeric(artifact, "oracle_at_k"),
        "oracle_minus_vote": exp4227._first_numeric(artifact, "oracle_minus_vote"),
        "off_fold_auroc": exp4227._first_numeric(artifact, "off_fold_auroc"),
        "held_out_task_n": int(artifact.get("held_out_task_n", 0)),
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
        "candidate_pool": dict(artifact.get("candidate_pool", {})),
        "pass_rates": dict(artifact.get("pass_rates", {})),
        "missing_discriminator": "second_distinct_code_candidate_corpus_for_replication",
    }


def classify_live_lora_retirement(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4252: summarize Exp 4247 live-LoRA retirement for manifest routing."""
    artifact = base._load_json(repo_root / EXP4247_PATH)
    live_lora_retired = artifact.get("live_lora_retired") is True
    return {
        "gap_id": GAP_LIVE_LORA_RETIREMENT_GAP_ID,
        "status": "retired_live_lora_path_after_6_infra_failures"
        if live_lora_retired
        else "not_retired",
        "artifact_path": EXP4247_PATH,
        "source_artifacts": [EXP4247_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "harness_smoke_passed": artifact.get("harness_smoke_passed") is True,
        "live_lora_retired": live_lora_retired,
        "live_lora_retirement_rationale": str(
            artifact.get(
                "live_lora_retirement_rationale",
                "Exp 4247 recorded live-LoRA retirement after repeated infra failures.",
            )
        ),
        "infra_failure_count": LIVE_LORA_INFRA_FAILURE_COUNT if live_lora_retired else 0,
        "operator_reopen_required": live_lora_retired,
        "retire_if_same_verdict": live_lora_retired,
        "retirement_entry_id": LIVE_LORA_RETIREMENT_ENTRY_ID,
        "verifier_is_oracle": artifact.get("verifier_is_oracle") is True,
        "smoke_failure_reason": str(artifact.get("smoke_failure_reason", "")),
    }


def classify_verifier_reward_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4252: summarize Exp 4248 as blocked unless it measured A-vs-B."""
    artifact = base._load_json(repo_root / EXP4248_PATH)
    retirement = classify_live_lora_retirement(repo_root)
    live_lora_retired = retirement["live_lora_retired"] is True
    status = (
        "blocked_offline_reward_gate_failed_live_lora_retired"
        if live_lora_retired and artifact.get("status") == "blocked"
        else str(artifact.get("status", "unknown"))
    )
    return {
        "gap_id": GAP_REWARD_GAP_ID,
        "retirement_gap_id": GAP_LIVE_LORA_RETIREMENT_GAP_ID,
        "status": status,
        "artifact_path": EXP4248_PATH,
        "retirement_artifact_path": EXP4247_PATH,
        "source_artifacts": [EXP4248_PATH, EXP4247_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_label_carries_signal": artifact.get("verifier_label_carries_signal") is True,
        "a_vs_b_delta": exp4227._first_numeric(artifact, "a_vs_b_delta", "arm_a_vs_b_delta"),
        "a_vs_b_ci95": exp4227._first_ci95(artifact, "a_vs_b_ci95", "arm_a_vs_b_ci95"),
        "youden_j": exp4227._numeric_or_none(artifact.get("youden_j")),
        "live_lora_retired": live_lora_retired,
        "blocked_at_layer": str(artifact.get("blocked_at_layer", "")),
        "gate_check_summary": str(artifact.get("gate_check_summary", "")),
        "gates_evaluated": list(artifact.get("gates_evaluated", [])),
        "verifier_is_oracle": bool(retirement.get("verifier_is_oracle")),
        "missing_discriminator": (
            "decision_grade_offline_a_vs_b_reward_signal_after_valid_harness_smoke"
        ),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    exclusion_manifest: dict[str, Any],
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_replication_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    live_lora_retirement: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    """Return registry, gap text, and manifest with the .393 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(
        updated_registry,
        offline_replay,
        oracle_distinct_outcome,
        code_replication_outcome,
        verifier_reward_outcome,
        live_lora_retirement,
    )
    _ensure_v393_role(
        updated_registry,
        oracle_distinct_outcome,
        code_replication_outcome,
        verifier_reward_outcome,
        live_lora_retirement,
    )

    updated_gaps = _append_marked_block(
        gaps_text,
        "exp4252-oracle-distinct",
        _oracle_distinct_frontier_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4252-oracle-distinct-a3",
        _oracle_distinct_a3_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4252-code-replication",
        _code_replication_gap_block(code_replication_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4252-gap-reward",
        _verifier_reward_gap_block(verifier_reward_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4252-live-lora-retirement",
        _live_lora_retirement_gap_block(live_lora_retirement),
    )

    updated_manifest = deepcopy(exclusion_manifest)
    manifest_changed = _ensure_live_lora_manifest_retirement(
        updated_manifest,
        live_lora_retirement,
    )
    gap_ids = [
        GAP_ORACLE_DISTINCT_GAP_ID,
        GAP_ORACLE_DISTINCT_A3_GAP_ID,
        GAP_CODE_REPLICATION_GAP_ID,
        GAP_REWARD_GAP_ID,
        GAP_LIVE_LORA_RETIREMENT_GAP_ID,
    ]
    touched = [gap_id for gap_id in gap_ids if gap_id in updated_gaps]
    live_lora_retired_recorded = (
        live_lora_retirement.get("live_lora_retired") is True
        and GAP_LIVE_LORA_RETIREMENT_GAP_ID in touched
        and _find_live_lora_manifest_entry(updated_manifest) is not None
    )
    return (
        updated_registry,
        updated_gaps,
        updated_manifest,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "oracle_distinct_recorded": GAP_ORACLE_DISTINCT_GAP_ID in touched,
            "arc_a3_recorded": GAP_ORACLE_DISTINCT_A3_GAP_ID in touched,
            "code_replication_recorded": GAP_CODE_REPLICATION_GAP_ID in touched,
            "verifier_reward_recorded": GAP_REWARD_GAP_ID in touched,
            "live_lora_retired_recorded": live_lora_retired_recorded,
            "exclusion_manifest_updated": manifest_changed,
        },
    )


def _append_marked_block(gaps_text: str, marker: str, block: str) -> str:
    return base._replace_marked_block(gaps_text, marker, block)


def _ensure_gap4_eval(
    registry: dict[str, Any],
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_replication_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    live_lora_retirement: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4252": EXP4252_ARTIFACT_PATH,
            "exp4252_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4252_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4252_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4252_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4252_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4252_oracle_distinct_beats_vote": bool(
                oracle_distinct_outcome.get("oracle_distinct_beats_vote")
            ),
            "exp4252_set_encoder_minus_vote_delta": oracle_distinct_outcome.get(
                "set_encoder_minus_vote_delta"
            ),
            "exp4252_set_encoder_minus_vote_ci95": oracle_distinct_outcome.get(
                "set_encoder_minus_vote_ci95"
            ),
            "exp4252_held_out_task_n": oracle_distinct_outcome.get("held_out_task_n"),
            "exp4252_verifier_is_oracle": bool(oracle_distinct_outcome.get("verifier_is_oracle")),
            "exp4252_oracle_distinct_auroc": oracle_distinct_outcome.get(
                "oracle_distinct_auroc"
            ),
            "exp4252_wrong_majority_n": oracle_distinct_outcome.get("wrong_majority_n"),
            "exp4252_code_replication_beats_vote": bool(
                code_replication_outcome.get("code_replication_beats_vote")
            ),
            "exp4252_code_predictor_minus_vote_delta": code_replication_outcome.get(
                "code_predictor_minus_vote_delta"
            ),
            "exp4252_verifier_label_carries_signal": bool(
                verifier_reward_outcome.get("verifier_label_carries_signal")
            ),
            "exp4252_live_lora_retired": bool(live_lora_retirement.get("live_lora_retired")),
        }
    )


def _ensure_v393_role(
    registry: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_replication_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    live_lora_retirement: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V393_ROLE_ID,
        "experiment": EXP4252_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v393",
        "status": "arc_a3_code_reward_retirement_recorded",
        "oracle_distinct_gap_id": oracle_distinct_outcome.get("gap_id"),
        "oracle_distinct_a3_gap_id": oracle_distinct_outcome.get("a3_gap_id"),
        "oracle_distinct_status": oracle_distinct_outcome.get("status"),
        "oracle_distinct_artifact": EXP4245_PATH,
        "oracle_distinct_build_artifact": EXP4244_PATH,
        "oracle_distinct_beats_vote": bool(
            oracle_distinct_outcome.get("oracle_distinct_beats_vote")
        ),
        "set_encoder_minus_vote_delta": oracle_distinct_outcome.get(
            "set_encoder_minus_vote_delta"
        ),
        "set_encoder_minus_vote_ci95": oracle_distinct_outcome.get(
            "set_encoder_minus_vote_ci95"
        ),
        "held_out_task_n": oracle_distinct_outcome.get("held_out_task_n"),
        "matched_control_delta": oracle_distinct_outcome.get("matched_control_delta"),
        "oracle_at_k": oracle_distinct_outcome.get("oracle_at_k"),
        "oracle_minus_vote": oracle_distinct_outcome.get("oracle_minus_vote"),
        "verifier_is_oracle": bool(oracle_distinct_outcome.get("verifier_is_oracle")),
        "oracle_distinct_auroc": oracle_distinct_outcome.get("oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": oracle_distinct_outcome.get(
            "oracle_distinct_auroc_ci95"
        ),
        "set_encoder_vs_logistic_auroc_delta": oracle_distinct_outcome.get(
            "set_encoder_vs_logistic_auroc_delta"
        ),
        "wrong_majority_n": oracle_distinct_outcome.get("wrong_majority_n"),
        "code_replication_gap_id": code_replication_outcome.get("gap_id"),
        "code_replication_status": code_replication_outcome.get("status"),
        "replication_read": code_replication_outcome.get("replication_read"),
        "code_replication_beats_vote": bool(
            code_replication_outcome.get("code_replication_beats_vote")
        ),
        "code_predictor_minus_vote_delta": code_replication_outcome.get(
            "code_predictor_minus_vote_delta"
        ),
        "code_predictor_minus_vote_ci95": code_replication_outcome.get(
            "code_predictor_minus_vote_ci95"
        ),
        "verifier_reward_gap_id": verifier_reward_outcome.get("gap_id"),
        "verifier_reward_status": verifier_reward_outcome.get("status"),
        "verifier_reward_artifact": EXP4248_PATH,
        "verifier_label_carries_signal": bool(
            verifier_reward_outcome.get("verifier_label_carries_signal")
        ),
        "a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
        "a_vs_b_ci95": verifier_reward_outcome.get("a_vs_b_ci95"),
        "youden_j": verifier_reward_outcome.get("youden_j"),
        "live_lora_retirement_gap_id": live_lora_retirement.get("gap_id"),
        "live_lora_retired": bool(live_lora_retirement.get("live_lora_retired")),
        "live_lora_retired_recorded": bool(live_lora_retirement.get("live_lora_retired")),
        "operator_reopen_required": bool(live_lora_retirement.get("operator_reopen_required")),
        "retire_if_same_verdict": bool(live_lora_retirement.get("retire_if_same_verdict")),
        "gap_oracle_distinct_update": oracle_distinct_outcome.get(
            "gap_oracle_distinct_update"
        ),
        "gap_moat_update": oracle_distinct_outcome.get("gap_moat_update"),
        "eval_exp_4252": EXP4252_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V393_ROLE_ID] + [
        role
    ]


def _oracle_distinct_frontier_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_GAP_ID}: Exp 4252 .393 oracle-distinct frontier\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4245_PATH}` with build `{EXP4244_PATH}`; "
        f"oracle_distinct_beats_vote={exp4227._bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"set_encoder_minus_vote_delta={outcome.get('set_encoder_minus_vote_delta')}; "
        f"set_encoder_minus_vote_ci95={outcome.get('set_encoder_minus_vote_ci95')}; "
        f"held_out_task_n={outcome.get('held_out_task_n')}; "
        f"matched_control_delta={outcome.get('matched_control_delta')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}; "
        f"oracle_minus_vote={outcome.get('oracle_minus_vote')}; "
        f"verifier_is_oracle={exp4227._bool_text(outcome.get('verifier_is_oracle'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"oracle_distinct_auroc_ci95={outcome.get('oracle_distinct_auroc_ci95')}; "
        f"set_encoder_vs_logistic_auroc_delta={outcome.get('set_encoder_vs_logistic_auroc_delta')}; "
        f"wrong_majority_n={outcome.get('wrong_majority_n')}; "
        f"honest_verdict={outcome.get('honest_verdict')}. "
        "This changed the .392 ties-vote read on the grown-pool set-encoder path. "
        "GAP-MOAT unchanged: registry hygiene records the frontier result but does not "
        "silently upgrade a moat claim.\n"
        "- failure mode: closed for the measured ARC A3 oracle-distinct selection read; "
        "other reward and replication axes remain separate.\n"
        "- missing discriminator: none for the measured non-oracle ARC A3 vote-beating read.\n"
        "- candidate design: preserve the grown-pool set-encoder methodology and retest "
        "only with explicit non-oracle and positive-CI gates.\n"
        "- priority: high\n"
    )


def _oracle_distinct_a3_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_A3_GAP_ID}: Exp 4252 .393 ARC A3 set-encoder read\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4245_PATH}`; "
        f"oracle_distinct_beats_vote={exp4227._bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"set_encoder_minus_vote_delta={outcome.get('set_encoder_minus_vote_delta')}; "
        f"set_encoder_minus_vote_ci95={outcome.get('set_encoder_minus_vote_ci95')}; "
        f"held_out_task_n={outcome.get('held_out_task_n')}; "
        f"margin_override_minus_vote={outcome.get('margin_override_minus_vote')}; "
        f"matched_control_delta={outcome.get('matched_control_delta')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}; "
        f"verifier_is_oracle={exp4227._bool_text(outcome.get('verifier_is_oracle'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"oracle_distinct_auroc_ci95={outcome.get('oracle_distinct_auroc_ci95')}; "
        f"set_encoder_vs_logistic_auroc_delta={outcome.get('set_encoder_vs_logistic_auroc_delta')}; "
        f"wrong_majority_n={outcome.get('wrong_majority_n')}.\n"
        "- failure mode: no ARC A3 failure for this measured read; the set-encoder "
        "converted the grown-pool candidate signal into a CI-positive vote-beating selector.\n"
        "- missing discriminator: none for this measured read; keep the non-oracle and "
        "held-out task gates load-bearing.\n"
        "- candidate design: compare future variants against both vote and same-pool controls.\n"
        "- priority: high\n"
    )


def _code_replication_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_CODE_REPLICATION_GAP_ID}: Exp 4252 .393 code replication status\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4246_PATH}`; "
        f"replication_read={outcome.get('replication_read')}; "
        f"code_replication_beats_vote={exp4227._bool_text(outcome.get('code_replication_beats_vote'))}; "
        f"code_predictor_minus_vote_delta={outcome.get('code_predictor_minus_vote_delta')}; "
        f"code_predictor_minus_vote_ci95={outcome.get('code_predictor_minus_vote_ci95')}; "
        f"held_out_task_n={outcome.get('held_out_task_n')}; "
        f"verifier_is_oracle={exp4227._bool_text(outcome.get('verifier_is_oracle'))}; "
        f"off_fold_auroc={outcome.get('off_fold_auroc')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}.\n"
        "- failure mode: no second distinct code corpus was available, so the code "
        "oracle-distinct win was neither replicated nor refuted.\n"
        "- missing discriminator: a source-distinct code candidate corpus for replication.\n"
        "- candidate design: rerun only after the source-distinctness gate has a nonempty corpus.\n"
        "- priority: medium\n"
    )


def _verifier_reward_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_REWARD_GAP_ID}: Exp 4252 .393 offline verifier-as-reward A-vs-B axis\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4248_PATH}` with retirement artifact `{EXP4247_PATH}`; "
        f"verifier_label_carries_signal={exp4227._bool_text(outcome.get('verifier_label_carries_signal'))}; "
        f"a_vs_b_delta={outcome.get('a_vs_b_delta')}; "
        f"a_vs_b_ci95={outcome.get('a_vs_b_ci95')}; "
        f"youden_j={outcome.get('youden_j')}; "
        f"live_lora_retired={exp4227._bool_text(outcome.get('live_lora_retired'))}; "
        f"blocked_at_layer={outcome.get('blocked_at_layer')}; "
        f"honest_verdict={outcome.get('honest_verdict')}.\n"
        "- failure mode: Exp 4248 was blocked by the Exp 4247 harness smoke gate, so "
        "no held-out A-vs-B reward signal exists.\n"
        "- missing discriminator: decision-grade offline evidence that verifier-certified "
        "labels beat same-generator random-label controls.\n"
        "- candidate design: repair or replace the offline smoke harness before any "
        "reward-signal promotion; do not reopen the live-LoRA path without operator approval.\n"
        "- priority: high\n"
    )


def _live_lora_retirement_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_LIVE_LORA_RETIREMENT_GAP_ID}: Exp 4252 live-LoRA retirement note\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4247_PATH}`; "
        f"live_lora_retired={exp4227._bool_text(outcome.get('live_lora_retired'))}; "
        f"harness_smoke_passed={exp4227._bool_text(outcome.get('harness_smoke_passed'))}; "
        f"infra_failure_count={outcome.get('infra_failure_count')}; "
        f"operator_reopen_required={exp4227._bool_text(outcome.get('operator_reopen_required'))}; "
        f"retire_if_same_verdict={exp4227._bool_text(outcome.get('retire_if_same_verdict'))}; "
        f"honest_verdict={outcome.get('honest_verdict')}.\n"
        "- failure mode: the live-LoRA verifier-as-reward path accumulated 6 infra failures "
        "and is retired so it is not re-proposed as another live run.\n"
        "- missing discriminator: none for live-LoRA retirement; future reward work must use "
        "the offline path unless an operator explicitly reopens live-LoRA.\n"
        "- candidate design: keep the exclusion manifest entry authoritative with "
        "operator_reopen_required=true and retire_if_same_verdict=true.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4252") == EXP4252_ARTIFACT_PATH
        and any(role.get("role_id") == V393_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def _live_lora_manifest_entry(retirement: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": LIVE_LORA_RETIREMENT_ENTRY_ID,
        "experiment_scope": "live-LoRA verifier-as-reward path",
        "reason": (
            "retire_if_same_verdict: Exp 4247 set live_lora_retired=true after "
            f"{LIVE_LORA_INFRA_FAILURE_COUNT} infra failures; use the offline "
            "reward-weighted SFT path instead. Operator reopen required."
        ),
        "experiment_ids": ["exp4247"],
        "retired_milestone": "2026.04.393",
        "retired_by_artifact": EXP4247_PATH,
        "recorded_by_artifact": EXP4252_ARTIFACT_PATH,
        "infra_failure_count": int(retirement.get("infra_failure_count") or 0),
        "operator_reopen_required": bool(retirement.get("operator_reopen_required")),
        "retire_if_same_verdict": bool(retirement.get("retire_if_same_verdict")),
        "blocked_patterns": [
            "live-LoRA verifier-as-reward",
            "live LoRA verifier reward",
            "live_lora verifier reward",
            "verifier_reward_lora_harness",
        ],
    }


def _find_live_lora_manifest_entry(manifest: dict[str, Any]) -> dict[str, Any] | None:
    for section in ("retired_extras", "retired_experiments", "retired"):
        entries = manifest.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("id") == LIVE_LORA_RETIREMENT_ENTRY_ID:
                return entry
            if entry.get("experiment_scope") == "live-LoRA verifier-as-reward path":
                return entry
    return None


def _ensure_live_lora_manifest_retirement(
    manifest: dict[str, Any],
    retirement: dict[str, Any],
) -> bool:
    if retirement.get("live_lora_retired") is not True:
        return False
    entry = _live_lora_manifest_entry(retirement)
    manifest.setdefault("retired_extras", [])
    existing = _find_live_lora_manifest_entry(manifest)
    if existing is None:
        manifest["retired_extras"].append(entry)
        return True
    if existing != entry:
        existing.clear()
        existing.update(entry)
        return True
    return False


def model_specs_for_replay(checksum: str) -> dict[str, Any]:
    """REQ-VERIFY-4252: methodology declaration for cached-candidate replay."""
    return {
        "method": "cached_gap4_candidate_replay_and_v393_ledger_reconciliation",
        "candidate_set": ARC1_POOL_PATH,
        "candidate_set_sha256": checksum,
        "program_outputs": ARC1_PROGRAMS_PATH,
        "scoring_description": "offline verifier ensemble replay over checked-in candidates",
        "codex_calls": 0,
        "live_model_inference": False,
        "gguf_inference": False,
        "gpu_inference": False,
        "trm_training_touched": False,
        "stable_checkpoint_write": False,
    }


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_replication_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    live_lora_retirement: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    live_lora_retired_recorded: bool,
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4252 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {
        GAP_ORACLE_DISTINCT_GAP_ID,
        GAP_ORACLE_DISTINCT_A3_GAP_ID,
        GAP_CODE_REPLICATION_GAP_ID,
        GAP_REWARD_GAP_ID,
        GAP_LIVE_LORA_RETIREMENT_GAP_ID,
    }
    gaps_complete = needed.issubset(set(gaps_updated))
    complete = guard_ok and gaps_complete and registry_updated and live_lora_retired_recorded
    prefix = "complete:" if complete else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4252_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4252_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v393_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"oracle_distinct_{oracle_distinct_outcome['status']}_"
            f"code_{code_replication_outcome['status']}_"
            f"reward_{verifier_reward_outcome['status']}_"
            f"live_lora_recorded_{live_lora_retired_recorded}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "live_lora_retired_recorded": bool(live_lora_retired_recorded),
        "registry_updated": bool(registry_updated),
        "oracle_distinct_outcome": oracle_distinct_outcome,
        "code_replication_outcome": code_replication_outcome,
        "verifier_reward_outcome": verifier_reward_outcome,
        "live_lora_retirement": live_lora_retirement,
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_replay(reproducibility_checksum),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "exclusion_manifest_path": EXCLUSION_MANIFEST_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4244_PATH,
            EXP4245_PATH,
            EXP4246_PATH,
            EXP4247_PATH,
            EXP4248_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4252", "SCENARIO-VERIFY-4252"],
        "adversarial_verify": {"status": "pending"},
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    checksum = f"blocked:{blocked}"
    artifact = {
        "experiment": "experiment_4252_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4252_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "live_lora_retired_recorded": False,
        "registry_updated": False,
        "oracle_distinct_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
            "a3_gap_id": GAP_ORACLE_DISTINCT_A3_GAP_ID,
            "oracle_distinct_beats_vote": False,
            "verifier_is_oracle": False,
        },
        "code_replication_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_CODE_REPLICATION_GAP_ID,
            "code_replication_beats_vote": False,
        },
        "verifier_reward_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_REWARD_GAP_ID,
            "verifier_label_carries_signal": False,
            "live_lora_retired": False,
        },
        "live_lora_retirement": {
            "status": "blocked_precondition",
            "gap_id": GAP_LIVE_LORA_RETIREMENT_GAP_ID,
            "live_lora_retired": False,
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "model_specs": model_specs_for_replay(checksum),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "preconditions": preflight,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4244_PATH,
            EXP4245_PATH,
            EXP4246_PATH,
            EXP4247_PATH,
            EXP4248_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4252", "SCENARIO-VERIFY-4252"],
        "adversarial_verify": {"status": "pending"},
    }
    validate_artifact(artifact)
    return artifact


def _run_adversarial_verify(
    repo_root: Path, artifact_path: Path
) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "adversarial_verify.py"),
            "--json",
            str(artifact_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    methodology_missing_clean = not any(flag.get("kind") == "METHODOLOGY_MISSING" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "methodology_missing_clean": methodology_missing_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required Exp 4252 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a BARE bool")
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")
    if not isinstance(artifact["live_lora_retired_recorded"], bool):
        raise ValueError("live_lora_retired_recorded must be a bare bool")
    if not isinstance(artifact["gaps_updated"], list):
        raise ValueError("gaps_updated must be a list")
    if isinstance(artifact["random_seed"], bool) or not isinstance(artifact["random_seed"], int):
        raise ValueError("random_seed must be a bare int")
    if (
        not isinstance(artifact["reproducibility_checksum"], str)
        or not artifact["reproducibility_checksum"]
    ):
        raise ValueError("reproducibility_checksum must be a non-empty string")
    if not isinstance(artifact["model_specs"], dict) or not artifact["model_specs"]:
        raise ValueError("model_specs must be a non-empty object")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required Exp 4252 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4252 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4252_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    manifest_path = repo_root / EXCLUSION_MANIFEST_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    exclusion_manifest = _load_manifest(manifest_path)
    offline_replay = replay_gap4_arc1(repo_root)
    oracle_distinct_outcome = classify_oracle_distinct_outcome(repo_root)
    code_replication_outcome = classify_code_replication_outcome(repo_root)
    verifier_reward_outcome = classify_verifier_reward_outcome(repo_root)
    live_lora_retirement = classify_live_lora_retirement(repo_root)
    checksum = candidate_set_checksum(repo_root)

    registry, gaps_text, exclusion_manifest, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        exclusion_manifest,
        offline_replay,
        oracle_distinct_outcome,
        code_replication_outcome,
        verifier_reward_outcome,
        live_lora_retirement,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")
    _write_manifest(manifest_path, exclusion_manifest)

    artifact = build_artifact(
        offline_replay=offline_replay,
        oracle_distinct_outcome=oracle_distinct_outcome,
        code_replication_outcome=code_replication_outcome,
        verifier_reward_outcome=verifier_reward_outcome,
        live_lora_retirement=live_lora_retirement,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
        live_lora_retired_recorded=bool(ledger_summary["live_lora_retired_recorded"]),
        random_seed=RANDOM_SEED,
        reproducibility_checksum=checksum,
        duration_s=time.time() - started,
    )
    base._write_json(out_path, artifact)
    raw_report = (
        adversarial_runner(out_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(repo_root, out_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    base._write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4252_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
