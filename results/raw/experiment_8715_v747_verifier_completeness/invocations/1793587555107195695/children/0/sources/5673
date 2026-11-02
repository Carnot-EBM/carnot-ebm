"""Exp 4239 registry/gaps hygiene for .392 oracle-distinct outcomes.

Spec refs: REQ-VERIFY-4239, SCENARIO-VERIFY-4239.

This runner is an offline ledger reconciler. It replays the canonical GAP-4
ARC-1 candidate set from cached artifacts, records the strengthened
oracle-distinct ARC A2 read, records the code disambiguation read, and records
the verifier-as-reward status without touching TRM training.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import time
from typing import Any, Callable

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base
from carnot.reporting import verifier_registry_gaps_hygiene_4227 as exp4227


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 4239
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4239_ARTIFACT_PATH = "results/experiment_4239_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = exp4227.ARC1_POOL_PATH
ARC1_PROGRAMS_PATH = exp4227.ARC1_PROGRAMS_PATH
EXP4231_PATH = "results/experiment_4231_oracle_distinct_arc_aggregator_build.json"
EXP4232_PATH = "results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json"
EXP4233_PATH = "results/experiment_4233_oracle_distinct_code_beats_vote.json"
EXP4235_PATH = "results/experiment_4235_verifier_as_reward_3arm_window_boxed.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
GAP_ORACLE_DISTINCT_GAP_ID = "GAP-ORACLE-DISTINCT"
GAP_ORACLE_DISTINCT_A2_GAP_ID = "GAP-ORACLE-DISTINCT-A2-4232"
GAP_CODE_DISAMBIGUATION_GAP_ID = "GAP-CODE-DISAMBIGUATION-4233"
GAP_REWARD_GAP_ID = "GAP-REWARD"
V392_ROLE_ID = "oracle_distinct_code_reward_hygiene_4239"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "regression_guard_passed",
    "gaps_updated",
    "registry_updated",
    "oracle_distinct_outcome",
    "code_disambiguation_outcome",
    "verifier_reward_outcome",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "inference_substrate",
    "adversarial_verify",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records the registry/gaps reconciled to the .392 truth.",
    "regression_guard_passed": (
        "BARE bool: the canonical GAP-4 numbers still reproduce bit-exact; catches a silent "
        "verifier regression."
    ),
    "gaps_updated": (
        "Lists the verifier_gaps entries touched (the GAP-ORACLE-DISTINCT frontier entry + "
        "the strengthened A2 + code disambiguation + verifier-as-reward notes)."
    ),
    "random_seed": (
        "Determinism precondition + the methodology field that prevents a METHODOLOGY_MISSING flag."
    ),
    "reproducibility_checksum": "Hash of the cached GAP-4 candidate set; catches silent candidate drift.",
}

replay_gap4_arc1 = exp4227.replay_gap4_arc1
candidate_set_checksum = exp4227.candidate_set_checksum


def check_preconditions(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4239: verify cached replay fixtures, upstream artifacts, and ledgers."""
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
        exp4227._check_json_resource(
            repo_root,
            "exp4231_oracle_distinct_aggregator_build",
            EXP4231_PATH,
        ),
        exp4227._check_json_resource(repo_root, "exp4232_oracle_distinct_a2", EXP4232_PATH),
        exp4227._check_json_resource(repo_root, "exp4233_code_disambiguation", EXP4233_PATH),
        exp4227._check_json_resource(repo_root, "exp4235_verifier_reward_a_vs_b", EXP4235_PATH),
    ]
    blocked = next((check["resource"] for check in checks if not check["available"]), None)
    return {"ok": blocked is None, "blocked_resource": blocked, "checks": checks}


def classify_oracle_distinct_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4239: summarize Exp 4232 A2 with the paired Exp 4231 build."""
    a2 = base._load_json(repo_root / EXP4232_PATH)
    build = base._load_json(repo_root / EXP4231_PATH)
    outcome = {
        "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
        "a2_gap_id": GAP_ORACLE_DISTINCT_A2_GAP_ID,
        "status": "open_a2_ties_vote_with_headroom_at_power",
        "artifact_path": EXP4232_PATH,
        "build_artifact_path": EXP4231_PATH,
        "source_artifacts": [EXP4232_PATH, EXP4231_PATH],
        "honest_verdict": str(a2.get("honest_verdict", "")),
        "build_honest_verdict": str(build.get("honest_verdict", "")),
        "headline_outcome": str(a2.get("headline_outcome", "")),
        "oracle_distinct_beats_vote": a2.get("oracle_distinct_beats_vote") is True,
        "aggregator_minus_vote_delta": exp4227._first_numeric(
            a2,
            "aggregator_minus_vote_delta",
        ),
        "aggregator_minus_vote_ci95": exp4227._first_ci95(
            a2,
            "aggregator_minus_vote_ci95",
        ),
        "margin_override_minus_vote": exp4227._first_numeric(
            a2,
            "margin_override_minus_vote",
        ),
        "matched_control_delta": exp4227._first_numeric(a2, "matched_control_delta"),
        "oracle_at_k": exp4227._first_numeric(a2, "oracle_at_k"),
        "oracle_minus_vote": exp4227._first_numeric(a2, "oracle_minus_vote"),
        "held_out_task_n": int(a2.get("held_out_task_n", 0)),
        "verifier_is_oracle": a2.get("verifier_is_oracle") is True,
        "headroom_exists": a2.get("headroom_exists") is True,
        "pass_rates": dict(a2.get("pass_rates", {})),
        "aggregator_trained": build.get("aggregator_trained") is True,
        "oracle_distinct_auroc": exp4227._first_numeric(build, "oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": exp4227._first_ci95(
            build,
            "oracle_distinct_auroc_ci95",
        ),
        "wrong_majority_n": build.get("wrong_majority_n"),
        "learned_verifier_path": str(build.get("learned_verifier_path", "")),
        "build_flagged_adversarial": build.get("flagged_adversarial") is True,
        "build_corrigendum_pending": list(build.get("corrigendum_pending", [])),
        "gap_moat_update": "unchanged_v392_ties_vote_with_headroom",
        "missing_discriminator": ("learned_non_oracle_arc_selector_with_positive_vote_beating_ci"),
    }
    outcome["verifier_is_oracle"] = bool(outcome["verifier_is_oracle"])
    return outcome


def classify_code_disambiguation_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4239: summarize Exp 4233 code disambiguation."""
    artifact = base._load_json(repo_root / EXP4233_PATH)
    return {
        "gap_id": GAP_CODE_DISAMBIGUATION_GAP_ID,
        "status": "filled_code_oracle_distinct_beats_vote",
        "artifact_path": EXP4233_PATH,
        "source_artifacts": [EXP4233_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "disambiguation_read": str(artifact.get("disambiguation_read", "")),
        "code_oracle_distinct_beats_vote": artifact.get("code_oracle_distinct_beats_vote") is True,
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
        "missing_discriminator": ("arc_pool_scale_or_features_matching_code_domain_power"),
    }


def classify_verifier_reward_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """REQ-VERIFY-4239: summarize Exp 4235 without inventing an A-vs-B read."""
    artifact = base._load_json(repo_root / EXP4235_PATH)
    return {
        "gap_id": GAP_REWARD_GAP_ID,
        "status": "open_live_lora_blocked_pre_gate",
        "artifact_path": EXP4235_PATH,
        "source_artifacts": [EXP4235_PATH],
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "verifier_label_carries_signal": artifact.get("verifier_label_carries_signal") is True,
        "a_vs_b_delta": exp4227._first_numeric(
            artifact,
            "a_vs_b_delta",
            "arm_a_vs_b_delta",
        ),
        "a_vs_b_ci95": exp4227._first_ci95(artifact, "a_vs_b_ci95", "arm_a_vs_b_ci95"),
        "youden_j": exp4227._numeric_or_none(artifact.get("youden_j")),
        "live_lora_retired": artifact.get("live_lora_retired") is True,
        "gate_check_summary": str(artifact.get("gate_check_summary", "")),
        "blocked_at_layer": str(artifact.get("blocked_at_layer", "")),
        "gates_evaluated": list(artifact.get("gates_evaluated", [])),
        "missing_discriminator": (
            "decision_grade_a_vs_b_training_signal_or_declared_live_lora_retirement"
        ),
    }


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_disambiguation_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Return registry and gap text with the .392 outcomes represented."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(
        updated_registry,
        offline_replay,
        oracle_distinct_outcome,
        code_disambiguation_outcome,
        verifier_reward_outcome,
    )
    _ensure_v392_role(
        updated_registry,
        oracle_distinct_outcome,
        code_disambiguation_outcome,
        verifier_reward_outcome,
    )

    updated_gaps = _append_marked_block(
        gaps_text,
        "exp4239-oracle-distinct",
        _oracle_distinct_frontier_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4239-oracle-distinct-a2",
        _oracle_distinct_a2_gap_block(oracle_distinct_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4239-code-disambiguation",
        _code_disambiguation_gap_block(code_disambiguation_outcome),
    )
    updated_gaps = _append_marked_block(
        updated_gaps,
        "exp4239-gap-reward",
        _verifier_reward_gap_block(verifier_reward_outcome),
    )
    gap_ids = [
        GAP_ORACLE_DISTINCT_GAP_ID,
        GAP_ORACLE_DISTINCT_A2_GAP_ID,
        GAP_CODE_DISAMBIGUATION_GAP_ID,
        GAP_REWARD_GAP_ID,
    ]
    touched = [gap_id for gap_id in gap_ids if gap_id in updated_gaps]
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": touched,
            "oracle_distinct_recorded": GAP_ORACLE_DISTINCT_GAP_ID in touched,
            "strengthened_a2_recorded": GAP_ORACLE_DISTINCT_A2_GAP_ID in touched,
            "code_disambiguation_recorded": GAP_CODE_DISAMBIGUATION_GAP_ID in touched,
            "verifier_reward_recorded": GAP_REWARD_GAP_ID in touched,
        },
    )


def _append_marked_block(gaps_text: str, marker: str, block: str) -> str:
    return base._replace_marked_block(gaps_text, marker, block)


def _ensure_gap4_eval(
    registry: dict[str, Any],
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_disambiguation_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    arc1 = offline_replay.get("arc1_rule_exec", {})
    entry.setdefault("eval", {}).update(
        {
            "eval_exp_4239": EXP4239_ARTIFACT_PATH,
            "exp4239_regression_guard_passed": bool(offline_replay.get("regression_guard_passed")),
            "exp4239_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4239_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4239_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4239_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
            "exp4239_oracle_distinct_beats_vote": bool(
                oracle_distinct_outcome.get("oracle_distinct_beats_vote")
            ),
            "exp4239_aggregator_minus_vote_delta": oracle_distinct_outcome.get(
                "aggregator_minus_vote_delta"
            ),
            "exp4239_aggregator_minus_vote_ci95": oracle_distinct_outcome.get(
                "aggregator_minus_vote_ci95"
            ),
            "exp4239_held_out_task_n": oracle_distinct_outcome.get("held_out_task_n"),
            "exp4239_verifier_is_oracle": bool(oracle_distinct_outcome.get("verifier_is_oracle")),
            "exp4239_oracle_distinct_auroc": oracle_distinct_outcome.get("oracle_distinct_auroc"),
            "exp4239_wrong_majority_n": oracle_distinct_outcome.get("wrong_majority_n"),
            "exp4239_code_oracle_distinct_beats_vote": bool(
                code_disambiguation_outcome.get("code_oracle_distinct_beats_vote")
            ),
            "exp4239_code_predictor_minus_vote_delta": code_disambiguation_outcome.get(
                "code_predictor_minus_vote_delta"
            ),
            "exp4239_verifier_label_carries_signal": bool(
                verifier_reward_outcome.get("verifier_label_carries_signal")
            ),
            "exp4239_live_lora_retired": bool(verifier_reward_outcome.get("live_lora_retired")),
        }
    )


def _ensure_v392_role(
    registry: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_disambiguation_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        return
    role = {
        "role_id": V392_ROLE_ID,
        "experiment": EXP4239_ARTIFACT_PATH,
        "role": "registry_gap_ledger_hygiene_v392",
        "status": "oracle_distinct_a2_code_reward_recorded",
        "oracle_distinct_gap_id": oracle_distinct_outcome.get("gap_id"),
        "oracle_distinct_a2_gap_id": oracle_distinct_outcome.get("a2_gap_id"),
        "oracle_distinct_status": oracle_distinct_outcome.get("status"),
        "oracle_distinct_artifact": EXP4232_PATH,
        "oracle_distinct_build_artifact": EXP4231_PATH,
        "oracle_distinct_beats_vote": bool(
            oracle_distinct_outcome.get("oracle_distinct_beats_vote")
        ),
        "aggregator_minus_vote_delta": oracle_distinct_outcome.get("aggregator_minus_vote_delta"),
        "aggregator_minus_vote_ci95": oracle_distinct_outcome.get("aggregator_minus_vote_ci95"),
        "held_out_task_n": oracle_distinct_outcome.get("held_out_task_n"),
        "matched_control_delta": oracle_distinct_outcome.get("matched_control_delta"),
        "oracle_at_k": oracle_distinct_outcome.get("oracle_at_k"),
        "verifier_is_oracle": bool(oracle_distinct_outcome.get("verifier_is_oracle")),
        "oracle_distinct_auroc": oracle_distinct_outcome.get("oracle_distinct_auroc"),
        "oracle_distinct_auroc_ci95": oracle_distinct_outcome.get("oracle_distinct_auroc_ci95"),
        "wrong_majority_n": oracle_distinct_outcome.get("wrong_majority_n"),
        "build_flagged_adversarial": bool(oracle_distinct_outcome.get("build_flagged_adversarial")),
        "code_disambiguation_gap_id": code_disambiguation_outcome.get("gap_id"),
        "code_disambiguation_status": code_disambiguation_outcome.get("status"),
        "code_disambiguation_read": code_disambiguation_outcome.get("disambiguation_read"),
        "code_oracle_distinct_beats_vote": bool(
            code_disambiguation_outcome.get("code_oracle_distinct_beats_vote")
        ),
        "code_predictor_minus_vote_delta": code_disambiguation_outcome.get(
            "code_predictor_minus_vote_delta"
        ),
        "code_predictor_minus_vote_ci95": code_disambiguation_outcome.get(
            "code_predictor_minus_vote_ci95"
        ),
        "verifier_reward_gap_id": verifier_reward_outcome.get("gap_id"),
        "verifier_reward_status": verifier_reward_outcome.get("status"),
        "verifier_reward_artifact": EXP4235_PATH,
        "verifier_label_carries_signal": bool(
            verifier_reward_outcome.get("verifier_label_carries_signal")
        ),
        "a_vs_b_delta": verifier_reward_outcome.get("a_vs_b_delta"),
        "a_vs_b_ci95": verifier_reward_outcome.get("a_vs_b_ci95"),
        "youden_j": verifier_reward_outcome.get("youden_j"),
        "live_lora_retired": bool(verifier_reward_outcome.get("live_lora_retired")),
        "gap_moat_update": oracle_distinct_outcome.get("gap_moat_update"),
        "eval_exp_4239": EXP4239_ARTIFACT_PATH,
    }
    old_roles = list(entry.get("registry_roles", []))
    entry["registry_roles"] = [old for old in old_roles if old.get("role_id") != V392_ROLE_ID] + [
        role
    ]


def _oracle_distinct_frontier_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_GAP_ID}: Exp 4239 .392 oracle-distinct frontier\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4232_PATH}` with build `{EXP4231_PATH}`; "
        f"oracle_distinct_beats_vote={exp4227._bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"aggregator_minus_vote_delta={outcome.get('aggregator_minus_vote_delta')}; "
        f"aggregator_minus_vote_ci95={outcome.get('aggregator_minus_vote_ci95')}; "
        f"held_out_task_n={outcome.get('held_out_task_n')}; "
        f"matched_control_delta={outcome.get('matched_control_delta')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}; "
        f"verifier_is_oracle={exp4227._bool_text(outcome.get('verifier_is_oracle'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"oracle_distinct_auroc_ci95={outcome.get('oracle_distinct_auroc_ci95')}; "
        f"wrong_majority_n={outcome.get('wrong_majority_n')}; "
        f"build_flagged_adversarial={exp4227._bool_text(outcome.get('build_flagged_adversarial'))}; "
        f"honest_verdict={outcome.get('honest_verdict')}. GAP-MOAT unchanged: "
        ".392 stronger build + power did not change the .391 ties-vote read.\n"
        "- failure mode: the strengthened non-oracle ARC aggregator still tied vote "
        "despite headroom and a larger held-out task count.\n"
        "- missing discriminator: a learned non-oracle ARC selector whose vote-beating "
        "delta has a positive CI on the headroom-present slice.\n"
        "- candidate design: grow the ARC pool or feature set using the code-domain "
        "disambiguation result before re-testing A2.\n"
        "- priority: high\n"
    )


def _oracle_distinct_a2_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_ORACLE_DISTINCT_A2_GAP_ID}: Exp 4239 .392 strengthened A2 read\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4232_PATH}`; "
        f"oracle_distinct_beats_vote={exp4227._bool_text(outcome.get('oracle_distinct_beats_vote'))}; "
        f"aggregator_minus_vote_delta={outcome.get('aggregator_minus_vote_delta')}; "
        f"aggregator_minus_vote_ci95={outcome.get('aggregator_minus_vote_ci95')}; "
        f"held_out_task_n={outcome.get('held_out_task_n')}; "
        f"margin_override_minus_vote={outcome.get('margin_override_minus_vote')}; "
        f"matched_control_delta={outcome.get('matched_control_delta')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}; "
        f"verifier_is_oracle={exp4227._bool_text(outcome.get('verifier_is_oracle'))}; "
        f"oracle_distinct_auroc={outcome.get('oracle_distinct_auroc')}; "
        f"oracle_distinct_auroc_ci95={outcome.get('oracle_distinct_auroc_ci95')}; "
        f"wrong_majority_n={outcome.get('wrong_majority_n')}.\n"
        "- failure mode: the power increase made the null cleaner rather than closing "
        "the oracle-distinct frontier.\n"
        "- missing discriminator: an ARC candidate scorer that converts off-fold "
        "discrimination into vote-beating top-1 selection.\n"
        "- candidate design: scale ARC positives or use richer candidate-set features, "
        "then require CI-exclusive lift before upgrading the frontier.\n"
        "- priority: high\n"
    )


def _code_disambiguation_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_CODE_DISAMBIGUATION_GAP_ID}: Exp 4239 .392 code disambiguation note\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4233_PATH}`; "
        f"disambiguation_read={outcome.get('disambiguation_read')}; "
        f"code_oracle_distinct_beats_vote={exp4227._bool_text(outcome.get('code_oracle_distinct_beats_vote'))}; "
        f"code_predictor_minus_vote_delta={outcome.get('code_predictor_minus_vote_delta')}; "
        f"code_predictor_minus_vote_ci95={outcome.get('code_predictor_minus_vote_ci95')}; "
        f"held_out_task_n={outcome.get('held_out_task_n')}; "
        f"verifier_is_oracle={exp4227._bool_text(outcome.get('verifier_is_oracle'))}; "
        f"off_fold_auroc={outcome.get('off_fold_auroc')}; "
        f"oracle_at_k={outcome.get('oracle_at_k')}.\n"
        "- failure mode: ARC's null does not generalize to high-power code; the ARC "
        "frontier is more likely data/positive-sparsity bound.\n"
        "- missing discriminator: ARC-scale positives or features with the power that "
        "made code vote-beating.\n"
        "- candidate design: build a larger ARC oracle-distinct candidate pool before "
        "retiring the selection thesis.\n"
        "- priority: high\n"
    )


def _verifier_reward_gap_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {GAP_REWARD_GAP_ID}: Exp 4239 .392 verifier-as-reward A-vs-B axis\n"
        f"- status: {outcome['status']}\n"
        f"- evidence: `{EXP4235_PATH}`; "
        f"verifier_label_carries_signal={exp4227._bool_text(outcome.get('verifier_label_carries_signal'))}; "
        f"a_vs_b_delta={outcome.get('a_vs_b_delta')}; "
        f"a_vs_b_ci95={outcome.get('a_vs_b_ci95')}; "
        f"youden_j={outcome.get('youden_j')}; "
        f"live_lora_retired={exp4227._bool_text(outcome.get('live_lora_retired'))}; "
        f"blocked_at_layer={outcome.get('blocked_at_layer')}; "
        f"honest_verdict={outcome.get('honest_verdict')}.\n"
        "- failure mode: Exp 4235 did not reach a held-out A-vs-B measurement because "
        "the real-training smoke pre-gate failed.\n"
        "- missing discriminator: decision-grade evidence that verifier-certified "
        "labels beat same-generator random-label controls, or an explicit live-LoRA "
        "retirement artifact.\n"
        "- candidate design: re-scope to an offline reward-weighted form or land a "
        "valid non-blocked A-vs-B eval before promotion.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    return bool(
        gap4
        and gap4.get("eval", {}).get("eval_exp_4239") == EXP4239_ARTIFACT_PATH
        and any(role.get("role_id") == V392_ROLE_ID for role in gap4.get("registry_roles", []))
    )


def model_specs_for_replay(checksum: str) -> dict[str, Any]:
    """REQ-VERIFY-4239: methodology declaration for cached-candidate replay."""
    return {
        "method": "cached_gap4_candidate_replay_and_v392_ledger_reconciliation",
        "candidate_set": ARC1_POOL_PATH,
        "candidate_set_sha256": checksum,
        "program_outputs": ARC1_PROGRAMS_PATH,
        "scoring_description": "offline verifier ensemble replay over checked-in candidates",
        "codex_calls": 0,
        "live_model_inference": False,
        "trm_training_touched": False,
        "stable_checkpoint_write": False,
    }


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    oracle_distinct_outcome: dict[str, Any],
    code_disambiguation_outcome: dict[str, Any],
    verifier_reward_outcome: dict[str, Any],
    registry_updated: bool,
    gaps_updated: list[str],
    random_seed: int,
    reproducibility_checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4239 terminal JSON payload."""
    guard_ok = bool(offline_replay.get("regression_guard_passed"))
    needed = {
        GAP_ORACLE_DISTINCT_GAP_ID,
        GAP_ORACLE_DISTINCT_A2_GAP_ID,
        GAP_CODE_DISAMBIGUATION_GAP_ID,
        GAP_REWARD_GAP_ID,
    }
    gaps_complete = needed.issubset(set(gaps_updated))
    prefix = "complete:" if guard_ok and gaps_complete and registry_updated else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4239_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4239_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}registry_gaps_reconciled_to_v392_truth_"
            f"regression_guard_passed_{guard_ok}_"
            f"oracle_distinct_{oracle_distinct_outcome['status']}_"
            f"code_{code_disambiguation_outcome['status']}_"
            f"reward_{verifier_reward_outcome['status']}"
        ),
        "regression_guard_passed": guard_ok,
        "gaps_updated": list(gaps_updated),
        "registry_updated": bool(registry_updated),
        "oracle_distinct_outcome": oracle_distinct_outcome,
        "code_disambiguation_outcome": code_disambiguation_outcome,
        "verifier_reward_outcome": verifier_reward_outcome,
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum,
        "model_specs": model_specs_for_replay(reproducibility_checksum),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "offline_replay": offline_replay,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4231_PATH,
            EXP4232_PATH,
            EXP4233_PATH,
            EXP4235_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4239", "SCENARIO-VERIFY-4239"],
        "adversarial_verify": {"status": "pending"},
    }
    validate_artifact(artifact)
    return artifact


def _blocked_artifact(preflight: dict[str, Any], duration_s: float) -> dict[str, Any]:
    blocked = str(preflight.get("blocked_resource") or "unknown_resource")
    checksum = f"blocked:{blocked}"
    artifact = {
        "experiment": "experiment_4239_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4239_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": f"blocked_{blocked}",
        "regression_guard_passed": False,
        "gaps_updated": [],
        "registry_updated": False,
        "oracle_distinct_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_ORACLE_DISTINCT_GAP_ID,
            "a2_gap_id": GAP_ORACLE_DISTINCT_A2_GAP_ID,
            "oracle_distinct_beats_vote": False,
            "verifier_is_oracle": False,
        },
        "code_disambiguation_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_CODE_DISAMBIGUATION_GAP_ID,
            "code_oracle_distinct_beats_vote": False,
        },
        "verifier_reward_outcome": {
            "status": "blocked_precondition",
            "gap_id": GAP_REWARD_GAP_ID,
            "verifier_label_carries_signal": False,
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
            EXP4231_PATH,
            EXP4232_PATH,
            EXP4233_PATH,
            EXP4235_PATH,
        ],
        "spec_refs": ["REQ-VERIFY-4239", "SCENARIO-VERIFY-4239"],
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
    """Validate required Exp 4239 fields before writing the artifact."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["regression_guard_passed"], bool):
        raise ValueError("regression_guard_passed must be a BARE bool")
    if not isinstance(artifact["registry_updated"], bool):
        raise ValueError("registry_updated must be a bare bool")
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
        raise ValueError("field_principles must match the required Exp 4239 principles")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")


def run_hygiene(
    repo_root: Path = REPO_ROOT,
    *,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4239 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    preflight = check_preconditions(repo_root)
    out_path = repo_root / EXP4239_ARTIFACT_PATH
    if not preflight["ok"]:
        artifact = _blocked_artifact(preflight, time.time() - started)
        base._write_json(out_path, artifact)
        return artifact

    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH
    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    oracle_distinct_outcome = classify_oracle_distinct_outcome(repo_root)
    code_disambiguation_outcome = classify_code_disambiguation_outcome(repo_root)
    verifier_reward_outcome = classify_verifier_reward_outcome(repo_root)
    checksum = candidate_set_checksum(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        oracle_distinct_outcome,
        code_disambiguation_outcome,
        verifier_reward_outcome,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        oracle_distinct_outcome=oracle_distinct_outcome,
        code_disambiguation_outcome=code_disambiguation_outcome,
        verifier_reward_outcome=verifier_reward_outcome,
        registry_updated=bool(ledger_summary["registry_updated"]),
        gaps_updated=list(ledger_summary["gaps_updated"]),
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
    print(f"Wrote {REPO_ROOT / EXP4239_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
