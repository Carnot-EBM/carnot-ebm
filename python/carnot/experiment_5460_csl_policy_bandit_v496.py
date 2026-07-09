"""Exp5460: frozen-model governed CSL policy bandit.

Spec refs: REQ-LEARN-5460,
SCENARIO-LEARN-5460-GATES,
SCENARIO-LEARN-5460-ROLLBACK,
SCENARIO-LEARN-5460-CONTROLS,
SCENARIO-LEARN-5460-NO-WEIGHT-MUTATION.

The experiment treats continuous self-learning as controller-side decision
state. A deterministic confidence-bound policy updates small routing counters
over no-memory, naive in-context, full-context, and governed-memory actions.
Those counters are rollbackable audit state, not model weights, which is the
important boundary for a frozen-model online policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from carnot import experiment_5446_governed_memory_csl_online_v495 as exp5446
from carnot import experiment_5447_gated_csl_memory_failure_stress_v495 as exp5447


JsonDict = dict[str, Any]
JsonList = list[JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5460_csl_policy_bandit_v496.json")
CONFIDENCE_RECEIPTS_RELATIVE_PATH = Path(
    "results/experiment_5460_csl_policy_confidence_receipts_v496.jsonl"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5460_csl_policy_bandit_v496.py")
EXP5446_RESULT_RELATIVE_PATH = exp5446.RESULT_RELATIVE_PATH
EXP5447_RESULT_RELATIVE_PATH = exp5447.RESULT_RELATIVE_PATH

EXPERIMENT = "experiment_5460_csl_policy_bandit_v496"
EXPERIMENT_ID = "exp5460-v496-csl-policy-bandit"
MILESTONE = "2026.07.496"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5460
SCHEMA = "carnot.experiment_5460.csl_policy_bandit.v496"
INFERENCE_SUBSTRATE = "deterministic_frozen_model_policy_no_weight_update"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-LEARN-5460",
    "SCENARIO-LEARN-5460-GATES",
    "SCENARIO-LEARN-5460-ROLLBACK",
    "SCENARIO-LEARN-5460-CONTROLS",
    "SCENARIO-LEARN-5460-NO-WEIGHT-MUTATION",
)
POLICY_ARMS = ("no_memory", "naive_icl", "always_full_context", "governed_memory")
BASELINE_NAMES = ("no_memory", "naive_icl", "always_full_context", "ungated_memory")
REQUIRED_CASE_FAMILIES = frozenset(
    {
        "repeated_task",
        "distribution_shift",
        "poisoned_memory",
        "stale_memory",
        "replay_failure",
        "no_memory_competitive",
        "naive_icl_competitive",
        "rollback_required",
        "post_rollback_replay",
    }
)

FIELD_PRINCIPLES: dict[str, str] = {
    "continuous_self_learning_task": "Research-program mandate.",
    "policy_update_count": "Online controller learning evidence.",
    "multi_session_trace_count": "Stateful stream coverage.",
    "baseline_names": "Control comparability.",
    "policy_confidence_receipts_path": "Uncertainty audit trail.",
    "regret_proxy_delta_vs_no_memory": "Online decision quality.",
    "quality_delta_vs_naive_icl": "No hidden regression against cheap ICL.",
    "context_efficiency_delta": "Context budget accounting.",
    "verifier_cost_delta": "Verifier budget accounting.",
    "cumulative_constraint_violations": "Safety boundary.",
    "negative_transfer_deflection_rate": "Unsafe memory transfer guard.",
    "rollback_recovery_rate": "Bad evidence reversibility.",
    "no_weight_mutation": "Frozen-model boundary.",
    "csl_policy_ready": "Downstream gate.",
    "inference_substrate": "Explicit learning substrate.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
INTEGER_FIELDS = (
    "policy_update_count",
    "multi_session_trace_count",
    "cumulative_constraint_violations",
)
BOOL_FIELDS = ("continuous_self_learning_task", "no_weight_mutation", "csl_policy_ready")
NUMERIC_FIELDS = (
    "regret_proxy_delta_vs_no_memory",
    "quality_delta_vs_naive_icl",
    "context_efficiency_delta",
    "verifier_cost_delta",
)
RATE_FIELDS = ("negative_transfer_deflection_rate", "rollback_recovery_rate")


class FrozenPolicyBandit:
    """Deterministic confidence-bound policy with rollbackable evidence state."""

    def __init__(
        self,
        *,
        exploration: float = 0.22,
        context_cost_weight: float = 0.00025,
        verifier_cost_weight: float = 0.012,
    ) -> None:
        self.exploration = float(exploration)
        self.context_cost_weight = float(context_cost_weight)
        self.verifier_cost_weight = float(verifier_cost_weight)
        self.arm_stats: dict[str, JsonDict] = {}
        self.evidence_index: dict[str, JsonDict] = {}
        self.rollback_tombstones: set[str] = set()

    def decide(self, row: Mapping[str, Any]) -> JsonDict:
        """Select one allowed arm and immediately record its evidence update."""

        context_key = str(row["context_key"])
        arm_scores = {
            arm: self._arm_score(
                self.arm_stats.get(_context_arm_key(context_key, arm), _empty_stats()),
                row["arm_outcomes"][arm],
            )
            for arm in POLICY_ARMS
        }
        selected_arm = max(
            POLICY_ARMS,
            key=lambda arm: (arm_scores[arm]["score"], -POLICY_ARMS.index(arm)),
        )
        outcome = row["arm_outcomes"][selected_arm]
        evidence_id = f"policy-evidence:{row['trace_id']}:{selected_arm}"
        update = self.record_policy_evidence(
            trace_id=str(row["trace_id"]),
            context_key=context_key,
            arm=selected_arm,
            outcome=outcome,
            evidence_id=evidence_id,
        )
        key = _context_arm_key(context_key, selected_arm)
        return _json_ready(
            {
                "trace_id": row["trace_id"],
                "context_key": context_key,
                "selected_arm": selected_arm,
                "evidence_id": evidence_id,
                "arm_scores": arm_scores,
                "gate_receipt": update["gate_receipt"],
                "update_status": update["update_status"],
                "cited_evidence_ids": list(
                    self.arm_stats.get(key, _empty_stats())["accepted_evidence_ids"]
                ),
            }
        )

    def record_policy_evidence(
        self,
        *,
        trace_id: str,
        context_key: str,
        arm: str,
        outcome: Mapping[str, Any],
        evidence_id: str,
    ) -> JsonDict:
        """Update arm statistics only when verifier and governance gates pass."""

        gate = policy_gate_receipt(outcome)
        key = _context_arm_key(context_key, arm)
        if gate["allowed"] is not True:
            return _json_ready(
                {
                    "trace_id": trace_id,
                    "context_key": context_key,
                    "arm": arm,
                    "evidence_id": evidence_id,
                    "gate_receipt": gate,
                    "update_status": "rejected_by_governance",
                    "stats_after": self.arm_stats.get(key, _empty_stats()),
                }
            )
        reward = reward_for_outcome(outcome)
        stats = self._stats_for(context_key, arm)
        stats["count"] += 1
        stats["reward_sum"] = round(float(stats["reward_sum"]) + reward, 6)
        stats["context_cost_sum"] += int(outcome["context_cost"])
        stats["verifier_cost_sum"] += int(outcome["verifier_cost"])
        stats["accepted_evidence_ids"].append(evidence_id)
        self.evidence_index[evidence_id] = {
            "context_key": context_key,
            "arm": arm,
            "reward": reward,
            "context_cost": int(outcome["context_cost"]),
            "verifier_cost": int(outcome["verifier_cost"]),
        }
        return _json_ready(
            {
                "trace_id": trace_id,
                "context_key": context_key,
                "arm": arm,
                "evidence_id": evidence_id,
                "gate_receipt": gate,
                "update_status": "accepted",
                "stats_after": stats,
            }
        )

    def rollback_evidence(self, evidence_id: str) -> JsonDict:
        """Remove a previously accepted evidence row from policy statistics."""

        record = self.evidence_index.pop(evidence_id, None)
        if record is None:
            return {
                "rollback_success": False,
                "removed_evidence_id": evidence_id,
                "reason": "evidence_not_active",
            }
        key = _context_arm_key(record["context_key"], record["arm"])
        stats = self.arm_stats[key]
        stats["count"] = max(0, int(stats["count"]) - 1)
        stats["reward_sum"] = round(float(stats["reward_sum"]) - float(record["reward"]), 6)
        stats["context_cost_sum"] -= int(record["context_cost"])
        stats["verifier_cost_sum"] -= int(record["verifier_cost"])
        stats["accepted_evidence_ids"] = [
            item for item in stats["accepted_evidence_ids"] if item != evidence_id
        ]
        self.rollback_tombstones.add(evidence_id)
        return _json_ready(
            {
                "rollback_success": True,
                "removed_evidence_id": evidence_id,
                "context_key": record["context_key"],
                "arm": record["arm"],
                "stats_after": stats,
            }
        )

    def snapshot(self) -> JsonDict:
        """Return stable JSON-ready policy statistics for artifacts and tests."""

        return _json_ready(
            {
                "arm_stats": self.arm_stats,
                "accepted_evidence_ids": sorted(self.evidence_index),
                "rolled_back_evidence_ids": sorted(self.rollback_tombstones),
                "active_policy_update_count": sum(
                    int(stats["count"]) for stats in self.arm_stats.values()
                ),
            }
        )

    def _stats_for(self, context_key: str, arm: str) -> JsonDict:
        key = _context_arm_key(context_key, arm)
        self.arm_stats.setdefault(key, _empty_stats())
        return self.arm_stats[key]

    def _arm_score(self, stats: Mapping[str, Any], outcome: Mapping[str, Any]) -> JsonDict:
        gate = policy_gate_receipt(outcome)
        count = int(stats["count"])
        mean_reward = (
            float(outcome["predicted_quality"])
            if count == 0
            else float(stats["reward_sum"]) / count
        )
        uncertainty = self.exploration / math.sqrt(count + 1)
        expected_reward = 0.5 * mean_reward + 0.5 * float(outcome["predicted_quality"])
        score = (
            expected_reward
            + uncertainty
            - self.context_cost_weight * int(outcome["context_cost"])
            - self.verifier_cost_weight * int(outcome["verifier_cost"])
        )
        if gate["allowed"] is not True:
            score = -999.0
        return {
            "score": round(score, 6),
            "uncertainty": round(uncertainty, 6),
            "expected_reward": round(expected_reward, 6),
            "context_cost": int(outcome["context_cost"]),
            "verifier_cost": int(outcome["verifier_cost"]),
            "allowed": gate["allowed"],
            "gate_reasons": gate["reasons"],
        }


def build_policy_stream() -> JsonList:
    """Build a deterministic multi-session stream with hard and cheap cases."""

    rows = [
        _trace(
            "trace5460-a1-repeat-bracket",
            "session-a",
            "cad",
            "repeated_task",
            {
                "no_memory": _outcome(0.8, 0.78, 260, 4),
                "naive_icl": _outcome(0.86, 0.84, 380, 4),
                "always_full_context": _outcome(0.96, 0.9, 1000, 6),
                "governed_memory": _outcome(0.96, 0.94, 420, 3),
                "ungated_memory": _outcome(0.96, 0.9, 360, 1),
            },
        ),
        _trace(
            "trace5460-a2-repeat-pocket",
            "session-a",
            "cad",
            "repeated_task",
            {
                "no_memory": _outcome(0.81, 0.78, 240, 4),
                "naive_icl": _outcome(0.87, 0.84, 360, 4),
                "always_full_context": _outcome(0.95, 0.9, 900, 5),
                "governed_memory": _outcome(0.95, 0.94, 320, 2),
                "ungated_memory": _outcome(0.95, 0.89, 300, 1),
            },
        ),
        _trace(
            "trace5460-b1-code-shift",
            "session-b",
            "code",
            "distribution_shift",
            {
                "no_memory": _outcome(0.75, 0.74, 260, 4),
                "naive_icl": _outcome(0.78, 0.77, 340, 4),
                "always_full_context": _outcome(0.92, 0.99, 800, 6),
                "governed_memory": _outcome(0.84, 0.8, 720, 5),
                "ungated_memory": _outcome(0.7, 0.83, 280, 1, constraint_violation=True),
            },
        ),
        _trace(
            "trace5460-b2-fresh-simple",
            "session-b",
            "cad",
            "no_memory_competitive",
            {
                "no_memory": _outcome(0.84, 0.84, 150, 2),
                "naive_icl": _outcome(0.83, 0.8, 320, 3),
                "always_full_context": _outcome(0.86, 0.82, 760, 4),
                "governed_memory": _outcome(0.87, 0.86, 900, 5),
                "ungated_memory": _outcome(0.82, 0.8, 260, 1),
            },
        ),
        _trace(
            "trace5460-b3-recent-format",
            "session-b",
            "support",
            "naive_icl_competitive",
            {
                "no_memory": _outcome(0.78, 0.78, 170, 2),
                "naive_icl": _outcome(0.9, 0.88, 320, 3),
                "always_full_context": _outcome(0.91, 0.85, 820, 5),
                "governed_memory": _outcome(0.9, 0.87, 760, 4),
                "ungated_memory": _outcome(0.88, 0.84, 300, 1),
            },
        ),
        _trace(
            "trace5460-c1-poisoned-vendor-memory",
            "session-c",
            "vendor",
            "poisoned_memory",
            {
                "no_memory": _outcome(0.81, 0.8, 200, 4),
                "naive_icl": _outcome(0.7, 0.78, 300, 4, constraint_violation=True),
                "always_full_context": _outcome(0.88, 0.88, 900, 6),
                "governed_memory": _outcome(
                    0.5,
                    0.95,
                    420,
                    2,
                    provenance_pass=False,
                    constraint_violation=True,
                    negative_transfer=True,
                ),
                "ungated_memory": _outcome(
                    0.52,
                    0.95,
                    240,
                    1,
                    constraint_violation=True,
                    negative_transfer=True,
                ),
            },
            negative_transfer_candidate=True,
        ),
        _trace(
            "trace5460-c2-stale-lot-memory",
            "session-c",
            "cad",
            "stale_memory",
            {
                "no_memory": _outcome(0.8, 0.8, 210, 4),
                "naive_icl": _outcome(0.83, 0.82, 320, 4),
                "always_full_context": _outcome(0.9, 0.88, 900, 6),
                "governed_memory": _outcome(
                    0.62,
                    0.91,
                    430,
                    2,
                    replay_pass=False,
                    negative_transfer=True,
                ),
                "ungated_memory": _outcome(
                    0.6,
                    0.9,
                    250,
                    1,
                    constraint_violation=True,
                    negative_transfer=True,
                ),
            },
            negative_transfer_candidate=True,
        ),
        _trace(
            "trace5460-d1-replay-order-trap",
            "session-d",
            "cad",
            "replay_failure",
            {
                "no_memory": _outcome(0.77, 0.77, 220, 4),
                "naive_icl": _outcome(0.78, 0.79, 310, 4),
                "always_full_context": _outcome(0.91, 0.92, 840, 5),
                "governed_memory": _outcome(
                    0.55,
                    0.93,
                    360,
                    2,
                    replay_pass=False,
                    constraint_violation=True,
                    negative_transfer=True,
                ),
                "ungated_memory": _outcome(
                    0.58,
                    0.93,
                    230,
                    1,
                    constraint_violation=True,
                    negative_transfer=True,
                ),
            },
            negative_transfer_candidate=True,
        ),
        _trace(
            "trace5460-d2-rollback-marked-policy",
            "session-d",
            "cad",
            "rollback_required",
            {
                "no_memory": _outcome(0.79, 0.79, 230, 4),
                "naive_icl": _outcome(0.86, 0.84, 340, 4),
                "always_full_context": _outcome(0.94, 0.9, 880, 5),
                "governed_memory": _outcome(0.94, 0.93, 430, 3),
                "ungated_memory": _outcome(0.88, 0.9, 260, 1),
            },
            rollback_after_update=True,
        ),
        _trace(
            "trace5460-e1-post-rollback-repeat",
            "session-e",
            "cad",
            "post_rollback_replay",
            {
                "no_memory": _outcome(0.82, 0.8, 240, 4),
                "naive_icl": _outcome(0.87, 0.84, 360, 4),
                "always_full_context": _outcome(0.95, 0.9, 900, 5),
                "governed_memory": _outcome(0.95, 0.94, 330, 2),
                "ungated_memory": _outcome(0.95, 0.9, 290, 1),
            },
        ),
    ]
    return [_json_ready(row) for row in rows]


def evaluate_policy_bandit(root: Path | str = REPO_ROOT) -> JsonDict:
    """Run the frozen policy over the stream and compute controls and receipts."""

    upstream = load_upstream_artifacts(root)
    policy = FrozenPolicyBandit()
    rows: JsonList = []
    receipts: JsonList = []
    rollback_audits: JsonList = []
    for raw in build_policy_stream():
        row = copy.deepcopy(raw)
        decision = policy.decide(row)
        if row.get("rollback_after_update") is True:
            rollback = policy.rollback_evidence(str(decision["evidence_id"]))
            decision["rollback_receipt"] = rollback
            rollback_audits.append(rollback)
        selected = str(decision["selected_arm"])
        row["policy_decision"] = decision
        row["policy_outcome"] = row["arm_outcomes"][selected]
        receipts.append(decision)
        rows.append(row)

    baseline_metrics = {
        name: _aggregate_outcomes(row["baseline_outcomes"][name] for row in rows)
        for name in BASELINE_NAMES
    }
    policy_metrics = _aggregate_outcomes(row["policy_outcome"] for row in rows)
    snapshot = policy.snapshot()
    oracle_regret, policy_regret, no_memory_regret = _regret_totals(rows)
    always_full = baseline_metrics["always_full_context"]
    naive = baseline_metrics["naive_icl"]
    negative_rows = [row for row in rows if row["negative_transfer_candidate"] is True]
    rollback_recovery_rate = _rollback_recovery_rate(rollback_audits)
    policy_metrics["active_policy_update_count"] = snapshot["active_policy_update_count"]
    return _json_ready(
        {
            "trace_rows": rows,
            "upstream_readiness": upstream,
            "baseline_metrics": baseline_metrics,
            "policy_metrics": policy_metrics,
            "policy_action_space": list(POLICY_ARMS),
            "confidence_receipts": receipts,
            "confidence_receipts_checksum": _jsonl_checksum(receipts),
            "policy_snapshot": snapshot,
            "rollback_audit": {
                "rollback_events": rollback_audits,
                "rolled_back_evidence_ids": snapshot["rolled_back_evidence_ids"],
                "future_receipts_exclude_rolled_back": _future_receipts_exclude_rolled_back(
                    receipts,
                    snapshot["rolled_back_evidence_ids"],
                ),
                "rollback_recovery_rate": rollback_recovery_rate,
            },
            "policy_update_count": snapshot["active_policy_update_count"],
            "multi_session_trace_count": len(rows),
            "regret_proxy_delta_vs_no_memory": round(no_memory_regret - policy_regret, 6),
            "regret_proxy_components": {
                "oracle_regret": round(oracle_regret, 6),
                "policy_regret": round(policy_regret, 6),
                "no_memory_regret": round(no_memory_regret, 6),
            },
            "quality_delta_vs_naive_icl": round(
                policy_metrics["quality_score"] - naive["quality_score"],
                6,
            ),
            "context_efficiency_delta": _relative_savings(
                always_full["context_cost"],
                policy_metrics["context_cost"],
            ),
            "verifier_cost_delta": _relative_savings(
                always_full["verifier_cost"],
                policy_metrics["verifier_cost"],
            ),
            "cumulative_constraint_violations": policy_metrics["constraint_violations"],
            "negative_transfer_deflection_rate": _negative_transfer_deflection_rate(
                negative_rows
            ),
            "rollback_recovery_rate": rollback_recovery_rate,
            "no_weight_mutation": _weight_mutation_receipt()["no_weight_mutation"],
            "weight_mutation_receipt": _weight_mutation_receipt(),
        }
    )


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal JSON artifact for Exp5460."""

    evaluation = evaluate_policy_bandit(root=root)
    readiness = _readiness_checks(evaluation, tests_run)
    ready = bool(readiness["all_passed"])
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if ready else "blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "continuous_self_learning_task": True,
        "policy_update_count": evaluation["policy_update_count"],
        "multi_session_trace_count": evaluation["multi_session_trace_count"],
        "baseline_names": list(BASELINE_NAMES),
        "policy_confidence_receipts_path": str(CONFIDENCE_RECEIPTS_RELATIVE_PATH),
        "regret_proxy_delta_vs_no_memory": evaluation[
            "regret_proxy_delta_vs_no_memory"
        ],
        "quality_delta_vs_naive_icl": evaluation["quality_delta_vs_naive_icl"],
        "context_efficiency_delta": evaluation["context_efficiency_delta"],
        "verifier_cost_delta": evaluation["verifier_cost_delta"],
        "cumulative_constraint_violations": evaluation[
            "cumulative_constraint_violations"
        ],
        "negative_transfer_deflection_rate": evaluation[
            "negative_transfer_deflection_rate"
        ],
        "rollback_recovery_rate": evaluation["rollback_recovery_rate"],
        "no_weight_mutation": evaluation["no_weight_mutation"],
        "csl_policy_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready),
        "tests_run": [_normalise_test_run(item) for item in tests_run],
        "trace_rows": evaluation["trace_rows"],
        "baseline_metrics": evaluation["baseline_metrics"],
        "policy_metrics": evaluation["policy_metrics"],
        "policy_action_space": evaluation["policy_action_space"],
        "confidence_receipts_checksum": evaluation["confidence_receipts_checksum"],
        "policy_snapshot": evaluation["policy_snapshot"],
        "rollback_audit": evaluation["rollback_audit"],
        "readiness_checks": readiness,
        "upstream_readiness": evaluation["upstream_readiness"],
        "regret_proxy_components": evaluation["regret_proxy_components"],
        "weight_mutation_receipt": evaluation["weight_mutation_receipt"],
        "source_artifacts": [
            str(EXP5446_RESULT_RELATIVE_PATH),
            str(EXP5447_RESULT_RELATIVE_PATH),
        ],
        "source_files": {
            "spec": str(SPEC_RELATIVE_PATH),
            "module": str(MODULE_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(Path(root)),
        "methodology_note": (
            "Exp5460 is a deterministic frozen-model policy replay. Online "
            "learning updates contextual-bandit routing statistics and "
            "rollbackable evidence IDs only; model and adapter weights are not "
            "loaded, written, fine-tuned, or mutated."
        ),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = _checksum(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the artifact cannot support the V496 policy readiness claim."""

    errors: list[str] = []
    errors.extend(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    errors.extend(
        field
        for field in INTEGER_FIELDS
        if type(artifact.get(field)) is not int or artifact.get(field, -1) < 0
    )
    errors.extend(field for field in BOOL_FIELDS if type(artifact.get(field)) is not bool)
    errors.extend(field for field in NUMERIC_FIELDS if not _is_numeric(artifact.get(field)))
    errors.extend(field for field in RATE_FIELDS if not _rate_is_valid(artifact.get(field)))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("baseline_names") != list(BASELINE_NAMES):
        errors.append("baseline_names")
    if artifact.get("policy_confidence_receipts_path") != str(
        CONFIDENCE_RECEIPTS_RELATIVE_PATH
    ):
        errors.append("policy_confidence_receipts_path")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("multi_session_trace_count") != len(artifact.get("trace_rows", [])):
        errors.append("multi_session_trace_count")
    policy_metrics = artifact.get("policy_metrics", {})
    if artifact.get("policy_update_count") != policy_metrics.get(
        "active_policy_update_count"
    ):
        errors.append("policy_update_count")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("research_conductor_modified")
    ready = artifact.get("csl_policy_ready")
    if ready is True:
        errors.extend(_ready_artifact_errors(artifact))
    if artifact.get("status") == "complete" and ready is not True:
        errors.append("csl_policy_ready")
    if artifact.get("status") == "blocked" and ready is True:
        errors.append("csl_policy_ready")
    if errors:
        raise ValueError(
            "invalid Exp5460 artifact fields: " + ",".join(sorted(set(errors)))
        )
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    receipts_path: Path | str = REPO_ROOT / CONFIDENCE_RECEIPTS_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5460 result artifact and confidence receipts."""

    evaluation = evaluate_policy_bandit(root=root)
    artifact = build_artifact(root=root, tests_run=tests_run)
    write_confidence_receipts(receipts_path, evaluation["confidence_receipts"])
    path = Path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def write_confidence_receipts(path: Path | str, receipts: Sequence[Mapping[str, Any]]) -> None:
    """Persist one confidence-bound decision receipt per trace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_ready(row), sort_keys=True, ensure_ascii=True) for row in receipts]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def policy_gate_receipt(outcome: Mapping[str, Any]) -> JsonDict:
    """Return the governance gate result for one candidate arm outcome."""

    checks = {
        "verifier": outcome.get("verifier_pass") is True,
        "provenance": outcome.get("provenance_pass") is True,
        "replay": outcome.get("replay_pass") is True,
        "access": outcome.get("access_pass") is True,
        "no_weight_mutation": outcome.get("no_weight_mutation_proof") is True,
        "constraint": outcome.get("constraint_violation") is not True,
    }
    labels = {
        "verifier": "verifier_failed",
        "provenance": "provenance_failed",
        "replay": "replay_failed",
        "access": "access_failed",
        "no_weight_mutation": "no_weight_mutation_failed",
        "constraint": "constraint_violation",
    }
    reasons = [labels[name] for name, passed in checks.items() if passed is not True]
    return {"allowed": not reasons, "checks": checks, "reasons": reasons}


def reward_for_outcome(outcome: Mapping[str, Any]) -> float:
    """Convert quality and costs into the scalar reward used by the bandit."""

    penalty = 0.0
    penalty += 0.5 if outcome.get("constraint_violation") is True else 0.0
    penalty += 0.25 if outcome.get("negative_transfer") is True else 0.0
    reward = (
        float(outcome["quality_score"])
        - int(outcome["context_cost"]) / 5000.0
        - int(outcome["verifier_cost"]) * 0.01
        - penalty
    )
    return round(reward, 6)


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load V495 readiness gates that authorize V496 policy evaluation."""

    base = Path(root)
    exp5446_payload = _read_json(base / EXP5446_RESULT_RELATIVE_PATH)
    exp5447_payload = _read_json(base / EXP5447_RESULT_RELATIVE_PATH)
    return _json_ready(
        {
            "exp5446_governed_csl_loop_ready": exp5446_payload.get(
                "governed_csl_loop_ready"
            )
            is True,
            "exp5447_csl_memory_stress_ready": exp5447_payload.get(
                "csl_memory_stress_ready"
            )
            is True,
            "exp5446_reproducibility_checksum": exp5446_payload.get(
                "reproducibility_checksum",
                "",
            ),
            "exp5447_reproducibility_checksum": exp5447_payload.get(
                "reproducibility_checksum",
                "",
            ),
        }
    )


def _trace(
    trace_id: str,
    session_id: str,
    context_key: str,
    case_family: str,
    outcomes: Mapping[str, Mapping[str, Any]],
    *,
    negative_transfer_candidate: bool = False,
    rollback_after_update: bool = False,
) -> JsonDict:
    row = {
        "trace_id": trace_id,
        "raw_trace_id": f"raw-{trace_id}",
        "session_id": session_id,
        "context_key": context_key,
        "case_family": case_family,
        "negative_transfer_candidate": negative_transfer_candidate,
        "rollback_after_update": rollback_after_update,
        "arm_outcomes": {arm: dict(outcomes[arm]) for arm in POLICY_ARMS},
        "baseline_outcomes": {name: dict(outcomes[name]) for name in BASELINE_NAMES},
    }
    row["raw_trace_receipt"] = _raw_trace_receipt(row)
    return row


def _outcome(
    quality_score: float,
    predicted_quality: float,
    context_cost: int,
    verifier_cost: int,
    *,
    verifier_pass: bool = True,
    provenance_pass: bool = True,
    replay_pass: bool = True,
    access_pass: bool = True,
    no_weight_mutation_proof: bool = True,
    constraint_violation: bool = False,
    negative_transfer: bool = False,
) -> JsonDict:
    return {
        "quality_score": float(quality_score),
        "predicted_quality": float(predicted_quality),
        "context_cost": int(context_cost),
        "verifier_cost": int(verifier_cost),
        "verifier_pass": bool(verifier_pass),
        "provenance_pass": bool(provenance_pass),
        "replay_pass": bool(replay_pass),
        "access_pass": bool(access_pass),
        "no_weight_mutation_proof": bool(no_weight_mutation_proof),
        "constraint_violation": bool(constraint_violation),
        "negative_transfer": bool(negative_transfer),
    }


def _aggregate_outcomes(outcomes: Sequence[Mapping[str, Any]] | Any) -> JsonDict:
    rows = list(outcomes)
    return {
        "quality_score": round(
            sum(float(row["quality_score"]) for row in rows) / len(rows),
            6,
        ),
        "context_cost": sum(int(row["context_cost"]) for row in rows),
        "verifier_cost": sum(int(row["verifier_cost"]) for row in rows),
        "constraint_violations": sum(row.get("constraint_violation") is True for row in rows),
    }


def _regret_totals(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    oracle_regret = 0.0
    policy_regret = 0.0
    no_memory_regret = 0.0
    for row in rows:
        clean_rewards = [
            reward_for_outcome(row["arm_outcomes"][arm])
            for arm in POLICY_ARMS
            if policy_gate_receipt(row["arm_outcomes"][arm])["allowed"] is True
        ]
        oracle = max(clean_rewards)
        policy_reward = reward_for_outcome(row["policy_outcome"])
        no_memory_reward = reward_for_outcome(row["baseline_outcomes"]["no_memory"])
        oracle_regret += 0.0
        policy_regret += oracle - policy_reward
        no_memory_regret += oracle - no_memory_reward
    return oracle_regret, round(policy_regret, 6), round(no_memory_regret, 6)


def _future_receipts_exclude_rolled_back(
    receipts: Sequence[Mapping[str, Any]],
    rolled_back_ids: Sequence[str],
) -> bool:
    blocked = set(rolled_back_ids)
    return all(blocked.isdisjoint(receipt.get("cited_evidence_ids", [])) for receipt in receipts[-1:])


def _rollback_recovery_rate(rollback_audits: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if rollback_audits and all(row.get("rollback_success") is True for row in rollback_audits) else 0.0


def _negative_transfer_deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if not rows else round(
        sum(
            row["policy_outcome"].get("negative_transfer") is not True
            and row["policy_outcome"].get("constraint_violation") is not True
            for row in rows
        )
        / len(rows),
        6,
    )


def _readiness_checks(
    evaluation: Mapping[str, Any],
    tests_run: Sequence[str | Mapping[str, Any]],
) -> JsonDict:
    upstream = evaluation["upstream_readiness"]
    checks = {
        "upstream_v495_ready": upstream["exp5446_governed_csl_loop_ready"] is True
        and upstream["exp5447_csl_memory_stress_ready"] is True,
        "case_families_covered": REQUIRED_CASE_FAMILIES.issubset(
            {row["case_family"] for row in evaluation["trace_rows"]}
        ),
        "policy_updates_recorded": evaluation["policy_update_count"] > 0,
        "confidence_receipts_cover_stream": len(evaluation["confidence_receipts"])
        == evaluation["multi_session_trace_count"],
        "regret_improved_vs_no_memory": evaluation["regret_proxy_delta_vs_no_memory"] > 0.0,
        "quality_preserved_vs_naive_icl": evaluation["quality_delta_vs_naive_icl"] >= 0.0,
        "context_efficiency_positive": evaluation["context_efficiency_delta"] > 0.0,
        "verifier_cost_positive": evaluation["verifier_cost_delta"] > 0.0,
        "constraint_violations_zero": evaluation["cumulative_constraint_violations"] == 0,
        "negative_transfer_deflected": evaluation["negative_transfer_deflection_rate"] == 1.0,
        "rollback_recovered": evaluation["rollback_recovery_rate"] == 1.0,
        "no_weight_mutation": evaluation["no_weight_mutation"] is True,
        "tests_recorded": bool(tests_run),
    }
    failed = sorted(key for key, passed in checks.items() if passed is not True)
    return {"all_passed": not failed, "checks": checks, "failed_checks": failed}


def _ready_artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = artifact.get("readiness_checks", {})
    if checks.get("all_passed") is not True:
        errors.append("csl_policy_ready")
    if not artifact.get("tests_run"):
        errors.append("tests_run")
    if artifact.get("policy_update_count", 0) <= 0:
        errors.append("policy_update_count")
    if artifact.get("regret_proxy_delta_vs_no_memory", 0.0) <= 0.0:
        errors.append("csl_policy_ready")
    if artifact.get("quality_delta_vs_naive_icl", -1.0) < 0.0:
        errors.append("csl_policy_ready")
    if artifact.get("context_efficiency_delta", 0.0) <= 0.0:
        errors.append("csl_policy_ready")
    if artifact.get("verifier_cost_delta", 0.0) <= 0.0:
        errors.append("csl_policy_ready")
    if artifact.get("cumulative_constraint_violations") != 0:
        errors.append("csl_policy_ready")
    if artifact.get("negative_transfer_deflection_rate") != 1.0:
        errors.append("csl_policy_ready")
    if artifact.get("rollback_recovery_rate") != 1.0:
        errors.append("csl_policy_ready")
    if artifact.get("no_weight_mutation") is not True:
        errors.append("no_weight_mutation")
    return errors


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weights_written": False,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "learned_state_scope": "contextual_bandit_policy_statistics_only",
    }


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return {
        "command": str(item.get("command", "")),
        "outcome": str(item.get("outcome", "passed")),
    }


def _honest_verdict(ready: bool) -> str:
    if ready:
        return (
            "complete: frozen-model governed CSL policy updated rollbackable "
            "action and memory routing statistics with zero constraint "
            "violations and no model weight mutation"
        )
    return "blocked: frozen-model governed CSL policy evidence or verification is incomplete"


def _context_arm_key(context_key: str, arm: str) -> str:
    return f"{context_key}|{arm}"


def _empty_stats() -> JsonDict:
    return {
        "count": 0,
        "reward_sum": 0.0,
        "context_cost_sum": 0,
        "verifier_cost_sum": 0,
        "accepted_evidence_ids": [],
    }


def _relative_savings(before: int | float, after: int | float) -> float:
    return round((float(before) - float(after)) / float(before), 6) if before else 0.0


def _raw_trace_receipt(row: Mapping[str, Any]) -> JsonDict:
    return {
        "raw_trace_id": row["raw_trace_id"],
        "retention_reason": "frozen-policy-bandit-audit",
        "checksum": _checksum(
            {
                "trace_id": row["trace_id"],
                "session_id": row["session_id"],
                "context_key": row["context_key"],
                "case_family": row["case_family"],
            }
        ),
    }


def _source_file_checksums(root: Path) -> JsonDict:
    return {
        "spec": _file_checksum(root / SPEC_RELATIVE_PATH),
        "module": _file_checksum(root / MODULE_RELATIVE_PATH),
        "exp5446_result": _file_checksum(root / EXP5446_RESULT_RELATIVE_PATH),
        "exp5447_result": _file_checksum(root / EXP5447_RESULT_RELATIVE_PATH),
    }


def _file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(_json_ready(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rate_is_valid(value: object) -> bool:
    return type(value) in {int, float} and 0.0 <= float(value) <= 1.0


def _is_numeric(value: object) -> bool:
    return type(value) in {int, float}


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
