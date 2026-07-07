"""Exp5330: SEA anytime certificate gate for lifecycle policy promotion.

Spec refs: REQ-LEARN-5330, SCENARIO-LEARN-5330-PROMOTE,
SCENARIO-LEARN-5330-REJECT, SCENARIO-LEARN-5330-DEFER,
SCENARIO-LEARN-5330-NOOP, SCENARIO-LEARN-5330-ROLLBACK.

This experiment is deliberately a frozen-model policy gate. The only proposed
"self-evolution" is whether Carnot should promote deterministic context
lifecycle policy choices over the Exp5328 fixture. No model is loaded, no
adapter is updated, and no foundation weights are touched. That separation is
important because SEA-style claims can sound like recursive model improvement;
here the certificate only governs auditable context-bank behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5328_context_object_lifecycle_self_learning_v486 as exp5328
from carnot import experiment_5329_memory_context_policy_rollout_v486 as exp5329


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5330_sea_anytime_certificate_gate_v486"
EXPERIMENT_ID = 5330
MILESTONE = "v486"
SCHEMA = "carnot.experiment_5330.sea_anytime_certificate_gate.v486"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5330
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5330_sea_anytime_certificate_gate_v486.json"
)
EXP5328_RELATIVE_PATH = Path(
    "results/experiment_5328_context_object_lifecycle_self_learning_v486.json"
)
EXP5329_RELATIVE_PATH = Path(
    "results/experiment_5329_memory_context_policy_rollout_v486.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5330_sea_anytime_certificate_gate_v486.py"
)
EXP5328_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5328_context_object_lifecycle_self_learning_v486.py"
)
EXP5329_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5329_memory_context_policy_rollout_v486.py"
)

INFERENCE_SUBSTRATE = "deterministic_anytime_certificate_gate"
SPEC_REFS = (
    "REQ-LEARN-5330",
    "SCENARIO-LEARN-5330-PROMOTE",
    "SCENARIO-LEARN-5330-REJECT",
    "SCENARIO-LEARN-5330-DEFER",
    "SCENARIO-LEARN-5330-NOOP",
    "SCENARIO-LEARN-5330-ROLLBACK",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

DECISION_PROMOTE = "promote"
DECISION_REJECT = "reject"
DECISION_DEFER = "defer"
ANYTIME_MIN_EVIDENCE = 8
DECISIONS = (DECISION_PROMOTE, DECISION_REJECT, DECISION_DEFER)

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5330 certificate-gate artifact so downstream "
        "self-learning cannot confuse policy promotion with model-weight updates."
    ),
    "milestone": (
        "Binds the anytime certificate gate to milestone v486 where context "
        "lifecycle policy promotion is the frozen-model SEA target."
    ),
    "status": (
        "Reports whether candidate policy promotion completed under certificate, "
        "no-op-control, rollback, and frozen-model gates."
    ),
    "honest_verdict": (
        "Terminal Exp5330 verdict; starts with complete: or blocked_ and states "
        "whether reproducible policy promotion avoided unsafe accepts and "
        "model-weight mutation."
    ),
    "inference_substrate": (
        "Declares deterministic anytime certificate evaluation over fixture rows "
        "with no live LLM, API judge, fine-tuning, adapter update, or "
        "foundation-weight mutation."
    ),
    "continuous_self_learning_target": (
        "Bare gate showing the certificate governs continuous self-learning "
        "policy promotion rather than static reporting."
    ),
    "no_weight_mutation": (
        "Bare gate confirming the experiment promotes only deterministic "
        "lifecycle policy choices and never mutates model weights or adapters."
    ),
    "candidate_policy_count": (
        "Bare count of candidate lifecycle policy updates evaluated by the "
        "certificate."
    ),
    "policy_promotions": (
        "Bare count of candidates accepted for promotion by the certificate gate."
    ),
    "policy_rejections": (
        "Bare count of candidates rejected by the certificate gate."
    ),
    "policy_deferrals": (
        "Bare count of candidates deferred because bounded evidence was "
        "insufficient for safe promotion or rejection."
    ),
    "no_op_control_delta": (
        "Bare numeric best improvement delta achieved by the no-op/shuffled "
        "control; candidates must clear this delta before any promotion can be "
        "reported as improvement."
    ),
    "unsafe_promotions": (
        "Bare count of promoted candidates that accept unsafe fixture rows; "
        "value 0 is required for readiness."
    ),
    "rollback_events": (
        "Bare count of rejected or later-invalidated promotions that recorded "
        "rollback behavior."
    ),
    "anytime_certificate_gate_ready": (
        "Bare gate true only when promotion decisions are reproducible, no "
        "unsafe policy is promoted, no-op controls are cleared, rollback is "
        "recorded, tests are recorded, and no model weights mutate."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the "
        "certificate module and artifact are stable."
    ),
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
BARE_INTEGER_FIELDS = (
    "candidate_policy_count",
    "policy_promotions",
    "policy_rejections",
    "policy_deferrals",
    "unsafe_promotions",
    "rollback_events",
)
BARE_NUMERIC_FIELDS = ("no_op_control_delta",)


@dataclass(frozen=True)
class CandidatePolicy:
    """One deterministic lifecycle-policy update considered by the certificate.

    The policy id is an auditable update name, while ``decision_mode`` chooses a
    deterministic evaluator below. Keeping these modes explicit makes the
    certificate easy to audit: the test fixture says what happened, and the gate
    says whether that evidence is enough to promote, reject, or defer.
    """

    policy_id: str
    decision_mode: str
    lifecycle_actions: tuple[str, ...]
    is_control: bool = False
    preliminary_promoted_then_invalidated: bool = False


def load_fixture_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the Exp5328 lifecycle fixture artifact that supplies evidence rows."""

    return _read_json(Path(root) / EXP5328_RELATIVE_PATH)


def confirm_fixture_gate(
    *,
    root: Path | str = REPO_ROOT,
    artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Confirm the source fixture is ready before evaluating promotion claims."""

    source = dict(artifact or load_fixture_artifact(root))
    rows = source.get("lifecycle_rows", [])
    checks = {
        "context_lifecycle_fixture_ready": source.get("context_lifecycle_fixture_ready")
        is True,
        "no_weight_mutation": source.get("no_weight_mutation") is True,
        "lifecycle_rows_present": isinstance(rows, list) and bool(rows),
        "action_counts_present": isinstance(source.get("lifecycle_action_counts"), Mapping),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def build_candidate_policies() -> tuple[CandidatePolicy, ...]:
    """Define the candidate lifecycle updates and the no-op/shuffled control."""

    actions = tuple(exp5328.LIFECYCLE_ACTION_SET)
    shuffled_control_actions = tuple(reversed(actions))
    return (
        CandidatePolicy(
            policy_id="context_lifecycle_certificate_update",
            decision_mode="context_lifecycle",
            lifecycle_actions=actions,
        ),
        CandidatePolicy(
            policy_id="unsafe_accept_all_lifecycle_actions",
            decision_mode="unsafe_pass_through",
            lifecycle_actions=actions,
        ),
        CandidatePolicy(
            policy_id="thin_evidence_context_lifecycle",
            decision_mode="thin_context_lifecycle",
            lifecycle_actions=actions[:3],
        ),
        CandidatePolicy(
            policy_id="no_op_shuffled_control",
            decision_mode="no_op_shuffled_control",
            lifecycle_actions=shuffled_control_actions,
            is_control=True,
        ),
        CandidatePolicy(
            policy_id="retrieval_fast_path_later_invalidated",
            decision_mode="later_invalidated_fast_path",
            lifecycle_actions=tuple(
                action for action in actions if action != "retrieve"
            ),
            preliminary_promoted_then_invalidated=True,
        ),
    )


def evaluate_certificate_gate(
    policies: Sequence[CandidatePolicy],
    fixture_artifact: Mapping[str, Any],
) -> JsonDict:
    """Evaluate promote/reject/defer decisions from deterministic fixture evidence."""

    cases = exp5328.build_lifecycle_fixture()
    policy_rollout = exp5329.evaluate_policy_rollout(cases)
    full_rows = list(fixture_artifact["lifecycle_rows"])
    transition_rows = list(
        policy_rollout["policy_rows"][exp5329.TRANSITION_ONLY_POLICY]
    )
    baseline_quality = float(
        policy_rollout["policy_metrics"][exp5329.TRANSITION_ONLY_POLICY][
            "final_quality"
        ]
    )
    rows = _certificate_rows(
        policies=policies,
        full_rows=full_rows,
        transition_rows=transition_rows,
        baseline_quality=baseline_quality,
    )
    repeat_rows = _certificate_rows(
        policies=policies,
        full_rows=full_rows,
        transition_rows=transition_rows,
        baseline_quality=baseline_quality,
    )
    counts = _decision_counts(rows)
    no_weight_mutation = bool(
        fixture_artifact.get("no_weight_mutation") is True
        and all(not bool(row.get("model_weights_mutated")) for row in full_rows)
    )
    unsafe_promotions = sum(
        1
        for row in rows
        if row["decision"] == DECISION_PROMOTE and row["unsafe_accepts"] > 0
    )
    rollback_events = sum(1 for row in rows if bool(row["rollback_event"]))
    no_op_present = any(bool(row["is_control"]) for row in rows)
    decisions_reproducible = rows == repeat_rows
    ready = bool(
        rows
        and counts[DECISION_PROMOTE] > 0
        and counts[DECISION_REJECT] > 0
        and counts[DECISION_DEFER] > 0
        and no_op_present
        and decisions_reproducible
        and unsafe_promotions == 0
        and rollback_events > 0
        and no_weight_mutation
    )
    return {
        "certificate_rows": rows,
        "candidate_policy_count": len(rows),
        "policy_promotions": counts[DECISION_PROMOTE],
        "policy_rejections": counts[DECISION_REJECT],
        "policy_deferrals": counts[DECISION_DEFER],
        "no_op_control_delta": _no_op_control_delta(rows),
        "unsafe_promotions": unsafe_promotions,
        "rollback_events": rollback_events,
        "decisions_reproducible": decisions_reproducible,
        "no_weight_mutation": no_weight_mutation,
        "anytime_certificate_gate_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5330 artifact from deterministic certificate decisions."""

    fixture_artifact = load_fixture_artifact(root)
    gate = confirm_fixture_gate(root=root)
    evaluation = (
        evaluate_certificate_gate(build_candidate_policies(), fixture_artifact)
        if gate["all_passed"]
        else _blocked_evaluation()
    )
    complete = _certificate_complete(evaluation, gate, tests_run)
    status = "anytime_certificate_gate_ready" if complete else "blocked_fixture_gate_or_tests"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5328_RELATIVE_PATH), str(EXP5329_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, evaluation, gate),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "candidate_policy_count": evaluation["candidate_policy_count"],
        "policy_promotions": evaluation["policy_promotions"],
        "policy_rejections": evaluation["policy_rejections"],
        "policy_deferrals": evaluation["policy_deferrals"],
        "no_op_control_delta": evaluation["no_op_control_delta"],
        "unsafe_promotions": evaluation["unsafe_promotions"],
        "rollback_events": evaluation["rollback_events"],
        "anytime_certificate_gate_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "fixture_gate": gate,
        "certificate_rows": evaluation["certificate_rows"],
        "decisions_reproducible": evaluation["decisions_reproducible"],
        "candidate_policies": [
            {
                "policy_id": policy.policy_id,
                "decision_mode": policy.decision_mode,
                "lifecycle_actions": list(policy.lifecycle_actions),
                "is_control": policy.is_control,
                "preliminary_promoted_then_invalidated": (
                    policy.preliminary_promoted_then_invalidated
                ),
            }
            for policy in build_candidate_policies()
        ]
        if gate["all_passed"]
        else [],
        "weight_mutation_receipt": _weight_mutation_receipt(evaluation),
        "methodology_note": (
            "The anytime certificate is bounded over the fixed Exp5328 fixture. "
            "It is a deterministic policy-promotion gate, not a statistical "
            "claim about model learning and not a model-weight update."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by downstream conductor gates."""

    for field in WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if (
            not isinstance(wrapped, Mapping)
            or "value" not in wrapped
            or wrapped.get("principle") != REQUIRED_FIELD_PRINCIPLES[field]
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    if not str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("continuous_self_learning_target") is not True:
        raise ValueError("continuous_self_learning_target must be bare true")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be bare true")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    for field in BARE_NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    if artifact.get("unsafe_promotions") != 0:
        raise ValueError("unsafe_promotions must be 0")
    if not isinstance(artifact.get("anytime_certificate_gate_ready"), bool):
        raise ValueError("anytime_certificate_gate_ready must be bare bool")
    if artifact["anytime_certificate_gate_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready certificate gate")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5330 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5328": _sha256_file(root_path / EXP5328_RELATIVE_PATH),
        "exp5329": _sha256_file(root_path / EXP5329_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5328_module": _sha256_file(root_path / EXP5328_MODULE_RELATIVE_PATH),
        "exp5329_module": _sha256_file(root_path / EXP5329_MODULE_RELATIVE_PATH),
    }


def _certificate_rows(
    *,
    policies: Sequence[CandidatePolicy],
    full_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    baseline_quality: float,
) -> list[JsonDict]:
    control_metrics = [
        _policy_evidence(policy, full_rows, transition_rows, baseline_quality)
        for policy in policies
        if policy.is_control
    ]
    control_delta = max(
        (float(metrics["certified_delta"]) for metrics in control_metrics),
        default=0.0,
    )
    return [
        _certify_policy(
            policy,
            _policy_evidence(policy, full_rows, transition_rows, baseline_quality),
            control_delta,
        )
        for policy in policies
    ]


def _policy_evidence(
    policy: CandidatePolicy,
    full_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    baseline_quality: float,
) -> JsonDict:
    if policy.decision_mode == "context_lifecycle":
        return _evidence_from_fixture_rows(policy, full_rows, baseline_quality)
    if policy.decision_mode == "unsafe_pass_through":
        return _unsafe_pass_through_evidence(policy, full_rows, baseline_quality)
    if policy.decision_mode == "thin_context_lifecycle":
        thin_rows = full_rows[: ANYTIME_MIN_EVIDENCE - 1]
        return _evidence_from_fixture_rows(policy, thin_rows, baseline_quality)
    if policy.decision_mode == "no_op_shuffled_control":
        return _control_evidence(policy, transition_rows)
    if policy.decision_mode == "later_invalidated_fast_path":
        return _later_invalidated_evidence(policy, full_rows, baseline_quality)
    raise ValueError(f"unknown policy decision_mode: {policy.decision_mode}")


def _certify_policy(
    policy: CandidatePolicy,
    metrics: Mapping[str, Any],
    control_delta: float,
) -> JsonDict:
    reasons: list[str] = []
    preliminary_decision = DECISION_PROMOTE if policy.preliminary_promoted_then_invalidated else None
    rollback_event = False
    if policy.is_control:
        decision = DECISION_REJECT
        reasons.append("control_policy_not_promotable")
    elif metrics["evidence_count"] < ANYTIME_MIN_EVIDENCE:
        decision = DECISION_DEFER
        reasons.append("insufficient_evidence")
    elif policy.preliminary_promoted_then_invalidated:
        decision = DECISION_REJECT
        rollback_event = True
        reasons.append("later_invalidated_promotion")
    elif metrics["unsafe_accepts"] > 0:
        decision = DECISION_REJECT
        rollback_event = True
        reasons.append("unsafe_accepts")
    elif metrics["certified_delta"] <= control_delta:
        decision = DECISION_REJECT
        reasons.append("no_better_than_no_op_control")
    else:
        decision = DECISION_PROMOTE
        reasons.append("certificate_cleared")
    if metrics["unsafe_accepts"] > 0 and "unsafe_accepts" not in reasons:
        reasons.append("unsafe_accepts")
    return {
        "policy_id": policy.policy_id,
        "decision_mode": policy.decision_mode,
        "decision": decision,
        "preliminary_decision": preliminary_decision,
        "is_control": policy.is_control,
        "evidence_count": metrics["evidence_count"],
        "final_quality": metrics["final_quality"],
        "baseline_quality": metrics["baseline_quality"],
        "observed_delta": metrics["observed_delta"],
        "anytime_bound": metrics["anytime_bound"],
        "certified_delta": metrics["certified_delta"],
        "unsafe_accepts": metrics["unsafe_accepts"],
        "rollback_event": rollback_event,
        "model_weights_mutated": False,
        "reasons": reasons,
    }


def _evidence_from_fixture_rows(
    policy: CandidatePolicy,
    rows: Sequence[Mapping[str, Any]],
    baseline_quality: float,
) -> JsonDict:
    final_quality = _fixture_quality(rows)
    return _evidence_metrics(
        policy=policy,
        evidence_count=len(rows),
        final_quality=final_quality,
        baseline_quality=baseline_quality,
        unsafe_accepts=_unsafe_accepts(rows),
        total_evidence=len(exp5328.build_lifecycle_fixture()),
    )


def _unsafe_pass_through_evidence(
    policy: CandidatePolicy,
    rows: Sequence[Mapping[str, Any]],
    baseline_quality: float,
) -> JsonDict:
    safe_count = sum(1 for row in rows if row.get("failure_family") is None)
    final_quality = _rate(safe_count, len(rows))
    return _evidence_metrics(
        policy=policy,
        evidence_count=len(rows),
        final_quality=final_quality,
        baseline_quality=baseline_quality,
        unsafe_accepts=sum(1 for row in rows if row.get("failure_family") is not None),
        total_evidence=len(rows),
    )


def _control_evidence(
    policy: CandidatePolicy,
    transition_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    final_quality = _rate(
        sum(1 for row in transition_rows if bool(row["final_correct"])),
        len(transition_rows),
    )
    return {
        "policy_id": policy.policy_id,
        "evidence_count": len(transition_rows),
        "final_quality": final_quality,
        "baseline_quality": final_quality,
        "observed_delta": 0.0,
        "anytime_bound": 0.0,
        "certified_delta": 0.0,
        "unsafe_accepts": 0,
    }


def _later_invalidated_evidence(
    policy: CandidatePolicy,
    rows: Sequence[Mapping[str, Any]],
    baseline_quality: float,
) -> JsonDict:
    final_correct = 0
    unsafe_accepts = 0
    for row in rows:
        family = row.get("failure_family")
        if family == exp5328.RETRIEVAL_FAILURE_FAMILY:
            unsafe_accepts += 1
            continue
        final_correct += 1
    return _evidence_metrics(
        policy=policy,
        evidence_count=len(rows),
        final_quality=_rate(final_correct, len(rows)),
        baseline_quality=baseline_quality,
        unsafe_accepts=unsafe_accepts,
        total_evidence=len(rows),
    )


def _evidence_metrics(
    *,
    policy: CandidatePolicy,
    evidence_count: int,
    final_quality: float,
    baseline_quality: float,
    unsafe_accepts: int,
    total_evidence: int,
) -> JsonDict:
    observed_delta = _delta(final_quality, baseline_quality)
    anytime_bound = _anytime_bound(evidence_count, total_evidence)
    certified_delta = _delta(observed_delta, anytime_bound)
    return {
        "policy_id": policy.policy_id,
        "evidence_count": evidence_count,
        "final_quality": final_quality,
        "baseline_quality": baseline_quality,
        "observed_delta": observed_delta,
        "anytime_bound": anytime_bound,
        "certified_delta": certified_delta,
        "unsafe_accepts": unsafe_accepts,
    }


def _fixture_quality(rows: Sequence[Mapping[str, Any]]) -> float:
    final_correct = 0
    for row in rows:
        safe = row.get("failure_family") is None
        accepted = bool(row.get("accepted"))
        final_correct += int(accepted if safe else not accepted)
    return _rate(final_correct, len(rows))


def _unsafe_accepts(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("failure_family") is not None and bool(row.get("accepted"))
    )


def _no_op_control_delta(rows: Sequence[Mapping[str, Any]]) -> float:
    return max(
        (float(row["certified_delta"]) for row in rows if bool(row["is_control"])),
        default=0.0,
    )


def _anytime_bound(evidence_count: int, total_evidence: int) -> float:
    if total_evidence <= 0:
        return 1.0
    return _rate(max(total_evidence - evidence_count, 0), total_evidence)


def _decision_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        decision: sum(1 for row in rows if row.get("decision") == decision)
        for decision in DECISIONS
    }


def _blocked_evaluation() -> JsonDict:
    return {
        "certificate_rows": [],
        "candidate_policy_count": 0,
        "policy_promotions": 0,
        "policy_rejections": 0,
        "policy_deferrals": 0,
        "no_op_control_delta": 0.0,
        "unsafe_promotions": 0,
        "rollback_events": 0,
        "decisions_reproducible": False,
        "no_weight_mutation": True,
        "anytime_certificate_gate_ready": False,
    }


def _certificate_complete(
    evaluation: Mapping[str, Any],
    gate: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        gate["all_passed"]
        and evaluation["anytime_certificate_gate_ready"]
        and evaluation["policy_promotions"] > 0
        and evaluation["policy_rejections"] > 0
        and evaluation["policy_deferrals"] > 0
        and evaluation["unsafe_promotions"] == 0
        and evaluation["rollback_events"] > 0
        and evaluation["decisions_reproducible"]
        and evaluation["no_weight_mutation"]
        and bool(tests_run)
    )


def _honest_verdict(
    complete: bool,
    evaluation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    if complete:
        return (
            "complete: deterministic anytime certificate promoted "
            f"{evaluation['policy_promotions']} lifecycle policy, rejected "
            f"{evaluation['policy_rejections']}, deferred "
            f"{evaluation['policy_deferrals']}, kept unsafe promotions at 0, "
            f"recorded {evaluation['rollback_events']} rollback events, and "
            "preserved frozen-model discipline"
        )
    failed = ",".join(gate.get("failed_gates", [])) or "tests_or_certificate_gate"
    return f"blocked_fixture_gate_not_ready: {failed}"


def _weight_mutation_receipt(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": ["deterministic_context_lifecycle_policy"],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_update": False,
        },
    }


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": REQUIRED_FIELD_PRINCIPLES[field], "value": value}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _is_numeric(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 12)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
