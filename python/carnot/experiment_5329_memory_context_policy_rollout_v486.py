"""Exp5329: memory/context policy rollout on the lifecycle fixture.

Spec refs: REQ-LEARN-5329, SCENARIO-LEARN-5329-POLICY.

This experiment compares policy choices over the deterministic Exp5328
context-object lifecycle fixture. It does not call an LLM and it does not touch
model weights. The only changing state is the JSON-like context bank from the
fixture, which lets the rollout separate final quality from process metrics such
as verifier calls, missed bank/retrieval/answer failures, rollback, and
recoverability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5328_context_object_lifecycle_self_learning_v486 as exp5328
from carnot.pipeline.memory_transition_verifier import MemoryTransitionProposal
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5329_memory_context_policy_rollout_v486"
EXPERIMENT_ID = 5329
MILESTONE = "v486"
SCHEMA = "carnot.experiment_5329.memory_context_policy_rollout.v486"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5329
RESULT_RELATIVE_PATH = Path("results/experiment_5329_memory_context_policy_rollout_v486.json")
EXP5328_RELATIVE_PATH = Path(
    "results/experiment_5328_context_object_lifecycle_self_learning_v486.json"
)
EXP5313_RELATIVE_PATH = Path(
    "results/experiment_5313_gated_memory_transition_policy_rollout_v485.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5329_memory_context_policy_rollout_v486.py")
EXP5328_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5328_context_object_lifecycle_self_learning_v486.py"
)
INFERENCE_SUBSTRATE = "deterministic_context_policy_rollout"
SPEC_REFS = ("REQ-LEARN-5329", "SCENARIO-LEARN-5329-POLICY")
TERMINAL_PREFIXES = ("complete:", "blocked_")

ALWAYS_FULL_POLICY = "always_full_verification"
TRANSITION_ONLY_POLICY = "transition_only_verifier"
CONTEXT_LIFECYCLE_POLICY = "context_lifecycle_policy_with_rollback"
POLICY_ARMS = (
    ALWAYS_FULL_POLICY,
    TRANSITION_ONLY_POLICY,
    CONTEXT_LIFECYCLE_POLICY,
)

ROUTE_FULL_CONTEXT = "full_context_verifier"
ROUTE_TRANSITION_ONLY = "transition_only_verifier"
ROUTE_CONTEXT_LIFECYCLE = "context_lifecycle_policy"
ROUTE_CONTEXT_PASS_THROUGH = "context_pass_through_without_lifecycle_checks"

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5329 rollout artifact so downstream gates cannot "
        "confuse lifecycle-aware context policy selection with Exp5313 "
        "transition-only memory rollout or Exp5328 fixture construction."
    ),
    "milestone": (
        "Binds the rollout to milestone v486 where lifecycle-aware context policy "
        "replaces fixed transition-only verifier dosing as the self-learning target."
    ),
    "status": (
        "Reports whether the lifecycle policy comparison completed after fixture "
        "checks instead of merely finding source artifacts."
    ),
    "honest_verdict": (
        "Terminal Exp5329 verdict; starts with complete: or blocked_ and states "
        "whether lifecycle policy avoided verifier calls without lowering quality "
        "or accepting unsafe state changes."
    ),
    "inference_substrate": (
        "Declares deterministic context-policy rollout with no live LLM, API judge, "
        "model generation, fine-tuning, adapter update, or model weight mutation."
    ),
    "continuous_self_learning_target": (
        "Bare gate showing the rollout evaluates policy choices for continuous "
        "self-learning rather than static context documentation."
    ),
    "no_weight_mutation": (
        "Bare gate confirming only deterministic context-object bank state changed, "
        "never model weights or adapters."
    ),
    "quality_delta_vs_always_full": (
        "Bare final-quality delta comparing the context-lifecycle policy with "
        "rollback against always-full verification."
    ),
    "verifier_calls_avoided": (
        "Bare integer count of verifier calls avoided by the context-lifecycle "
        "policy with rollback relative to always-full on identical cases."
    ),
    "bank_failure_delta": (
        "Bare bank failure-rate delta comparing the context-lifecycle policy with "
        "rollback against always-full verification."
    ),
    "retrieval_failure_delta": (
        "Bare retrieval failure-rate delta comparing the context-lifecycle policy "
        "with rollback against always-full verification."
    ),
    "answer_failure_delta": (
        "Bare answer failure-rate delta comparing the context-lifecycle policy with "
        "rollback against always-full verification."
    ),
    "unsafe_false_accepts": (
        "Bare integer count of unsafe state-change accepts by the context-lifecycle "
        "policy; any positive count blocks rollout readiness."
    ),
    "rollback_events": (
        "Bare integer count of rollback transitions exercised by the "
        "context-lifecycle policy with rollback."
    ),
    "policy_rollout_ready": (
        "Bare gate showing all three policy variants ran on the same deterministic "
        "lifecycle cases and preserved the safety gates."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the rollout "
        "module and artifact are stable."
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
BARE_NUMERIC_FIELDS = (
    "quality_delta_vs_always_full",
    "bank_failure_delta",
    "retrieval_failure_delta",
    "answer_failure_delta",
)
BARE_INTEGER_FIELDS = (
    "verifier_calls_avoided",
    "unsafe_false_accepts",
    "rollback_events",
)


@dataclass(frozen=True)
class PolicyPanel:
    """The shared deterministic lifecycle cases for every policy arm."""

    cases: tuple[exp5328.LifecycleCase, ...]


def build_policy_panel() -> tuple[exp5328.LifecycleCase, ...]:
    """Return the Exp5328 lifecycle cases used by every rollout policy."""

    return exp5328.build_lifecycle_fixture()


def load_fixture_artifact(root: Path | str = REPO_ROOT) -> JsonDict:
    """Read the Exp5328 artifact that gates this rollout."""

    return _read_json(Path(root) / EXP5328_RELATIVE_PATH)


def confirm_fixture_gate(
    *,
    root: Path | str = REPO_ROOT,
    artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Confirm Exp5328 exposes lifecycle, rollback, and recoverability metrics."""

    source = dict(artifact or load_fixture_artifact(root))
    rows = source.get("lifecycle_rows", [])
    checks = {
        "context_lifecycle_fixture_ready": source.get("context_lifecycle_fixture_ready") is True,
        "bank_failure_detection_rate_present": _is_numeric(
            source.get("bank_failure_detection_rate")
        ),
        "retrieval_failure_detection_rate_present": _is_numeric(
            source.get("retrieval_failure_detection_rate")
        ),
        "answer_failure_detection_rate_present": _is_numeric(
            source.get("answer_failure_detection_rate")
        ),
        "rollback_success_rate_present": _is_numeric(source.get("rollback_success_rate")),
        "recoverability_metrics_present": _recoverability_metrics_present(rows),
        "no_weight_mutation": source.get("no_weight_mutation") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def evaluate_policy_rollout(panel: Sequence[exp5328.LifecycleCase]) -> JsonDict:
    """Evaluate all policy arms on identical deterministic lifecycle cases."""

    full_rows = exp5328.evaluate_lifecycle_fixture(panel)["lifecycle_rows"]
    policy_rows = {
        policy: [
            _evaluate_policy_case(case, policy, full_row)
            for case, full_row in zip(panel, full_rows, strict=True)
        ]
        for policy in POLICY_ARMS
    }
    policy_metrics = {policy: _policy_metrics(rows) for policy, rows in policy_rows.items()}
    always = policy_metrics[ALWAYS_FULL_POLICY]
    lifecycle = policy_metrics[CONTEXT_LIFECYCLE_POLICY]
    all_variants_ran = bool(
        panel
        and set(policy_rows) == set(POLICY_ARMS)
        and all(len(rows) == len(panel) for rows in policy_rows.values())
        and _same_case_ids(policy_rows)
    )
    no_weight_mutation = all(
        row["model_weights_mutated"] is False for rows in policy_rows.values() for row in rows
    )
    quality_delta = _delta(lifecycle["final_quality"], always["final_quality"])
    bank_delta = _delta(lifecycle["bank_failure_rate"], always["bank_failure_rate"])
    retrieval_delta = _delta(
        lifecycle["retrieval_failure_rate"],
        always["retrieval_failure_rate"],
    )
    answer_delta = _delta(lifecycle["answer_failure_rate"], always["answer_failure_rate"])
    verifier_calls_avoided = int(always["verifier_calls"] - lifecycle["verifier_calls"])
    unsafe_false_accepts = int(lifecycle["unsafe_false_accepts"])
    rollback_events = int(lifecycle["rollback_events"])
    recoveries = int(lifecycle["recoveries"])
    ready = bool(
        all_variants_ran
        and quality_delta >= 0.0
        and bank_delta <= 0.0
        and retrieval_delta <= 0.0
        and answer_delta <= 0.0
        and verifier_calls_avoided > 0
        and unsafe_false_accepts == 0
        and rollback_events > 0
        and recoveries > 0
        and no_weight_mutation
    )
    return {
        "all_variants_ran": all_variants_ran,
        "policy_rollout_ready": ready,
        "policy_rows": policy_rows,
        "policy_metrics": policy_metrics,
        "quality_delta_vs_always_full": quality_delta,
        "verifier_calls_avoided": verifier_calls_avoided,
        "bank_failure_delta": bank_delta,
        "retrieval_failure_delta": retrieval_delta,
        "answer_failure_delta": answer_delta,
        "unsafe_false_accepts": unsafe_false_accepts,
        "rollback_events": rollback_events,
        "recoveries": recoveries,
        "no_weight_mutation": no_weight_mutation,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5329 artifact from deterministic policy rollout."""

    gate = confirm_fixture_gate(root=root)
    evaluation = (
        evaluate_policy_rollout(build_policy_panel())
        if gate["all_passed"]
        else _blocked_evaluation()
    )
    complete = _rollout_complete(evaluation, gate, tests_run)
    status = "policy_rollout_ready" if complete else "blocked_fixture_gate_or_tests"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5328_RELATIVE_PATH), str(EXP5313_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(complete, evaluation, gate)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "quality_delta_vs_always_full": evaluation["quality_delta_vs_always_full"],
        "verifier_calls_avoided": evaluation["verifier_calls_avoided"],
        "bank_failure_delta": evaluation["bank_failure_delta"],
        "retrieval_failure_delta": evaluation["retrieval_failure_delta"],
        "answer_failure_delta": evaluation["answer_failure_delta"],
        "unsafe_false_accepts": evaluation["unsafe_false_accepts"],
        "rollback_events": evaluation["rollback_events"],
        "recoveries": evaluation["recoveries"],
        "policy_rollout_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "fixture_gate": gate,
        "policy_metrics": evaluation["policy_metrics"],
        "policy_rows": evaluation["policy_rows"],
        "weight_mutation_receipt": _weight_mutation_receipt(evaluation),
        "methodology_note": (
            "Failure rates are deterministic missed-failure rates over the Exp5328 "
            "fixture families, not statistical estimates. Transition-only misses "
            "retrieval and answer-time context failures because it checks bank "
            "transitions but not lifecycle context use."
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
    for field in BARE_NUMERIC_FIELDS:
        value = artifact.get(field)
        if not _is_numeric(value):
            raise ValueError(f"{field} must be bare numeric")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    if not isinstance(artifact.get("policy_rollout_ready"), bool):
        raise ValueError("policy_rollout_ready must be bare bool")
    if artifact["policy_rollout_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready rollout")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5329 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5328": _sha256_file(root_path / EXP5328_RELATIVE_PATH),
        "exp5313": _sha256_file(root_path / EXP5313_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5328_module": _sha256_file(root_path / EXP5328_MODULE_RELATIVE_PATH),
    }


def _evaluate_policy_case(
    case: exp5328.LifecycleCase,
    policy: str,
    full_row: Mapping[str, Any],
) -> JsonDict:
    if policy == ALWAYS_FULL_POLICY:
        return _row_from_full_lifecycle(case, policy, full_row, verifier_call=True)
    if policy == CONTEXT_LIFECYCLE_POLICY:
        return _row_from_full_lifecycle(
            case,
            policy,
            full_row,
            verifier_call=bool(full_row["transition_verifier_reused"]),
        )
    return _transition_only_row(case)


def _row_from_full_lifecycle(
    case: exp5328.LifecycleCase,
    policy: str,
    full_row: Mapping[str, Any],
    *,
    verifier_call: bool,
) -> JsonDict:
    accepted = bool(full_row["accepted"])
    detected_failure = bool(full_row["detected_failure"])
    changed = bool(full_row["committed_state_changed"])
    return {
        "case_id": case.case_id,
        "action": case.action,
        "policy": policy,
        "route": ROUTE_FULL_CONTEXT if policy == ALWAYS_FULL_POLICY else ROUTE_CONTEXT_LIFECYCLE,
        "failure_family": case.failure_family,
        "failure_mode": case.failure_mode,
        "safe_expected": case.safe_expected,
        "accepted": accepted,
        "detected_failure": detected_failure,
        "final_correct": _final_correct(case, accepted),
        "verifier_call": verifier_call,
        "persistent_state_changed": changed,
        "unsafe_false_accept": _unsafe_state_change_accept(case, accepted, changed),
        "rollback_event": bool(full_row["rollback_success"]),
        "recovered_from_sidecar": bool(full_row["recovered_from_sidecar"]),
        "model_weights_mutated": False,
        "rejection_reasons": list(full_row["rejection_reasons"]),
    }


def _transition_only_row(case: exp5328.LifecycleCase) -> JsonDict:
    verifier_call = case.action in exp5328.BANK_MUTATION_ACTIONS
    if verifier_call:
        transition = _transition_only_transition(case)
        route = ROUTE_TRANSITION_ONLY
    else:
        transition = {
            "accepted": True,
            "detected_failure": False,
            "persistent_state_changed": False,
            "rejection_reasons": [],
        }
        route = ROUTE_CONTEXT_PASS_THROUGH
    accepted = bool(transition["accepted"])
    detected_failure = bool(transition["detected_failure"])
    changed = bool(transition["persistent_state_changed"])
    recovered = bool(case.action == "rollback" and accepted and _recoverable_from_sidecar(case))
    rollback_event = bool(
        case.rollback_expected
        and (recovered or (not case.safe_expected and detected_failure and not changed))
    )
    return {
        "case_id": case.case_id,
        "action": case.action,
        "policy": TRANSITION_ONLY_POLICY,
        "route": route,
        "failure_family": case.failure_family,
        "failure_mode": case.failure_mode,
        "safe_expected": case.safe_expected,
        "accepted": accepted,
        "detected_failure": detected_failure,
        "final_correct": _final_correct(case, accepted),
        "verifier_call": verifier_call,
        "persistent_state_changed": changed,
        "unsafe_false_accept": _unsafe_state_change_accept(case, accepted, changed),
        "rollback_event": rollback_event,
        "recovered_from_sidecar": recovered,
        "model_weights_mutated": False,
        "rejection_reasons": list(transition["rejection_reasons"]),
    }


def _transition_only_transition(case: exp5328.LifecycleCase) -> JsonDict:
    verifier = exp5328.build_lifecycle_verifier()
    proposal = MemoryTransitionProposal(
        transition_id=f"t5329-transition-only-{case.case_id}",
        label=case.action if case.safe_expected else str(case.failure_mode),
        source_stress_event_id=case.case_id,
        prior_state=case.prior_state,
        proposed_state=case.proposed_state,
        expected_state=case.expected_state,
        protected_keys=case.protected_keys,
        safe_expected=case.safe_expected,
    )
    before = deepcopy(case.prior_state)
    decision, committed = verifier.commit_if_safe(before, proposal)
    accepted = bool(decision.accepted)
    changed = committed != case.prior_state
    detected_failure = bool(case.failure_family is not None and not accepted)
    return {
        "accepted": accepted,
        "detected_failure": detected_failure,
        "persistent_state_changed": changed,
        "rejection_reasons": list(decision.rejection_reasons),
    }


def _policy_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    n_rows = len(rows)
    final_correct = sum(1 for row in rows if bool(row["final_correct"]))
    return {
        "n": n_rows,
        "final_correct": final_correct,
        "final_quality": _rate(final_correct, n_rows),
        "verifier_calls": sum(1 for row in rows if bool(row["verifier_call"])),
        "bank_failure_rate": _family_failure_rate(rows, exp5328.BANK_FAILURE_FAMILY),
        "retrieval_failure_rate": _family_failure_rate(
            rows,
            exp5328.RETRIEVAL_FAILURE_FAMILY,
        ),
        "answer_failure_rate": _family_failure_rate(rows, exp5328.ANSWER_FAILURE_FAMILY),
        "unsafe_false_accepts": sum(1 for row in rows if bool(row["unsafe_false_accept"])),
        "rollback_events": sum(1 for row in rows if bool(row["rollback_event"])),
        "recoveries": sum(1 for row in rows if bool(row["recovered_from_sidecar"])),
        "model_weights_mutated": any(bool(row["model_weights_mutated"]) for row in rows),
    }


def _blocked_evaluation() -> JsonDict:
    empty_metrics = {
        policy: {
            "n": 0,
            "final_correct": 0,
            "final_quality": 0.0,
            "verifier_calls": 0,
            "bank_failure_rate": 0.0,
            "retrieval_failure_rate": 0.0,
            "answer_failure_rate": 0.0,
            "unsafe_false_accepts": 0,
            "rollback_events": 0,
            "recoveries": 0,
            "model_weights_mutated": False,
        }
        for policy in POLICY_ARMS
    }
    return {
        "all_variants_ran": False,
        "policy_rollout_ready": False,
        "policy_rows": {policy: [] for policy in POLICY_ARMS},
        "policy_metrics": empty_metrics,
        "quality_delta_vs_always_full": 0.0,
        "verifier_calls_avoided": 0,
        "bank_failure_delta": 0.0,
        "retrieval_failure_delta": 0.0,
        "answer_failure_delta": 0.0,
        "unsafe_false_accepts": 0,
        "rollback_events": 0,
        "recoveries": 0,
        "no_weight_mutation": True,
    }


def _rollout_complete(
    evaluation: Mapping[str, Any],
    gate: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        gate["all_passed"]
        and evaluation["policy_rollout_ready"]
        and evaluation["quality_delta_vs_always_full"] >= 0.0
        and evaluation["bank_failure_delta"] <= 0.0
        and evaluation["retrieval_failure_delta"] <= 0.0
        and evaluation["answer_failure_delta"] <= 0.0
        and evaluation["verifier_calls_avoided"] > 0
        and evaluation["unsafe_false_accepts"] == 0
        and evaluation["rollback_events"] > 0
        and evaluation["recoveries"] > 0
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
            "complete: context-lifecycle policy matched always-full quality, "
            f"avoided {evaluation['verifier_calls_avoided']} verifier calls, kept "
            "bank/retrieval/answer failure deltas at 0, exercised "
            f"{evaluation['rollback_events']} rollbacks and {evaluation['recoveries']} "
            "recoveries, and preserved no weight mutation"
        )
    failed = ",".join(gate.get("failed_gates", [])) or "tests_or_policy_rollout_gate"
    return f"blocked_fixture_gate_not_ready: {failed}"


def _weight_mutation_receipt(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": ["deterministic_context_object_bank"],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "cross_model_transfer_claim": False,
        },
    }


def _same_case_ids(policy_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> bool:
    case_id_sets = [tuple(row["case_id"] for row in rows) for rows in policy_rows.values()]
    return len(set(case_id_sets)) <= 1


def _family_failure_rate(rows: Sequence[Mapping[str, Any]], family: str) -> float:
    family_rows = [row for row in rows if row["failure_family"] == family]
    missed = sum(1 for row in family_rows if not bool(row["detected_failure"]))
    return _rate(missed, len(family_rows))


def _final_correct(case: exp5328.LifecycleCase, accepted: bool) -> bool:
    if case.failure_family is None:
        return bool(accepted)
    return not bool(accepted)


def _unsafe_state_change_accept(
    case: exp5328.LifecycleCase,
    accepted: bool,
    changed: bool,
) -> bool:
    return bool(case.failure_family is not None and accepted and changed)


def _recoverable_from_sidecar(case: exp5328.LifecycleCase) -> bool:
    if case.recover_object_id is None:
        return False
    obj = case.proposed_state.get(case.recover_object_id) or case.prior_state.get(
        case.recover_object_id
    )
    if obj is None:
        return False
    sidecar = obj.get("recoverable_sidecar", {})
    return bool(sidecar.get("recoverable") and sidecar.get("restore_payload"))


def _recoverability_metrics_present(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    return bool(
        any(bool(row.get("recovered_from_sidecar")) for row in rows if isinstance(row, Mapping))
        and any(bool(row.get("sidecar_preserved")) for row in rows if isinstance(row, Mapping))
    )


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
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


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
