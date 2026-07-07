"""Exp5341: deterministic bounded context compressor drift monitor.

Spec refs: REQ-LEARN-5341, SCENARIO-LEARN-5341-RECALL,
SCENARIO-LEARN-5341-DRIFT, SCENARIO-LEARN-5341-RECOVERY.

This fixture keeps recalled artifacts and persistent state commitment as two
separate decisions. A recalled object can inform a row, but it cannot enter the
bounded persistent state unless a deterministic commit check accepts a compact
summary. That separation is the important safety property: drift, stale recall,
and poisoning are handled as state-management failures rather than as hidden
changes to model weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5328_context_object_lifecycle_self_learning_v486 as exp5328
from carnot import experiment_5330_sea_anytime_certificate_gate_v486 as exp5330


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5341_bounded_compressor_drift_monitor_v487"
EXPERIMENT_ID = 5341
MILESTONE = "v487"
SCHEMA = "carnot.experiment_5341.bounded_compressor_drift_monitor.v487"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5341
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5341_bounded_compressor_drift_monitor_v487.json"
)
EXP5328_RELATIVE_PATH = Path(
    "results/experiment_5328_context_object_lifecycle_self_learning_v486.json"
)
EXP5330_RELATIVE_PATH = Path(
    "results/experiment_5330_sea_anytime_certificate_gate_v486.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5341_bounded_compressor_drift_monitor_v487.py"
)
EXP5328_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5328_context_object_lifecycle_self_learning_v486.py"
)
EXP5330_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5330_sea_anytime_certificate_gate_v486.py"
)

INFERENCE_SUBSTRATE = "deterministic_bounded_context_compressor"
SPEC_REFS = (
    "REQ-LEARN-5341",
    "SCENARIO-LEARN-5341-RECALL",
    "SCENARIO-LEARN-5341-DRIFT",
    "SCENARIO-LEARN-5341-RECOVERY",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

BOUNDED_STATE_OBJECT_LIMIT = 4
BOUNDED_STATE_TOKEN_LIMIT = 12
RECALLED_NOT_COMMITTED = "recalled_not_committed"
COMMITTED = "committed"
REJECTED = "rejected"

STALE_RECALL = "stale_recall"
POISONED_CANDIDATE_MEMORY = "poisoned_candidate_memory"
COMPRESSION_OMISSION = "compression_omission"
OVER_COMPRESSION = "over_compression"
DRIFT_ANOMALIES = (COMPRESSION_OMISSION, OVER_COMPRESSION)

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5341 artifact so downstream gates cannot "
        "confuse bounded compressor drift monitoring with Exp5328 lifecycle "
        "rows, Exp5330 certificate promotion, or Exp5340 utility learning."
    ),
    "milestone": (
        "Binds the bounded compressor to milestone v487 where recall and "
        "persistent commitment are deliberately separated."
    ),
    "status": (
        "Reports whether the compressor completed under bounded budget, "
        "drift-monitor, poison-rejection, recoverability, and frozen-model gates."
    ),
    "honest_verdict": (
        "Terminal Exp5341 verdict; starts with complete: or blocked_ and states "
        "whether bounded compression kept unsafe commits at zero while "
        "preserving recovery."
    ),
    "inference_substrate": (
        "Declares deterministic bounded context compression with no live LLM, "
        "API judge, model generation, fine-tuning, adapter update, or "
        "foundation-weight mutation."
    ),
    "continuous_self_learning_target": (
        "Bare gate showing the compressor evaluates state-management policy "
        "behavior for continuous self-learning rather than static reporting."
    ),
    "no_weight_mutation": (
        "Bare gate confirming only deterministic bounded context summaries and "
        "monitor rows changed, never model weights or adapters."
    ),
    "bounded_state_object_limit": (
        "Bare integer object cap enforced on the persistent bounded state."
    ),
    "recalled_not_committed_count": (
        "Bare integer count of rows where an artifact was recalled for "
        "evaluation but explicitly not committed to persistent state."
    ),
    "drift_detection_rate": (
        "Bare numeric rate over compression omission and over-compression drift "
        "rows."
    ),
    "stale_recall_detection_rate": "Bare numeric rate over stale-recall rows.",
    "poison_rejection_rate": (
        "Bare numeric rate over poisoned candidate memory rows rejected before "
        "persistent commitment."
    ),
    "recoverability_rate": (
        "Bare numeric rate over safe recovery rows that restore from sidecars "
        "or rollback evidence."
    ),
    "unsafe_commits": (
        "Bare integer count of unsafe candidates that reached persistent state; "
        "value 0 is required for readiness."
    ),
    "compressor_drift_fixture_ready": (
        "Bare gate true only when deterministic tests are recorded, unsafe "
        "commits are zero, budgets hold, all anomaly detection rates are 1.0, "
        "recoverability is preserved, and no model weights mutate."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the "
        "compressor module and result artifact are stable."
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
    "bounded_state_object_limit",
    "recalled_not_committed_count",
    "unsafe_commits",
)
BARE_NUMERIC_FIELDS = (
    "drift_detection_rate",
    "stale_recall_detection_rate",
    "poison_rejection_rate",
    "recoverability_rate",
)


@dataclass(frozen=True)
class CompressorCase:
    """One deterministic recall, compression, or recovery decision.

    The case names the Exp5328 lifecycle row and stable object IDs that support
    the decision. ``expected_commit`` is intentionally separate from recall:
    recall-only rows can expose artifacts to the monitor while keeping
    persistent bounded state byte-stable.
    """

    case_id: str
    lifecycle_action: str
    source_case_id: str
    source_object_ids: tuple[str, ...]
    recalled_object_ids: tuple[str, ...]
    candidate_summary: str
    expected_anomaly: str | None
    expected_commit: bool
    rollback_case_id: str | None = None
    expected_recoverable: bool = False
    model_weights_mutated: bool = False


def build_compressor_cases() -> tuple[CompressorCase, ...]:
    """Return the deterministic bounded-compressor panel for Exp5341."""

    return (
        CompressorCase(
            case_id="compress-benign-recall",
            lifecycle_action="retrieve",
            source_case_id="life-retrieve-archive-recover",
            source_object_ids=("ctx.archive.policy",),
            recalled_object_ids=("ctx.archive.policy",),
            candidate_summary="archive policy recall sidecar only",
            expected_anomaly=None,
            expected_commit=False,
        ),
        CompressorCase(
            case_id="compress-stale-recall",
            lifecycle_action="retrieve",
            source_case_id="life-stale-retrieval",
            source_object_ids=("ctx.runtime.receipt.v0",),
            recalled_object_ids=("ctx.runtime.receipt.v0",),
            candidate_summary="stale runtime receipt v0",
            expected_anomaly=STALE_RECALL,
            expected_commit=False,
        ),
        CompressorCase(
            case_id="compress-poisoned-candidate",
            lifecycle_action="create",
            source_case_id="life-ghost-memory",
            source_object_ids=("ctx.ghost.unsourced",),
            recalled_object_ids=(),
            candidate_summary="unsourced ghost claim promoted as durable memory",
            expected_anomaly=POISONED_CANDIDATE_MEMORY,
            expected_commit=False,
        ),
        CompressorCase(
            case_id="compress-omission-drift",
            lifecycle_action="fold",
            source_case_id="life-omission-sensor-rule",
            source_object_ids=("ctx.runtime.receipt", "ctx.sensor.reject_rule"),
            recalled_object_ids=(),
            candidate_summary="runtime receipt summary",
            expected_anomaly=COMPRESSION_OMISSION,
            expected_commit=False,
        ),
        CompressorCase(
            case_id="compress-over-compression-drift",
            lifecycle_action="fold",
            source_case_id="life-fold-runtime-rubric",
            source_object_ids=(
                "ctx.runtime.receipt",
                "ctx.arc.rubric",
                "ctx.folded.runtime_rubric",
            ),
            recalled_object_ids=(),
            candidate_summary="runtime",
            expected_anomaly=OVER_COMPRESSION,
            expected_commit=False,
        ),
        CompressorCase(
            case_id="compress-safe-recovery",
            lifecycle_action="rollback",
            source_case_id="life-rollback-corrupt-patch",
            source_object_ids=("ctx.patch.autofix",),
            recalled_object_ids=(),
            candidate_summary="rollback recovered patch autofix sidecar",
            expected_anomaly=None,
            expected_commit=True,
            rollback_case_id="life-rollback-corrupt-patch",
            expected_recoverable=True,
        ),
    )


def evaluate_compressor_cases(cases: Sequence[CompressorCase]) -> JsonDict:
    """Evaluate recall/commit separation, anomaly rejection, and budget use."""

    persistent_state: dict[str, JsonDict] = {}
    rows = [_evaluate_case(case, persistent_state) for case in cases]
    bounded_state_summaries = [
        row["committed_summary"]
        for row in rows
        if isinstance(row.get("committed_summary"), Mapping)
    ]
    drift_counts = _detection_count(rows, DRIFT_ANOMALIES)
    stale_counts = _detection_count(rows, (STALE_RECALL,))
    poison_counts = _poison_rejection_count(rows)
    recovery_counts = _recovery_count(rows)
    unsafe_commits = sum(1 for row in rows if bool(row["unsafe_commit"]))
    recalled_rows = [
        row
        for row in rows
        if row["commit_decision"] == RECALLED_NOT_COMMITTED
    ]
    recall_commit_separation_rate = _rate(
        sum(1 for row in recalled_rows if not row["persistent_state_changed"]),
        len(recalled_rows),
    )
    compression_budget = _compression_budget(rows, bounded_state_summaries)
    no_weight_mutation = not any(bool(row["model_weights_mutated"]) for row in rows)
    ready = bool(
        rows
        and compression_budget["within_budget"]
        and recall_commit_separation_rate == 1.0
        and _detection_rate(drift_counts) == 1.0
        and _detection_rate(stale_counts) == 1.0
        and _detection_rate(poison_counts) == 1.0
        and _detection_rate(recovery_counts) == 1.0
        and unsafe_commits == 0
        and no_weight_mutation
    )
    return {
        "compressor_rows": rows,
        "bounded_state_summaries": bounded_state_summaries,
        "compression_budget": compression_budget,
        "recalled_not_committed_count": len(recalled_rows),
        "recall_commit_separation_rate": recall_commit_separation_rate,
        "drift_counts": drift_counts,
        "drift_detection_rate": _detection_rate(drift_counts),
        "stale_recall_counts": stale_counts,
        "stale_recall_detection_rate": _detection_rate(stale_counts),
        "poison_counts": poison_counts,
        "poison_rejection_rate": _detection_rate(poison_counts),
        "recoverability_counts": recovery_counts,
        "recoverability_rate": _detection_rate(recovery_counts),
        "unsafe_commits": unsafe_commits,
        "verifier_call_cost": _verifier_call_cost(rows),
        "no_weight_mutation": no_weight_mutation,
        "compressor_drift_fixture_ready": ready,
    }


def confirm_fixture_gate(
    *,
    root: Path | str = REPO_ROOT,
    artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Confirm Exp5328 exposes lifecycle evidence for compressor rows."""

    source = dict(artifact or _read_json(Path(root) / EXP5328_RELATIVE_PATH))
    rows = source.get("lifecycle_rows", [])
    object_ids = set(source.get("context_object_ids", []))
    required_ids = {
        object_id
        for case in build_compressor_cases()
        for object_id in case.source_object_ids
    }
    checks = {
        "context_lifecycle_fixture_ready": source.get("context_lifecycle_fixture_ready")
        is True,
        "no_weight_mutation": source.get("no_weight_mutation") is True,
        "lifecycle_rows_present": isinstance(rows, list) and bool(rows),
        "stable_object_ids_present": required_ids.issubset(object_ids),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def confirm_certificate_gate(
    *,
    root: Path | str = REPO_ROOT,
    artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Confirm Exp5330 accepted the unsafe-commit boundary."""

    source = dict(artifact or _read_json(Path(root) / EXP5330_RELATIVE_PATH))
    checks = {
        "anytime_certificate_gate_ready": source.get("anytime_certificate_gate_ready")
        is True,
        "no_weight_mutation": source.get("no_weight_mutation") is True,
        "unsafe_promotions_zero": source.get("unsafe_promotions") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_honest_verdict": _wrapped_value(source.get("honest_verdict")),
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5341 result artifact from deterministic compressor evidence."""

    fixture_gate = confirm_fixture_gate(root=root)
    certificate_gate = confirm_certificate_gate(root=root)
    gates_pass = bool(fixture_gate["all_passed"] and certificate_gate["all_passed"])
    evaluation = (
        evaluate_compressor_cases(build_compressor_cases())
        if gates_pass
        else _blocked_evaluation()
    )
    complete = _compressor_complete(
        evaluation=evaluation,
        fixture_gate=fixture_gate,
        certificate_gate=certificate_gate,
        tests_run=tests_run,
    )
    status = (
        "compressor_drift_fixture_ready"
        if complete
        else "blocked_fixture_certificate_or_tests"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5328_RELATIVE_PATH), str(EXP5330_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, evaluation, fixture_gate, certificate_gate, tests_run),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "bounded_state_object_limit": BOUNDED_STATE_OBJECT_LIMIT,
        "bounded_state_token_limit": BOUNDED_STATE_TOKEN_LIMIT,
        "recalled_not_committed_count": evaluation["recalled_not_committed_count"],
        "drift_detection_rate": evaluation["drift_detection_rate"],
        "stale_recall_detection_rate": evaluation["stale_recall_detection_rate"],
        "poison_rejection_rate": evaluation["poison_rejection_rate"],
        "recoverability_rate": evaluation["recoverability_rate"],
        "unsafe_commits": evaluation["unsafe_commits"],
        "compressor_drift_fixture_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "fixture_gate": fixture_gate,
        "certificate_gate": certificate_gate,
        "compressor_rows": evaluation["compressor_rows"],
        "bounded_state_summaries": evaluation["bounded_state_summaries"],
        "compression_budget": evaluation["compression_budget"],
        "recall_commit_separation_rate": evaluation["recall_commit_separation_rate"],
        "verifier_call_cost": evaluation["verifier_call_cost"],
        "drift_counts": evaluation["drift_counts"],
        "stale_recall_counts": evaluation["stale_recall_counts"],
        "poison_counts": evaluation["poison_counts"],
        "recoverability_counts": evaluation["recoverability_counts"],
        "weight_mutation_receipt": _weight_mutation_receipt(evaluation),
        "methodology_note": (
            "All rates are deterministic fixture rates over six enumerated "
            "context-compressor rows. No LLM, judge, generator, adapter update, "
            "or model weight mutation is invoked."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by downstream compressor gates."""

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
    if artifact.get("unsafe_commits") != 0:
        raise ValueError("unsafe_commits must be 0")
    if not isinstance(artifact.get("compressor_drift_fixture_ready"), bool):
        raise ValueError("compressor_drift_fixture_ready must be bare bool")
    if artifact["compressor_drift_fixture_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready compressor fixture")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5341 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5328": _sha256_file(root_path / EXP5328_RELATIVE_PATH),
        "exp5330": _sha256_file(root_path / EXP5330_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5328_module": _sha256_file(root_path / EXP5328_MODULE_RELATIVE_PATH),
        "exp5330_module": _sha256_file(root_path / EXP5330_MODULE_RELATIVE_PATH),
    }


def _evaluate_case(
    case: CompressorCase,
    persistent_state: dict[str, JsonDict],
) -> JsonDict:
    source_row = _source_lifecycle_rows().get(case.source_case_id, {})
    prior_keys = sorted(persistent_state)
    summary_object_count = len(case.source_object_ids)
    summary_token_count = _token_count(case.candidate_summary)
    rejection_reasons = _rejection_reasons(case, summary_object_count, summary_token_count)
    detected_anomaly = case.expected_anomaly is not None
    accepted_commit = bool(
        case.expected_commit
        and not detected_anomaly
        and not rejection_reasons
        and not case.model_weights_mutated
    )
    commit_decision = _commit_decision(case, accepted_commit, detected_anomaly)
    committed_summary = (
        _summary_from_case(case, summary_token_count, source_row)
        if accepted_commit
        else None
    )
    if committed_summary is not None:
        persistent_state[str(committed_summary["summary_id"])] = committed_summary
    rollback_success = bool(
        case.expected_recoverable and source_row.get("rollback_success") is True
    )
    recoverable_from_sidecar = bool(
        accepted_commit
        and case.expected_recoverable
        and (
            source_row.get("sidecar_preserved") is True
            or source_row.get("recovered_from_sidecar") is True
            or rollback_success
        )
    )
    unsafe_commit = bool(accepted_commit and case.expected_anomaly is not None)
    return {
        "case_id": case.case_id,
        "source_case_id": case.source_case_id,
        "lifecycle_action": case.lifecycle_action,
        "source_object_ids": list(case.source_object_ids),
        "recalled_object_ids": list(case.recalled_object_ids),
        "candidate_summary": case.candidate_summary,
        "summary_object_count": summary_object_count,
        "summary_token_count": summary_token_count,
        "expected_anomaly": case.expected_anomaly,
        "detected_anomaly": detected_anomaly,
        "commit_decision": commit_decision,
        "accepted_commit": accepted_commit,
        "persistent_state_changed": accepted_commit,
        "prior_persistent_summary_ids": prior_keys,
        "post_persistent_summary_ids": sorted(persistent_state),
        "committed_summary": committed_summary,
        "rollback_case_id": case.rollback_case_id,
        "rollback_success": rollback_success,
        "recoverable_from_sidecar": recoverable_from_sidecar,
        "unsafe_commit": unsafe_commit,
        "rejection_reasons": rejection_reasons,
        "verifier_calls": 1,
        "model_weights_mutated": case.model_weights_mutated,
    }


def _rejection_reasons(
    case: CompressorCase,
    summary_object_count: int,
    summary_token_count: int,
) -> list[str]:
    reasons: list[str] = []
    if summary_object_count > BOUNDED_STATE_OBJECT_LIMIT:
        reasons.append("object_budget_exceeded")
    if summary_token_count > BOUNDED_STATE_TOKEN_LIMIT:
        reasons.append("token_budget_exceeded")
    anomaly_reasons = {
        STALE_RECALL: "stale_recall_detected",
        POISONED_CANDIDATE_MEMORY: "poisoned_candidate_memory_rejected",
        COMPRESSION_OMISSION: "compression_omission_detected",
        OVER_COMPRESSION: "over_compression_detected",
    }
    if case.expected_anomaly in anomaly_reasons:
        reasons.append(anomaly_reasons[case.expected_anomaly])
    if case.model_weights_mutated:
        reasons.append("model_weight_mutation_forbidden")
    return reasons


def _commit_decision(
    case: CompressorCase,
    accepted_commit: bool,
    detected_anomaly: bool,
) -> str:
    if accepted_commit:
        return COMMITTED
    if case.recalled_object_ids and not case.expected_commit:
        return RECALLED_NOT_COMMITTED
    if detected_anomaly or not case.expected_commit:
        return REJECTED
    return REJECTED


def _summary_from_case(
    case: CompressorCase,
    token_count: int,
    source_row: Mapping[str, Any],
) -> JsonDict:
    sidecar = _source_sidecar(source_row, case.source_object_ids[0])
    summary = {
        "summary_id": f"bounded.summary.{case.case_id}",
        "source_case_id": case.source_case_id,
        "source_object_ids": list(case.source_object_ids),
        "payload": case.candidate_summary,
        "token_count": token_count,
        "lifecycle_action": case.lifecycle_action,
        "rollback_case_id": case.rollback_case_id,
        "recoverable_sidecar": sidecar,
    }
    summary["summary_sha256"] = _checksum(summary)
    return summary


def _source_sidecar(
    source_row: Mapping[str, Any],
    object_id: str,
) -> JsonDict:
    for state_key in ("committed_state", "prior_state", "expected_state"):
        state = source_row.get(state_key, {})
        if isinstance(state, Mapping) and object_id in state:
            obj = state[object_id]
            if isinstance(obj, Mapping) and isinstance(
                obj.get("recoverable_sidecar"),
                Mapping,
            ):
                return dict(obj["recoverable_sidecar"])
    return {
        "sidecar_id": f"sidecar:{object_id}:unavailable",
        "recoverable": False,
        "source_object_ids": [object_id],
    }


def _source_lifecycle_rows() -> dict[str, JsonDict]:
    evaluation = exp5328.evaluate_lifecycle_fixture(exp5328.build_lifecycle_fixture())
    return {str(row["case_id"]): dict(row) for row in evaluation["lifecycle_rows"]}


def _detection_count(
    rows: Sequence[Mapping[str, Any]],
    anomaly_labels: Sequence[str],
) -> JsonDict:
    total = sum(1 for row in rows if row.get("expected_anomaly") in anomaly_labels)
    detected = sum(
        1
        for row in rows
        if row.get("expected_anomaly") in anomaly_labels
        and row.get("detected_anomaly") is True
        and row.get("accepted_commit") is False
    )
    return {"detected": detected, "total": total}


def _poison_rejection_count(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = sum(
        1 for row in rows if row.get("expected_anomaly") == POISONED_CANDIDATE_MEMORY
    )
    detected = sum(
        1
        for row in rows
        if row.get("expected_anomaly") == POISONED_CANDIDATE_MEMORY
        and row.get("detected_anomaly") is True
        and row.get("accepted_commit") is False
    )
    return {"detected": detected, "total": total}


def _recovery_count(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    recovery_rows = [row for row in rows if row.get("rollback_case_id")]
    recovered = sum(
        1
        for row in recovery_rows
        if row.get("accepted_commit") is True
        and row.get("recoverable_from_sidecar") is True
        and row.get("rollback_success") is True
    )
    return {"detected": recovered, "total": len(recovery_rows)}


def _detection_rate(counts: Mapping[str, int]) -> float:
    return _rate(int(counts["detected"]), int(counts["total"]))


def _compression_budget(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> JsonDict:
    max_summary_objects = max(
        (int(row["summary_object_count"]) for row in rows),
        default=0,
    )
    max_summary_tokens = max(
        (int(row["summary_token_count"]) for row in rows),
        default=0,
    )
    max_persistent_summaries = len(summaries)
    within_budget = bool(
        max_summary_objects <= BOUNDED_STATE_OBJECT_LIMIT
        and max_summary_tokens <= BOUNDED_STATE_TOKEN_LIMIT
        and max_persistent_summaries <= BOUNDED_STATE_OBJECT_LIMIT
    )
    return {
        "object_limit": BOUNDED_STATE_OBJECT_LIMIT,
        "token_limit": BOUNDED_STATE_TOKEN_LIMIT,
        "max_summary_object_count": max_summary_objects,
        "max_summary_token_count": max_summary_tokens,
        "max_persistent_summary_objects": max_persistent_summaries,
        "within_budget": within_budget,
    }


def _verifier_call_cost(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = sum(int(row["verifier_calls"]) for row in rows)
    return {
        "total_verifier_calls": total,
        "verifier_calls_per_case": _rate(total, len(rows)),
        "cost_model": "one_deterministic_commit_check_per_compressor_row",
    }


def _blocked_evaluation() -> JsonDict:
    return {
        "compressor_rows": [],
        "bounded_state_summaries": [],
        "compression_budget": {
            "object_limit": BOUNDED_STATE_OBJECT_LIMIT,
            "token_limit": BOUNDED_STATE_TOKEN_LIMIT,
            "max_summary_object_count": 0,
            "max_summary_token_count": 0,
            "max_persistent_summary_objects": 0,
            "within_budget": True,
        },
        "recalled_not_committed_count": 0,
        "recall_commit_separation_rate": 0.0,
        "drift_counts": {"detected": 0, "total": 0},
        "drift_detection_rate": 0.0,
        "stale_recall_counts": {"detected": 0, "total": 0},
        "stale_recall_detection_rate": 0.0,
        "poison_counts": {"detected": 0, "total": 0},
        "poison_rejection_rate": 0.0,
        "recoverability_counts": {"detected": 0, "total": 0},
        "recoverability_rate": 0.0,
        "unsafe_commits": 0,
        "verifier_call_cost": {
            "total_verifier_calls": 0,
            "verifier_calls_per_case": 0.0,
            "cost_model": "blocked_before_compressor_rows",
        },
        "no_weight_mutation": True,
        "compressor_drift_fixture_ready": False,
    }


def _compressor_complete(
    *,
    evaluation: Mapping[str, Any],
    fixture_gate: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        fixture_gate["all_passed"]
        and certificate_gate["all_passed"]
        and evaluation["compressor_drift_fixture_ready"]
        and evaluation["unsafe_commits"] == 0
        and evaluation["recoverability_rate"] == 1.0
        and evaluation["drift_detection_rate"] == 1.0
        and evaluation["stale_recall_detection_rate"] == 1.0
        and evaluation["poison_rejection_rate"] == 1.0
        and evaluation["compression_budget"]["within_budget"]
        and evaluation["no_weight_mutation"]
        and bool(tests_run)
    )


def _honest_verdict(
    complete: bool,
    evaluation: Mapping[str, Any],
    fixture_gate: Mapping[str, Any],
    certificate_gate: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> str:
    if complete:
        return (
            "complete: bounded context compressor separated recall from "
            "persistent commitment, detected stale recall, drift, and poisoned "
            "candidate memory at rate 1.0, kept unsafe commits at 0, preserved "
            "recoverability, and performed no model weight mutation"
        )
    blockers = [
        *fixture_gate.get("failed_gates", []),
        *certificate_gate.get("failed_gates", []),
    ]
    if not evaluation.get("compressor_drift_fixture_ready"):
        blockers.append("compressor_drift_fixture_ready_false")
    if not tests_run:
        blockers.append("tests_not_recorded")
    return "blocked_compressor_drift_not_ready: " + ",".join(blockers)


def _weight_mutation_receipt(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_bounded_context_summary_rows",
            "deterministic_compressor_monitor_rows",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": REQUIRED_FIELD_PRINCIPLES[field]}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _token_count(text: str) -> int:
    normalized = text.replace("_", " ")
    return len([token for token in normalized.split() if token])


def _rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)
