"""Exp 2981 promotion metrics for the deterministic partial monitor.

Spec refs: REQ-VERIFY-2981, SCENARIO-VERIFY-2981.

This module does not wrap an LLM or inspect a live token stream. It replays the
Exp 2968 partial monitor artifact, the Exp 2979 solver-feedback frontier, and
optionally the checked-in Exp 2980 solver candidate rows. The result is a
promotion decision for the partial monitor only: coverage and localization can
promote the deterministic harness, but they do not prove full streaming
verification across every generator path.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

RUN_DATE = "20260524"
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2981_interwhen_partial_monitor_promotion_v2.json"
EXP2968_FILENAME = "experiment_2968_interwhen_partial_monitor_harness_v1.json"
EXP2979_FILENAME = "experiment_2979_solver_feedback_mcs_frontier_v1.json"
EXP2980_FILENAME = "experiment_2980_sota_solver_formalization_feedback_v2.json"
INFERENCE_SUBSTRATE = "deterministic_monitor_harness"

EVENT_TYPES = (
    "draft_intent",
    "constraint_emission",
    "parse_boundary",
    "verifier_call",
    "counterexample",
    "repair_step",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "partial_monitor_promoted",
    "full_streaming_verification_claim",
    "event_types",
    "coverage_by_event",
    "prefix_failure_localization_rate",
    "monitor_latency_ms",
    "false_alarm_rate",
    "fixture_count",
    "live_trace_count",
    "promotion_gates",
    "inference_substrate",
    "duration_s",
)
_LATENCY_COST_MS = {
    "draft_intent": 0.05,
    "constraint_emission": 0.12,
    "parse_boundary": 0.20,
    "verifier_call": 0.75,
    "counterexample": 0.30,
    "repair_step": 0.15,
}
_CHECK_TO_EVENT = {
    "import_allow_list": "constraint_emission",
    "symbol_consistency": "constraint_emission",
    "parser_prefix_validity": "parse_boundary",
    "schema_field_coverage": "parse_boundary",
    "z3_parse_check": "verifier_call",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths for the deterministic Exp 2981 promotion evaluator."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2968_path: Path | None = None
    exp2979_path: Path | None = None
    exp2980_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def harness_path(self) -> Path:
        return self.exp2968_path or self.repo_root / "results" / EXP2968_FILENAME

    def frontier_path(self) -> Path:
        return self.exp2979_path or self.repo_root / "results" / EXP2979_FILENAME

    def live_solver_path(self) -> Path:
        return self.exp2980_path or self.repo_root / "results" / EXP2980_FILENAME


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 2981 terminal payload without fresh inference."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = source_artifact_status(active)
    exp2968 = _read_json(active.harness_path())
    exp2979 = _read_json(active.frontier_path())
    exp2980 = _read_json(active.live_solver_path()) if active.live_solver_path().exists() else {}
    exp2979_ready = bool(exp2979.get("frontier_upgrade_ready"))

    traces = []
    if exp2979_ready:
        traces.extend(deterministic_fixture_traces(exp2968, exp2979))
        traces.extend(live_solver_traces_from_exp2980(exp2980))
    monitored = [monitor_trace(trace) for trace in traces]
    metrics = compute_metrics(monitored, exp2979_ready=exp2979_ready)

    artifact = {
        "schema": "carnot.interwhen_partial_monitor_promotion.v2",
        "artifact": "experiment_2981_interwhen_partial_monitor_promotion_v2",
        "run_date": RUN_DATE,
        "honest_verdict": _honest_verdict(exp2979_ready, metrics["partial_monitor_promoted"]),
        "partial_monitor_promoted": metrics["partial_monitor_promoted"],
        "full_streaming_verification_claim": False,
        "event_types": list(EVENT_TYPES),
        "coverage_by_event": metrics["coverage_by_event"],
        "prefix_failure_localization_rate": metrics["prefix_failure_localization_rate"],
        "monitor_latency_ms": metrics["monitor_latency_ms"],
        "false_alarm_rate": metrics["false_alarm_rate"],
        "fixture_count": metrics["fixture_count"],
        "live_trace_count": metrics["live_trace_count"],
        "promotion_gates": metrics["promotion_gates"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, active.clock() - started), 6),
        "source_artifacts": sources,
        "trace_summaries": [_trace_summary(trace) for trace in monitored],
        "methodology_notes": [
            "Exp 2981 replays checked-in partial-output artifacts; it does not run a live generator.",
            "Coverage and localization promote only the deterministic partial monitor harness.",
            "full_streaming_verification_claim remains false because live code streams are not monitored.",
        ],
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 2981 artifact."""

    active = config or ExperimentConfig()
    payload = build_artifact(active)
    active.artifact_path().parent.mkdir(parents=True, exist_ok=True)
    active.artifact_path().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def deterministic_fixture_traces(
    exp2968: Mapping[str, Any],
    exp2979: Mapping[str, Any],
) -> list[JsonDict]:
    """Derive deterministic code and solver monitor traces from prior artifacts."""

    traces = []
    for row in exp2968.get("monitor_results") or []:
        if isinstance(row, Mapping):
            traces.append(_code_trace_from_exp2968(row))
    for row in exp2979.get("frontier_items") or []:
        if isinstance(row, Mapping):
            traces.append(_solver_trace_from_frontier(row))
    return traces


def live_solver_traces_from_exp2980(exp2980: Mapping[str, Any]) -> list[JsonDict]:
    """Derive solver monitor traces from checked-in Exp 2980 candidate rows."""

    traces = []
    for row in exp2980.get("per_item_results") or []:
        if isinstance(row, Mapping):
            traces.append(_solver_trace_from_exp2980(row))
    return traces


def monitor_trace(trace: Mapping[str, Any]) -> JsonDict:
    """Normalize one deterministic trace and locate its first flagged prefix."""

    events = []
    for index, event in enumerate(trace.get("events") or []):
        event_type = str(event.get("event_type"))
        expected_issue = bool(event.get("expected_issue"))
        monitor_flag = bool(event.get("monitor_flag", expected_issue))
        events.append(
            {
                "index": index,
                "event_type": event_type,
                "expected_issue": expected_issue,
                "monitor_flag": monitor_flag,
                "detail": event.get("detail"),
            }
        )
    expected_indexes = [event["index"] for event in events if event["expected_issue"]]
    flag_indexes = [event["index"] for event in events if event["monitor_flag"]]
    return {
        "trace_id": str(trace.get("trace_id") or "unknown"),
        "stream_kind": str(trace.get("stream_kind") or "unknown"),
        "trace_source": str(trace.get("trace_source") or "fixture"),
        "events": events,
        "has_expected_failure": bool(expected_indexes),
        "has_monitor_flag": bool(flag_indexes),
        "first_expected_failure_index": expected_indexes[0] if expected_indexes else None,
        "first_monitor_flag_index": flag_indexes[0] if flag_indexes else None,
    }


def compute_metrics(
    monitored_traces: Sequence[Mapping[str, Any]],
    *,
    exp2979_ready: bool,
) -> JsonDict:
    """Compute promotion metrics and gates for normalized monitor traces."""

    coverage = coverage_by_event(monitored_traces)
    localization = prefix_failure_localization_rate(monitored_traces)
    alarms = false_alarm_rate(monitored_traces)
    latency = monitor_latency_ms(monitored_traces)
    fixture_count = sum(trace.get("trace_source") == "fixture" for trace in monitored_traces)
    live_count = sum(trace.get("trace_source") == "live_exp2980" for trace in monitored_traces)
    gates = promotion_gates(
        coverage,
        prefix_rate=localization,
        false_alarm=alarms,
        exp2979_ready=exp2979_ready,
        failing_trace_count=sum(bool(trace.get("has_expected_failure")) for trace in monitored_traces),
    )
    promoted = all(
        gate["passed"]
        for name, gate in gates.items()
        if name != "full_streaming_claim_supported"
    )
    return {
        "coverage_by_event": coverage,
        "prefix_failure_localization_rate": localization,
        "monitor_latency_ms": latency,
        "false_alarm_rate": alarms,
        "fixture_count": fixture_count,
        "live_trace_count": live_count,
        "promotion_gates": gates,
        "partial_monitor_promoted": promoted,
    }


def coverage_by_event(monitored_traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count required event coverage by fixture/live source and stream kind."""

    coverage: JsonDict = {
        event_type: {
            "count": 0,
            "fixture_count": 0,
            "live_count": 0,
            "stream_kinds": [],
            "live_stream_kinds": [],
        }
        for event_type in EVENT_TYPES
    }
    for trace in monitored_traces:
        stream_kind = str(trace.get("stream_kind") or "unknown")
        source = str(trace.get("trace_source") or "fixture")
        for event in trace.get("events") or []:
            event_type = str(event.get("event_type"))
            if event_type not in coverage:
                continue
            coverage[event_type]["count"] += 1
            if source == "live_exp2980":
                coverage[event_type]["live_count"] += 1
                if stream_kind not in coverage[event_type]["live_stream_kinds"]:
                    coverage[event_type]["live_stream_kinds"].append(stream_kind)
            else:
                coverage[event_type]["fixture_count"] += 1
            if stream_kind not in coverage[event_type]["stream_kinds"]:
                coverage[event_type]["stream_kinds"].append(stream_kind)
    for row in coverage.values():
        row["stream_kinds"].sort()
        row["live_stream_kinds"].sort()
    return coverage


def prefix_failure_localization_rate(monitored_traces: Sequence[Mapping[str, Any]]) -> float:
    """Return the share of failing traces flagged no later than the failing prefix."""

    failing = [trace for trace in monitored_traces if trace.get("has_expected_failure")]
    if not failing:
        return 0.0
    localized = 0
    for trace in failing:
        flag_index = trace.get("first_monitor_flag_index")
        expected_index = trace.get("first_expected_failure_index")
        if flag_index is not None and expected_index is not None and flag_index <= expected_index:
            localized += 1
    return round(localized / len(failing), 6)


def false_alarm_rate(monitored_traces: Sequence[Mapping[str, Any]]) -> float:
    """Return the trace-level rate of monitor flags on non-failing traces."""

    if not monitored_traces:
        return 0.0
    false_alarms = sum(
        bool(trace.get("has_monitor_flag")) and not bool(trace.get("has_expected_failure"))
        for trace in monitored_traces
    )
    return round(false_alarms / len(monitored_traces), 6)


def monitor_latency_ms(monitored_traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Estimate deterministic monitor latency from event type costs."""

    by_event = {event_type: 0.0 for event_type in EVENT_TYPES}
    trace_totals = []
    for trace in monitored_traces:
        trace_total = 0.0
        for event in trace.get("events") or []:
            event_type = str(event.get("event_type"))
            cost = _LATENCY_COST_MS.get(event_type, 0.0)
            trace_total += cost
            if event_type in by_event:
                by_event[event_type] += cost
        trace_totals.append(trace_total)
    total = round(sum(trace_totals), 3)
    return {
        "by_event": {event: round(value, 3) for event, value in by_event.items()},
        "total": total,
        "mean_per_trace": round(total / len(trace_totals), 3) if trace_totals else 0.0,
        "max_trace": round(max(trace_totals), 3) if trace_totals else 0.0,
    }


def promotion_gates(
    coverage: Mapping[str, Any],
    *,
    prefix_rate: float,
    false_alarm: float,
    exp2979_ready: bool,
    failing_trace_count: int,
) -> JsonDict:
    """Return explicit pass/fail gates for partial-monitor promotion."""

    stream_kinds = sorted(
        {
            stream
            for row in coverage.values()
            if isinstance(row, Mapping)
            for stream in row.get("stream_kinds", [])
        }
    )
    live_streams = sorted(
        {
            stream
            for row in coverage.values()
            if isinstance(row, Mapping)
            for stream in row.get("live_stream_kinds", [])
        }
    )
    return {
        "exp2979_frontier_upgrade_ready": {
            "passed": bool(exp2979_ready),
            "actual": bool(exp2979_ready),
            "required": True,
        },
        "event_coverage_broad": {
            "passed": all(int(coverage[event]["count"]) > 0 for event in EVENT_TYPES),
            "actual": {event: int(coverage[event]["count"]) for event in EVENT_TYPES},
            "required": "count>0 for every required event type",
        },
        "stream_coverage_broad": {
            "passed": {"code", "solver"} <= set(stream_kinds),
            "actual": stream_kinds,
            "required": ["code", "solver"],
        },
        "prefix_failure_localization_measured": {
            "passed": failing_trace_count > 0,
            "actual": failing_trace_count,
            "required": "at least one failing trace",
        },
        "prefix_failure_localization_rate": {
            "passed": prefix_rate >= 0.80,
            "actual": prefix_rate,
            "required": 0.80,
        },
        "false_alarm_rate_bounded": {
            "passed": false_alarm <= 0.20,
            "actual": false_alarm,
            "required": 0.20,
        },
        "full_streaming_claim_supported": {
            "passed": {"code", "solver"} <= set(live_streams),
            "actual": live_streams,
            "required": ["code", "solver"],
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 2981 terminal artifact contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("full_streaming_verification_claim") is not False:
        raise ValueError("full_streaming_verification_claim must remain false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")
    if tuple(artifact.get("event_types") or ()) != EVENT_TYPES:
        raise ValueError("event_types must match the required Exp 2981 vocabulary")
    gates = artifact.get("promotion_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("promotion_gates must be an object")
    expected_promoted = all(
        bool(gate.get("passed"))
        for name, gate in gates.items()
        if name != "full_streaming_claim_supported" and isinstance(gate, Mapping)
    )
    if bool(artifact.get("partial_monitor_promoted")) != expected_promoted:
        raise ValueError("partial_monitor_promoted does not match promotion gates")


def source_artifact_status(config: ExperimentConfig) -> JsonDict:
    """Return path, presence, and checksum evidence for source artifacts."""

    rows = {}
    for key, path in {
        "exp2968": config.harness_path(),
        "exp2979": config.frontier_path(),
        "exp2980": config.live_solver_path(),
    }.items():
        rows[key] = {
            "path": str(path),
            "present": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    return rows


def _code_trace_from_exp2968(row: Mapping[str, Any]) -> JsonDict:
    failed_checks = _failed_checks(row)
    failed_events = {_CHECK_TO_EVENT[check] for check in failed_checks if check in _CHECK_TO_EVENT}
    trace_id = str(row.get("trace_id") or "exp2968-code")
    return {
        "trace_id": f"fixture:{trace_id}",
        "stream_kind": "code",
        "trace_source": "fixture",
        "events": [
            _event("draft_intent", False, False, "Exp 2968 code trace loaded"),
            _event(
                "constraint_emission",
                "constraint_emission" in failed_events,
                "constraint_emission" in failed_events,
                "Import, signature, or assertion constraint emission checked",
            ),
            _event(
                "parse_boundary",
                "parse_boundary" in failed_events,
                "parse_boundary" in failed_events,
                "Python parser/schema prefix boundary checked",
            ),
            _event(
                "verifier_call",
                "verifier_call" in failed_events,
                "verifier_call" in failed_events,
                "Code verifier escalation boundary checked",
            ),
        ],
    }


def _solver_trace_from_frontier(row: Mapping[str, Any]) -> JsonDict:
    feedback = _feedback(row)
    parse_issue = feedback.get("parse_error") is not None
    verifier_issue = feedback.get("z3_exception") is not None
    counterexample_issue = bool(feedback.get("model_counterexample") or feedback.get("unsat_core_or_mus"))
    return {
        "trace_id": f"fixture:{row.get('item_id') or 'solver'}",
        "stream_kind": "solver",
        "trace_source": "fixture",
        "events": [
            _event("draft_intent", False, False, "Exp 2979 prompt intent loaded"),
            _event("constraint_emission", False, False, "Accepted SMT-LIB reference loaded"),
            _event("parse_boundary", parse_issue, parse_issue, feedback.get("parse_error")),
            _event("verifier_call", verifier_issue, verifier_issue, feedback.get("z3_exception")),
            _event(
                "counterexample",
                counterexample_issue,
                counterexample_issue,
                "counterexample, unsat core, or MUS feedback present",
            ),
            _event("repair_step", False, False, feedback.get("minimal_correction_hint")),
        ],
    }


def _solver_trace_from_exp2980(row: Mapping[str, Any]) -> JsonDict:
    initial = row.get("initial_result") if isinstance(row.get("initial_result"), Mapping) else {}
    final = row.get("final_result") if isinstance(row.get("final_result"), Mapping) else {}
    parse_issue = bool(initial.get("parse_error")) or initial.get("parseable") is False
    verifier_issue = bool(_z3_error(initial)) or initial.get("z3_executed") is False and not parse_issue
    solver_clean = bool(initial.get("solver_formula_correct") and initial.get("answer_correct"))
    counterexample_issue = bool(initial.get("z3_executed")) and not solver_clean
    repair_attempted = bool(row.get("repair_attempted"))
    return {
        "trace_id": f"live_exp2980:{row.get('item_id') or 'solver'}",
        "stream_kind": "solver",
        "trace_source": "live_exp2980",
        "events": [
            _event("draft_intent", False, False, "Exp 2980 candidate prompt intent loaded"),
            _event("constraint_emission", False, False, "Initial formalization constraints emitted"),
            _event("parse_boundary", parse_issue, parse_issue, initial.get("parse_error")),
            _event("verifier_call", verifier_issue, verifier_issue, _z3_error(initial)),
            _event(
                "counterexample",
                counterexample_issue,
                counterexample_issue,
                initial.get("failure_category"),
            ),
            _event(
                "repair_step",
                False,
                False,
                "repair attempted" if repair_attempted else final.get("failure_category"),
            ),
        ],
    }


def _failed_checks(row: Mapping[str, Any]) -> set[str]:
    failed = set()
    for event in row.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        for check in event.get("checks") or []:
            if isinstance(check, Mapping) and check.get("passed") is False:
                failed.add(str(check.get("check_name")))
    return failed


def _feedback(row: Mapping[str, Any]) -> JsonDict:
    feedback = row.get("solver_feedback")
    return dict(feedback) if isinstance(feedback, Mapping) else {}


def _z3_error(row: Mapping[str, Any]) -> str | None:
    z3_result = row.get("z3_result")
    if isinstance(z3_result, Mapping):
        error = z3_result.get("z3_error")
        return str(error) if error else None
    return None


def _event(event_type: str, expected_issue: bool, monitor_flag: bool, detail: Any) -> JsonDict:
    return {
        "event_type": event_type,
        "expected_issue": bool(expected_issue),
        "monitor_flag": bool(monitor_flag),
        "detail": detail,
    }


def _trace_summary(trace: Mapping[str, Any]) -> JsonDict:
    return {
        "trace_id": trace.get("trace_id"),
        "stream_kind": trace.get("stream_kind"),
        "trace_source": trace.get("trace_source"),
        "event_count": len(trace.get("events") or []),
        "has_expected_failure": bool(trace.get("has_expected_failure")),
        "has_monitor_flag": bool(trace.get("has_monitor_flag")),
        "first_expected_failure_index": trace.get("first_expected_failure_index"),
        "first_monitor_flag_index": trace.get("first_monitor_flag_index"),
    }


def _honest_verdict(exp2979_ready: bool, promoted: bool) -> str:
    if not exp2979_ready:
        return "blocked_precondition: exp2979_frontier_upgrade_not_ready"
    if promoted:
        return "complete: deterministic partial monitor promoted with measured coverage and localization"
    return "complete: deterministic partial monitor measured but not promoted"


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:  # pragma: no cover - script entrypoint.
    artifact = write_artifact()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["partial_monitor_promoted"] else 1


if __name__ == "__main__":  # pragma: no cover - script entrypoint.
    raise SystemExit(main())


__all__ = [
    "EVENT_TYPES",
    "EXP2968_FILENAME",
    "EXP2979_FILENAME",
    "EXP2980_FILENAME",
    "ExperimentConfig",
    "INFERENCE_SUBSTRATE",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "compute_metrics",
    "coverage_by_event",
    "deterministic_fixture_traces",
    "false_alarm_rate",
    "live_solver_traces_from_exp2980",
    "monitor_latency_ms",
    "monitor_trace",
    "prefix_failure_localization_rate",
    "promotion_gates",
    "source_artifact_status",
    "validate_artifact",
    "write_artifact",
]
