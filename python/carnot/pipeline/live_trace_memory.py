"""Live trace ingestion for Exp 222 memory, repair, and policy analysis.

Spec: REQ-VERIFY-030, REQ-VERIFY-031, REQ-VERIFY-032,
SCENARIO-VERIFY-030, SCENARIO-VERIFY-031, SCENARIO-VERIFY-032
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.pipeline.memory import ConstraintMemory

RUN_DATE = "20260412"
RESULT_OUTPUT = Path("results/experiment_222_results.json")
MEMORY_OUTPUT = Path("results/constraint_memory_live_222.json")
POLICY_PATH = Path("results/monitorability_policy_213.json")
SOURCE_ARTIFACTS = (
    Path("results/experiment_219_results.json"),
    Path("results/experiment_220_results.json"),
    Path("results/experiment_221_results.json"),
)
CONFIDENCE_THRESHOLD = 0.8


@dataclass(frozen=True)
class TraceEvent:
    """Normalized live trace event used for replay and memory gating."""

    source_experiment: int
    benchmark: str
    domain: str
    model_name: str
    case_id: str
    response_mode: str
    verifier_path: str
    actual_error: bool
    detected: bool
    error_types: tuple[str, ...]
    descriptions: tuple[str, ...]
    confidence: float
    ambiguous: bool = False

    @property
    def outcome(self) -> str:
        if self.actual_error and self.detected:
            return "true_positive"
        if (not self.actual_error) and self.detected:
            return "false_positive"
        if self.actual_error and (not self.detected):
            return "false_negative"
        return "true_negative"

    @property
    def eligible_for_memory(self) -> bool:
        return (
            self.outcome == "true_positive"
            and self.confidence >= CONFIDENCE_THRESHOLD
            and not self.ambiguous
            and bool(self.error_types)
        )

    @property
    def exclusion_reason(self) -> str | None:
        if self.eligible_for_memory:
            return None
        if self.ambiguous:
            return "ambiguous_trace"
        if self.outcome == "false_positive":
            return "false_positive"
        if self.outcome == "false_negative":
            return "false_negative"
        if self.outcome == "true_positive" and not self.error_types:
            return "missing_error_taxonomy"
        if self.outcome == "true_positive":
            return "low_confidence"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment": self.source_experiment,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "model_name": self.model_name,
            "case_id": self.case_id,
            "response_mode": self.response_mode,
            "verifier_path": self.verifier_path,
            "outcome": self.outcome,
            "actual_error": self.actual_error,
            "detected": self.detected,
            "error_types": list(self.error_types),
            "descriptions": list(self.descriptions),
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
        }


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_source_artifacts(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    exp219 = load_json(repo_root / SOURCE_ARTIFACTS[0])
    exp220 = load_json(repo_root / SOURCE_ARTIFACTS[1])
    exp221 = load_json(repo_root / SOURCE_ARTIFACTS[2])
    return exp219, exp220, exp221


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _dedupe_preserve(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _combine_error_type(taxonomy_hint: str | None, violation_type: str | None) -> str:
    if taxonomy_hint:
        return f"{taxonomy_hint}:{violation_type or 'unknown_violation'}"
    return str(violation_type or "unknown_violation")


def _extract_exp219_events(payload: dict[str, Any]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name = str(run.get("model_name") or "")
        for case in run.get("cases", []):
            verification = _as_dict(case.get("verification"))
            semantic_grounding = _as_dict(verification.get("semantic_grounding"))
            raw_error_types: list[str] = []
            raw_descriptions: list[str] = []
            for violation in semantic_grounding.get("violations", []):
                if not isinstance(violation, dict):
                    continue
                metadata = _as_dict(violation.get("metadata"))
                raw_error_types.append(
                    _combine_error_type(
                        str(metadata.get("taxonomy_hint") or "") or None,
                        str(
                            violation.get("violation_type") or metadata.get("violation_type") or ""
                        ),
                    )
                )
                raw_descriptions.append(
                    str(violation.get("description") or "semantic grounding violation")
                )
            if not raw_error_types:
                for violation in verification.get("violations", []):
                    if not isinstance(violation, dict):
                        continue
                    metadata = _as_dict(violation.get("metadata"))
                    raw_error_types.append(
                        _combine_error_type(
                            str(metadata.get("taxonomy_hint") or "") or None,
                            str(
                                metadata.get("violation_type")
                                or violation.get("constraint_type")
                                or ""
                            ),
                        )
                    )
                    raw_descriptions.append(
                        str(
                            violation.get("description")
                            or violation.get("constraint_type")
                            or "verification violation"
                        )
                    )
            actual_error = not bool(case.get("correct"))
            detected = bool(case.get("flagged"))
            confidence = (
                0.95
                if actual_error and detected and raw_error_types
                else 0.25
                if detected and not actual_error
                else 0.4
                if actual_error
                else 0.6
            )
            events.append(
                TraceEvent(
                    source_experiment=219,
                    benchmark="gsm8k_semantic",
                    domain="live_gsm8k_semantic_failure",
                    model_name=model_name,
                    case_id=str(case.get("case_id") or ""),
                    response_mode=str(case.get("response_mode") or "unknown"),
                    verifier_path="semantic_grounding",
                    actual_error=actual_error,
                    detected=detected,
                    error_types=_dedupe_preserve(raw_error_types),
                    descriptions=_dedupe_preserve(raw_descriptions),
                    confidence=confidence,
                    ambiguous=bool(semantic_grounding.get("refinement_applied")),
                )
            )
    return events


def _classify_humaneval_text(text: str) -> str:
    lowered = text.lower()
    if "official_tests" in lowered or "assertionerror" in lowered:
        return "official_test_failure"
    if "syntax" in lowered or "unterminated" in lowered:
        return "syntax_error"
    if "indexerror" in lowered:
        return "index_error"
    if "parameter" in lowered and "annotated" in lowered:
        return "annotation_feedback"
    if "property" in lowered or "prompt_examples" in lowered or "prompt_intent" in lowered:
        return "property_violation"
    return "humaneval_failure"


def _extract_humaneval_errors(case: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    verifier = _as_dict(case.get("execution_plus_property"))
    raw_descriptions: list[str] = []
    for key in (
        "property_violations",
        "static_violations",
        "dynamic_violations",
        "constraint_feedback",
    ):
        values = verifier.get(key)
        if isinstance(values, list):
            raw_descriptions.extend(str(value) for value in values if value)
    if not raw_descriptions:
        if case.get("error_message"):
            raw_descriptions.append(str(case["error_message"]))
        elif case.get("error_type"):
            raw_descriptions.append(str(case["error_type"]))
    raw_error_types = [_classify_humaneval_text(description) for description in raw_descriptions]
    return _dedupe_preserve(raw_error_types), _dedupe_preserve(raw_descriptions)


def _extract_exp220_events(payload: dict[str, Any]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name = str(run.get("model_name") or "")
        for case in run.get("cases", []):
            verifier = _as_dict(case.get("execution_plus_property"))
            error_types, descriptions = _extract_humaneval_errors(case)
            actual_error = not bool(case.get("passed"))
            detected = bool(verifier.get("detected"))
            confidence = (
                0.99
                if actual_error and detected
                else 0.55
                if actual_error
                else 0.35
                if detected
                else 0.6
            )
            events.append(
                TraceEvent(
                    source_experiment=220,
                    benchmark="humaneval_property",
                    domain="code_typed_properties",
                    model_name=model_name,
                    case_id=str(case.get("case_id") or ""),
                    response_mode=str(case.get("response_mode") or "unknown"),
                    verifier_path="execution_plus_property",
                    actual_error=actual_error,
                    detected=detected,
                    error_types=error_types,
                    descriptions=descriptions,
                    confidence=confidence,
                )
            )
    return events


def _extract_constraint_ir_errors(case: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    evaluation = _as_dict(case.get("evaluation"))
    raw_error_types: list[str] = []
    raw_descriptions: list[str] = []
    for result in evaluation.get("constraint_results", []):
        if not isinstance(result, dict) or result.get("status") != "violated":
            continue
        family = str(result.get("family") or "")
        error_type = str(result.get("type") or "constraint_violation")
        raw_error_types.append(f"{family}:{error_type}" if family else error_type)
        raw_descriptions.append(f"{family}:{error_type}" if family else error_type)
    return _dedupe_preserve(raw_error_types), _dedupe_preserve(raw_descriptions)


def _extract_exp221_events(payload: dict[str, Any]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for run in payload.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name = str(run.get("model_name") or "")
        for case in run.get("cases", []):
            evaluation = _as_dict(case.get("evaluation"))
            judging = _as_dict(evaluation.get("judging_summary"))
            error_types, descriptions = _extract_constraint_ir_errors(case)
            actual_error = not bool(case.get("exact_satisfaction"))
            detected = bool(case.get("flagged"))
            ambiguous = (
                int(judging.get("model_assisted", 0)) > 0
                and int(judging.get("deterministic", 0)) == 0
            )
            confidence = (
                0.95
                if actual_error and detected and not ambiguous
                else 0.4
                if ambiguous
                else 0.25
                if detected and not actual_error
                else 0.5
            )
            events.append(
                TraceEvent(
                    source_experiment=221,
                    benchmark="constraint_ir",
                    domain=str(evaluation.get("task_slice") or "constraint_ir"),
                    model_name=model_name,
                    case_id=str(case.get("case_id") or ""),
                    response_mode=str(case.get("response_mode") or "unknown"),
                    verifier_path="constraint_ir_scoring",
                    actual_error=actual_error,
                    detected=detected,
                    error_types=error_types,
                    descriptions=descriptions,
                    confidence=confidence,
                    ambiguous=ambiguous,
                )
            )
    return events


def _memory_patterns_payload(memory: ConstraintMemory) -> dict[str, Any]:
    patterns: dict[str, Any] = {}
    for domain, domain_patterns in memory._patterns.items():
        patterns[domain] = {}
        for error_type, record in domain_patterns.items():
            patterns[domain][error_type] = {
                "frequency": record.frequency,
                "constraint_examples": list(record.constraint_examples),
                "auto_generated": record.auto_generated,
            }
    return patterns


def _replay_memory(
    events: list[TraceEvent],
) -> tuple[ConstraintMemory, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    memory = ConstraintMemory()
    accepted_events: list[dict[str, Any]] = []
    quarantined_events: list[dict[str, Any]] = []
    events_with_suggestions = 0
    helpful_retrieval_events = 0
    total_suggestions = 0
    matching_suggestions = 0

    for event in events:
        suggestions = memory.suggest_constraints("", event.domain)
        suggestion_error_types = [
            str(suggestion.metadata.get("error_type"))
            for suggestion in suggestions
            if isinstance(suggestion.metadata, dict) and suggestion.metadata.get("error_type")
        ]
        if suggestion_error_types:
            events_with_suggestions += 1
            total_suggestions += len(suggestion_error_types)
            matches = [
                error_type
                for error_type in suggestion_error_types
                if error_type in event.error_types
            ]
            if matches:
                helpful_retrieval_events += 1
                matching_suggestions += len(matches)

        record = event.to_dict()
        record["retrieved_error_types"] = suggestion_error_types
        record["matched_retrieval"] = any(
            error_type in event.error_types for error_type in suggestion_error_types
        )

        if event.eligible_for_memory:
            for index, error_type in enumerate(event.error_types):
                description = (
                    event.descriptions[index]
                    if index < len(event.descriptions)
                    else event.descriptions[0]
                )
                memory.record_pattern(event.domain, error_type, description)
            accepted_events.append(record)
            continue

        exclusion_reason = event.exclusion_reason
        if exclusion_reason is not None:
            record["exclusion_reason"] = exclusion_reason
            quarantined_events.append(record)

    reused_pattern_precision = (
        matching_suggestions / total_suggestions if total_suggestions else 0.0
    )
    replay = {
        "events_with_suggestions": events_with_suggestions,
        "helpful_retrieval_events": helpful_retrieval_events,
        "reused_pattern_suggestions": total_suggestions,
        "reused_pattern_precision": reused_pattern_precision,
    }
    return memory, accepted_events, quarantined_events, replay


def _sanitize_repair_prompt(benchmark: str, prompt: str) -> str:
    if benchmark == "humaneval_property" and "HumanEval test failure:" in prompt:
        return (
            "HumanEval test failure:\n"
            "  - {failure_message}\n\n"
            "Write ONLY the corrected function body. No markdown fences.\n"
            "Indent with 4 spaces."
        )
    if (
        benchmark == "constraint_ir"
        and "Your previous response did not satisfy the required contract." in prompt
    ):
        return (
            "Your previous response did not satisfy the required contract.\n"
            "Issues:\n"
            "- parseable={parseable}\n"
            "- answer_quality={answer_quality}\n"
            "- constraint_coverage={constraint_coverage}\n\n"
            "Answer again using the same response contract."
        )
    return prompt.strip()


def _collect_repair_snippets(
    exp220: dict[str, Any],
    exp221: dict[str, Any],
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for payload in (exp220, exp221):
        benchmark = str(payload.get("benchmark") or "")
        verify_only_lookup: dict[str, dict[str, Any]] = {}
        for run in payload.get("paired_runs", []):
            if run.get("mode") != "verify_only":
                continue
            for case in run.get("cases", []):
                if case.get("case_id"):
                    verify_only_lookup[str(case["case_id"])] = case
        for run in payload.get("paired_runs", []):
            if run.get("mode") != "verify_repair":
                continue
            model_name = str(run.get("model_name") or "")
            for case in run.get("cases", []):
                reference_case = verify_only_lookup.get(str(case.get("case_id") or ""), case)
                if benchmark == "humaneval_property":
                    error_types, _ = _extract_humaneval_errors(reference_case)
                    if not error_types:
                        history = (
                            case.get("history") if isinstance(case.get("history"), list) else []
                        )
                        error_types = _dedupe_preserve(
                            [
                                _classify_humaneval_text(str(item.get("repair_prompt") or ""))
                                for item in history
                                if isinstance(item, dict) and item.get("repair_prompt")
                            ]
                        )
                    domain = "code_typed_properties"
                else:
                    error_types, _ = _extract_constraint_ir_errors(reference_case)
                    evaluation = (
                        reference_case.get("evaluation")
                        if isinstance(reference_case.get("evaluation"), dict)
                        else {}
                    )
                    domain = str(evaluation.get("task_slice") or "constraint_ir")
                primary_error_type = error_types[0] if error_types else "repair_feedback"
                snippet_id = f"{benchmark}:{primary_error_type}"
                history = case.get("history") if isinstance(case.get("history"), list) else []
                for item in history:
                    if not isinstance(item, dict) or not item.get("repair_prompt"):
                        continue
                    entry = aggregates.setdefault(
                        snippet_id,
                        {
                            "snippet_id": snippet_id,
                            "benchmark": benchmark,
                            "domain": domain,
                            "model_names": set(),
                            "trigger_error_type": primary_error_type,
                            "template": _sanitize_repair_prompt(
                                benchmark,
                                str(item["repair_prompt"]),
                            ),
                            "support": 0,
                            "_successful_cases": set(),
                            "_failed_cases": set(),
                            "provenance": [],
                        },
                    )
                    entry["model_names"].add(model_name)
                    entry["support"] += 1
                    if case.get("repaired"):
                        entry["_successful_cases"].add(str(case.get("case_id") or ""))
                    else:
                        entry["_failed_cases"].add(str(case.get("case_id") or ""))
                    entry["provenance"].append(
                        {
                            "source_experiment": payload.get("experiment"),
                            "model_name": model_name,
                            "case_id": case.get("case_id"),
                            "iteration": item.get("iteration"),
                        }
                    )
    snippets: list[dict[str, Any]] = []
    for snippet_id in sorted(aggregates):
        entry = aggregates[snippet_id]
        snippets.append(
            {
                "snippet_id": entry["snippet_id"],
                "benchmark": entry["benchmark"],
                "domain": entry["domain"],
                "model_names": sorted(entry["model_names"]),
                "trigger_error_type": entry["trigger_error_type"],
                "template": entry["template"],
                "support": entry["support"],
                "successful_cases": len(entry["_successful_cases"]),
                "failed_cases": len(entry["_failed_cases"]),
                "provenance": entry["provenance"],
            }
        )
    return snippets


def _build_reliability_stats(events: list[TraceEvent]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        key = (event.model_name, event.benchmark, event.domain)
        bucket = buckets.setdefault(
            key,
            {
                "model_name": event.model_name,
                "benchmark": event.benchmark,
                "domain": event.domain,
                "n_cases": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
            },
        )
        bucket["n_cases"] += 1
        if event.outcome == "true_positive":
            bucket["true_positives"] += 1
        elif event.outcome == "false_positive":
            bucket["false_positives"] += 1
        elif event.outcome == "false_negative":
            bucket["false_negatives"] += 1
        else:
            bucket["true_negatives"] += 1
    stats: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        precision_denom = bucket["true_positives"] + bucket["false_positives"]
        recall_denom = bucket["true_positives"] + bucket["false_negatives"]
        bucket["precision"] = bucket["true_positives"] / precision_denom if precision_denom else 0.0
        bucket["recall"] = bucket["true_positives"] / recall_denom if recall_denom else 0.0
        stats.append(bucket)
    return stats


def _build_policy_updates(
    reliability: list[dict[str, Any]],
    base_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    per_task_slice = _as_dict(base_policy.get("per_task_slice"))
    updates: list[dict[str, Any]] = []
    for entry in reliability:
        recommended_mode = None
        domain_policy = per_task_slice.get(entry["domain"])
        if isinstance(domain_policy, dict):
            recommended_mode = domain_policy.get("recommended_mode")
        if entry["precision"] < 0.9 and entry["true_positives"] > 0:
            recommended_action = "guarded_memory_only"
        elif entry["benchmark"] == "humaneval_property":
            recommended_action = "repair_guidance_only"
        elif entry["benchmark"] == "constraint_ir":
            recommended_action = "reuse_contract_patch"
        else:
            recommended_action = "promote_memory_reuse"
        updates.append(
            {
                "update_id": (
                    f"{entry['benchmark']}:{entry['model_name'].lower()}:{entry['domain']}"
                ),
                "model_name": entry["model_name"],
                "benchmark": entry["benchmark"],
                "domain": entry["domain"],
                "recommended_mode": recommended_mode,
                "recommended_action": recommended_action,
                "evidence": {
                    "n_cases": entry["n_cases"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "true_positives": entry["true_positives"],
                    "false_positives": entry["false_positives"],
                    "false_negatives": entry["false_negatives"],
                },
            }
        )
    return updates


def build_live_trace_memory_bundle(
    *,
    exp219: dict[str, Any],
    exp220: dict[str, Any],
    exp221: dict[str, Any],
    base_policy: dict[str, Any],
) -> dict[str, Any]:
    """Build the in-memory Exp 222 result bundle without writing files."""

    events = [
        *_extract_exp219_events(exp219),
        *_extract_exp220_events(exp220),
        *_extract_exp221_events(exp221),
    ]
    memory, accepted_events, quarantined_events, replay = _replay_memory(events)
    repair_snippets = _collect_repair_snippets(exp220, exp221)
    reliability = _build_reliability_stats(events)
    policy_updates = _build_policy_updates(reliability, base_policy)

    patterns = _memory_patterns_payload(memory)
    memory_summary = memory.summary()
    total_patterns = sum(domain["total_patterns"] for domain in memory_summary.values())
    mature_patterns = sum(domain["mature_patterns"] for domain in memory_summary.values())

    memory_payload = {
        "version": 1,
        "experiment": 222,
        "run_date": RUN_DATE,
        "title": "Live constraint memory built from Exp 219-221",
        "result_source": str(RESULT_OUTPUT),
        "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
        "patterns": patterns,
        "accepted_events": accepted_events,
        "quarantined_events": quarantined_events,
        "repair_snippets": repair_snippets,
        "monitorability_policy_updates": policy_updates,
        "summary": {
            "accepted_trace_events": len(accepted_events),
            "quarantined_trace_events": len(quarantined_events),
            "total_patterns": total_patterns,
            "mature_patterns": mature_patterns,
            "patterns_by_domain": memory_summary,
        },
    }
    result_payload = {
        "experiment": 222,
        "run_date": RUN_DATE,
        "title": "Live trace memory and repair guidance",
        "metadata": {
            "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
            "policy_source": str(POLICY_PATH),
            "memory_output_path": str(MEMORY_OUTPUT),
            "output_path": str(RESULT_OUTPUT),
        },
        "summary": {
            "n_trace_events": len(events),
            "memory_growth": memory_payload["summary"],
            "retrieval_usefulness": replay,
        },
        "verifier_reliability": reliability,
        "repair_snippets": repair_snippets,
        "monitorability_policy_updates": policy_updates,
    }
    return {
        "events": [event.to_dict() for event in events],
        "memory_payload": memory_payload,
        "result_payload": result_payload,
    }


def run_experiment(
    repo_root: Path | None = None,
    result_path: Path | None = None,
    memory_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and write Exp 222 artifacts for the current repository checkout."""

    resolved_repo = (repo_root or get_repo_root()).resolve()
    resolved_result_path = resolved_repo / (result_path or RESULT_OUTPUT)
    resolved_memory_path = resolved_repo / (memory_path or MEMORY_OUTPUT)
    exp219, exp220, exp221 = load_source_artifacts(resolved_repo)
    base_policy = load_json(resolved_repo / POLICY_PATH)
    bundle = build_live_trace_memory_bundle(
        exp219=exp219,
        exp220=exp220,
        exp221=exp221,
        base_policy=base_policy,
    )
    bundle["memory_payload"]["result_source"] = _relative_path(resolved_result_path, resolved_repo)
    bundle["result_payload"]["metadata"]["memory_output_path"] = _relative_path(
        resolved_memory_path,
        resolved_repo,
    )
    bundle["result_payload"]["metadata"]["output_path"] = _relative_path(
        resolved_result_path,
        resolved_repo,
    )
    write_json(resolved_result_path, bundle["result_payload"])
    write_json(resolved_memory_path, bundle["memory_payload"])
    return bundle["result_payload"], bundle["memory_payload"]


__all__ = [
    "CONFIDENCE_THRESHOLD",
    "MEMORY_OUTPUT",
    "POLICY_PATH",
    "RESULT_OUTPUT",
    "RUN_DATE",
    "SOURCE_ARTIFACTS",
    "TraceEvent",
    "build_live_trace_memory_bundle",
    "get_repo_root",
    "load_json",
    "load_source_artifacts",
    "run_experiment",
    "write_json",
]
