"""Exp 2891 tiny CCTU-style executable constraint validator pilot.

Spec: REQ-VERIFY-2891, SCENARIO-VERIFY-2891.

The pilot is deliberately narrower than the public CCTU benchmark. It does not
call a model and it does not report benchmark performance. Instead, it creates
small local transcript perturbations from the existing Exp 1486 executable
cases and checks whether Carnot can localize failures into CCTU-shaped
constraint categories before any broader claim is attempted.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2891_cctu_executable_constraint_validator_pilot_v1.json"
CCTU_SOURCE_MODULE = "python/carnot/eval/cctu_executable_constraint_microbenchmark.py"
SOURCE_MODULES = (
    CCTU_SOURCE_MODULE,
    "python/carnot/eval/cctu_micro_benchmark_adapter.py",
    "python/carnot/eval/cctu_trigger_certificate_export.py",
)
STEP_IDS = (
    "parse_json",
    "resource",
    "toolset",
    "behavior",
    "response",
    "response_verifier",
)
REQUIRED_CATEGORIES = ("behavior", "resource", "response", "response_verifier", "toolset")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "cctu_validator_ready",
    "source_modules",
    "n_cases",
    "constraint_categories",
    "category_coverage",
    "executable_validation_used",
    "validation_rows",
    "unsupported_categories",
    "live_llm_called",
    "headline_metric_claim_made",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict; complete means the local validator shape is exercised, not benchmark performance.",
    "cctu_validator_ready": "True only when required CCTU-shaped categories have deterministic rows and executable validation ran.",
    "source_modules": "Local modules inspected or reused for the pilot; no remote benchmark loader is implied.",
    "n_cases": "Count of local deterministic transcript probes, not public CCTU task count.",
    "constraint_categories": "CCTU-style categories covered by local probes.",
    "category_coverage": "Per-category validation pass counts over injected local probes.",
    "executable_validation_used": "True only when rows replay through the Exp 1486 local executable validator.",
    "validation_rows": "Step-level pass/fail rows with violation localization and stable checksums.",
    "unsupported_categories": "Categories intentionally out of scope for this tiny pilot.",
    "live_llm_called": "Must remain false; this pilot validates benchmark shape without generation.",
    "headline_metric_claim_made": "Must remain false; no CCTU benchmark score or model-performance headline is reported.",
    "tests_run": "Commands used to validate the module and artifact.",
    "duration_s": "Measured wall-clock runtime; no padding.",
}
UNSUPPORTED_CATEGORIES = {
    "multi_turn_state": {
        "supported": False,
        "reason": "The pilot validates one JSON transcript per case and does not model multi-turn state carryover.",
    },
    "external_api_side_effects": {
        "supported": False,
        "reason": "Only deterministic in-process tools are executed; network and external side effects are excluded.",
    },
    "long_context_budget": {
        "supported": False,
        "reason": "The local cases are tiny and do not test the long-prompt budgets described by the full CCTU benchmark.",
    },
}


@dataclass(frozen=True)
class PilotCase:
    """One local transcript probe for a CCTU-shaped constraint category.

    Each probe starts from an Exp 1486 gold transcript, mutates exactly the
    field that should exercise one category, and is then replayed through the
    executable validator. Keeping the mutation local makes the expected
    violation auditable without model sampling noise.
    """

    case_id: str
    category: str
    source_case_id: str
    description: str
    transcript_text: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for building the Exp 2891 artifact."""

    output_path: Path | None = None
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME


def build_local_pilot_cases() -> list[PilotCase]:
    """Return deterministic transcript probes for required CCTU categories."""

    source_cases = cctu.build_benchmark_cases()
    resource_case = source_cases[0]
    toolset_case = source_cases[1]
    behavior_case = source_cases[2]
    response_case = source_cases[3]
    verifier_case = source_cases[4]

    return [
        PilotCase(
            case_id="cctu-2891-resource-001",
            category="resource",
            source_case_id=resource_case.case_id,
            description="Missing tool call exercises resource/tool-call allocation validation.",
            transcript_text=_mutated_transcript(resource_case, drop_tool_call=True),
        ),
        PilotCase(
            case_id="cctu-2891-toolset-001",
            category="toolset",
            source_case_id=toolset_case.case_id,
            description="Wrong tool arguments exercise allowed-tool and argument-shape validation.",
            transcript_text=_mutated_transcript(
                toolset_case,
                tool_arguments={"operation": "sum", "numbers": [1, 2, 3]},
            ),
        ),
        PilotCase(
            case_id="cctu-2891-behavior-001",
            category="behavior",
            source_case_id=behavior_case.case_id,
            description="Wrong declared tool result exercises executable behavior replay.",
            transcript_text=_mutated_transcript(behavior_case, tool_result={"value": -999}),
        ),
        PilotCase(
            case_id="cctu-2891-response-001",
            category="response",
            source_case_id=response_case.case_id,
            description="Wrong final answer exercises response constraint validation.",
            transcript_text=_mutated_transcript(response_case, final_answer="wrong final answer"),
        ),
        PilotCase(
            case_id="cctu-2891-response-verifier-001",
            category="response_verifier",
            source_case_id=verifier_case.case_id,
            description="Incorrect verifier decision exercises response-side verifier consistency.",
            transcript_text=_mutated_transcript(verifier_case, verifier_accept=False),
        ),
    ]


def validate_pilot_case(case: PilotCase) -> JsonDict:
    """Replay one pilot transcript and return localized step-level checks."""

    source_case = _source_case_by_id(case.source_case_id)
    parsed = cctu.extract_json_object(case.transcript_text)
    validation = cctu.validate_transcript(source_case, case.transcript_text)
    validator = validation["validator_result"]
    verifier = validation["verifier_result"]
    steps = _step_results(parsed, validator)
    violations = [
        {
            "step_id": step["step_id"],
            "category": step["category"],
            "localized_to": step["localized_to"],
            "detail": step["detail"],
        }
        for step in steps
        if not step["passed"]
    ]
    row: JsonDict = {
        "case_id": case.case_id,
        "source_case_id": case.source_case_id,
        "category": case.category,
        "description": case.description,
        "source_module": CCTU_SOURCE_MODULE,
        "executable_validation_used": True,
        "overall_passed": bool(verifier["accepted"]) and all(step["passed"] for step in steps),
        "step_results": steps,
        "violations": violations,
        "validator_result": validator,
        "verifier_result": verifier,
        "input_checksum": _checksum(
            {
                "case_id": case.case_id,
                "category": case.category,
                "source_case_id": case.source_case_id,
                "transcript_text": case.transcript_text,
            }
        ),
    }
    row["validation_checksum"] = _checksum({key: row[key] for key in sorted(row)})
    return row


def category_coverage(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize per-category pass counts for validation rows."""

    totals: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for row in rows:
        category = str(row["category"])
        totals[category] += 1
        if row.get("overall_passed") is True:
            passed[category] += 1
    return {
        category: {"passed": int(passed[category]), "total": int(totals[category])}
        for category in sorted(totals)
    }


def build_experiment_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 2891 validator-pilot artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    rows = [validate_pilot_case(case) for case in build_local_pilot_cases()]
    categories = sorted({str(row["category"]) for row in rows})
    executable_used = bool(rows) and all(row["executable_validation_used"] for row in rows)
    required_covered = set(REQUIRED_CATEGORIES) <= set(categories)
    checksummed = all(row.get("input_checksum") and row.get("validation_checksum") for row in rows)
    ready = bool(executable_used and required_covered and checksummed)
    artifact: JsonDict = {
        "artifact": "experiment_2891_cctu_executable_constraint_validator_pilot_v1",
        "schema": "carnot.cctu_executable_constraint_validator_pilot.v1",
        "honest_verdict": (
            "complete: local CCTU-style executable constraint validator pilot ready"
            if ready
            else "blocked_cctu_validator_pilot_not_ready"
        ),
        "cctu_validator_ready": ready,
        "source_modules": list(SOURCE_MODULES),
        "n_cases": len(rows),
        "constraint_categories": categories,
        "category_coverage": category_coverage(rows),
        "executable_validation_used": executable_used,
        "validation_rows": rows,
        "unsupported_categories": dict(UNSUPPORTED_CATEGORIES),
        "live_llm_called": False,
        "headline_metric_claim_made": False,
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, config.clock() - started),
    }
    return artifact


def write_experiment_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2891 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_experiment_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _mutated_transcript(
    case: cctu.BenchmarkCase,
    *,
    drop_tool_call: bool = False,
    tool_arguments: JsonDict | None = None,
    tool_result: Any | None = None,
    final_answer: str | None = None,
    verifier_accept: bool = False,
) -> str:
    payload = json.loads(cctu.compliant_transcript_for_case(case))
    if drop_tool_call:
        payload.pop("tool_call", None)
    if tool_arguments is not None:
        payload["tool_call"]["arguments"] = tool_arguments
    if tool_result is not None:
        payload["tool_result"] = tool_result
    if final_answer is not None:
        payload["final_answer"] = final_answer
    payload["verifier"] = {"accept": verifier_accept}
    return json.dumps(payload, sort_keys=True)


def _source_case_by_id(case_id: str) -> cctu.BenchmarkCase:
    for source_case in cctu.build_benchmark_cases():
        if source_case.case_id == case_id:
            return source_case
    raise ValueError(f"unknown source CCTU case: {case_id}")


def _step_results(parsed: JsonDict | None, validator: Mapping[str, Any]) -> list[JsonDict]:
    return [
        _step(
            "parse_json",
            "response",
            parsed is not None and validator.get("parse_error") is None,
            "json_object",
            f"parse_error={validator.get('parse_error')!r}",
        ),
        _step(
            "resource",
            "resource",
            isinstance(parsed, dict) and isinstance(parsed.get("tool_call"), dict),
            "tool_call",
            "expected exactly one JSON tool_call object",
        ),
        _step(
            "toolset",
            "toolset",
            bool(validator.get("tool_call_structure_valid")),
            "tool_call",
            "tool name and arguments must match the deterministic case contract",
        ),
        _step(
            "behavior",
            "behavior",
            bool(validator.get("tool_result_consistent")),
            "tool_result",
            f"tool_result_error={validator.get('tool_result_error')!r}",
        ),
        _step(
            "response",
            "response",
            bool(validator.get("final_answer_valid")),
            "final_answer",
            "final_answer must match the executable tool result",
        ),
        _step(
            "response_verifier",
            "response_verifier",
            bool(validator.get("verifier_outcome_valid")),
            "verifier.accept",
            f"model_declared_accept={validator.get('model_declared_accept')!r}",
        ),
    ]


def _step(
    step_id: str,
    category: str,
    passed: bool,
    localized_to: str,
    detail: str,
) -> JsonDict:
    return {
        "step_id": step_id,
        "category": category,
        "passed": bool(passed),
        "localized_to": localized_to,
        "detail": detail,
    }


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CCTU_SOURCE_MODULE",
    "FIELD_PRINCIPLES",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "STEP_IDS",
    "ExperimentConfig",
    "PilotCase",
    "build_experiment_artifact",
    "build_local_pilot_cases",
    "category_coverage",
    "validate_pilot_case",
    "write_experiment_artifact",
]
