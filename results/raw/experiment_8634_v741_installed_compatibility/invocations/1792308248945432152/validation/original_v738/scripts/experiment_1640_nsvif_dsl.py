#!/usr/bin/env python3
"""Exp 1640 NSVIF-style instruction-to-constraint DSL workflow.

The script is deliberately local and deterministic.  It reuses the bounded
safe DSL from `carnot.verifiers.dsl`, converts each parsed DSL item into the
pipeline's `ConstraintResult` shape, and compiles the same pack into fixed
Python validator functions.  That keeps natural-language instructions as data:
unsupported shapes fail closed instead of creating or executing Python code.

Spec: REQ-VERIFY-1640, SCENARIO-VERIFY-1640.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.verifiers import dsl


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = "experiment_1640_nsvif_dsl.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE

EXPERIMENT_ID = 1640
EXPERIMENT = "1640_nsvif_dsl"
SCHEMA = "nsvif_instruction_constraint_workflow_v1"
RUN_DATE = "20260509"
SPEC_TRACES = ["REQ-VERIFY-1640", "SCENARIO-VERIFY-1640"]
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "experiment_id",
    "dsl_schema_version",
    "parser_success",
    "instructions_tested",
    "constraints_extracted",
    "carnot_constraints_emitted",
    "validators_compiled",
    "known_good_pass_rate",
    "known_bad_reject_rate",
    "false_accept_rate",
    "arbitrary_code_execution_path_introduced",
    "tests_run",
    "honest_verdict",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def default_instruction_cases() -> list[JsonDict]:
    """Return the fixed Exp 1640 instruction cases used by tests and artifact writes."""

    return [
        {
            "case_id": "json-text-bound",
            "instruction": (
                'Respond in JSON with keys answer and confidence. Include "approved". '
                'Do not mention "secret". Use at most 12 words.'
            ),
            "known_good": '{"answer": "approved", "confidence": "high"}',
            "known_bad": '{"answer": "secret", "extra": true}',
        },
        {
            "case_id": "enum-answer",
            "instruction": 'Answer must be one of "yes", "no".',
            "known_good": "yes",
            "known_bad": "maybe",
        },
        {
            "case_id": "exact-bullets",
            "instruction": "Use exactly 2 bullet points.",
            "known_good": "- first\n- second",
            "known_bad": "- only one",
        },
        {
            "case_id": "text-min-max",
            "instruction": 'Include "north". Use at least 2 words and at most 5 words.',
            "known_good": "north star",
            "known_bad": "north",
        },
    ]


def constraint_to_carnot_result(
    constraint: dsl.ConstraintSpec,
    *,
    instruction: str,
) -> ConstraintResult:
    """Convert one bounded DSL constraint into Carnot's pipeline constraint row."""

    description = (
        f"Instruction constraint {constraint.id}: {constraint.op} "
        f"on {constraint.field}"
    )
    return ConstraintResult(
        constraint_type="instruction_constraint",
        description=description,
        metadata={
            "constraint_id": constraint.id,
            "instruction": instruction,
            "nsvif_operator": constraint.op,
            "field": constraint.field,
            "value": constraint.value,
            "source_text": constraint.source_text,
            "dsl_schema_version": dsl.DSL_SCHEMA_VERSION,
            "spec_traces": SPEC_TRACES,
        },
    )


def carnot_constraint_to_dict(result: ConstraintResult) -> JsonDict:
    """Return a JSON-safe representation of a Carnot `ConstraintResult`."""

    return {
        "constraint_type": result.constraint_type,
        "description": result.description,
        "energy_term": None,
        "metadata": dict(result.metadata),
    }


def parse_instruction_to_carnot_constraints(instruction: str) -> JsonDict:
    """Parse one instruction into schema-valid DSL and Carnot constraint rows."""

    try:
        pack = dsl.parse_instruction_constraints(instruction)
        if not pack.constraints:
            raise dsl.ConstraintDslError("no supported constraints")
    except dsl.ConstraintDslError as exc:
        return {
            "parser_success": False,
            "dsl_pack": None,
            "constraints": [],
            "error": str(exc),
        }

    carnot_constraints = [
        carnot_constraint_to_dict(
            constraint_to_carnot_result(constraint, instruction=pack.instruction)
        )
        for constraint in pack.constraints
    ]
    return {
        "parser_success": True,
        "dsl_pack": pack.to_dict(),
        "constraints": carnot_constraints,
        "error": "",
    }


def evaluate_instruction_case(case: Mapping[str, Any]) -> JsonDict:
    """Parse, compile, and evaluate one Exp 1640 instruction case."""

    case_id = str(case.get("case_id") or "unknown")
    instruction = str(case.get("instruction") or "")
    parsed = parse_instruction_to_carnot_constraints(instruction)
    if not parsed["parser_success"]:
        return {
            "case_id": case_id,
            "instruction": instruction,
            "parser_success": False,
            "validator_compiled": False,
            "constraint_count": 0,
            "carnot_constraint_count": 0,
            "known_good": {"accepted": False, "failure_ids": []},
            "known_bad": {"accepted": False, "failure_ids": []},
            "dsl_pack": None,
            "carnot_constraints": [],
            "error": parsed["error"],
        }

    validator = dsl.compile_constraint_pack(parsed["dsl_pack"])
    known_good = validator.validate(str(case.get("known_good") or ""))
    known_bad = validator.validate(str(case.get("known_bad") or ""))
    constraints = list(parsed["constraints"])
    return {
        "case_id": case_id,
        "instruction": instruction,
        "parser_success": True,
        "validator_compiled": True,
        "constraint_count": len(validator.pack.constraints),
        "carnot_constraint_count": len(constraints),
        "known_good": known_good.to_dict(),
        "known_bad": known_bad.to_dict(),
        "dsl_pack": parsed["dsl_pack"],
        "carnot_constraints": constraints,
        "error": "",
    }


def build_artifact(
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Build the Exp 1640 terminal artifact without writing it."""

    rows = [evaluate_instruction_case(case) for case in (cases or default_instruction_cases())]
    instructions_tested = len(rows)
    parser_success = all(row["parser_success"] for row in rows) and bool(rows)
    validators_compiled = sum(1 for row in rows if row["validator_compiled"])
    constraints_extracted = sum(int(row["constraint_count"]) for row in rows)
    carnot_constraints_emitted = sum(int(row["carnot_constraint_count"]) for row in rows)
    good_passes = sum(1 for row in rows if row["known_good"]["accepted"])
    bad_rejects = sum(1 for row in rows if not row["known_bad"]["accepted"] and row["parser_success"])
    false_accepts = sum(1 for row in rows if row["known_bad"]["accepted"])
    known_good_pass_rate = _rate(good_passes, instructions_tested)
    known_bad_reject_rate = _rate(bad_rejects, instructions_tested)
    false_accept_rate = _rate(false_accepts, instructions_tested)
    arbitrary_code = dsl.compiler_uses_arbitrary_code_execution()
    complete = (
        parser_success
        and validators_compiled == instructions_tested
        and known_good_pass_rate == 1.0
        and known_bad_reject_rate == 1.0
        and false_accept_rate == 0.0
        and not arbitrary_code
    )
    return {
        "status": "complete" if complete else "partial",
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": SPEC_TRACES,
        "dsl_schema_version": dsl.DSL_SCHEMA_VERSION,
        "dsl_schema": dsl.DSL_SCHEMA,
        "parser_success": parser_success,
        "instructions_tested": instructions_tested,
        "constraints_extracted": constraints_extracted,
        "carnot_constraints_emitted": carnot_constraints_emitted,
        "validators_compiled": validators_compiled,
        "known_good_pass_rate": known_good_pass_rate,
        "known_bad_reject_rate": known_bad_reject_rate,
        "false_accept_rate": false_accept_rate,
        "arbitrary_code_execution_path_introduced": arbitrary_code,
        "case_results": rows,
        "tests_run": list(tests_run or []),
        "honest_verdict": _honest_verdict(complete, parser_success, false_accept_rate),
    }


def _honest_verdict(complete: bool, parser_success: bool, false_accept_rate: float) -> str:
    if complete:
        return (
            "complete: NSVIF instruction DSL parsed into Carnot constraints and "
            "compiled local Python validators with zero false accepts"
        )
    return (
        "partial: NSVIF instruction DSL did not satisfy all completion gates; "
        f"parser_success={parser_success}, false_accept_rate={false_accept_rate}"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that an Exp 1640 artifact is internally consistent."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert artifact["dsl_schema_version"] == dsl.DSL_SCHEMA_VERSION, "dsl_schema_version mismatch"
    assert artifact["instructions_tested"] >= 1, "instructions_tested must be positive"
    assert 0.0 <= artifact["false_accept_rate"] <= 1.0, "false_accept_rate out of range"
    if artifact["status"] == "complete":
        assert artifact["parser_success"] is True, "complete artifact requires parser_success"
        assert artifact["known_good_pass_rate"] == 1.0, "complete artifact requires good pass rate"
        assert artifact["known_bad_reject_rate"] == 1.0, "complete artifact requires bad reject rate"
        assert artifact["false_accept_rate"] == 0.0, "complete artifact requires false_accept_rate=0"
        assert artifact["arbitrary_code_execution_path_introduced"] is False, (
            "complete artifact cannot introduce arbitrary code execution"
        )


def run_experiment(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    tests_run: list[str] | None = None,
    cases: Iterable[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run Exp 1640 and write `results/experiment_1640_nsvif_dsl.json`."""

    artifact = build_artifact(cases=cases, tests_run=tests_run)
    artifact["artifact_path"] = str(output_path)
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
