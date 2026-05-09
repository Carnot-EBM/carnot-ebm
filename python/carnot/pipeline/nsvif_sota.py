"""Adapter from SOTA GGUF generation rows to NSVIF DSL inputs.

Spec: REQ-VERIFY-1641, SCENARIO-VERIFY-1641.

Local SOTA models are allowed to suggest instructions or bounded constraint
objects, but this module keeps those suggestions outside the execution trust
boundary.  Raw model text is parsed as JSON data, normalized into the existing
`carnot.verifiers.dsl.ConstraintPack` schema, and compiled only by Carnot's
fixed local DSL compiler.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.verifiers import dsl

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
EXPERIMENT_ID = 1641
EXPERIMENT = "1641_nsvif_sota"
ADAPTER_SCHEMA_VERSION = "carnot.nsvif_sota_adapter.v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1641_nsvif_sota.json")
SPEC_TRACES = ["REQ-VERIFY-1641", "SCENARIO-VERIFY-1641"]
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "adapter_schema_version",
    "dsl_schema_version",
    "sota_outputs_seen",
    "dsl_inputs_emitted",
    "validators_compiled",
    "known_good_pass_rate",
    "known_bad_reject_rate",
    "false_accept_rate",
    "arbitrary_code_execution_path_introduced",
    "tests_run",
    "honest_verdict",
)
MANDATED_SOTA_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)


def sha256_text(text: str) -> str:
    """Return the stable SHA-256 digest used for raw GGUF output provenance."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_sota_output_cases() -> list[JsonDict]:
    """Return deterministic SOTA-shaped rows used by tests and artifact writes."""

    return [
        {
            "case_id": "json-text-bound",
            "model_hf_id": MANDATED_SOTA_MODEL_IDS[0],
            "model_name": "Qwen3.6-35B-A3B",
            "model_path": "/models/qwen.gguf",
            "output_text": json.dumps(
                {
                    "instruction": (
                        'Respond in JSON with keys answer and confidence. Include "approved". '
                        'Do not mention "secret". Use at most 12 words.'
                    )
                }
            ),
            "known_good": '{"answer": "approved", "confidence": "high"}',
            "known_bad": '{"answer": "secret", "extra": true}',
        },
        {
            "case_id": "enum-answer",
            "model_hf_id": MANDATED_SOTA_MODEL_IDS[1],
            "model_name": "Gemma4-31B-it",
            "model_path": "/models/gemma31.gguf",
            "output_text": json.dumps(
                {
                    "instruction": 'Answer one of "yes", "no".',
                    "constraints": [
                        {
                            "id": "choice",
                            "op": "enum",
                            "field": "text",
                            "value": ["yes", "no"],
                            "source_text": "bounded yes/no answer",
                        }
                    ],
                }
            ),
            "known_good": "yes",
            "known_bad": "maybe",
        },
        {
            "case_id": "existing-pack",
            "model_hf_id": MANDATED_SOTA_MODEL_IDS[2],
            "model_name": "Gemma4-26B-it",
            "model_path": "/models/gemma26.gguf",
            "output_text": json.dumps(
                {
                    "constraint_pack": {
                        "schema_version": dsl.DSL_SCHEMA_VERSION,
                        "instruction": 'Include "north". Use at least 2 words.',
                        "constraints": [
                            {
                                "id": "c001-contains",
                                "op": "contains",
                                "field": "text",
                                "value": "north",
                            },
                            {
                                "id": "c002-min_words",
                                "op": "min_words",
                                "field": "text",
                                "value": 2,
                            },
                        ],
                    }
                }
            ),
            "known_good": "north star",
            "known_bad": "north",
        },
    ]


def adapt_sota_output(row: Mapping[str, Any]) -> JsonDict:
    """Normalize one raw SOTA GGUF output row into an NSVIF DSL input."""

    case_id = str(row.get("case_id") or "unknown")
    raw_output = str(row.get("output_text") or "")
    base = _base_result(row, case_id=case_id, raw_output=raw_output)
    try:
        payload = _first_supported_payload(raw_output)
        pack = _pack_from_payload(payload)
        if not pack.constraints:
            raise dsl.ConstraintDslError("no supported constraints")
        validator = dsl.compile_constraint_pack(pack)
    except dsl.ConstraintDslError as exc:
        return {
            **base,
            "adapter_success": False,
            "validator_compiled": False,
            "dsl_input": None,
            "carnot_constraints": [],
            "known_good": {"accepted": False, "failure_ids": []},
            "known_bad": {"accepted": False, "failure_ids": []},
            "error": str(exc),
        }

    good = validator.validate(str(row.get("known_good") or ""))
    bad = validator.validate(str(row.get("known_bad") or ""))
    carnot_constraints = [
        _carnot_constraint_to_dict(
            _constraint_to_carnot_result(
                constraint,
                instruction=pack.instruction,
                row=row,
                raw_output_sha256=base["raw_output_sha256"],
            )
        )
        for constraint in pack.constraints
    ]
    return {
        **base,
        "adapter_success": True,
        "validator_compiled": True,
        "dsl_input": pack.to_dict(),
        "carnot_constraints": carnot_constraints,
        "known_good": good.to_dict(),
        "known_bad": bad.to_dict(),
        "error": "",
    }


def build_artifact(
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 1641 artifact without writing it."""

    output_rows = [adapt_sota_output(row) for row in (rows or default_sota_output_cases())]
    sota_outputs_seen = len(output_rows)
    dsl_inputs_emitted = sum(1 for row in output_rows if row["adapter_success"])
    validators_compiled = sum(1 for row in output_rows if row["validator_compiled"])
    good_passes = sum(1 for row in output_rows if row["known_good"]["accepted"])
    bad_rejects = sum(
        1 for row in output_rows if row["adapter_success"] and not row["known_bad"]["accepted"]
    )
    false_accepts = sum(1 for row in output_rows if row["known_bad"]["accepted"])
    known_good_pass_rate = _rate(good_passes, sota_outputs_seen)
    known_bad_reject_rate = _rate(bad_rejects, sota_outputs_seen)
    false_accept_rate = _rate(false_accepts, sota_outputs_seen)
    arbitrary_code = dsl.compiler_uses_arbitrary_code_execution()
    complete = (
        bool(output_rows)
        and dsl_inputs_emitted == sota_outputs_seen
        and validators_compiled == dsl_inputs_emitted
        and known_good_pass_rate == 1.0
        and known_bad_reject_rate == 1.0
        and false_accept_rate == 0.0
        and not arbitrary_code
    )
    return {
        "status": "complete" if complete else "partial",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": SPEC_TRACES,
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "dsl_schema_version": dsl.DSL_SCHEMA_VERSION,
        "dsl_schema": dsl.DSL_SCHEMA,
        "model_specs": list(MANDATED_SOTA_MODEL_IDS),
        "live_sota_model_inference_used": False,
        "sota_outputs_seen": sota_outputs_seen,
        "dsl_inputs_emitted": dsl_inputs_emitted,
        "validators_compiled": validators_compiled,
        "known_good_pass_rate": known_good_pass_rate,
        "known_bad_reject_rate": known_bad_reject_rate,
        "false_accept_rate": false_accept_rate,
        "false_accepts": false_accepts,
        "arbitrary_code_execution_path_introduced": arbitrary_code,
        "adapter_rows": output_rows,
        "tests_run": list(tests_run or []),
        "honest_verdict": _honest_verdict(complete, dsl_inputs_emitted, false_accept_rate),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that a terminal Exp 1641 artifact is internally consistent."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert artifact["adapter_schema_version"] == ADAPTER_SCHEMA_VERSION, (
        "adapter_schema_version mismatch"
    )
    assert artifact["dsl_schema_version"] == dsl.DSL_SCHEMA_VERSION, "dsl_schema_version mismatch"
    assert artifact["sota_outputs_seen"] >= 1, "sota_outputs_seen must be positive"
    assert 0.0 <= artifact["false_accept_rate"] <= 1.0, "false_accept_rate out of range"
    if artifact["status"] == "complete":
        assert artifact["dsl_inputs_emitted"] == artifact["sota_outputs_seen"], (
            "complete artifact requires all SOTA rows to emit DSL inputs"
        )
        assert artifact["validators_compiled"] == artifact["dsl_inputs_emitted"], (
            "complete artifact requires validators_compiled=dsl_inputs_emitted"
        )
        assert artifact["known_good_pass_rate"] == 1.0, "complete artifact requires good pass rate"
        assert artifact["known_bad_reject_rate"] == 1.0, (
            "complete artifact requires bad reject rate"
        )
        assert artifact["false_accept_rate"] == 0.0, (
            "complete artifact requires false_accept_rate=0"
        )
        assert artifact["arbitrary_code_execution_path_introduced"] is False, (
            "complete artifact cannot introduce arbitrary code execution"
        )


def run_experiment(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    tests_run: list[str] | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the NSVIF SOTA adapter fixture and write the JSON deliverable."""

    artifact = build_artifact(rows=rows, tests_run=tests_run)
    artifact["artifact_path"] = str(output_path)
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def _base_result(row: Mapping[str, Any], *, case_id: str, raw_output: str) -> JsonDict:
    return {
        "case_id": case_id,
        "model_hf_id": str(row.get("model_hf_id") or ""),
        "model_name": str(row.get("model_name") or ""),
        "model_path": str(row.get("model_path") or ""),
        "raw_output_sha256": sha256_text(raw_output),
        "raw_output_excerpt": raw_output[:500],
    }


def _first_supported_payload(raw_output: str) -> JsonDict:
    for obj in _extract_json_objects(raw_output):
        if _is_supported_payload(obj):
            return obj
    raise dsl.ConstraintDslError("no_json_object")


def _extract_json_objects(text: str) -> list[JsonDict]:
    decoder = json.JSONDecoder()
    objects: list[JsonDict] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def _is_supported_payload(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "instruction",
            "nsvif_instruction",
            "dsl_pack",
            "constraint_pack",
            "constraints",
        )
    )


def _pack_from_payload(payload: Mapping[str, Any]) -> dsl.ConstraintPack:
    pack_payload = payload.get("dsl_pack") or payload.get("constraint_pack")
    if isinstance(pack_payload, Mapping):
        return dsl.constraint_pack_from_dict(pack_payload)
    instruction = str(
        payload.get("instruction")
        or payload.get("nsvif_instruction")
        or "SOTA proposed constraints"
    )
    constraints = payload.get("constraints")
    if isinstance(constraints, list):
        return dsl.constraint_pack_from_dict(
            {
                "schema_version": dsl.DSL_SCHEMA_VERSION,
                "instruction": instruction,
                "constraints": constraints,
            }
        )
    return dsl.parse_instruction_constraints(instruction)


def _constraint_to_carnot_result(
    constraint: dsl.ConstraintSpec,
    *,
    instruction: str,
    row: Mapping[str, Any],
    raw_output_sha256: str,
) -> ConstraintResult:
    return ConstraintResult(
        constraint_type="instruction_constraint",
        description=(
            f"SOTA NSVIF constraint {constraint.id}: {constraint.op} on {constraint.field}"
        ),
        metadata={
            "constraint_id": constraint.id,
            "case_id": str(row.get("case_id") or "unknown"),
            "instruction": instruction,
            "nsvif_operator": constraint.op,
            "field": constraint.field,
            "value": constraint.value,
            "source_text": constraint.source_text,
            "dsl_schema_version": dsl.DSL_SCHEMA_VERSION,
            "model_hf_id": str(row.get("model_hf_id") or ""),
            "model_name": str(row.get("model_name") or ""),
            "model_path": str(row.get("model_path") or ""),
            "raw_output_sha256": raw_output_sha256,
            "spec_traces": SPEC_TRACES,
        },
    )


def _carnot_constraint_to_dict(result: ConstraintResult) -> JsonDict:
    return {
        "constraint_type": result.constraint_type,
        "description": result.description,
        "energy_term": None,
        "metadata": dict(result.metadata),
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _honest_verdict(complete: bool, dsl_inputs_emitted: int, false_accept_rate: float) -> str:
    if complete:
        return (
            "complete: SOTA GGUF outputs normalize into NSVIF DSL inputs and "
            "compile to local validators with zero false accepts"
        )
    return (
        "partial: SOTA GGUF output adapter did not satisfy all completion gates; "
        f"dsl_inputs_emitted={dsl_inputs_emitted}, false_accept_rate={false_accept_rate}"
    )


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact
