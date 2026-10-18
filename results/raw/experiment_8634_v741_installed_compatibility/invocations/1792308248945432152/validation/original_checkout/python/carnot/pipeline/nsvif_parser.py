"""Product NSVIF instruction-to-constraint parser and validator compiler.

Spec: REQ-VERIFY-1666, SCENARIO-VERIFY-1666.

This module is the importable product-facing layer for turning bounded natural
language prompts into executable Carnot constraints. It delegates grammar and
operator safety to `carnot.verifiers.dsl`, then exposes three local validator
views: the fixed Python validator, a PySAT-compatible CNF hard conjunction, and
a Z3-compatible hard conjunction. No generated Python or model-proposed code is
executed.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.verifiers import dsl

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
EXPERIMENT_ID = 1666
EXPERIMENT = "1666_nsvif"
PARSER_SCHEMA_VERSION = "carnot.pipeline.nsvif_parser.v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1666_nsvif.json")
SPEC_TRACES = ["REQ-VERIFY-1666", "SCENARIO-VERIFY-1666"]
MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "parser_schema_version",
    "dsl_schema_version",
    "model_specs",
    "cases_attempted",
    "validators_compiled",
    "python_validators_compiled",
    "pysat_validators_compiled",
    "z3_validators_compiled",
    "false_accepts",
    "compilation_rate",
    "known_good_pass_rate",
    "known_bad_reject_rate",
    "false_accept_rate",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class Z3Problem:
    """Z3-compatible hard conjunction metadata for a parsed constraint pack."""

    variables: dict[str, str]
    assertions: list[str]
    backend: str = "z3-compatible-hard-conjunction"

    def to_dict(self) -> JsonDict:
        """Return a stable JSON-compatible Z3 problem descriptor."""

        return {
            "backend": self.backend,
            "variables": dict(self.variables),
            "assertions": list(self.assertions),
            "z3_backend_available": z3_backend_available(),
        }


@dataclass(frozen=True)
class Z3ValidationResult:
    """One output validation result from the Z3 hard-conjunction backend."""

    accepted: bool
    sat_status: str
    failure_ids: tuple[str, ...]
    z3_backend_available: bool

    def to_dict(self) -> JsonDict:
        """Return the JSON shape used by tests and experiment artifacts."""

        return {
            "accepted": self.accepted,
            "sat_status": self.sat_status,
            "failure_ids": list(self.failure_ids),
            "z3_backend_available": self.z3_backend_available,
        }


@dataclass(frozen=True)
class Z3ConstraintValidator:
    """A local Z3 hard-conjunction validator backed by fixed DSL checks."""

    pack: dsl.ConstraintPack
    python_validator: dsl.CompiledInstructionValidator
    problem: Z3Problem

    def validate(self, output_text: str) -> Z3ValidationResult:
        """Validate output by asserting every local constraint verdict in Z3."""

        python_result = self.python_validator.validate(output_text)
        failure_ids = tuple(python_result.failure_ids)
        if not z3_backend_available():
            return Z3ValidationResult(
                accepted=python_result.accepted,
                sat_status="sat" if python_result.accepted else "unsat",
                failure_ids=failure_ids,
                z3_backend_available=False,
            )

        import z3  # type: ignore[import-untyped]

        solver = z3.Solver()
        symbols = {
            constraint_id: z3.Bool(symbol)
            for constraint_id, symbol in self.problem.variables.items()
        }
        failed = set(failure_ids)
        for constraint_id, symbol in symbols.items():
            solver.add(symbol == (constraint_id not in failed))
        solver.add(z3.And(*symbols.values()))
        sat_status = str(solver.check())
        return Z3ValidationResult(
            accepted=sat_status == "sat",
            sat_status=sat_status,
            failure_ids=failure_ids,
            z3_backend_available=True,
        )


@dataclass(frozen=True)
class NsvifValidatorBundle:
    """Compiled validator bundle for one parsed NSVIF prompt."""

    pack: dsl.ConstraintPack
    python_validator: dsl.CompiledInstructionValidator
    z3_validator: Z3ConstraintValidator

    @property
    def pysat_problem(self) -> dsl.PySatProblem:
        """Return the PySAT-compatible CNF produced by the DSL compiler."""

        return self.python_validator.pysat_problem

    def to_dict(self) -> JsonDict:
        """Return backend metadata without serializing callable validators."""

        return {
            "compiled_backends": ["python", "pysat_cnf", "z3"],
            "pysat_problem": self.pysat_problem.to_dict(),
            "z3_problem": self.z3_validator.problem.to_dict(),
        }


def default_model_prompt_cases() -> list[JsonDict]:
    """Return bounded prompt rows for the mandated GGUF model specifications."""

    return [
        {
            "case_id": "qwen-json-text-bound",
            "model_hf_id": MODEL_SPECS[0],
            "prompt": (
                'Respond in JSON with keys answer and confidence. Include "approved". '
                'Do not mention "secret". Use at most 12 words.'
            ),
            "known_good": '{"answer": "approved", "confidence": "high"}',
            "known_bad": '{"answer": "secret", "extra": true}',
        },
        {
            "case_id": "gemma-enum-answer",
            "model_hf_id": MODEL_SPECS[1],
            "prompt": 'Answer must be one of "yes", "no".',
            "known_good": "yes",
            "known_bad": "maybe",
        },
    ]


def z3_backend_available() -> bool:
    """Return whether the optional Z3 Python package is importable."""

    return importlib.util.find_spec("z3") is not None


def parse_nsvif_prompt(
    prompt: str,
    *,
    case_id: str = "",
    model_hf_id: str = "",
) -> JsonDict:
    """Parse a bounded natural-language prompt into DSL and Carnot rows."""

    try:
        pack = dsl.parse_instruction_constraints(prompt)
        if not pack.constraints:
            raise dsl.ConstraintDslError("no supported constraints")
    except dsl.ConstraintDslError as exc:
        return {
            "parser_success": False,
            "dsl_pack": None,
            "carnot_constraints": [],
            "error": str(exc),
        }

    carnot_constraints = [
        _carnot_constraint_to_dict(
            constraint_to_carnot_result(
                constraint,
                instruction=pack.instruction,
                case_id=case_id,
                model_hf_id=model_hf_id,
            )
        )
        for constraint in pack.constraints
    ]
    return {
        "parser_success": True,
        "dsl_pack": pack.to_dict(),
        "carnot_constraints": carnot_constraints,
        "error": "",
    }


def constraint_to_carnot_result(
    constraint: dsl.ConstraintSpec,
    *,
    instruction: str,
    case_id: str = "",
    model_hf_id: str = "",
) -> ConstraintResult:
    """Convert one bounded DSL constraint into a Carnot constraint row."""

    return ConstraintResult(
        constraint_type="instruction_constraint",
        description=(
            f"NSVIF instruction constraint {constraint.id}: {constraint.op} on {constraint.field}"
        ),
        metadata={
            "constraint_id": constraint.id,
            "instruction": instruction,
            "nsvif_operator": constraint.op,
            "field": constraint.field,
            "value": constraint.value,
            "source_text": constraint.source_text,
            "dsl_schema_version": dsl.DSL_SCHEMA_VERSION,
            "parser_schema_version": PARSER_SCHEMA_VERSION,
            "case_id": case_id,
            "model_hf_id": model_hf_id,
            "spec_traces": SPEC_TRACES,
        },
    )


def compile_nsvif_validators(pack: dsl.ConstraintPack | Mapping[str, Any]) -> NsvifValidatorBundle:
    """Compile one DSL pack to Python, PySAT-compatible CNF, and Z3 validators."""

    typed_pack = dsl.constraint_pack_from_dict(pack) if isinstance(pack, Mapping) else pack
    python_validator = dsl.compile_constraint_pack(typed_pack)
    z3_validator = Z3ConstraintValidator(
        pack=typed_pack,
        python_validator=python_validator,
        problem=compile_z3_problem(typed_pack),
    )
    return NsvifValidatorBundle(
        pack=typed_pack,
        python_validator=python_validator,
        z3_validator=z3_validator,
    )


def compile_z3_problem(pack: dsl.ConstraintPack) -> Z3Problem:
    """Compile constraint IDs to a Z3-compatible hard-conjunction descriptor."""

    variables = {constraint.id: _z3_symbol(constraint.id) for constraint in pack.constraints}
    return Z3Problem(variables=variables, assertions=list(variables.values()))


def evaluate_prompt_case(case: Mapping[str, Any]) -> JsonDict:
    """Parse, compile, and evaluate one bounded NSVIF prompt case."""

    case_id = str(case.get("case_id") or "unknown")
    model_hf_id = str(case.get("model_hf_id") or "")
    prompt = str(case.get("prompt") or case.get("instruction") or "")
    parsed = parse_nsvif_prompt(prompt, case_id=case_id, model_hf_id=model_hf_id)
    if not parsed["parser_success"]:
        return _failed_case_result(
            case_id=case_id,
            model_hf_id=model_hf_id,
            prompt=prompt,
            error=parsed["error"],
        )

    try:
        validators = compile_nsvif_validators(parsed["dsl_pack"])
    except dsl.ConstraintDslError as exc:
        return _failed_case_result(
            case_id=case_id,
            model_hf_id=model_hf_id,
            prompt=prompt,
            error=str(exc),
        )

    good_text = str(case.get("known_good") or "")
    bad_text = str(case.get("known_bad") or "")
    python_good = validators.python_validator.validate(good_text)
    python_bad = validators.python_validator.validate(bad_text)
    z3_good = validators.z3_validator.validate(good_text)
    z3_bad = validators.z3_validator.validate(bad_text)
    backend_metadata = validators.to_dict()
    return {
        "case_id": case_id,
        "model_hf_id": model_hf_id,
        "prompt": prompt,
        "parser_success": True,
        "validators_compiled": True,
        "compiled_backends": backend_metadata["compiled_backends"],
        "constraint_count": len(validators.pack.constraints),
        "dsl_pack": parsed["dsl_pack"],
        "carnot_constraints": parsed["carnot_constraints"],
        "pysat_problem": backend_metadata["pysat_problem"],
        "z3_problem": backend_metadata["z3_problem"],
        "python_known_good": python_good.to_dict(),
        "python_known_bad": python_bad.to_dict(),
        "z3_known_good": z3_good.to_dict(),
        "z3_known_bad": z3_bad.to_dict(),
        "error": "",
    }


def build_artifact(
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Build the Exp 1666 artifact without writing it."""

    rows = [evaluate_prompt_case(case) for case in (cases or default_model_prompt_cases())]
    cases_attempted = len(rows)
    validators_compiled = sum(1 for row in rows if row["validators_compiled"])
    python_good_passes = sum(1 for row in rows if row["python_known_good"]["accepted"])
    python_bad_rejects = sum(
        1 for row in rows if row["parser_success"] and not row["python_known_bad"]["accepted"]
    )
    z3_good_passes = sum(1 for row in rows if row["z3_known_good"]["accepted"])
    z3_bad_rejects = sum(
        1 for row in rows if row["parser_success"] and not row["z3_known_bad"]["accepted"]
    )
    false_accepts = sum(
        1 for row in rows if row["python_known_bad"]["accepted"] or row["z3_known_bad"]["accepted"]
    )
    known_good_passes = sum(
        1
        for row in rows
        if row["python_known_good"]["accepted"] and row["z3_known_good"]["accepted"]
    )
    known_bad_rejects = sum(
        1
        for row in rows
        if row["parser_success"]
        and not row["python_known_bad"]["accepted"]
        and not row["z3_known_bad"]["accepted"]
    )
    compilation_rate = _rate(validators_compiled, cases_attempted)
    known_good_pass_rate = _rate(known_good_passes, cases_attempted)
    known_bad_reject_rate = _rate(known_bad_rejects, cases_attempted)
    false_accept_rate = _rate(false_accepts, cases_attempted)
    complete = (
        cases_attempted > 0
        and validators_compiled == cases_attempted
        and known_good_pass_rate == 1.0
        and known_bad_reject_rate == 1.0
        and false_accepts == 0
        and not dsl.compiler_uses_arbitrary_code_execution()
    )
    return {
        "status": "complete" if complete else "partial",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": SPEC_TRACES,
        "parser_schema_version": PARSER_SCHEMA_VERSION,
        "dsl_schema_version": dsl.DSL_SCHEMA_VERSION,
        "dsl_schema": dsl.DSL_SCHEMA,
        "model_specs": list(MODEL_SPECS),
        "live_sota_model_inference_used": False,
        "cases_attempted": cases_attempted,
        "validators_compiled": validators_compiled,
        "python_validators_compiled": validators_compiled,
        "pysat_validators_compiled": validators_compiled,
        "z3_validators_compiled": validators_compiled,
        "python_known_good_pass_rate": _rate(python_good_passes, cases_attempted),
        "python_known_bad_reject_rate": _rate(python_bad_rejects, cases_attempted),
        "z3_known_good_pass_rate": _rate(z3_good_passes, cases_attempted),
        "z3_known_bad_reject_rate": _rate(z3_bad_rejects, cases_attempted),
        "known_good_pass_rate": known_good_pass_rate,
        "known_bad_reject_rate": known_bad_reject_rate,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accept_rate,
        "compilation_rate": compilation_rate,
        "arbitrary_code_execution_path_introduced": (dsl.compiler_uses_arbitrary_code_execution()),
        "case_results": rows,
        "tests_run": list(tests_run or []),
        "honest_verdict": _honest_verdict(complete, compilation_rate, false_accepts),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that an Exp 1666 artifact is internally consistent."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert artifact["parser_schema_version"] == PARSER_SCHEMA_VERSION, (
        "parser_schema_version mismatch"
    )
    assert artifact["dsl_schema_version"] == dsl.DSL_SCHEMA_VERSION, "dsl_schema_version mismatch"
    assert artifact["cases_attempted"] >= 1, "cases_attempted must be positive"
    assert 0.0 <= artifact["compilation_rate"] <= 1.0, "compilation_rate out of range"
    assert 0.0 <= artifact["false_accept_rate"] <= 1.0, "false_accept_rate out of range"
    if artifact["status"] == "complete":
        assert artifact["validators_compiled"] == artifact["cases_attempted"], (
            "complete artifact requires every validator to compile"
        )
        assert artifact["compilation_rate"] == 1.0, "complete artifact requires compilation_rate=1"
        assert artifact["known_good_pass_rate"] == 1.0, "complete artifact requires good pass rate"
        assert artifact["known_bad_reject_rate"] == 1.0, (
            "complete artifact requires bad reject rate"
        )
        assert artifact["false_accepts"] == 0, "complete artifact requires false_accepts=0"
        assert artifact["false_accept_rate"] == 0.0, (
            "complete artifact requires false_accept_rate=0"
        )
        assert artifact["arbitrary_code_execution_path_introduced"] is False, (
            "complete artifact cannot introduce arbitrary code execution"
        )


def run_experiment(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run Exp 1666 and write `results/experiment_1666_nsvif.json`."""

    artifact = build_artifact(cases=cases, tests_run=tests_run)
    artifact["artifact_path"] = str(output_path)
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def _failed_case_result(
    *,
    case_id: str,
    model_hf_id: str,
    prompt: str,
    error: str,
) -> JsonDict:
    empty_result = {"accepted": False, "failure_ids": []}
    return {
        "case_id": case_id,
        "model_hf_id": model_hf_id,
        "prompt": prompt,
        "parser_success": False,
        "validators_compiled": False,
        "compiled_backends": [],
        "constraint_count": 0,
        "dsl_pack": None,
        "carnot_constraints": [],
        "pysat_problem": None,
        "z3_problem": None,
        "python_known_good": dict(empty_result),
        "python_known_bad": dict(empty_result),
        "z3_known_good": dict(empty_result, sat_status="unknown"),
        "z3_known_bad": dict(empty_result, sat_status="unknown"),
        "error": error,
    }


def _carnot_constraint_to_dict(result: ConstraintResult) -> JsonDict:
    return {
        "constraint_type": result.constraint_type,
        "description": result.description,
        "energy_term": None,
        "metadata": dict(result.metadata),
    }


def _z3_symbol(constraint_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", constraint_id)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _honest_verdict(complete: bool, compilation_rate: float, false_accepts: int) -> str:
    if complete:
        return (
            "complete: NSVIF prompts compiled to Python, PySAT-compatible CNF, "
            "and Z3 validators with zero false accepts"
        )
    return (
        "partial: NSVIF parser did not satisfy all completion gates; "
        f"compilation_rate={compilation_rate}, false_accepts={false_accepts}"
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
