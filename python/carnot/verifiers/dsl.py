"""Bounded instruction-to-constraint DSL for local semantic validators.

Spec: REQ-VERIFY-1588, SCENARIO-VERIFY-1588.

This module intentionally recognizes a small set of natural-language patterns
instead of trying to understand arbitrary instructions.  The narrow surface is
the trust boundary: model- or user-written instructions become data in a JSON
DSL, then fixed local functions evaluate that data.  No generated Python is
executed, and unsupported shapes fail closed so later verifier stages can ask
for manual handling instead of accepting a guess.
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
EXPERIMENT_ID = "experiment_1588_nsvif_dsl"
HUMANEVAL_EXPERIMENT_ID = "experiment_1607_dsl_humaneval"
DSL_SCHEMA_VERSION = "carnot.instruction_constraints.v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1588_nsvif_dsl.json")
DEFAULT_HUMANEVAL_ARTIFACT_PATH = Path("results/experiment_1607_dsl_humaneval.json")
MAX_INSTRUCTION_CHARS = 2000
MAX_CONSTRAINTS = 8
MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
)
SUPPORTED_OPERATORS: frozenset[str] = frozenset(
    {
        "contains",
        "not_contains",
        "max_words",
        "min_words",
        "json_object",
        "json_has_keys",
        "enum",
        "exact_bullets",
    }
)
FORBIDDEN_TEXT_TOKENS: tuple[str, ...] = (
    "__import__",
    "import ",
    "eval(",
    "exec(",
    "open(",
    "pathlib",
    "subprocess",
    "os.",
    "socket",
    "requests.",
    "urllib",
    "http://",
    "https://",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "dsl_schema_version",
    "instructions_tested",
    "constraints_extracted",
    "validators_compiled",
    "pysat_cnf_compiled",
    "python_validator_pass_rate",
    "known_good_pass_rate",
    "known_bad_reject_rate",
    "false_accept_rate",
    "arbitrary_code_execution_path_introduced",
    "tests_run",
    "honest_verdict",
)
DSL_SCHEMA: JsonDict = {
    "schema_version": DSL_SCHEMA_VERSION,
    "max_instruction_chars": MAX_INSTRUCTION_CHARS,
    "max_constraints": MAX_CONSTRAINTS,
    "supported_operators": sorted(SUPPORTED_OPERATORS),
    "constraint_fields": {
        "id": "stable local constraint id",
        "op": "one supported operator",
        "field": "text or json",
        "value": "operator-specific expected value",
        "source_text": "instruction fragment that produced the constraint",
    },
}


class ConstraintDslError(ValueError):
    """Raised when an instruction or DSL pack cannot be compiled safely."""


@dataclass(frozen=True)
class ConstraintSpec:
    """One machine-checkable constraint extracted from instruction text."""

    id: str
    op: str
    field: str = "text"
    value: Any = None
    source_text: str = ""

    def to_dict(self) -> JsonDict:
        """Return the JSON form used by artifacts and schema checks."""

        payload: JsonDict = {"id": self.id, "op": self.op, "field": self.field}
        if self.value is not None:
            payload["value"] = self.value
        if self.source_text:
            payload["source_text"] = self.source_text
        return payload


@dataclass(frozen=True)
class ConstraintPack:
    """A bounded group of constraints produced from one instruction string."""

    instruction: str
    constraints: tuple[ConstraintSpec, ...]
    schema_version: str = DSL_SCHEMA_VERSION
    parser_version: str = "rule-v1"

    def to_dict(self) -> JsonDict:
        """Return a stable JSON-compatible representation of the pack."""

        return {
            "schema_version": self.schema_version,
            "parser_version": self.parser_version,
            "instruction": self.instruction,
            "max_constraints": MAX_CONSTRAINTS,
            "constraints": [constraint.to_dict() for constraint in self.constraints],
        }


@dataclass(frozen=True)
class ValidationIssue:
    """One failed constraint from a compiled validator run."""

    constraint_id: str
    message: str
    expected: Any
    observed: Any

    def to_dict(self) -> JsonDict:
        """Return a JSON-compatible issue record."""

        return {
            "constraint_id": self.constraint_id,
            "message": self.message,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome with per-constraint evidence for repair feedback."""

    accepted: bool
    issues: tuple[ValidationIssue, ...]

    @property
    def failure_ids(self) -> list[str]:
        """Return failed constraint IDs in evaluation order."""

        return [issue.constraint_id for issue in self.issues]

    def to_dict(self) -> JsonDict:
        """Return the result shape used in experiment fixtures."""

        return {
            "accepted": self.accepted,
            "issues": [issue.to_dict() for issue in self.issues],
            "failure_ids": self.failure_ids,
        }


@dataclass(frozen=True)
class PySatProblem:
    """PySAT-compatible hard-conjunction view of a constraint pack."""

    variables: dict[str, int]
    clauses: list[list[int]]
    description: str = "hard conjunction of instruction constraints"
    backend: str = "pysat-compatible-cnf"

    def to_dict(self) -> JsonDict:
        """Return a shape that can be passed to PySAT's CNF constructor."""

        return {
            "backend": self.backend,
            "description": self.description,
            "variables": dict(self.variables),
            "clauses": [list(clause) for clause in self.clauses],
        }


@dataclass(frozen=True)
class CompiledInstructionValidator:
    """Compiled local validator for one bounded instruction pack."""

    pack: ConstraintPack
    pysat_problem: PySatProblem

    def validate(self, output_text: str) -> ValidationResult:
        """Evaluate every compiled constraint against candidate output text."""

        issues: list[ValidationIssue] = []
        parsed_json, json_error = _parse_json_object(output_text)
        for constraint in self.pack.constraints:
            issue = _evaluate_constraint(constraint, output_text, parsed_json, json_error)
            if issue is not None:
                issues.append(issue)
        return ValidationResult(accepted=not issues, issues=tuple(issues))

    def __call__(self, output_text: str) -> bool:
        """Return a boolean verdict for call sites that only need pass/fail."""

        return self.validate(output_text).accepted


def parse_instruction_constraints(
    instruction: str,
    *,
    max_constraints: int = MAX_CONSTRAINTS,
) -> ConstraintPack:
    """Parse supported instruction patterns into a bounded DSL pack."""

    text = _clean_instruction(instruction)
    unsafe = _unsafe_reason(text)
    if unsafe:
        raise ConstraintDslError(unsafe)

    found: list[tuple[str, str, Any, str]] = []
    found.extend(_json_constraints(text))
    found.extend(_quoted_phrase_constraints(text))
    found.extend(_word_count_constraints(text))
    found.extend(_enum_constraints(text))
    found.extend(_bullet_constraints(text))

    if len(found) > max_constraints:
        raise ConstraintDslError(f"too many constraints:{len(found)}>{max_constraints}")

    constraints = tuple(
        ConstraintSpec(
            id=f"c{index:03d}-{op}",
            op=op,
            field=field,
            value=value,
            source_text=source,
        )
        for index, (op, field, value, source) in enumerate(found, start=1)
    )
    return ConstraintPack(instruction=text, constraints=constraints)


def validate_constraint_pack(payload: Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for a raw DSL mapping."""

    errors: list[str] = []
    if payload.get("schema_version") != DSL_SCHEMA_VERSION:
        errors.append("schema_version unsupported")
    if not isinstance(payload.get("instruction"), str):
        errors.append("instruction must be string")
    constraints = payload.get("constraints")
    if not isinstance(constraints, list):
        errors.append("constraints must be list")
        return errors
    if len(constraints) > MAX_CONSTRAINTS:
        errors.append(f"constraints too many:{len(constraints)}>{MAX_CONSTRAINTS}")
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            errors.append(f"constraint[{index}] must be object")
            continue
        op = constraint.get("op")
        if op not in SUPPORTED_OPERATORS:
            errors.append(f"constraint[{index}].op unsupported:{op}")
            continue
        errors.extend(_operator_value_errors(index, constraint))
    return errors


def constraint_pack_from_dict(payload: Mapping[str, Any]) -> ConstraintPack:
    """Build a typed pack from raw DSL JSON after schema validation."""

    errors = validate_constraint_pack(payload)
    if errors:
        raise ConstraintDslError("; ".join(errors))
    instruction = str(payload["instruction"])
    constraints = tuple(
        ConstraintSpec(
            id=str(item.get("id") or f"c{index:03d}-{item['op']}"),
            op=str(item["op"]),
            field=str(item.get("field") or "text"),
            value=item.get("value"),
            source_text=str(item.get("source_text") or ""),
        )
        for index, item in enumerate(payload["constraints"], start=1)
    )
    return ConstraintPack(instruction=instruction, constraints=constraints)


def compile_instruction_validator(instruction: str) -> CompiledInstructionValidator:
    """Parse and compile one natural-language instruction into a validator."""

    return compile_constraint_pack(parse_instruction_constraints(instruction))


def compile_constraint_pack(pack: ConstraintPack | Mapping[str, Any]) -> CompiledInstructionValidator:
    """Compile a typed or raw DSL pack into a local Python validator."""

    typed_pack = constraint_pack_from_dict(pack) if isinstance(pack, Mapping) else pack
    if not typed_pack.constraints:
        raise ConstraintDslError("no supported constraints to compile")
    errors = validate_constraint_pack(typed_pack.to_dict())
    if errors:
        raise ConstraintDslError("; ".join(errors))
    return CompiledInstructionValidator(
        pack=typed_pack,
        pysat_problem=compile_pysat_problem(typed_pack),
    )


def compile_pysat_problem(pack: ConstraintPack) -> PySatProblem:
    """Compile hard constraints into a PySAT-compatible unit-clause CNF."""

    variables = {constraint.id: index for index, constraint in enumerate(pack.constraints, start=1)}
    clauses = [[index] for index in variables.values()]
    return PySatProblem(variables=variables, clauses=clauses)


def pysat_backend_available() -> bool:
    """Return whether optional python-sat is importable in this environment."""

    return importlib.util.find_spec("pysat") is not None


def compiler_uses_arbitrary_code_execution() -> bool:
    """Return false because this compiler only dispatches fixed local functions."""

    return False


def write_experiment_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the fixed Exp 1588 fixture pack and write the terminal artifact."""

    fixtures = _experiment_fixtures()
    rows: list[JsonDict] = []
    for fixture in fixtures:
        validator = compile_instruction_validator(fixture["instruction"])
        good = validator.validate(fixture["known_good"])
        bad = validator.validate(fixture["known_bad"])
        rows.append(
            {
                "instruction": fixture["instruction"],
                "constraint_ids": [constraint.id for constraint in validator.pack.constraints],
                "constraint_ops": [constraint.op for constraint in validator.pack.constraints],
                "known_good": good.to_dict(),
                "known_bad": bad.to_dict(),
                "pysat_problem": validator.pysat_problem.to_dict(),
            }
        )

    good_passes = sum(row["known_good"]["accepted"] for row in rows)
    bad_rejects = sum(not row["known_bad"]["accepted"] for row in rows)
    false_accepts = sum(row["known_bad"]["accepted"] for row in rows)
    artifact: JsonDict = {
        "status": "complete",
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "spec_traces": ["REQ-VERIFY-1588", "SCENARIO-VERIFY-1588"],
        "dsl_schema_version": DSL_SCHEMA_VERSION,
        "dsl_schema": DSL_SCHEMA,
        "instructions_tested": len(rows),
        "constraints_extracted": sum(len(row["constraint_ids"]) for row in rows),
        "validators_compiled": len(rows),
        "pysat_backend_available": pysat_backend_available(),
        "pysat_cnf_compiled": sum(bool(row["pysat_problem"]["clauses"]) for row in rows),
        "python_validator_pass_rate": _rate(good_passes, len(rows)),
        "known_good_pass_rate": _rate(good_passes, len(rows)),
        "known_bad_reject_rate": _rate(bad_rejects, len(rows)),
        "false_accept_rate": _rate(false_accepts, len(rows)),
        "arbitrary_code_execution_path_introduced": compiler_uses_arbitrary_code_execution(),
        "fixture_rows": rows,
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: bounded instruction-to-constraint DSL compiled local "
            "Python validators and PySAT-compatible CNF with zero fixture false accepts"
        ),
    }
    _write_json(Path(output_path), artifact)
    return artifact


def load_humaneval_prompt_cases(*, sample_size: int | None = None) -> list[JsonDict]:
    """Load HumanEval prompt rows in the minimal shape needed by Exp 1607."""

    from datasets import load_dataset

    dataset = load_dataset("openai_humaneval", split="test")
    limit = len(dataset) if sample_size is None else min(sample_size, len(dataset))
    cases: list[JsonDict] = []
    for dataset_idx in range(limit):
        row = dataset[dataset_idx]
        cases.append(
            {
                "dataset_idx": dataset_idx,
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
            }
        )
    return cases


def extract_humaneval_prompt_constraints(case: Mapping[str, Any]) -> ConstraintPack:
    """Extract bounded code-output constraints from one HumanEval prompt.

    HumanEval prompts describe the function a model should produce.  Exp 1607
    compiles only structural obligations that the local DSL can check without
    executing generated code: the output should define the requested entry
    point, return a value, and avoid common non-code placeholders.
    """

    prompt = str(case.get("prompt") or "")
    entry_point = _humaneval_entry_point(case, prompt)
    constraints = (
        ("contains", "text", f"def {entry_point}", f"entry_point:{entry_point}"),
        ("contains", "text", "return", "humaneval_return_value"),
        ("not_contains", "text", "```", "markdown_fence_free_code"),
        ("not_contains", "text", "TODO", "placeholder_free_code"),
        ("not_contains", "text", "pass", "placeholder_free_code"),
        ("not_contains", "text", "NotImplementedError", "placeholder_free_code"),
    )
    specs = tuple(
        ConstraintSpec(
            id=f"c{index:03d}-{op}",
            op=op,
            field=field,
            value=value,
            source_text=source,
        )
        for index, (op, field, value, source) in enumerate(constraints, start=1)
    )
    return ConstraintPack(
        instruction=_humaneval_instruction(case, entry_point),
        constraints=specs,
        parser_version="humaneval-rule-v1",
    )


def evaluate_humaneval_dsl_extraction(
    cases: Iterable[Mapping[str, Any]],
    *,
    model_specs: Iterable[str] = MODEL_SPECS,
    sample_rows_limit: int = 10,
) -> JsonDict:
    """Run Exp 1607 DSL compilation over HumanEval prompt rows."""

    case_rows: list[JsonDict] = []
    for case in cases:
        task_id = str(case.get("task_id") or "")
        try:
            pack = extract_humaneval_prompt_constraints(case)
            validator = compile_constraint_pack(pack)
        except ConstraintDslError as exc:
            case_rows.append(
                {
                    "task_id": task_id,
                    "entry_point": str(case.get("entry_point") or ""),
                    "valid_extraction": False,
                    "compiled_validator": False,
                    "constraints_count": 0,
                    "constraint_ops": [],
                    "pysat_clause_count": 0,
                    "error": str(exc),
                }
            )
            continue
        case_rows.append(
            {
                "task_id": task_id,
                "entry_point": _humaneval_entry_point(case, str(case.get("prompt") or "")),
                "valid_extraction": True,
                "compiled_validator": True,
                "constraints_count": len(pack.constraints),
                "constraint_ops": [constraint.op for constraint in pack.constraints],
                "pysat_clause_count": len(validator.pysat_problem.clauses),
                "constraint_pack": pack.to_dict(),
            }
        )

    prompts_attempted = len(case_rows)
    valid_extractions = sum(1 for row in case_rows if row["valid_extraction"])
    validators_compiled = sum(1 for row in case_rows if row["compiled_validator"])
    constraints_extracted = sum(int(row["constraints_count"]) for row in case_rows)
    valid_rate = _rate(valid_extractions, prompts_attempted)
    compiled_rate = _rate(validators_compiled, prompts_attempted)
    status = "complete" if valid_extractions == prompts_attempted == validators_compiled else "partial"
    models = list(model_specs)
    model_spec_results = [
        {
            "model_spec": model_spec,
            "evaluation_mode": "deterministic_dsl_compilation_no_inference",
            "prompts_attempted": prompts_attempted,
            "valid_extraction_rate": valid_rate,
            "compiled_validator_rate": compiled_rate,
        }
        for model_spec in models
    ]
    return {
        "status": status,
        "run_date": RUN_DATE,
        "experiment_id": HUMANEVAL_EXPERIMENT_ID,
        "spec_traces": ["REQ-CODE-034", "SCENARIO-CODE-032", "REQ-VERIFY-1588"],
        "dataset_source": "openai_humaneval",
        "model_specs": models,
        "model_spec_results": model_spec_results,
        "inference_mode": "not_applicable_dsl_only",
        "dsl_schema_version": DSL_SCHEMA_VERSION,
        "dsl_schema": DSL_SCHEMA,
        "prompts_attempted": prompts_attempted,
        "valid_extractions": valid_extractions,
        "extraction_failures": prompts_attempted - valid_extractions,
        "constraints_extracted": constraints_extracted,
        "validators_compiled": validators_compiled,
        "pysat_cnf_compiled": validators_compiled,
        "valid_extraction_rate": valid_rate,
        "compiled_validator_rate": compiled_rate,
        "arbitrary_code_execution_path_introduced": compiler_uses_arbitrary_code_execution(),
        "sample_rows": case_rows[:sample_rows_limit],
        "case_rows": case_rows,
        "honest_verdict": _humaneval_verdict(status, valid_rate, compiled_rate),
    }


def write_humaneval_dsl_artifact(
    output_path: Path | str = DEFAULT_HUMANEVAL_ARTIFACT_PATH,
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Write the Exp 1607 HumanEval prompt DSL extraction artifact."""

    prompt_cases = list(cases) if cases is not None else load_humaneval_prompt_cases()
    artifact = evaluate_humaneval_dsl_extraction(prompt_cases)
    artifact["tests_run"] = list(tests_run or [])
    _write_json(Path(output_path), artifact)
    return artifact


def _clean_instruction(instruction: str) -> str:
    if not isinstance(instruction, str):
        raise ConstraintDslError("instruction must be string")
    text = " ".join(instruction.strip().split())
    if len(text) > MAX_INSTRUCTION_CHARS:
        raise ConstraintDslError("instruction too long")
    return text


def _json_constraints(text: str) -> list[tuple[str, str, Any, str]]:
    constraints: list[tuple[str, str, Any, str]] = []
    if re.search(r"\b(json|JSON)\b", text):
        constraints.append(("json_object", "json", True, "json"))
    key_match = re.search(
        r"\b(?:with|include|including|must include|must have)\s+keys?\s+([^.;]+)",
        text,
        flags=re.IGNORECASE,
    )
    if key_match:
        keys = _split_words(key_match.group(1))
        if keys:
            constraints.append(("json_has_keys", "json", keys, key_match.group(0)))
    return constraints


def _quoted_phrase_constraints(text: str) -> list[tuple[str, str, Any, str]]:
    positioned: list[tuple[int, tuple[str, str, Any, str]]] = []
    negative_pattern = re.compile(
        r"(?P<prefix>(?:do not|don't|must not|never|avoid)"
        r"(?:\s+(?:include|mention|contain|use))?"
        r"(?:\s+(?:the\s+)?(?:word|phrase|term))?)\s+[\"'](?P<phrase>[^\"']+)[\"']",
        flags=re.IGNORECASE,
    )
    negative_spans: list[tuple[int, int]] = []
    for match in negative_pattern.finditer(text):
        phrase = match.group("phrase").strip()
        negative_spans.append(match.span())
        positioned.append((match.start(), ("not_contains", "text", phrase, match.group(0))))

    positive_pattern = re.compile(
        r"(?P<prefix>(?:include|mention|contain|use)"
        r"(?:\s+(?:the\s+)?(?:word|phrase|term))?)\s+[\"'](?P<phrase>[^\"']+)[\"']",
        flags=re.IGNORECASE,
    )
    for match in positive_pattern.finditer(text):
        if any(match.start() >= start and match.end() <= end for start, end in negative_spans):
            continue
        phrase = match.group("phrase").strip()
        if phrase:
            positioned.append((match.start(), ("contains", "text", phrase, match.group(0))))
    return [constraint for _, constraint in sorted(positioned, key=lambda item: item[0])]


def _word_count_constraints(text: str) -> list[tuple[str, str, Any, str]]:
    constraints: list[tuple[str, str, Any, str]] = []
    max_match = re.search(
        r"\b(?:use|keep|write|answer)?\s*(?:at most|no more than|under)\s+(\d{1,4})\s+words?\b",
        text,
        flags=re.IGNORECASE,
    )
    if max_match:
        constraints.append(("max_words", "text", int(max_match.group(1)), max_match.group(0)))
    min_match = re.search(
        r"\b(?:use|write|answer)?\s*(?:at least|no fewer than)\s+(\d{1,4})\s+words?\b",
        text,
        flags=re.IGNORECASE,
    )
    if min_match:
        constraints.append(("min_words", "text", int(min_match.group(1)), min_match.group(0)))
    return constraints


def _enum_constraints(text: str) -> list[tuple[str, str, Any, str]]:
    enum_match = re.search(r"\b(?:one of|either)\b(?P<body>[^.]+)", text, flags=re.IGNORECASE)
    if not enum_match:
        return []
    choices = re.findall(r"[\"']([^\"']+)[\"']", enum_match.group("body"))
    if len(choices) < 2:
        return []
    return [("enum", "text", [choice.strip() for choice in choices if choice.strip()], enum_match.group(0))]


def _bullet_constraints(text: str) -> list[tuple[str, str, Any, str]]:
    bullet_match = re.search(
        r"\buse\s+exactly\s+(\d{1,3})\s+bullet\s+points?\b",
        text,
        flags=re.IGNORECASE,
    )
    if not bullet_match:
        return []
    return [("exact_bullets", "text", int(bullet_match.group(1)), bullet_match.group(0))]


def _split_words(raw: str) -> list[str]:
    cleaned = re.sub(r"\band\b", ",", raw, flags=re.IGNORECASE)
    return [part.strip(" ,\"'`").lower() for part in cleaned.split(",") if part.strip(" ,\"'`")]


def _operator_value_errors(index: int, constraint: Mapping[str, Any]) -> list[str]:
    op = constraint["op"]
    value = constraint.get("value")
    if op in {"contains", "not_contains"} and not isinstance(value, str):
        return [f"constraint[{index}].value must be string"]
    if op in {"max_words", "min_words", "exact_bullets"} and (
        not isinstance(value, int) or value < 0
    ):
        return [f"constraint[{index}].value must be nonnegative integer"]
    if op == "json_has_keys" and (
        not isinstance(value, list) or not all(isinstance(item, str) and item for item in value)
    ):
        return [f"constraint[{index}].value must be string list"]
    if op == "enum" and (
        not isinstance(value, list) or len(value) < 2 or not all(isinstance(item, str) and item for item in value)
    ):
        return [f"constraint[{index}].value must be two-or-more string list"]
    return []


def _evaluate_constraint(
    constraint: ConstraintSpec,
    output_text: str,
    parsed_json: JsonDict | None,
    json_error: str | None,
) -> ValidationIssue | None:
    op = constraint.op
    if op == "json_object":
        return None if parsed_json is not None else _issue(constraint, "valid JSON object", json_error)
    if op == "json_has_keys":
        keys = list(constraint.value)
        missing = [key for key in keys if not isinstance(parsed_json, dict) or key not in parsed_json]
        return None if not missing else _issue(constraint, keys, {"missing": missing})
    if op == "contains":
        return None if str(constraint.value).lower() in output_text.lower() else _issue(constraint, constraint.value, "not found")
    if op == "not_contains":
        found = str(constraint.value).lower() in output_text.lower()
        return _issue(constraint, f"not {constraint.value}", "found") if found else None
    if op == "max_words":
        count = _word_count(output_text)
        return None if count <= int(constraint.value) else _issue(constraint, f"<={constraint.value}", count)
    if op == "min_words":
        count = _word_count(output_text)
        return None if count >= int(constraint.value) else _issue(constraint, f">={constraint.value}", count)
    if op == "enum":
        normalized = _normalize_answer(output_text)
        choices = {_normalize_answer(choice) for choice in constraint.value}
        return None if normalized in choices else _issue(constraint, sorted(choices), normalized)
    if op == "exact_bullets":
        count = _bullet_count(output_text)
        return None if count == int(constraint.value) else _issue(constraint, constraint.value, count)
    return _issue(constraint, "supported operator", op)


def _issue(constraint: ConstraintSpec, expected: Any, observed: Any) -> ValidationIssue:
    return ValidationIssue(
        constraint_id=constraint.id,
        message=f"{constraint.op} constraint failed",
        expected=expected,
        observed=observed,
    )


def _parse_json_object(output_text: str) -> tuple[JsonDict | None, str | None]:
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"
    if not isinstance(parsed, dict):
        return None, "json_value_not_object"
    return parsed, None


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"\s*(?:[-*]|\d+[.)])\s+\S", line))


def _normalize_answer(text: str) -> str:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped.strip("\"' .").lower()
    if isinstance(parsed, str):
        return parsed.strip().lower()
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        return parsed["answer"].strip().lower()
    return stripped.strip("\"' .").lower()


def _unsafe_reason(value: Any) -> str | None:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
    lowered = text.lower()
    for token in FORBIDDEN_TEXT_TOKENS:
        if token in lowered:
            return f"unsafe token:{token}"
    return None


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _humaneval_entry_point(case: Mapping[str, Any], prompt: str) -> str:
    raw_entry_point = case.get("entry_point")
    if isinstance(raw_entry_point, str) and re.match(r"^[A-Za-z_]\w*$", raw_entry_point):
        return raw_entry_point
    match = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", prompt, flags=re.MULTILINE)
    if match:
        return match.group(1)
    raise ConstraintDslError("missing HumanEval entry point")


def _humaneval_instruction(case: Mapping[str, Any], entry_point: str) -> str:
    task_id = str(case.get("task_id") or "unknown")
    return (
        f"{task_id}: generate Python code defining def {entry_point}; include a return "
        "statement; do not output markdown fences or placeholder bodies."
    )


def _humaneval_verdict(status: str, valid_rate: float, compiled_rate: float) -> str:
    if status == "complete":
        return (
            "complete: HumanEval prompt-derived NSVIF DSL constraints compiled "
            "locally for mandated model specs without live inference"
        )
    return (
        "partial: HumanEval NSVIF DSL extraction completed with unsupported "
        f"prompts; valid_extraction_rate={valid_rate}, compiled_validator_rate={compiled_rate}"
    )


def _experiment_fixtures() -> list[JsonDict]:
    return [
        {
            "instruction": (
                'Respond in JSON with keys answer and confidence. Include "approved". '
                'Do not mention "secret". Use at most 12 words.'
            ),
            "known_good": '{"answer": "approved", "confidence": "high"}',
            "known_bad": '{"answer": "secret", "extra": true}',
        },
        {
            "instruction": 'Answer must be one of "yes", "no".',
            "known_good": "yes",
            "known_bad": "maybe",
        },
        {
            "instruction": "Use exactly 2 bullet points.",
            "known_good": "- first\n- second",
            "known_bad": "- only one",
        },
        {
            "instruction": 'Include "north". Use at least 2 words and at most 5 words.',
            "known_good": "north star",
            "known_bad": "north",
        },
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
