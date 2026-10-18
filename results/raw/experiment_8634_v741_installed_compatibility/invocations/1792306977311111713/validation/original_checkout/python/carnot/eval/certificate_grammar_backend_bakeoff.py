"""Exp 1283 local grammar backend bakeoff for Carnot certificates.

Spec: REQ-VERIFY-1283, SCENARIO-VERIFY-1283
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1283_certificate_grammar_backend_bakeoff.json"
)
EXPERIMENT_NAME = "1283_certificate_grammar_backend_bakeoff"
SCHEMA = "certificate_grammar_backend_bakeoff_v1"
RUN_DATE = "20260504"

REQUIRED_CERTIFICATE_FIELDS = [
    "claims",
    "equations",
    "final_answer",
    "confidence",
    "verifier_routes",
    "proof_numbers",
]


@dataclass(frozen=True)
class BackendDefinition:
    """Static description of one backend probe target.

    The bakeoff is a CPU-only preflight, so these definitions describe import
    and CLI surfaces that can be inspected without loading a model.  The
    `priority` value is the local-friction ranking used after availability is
    known: lower values are easier to wire into the current GGUF path.
    """

    name: str
    import_name: str | None
    cli_candidates: tuple[str, ...]
    schema_support: str
    unsupported_features: tuple[str, ...]
    constrained_generation: bool
    priority: int
    help_markers: tuple[str, ...] = ()


class ConstantStepTimer:
    """Deterministic timer used by tests for validation-overhead measurement."""

    def __init__(self, step: float = 0.001) -> None:
        self._current = 0.0
        self._step = float(step)

    def __call__(self) -> float:
        self._current += self._step
        return self._current


CONTEXT_SENSITIVE_UNSUPPORTED = (
    "claim id uniqueness across arrays",
    "every verifier route target must reference an emitted claim",
    "proof number ordering and equation-to-proof consistency",
    "confidence calibration against verifier energy",
    "truth of final_answer relative to the claims",
)

BACKEND_DEFINITIONS: tuple[BackendDefinition, ...] = (
    BackendDefinition(
        name="llama_cpp_gbnf",
        import_name="llama_cpp",
        cli_candidates=("llama-cli", "llama", "main"),
        schema_support="GBNF grammar support; JSON schema needs translation to bounded grammar",
        unsupported_features=CONTEXT_SENSITIVE_UNSUPPORTED,
        constrained_generation=True,
        priority=10,
        help_markers=("--grammar", "--grammar-file", "--grammar-json"),
    ),
    BackendDefinition(
        name="llguidance",
        import_name="llguidance",
        cli_candidates=("llguidance",),
        schema_support="JSON-schema/grammar guidance when package bindings are installed",
        unsupported_features=CONTEXT_SENSITIVE_UNSUPPORTED,
        constrained_generation=True,
        priority=20,
        help_markers=("grammar", "json"),
    ),
    BackendDefinition(
        name="xgrammar",
        import_name="xgrammar",
        cli_candidates=("xgrammar",),
        schema_support="JSON-schema grammar compilation when xgrammar is importable",
        unsupported_features=CONTEXT_SENSITIVE_UNSUPPORTED,
        constrained_generation=True,
        priority=30,
        help_markers=("grammar", "json"),
    ),
    BackendDefinition(
        name="outlines",
        import_name="outlines",
        cli_candidates=("outlines",),
        schema_support="JSON-schema structured generation through Python API",
        unsupported_features=CONTEXT_SENSITIVE_UNSUPPORTED,
        constrained_generation=True,
        priority=40,
        help_markers=("json", "schema"),
    ),
    BackendDefinition(
        name="lm_format_enforcer",
        import_name="lmformatenforcer",
        cli_candidates=("lm-format-enforcer",),
        schema_support="JSON-schema parser/logits processor through Python API",
        unsupported_features=CONTEXT_SENSITIVE_UNSUPPORTED,
        constrained_generation=True,
        priority=50,
        help_markers=("json", "schema"),
    ),
    BackendDefinition(
        name="pure_python_validation",
        import_name=None,
        cli_candidates=(),
        schema_support="post-hoc validation of the bounded JSON schema subset",
        unsupported_features=("token-level constrained generation",) + CONTEXT_SENSITIVE_UNSUPPORTED,
        constrained_generation=False,
        priority=100,
    ),
)


def certificate_schema() -> dict[str, Any]:
    """Return the bounded minimal Carnot certificate JSON schema."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Minimal Carnot Certificate",
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_CERTIFICATE_FIELDS),
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "pattern": "^c[0-9]+$",
                            "minLength": 2,
                            "maxLength": 16,
                        },
                        "text": {"type": "string", "minLength": 1, "maxLength": 320},
                    },
                },
            },
            "equations": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["lhs", "relation", "rhs"],
                    "properties": {
                        "lhs": {"type": "string", "minLength": 1, "maxLength": 120},
                        "relation": {"type": "string", "enum": ["=", "!=", "<=", ">="]},
                        "rhs": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                },
            },
            "final_answer": {"type": "string", "minLength": 1, "maxLength": 160},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "verifier_routes": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_id", "verifier"],
                    "properties": {
                        "claim_id": {
                            "type": "string",
                            "pattern": "^c[0-9]+$",
                            "minLength": 2,
                            "maxLength": 16,
                        },
                        "verifier": {
                            "type": "string",
                            "enum": [
                                "z3_math",
                                "symcode",
                                "causal",
                                "semenergy",
                                "soskan",
                            ],
                        },
                    },
                },
            },
            "proof_numbers": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "number"},
            },
        },
    }


def sample_certificate() -> dict[str, Any]:
    """Return a tiny certificate that exercises every schema field."""

    return {
        "claims": [{"id": "c1", "text": "2 + 2 equals 4."}],
        "equations": [{"lhs": "2 + 2", "relation": "=", "rhs": "4"}],
        "final_answer": "4",
        "confidence": 0.91,
        "verifier_routes": [{"claim_id": "c1", "verifier": "z3_math"}],
        "proof_numbers": [4],
    }


def validate_certificate(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Validate the bounded schema subset needed by the fallback path."""

    schema = schema or certificate_schema()
    errors: list[str] = []
    for field in schema.get("required", []):
        if field not in payload:
            errors.append(f"missing {field}")

    properties = schema.get("properties", {})
    for field, spec in properties.items():
        if field not in payload:
            continue
        errors.extend(_validate_value(field, payload[field], spec))
    return not errors, errors


def _validate_value(path: str, value: Any, spec: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_type = spec.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            return [f"{path} must be object"]
        for field in spec.get("required", []):
            if field not in value:
                errors.append(f"missing {path}.{field}")
        for field, child_spec in spec.get("properties", {}).items():
            if field in value:
                errors.extend(_validate_value(f"{path}.{field}", value[field], child_spec))
    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path} must be array"]
        min_items = spec.get("minItems")
        max_items = spec.get("maxItems")
        if min_items is not None and len(value) < int(min_items):
            errors.append(f"{path} must contain at least {min_items} items")
        if max_items is not None and len(value) > int(max_items):
            errors.append(f"{path} must contain at most {max_items} items")
        child_spec = spec.get("items")
        if isinstance(child_spec, Mapping):
            for index, item in enumerate(value):
                errors.extend(_validate_value(f"{path}[{index}]", item, child_spec))
    elif expected_type == "string":
        if not isinstance(value, str):
            return [f"{path} must be string"]
        enum_values = spec.get("enum")
        if enum_values is not None and value not in enum_values:
            errors.append(f"{path} must be one of {list(enum_values)}")
        min_length = spec.get("minLength")
        max_length = spec.get("maxLength")
        if min_length is not None and len(value) < int(min_length):
            errors.append(f"{path} is too short")
        if max_length is not None and len(value) > int(max_length):
            errors.append(f"{path} is too long")
        pattern = spec.get("pattern")
        if pattern is not None and re.fullmatch(str(pattern), value) is None:
            errors.append(f"{path} does not match pattern {pattern}")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [f"{path} must be number"]
        number = float(value)
        if not math.isfinite(number):
            errors.append(f"{path} must be finite")
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and number < float(minimum):
            errors.append(f"{path} below minimum {minimum}")
        if maximum is not None and number > float(maximum):
            errors.append(f"{path} above maximum {maximum}")
    return errors


def bounded_vocab_constraint_count(schema: Mapping[str, Any]) -> int:
    """Count enum vocabulary entries in the schema for the STATIC note."""

    total = 0
    enum_values = schema.get("enum")
    if isinstance(enum_values, Sequence) and not isinstance(enum_values, str):
        total += len(enum_values)
    for child in schema.get("properties", {}).values():
        if isinstance(child, Mapping):
            total += bounded_vocab_constraint_count(child)
    item_spec = schema.get("items")
    if isinstance(item_spec, Mapping):
        total += bounded_vocab_constraint_count(item_spec)
    return total


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _find_cli(name: str) -> str | None:
    return shutil.which(name)


def _help_text(path: str) -> str:
    try:
        result = subprocess.run(
            [path, "--help"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{result.stdout}\n{result.stderr}"


def probe_backends(
    definitions: Sequence[BackendDefinition] = BACKEND_DEFINITIONS,
    *,
    import_checker: Callable[[str], bool] = _module_available,
    cli_finder: Callable[[str], str | None] = _find_cli,
    help_runner: Callable[[str], str] = _help_text,
    overhead_timer: Callable[[], float] = time.perf_counter,
) -> list[dict[str, Any]]:
    """Probe every backend surface without loading a model or running inference."""

    records: list[dict[str, Any]] = []
    for definition in definitions:
        import_available = (
            True if definition.import_name is None else import_checker(definition.import_name)
        )
        cli_path = _first_cli_path(definition.cli_candidates, cli_finder)
        help_text = help_runner(cli_path) if cli_path and definition.help_markers else ""
        cli_supports_grammar = bool(
            cli_path
            and (
                not definition.help_markers
                or any(marker.lower() in help_text.lower() for marker in definition.help_markers)
            )
        )
        available = (
            import_available if not definition.cli_candidates else import_available or bool(cli_path)
        )
        records.append(
            {
                "name": definition.name,
                "import_name": definition.import_name,
                "import_available": import_available,
                "cli_candidates": list(definition.cli_candidates),
                "cli_path": cli_path,
                "cli_available": cli_path is not None,
                "cli_supports_grammar": cli_supports_grammar,
                "schema_support": definition.schema_support,
                "unsupported_features": list(definition.unsupported_features),
                "constrained_generation": definition.constrained_generation,
                "priority": definition.priority,
                "available": available,
                "estimated_overhead": _estimated_overhead(definition, overhead_timer),
                "failure_reason": _failure_reason(definition, import_available, cli_path),
            }
        )
    return records


def _first_cli_path(
    candidates: Sequence[str],
    cli_finder: Callable[[str], str | None],
) -> str | None:
    for candidate in candidates:
        path = cli_finder(candidate)
        if path:
            return path
    return None


def _estimated_overhead(
    definition: BackendDefinition,
    overhead_timer: Callable[[], float],
) -> str | dict[str, float]:
    if definition.name != "pure_python_validation":
        return "not_measured_no_model_inference"
    sample = sample_certificate()
    schema = certificate_schema()
    start = overhead_timer()
    for _ in range(1000):
        validate_certificate(sample, schema)
    elapsed_ms = (overhead_timer() - start) * 1000.0
    return {"validation_1000_docs_ms": round(elapsed_ms, 6)}


def _failure_reason(
    definition: BackendDefinition,
    import_available: bool,
    cli_path: str | None,
) -> str | None:
    if definition.name == "pure_python_validation":
        return None
    if import_available or cli_path:
        return None
    if definition.import_name and definition.cli_candidates:
        return "import_and_cli_absent"
    if definition.import_name:
        return "import_absent"
    return "cli_absent"


def select_backend(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the lowest-friction constrained backend or the validation fallback."""

    available_generation = [
        record
        for record in records
        if record.get("available") and record.get("constrained_generation")
    ]
    if available_generation:
        selected = min(available_generation, key=lambda record: int(record.get("priority", 999)))
        return {
            "name": str(selected["name"]),
            "grammar_backend_available": True,
            "fallback_only": False,
        }

    fallback = next(
        (
            record
            for record in records
            if record.get("name") == "pure_python_validation" and record.get("available")
        ),
        None,
    )
    if fallback is not None:
        return {
            "name": "pure_python_validation",
            "grammar_backend_available": False,
            "fallback_only": True,
        }
    return {"name": "none", "grammar_backend_available": False, "fallback_only": True}


def build_bakeoff_artifact(
    backend_records: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any] | None = None,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the completed Exp 1283 artifact from probe records."""

    schema = dict(schema or certificate_schema())
    selected = select_backend(backend_records)
    grammar_available = bool(selected["grammar_backend_available"])
    status = "complete" if grammar_available else "blocked"
    honest_verdict = (
        f"selected_{selected['name']}"
        if grammar_available
        else "blocked_no_local_constrained_generation_backend_pure_python_validation_only"
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": status,
        "llm_inference_run": False,
        "certificate_schema": schema,
        "certificate_schema_required_fields": list(REQUIRED_CERTIFICATE_FIELDS),
        "backend_probes": {str(record["name"]): dict(record) for record in backend_records},
        "grammar_backend_available": grammar_available,
        "grammar_backend_selected": str(selected["name"]),
        "selected_backend_is_fallback_only": bool(selected["fallback_only"]),
        "cdot_expressiveness_note": (
            "CDoT-style context-sensitive obligations remain outside the selected "
            "syntax backend: route targets must match emitted claim ids, proof "
            "numbers must stay consistent with equations, and confidence must be "
            "calibrated against verifier evidence after decoding."
        ),
        "static_trie_note": (
            "The bounded enum vocabularies for verifier routes and equation relations "
            "are small enough for STATIC-style trie or vectorized token-mask handling; "
            "free-text claim and answer strings remain open vocabulary."
        ),
        "bounded_vocab_constraint_count": bounded_vocab_constraint_count(schema),
        "bounded_vocab_constraints": {
            "equation_relations": ["=", "!=", "<=", ">="],
            "verifier_routes": ["z3_math", "symcode", "causal", "semenergy", "soskan"],
        },
        "automata_fallback_viable": True,
        "dfa_checkable_fields": list(REQUIRED_CERTIFICATE_FIELDS),
        "structure_snowballing_risk": "medium",
        "structure_snowballing_risk_reason": (
            "Nested claims, equations, and routes can inflate grammar states as maxItems "
            "rises; the v1 schema keeps every repeated field capped at eight items."
        ),
        "honest_verdict": honest_verdict,
    }


def write_in_progress_artifact(path: Path | str, *, run_date: str = RUN_DATE) -> dict[str, Any]:
    """Write the required bootstrap artifact before probing starts."""

    artifact = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
    }
    _write_json(Path(path), artifact)
    return artifact


def run_bakeoff(
    *,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    run_date: str = RUN_DATE,
    import_checker: Callable[[str], bool] = _module_available,
    cli_finder: Callable[[str], str | None] = _find_cli,
    help_runner: Callable[[str], str] = _help_text,
    overhead_timer: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Probe local grammar backends and write the stable Exp 1283 artifact."""

    output = Path(output_path)
    write_in_progress_artifact(output, run_date=run_date)
    schema = certificate_schema()
    records = probe_backends(
        import_checker=import_checker,
        cli_finder=cli_finder,
        help_runner=help_runner,
        overhead_timer=overhead_timer,
    )
    artifact = build_bakeoff_artifact(records, schema=schema, run_date=run_date)
    _write_json(output, artifact)
    return artifact


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
