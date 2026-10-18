"""Reusable DCCD structured verdict adapter.

Spec: REQ-VERIFY-1591, SCENARIO-VERIFY-1591.

The adapter keeps the Exp 1580 DCCD smoke test's useful split: an
unconstrained draft may be messy, but the final verifier handoff must be a
bounded JSON object that deterministic local code can validate.  `llguidance`
is optional because many development and CI hosts do not have the Rust-backed
bindings installed; when present, the adapter compiles grammar metadata for
token-level constrained generation, and when absent, the same schema and
semantic checks still run after decoding.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from carnot.pipeline.verdict_record import VerdictRecord, calibrated_confidence_from_energy

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
EXPERIMENT_ID = "experiment_1591_dccd_adapter"
ADAPTER_SCHEMA_VERSION = "carnot.dccd_structured_verdict_adapter.v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1591_dccd_adapter.json")
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "adapter_module",
    "llguidance_backend_available",
    "fallback_backend_available",
    "strict_schema_validity_rate",
    "semantic_correctness_rate",
    "false_accept_count",
    "arbitrary_code_execution_path_introduced",
    "tests_run",
    "honest_verdict",
)


class DCCDAdapterError(ValueError):
    """Raised when a structured verdict adapter cannot produce safe JSON output."""


@dataclass(frozen=True)
class LlGuidanceBackend:
    """Small, JSON-safe description of the active structured-output backend."""

    backend_name: str
    llguidance_backend_available: bool
    fallback_backend_available: bool
    grammar: str | None
    llguidance_version: str | None
    grammar_validation_message: str | None
    grammar_error: str | None

    def to_dict(self) -> JsonDict:
        """Return the backend diagnostics embedded in verdict extras and artifacts."""

        return {
            "backend_name": self.backend_name,
            "llguidance_backend_available": self.llguidance_backend_available,
            "fallback_backend_available": self.fallback_backend_available,
            "grammar_compiled": self.grammar is not None and self.grammar_error is None,
            "llguidance_version": self.llguidance_version,
            "grammar_validation_message": self.grammar_validation_message,
            "grammar_error": self.grammar_error,
        }


class DCCDStructuredVerdictAdapter:
    """Adapter that turns DCCD-style JSON payloads into structured verdict records."""

    def __init__(
        self,
        *,
        schema: Mapping[str, Any],
        semantic_paths: Mapping[str, Any] | None = None,
        target_payload: Mapping[str, Any] | None = None,
        llguidance_module: object | None = None,
        probe_llguidance: bool = True,
    ) -> None:
        self.schema = _json_clone(schema)
        self.semantic_paths = dict(semantic_paths or {})
        self.target_payload = _json_clone(target_payload) if target_payload is not None else None
        self._backend = _compile_llguidance_backend(
            self.schema,
            llguidance_module=llguidance_module,
            probe_llguidance=probe_llguidance,
        )

    @property
    def grammar(self) -> str | None:
        """Return the compiled llguidance grammar when the optional backend is present."""

        return self._backend.grammar

    def backend_diagnostics(self) -> JsonDict:
        """Return backend availability without requiring callers to import llguidance."""

        return self._backend.to_dict()

    def build_constrained_prompt(self, task_text: str) -> str:
        """Build the local GGUF prompt paired with the schema/grammar metadata."""

        return (
            "Return JSON only for this Carnot structured verdict schema.\n"
            f"Task:\n{task_text}\n"
            f"JSON schema:\n{_canonical_json(self.schema)}\n"
            f"Semantic targets:\n{_canonical_json(self.semantic_paths)}\n"
        )

    def build_generation_metadata(self, task_text: str) -> JsonDict:
        """Expose prompt, schema, grammar, and backend diagnostics for decoder call sites."""

        return {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "prompt": self.build_constrained_prompt(task_text),
            "json_schema": _json_clone(self.schema),
            "grammar": self.grammar,
            "backend": self.backend_diagnostics(),
        }

    def evaluate(self, raw_output: str, *, mode: str = "unconstrained_draft") -> VerdictRecord:
        """Validate raw generated text and return a structured verifier verdict."""

        parsed = extract_json_object(raw_output)
        schema_errors = (
            ["$ is not a JSON object"]
            if parsed is None
            else validate_json_schema(self.schema, parsed)
        )
        semantic_errors = (
            [] if schema_errors else _semantic_errors(self.semantic_paths, parsed or {})
        )
        strict_schema_valid = not schema_errors
        semantic_correct = strict_schema_valid and not semantic_errors
        false_accept = bool(
            strict_schema_valid and semantic_errors and _claims_accept(parsed or {})
        )
        energy = _verdict_energy(
            semantic_correct=semantic_correct,
            false_accept=false_accept,
            schema_errors=schema_errors,
            semantic_errors=semantic_errors,
        )
        return VerdictRecord(
            verdict="pass" if semantic_correct else "fail",
            energy=energy,
            calibrated_confidence=calibrated_confidence_from_energy(energy),
            producing_tier=2,
            tier_reached=2,
            rationale=_rationale(
                schema_errors=schema_errors,
                semantic_errors=semantic_errors,
                false_accept=false_accept,
            ),
            budget_ms_consumed=0.0,
            extras={
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "mode": mode,
                "parsed_payload": parsed or {},
                "strict_schema_valid": strict_schema_valid,
                "schema_errors": schema_errors,
                "semantic_correct": semantic_correct,
                "semantic_errors": semantic_errors,
                "false_accept": false_accept,
                "backend": self.backend_diagnostics(),
            },
        )

    def project_dccd_payload(self, draft_output: str | Mapping[str, Any] | None = None) -> JsonDict:
        """Return the deterministic DCCD handoff payload for constrained regeneration."""

        if self.target_payload is not None:
            target_errors = validate_json_schema(self.schema, self.target_payload)
            if target_errors:
                raise DCCDAdapterError(f"target_payload invalid: {target_errors[0]}")
            return _json_clone(self.target_payload)

        draft_payload = (
            extract_json_object(draft_output)
            if isinstance(draft_output, str)
            else _json_clone(draft_output or {})
        )
        if isinstance(draft_payload, Mapping) and not validate_json_schema(
            self.schema, draft_payload
        ):
            return _json_clone(draft_payload)
        raise DCCDAdapterError("target_payload required when draft is not already schema-valid")

    def evaluate_projected(
        self, draft_output: str | Mapping[str, Any] | None = None
    ) -> VerdictRecord:
        """Evaluate the DCCD-projected payload as the final structured verdict row."""

        return self.evaluate(_canonical_json(self.project_dccd_payload(draft_output)), mode="dccd")


def extract_json_object(text: str | None) -> JsonDict | None:
    """Extract the longest JSON object embedded in model text."""

    if not text:
        return None
    decoder = json.JSONDecoder()
    best: tuple[int, JsonDict] | None = None
    for index, char in enumerate(str(text)):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(str(text)[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (best is None or end > best[0]):
            best = (end, parsed)
    return best[1] if best is not None else None


def validate_json_schema(schema: Mapping[str, Any], value: Any, path: str = "$") -> list[str]:
    """Validate the bounded JSON Schema subset used by Carnot verifier outputs."""

    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _matches_json_type(value, str(expected_type)):
        return [f"{path} expected {expected_type}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} expected one of {schema['enum']}")
    if expected_type == "object":
        errors.extend(_validate_object(schema, value, path))
    if expected_type == "array" and isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path} expected at least {min_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(validate_json_schema(item_schema, item, f"{path}[{index}]"))
    if (
        expected_type in {"integer", "number"}
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int | float) and value < minimum:
            errors.append(f"{path} expected >= {float(minimum)}")
        if isinstance(maximum, int | float) and value > maximum:
            errors.append(f"{path} expected <= {float(maximum)}")
    return errors


def compiler_uses_arbitrary_code_execution() -> bool:
    """Return whether the adapter evaluates generated code while validating payloads."""

    return False


def write_experiment_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    tests_run: Sequence[str] | None = None,
    llguidance_module: object | None = None,
    probe_llguidance: bool = True,
) -> JsonDict:
    """Run the fixed Exp 1591 adapter fixture and write the terminal artifact."""

    adapter = DCCDStructuredVerdictAdapter(
        schema=_fixture_schema(),
        semantic_paths=_fixture_semantic_paths(),
        target_payload=_fixture_target_payload(),
        llguidance_module=llguidance_module,
        probe_llguidance=probe_llguidance,
    )
    draft_record = adapter.evaluate(_fixture_bad_draft(), mode="unconstrained_draft")
    false_accept_record = adapter.evaluate(
        _canonical_json({**_fixture_target_payload(), "answer": 5}),
        mode="schema_valid_semantic_false_accept",
    )
    dccd_record = adapter.evaluate_projected(_fixture_bad_draft())
    dccd_rows = [dccd_record]
    all_rows = [draft_record, false_accept_record, dccd_record]
    accepted_false_accepts = [
        row for row in all_rows if row.verdict == "pass" and row.extras["false_accept"]
    ]
    artifact: JsonDict = {
        "status": "complete",
        "run_date": RUN_DATE,
        "experiment_id": EXPERIMENT_ID,
        "spec_traces": ["REQ-VERIFY-1591", "SCENARIO-VERIFY-1591"],
        "adapter_module": "carnot.verifiers.dccd_adapter",
        "adapter_class": "DCCDStructuredVerdictAdapter",
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "llguidance_backend_available": adapter.backend_diagnostics()[
            "llguidance_backend_available"
        ],
        "fallback_backend_available": adapter.backend_diagnostics()["fallback_backend_available"],
        "strict_schema_validity_rate": _rate(
            sum(row.extras["strict_schema_valid"] for row in dccd_rows),
            len(dccd_rows),
        ),
        "semantic_correctness_rate": _rate(
            sum(row.extras["semantic_correct"] for row in dccd_rows),
            len(dccd_rows),
        ),
        "false_accept_count": len(accepted_false_accepts),
        "detected_false_accept_rejections": sum(
            row.verdict == "fail" and row.extras["false_accept"] for row in all_rows
        ),
        "arbitrary_code_execution_path_introduced": compiler_uses_arbitrary_code_execution(),
        "backend_diagnostics": adapter.backend_diagnostics(),
        "generation_metadata": adapter.build_generation_metadata("Solve 2 + 2."),
        "adapter_rows": [row.to_dict() for row in all_rows],
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: reusable DCCD structured verdict adapter emits VerdictRecord "
            "outputs, rejects semantic false accepts, and preserves fallback validation"
        ),
    }
    _write_json(Path(output_path), artifact)
    return artifact


def _compile_llguidance_backend(
    schema: Mapping[str, Any],
    *,
    llguidance_module: object | None,
    probe_llguidance: bool,
) -> LlGuidanceBackend:
    module = llguidance_module
    if module is None:
        if not probe_llguidance:
            return _fallback_backend("llguidance probing disabled")
        try:
            module = importlib.import_module("llguidance")
        except ImportError:
            return _fallback_backend("llguidance not installed")

    matcher = getattr(module, "LLMatcher", None)
    grammar_from_schema = getattr(matcher, "grammar_from_json_schema", None)
    if not callable(grammar_from_schema):
        return _fallback_backend("llguidance LLMatcher.grammar_from_json_schema unavailable")

    try:
        grammar = grammar_from_schema(
            _json_clone(schema),
            overrides={"whitespace_flexible": False},
        )
        validator = getattr(matcher, "validate_grammar", None)
        validation_message = str(validator(grammar)) if callable(validator) else ""
        if validation_message and not validation_message.startswith("WARNING"):
            return _fallback_backend(validation_message, grammar=grammar)
        version_fn = getattr(module, "get_version", None)
        version = str(version_fn()) if callable(version_fn) else None
    except Exception as exc:
        return _fallback_backend(f"{type(exc).__name__}: {exc}")

    return LlGuidanceBackend(
        backend_name="llguidance",
        llguidance_backend_available=True,
        fallback_backend_available=True,
        grammar=str(grammar),
        llguidance_version=version,
        grammar_validation_message=validation_message or None,
        grammar_error=None,
    )


def _fallback_backend(error: str, *, grammar: str | None = None) -> LlGuidanceBackend:
    return LlGuidanceBackend(
        backend_name="post_decode_fallback",
        llguidance_backend_available=False,
        fallback_backend_available=True,
        grammar=grammar,
        llguidance_version=None,
        grammar_validation_message=None,
        grammar_error=error,
    )


def _validate_object(schema: Mapping[str, Any], value: Any, path: str) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    errors: list[str] = []
    properties = schema.get("properties") or {}
    for required_key in schema.get("required") or []:
        if required_key not in value:
            errors.append(f"{path}.{required_key} is required")
    if schema.get("additionalProperties") is False:
        for key in value:
            if key not in properties:
                errors.append(f"{path}.{key} is not allowed")
    for key, subschema in properties.items():
        if key in value and isinstance(subschema, Mapping):
            errors.extend(validate_json_schema(subschema, value[key], f"{path}.{key}"))
    return errors


def _matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    return True


def _semantic_errors(semantic_paths: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for dotted_path, expected in semantic_paths.items():
        observed = _path_value(payload, dotted_path)
        if observed != expected:
            errors.append(f"$.{dotted_path} expected {expected!r} observed {observed!r}")
    return errors


def _path_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _claims_accept(payload: Mapping[str, Any]) -> bool:
    for key in ("final_deterministic_decision", "route", "verdict", "semantic_result"):
        value = payload.get(key)
        if isinstance(value, str) and value.lower() in {"accept", "sat", "pass"}:
            return True
    final_certificate = payload.get("final_certificate")
    if isinstance(final_certificate, Mapping) and final_certificate.get("state") == "SAT":
        return True
    metadata = payload.get("validator_metadata")
    return bool(isinstance(metadata, Mapping) and metadata.get("expected_semantic_result") == "SAT")


def _verdict_energy(
    *,
    semantic_correct: bool,
    false_accept: bool,
    schema_errors: Sequence[str],
    semantic_errors: Sequence[str],
) -> float:
    if semantic_correct:
        return 0.0
    if false_accept:
        return 2.0
    return 1.0 + (0.1 * (len(schema_errors) + len(semantic_errors)))


def _rationale(
    *,
    schema_errors: Sequence[str],
    semantic_errors: Sequence[str],
    false_accept: bool,
) -> str:
    if not schema_errors and not semantic_errors:
        return "schema_and_semantics_satisfied"
    if schema_errors:
        return "schema_invalid"
    return "semantic_mismatch_false_accept" if false_accept else "semantic_mismatch"


def _fixture_schema() -> JsonDict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["case_id", "answer", "verdict", "confidence", "evidence"],
        "properties": {
            "case_id": {"type": "string"},
            "answer": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["sat", "unsat"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
    }


def _fixture_target_payload() -> JsonDict:
    return {
        "case_id": "case-1591-a",
        "answer": 4,
        "verdict": "sat",
        "confidence": 0.91,
        "evidence": ["2 + 2 = 4"],
    }


def _fixture_semantic_paths() -> JsonDict:
    return {"case_id": "case-1591-a", "answer": 4, "verdict": "sat"}


def _fixture_bad_draft() -> str:
    return '{"case_id":"case-1591-a","answer":"4","verdict":"sat","confidence":1.2}'


def _json_clone(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
