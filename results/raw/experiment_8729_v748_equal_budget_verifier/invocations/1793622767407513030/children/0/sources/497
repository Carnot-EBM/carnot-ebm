"""Exp 2951 structured candidate manifest adapter for code-repair rows.

The adapter turns a code-repair candidate into a bounded JSON object before any
verifier reads it.  This matters because the .277 failure pattern was not just
"bad code"; it was unbounded prose that later tools had to scrape, classify,
and repair.  The schema here makes the repair candidate, parser state, test
state, taxonomy, and provenance checksums explicit at the manifest boundary.

Spec: REQ-CODE-2951, SCENARIO-CODE-2951.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]
Importer = Callable[[str], object]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_FILENAME = "experiment_2951_structured_candidate_manifest_adapter_v1.json"
ARTIFACT = "experiment_2951_structured_candidate_manifest_adapter_v1"
SCHEMA_VERSION = "carnot.structured_candidate_manifest.v1"
INFERENCE_SUBSTRATE = "deterministic_wiring"
SPEC_TRACES = ["REQ-CODE-2951", "SCENARIO-CODE-2951"]

CANDIDATE_SCHEMA_FIELDS = (
    "task_id",
    "prompt_id",
    "model_id",
    "raw_completion_ref",
    "repaired_code",
    "failure_taxonomy",
    "parser_status",
    "test_status",
    "verifier_score",
    "provenance_checksums",
)

FAILURE_TAXONOMY_LABELS = (
    "none",
    "syntax_error",
    "unsupported_import",
    "unsupported_api_hallucination",
    "failed_tests",
)
PARSER_STATUSES = ("parsed", "syntax_error")
TEST_STATUSES = ("passed", "failed", "not_run")
CHECKSUM_FIELDS = (
    "raw_completion_sha256",
    "repaired_code_sha256",
    "manifest_schema_sha256",
)

MODEL_SPECS = (
    {
        "name": "Qwen3.6-35B-A3B-GGUF",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "downstream_live_structured_repair_candidate_generation",
    },
    {
        "name": "gemma-4-31B-it-GGUF",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "downstream_live_structured_repair_candidate_generation",
    },
    {
        "name": "gemma-4-26B-A4B-it-GGUF",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "downstream_live_structured_repair_candidate_generation",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "structured_decode_manifest_ready",
    "schema_version",
    "schema_fields",
    "local_backends_checked",
    "llguidance_available",
    "llama_cpp_grammar_available",
    "validation_fixture_count",
    "validation_fixture_passed",
    "model_specs_for_downstream_live_use",
    "inference_substrate",
    "duration_s",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationResult:
    """Deterministic schema-validation result for a single candidate manifest."""

    ok: bool
    errors: list[str]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for writing the Exp 2951 deterministic artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


class StructuredCandidateManifestAdapter:
    """Validate candidate manifests and expose downstream grammar metadata.

    The class deliberately has no model-call method.  Live tasks can use the
    emitted schema with llguidance or llama.cpp grammar support, but this
    adapter's local contract is deterministic validation of the resulting JSON.
    """

    def __init__(
        self,
        *,
        schema: Mapping[str, Any] | None = None,
        local_backends: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.schema = _json_clone(candidate_manifest_schema() if schema is None else schema)
        self.local_backends = [dict(row) for row in (local_backends or [])]

    def validate_record(self, record: Mapping[str, Any]) -> ValidationResult:
        errors = _validate_schema(record, self.schema, "$")
        return ValidationResult(ok=not errors, errors=errors)

    def validate_records(self, records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        results: list[JsonDict] = []
        for record in records:
            validation = self.validate_record(record)
            results.append(
                {
                    "fixture_id": str(record.get("task_id", "unknown")),
                    "schema_valid": validation.ok,
                    "errors": validation.errors,
                    "parser_status": record.get("parser_status"),
                    "test_status": record.get("test_status"),
                    "failure_taxonomy": list(record.get("failure_taxonomy", [])),
                }
            )
        return results

    def generation_metadata(self) -> JsonDict:
        backend = preferred_backend(self.local_backends)
        return {
            "preferred_backend": backend,
            "json_schema": _json_clone(self.schema),
            "llama_cpp_kwargs": (
                {"grammar_source": "json_schema", "temperature": 0.0}
                if backend == "llama_cpp_grammar"
                else {"response_format": {"type": "json_object"}, "temperature": 0.0}
            ),
            "post_decode_fallback": "deterministic_schema_validation",
            "live_llm_call_required": False,
        }


def candidate_manifest_schema() -> JsonDict:
    """Return the reusable JSON-schema-shaped contract for a candidate row."""

    checksum_schema: JsonDict = {
        "type": "string",
        "pattern": "^[0-9a-f]{64}$",
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(CANDIDATE_SCHEMA_FIELDS),
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "prompt_id": {"type": "string", "minLength": 1},
            "model_id": {"type": "string", "minLength": 1},
            "raw_completion_ref": {"type": "string", "minLength": 1},
            "repaired_code": {"type": "string", "minLength": 1},
            "failure_taxonomy": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": list(FAILURE_TAXONOMY_LABELS)},
            },
            "parser_status": {"type": "string", "enum": list(PARSER_STATUSES)},
            "test_status": {"type": "string", "enum": list(TEST_STATUSES)},
            "verifier_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "provenance_checksums": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CHECKSUM_FIELDS),
                "properties": {field: checksum_schema for field in CHECKSUM_FIELDS},
            },
        },
    }


def synthetic_candidate_records() -> list[JsonDict]:
    """Return three deterministic schema-valid rows for the .278 repair adapter."""

    return [
        _candidate_record(
            task_id="valid_candidate",
            prompt_id="exp2951-fixture-valid",
            model_id="unsloth/Qwen3.6-35B-A3B-GGUF",
            raw_completion_ref="results/raw/experiment_2951/valid_candidate.txt",
            raw_completion_text="def add(a, b):\n    return a + b\n",
            repaired_code="def add(a, b):\n    return a + b\n",
            failure_taxonomy=["none"],
            parser_status="parsed",
            test_status="passed",
            verifier_score=1.0,
        ),
        _candidate_record(
            task_id="syntax_failure",
            prompt_id="exp2951-fixture-syntax",
            model_id="unsloth/gemma-4-31B-it-GGUF",
            raw_completion_ref="results/raw/experiment_2951/syntax_failure.txt",
            raw_completion_text="def broken(:\n",
            repaired_code="def broken(:\n",
            failure_taxonomy=["syntax_error"],
            parser_status="syntax_error",
            test_status="not_run",
            verifier_score=0.0,
        ),
        _candidate_record(
            task_id="unsupported_import_api_hallucination",
            prompt_id="exp2951-fixture-api",
            model_id="unsloth/gemma-4-26B-A4B-it-GGUF",
            raw_completion_ref="results/raw/experiment_2951/unsupported_api.txt",
            raw_completion_text="import magic_json\ndef parse(x):\n    return magic_json.parsefast(x)\n",
            repaired_code="import magic_json\ndef parse(x):\n    return magic_json.parsefast(x)\n",
            failure_taxonomy=["unsupported_import", "unsupported_api_hallucination"],
            parser_status="parsed",
            test_status="failed",
            verifier_score=0.0,
        ),
    ]


def probe_local_backends(importer: Importer = importlib.import_module) -> list[JsonDict]:
    """Check optional local structured-output backends without installing packages."""

    jsonschema_module, jsonschema_detail = _import_optional("jsonschema", importer)
    llguidance_module, llguidance_detail = _import_optional("llguidance", importer)
    llama_cpp_module, llama_cpp_detail = _import_optional("llama_cpp", importer)

    llguidance_available = _has_llguidance_json_schema(llguidance_module)
    llama_cpp_grammar_available = _has_llama_cpp_json_schema_grammar(llama_cpp_module)
    return [
        {
            "backend_name": "jsonschema",
            "available": jsonschema_module is not None,
            "detail": jsonschema_detail,
        },
        {
            "backend_name": "llguidance",
            "available": llguidance_available,
            "detail": (
                "LLMatcher.grammar_from_json_schema available"
                if llguidance_available
                else llguidance_detail
            ),
        },
        {
            "backend_name": "llama_cpp_grammar",
            "available": llama_cpp_grammar_available,
            "detail": (
                "LlamaGrammar.from_json_schema available"
                if llama_cpp_grammar_available
                else llama_cpp_detail
            ),
        },
    ]


def preferred_backend(local_backends: Sequence[Mapping[str, Any]]) -> str:
    available = {
        str(row.get("backend_name")) for row in local_backends if row.get("available") is True
    }
    if "llguidance" in available:
        return "llguidance"
    if "llama_cpp_grammar" in available:
        return "llama_cpp_grammar"
    return "deterministic_schema_validation"


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    local_backends: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp 2951 artifact without writing it."""

    config = config or ExperimentConfig()
    started = config.start_time()
    backends = [dict(row) for row in (local_backends or probe_local_backends())]
    adapter = StructuredCandidateManifestAdapter(local_backends=backends)
    records = synthetic_candidate_records()
    validation_results = adapter.validate_records(records)
    fixture_passed = all(row["schema_valid"] for row in validation_results)
    backend_map = {str(row["backend_name"]): bool(row["available"]) for row in backends}
    metadata = adapter.generation_metadata()

    return {
        "schema": "carnot.experiment_2951_structured_candidate_manifest_adapter.v1",
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "spec_traces": SPEC_TRACES,
        "honest_verdict": (
            "complete: structured candidate manifest adapter ready; no live LLM call made"
            if fixture_passed
            else "blocked_schema_fixture_validation_failed"
        ),
        "structured_decode_manifest_ready": fixture_passed,
        "schema_version": SCHEMA_VERSION,
        "schema_fields": list(CANDIDATE_SCHEMA_FIELDS),
        "candidate_manifest_schema": adapter.schema,
        "local_backends_checked": backends,
        "jsonschema_available": backend_map.get("jsonschema", False),
        "llguidance_available": backend_map.get("llguidance", False),
        "llama_cpp_grammar_available": backend_map.get("llama_cpp_grammar", False),
        "preferred_structured_output_backend": metadata["preferred_backend"],
        "structured_output_metadata": metadata,
        "validation_fixtures": records,
        "validation_fixture_results": validation_results,
        "validation_fixture_count": len(records),
        "validation_fixture_passed": fixture_passed,
        "model_specs_for_downstream_live_use": [dict(model) for model in MODEL_SPECS],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "duration_s": round(max(0.0, config.clock() - started), 6),
    }


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    local_backends: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and persist the deterministic Exp 2951 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config, local_backends=local_backends)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    schema_type = schema.get("type")
    if schema_type == "object":
        return _validate_object(value, schema, path)
    if schema_type == "array":
        return _validate_array(value, schema, path)
    if schema_type == "string":
        return _validate_string(value, schema, path)
    if schema_type == "number":
        return _validate_number(value, schema, path)
    return []


def _validate_object(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path} expected object"]
    errors: list[str] = []
    properties = dict(schema.get("properties", {}))
    for field in schema.get("required", []):
        if field not in value:
            errors.append(f"{path} missing required field {field}")
    if schema.get("additionalProperties") is False:
        for field in sorted(set(value) - set(properties)):
            errors.append(f"{path} unexpected field {field}")
    for field, field_schema in properties.items():
        if field in value:
            errors.extend(_validate_schema(value[field], field_schema, f"{path}.{field}"))
    return errors


def _validate_array(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{path} expected array"]
    errors: list[str] = []
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(f"{path} expected at least {min_items} item(s)")
    item_schema = schema.get("items")
    if isinstance(item_schema, Mapping):
        for index, item in enumerate(value):
            errors.extend(_validate_schema(item, item_schema, f"{path}[{index}]"))
    return errors


def _validate_string(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    if not isinstance(value, str):
        return [f"{path} expected string"]
    errors: list[str] = []
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        errors.append(f"{path} expected length >= {min_length}")
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{path} expected one of {enum_values}")
    pattern = schema.get("pattern")
    if pattern == "^[0-9a-f]{64}$" and _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path} expected 64 lowercase hex characters")
    return errors


def _validate_number(value: Any, schema: Mapping[str, Any], path: str) -> list[str]:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return [f"{path} expected number"]
    errors: list[str] = []
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, int | float) and value < minimum:
        errors.append(f"{path} expected >= {minimum}")
    if isinstance(maximum, int | float) and value > maximum:
        errors.append(f"{path} expected <= {maximum}")
    return errors


def _candidate_record(
    *,
    task_id: str,
    prompt_id: str,
    model_id: str,
    raw_completion_ref: str,
    raw_completion_text: str,
    repaired_code: str,
    failure_taxonomy: list[str],
    parser_status: str,
    test_status: str,
    verifier_score: float,
) -> JsonDict:
    return {
        "task_id": task_id,
        "prompt_id": prompt_id,
        "model_id": model_id,
        "raw_completion_ref": raw_completion_ref,
        "repaired_code": repaired_code,
        "failure_taxonomy": failure_taxonomy,
        "parser_status": parser_status,
        "test_status": test_status,
        "verifier_score": verifier_score,
        "provenance_checksums": {
            "raw_completion_sha256": _sha256_text(raw_completion_text),
            "repaired_code_sha256": _sha256_text(repaired_code),
            "manifest_schema_sha256": schema_checksum(),
        },
    }


def schema_checksum() -> str:
    return _sha256_text(json.dumps(candidate_manifest_schema(), sort_keys=True, separators=(",", ":")))


def _import_optional(name: str, importer: Importer) -> tuple[object | None, str]:
    try:
        module = importer(name)
    except ImportError as exc:
        return None, f"ImportError: {exc}"
    version = getattr(module, "__version__", None)
    detail = f"imported version {version}" if version else "imported"
    return module, detail


def _has_llguidance_json_schema(module: object | None) -> bool:
    matcher = getattr(module, "LLMatcher", None)
    return callable(getattr(matcher, "grammar_from_json_schema", None))


def _has_llama_cpp_json_schema_grammar(module: object | None) -> bool:
    grammar = getattr(module, "LlamaGrammar", None)
    return callable(getattr(grammar, "from_json_schema", None))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_clone(payload: Mapping[str, Any]) -> JsonDict:
    return json.loads(json.dumps(payload, sort_keys=True))
