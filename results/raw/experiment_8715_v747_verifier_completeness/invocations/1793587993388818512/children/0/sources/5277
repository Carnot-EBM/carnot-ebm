"""Build the Exp 3088 structured repair emitter preflight artifact.

Spec refs: REQ-REPORT-3088, SCENARIO-REPORT-3088.

This module deliberately stops at a local schema/parser preflight. XGrammar or
LLGuidance can reduce malformed structured output during decoding, but a repair
micro-panel still needs deterministic gates before it can treat syntax-valid
JSON as useful repair evidence. The fallback here validates cached payloads
only, so the artifact can say that a contract exists without claiming live LLM
repair quality.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence


JsonDict = dict[str, Any]
ImportChecker = Callable[[str], bool]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3088_xgrammar2_structured_repair_emitter_preflight_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
SCHEMA_REL_PATH = Path("python/carnot/schemas/structured_repair_candidate_v1.json")
EXP3074_REL_PATH = Path("results/experiment_3074_llguidance_aprad_repair_protocol_v1.json")
EXP3075_REL_PATH = Path(
    "results/experiment_3075_gated_grammar_constrained_sota_repair_micro_panel_v1.json"
)
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

REPAIR_CANDIDATE_REQUIRED_FIELDS = (
    "task_id",
    "task_intent_hash",
    "patch",
    "behavioral_tests",
    "semantic_drift_checks",
    "verifier_authority",
)
VERIFIER_AUTHORITIES = (
    "deterministic_tests",
    "exact_solver",
    "exact_verifier",
    "blocked_unavailable",
)
SEMANTIC_AUTHORITIES = ("deterministic_tests", "exact_solver", "exact_verifier")
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
SYNTAX_FAILURE_CLASSES_FOR_EXP3089 = (
    "json_decode_error",
    "not_json_object",
    "missing_required_field",
    "wrong_type",
    "enum_violation",
    "extra_property",
    "invalid_task_intent_hash",
    "empty_patch",
    "unchecked_semantic_drift",
)
REQUIRED_ARTIFACT_FIELDS = (
    "structured_generation_ready",
    "grammar_or_schema_path",
    "parser_validation_count",
    "invalid_payload_rejection_count",
    "structured_library_available",
    "fallback_contract_used",
    "tests_added_or_reused",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
TEST_PATH = "tests/python/test_experiment_3088_structured_repair_emitter_preflight.py"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RepairPayloadParseResult:
    """One deterministic parser result for a cached repair-candidate payload."""

    valid: bool
    payload: JsonDict
    errors: list[str]
    failure_class: str | None

    def to_dict(self) -> JsonDict:
        """Return a JSON-safe row for the terminal artifact."""

        return {
            "valid": self.valid,
            "payload": self.payload,
            "errors": list(self.errors),
            "failure_class": self.failure_class,
        }


@dataclass(frozen=True)
class InvalidPayloadCase:
    """One cached invalid example and the failure class Exp 3089 should measure."""

    name: str
    raw_text: str
    expected_failure_class: str

    def to_dict(self) -> JsonDict:
        """Return the fixture metadata without implying it came from a live model."""

        return {
            "name": self.name,
            "raw_text": self.raw_text,
            "expected_failure_class": self.expected_failure_class,
        }


def load_repair_candidate_schema(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the concrete JSON schema that downstream emitters can hand to decoders."""

    return json.loads((Path(root) / SCHEMA_REL_PATH).read_text(encoding="utf-8"))


def cached_valid_payloads() -> list[JsonDict]:
    """Return cached examples that are syntactically valid repair candidates."""

    return [
        {
            "task_id": "cached-repair-001",
            "task_intent_hash": "a" * 64,
            "patch": "--- a/foo.py\n+++ b/foo.py\n@@\n-return 1\n+return 2\n",
            "behavioral_tests": [
                {
                    "name": "unit",
                    "command": ".venv/bin/pytest tests/python/test_cached.py -q",
                    "expected": "pass",
                }
            ],
            "semantic_drift_checks": [
                {
                    "name": "intent",
                    "authority": "deterministic_tests",
                    "must_pass": True,
                }
            ],
            "verifier_authority": "deterministic_tests",
        },
        {
            "task_id": "cached-repair-002",
            "task_intent_hash": "b" * 64,
            "patch": "--- a/solver.py\n+++ b/solver.py\n@@\n-answer = None\n+answer = 4\n",
            "behavioral_tests": [
                {
                    "name": "solver-check",
                    "command": ".venv/bin/pytest tests/python/test_solver.py -q",
                    "expected": "pass",
                }
            ],
            "semantic_drift_checks": [
                {
                    "name": "exact-solver",
                    "authority": "exact_solver",
                    "must_pass": True,
                }
            ],
            "verifier_authority": "exact_solver",
        },
    ]


def cached_invalid_payloads() -> list[InvalidPayloadCase]:
    """Return cached malformed examples that exercise syntax failure classes."""

    base = cached_valid_payloads()[0]
    cases = [
        InvalidPayloadCase("malformed_json", "{bad", "json_decode_error"),
        InvalidPayloadCase(
            "missing_patch",
            _canonical_json({key: value for key, value in base.items() if key != "patch"}),
            "missing_required_field",
        ),
        InvalidPayloadCase(
            "wrong_behavioral_tests_type",
            _canonical_json(base | {"behavioral_tests": "pytest"}),
            "wrong_type",
        ),
        InvalidPayloadCase(
            "wrong_verifier_authority",
            _canonical_json(base | {"verifier_authority": "self_graded"}),
            "enum_violation",
        ),
        InvalidPayloadCase(
            "extra_field",
            _canonical_json(base | {"confidence": 0.9}),
            "extra_property",
        ),
        InvalidPayloadCase(
            "invalid_task_intent_hash",
            _canonical_json(base | {"task_intent_hash": "sha256:not-a-hash"}),
            "invalid_task_intent_hash",
        ),
        InvalidPayloadCase("empty_patch", _canonical_json(base | {"patch": ""}), "empty_patch"),
        InvalidPayloadCase(
            "unchecked_semantic_drift",
            _canonical_json(
                base
                | {
                    "semantic_drift_checks": [
                        {
                            "name": "intent",
                            "authority": "deterministic_tests",
                            "must_pass": False,
                        }
                    ]
                }
            ),
            "unchecked_semantic_drift",
        ),
    ]
    return cases


def parse_repair_payload_text(
    raw_text: str,
    *,
    schema: Mapping[str, Any] | None = None,
) -> RepairPayloadParseResult:
    """Parse exactly one JSON repair candidate and validate the local contract."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return RepairPayloadParseResult(
            valid=False,
            payload={},
            errors=[f"json_decode_error:{exc.msg}"],
            failure_class="json_decode_error",
        )
    errors = validate_repair_payload(parsed, schema=schema)
    return RepairPayloadParseResult(
        valid=not errors,
        payload=dict(parsed) if isinstance(parsed, Mapping) else {},
        errors=errors,
        failure_class=None if not errors else _failure_class(errors[0]),
    )


def validate_repair_payload(
    payload: Any,
    *,
    schema: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the schema subset plus Carnot's syntax-only repair guardrails."""

    active_schema = schema or load_repair_candidate_schema()
    properties = active_schema["properties"]
    if not isinstance(payload, Mapping):
        return ["not_json_object:$ expected object"]

    errors: list[str] = []
    for field in REPAIR_CANDIDATE_REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_required_field:$.{field}")
    for field in payload:
        if field not in properties:
            errors.append(f"extra_property:$.{field}")
    if errors:
        return errors

    _require_string(errors, payload, "task_id")
    _require_hash(errors, payload.get("task_intent_hash"))
    _require_patch(errors, payload.get("patch"))
    _require_behavioral_tests(errors, payload.get("behavioral_tests"))
    _require_semantic_drift_checks(errors, payload.get("semantic_drift_checks"))
    if payload.get("verifier_authority") not in VERIFIER_AUTHORITIES:
        errors.append("enum_violation:$.verifier_authority")
    return errors


def run_parser_validation(
    valid_payloads: Sequence[Mapping[str, Any]] | None = None,
    invalid_payloads: Sequence[InvalidPayloadCase] | None = None,
) -> JsonDict:
    """Run cached valid and invalid parser checks and count both sides."""

    schema = load_repair_candidate_schema()
    valid_rows = [
        parse_repair_payload_text(_canonical_json(payload), schema=schema)
        for payload in (valid_payloads or cached_valid_payloads())
    ]
    invalid_cases = list(invalid_payloads or cached_invalid_payloads())
    invalid_rows = [
        {
            "name": case.name,
            "expected_failure_class": case.expected_failure_class,
            "result": parse_repair_payload_text(case.raw_text, schema=schema).to_dict(),
        }
        for case in invalid_cases
    ]
    parser_validation_count = sum(row.valid for row in valid_rows)
    invalid_payload_rejection_count = sum(
        not row["result"]["valid"] and row["result"]["failure_class"] == row["expected_failure_class"]
        for row in invalid_rows
    )
    return {
        "parser_validation_count": parser_validation_count,
        "invalid_payload_rejection_count": invalid_payload_rejection_count,
        "accepted_invalid_count": sum(row["result"]["valid"] for row in invalid_rows),
        "rejected_valid_count": sum(not row.valid for row in valid_rows),
        "syntax_failure_classes_for_exp3089": list(SYNTAX_FAILURE_CLASSES_FOR_EXP3089),
        "valid_rows": [row.to_dict() for row in valid_rows],
        "invalid_rows": invalid_rows,
    }


def probe_structured_libraries(
    *,
    import_checker: ImportChecker = lambda name: importlib.util.find_spec(name) is not None,
) -> JsonDict:
    """Record native library availability while keeping the fallback explicit."""

    xgrammar_available = bool(import_checker("xgrammar"))
    llguidance_available = bool(import_checker("llguidance"))
    return {
        "xgrammar": {
            "import_name": "xgrammar",
            "available": xgrammar_available,
            "role": "native_token_mask_backend",
        },
        "llguidance": {
            "import_name": "llguidance",
            "available": llguidance_available,
            "role": "native_json_schema_grammar_backend",
        },
        "repo_native_fallback": {
            "available": True,
            "role": "post_decode_schema_contract",
            "schema_path": SCHEMA_REL_PATH.as_posix(),
        },
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
    import_checker: ImportChecker = lambda name: importlib.util.find_spec(name) is not None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the completed preflight artifact without running repair generation."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    validation = run_parser_validation()
    dependency_probe = probe_structured_libraries(import_checker=import_checker)
    native_available = bool(
        dependency_probe["xgrammar"]["available"] or dependency_probe["llguidance"]["available"]
    )
    fallback_used = not native_available
    substrate = _inference_substrate()
    ready = (
        (root_path / SCHEMA_REL_PATH).is_file()
        and validation["parser_validation_count"] > 0
        and validation["invalid_payload_rejection_count"] > 0
        and substrate["live_llm_inference"] is False
        and substrate["repair_quality_claimed"] is False
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": "carnot.structured_repair_emitter_preflight.v1",
        "run_date": RUN_DATE,
        "structured_generation_ready": ready,
        "grammar_or_schema_path": SCHEMA_REL_PATH.as_posix(),
        "grammar_or_schema_kind": "json_schema",
        "schema_checksum": _sha256_file(root_path / SCHEMA_REL_PATH),
        "parser_validation_count": validation["parser_validation_count"],
        "invalid_payload_rejection_count": validation["invalid_payload_rejection_count"],
        "structured_library_available": native_available,
        "fallback_contract_used": fallback_used,
        "blocked_library_missing": False if fallback_used else not native_available,
        "dependency_probe": dependency_probe,
        "tests_added_or_reused": list(tests_run or [TEST_PATH]),
        "source_artifacts": _source_artifacts(root_path),
        "validation_manifest": validation,
        "syntax_failure_classes_for_exp3089": validation["syntax_failure_classes_for_exp3089"],
        "cached_examples_only": True,
        "inference_substrate": substrate,
        "duration_s": _duration(start, now_s),
        "honest_verdict": _honest_verdict(ready, fallback_used, validation),
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str] | None = None,
    import_checker: ImportChecker = lambda name: importlib.util.find_spec(name) is not None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Write the terminal Exp 3088 JSON artifact and return it."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        tests_run=tests_run,
        import_checker=import_checker,
        started_s=started_s,
        now_s=now_s,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the artifact could be mistaken for a live repair result."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not (REPO_ROOT / str(artifact["grammar_or_schema_path"])).is_file():
        raise ValueError("grammar_or_schema_path must point to a checked-in contract")
    if int(artifact["parser_validation_count"]) <= 0:
        raise ValueError("parser_validation_count must be positive")
    if int(artifact["invalid_payload_rejection_count"]) <= 0:
        raise ValueError("invalid_payload_rejection_count must be positive")
    substrate = artifact["inference_substrate"]
    if not isinstance(substrate, Mapping) or substrate.get("live_llm_inference") is not False:
        raise ValueError("inference_substrate.live_llm_inference must be false")
    if substrate.get("repair_quality_claimed") is not False:
        raise ValueError("inference_substrate.repair_quality_claimed must be false")
    if artifact["structured_generation_ready"] is True:
        verdict = str(artifact["honest_verdict"])
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must start with a terminal success prefix")


def _require_string(
    errors: list[str],
    payload: Mapping[str, Any],
    field: str,
    *,
    path: str | None = None,
) -> None:
    value = payload.get(field)
    error_path = path or field
    if not isinstance(value, str):
        errors.append(f"wrong_type:$.{error_path}")
    elif not value.strip():
        errors.append(f"empty_string:$.{error_path}")


def _require_hash(errors: list[str], value: Any) -> None:
    if not isinstance(value, str):
        errors.append("wrong_type:$.task_intent_hash")
    elif not _HASH_RE.match(value):
        errors.append("invalid_task_intent_hash:$.task_intent_hash")


def _require_patch(errors: list[str], value: Any) -> None:
    if not isinstance(value, str):
        errors.append("wrong_type:$.patch")
    elif not value.strip():
        errors.append("empty_patch:$.patch")


def _require_behavioral_tests(errors: list[str], value: Any) -> None:
    if not isinstance(value, list):
        errors.append("wrong_type:$.behavioral_tests")
        return
    if not value:
        errors.append("wrong_type:$.behavioral_tests")
        return
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"wrong_type:$.behavioral_tests[{index}]")
            continue
        _require_string(errors, item, "name", path=f"behavioral_tests[{index}].name")
        _require_string(errors, item, "command", path=f"behavioral_tests[{index}].command")
        if item.get("expected") not in {"pass", "fail"}:
            errors.append(f"enum_violation:$.behavioral_tests[{index}].expected")


def _require_semantic_drift_checks(errors: list[str], value: Any) -> None:
    if not isinstance(value, list):
        errors.append("wrong_type:$.semantic_drift_checks")
        return
    if not value:
        errors.append("unchecked_semantic_drift:$.semantic_drift_checks")
        return
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"wrong_type:$.semantic_drift_checks[{index}]")
            continue
        _require_string(errors, item, "name", path=f"semantic_drift_checks[{index}].name")
        if item.get("authority") not in SEMANTIC_AUTHORITIES:
            errors.append(f"enum_violation:$.semantic_drift_checks[{index}].authority")
        if item.get("must_pass") is not True:
            errors.append(f"unchecked_semantic_drift:$.semantic_drift_checks[{index}].must_pass")


def _source_artifacts(root: Path) -> list[JsonDict]:
    rows = [
        ("exp3074", EXP3074_REL_PATH, "llguidance_aprad_repair_protocol", True),
        ("exp3075", EXP3075_REL_PATH, "referenced_gated_repair_micro_panel", False),
        ("research_references", RESEARCH_REFERENCES_REL_PATH, "structured_generation_context", False),
    ]
    return [
        {
            "experiment_id": experiment_id,
            "path": rel_path.as_posix(),
            "role": role,
            "required": required,
            "present": (root / rel_path).is_file(),
            "sha256": _sha256_file(root / rel_path),
        }
        for experiment_id, rel_path, role, required in rows
    ]


def _inference_substrate() -> JsonDict:
    return {
        "mode": "cached_payload_schema_preflight",
        "cached_examples_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "model_load_attempted": False,
        "fresh_repair_generation": False,
        "fresh_verifier_scoring": False,
        "fresh_solver_execution": False,
        "repair_quality_claimed": False,
        "conductor_invoked": False,
    }


def _honest_verdict(
    ready: bool,
    fallback_used: bool,
    validation: Mapping[str, Any],
) -> str:
    if not ready:
        return "blocked_preflight_contract_invalid"
    mode = "fallback_contract" if fallback_used else "native_structured_library"
    return (
        f"complete: structured_repair_emitter_preflight_ready_{mode}; "
        f"valid={validation['parser_validation_count']}; "
        f"invalid_rejected={validation['invalid_payload_rejection_count']}; "
        "no_live_repair_quality_claim"
    )


def _failure_class(error: str) -> str:
    return error.split(":", 1)[0]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
