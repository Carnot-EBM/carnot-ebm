"""Exp 2965 BEAVER-style certificate audit for structured repair manifests.

This is intentionally a bounded audit, not a full BEAVER proof.  BEAVER-style
here means that each repair candidate carries a deterministic certificate over
prefix-closed gates: once schema, parser, import, function-name, test, or
verifier evidence fails, a later verifier score cannot turn the row into a
pass.  That boundary prevents structured repair from becoming another
verifier-only false-accept path while leaving probability bounds unclaimed.

Spec: REQ-CODE-2965, SCENARIO-CODE-2965.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import structured_candidate_manifest_adapter as exp2951


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2965_beaver_style_repair_certificate_v1.json"
ARTIFACT = "experiment_2965_beaver_style_repair_certificate_v1"
SCHEMA = "carnot.beaver_style_repair_certificate_artifact.v1"
CERTIFICATE_SCHEMA_VERSION = "carnot.beaver_style_repair_certificate.v1"
INFERENCE_SUBSTRATE = "deterministic_wiring"

EXP2951_REL_PATH = Path("results/experiment_2951_structured_candidate_manifest_adapter_v1.json")
EXP2952_REL_PATH = Path("results/experiment_2952_sota_taxonomy_guided_code_repair_eval_v1.json")
EXP2953_REL_PATH = Path("results/experiment_2953_code_verifier_threshold_policy_v1.json")
EXP2963_REL_PATH = Path("results/experiment_2963_dccd_repair_protocol_manifest_v1.json")

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "bisect",
        "collections",
        "functools",
        "heapq",
        "itertools",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "typing",
    }
)

PREFIX_CLOSED_CONSTRAINTS = (
    {
        "constraint_id": "schema_validity",
        "prefix_closed": True,
        "description": "The candidate manifest must validate against the Exp 2951 schema.",
    },
    {
        "constraint_id": "code_block_completeness",
        "prefix_closed": True,
        "description": "The repaired_code field must be a complete parseable Python block.",
    },
    {
        "constraint_id": "import_allowlist",
        "prefix_closed": True,
        "description": "Imports must stay inside the deterministic repair allow-list.",
    },
    {
        "constraint_id": "function_name_preservation",
        "prefix_closed": True,
        "description": "When an expected entry point is known, repaired code must define it.",
    },
    {
        "constraint_id": "test_verifier_status_fields",
        "prefix_closed": True,
        "description": "Parser, test, and verifier fields must support deterministic acceptance.",
    },
)

FALSE_ACCEPT_AUDIT_FIELDS = (
    "verifier_accepted",
    "deterministic_accept",
    "schema_valid",
    "parser_valid",
    "code_block_complete",
    "import_allowlist_passed",
    "function_name_preserved",
    "test_status",
    "verifier_score",
    "false_accept",
    "reasons",
)

FILES_CHANGED = (
    "openspec/capabilities/code-verification/spec.md",
    "python/carnot/eval/beaver_style_repair_certificate.py",
    "tests/python/test_experiment_2965_beaver_style_repair_certificate.py",
    "results/experiment_2965_beaver_style_repair_certificate_v1.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "beaver_style_certificate_ready",
    "full_beaver_claim",
    "source_artifacts",
    "certificate_schema_version",
    "prefix_closed_constraints",
    "validation_fixture_count",
    "validation_fixture_passed",
    "local_backends_checked",
    "llguidance_available",
    "llama_cpp_grammar_available",
    "false_accept_audit_fields",
    "files_changed",
    "inference_substrate",
    "duration_s",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the deterministic Exp 2965 artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float = field(default_factory=time.time)
    clock: Callable[[], float] = time.time

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


def audit_candidate(
    record: Mapping[str, Any],
    *,
    schema: Mapping[str, Any],
    verifier_threshold: float,
    expected_function_names: Mapping[str, str],
    fixture_source: str,
) -> JsonDict:
    """Build one deterministic candidate certificate."""

    adapter = exp2951.StructuredCandidateManifestAdapter(schema=schema)
    validation = adapter.validate_record(record)
    code = str(record.get("repaired_code", ""))
    tree, parse_error = _parse_python(code)
    parser_valid = tree is not None
    code_block_complete = bool(code.strip()) and code.count("```") % 2 == 0 and parser_valid
    imports = _import_roots(tree)
    unsafe_imports = sorted(root for root in imports if root not in ALLOWED_IMPORT_ROOTS)
    import_allowlist_passed = parser_valid and not unsafe_imports
    function_preserved, expected_function, function_status = _function_name_preservation(
        record,
        tree,
        expected_function_names,
    )
    test_status = record.get("test_status")
    verifier_score = _numeric_score(record.get("verifier_score"))
    verifier_accepted = validation.ok and verifier_score >= verifier_threshold
    status_fields_valid = (
        record.get("parser_status") in exp2951.PARSER_STATUSES
        and test_status in exp2951.TEST_STATUSES
        and verifier_score >= 0.0
    )
    deterministic_accept = (
        validation.ok
        and code_block_complete
        and parser_valid
        and import_allowlist_passed
        and function_preserved is not False
        and test_status == "passed"
        and status_fields_valid
    )
    prefix_results = _prefix_results(
        schema_valid=validation.ok,
        code_block_complete=code_block_complete,
        import_allowlist_passed=import_allowlist_passed,
        function_name_preserved=function_preserved,
        status_fields_valid=status_fields_valid and test_status == "passed",
    )
    false_accept_audit = _false_accept_audit(
        verifier_accepted=verifier_accepted,
        deterministic_accept=deterministic_accept,
        schema_valid=validation.ok,
        parser_valid=parser_valid,
        code_block_complete=code_block_complete,
        import_allowlist_passed=import_allowlist_passed,
        function_name_preserved=function_preserved,
        test_status=str(test_status),
        verifier_score=verifier_score,
    )
    return {
        "candidate_id": str(record.get("task_id", "unknown")),
        "fixture_source": fixture_source,
        "explored_prefix_count": len(prefix_results),
        "blocked_prefix_count": sum(1 for result in prefix_results if not result["passed"]),
        "schema_valid": validation.ok,
        "schema_errors": validation.errors,
        "parser_valid": parser_valid,
        "parser_error": parse_error,
        "code_block_complete": code_block_complete,
        "imports_seen": imports,
        "unsafe_imports": unsafe_imports,
        "import_allowlist_passed": import_allowlist_passed,
        "expected_function_name": expected_function,
        "function_name_preserved": function_preserved,
        "function_name_status": function_status,
        "test_status": test_status,
        "verifier_score": verifier_score,
        "verifier_threshold_used": verifier_threshold,
        "deterministic_accept": deterministic_accept,
        "prefix_results": prefix_results,
        "false_accept_audit": false_accept_audit,
    }


def synthetic_candidate_records() -> tuple[list[JsonDict], dict[str, str]]:
    """Return five deterministic records that exercise accept and block paths."""

    records = [
        _candidate_record(
            task_id="synthetic_valid",
            repaired_code="def add(a, b):\n    return a + b\n",
            test_status="passed",
            verifier_score=1.0,
        ),
        _candidate_record(
            task_id="synthetic_unsafe_import",
            repaired_code="import os\ndef add(a, b):\n    return os.getcwd()\n",
            test_status="passed",
            verifier_score=1.0,
        ),
        _candidate_record(
            task_id="synthetic_function_mismatch",
            repaired_code="def subtract(a, b):\n    return a - b\n",
            test_status="passed",
            verifier_score=1.0,
        ),
        _candidate_record(
            task_id="synthetic_failed_tests",
            repaired_code="def add(a, b):\n    return a - b\n",
            failure_taxonomy=["failed_tests"],
            test_status="failed",
            verifier_score=1.0,
        ),
        _candidate_record(
            task_id="synthetic_syntax_error",
            repaired_code="def add(:\n",
            failure_taxonomy=["syntax_error"],
            parser_status="syntax_error",
            test_status="not_run",
            verifier_score=0.0,
        ),
    ]
    expected_names = {
        "synthetic_valid": "add",
        "synthetic_unsafe_import": "add",
        "synthetic_function_mismatch": "add",
        "synthetic_failed_tests": "add",
        "synthetic_syntax_error": "add",
    }
    return records, expected_names


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    local_backends: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the Exp 2965 artifact without writing it."""

    config = config or ExperimentConfig()
    source_artifacts = _source_artifacts(config)
    missing_sources = [
        source["experiment_id"]
        for source in source_artifacts
        if source["required"] and not source["present"]
    ]
    backends = [dict(row) for row in (local_backends or exp2951.probe_local_backends())]
    if missing_sources:
        return _blocked_artifact(
            config,
            "blocked_missing_required_source_artifact",
            source_artifacts,
            missing_sources,
            [],
            backends,
        )

    payloads, malformed_sources = _read_payloads(config, source_artifacts)
    if malformed_sources:
        return _blocked_artifact(
            config,
            "blocked_malformed_source_artifact",
            source_artifacts,
            [],
            malformed_sources,
            backends,
        )
    if payloads["exp2963"].get("dccd_repair_protocol_ready") is not True:
        return _blocked_artifact(
            config,
            "blocked_exp2963_protocol_not_ready",
            source_artifacts,
            [],
            [],
            backends,
        )

    schema = payloads["exp2951"].get("candidate_manifest_schema") or exp2951.candidate_manifest_schema()
    threshold = _verifier_threshold(payloads["exp2953"])
    synthetic_records, expected_names = synthetic_candidate_records()
    synthetic_certificates = [
        audit_candidate(
            record,
            schema=schema,
            verifier_threshold=threshold,
            expected_function_names=expected_names,
            fixture_source="synthetic",
        )
        for record in synthetic_records
    ]
    repair_candidates = list(payloads.get("exp2952", {}).get("candidate_manifests", []))
    repair_certificates = [
        audit_candidate(
            record,
            schema=schema,
            verifier_threshold=threshold,
            expected_function_names={},
            fixture_source="exp2952_available",
        )
        for record in repair_candidates
    ]
    validation_fixture_passed = _synthetic_validation_passed(synthetic_certificates)
    ready = validation_fixture_passed
    return _final_artifact(
        config=config,
        ready=ready,
        verdict=(
            "complete: bounded certificate audit ready; full BEAVER probability bound not claimed"
            if ready
            else "blocked_synthetic_certificate_validation_failed"
        ),
        source_artifacts=source_artifacts,
        missing_sources=[],
        malformed_sources=[],
        local_backends=backends,
        validation_fixture_count=len(synthetic_records),
        validation_fixture_passed=validation_fixture_passed,
        available_repair_candidate_count=len(repair_candidates),
        available_repair_candidate_audited_count=len(repair_certificates),
        candidate_certificates=synthetic_certificates + repair_certificates,
    )


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    local_backends: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build and persist the deterministic Exp 2965 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config, local_backends=local_backends)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _blocked_artifact(
    config: ExperimentConfig,
    verdict: str,
    source_artifacts: list[JsonDict],
    missing_sources: list[str],
    malformed_sources: list[str],
    local_backends: list[JsonDict],
) -> JsonDict:
    return _final_artifact(
        config=config,
        ready=False,
        verdict=verdict,
        source_artifacts=source_artifacts,
        missing_sources=missing_sources,
        malformed_sources=malformed_sources,
        local_backends=local_backends,
        validation_fixture_count=0,
        validation_fixture_passed=False,
        available_repair_candidate_count=0,
        available_repair_candidate_audited_count=0,
        candidate_certificates=[],
    )


def _final_artifact(
    *,
    config: ExperimentConfig,
    ready: bool,
    verdict: str,
    source_artifacts: list[JsonDict],
    missing_sources: list[str],
    malformed_sources: list[str],
    local_backends: list[JsonDict],
    validation_fixture_count: int,
    validation_fixture_passed: bool,
    available_repair_candidate_count: int,
    available_repair_candidate_audited_count: int,
    candidate_certificates: list[JsonDict],
) -> JsonDict:
    backend_map = {str(row.get("backend_name")): bool(row.get("available")) for row in local_backends}
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "beaver_style_certificate_ready": ready,
        "full_beaver_claim": False,
        "source_artifacts": source_artifacts,
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "prefix_closed_constraints": [dict(row) for row in PREFIX_CLOSED_CONSTRAINTS],
        "validation_fixture_count": validation_fixture_count,
        "validation_fixture_passed": validation_fixture_passed,
        "available_repair_candidate_count": available_repair_candidate_count,
        "available_repair_candidate_audited_count": available_repair_candidate_audited_count,
        "local_backends_checked": local_backends,
        "llguidance_available": backend_map.get("llguidance", False),
        "llama_cpp_grammar_available": backend_map.get("llama_cpp_grammar", False),
        "json_schema_validation_support": "deterministic_schema_validation_fallback",
        "false_accept_audit_fields": list(FALSE_ACCEPT_AUDIT_FIELDS),
        "candidate_certificates": candidate_certificates,
        "files_changed": list(FILES_CHANGED),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "missing_source_artifacts": missing_sources,
        "malformed_source_artifacts": malformed_sources,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
        "duration_s": max(0.0, config.clock() - config.started_at),
    }


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    specs = (
        (
            "exp2963",
            EXP2963_REL_PATH,
            True,
            "ready_dccd_repair_protocol_precondition",
            ["dccd_repair_protocol_ready"],
        ),
        (
            "exp2951",
            EXP2951_REL_PATH,
            True,
            "candidate_schema_and_local_backend_status",
            ["candidate_manifest_schema", "local_backends_checked"],
        ),
        (
            "exp2953",
            EXP2953_REL_PATH,
            True,
            "verifier_threshold_policy",
            ["selected_default_threshold"],
        ),
        (
            "exp2952",
            EXP2952_REL_PATH,
            False,
            "available_v278_repair_candidate_manifests",
            ["candidate_manifests"],
        ),
    )
    artifacts: list[JsonDict] = []
    for experiment_id, rel_path, required, role, fields in specs:
        path = _repo_path(config.repo_root, rel_path)
        present = path.is_file()
        artifacts.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "required": required,
                "present": present,
                "sha256": _sha256(path) if present else None,
                "fields_imported": list(fields),
            }
        )
    return artifacts


def _read_payloads(
    config: ExperimentConfig,
    source_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, JsonDict], list[str]]:
    payloads: dict[str, JsonDict] = {}
    malformed_sources: list[str] = []
    for source in source_artifacts:
        if not source["present"]:
            continue
        try:
            payloads[str(source["experiment_id"])] = _read_json(
                _repo_path(config.repo_root, Path(str(source["path"])))
            )
        except ValueError:
            malformed_sources.append(str(source["experiment_id"]))
    return payloads, malformed_sources


def _prefix_results(
    *,
    schema_valid: bool,
    code_block_complete: bool,
    import_allowlist_passed: bool,
    function_name_preserved: bool | None,
    status_fields_valid: bool,
) -> list[JsonDict]:
    values = {
        "schema_validity": schema_valid,
        "code_block_completeness": code_block_complete,
        "import_allowlist": import_allowlist_passed,
        "function_name_preservation": function_name_preserved is not False,
        "test_verifier_status_fields": status_fields_valid,
    }
    return [
        {
            "constraint_id": constraint["constraint_id"],
            "prefix_closed": True,
            "passed": values[constraint["constraint_id"]],
        }
        for constraint in PREFIX_CLOSED_CONSTRAINTS
    ]


def _false_accept_audit(
    *,
    verifier_accepted: bool,
    deterministic_accept: bool,
    schema_valid: bool,
    parser_valid: bool,
    code_block_complete: bool,
    import_allowlist_passed: bool,
    function_name_preserved: bool | None,
    test_status: str,
    verifier_score: float,
) -> JsonDict:
    reasons: list[str] = []
    if not schema_valid:
        reasons.append("schema_invalid")
    if not parser_valid:
        reasons.append("parser_invalid")
    if not code_block_complete:
        reasons.append("code_block_incomplete")
    if not import_allowlist_passed:
        reasons.append("unsafe_imports")
    if function_name_preserved is False:
        reasons.append("function_name_not_preserved")
    if test_status != "passed":
        reasons.append("test_status_not_passed")
    return {
        "verifier_accepted": verifier_accepted,
        "deterministic_accept": deterministic_accept,
        "schema_valid": schema_valid,
        "parser_valid": parser_valid,
        "code_block_complete": code_block_complete,
        "import_allowlist_passed": import_allowlist_passed,
        "function_name_preserved": function_name_preserved,
        "test_status": test_status,
        "verifier_score": verifier_score,
        "false_accept": verifier_accepted and not deterministic_accept,
        "reasons": reasons,
    }


def _synthetic_validation_passed(certificates: Sequence[Mapping[str, Any]]) -> bool:
    by_id = {str(cert["candidate_id"]): cert for cert in certificates}
    return (
        len(certificates) == 5
        and by_id["synthetic_valid"]["deterministic_accept"] is True
        and by_id["synthetic_unsafe_import"]["false_accept_audit"]["false_accept"] is True
        and by_id["synthetic_function_mismatch"]["false_accept_audit"]["false_accept"] is True
        and by_id["synthetic_failed_tests"]["false_accept_audit"]["false_accept"] is True
        and by_id["synthetic_syntax_error"]["parser_valid"] is False
    )


def _function_name_preservation(
    record: Mapping[str, Any],
    tree: ast.AST | None,
    expected_function_names: Mapping[str, str],
) -> tuple[bool | None, str | None, str]:
    task_id = str(record.get("task_id", ""))
    expected = expected_function_names.get(task_id) or expected_function_names.get(_base_task_id(task_id))
    if not expected:
        return None, None, "not_applicable_missing_expected_function"
    names = _function_names(tree)
    if expected in names:
        return True, expected, "preserved"
    return False, expected, "missing_expected_function"


def _parse_python(code: str) -> tuple[ast.Module | None, str | None]:
    try:
        return ast.parse(code), None
    except SyntaxError as exc:
        return None, f"{exc.__class__.__name__}: {exc.msg}"


def _function_names(tree: ast.AST | None) -> list[str]:
    if tree is None:
        return []
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _import_roots(tree: ast.AST | None) -> list[str]:
    if tree is None:
        return []
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".", 1)[0])
    return sorted(roots)


def _numeric_score(value: Any) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _verifier_threshold(exp2953: Mapping[str, Any]) -> float:
    threshold = exp2953.get("selected_default_threshold")
    if isinstance(threshold, int | float) and not isinstance(threshold, bool):
        return float(threshold)
    return 1.0


def _candidate_record(
    *,
    task_id: str,
    repaired_code: str,
    failure_taxonomy: list[str] | None = None,
    parser_status: str = "parsed",
    test_status: str,
    verifier_score: float,
) -> JsonDict:
    return {
        "task_id": task_id,
        "prompt_id": f"{task_id}:prompt",
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "raw_completion_ref": f"results/raw/experiment_2965/{task_id}.txt",
        "repaired_code": repaired_code,
        "failure_taxonomy": failure_taxonomy or ["none"],
        "parser_status": parser_status,
        "test_status": test_status,
        "verifier_score": verifier_score,
        "provenance_checksums": {
            "raw_completion_sha256": _sha256_text(repaired_code),
            "repaired_code_sha256": _sha256_text(repaired_code),
            "manifest_schema_sha256": exp2951.schema_checksum(),
        },
    }


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON source artifact: {path}") from exc


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _base_task_id(task_id: str) -> str:
    parts = task_id.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else task_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
