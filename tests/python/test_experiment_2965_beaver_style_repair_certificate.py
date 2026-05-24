"""Tests for Exp 2965 BEAVER-style structured-repair certificate audit.

Spec: REQ-CODE-2965, SCENARIO-CODE-2965.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import beaver_style_repair_certificate as exp
from carnot.eval import structured_candidate_manifest_adapter as exp2951


REQUIRED_FIELDS = {
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
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    *,
    task_id: str,
    code: str,
    parser_status: str = "parsed",
    test_status: str = "passed",
    verifier_score: float = 1.0,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "prompt_id": f"{task_id}:prompt",
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "raw_completion_ref": f"results/raw/{task_id}.txt",
        "repaired_code": code,
        "failure_taxonomy": ["none"] if test_status == "passed" else ["failed_tests"],
        "parser_status": parser_status,
        "test_status": test_status,
        "verifier_score": verifier_score,
        "provenance_checksums": {
            "raw_completion_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "repaired_code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "manifest_schema_sha256": exp2951.schema_checksum(),
        },
    }


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        exp.EXP2951_REL_PATH,
        {
            "artifact": "experiment_2951_structured_candidate_manifest_adapter_v1",
            "structured_decode_manifest_ready": True,
            "schema_version": exp2951.SCHEMA_VERSION,
            "schema_fields": list(exp2951.CANDIDATE_SCHEMA_FIELDS),
            "candidate_manifest_schema": exp2951.candidate_manifest_schema(),
            "local_backends_checked": [
                {"backend_name": "jsonschema", "available": False, "detail": "missing"},
                {"backend_name": "llguidance", "available": False, "detail": "missing"},
                {
                    "backend_name": "llama_cpp_grammar",
                    "available": True,
                    "detail": "LlamaGrammar.from_json_schema available",
                },
            ],
        },
    )
    _write_json(
        root,
        exp.EXP2952_REL_PATH,
        {
            "artifact": "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1",
            "candidate_manifests": [
                _candidate(
                    task_id="MBPP:mbpp-11:baseline_no_taxonomy:0",
                    code="public function signature: incomplete",
                    parser_status="syntax_error",
                    test_status="not_run",
                    verifier_score=0.0,
                ),
                _candidate(
                    task_id="MBPP:mbpp-12:taxonomy_guided:0",
                    code="def max_chain_length(pairs):\n    return len(pairs)\n",
                ),
            ],
        },
    )
    _write_json(
        root,
        exp.EXP2953_REL_PATH,
        {
            "artifact": "experiment_2953_code_verifier_threshold_policy_v1",
            "threshold_policy_ready": True,
            "selected_default_threshold": 1.0,
        },
    )
    _write_json(
        root,
        exp.EXP2963_REL_PATH,
        {
            "artifact": "experiment_2963_dccd_repair_protocol_manifest_v1",
            "dccd_repair_protocol_ready": True,
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 12.5,
        tests_run=("focused-pytest",),
    )


def _backend(name: str, available: bool, detail: str) -> dict[str, Any]:
    return {"backend_name": name, "available": available, "detail": detail}


def test_req_code_2965_spec_anchor_exists() -> None:
    """REQ-CODE-2965, SCENARIO-CODE-2965: the certificate audit is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2965" in spec
    assert "SCENARIO-CODE-2965" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "full_beaver_claim=false" in spec
    assert 'inference_substrate="deterministic_wiring"' in spec


def test_scenario_code_2965_builds_ready_certificate_artifact(tmp_path: Path) -> None:
    """SCENARIO-CODE-2965: synthetic and available .278 manifests are audited."""

    _write_ready_sources(tmp_path)
    local_backends = [
        _backend("jsonschema", False, "not installed"),
        _backend("llguidance", False, "not installed"),
        _backend("llama_cpp_grammar", True, "LlamaGrammar.from_json_schema available"),
    ]

    artifact = exp.write_artifact(_config(tmp_path), local_backends=local_backends)
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["beaver_style_certificate_ready"] is True
    assert artifact["full_beaver_claim"] is False
    assert "bounded certificate audit ready" in artifact["honest_verdict"]
    assert artifact["certificate_schema_version"] == exp.CERTIFICATE_SCHEMA_VERSION
    assert artifact["validation_fixture_count"] == 5
    assert artifact["validation_fixture_passed"] is True
    assert artifact["available_repair_candidate_count"] == 2
    assert artifact["available_repair_candidate_audited_count"] == 2
    assert artifact["llguidance_available"] is False
    assert artifact["llama_cpp_grammar_available"] is True
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["files_changed"] == list(exp.FILES_CHANGED)

    constraint_ids = {row["constraint_id"] for row in artifact["prefix_closed_constraints"]}
    assert {
        "schema_validity",
        "code_block_completeness",
        "import_allowlist",
        "function_name_preservation",
        "test_verifier_status_fields",
    } <= constraint_ids
    assert artifact["false_accept_audit_fields"] == list(exp.FALSE_ACCEPT_AUDIT_FIELDS)

    certificates = {row["candidate_id"]: row for row in artifact["candidate_certificates"]}
    assert certificates["synthetic_valid"]["deterministic_accept"] is True
    assert certificates["synthetic_unsafe_import"]["false_accept_audit"]["false_accept"] is True
    assert "unsafe_imports" in certificates["synthetic_unsafe_import"]["false_accept_audit"]["reasons"]
    assert certificates["synthetic_function_mismatch"]["false_accept_audit"]["false_accept"] is True
    assert "function_name_not_preserved" in (
        certificates["synthetic_function_mismatch"]["false_accept_audit"]["reasons"]
    )
    assert certificates["synthetic_failed_tests"]["false_accept_audit"]["false_accept"] is True
    assert "test_status_not_passed" in certificates["synthetic_failed_tests"]["false_accept_audit"][
        "reasons"
    ]
    assert certificates["synthetic_syntax_error"]["parser_valid"] is False
    assert certificates["MBPP:mbpp-12:taxonomy_guided:0"]["fixture_source"] == "exp2952_available"

    sources = {source["experiment_id"]: source for source in artifact["source_artifacts"]}
    assert sources["exp2963"]["sha256"] == _sha256(tmp_path / exp.EXP2963_REL_PATH)
    assert sources["exp2952"]["required"] is False


def test_req_code_2965_audit_candidate_blocks_verifier_only_accepts() -> None:
    """REQ-CODE-2965: verifier-threshold accepts still need deterministic gates."""

    schema = exp2951.candidate_manifest_schema()
    unsafe = exp.audit_candidate(
        _candidate(task_id="unsafe", code="import os\ndef target():\n    return os.getcwd()\n"),
        schema=schema,
        verifier_threshold=1.0,
        expected_function_names={"unsafe": "target"},
        fixture_source="unit",
    )
    mismatched = exp.audit_candidate(
        _candidate(task_id="mismatch", code="def other():\n    return 1\n"),
        schema=schema,
        verifier_threshold=1.0,
        expected_function_names={"mismatch": "target"},
        fixture_source="unit",
    )
    missing_expected = exp.audit_candidate(
        _candidate(task_id="unknown", code="def whatever():\n    return 1\n"),
        schema=schema,
        verifier_threshold=1.0,
        expected_function_names={},
        fixture_source="unit",
    )
    allowed_from_import = exp.audit_candidate(
        _candidate(task_id="math", code="from math import sqrt\ndef root(x):\n    return sqrt(x)\n"),
        schema=schema,
        verifier_threshold=1.0,
        expected_function_names={"math": "root"},
        fixture_source="unit",
    )
    malformed = exp.audit_candidate(
        {"task_id": "bad", "repaired_code": "def bad():\n    return 1\n"},
        schema=schema,
        verifier_threshold=1.0,
        expected_function_names={},
        fixture_source="unit",
    )

    assert unsafe["import_allowlist_passed"] is False
    assert unsafe["blocked_prefix_count"] > 0
    assert unsafe["false_accept_audit"]["false_accept"] is True
    assert "unsafe_imports" in unsafe["false_accept_audit"]["reasons"]
    assert mismatched["function_name_preserved"] is False
    assert missing_expected["function_name_status"] == "not_applicable_missing_expected_function"
    assert missing_expected["deterministic_accept"] is True
    assert allowed_from_import["imports_seen"] == ["math"]
    assert allowed_from_import["deterministic_accept"] is True
    assert malformed["schema_valid"] is False
    assert malformed["false_accept_audit"]["verifier_accepted"] is False
    assert exp._verifier_threshold({}) == pytest.approx(1.0)


def test_req_code_2965_blocks_missing_malformed_and_unready_sources(tmp_path: Path) -> None:
    """REQ-CODE-2965: required source problems block certificate readiness honestly."""

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2951_REL_PATH).unlink()
    missing = exp.build_artifact(_config(tmp_path), local_backends=[])
    assert missing["honest_verdict"] == "blocked_missing_required_source_artifact"
    assert missing["beaver_style_certificate_ready"] is False
    assert missing["validation_fixture_count"] == 0

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2953_REL_PATH).write_text("{not-json}\n", encoding="utf-8")
    malformed = exp.build_artifact(_config(tmp_path), local_backends=[])
    assert malformed["honest_verdict"] == "blocked_malformed_source_artifact"
    assert malformed["malformed_source_artifacts"] == ["exp2953"]

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        exp.EXP2963_REL_PATH,
        {
            "artifact": "experiment_2963_dccd_repair_protocol_manifest_v1",
            "dccd_repair_protocol_ready": False,
        },
    )
    unready = exp.build_artifact(_config(tmp_path), local_backends=[])
    assert unready["honest_verdict"] == "blocked_exp2963_protocol_not_ready"
    assert unready["beaver_style_certificate_ready"] is False

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2952_REL_PATH).unlink()
    optional_missing = exp.build_artifact(_config(tmp_path), local_backends=[])
    assert optional_missing["honest_verdict"].startswith("complete:")
    assert optional_missing["available_repair_candidate_count"] == 0
