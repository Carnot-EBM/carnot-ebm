"""Tests for Exp 2963 DCCD structured-repair protocol manifest.

Spec: REQ-CODE-2963, SCENARIO-CODE-2963.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import dccd_repair_protocol_manifest as exp


REQUIRED_FIELDS = {
    "honest_verdict",
    "dccd_repair_protocol_ready",
    "source_artifacts",
    "model_specs",
    "legacy_models_only_for_smoke",
    "n_tasks_planned_min",
    "fixed_seed_plan",
    "dccd_steps",
    "structured_backends_to_check",
    "deterministic_acceptance_checks",
    "false_accept_audit_plan",
    "downstream_gate",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_specs() -> list[dict[str, str]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B-GGUF",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "downstream_live_repair_generation",
        },
        {
            "name": "gemma-4-31B-it-GGUF",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "role": "downstream_live_repair_generation",
        },
        {
            "name": "gemma-4-26B-A4B-it-GGUF",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "role": "downstream_live_repair_generation",
        },
    ]


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        exp.EXP2950_REL_PATH,
        {
            "artifact": "experiment_2950_code_taxonomy_repair_prompt_manifest_v1",
            "honest_verdict": "complete: repair prompt manifest ready",
            "inference_substrate": exp.INFERENCE_SUBSTRATE,
            "repair_prompt_manifest_ready": True,
            "taxonomy_labels": [
                {
                    "label": "syntax_error",
                    "description": "Candidate cannot be parsed as Python after extraction.",
                    "deterministic_checks": ["parser_ast_parse", "function_extraction"],
                    "evidence_count": 373,
                    "sample_ids": ["MBPP:mbpp-11:c0:s2910"],
                },
                {
                    "label": "missing_symbol",
                    "description": "Candidate references a name that is not defined in scope.",
                    "deterministic_checks": ["parser_ast_parse", "static_import_name_checks"],
                    "evidence_count": 2,
                    "sample_ids": ["MBPP:mbpp-56:c2:s2976"],
                },
            ],
            "failure_evidence_summary": {
                "syntax_error": {"evidence_count": 373, "sample_ids": ["MBPP:mbpp-11:c0:s2910"]},
                "missing_symbol": {"evidence_count": 2, "sample_ids": ["MBPP:mbpp-56:c2:s2976"]},
            },
            "deterministic_checks": [
                {
                    "check_id": "parser_ast_parse",
                    "description": "The repaired candidate must parse with Python ast.parse.",
                    "required": True,
                },
                {
                    "check_id": "tests_where_present",
                    "description": "Official or manifest-local tests must pass when provided.",
                    "required": True,
                },
                {
                    "check_id": "exp2940_verifier_threshold",
                    "description": "The verifier score must meet the retained threshold.",
                    "required": True,
                    "threshold": 1.0,
                },
            ],
            "downstream_eval_plan": {
                "may_claim_this_manifest_improves_pass_rate": False,
                "acceptance_criteria": ["Every repair candidate must pass parser_ast_parse."],
            },
            "model_specs": _model_specs(),
        },
    )
    _write_json(
        root,
        exp.EXP2951_REL_PATH,
        {
            "artifact": "experiment_2951_structured_candidate_manifest_adapter_v1",
            "honest_verdict": "complete: structured candidate manifest adapter ready",
            "structured_decode_manifest_ready": True,
            "schema_version": "carnot.structured_candidate_manifest.v1",
            "schema_fields": [
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
            ],
            "candidate_manifest_schema": {
                "type": "object",
                "required": ["task_id", "repaired_code"],
            },
            "local_backends_checked": [
                {"backend_name": "jsonschema", "available": False, "detail": "missing"},
                {"backend_name": "llguidance", "available": False, "detail": "missing"},
                {
                    "backend_name": "llama_cpp_grammar",
                    "available": True,
                    "detail": "LlamaGrammar.from_json_schema available",
                },
            ],
            "model_specs_for_downstream_live_use": _model_specs(),
        },
    )
    _write_json(
        root,
        exp.EXP2952_REL_PATH,
        {
            "artifact": "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1",
            "honest_verdict": "complete: taxonomy-guided repair delta passed",
            "flagged_adversarial": True,
            "n_tasks": 4,
            "sample_budget_per_mode": 16,
            "samples_per_mode": 4,
            "selected_task_ids": [
                "MBPP:mbpp-11",
                "MBPP:mbpp-12",
                "MBPP:mbpp-14",
                "MBPP:mbpp-16",
            ],
            "baseline_pass_at_1": 0.0,
            "repair_pass_at_1": 0.25,
            "baseline_pass_at_k": 0.25,
            "repair_pass_at_k": 0.5,
            "baseline_syntax_failure_rate": 0.6875,
            "repair_syntax_failure_rate": 0.625,
            "repair_schema_failure_rate": 0.0,
            "repair_false_accept_rate": 0.0,
            "false_accept_delta": -0.125,
            "false_accept_audit_notes": [
                "false accepts decreased under taxonomy-guided repair",
                "baseline=0.125, repair=0.0",
            ],
            "corrigendum_pending": [
                {"kind": "TAUTOLOGY", "severity": "critical", "detail": "metric equality warning"},
                {
                    "kind": "METHODOLOGY_MISSING",
                    "severity": "warn",
                    "detail": "missing random_seed",
                },
            ],
            "candidate_evaluations": [
                {
                    "mode": "baseline_no_taxonomy",
                    "sample_id": "MBPP:mbpp-16:c0:s2934",
                    "seed": 2964,
                    "parser_status": "parsed",
                    "test_status": "failed",
                    "verifier_score": 1.0,
                    "verifier_accepted": True,
                    "passed": False,
                    "false_accept": True,
                }
            ],
        },
    )
    _write_json(
        root,
        exp.EXP2953_REL_PATH,
        {
            "artifact": "experiment_2953_code_verifier_threshold_policy_v1",
            "honest_verdict": "complete: threshold policy ready",
            "threshold_policy_ready": True,
            "selected_default_threshold": 1.0,
            "expected_ppv_at_default": 0.8888888888888888,
            "expected_recall_at_default": 1.0,
            "expected_false_accept_rate_at_default": 0.010135135135,
            "deployment_boundary": "not a standalone correctness oracle",
            "operating_points": [
                {
                    "policy_name": "conservative",
                    "threshold": 1.0,
                    "recommended_use": "automated_candidate_filtering",
                    "expected_false_accept_rate": 0.010135135135,
                }
            ],
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 13.25,
        tests_run=("focused-pytest",),
    )


def test_req_code_2963_spec_anchor_exists() -> None:
    """REQ-CODE-2963, SCENARIO-CODE-2963: the DCCD protocol is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2963" in spec
    assert "SCENARIO-CODE-2963" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="aggregation_from_upstream_artifacts"' in spec


def test_scenario_code_2963_builds_ready_protocol_manifest(tmp_path: Path) -> None:
    """SCENARIO-CODE-2963: DCCD manifest pre-registers flow without improvement claims."""

    _write_ready_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["dccd_repair_protocol_ready"] is True
    assert "no pass-rate improvement claimed" in artifact["honest_verdict"]
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["legacy_models_only_for_smoke"] is True
    assert [model["hf_id"] for model in artifact["model_specs"]] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert artifact["n_tasks_planned_min"] == 20
    assert artifact["fixed_seed_plan"] == list(range(296300, 296320))

    step_ids = [step["step_id"] for step in artifact["dccd_steps"]]
    assert step_ids == [
        "unconstrained_semantic_draft",
        "taxonomy_conditioned_repair",
        "constrained_manifest_emission",
        "parser_static_test_checks",
        "verifier_threshold_check",
        "false_accept_audit",
    ]
    assert artifact["extracted_failure_taxonomy"]["syntax_error"]["evidence_count"] == 373
    assert artifact["candidate_schema_summary"]["schema_fields"][-1] == "provenance_checksums"
    assert artifact["threshold_policy"]["selected_default_threshold"] == pytest.approx(1.0)

    backend_names = [
        backend["backend_name"] for backend in artifact["structured_backends_to_check"]
    ]
    assert backend_names == ["llguidance", "llama_cpp_grammar", "json_schema_validation_fallback"]
    check_ids = {check["check_id"] for check in artifact["deterministic_acceptance_checks"]}
    assert {
        "parser_ast_parse",
        "tests_where_present",
        "structured_candidate_schema_validation",
        "verifier_only_accept_rejection",
    } <= check_ids

    false_accept_plan = artifact["false_accept_audit_plan"]
    assert false_accept_plan["upstream_known_false_accept_count"] == 1
    assert false_accept_plan["known_false_accept_constraints"][0]["test_status"] == "failed"
    assert "TAUTOLOGY" in false_accept_plan["corrigendum_kinds_to_audit"]

    gate = artifact["downstream_gate"]
    assert gate["may_claim_pass_rate_improvement"] is False
    assert gate["requires_fresh_live_replication"] is True
    assert gate["selected_default_threshold"] == pytest.approx(1.0)
    assert gate["accounting_buckets"] == [
        "pass_at_1",
        "pass_at_k",
        "syntax_failures",
        "schema_failures",
        "test_failures",
        "verifier_only_accepts",
    ]

    sources = {source["experiment_id"]: source for source in artifact["source_artifacts"]}
    assert sources["exp2950"]["sha256"] == _sha256(tmp_path / exp.EXP2950_REL_PATH)
    assert sources["exp2951"]["sha256"] == _sha256(tmp_path / exp.EXP2951_REL_PATH)
    assert sources["exp2952"]["sha256"] == _sha256(tmp_path / exp.EXP2952_REL_PATH)
    assert sources["exp2953"]["sha256"] == _sha256(tmp_path / exp.EXP2953_REL_PATH)


def test_req_code_2963_blocks_when_required_artifact_is_missing(tmp_path: Path) -> None:
    """REQ-CODE-2963: missing .278 source artifacts prevent protocol readiness."""

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2952_REL_PATH).unlink()

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_missing_required_source_artifact"
    assert artifact["dccd_repair_protocol_ready"] is False
    assert artifact["dccd_steps"] == []
    assert artifact["false_accept_audit_plan"]["upstream_known_false_accept_count"] == 0
    assert artifact["downstream_gate"]["requires_fresh_live_replication"] is True
    assert any(
        source["experiment_id"] == "exp2952" and source["present"] is False
        for source in artifact["source_artifacts"]
    )


def test_req_code_2963_blocks_on_malformed_source_artifact(tmp_path: Path) -> None:
    """REQ-CODE-2963: malformed upstream JSON is reported instead of ignored."""

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2953_REL_PATH).write_text("{not-json}\n", encoding="utf-8")

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_malformed_source_artifact"
    assert artifact["dccd_repair_protocol_ready"] is False
    assert artifact["malformed_source_artifacts"] == ["exp2953"]
    assert artifact["threshold_policy"]["selected_default_threshold"] is None
