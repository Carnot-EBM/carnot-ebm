"""Tests for Exp 2950 code taxonomy repair-prompt manifest.

Spec: REQ-CODE-2950, SCENARIO-CODE-2950.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import code_taxonomy_repair_prompt_manifest as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ready_sources(tmp_path: Path) -> None:
    _write_json(
        tmp_path / exp.EXP2940_REL_PATH,
        {
            "artifact": "experiment_2940_verifier_ensemble_auprc_code_corpora_v1",
            "code_corpus_auprc": 0.8888888888888888,
            "max_f1_operating_point": {"threshold": 1.0, "ppv": 0.8888888888888888},
            "paper_v6_recommendation": {"value": "retain"},
            "ppv_50_operating_point": {"threshold": 0.5},
        },
    )
    _write_json(
        tmp_path / exp.EXP2943_REL_PATH,
        {
            "artifact": "experiment_2943_cross_corpus_matrix_v11",
            "matrix_v11_ready": True,
            "per_corpus_auprc": {
                "code_corpora": {
                    "source_experiment_id": "exp2940",
                    "value": 0.8888888888888888,
                }
            },
            "rows_clean": ["exp2940_code_corpus_auprc_corrigendum"],
            "rows_flagged": ["exp2911_code_hallucination_verifier"],
        },
    )
    _write_json(
        tmp_path / exp.EXP2946_REL_PATH,
        {
            "artifact": "experiment_2946_sota_code_generation_continuation_v1",
            "honest_verdict": "complete: pass@1=0.0600, pass@k=0.1600",
            "inference_substrate": "live_llm_inference",
            "pass_at_1": 0.06,
            "pass_at_k": 0.16,
            "protocol_artifact_path": str(exp.NESTED_EXP2946_REL_PATH),
        },
    )
    _write_json(
        tmp_path / exp.NESTED_EXP2946_REL_PATH,
        {
            "artifact": "experiment_2946_nested_exp2910_protocol",
            "candidate_results": [
                {
                    "candidate_index": 0,
                    "corpus": "MBPP",
                    "error_message": "invalid syntax",
                    "error_type": "SyntaxError",
                    "extracted_code": "def bad(:\n",
                    "passed": False,
                    "random_seed": 2910,
                    "row_status": "candidate_syntax_failed",
                    "runtime_success": False,
                    "stable_id": "mbpp-1",
                    "syntax_success": False,
                },
                {
                    "candidate_index": 1,
                    "corpus": "MBPP",
                    "error_message": "name 'missing_total' is not defined",
                    "error_type": "NameError",
                    "extracted_code": "def add(a, b):\n    return missing_total\n",
                    "passed": False,
                    "random_seed": 2911,
                    "row_status": "candidate_failed",
                    "runtime_success": False,
                    "stable_id": "mbpp-2",
                    "syntax_success": True,
                },
                {
                    "candidate_index": 2,
                    "corpus": "MBPP",
                    "error_message": "AssertionError: expected [1, 2] got '1,2'",
                    "error_type": "AssertionError",
                    "extracted_code": "def values():\n    return '1,2'\n",
                    "passed": False,
                    "random_seed": 2912,
                    "row_status": "candidate_failed",
                    "runtime_success": True,
                    "stable_id": "mbpp-3",
                    "syntax_success": True,
                },
                {
                    "candidate_index": 3,
                    "corpus": "HumanEval",
                    "error_message": "wrong return type: expected list got str",
                    "error_type": "TypeError",
                    "extracted_code": "def solve(xs):\n    return ','.join(xs)\n",
                    "passed": False,
                    "random_seed": 2913,
                    "row_status": "candidate_failed",
                    "runtime_success": False,
                    "stable_id": "HumanEval/1",
                    "syntax_success": True,
                },
                {
                    "candidate_index": 4,
                    "corpus": "HumanEval",
                    "error_message": "unsafe import subprocess",
                    "error_type": "SecurityError",
                    "extracted_code": (
                        "import subprocess\n"
                        "def solve(x):\n"
                        "    return subprocess.run(['echo', str(x)])\n"
                    ),
                    "passed": False,
                    "random_seed": 2914,
                    "row_status": "candidate_failed",
                    "runtime_success": False,
                    "stable_id": "HumanEval/2",
                    "syntax_success": True,
                },
                {
                    "candidate_index": 5,
                    "corpus": "HumanEval",
                    "error_message": "module 'json' has no attribute 'parse'",
                    "error_type": "AttributeError",
                    "extracted_code": "import json\ndef solve(text):\n    return json.parse(text)\n",
                    "passed": False,
                    "random_seed": 2915,
                    "row_status": "candidate_failed",
                    "runtime_success": False,
                    "stable_id": "HumanEval/3",
                    "syntax_success": True,
                },
            ],
            "per_task_results": [
                {"stable_id": "mbpp-1", "pass_at_k": 0.0},
                {"stable_id": "HumanEval/3", "pass_at_k": 0.0},
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


def test_req_code_2950_spec_anchor_exists() -> None:
    """REQ-CODE-2950, SCENARIO-CODE-2950: Exp 2950 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2950" in spec
    assert "SCENARIO-CODE-2950" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_scenario_code_2950_builds_ready_manifest_from_upstream_failures(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2950: manifest groups failures without claiming improvement."""

    _write_ready_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["repair_prompt_manifest_ready"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert "no pass-rate improvement claimed" in artifact["honest_verdict"]
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["legacy_models_only_for_smoke"] is True
    assert [model["hf_id"] for model in artifact["model_specs"]] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]

    labels = {row["label"]: row for row in artifact["taxonomy_labels"]}
    assert set(labels) == set(exp.TAXONOMY_LABELS)
    assert labels["syntax_error"]["sample_ids"] == ["MBPP:mbpp-1:c0:s2910"]
    assert labels["missing_symbol"]["sample_ids"] == ["MBPP:mbpp-2:c1:s2911"]
    assert labels["failed_tests"]["sample_ids"] == ["MBPP:mbpp-3:c2:s2912"]
    assert labels["wrong_return_type"]["sample_ids"] == ["HumanEval:HumanEval/1:c3:s2913"]
    assert labels["unsafe_import"]["sample_ids"] == ["HumanEval:HumanEval/2:c4:s2914"]
    assert labels["unsupported_api_hallucination"]["sample_ids"] == [
        "HumanEval:HumanEval/3:c5:s2915"
    ]

    assert set(artifact["repair_prompt_templates"]) == set(exp.TAXONOMY_LABELS)
    for label, template in artifact["repair_prompt_templates"].items():
        assert label in template["template"]
        assert "Do not introduce new imports" in template["template"]

    check_ids = {check["check_id"]: check for check in artifact["deterministic_checks"]}
    assert check_ids["parser_ast_parse"]["required"] is True
    assert check_ids["exp2940_verifier_threshold"]["threshold"] == pytest.approx(1.0)
    assert check_ids["exp2940_verifier_threshold"]["source_artifact"].endswith(
        "experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json"
    )
    assert artifact["downstream_eval_plan"]["may_claim_this_manifest_improves_pass_rate"] is False
    assert artifact["downstream_eval_plan"]["acceptance_criteria"]
    assert artifact["upstream_metrics"]["pass_at_1"] == pytest.approx(0.06)
    assert artifact["upstream_metrics"]["pass_at_k"] == pytest.approx(0.16)


def test_req_code_2950_blocks_when_required_upstream_artifact_is_missing(
    tmp_path: Path,
) -> None:
    """REQ-CODE-2950: required upstream artifacts must exist before readiness."""

    _write_ready_sources(tmp_path)
    (tmp_path / exp.EXP2943_REL_PATH).unlink()

    artifact = exp.build_artifact(_config(tmp_path))

    assert artifact["honest_verdict"] == "blocked_upstream_artifact_missing"
    assert artifact["repair_prompt_manifest_ready"] is False
    assert artifact["taxonomy_labels"] == []
    assert artifact["repair_prompt_templates"] == {}
    assert artifact["downstream_eval_plan"]["blocked_reason"] == "missing_required_source"
    assert any(
        source["path"].endswith("experiment_2943_cross_corpus_matrix_v11.json")
        and source["present"] is False
        for source in artifact["source_artifacts"]
    )
