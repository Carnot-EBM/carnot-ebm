"""Tests for Exp 2964 SOTA DCCD code-repair replication.

Spec: REQ-CODE-2964, SCENARIO-CODE-2964.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import sota_dccd_repair_replication as exp


REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "model_specs",
    "headline_models_used",
    "legacy_models_only_for_smoke",
    "n_tasks",
    "baseline_pass_at_1",
    "taxonomy_repair_pass_at_1",
    "dccd_repair_pass_at_1",
    "pass_at_1_delta",
    "baseline_pass_at_k",
    "dccd_repair_pass_at_k",
    "pass_at_k_delta",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "false_accept_delta",
    "dccd_repair_replication_clean",
    "candidate_manifest_sha256",
    "reproducibility_checksum",
    "duration_s",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _candidate(stable_id: str, seed: int, index: int = 0) -> dict[str, Any]:
    func = stable_id.replace("mbpp-", "solve_")
    return {
        "candidate_index": index,
        "corpus": "MBPP",
        "error_message": "invalid syntax",
        "error_type": "SyntaxError",
        "executed": True,
        "extracted_code": f"def {func}(:\n",
        "extraction_success": True,
        "raw_response": f"def {func}(:\n",
        "raw_response_path": f"results/raw/{stable_id}_{seed}.txt",
        "raw_response_sha256": "a" * 64,
        "random_seed": seed,
        "row_status": "candidate_syntax_failed",
        "runtime_success": False,
        "stable_id": stable_id,
        "syntax_success": False,
        "passed": False,
    }


def _write_ready_sources(root: Path, *, n_tasks: int = 20, dccd_ready: bool = True) -> None:
    per_task = []
    candidates = []
    for idx in range(n_tasks):
        stable_id = f"mbpp-{idx}"
        per_task.append(
            {
                "stable_id": stable_id,
                "corpus": "MBPP",
                "pass_at_1": 0.0,
                "pass_at_k": 0.0,
                "candidate_count": 2,
            }
        )
        candidates.append(_candidate(stable_id, 2910 + idx, 0))

    _write_json(
        root / exp.EXP2946_REL_PATH,
        {
            "artifact": "experiment_2946_sota_code_generation_continuation_v1",
            "honest_verdict": "complete: retain continuation executed",
            "inference_substrate": "live_llm_inference",
            "protocol_artifact_path": str(exp.NESTED_EXP2946_REL_PATH),
        },
    )
    _write_json(
        root / exp.NESTED_EXP2946_REL_PATH,
        {
            "artifact": "experiment_2946_nested_exp2910_protocol",
            "per_task_results": per_task,
            "candidate_results": candidates,
        },
    )
    _write_json(
        root / exp.EXP2950_REL_PATH,
        {
            "artifact": "experiment_2950_code_taxonomy_repair_prompt_manifest_v1",
            "repair_prompt_manifest_ready": True,
            "repair_prompt_templates": {
                "syntax_error": {
                    "template": (
                        "Taxonomy label: syntax_error\n"
                        "Sample: {sample_id}\n"
                        "Failure evidence: {failure_evidence}\n"
                        "Task context: {task_prompt}\n"
                        "Candidate code:\n{candidate_code}\n"
                        "Return only corrected Python code."
                    )
                }
            },
        },
    )
    _write_json(
        root / exp.EXP2951_REL_PATH,
        {
            "artifact": "experiment_2951_structured_candidate_manifest_adapter_v1",
            "structured_decode_manifest_ready": True,
            "schema_version": "carnot.structured_candidate_manifest.v1",
        },
    )
    _write_json(
        root / exp.EXP2952_REL_PATH,
        {
            "artifact": "experiment_2952_sota_taxonomy_guided_code_repair_eval_v1",
            "n_tasks": 4,
            "flagged_adversarial": True,
            "candidate_evaluations": [],
        },
    )
    _write_json(
        root / exp.EXP2953_REL_PATH,
        {
            "artifact": "experiment_2953_code_verifier_threshold_policy_v1",
            "threshold_policy_ready": True,
            "selected_default_threshold": 1.0,
        },
    )
    _write_json(
        root / exp.EXP2963_REL_PATH,
        {
            "artifact": "experiment_2963_dccd_repair_protocol_manifest_v1",
            "dccd_repair_protocol_ready": dccd_ready,
            "downstream_gate": {"n_tasks_min": 20},
        },
    )


def _task_rows(_config: exp.ExperimentConfig) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for idx in range(25):
        stable_id = f"mbpp-{idx}"
        func = f"solve_{idx}"
        rows[("MBPP", stable_id)] = {
            "stable_id": stable_id,
            "prompt": "Return x plus one.",
            "tests": [f"assert {func}(1) == 2"],
            "test_imports": [],
        }
    return rows


def _preconditions() -> exp.PreconditionReport:
    return exp.PreconditionReport(
        checks=[
            {"resource": "dual_rtx_3090_host", "available": True, "detail": "2x RTX 3090"},
            {"resource": "llama_cpp_runtime", "available": True, "detail": "imported"},
            {"resource": "runsc_sandbox", "available": True, "detail": "runsc"},
            {
                "resource": "cached_sota_pair_or_single_headline_gguf",
                "available": True,
                "detail": "single mandated GGUF resolved",
            },
        ],
        model_specs=[
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": "/models/gemma.gguf",
                "cached": True,
                "selected_for_live_repair": True,
            }
        ],
        runnable_model_specs=[
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 1,
                "model_path": "/models/gemma.gguf",
            }
        ],
    )


def _config(tmp_path: Path, *, n_tasks: int = 20) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        raw_response_dir=tmp_path / "results" / "raw" / "experiment_2964",
        n_tasks=n_tasks,
        samples_per_mode=2,
        started_at=10.0,
        clock=lambda: 75.0,
        tests_run=("focused-pytest",),
    )


def _generator(
    prompt: str,
    seed: int,
    _max_tokens: int,
    model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    match = re.search(r"solve_(\d+)", prompt)
    func = f"solve_{match.group(1) if match else 0}"
    if "mode_id: dccd_structured" in prompt:
        repaired = f"def {func}(x):\n    return x + 1\n"
        text = json.dumps(
            {
                "task_id": "model-filled-task",
                "prompt_id": f"seed-{seed}",
                "model_id": model_spec["hf_id"],
                "raw_completion_ref": "inline",
                "repaired_code": repaired,
                "failure_taxonomy": ["none"],
                "parser_status": "parsed",
                "test_status": "passed",
                "verifier_score": 1.0,
                "provenance_checksums": {
                    "raw_completion_sha256": "0" * 64,
                    "repaired_code_sha256": "1" * 64,
                    "manifest_schema_sha256": "2" * 64,
                },
            }
        )
    elif "Taxonomy label:" in prompt:
        text = f"```python\ndef {func}(x):\n    return x\n```"
    else:
        text = f"```python\ndef {func}(:\n```"
    return exp.GenerationOutcome(
        text=text,
        tokens_generated=32,
        duration_s=1.0,
        backend="fake-live-llama",
        backend_detail="fake",
    )


def _schema_failure_generator(
    prompt: str,
    seed: int,
    max_tokens: int,
    model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    if "mode_id: dccd_structured" in prompt:
        return exp.GenerationOutcome(
            text='{"repaired_code": "def missing_schema(x): return x"}',
            tokens_generated=8,
            duration_s=1.0,
            backend="fake-live-llama",
            backend_detail="fake",
        )
    return _generator(prompt, seed, max_tokens, model_spec)


def _executor(script: str, _timeout: float) -> exp.ExecutionOutcome:
    if "return x + 1" in script:
        return exp.ExecutionOutcome(passed=True)
    if "def " in script:
        return exp.ExecutionOutcome(
            passed=False,
            error_type="AssertionError",
            error_message="wrong answer",
        )
    return exp.ExecutionOutcome(
        passed=False,
        error_type="SyntaxError",
        error_message="invalid syntax",
    )


def test_req_code_2964_spec_anchor_exists() -> None:
    """REQ-CODE-2964, SCENARIO-CODE-2964: Exp 2964 is spec-anchored."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/code-verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-CODE-2964" in spec
    assert "SCENARIO-CODE-2964" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference"' in spec


def test_scenario_code_2964_clean_replication_promotes_only_when_gates_clear(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2964: DCCD promotes only with n>=20 and clean false accepts."""

    _write_ready_sources(tmp_path, n_tasks=20)
    artifact = exp.write_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=_task_rows,
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["n_tasks"] == 20
    assert artifact["baseline_pass_at_1"] == pytest.approx(0.0)
    assert artifact["taxonomy_repair_pass_at_1"] == pytest.approx(0.0)
    assert artifact["dccd_repair_pass_at_1"] == pytest.approx(1.0)
    assert artifact["baseline_pass_at_k"] == pytest.approx(0.0)
    assert artifact["dccd_repair_pass_at_k"] == pytest.approx(1.0)
    assert artifact["pass_at_1_delta"] == pytest.approx(1.0)
    assert artifact["pass_at_k_delta"] == pytest.approx(1.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(-1.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["false_accept_delta"] == pytest.approx(0.0)
    assert artifact["dccd_repair_replication_clean"] is True
    assert artifact["headline_models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["legacy_models_only_for_smoke"] is True
    assert len(artifact["candidate_manifest_sha256"]) == 64
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(artifact["mode_metrics"]) == {
        "baseline_no_taxonomy",
        "taxonomy_guided",
        "dccd_structured",
    }


def test_req_code_2964_blocks_when_protocol_or_model_preconditions_fail(tmp_path: Path) -> None:
    """REQ-CODE-2964: closed gates write a blocked artifact without fake metrics."""

    _write_ready_sources(tmp_path, n_tasks=20, dccd_ready=False)
    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=_task_rows,
    )

    assert artifact["honest_verdict"] == "blocked_preconditions_failed"
    assert artifact["n_tasks"] == 0
    assert artifact["baseline_pass_at_1"] == 0.0
    assert artifact["dccd_repair_pass_at_1"] == 0.0
    assert artifact["dccd_repair_replication_clean"] is False
    assert artifact["candidate_evaluations"] == []
    assert any(
        check["resource"] == "exp2963_dccd_repair_protocol_ready"
        and check["available"] is False
        for check in artifact["preconditions_checked"]
    )

    _write_ready_sources(tmp_path, n_tasks=20, dccd_ready=True)
    no_model = exp.PreconditionReport(
        checks=[{"resource": "headline_gguf_cache", "available": False, "detail": "none"}],
        model_specs=[],
        runnable_model_specs=[],
    )
    model_blocked = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: no_model,
        task_row_provider=_task_rows,
    )

    assert model_blocked["honest_verdict"] == "blocked_preconditions_failed"
    assert model_blocked["headline_models_used"] == []


def test_req_code_2964_schema_failures_do_not_count_as_passes(tmp_path: Path) -> None:
    """REQ-CODE-2964: malformed DCCD manifests stay out of pass metrics."""

    _write_ready_sources(tmp_path, n_tasks=20)
    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=2),
        generator=_schema_failure_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=_task_rows,
    )

    assert artifact["n_tasks"] == 2
    assert artifact["dccd_repair_pass_at_1"] == pytest.approx(0.0)
    assert artifact["mode_metrics"]["dccd_structured"]["schema_failure_rate"] == pytest.approx(1.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(1.0)
    assert artifact["dccd_repair_replication_clean"] is False
    assert all(
        not row["schema_valid"]
        for row in artifact["candidate_evaluations"]
        if row["mode"] == "dccd_structured"
    )


def test_req_code_2964_blocks_when_no_failed_candidates_are_selectable(tmp_path: Path) -> None:
    """REQ-CODE-2964: candidate selection must preserve real failed rows."""

    _write_ready_sources(tmp_path, n_tasks=0)
    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_generator,
        executor=_executor,
        precondition_probe=lambda _config: _preconditions(),
        task_row_provider=_task_rows,
    )

    assert artifact["honest_verdict"] == "blocked_no_failed_or_low_scoring_candidates"
    assert artifact["n_tasks"] == 0
    assert artifact["dccd_repair_replication_clean"] is False


def test_req_code_2964_json_parser_and_audit_note_edges() -> None:
    """REQ-CODE-2964: structured parsing and false-accept notes are deterministic."""

    parsed, errors = exp._parse_json_object('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}
    assert errors == []

    parsed, errors = exp._parse_json_object('prefix {"b": 2} suffix')
    assert parsed == {"b": 2}
    assert errors == []

    parsed, errors = exp._parse_json_object("no object")
    assert parsed is None
    assert errors == ["no JSON object found"]

    parsed, errors = exp._parse_json_object('prefix {"broken": }')
    assert parsed is None
    assert errors == ["invalid JSON object: Expecting value"]

    assert exp._false_accept_notes({"false_accept_delta": -0.1}) == [
        "false accepts decreased under DCCD structured repair"
    ]
    assert exp._false_accept_notes({"false_accept_delta": 0.1}) == [
        "false accepts increased or unavailable; DCCD promotion gate remains closed"
    ]
