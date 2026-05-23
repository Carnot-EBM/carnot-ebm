"""Tests for Exp 2934 AquaForte/BEAVER reformulation pipeline.

Spec: REQ-VERIFY-2934, SCENARIO-VERIFY-2934.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import aquaforte_beaver_reformulation_pipeline as exp
from carnot.eval import constraintbench_constrained_output_rerun as exp2926
from carnot.eval import constraintbench_mini_direct_optimization as base


MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _clock(*values: float):
    ticks = iter(values)
    return lambda: next(ticks)


def _cached_pair(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def provider(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": MANDATED,
                "gpu": 1,
                "model_path": "/tmp/gemma.gguf",
            },
        ]

    return provider


def _raw_text_for(index: int, task: base.OptimizationTask) -> str:
    if index % 4 == 0:
        return base.compliant_answer_for_task(task)
    if index % 4 == 1:
        return json.dumps({"solution": base.infeasible_answer_for_task(task)}, sort_keys=True)
    if index % 4 == 2:
        return json.dumps({"solution": base.suboptimal_answer_for_task(task)}, sort_keys=True)
    return "not json"


def _write_exp2926_fixture(
    root: Path,
    *,
    n_rows: int = 12,
    ready: bool = True,
    include_logprobs: bool = False,
) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(exp2926.build_task_manifest()[:n_rows]):
        raw_text = _raw_text_for(index, task)
        raw_path = raw_dir / f"{task.task_id}.json"
        raw_payload: dict[str, Any] = {
            "task_id": task.task_id,
            "model_hf_id": MANDATED,
            "model_name": "Gemma4-26B-A4B-it",
            "raw_response": raw_text,
            "raw_response_sha256": exp2926.sha256_text(raw_text),
        }
        if include_logprobs:
            raw_payload["token_logprobs"] = [{"token": "{", "logprob": -0.01}]
        raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(
            exp2926.evaluate_raw_output(
                task,
                raw_text,
                generation_metadata={
                    "model_hf_id": MANDATED,
                    "model_name": "Gemma4-26B-A4B-it",
                    "model_path": "/tmp/gemma.gguf",
                    "gpu_index": 1,
                    "prompt_hash": exp2926.prompt_hash(task.prompt),
                    "per_task_seed": 2926 + index,
                    "generation_source": "live_sota_llamacpp_prompt_schema",
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": exp2926.sha256_text(raw_text),
                    "elapsed_seconds": 0.1,
                    "blocker": None,
                },
            )
        )

    payload = {
        "honest_verdict": "complete: fixture exp2926",
        "constraintbench_corrigendum_ready": ready,
        "model_specs": [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": MANDATED,
                "gpu": 1,
                "model_path": "/tmp/gemma.gguf",
            }
        ],
        "models_used": [MANDATED],
        "per_task_results": rows,
        "raw_response_dir": str(raw_dir),
        "run_date": "20260523",
    }
    path = root / "results" / exp.EXP2926_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_req_verify_2934_spec_is_declared() -> None:
    """REQ-VERIFY-2934: OpenSpec declares the reformulation contract first."""

    spec = Path("openspec/capabilities/verifiable-reasoning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-2934" in spec
    assert "SCENARIO-VERIFY-2934" in spec
    assert "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1.json" in spec


def test_req_verify_2934_blocks_before_model_resolution_when_exp2926_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2934-1: missing corrigendum writes the required blocked artifact."""

    calls: list[dict[str, Any]] = []
    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            exp2926_path=tmp_path / "missing.json",
            started_at=10.0,
            clock=_clock(12.5),
            tests_run=("focused pytest",),
        ),
        cached_pair_provider=_cached_pair(calls),
    )

    assert calls == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert json.loads((tmp_path / exp.OUTPUT_FILENAME).read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_constraintbench_corrigendum_missing"
    assert artifact["reformulation_pipeline_ready"] is False
    assert artifact["proposal_count"] == 0
    assert artifact["selected_task_ids"] == []
    assert artifact["prefix_bound_available"] is False
    assert artifact["inference_substrate"] == "live_llm_inference_plus_exact_verifier"
    assert artifact["duration_s"] == pytest.approx(2.5)

    not_ready_path = _write_exp2926_fixture(tmp_path / "not_ready", ready=False)
    not_ready = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / "not_ready.json",
            exp2926_path=not_ready_path,
            started_at=1.0,
            clock=_clock(2.0),
        ),
        cached_pair_provider=_cached_pair(calls),
    )
    assert not_ready["honest_verdict"] == "blocked_constraintbench_corrigendum_missing"


def test_scenario_verify_2934_reformulates_retries_and_compares_to_exp2926(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2934: proposals are reformulated, retried, and exact-verified."""

    exp2926_path = _write_exp2926_fixture(tmp_path, n_rows=12)
    calls: list[dict[str, Any]] = []
    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            exp2926_path=exp2926_path,
            selected_count=12,
            started_at=5.0,
            clock=_clock(9.0),
        ),
        cached_pair_provider=_cached_pair(calls),
    )

    persisted = json.loads((tmp_path / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert persisted == artifact
    assert calls == [{"gpu_indices": (0, 1)}]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reformulation_pipeline_ready"] is True
    assert artifact["random_seed"] == 2934
    assert artifact["proposal_count"] == 12
    assert len(artifact["selected_task_ids"]) == 12
    assert artifact["selected_task_ids"] == [
        task.task_id for task in exp2926.build_task_manifest()[:12]
    ]
    assert artifact["model_specs"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["models_used"] == [MANDATED]
    assert artifact["verifier_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["feasibility_delta_vs_exp2926"] == pytest.approx(0.5)
    assert artifact["optimality_delta_vs_exp2926"] == pytest.approx(0.75)
    assert artifact["prefix_bound_available"] is False
    assert artifact["prefix_bound_summary"]["reason"] == "token_logprobs_or_frontier_unavailable"
    assert artifact["raw_response_dir"] == str(tmp_path / "raw")

    first = artifact["per_task_results"][0]
    rejected = artifact["per_task_results"][1]
    syntax_rejected = artifact["per_task_results"][3]
    assert first["initial_verifier"]["accepted"] is True
    assert first["retry"]["attempted"] is False
    assert rejected["initial_verifier"]["accepted"] is False
    assert rejected["retry"]["attempted"] is True
    assert "Exclude the rejected candidate" in rejected["retry"]["prompt"]
    assert rejected["final_verifier"]["accepted"] is True
    assert rejected["final_verifier"]["optimal"] is True
    assert syntax_rejected["reformulation"]["schema_valid"] is False
    assert syntax_rejected["retry"]["attempted"] is True

    expected_checksum = exp.compute_reproducibility_checksum(
        selected_task_ids=artifact["selected_task_ids"],
        model_specs=artifact["model_specs"],
        per_task_results=artifact["per_task_results"],
    )
    assert expected_checksum == artifact["reproducibility_checksum"]
    mutated = [dict(row) for row in artifact["per_task_results"]]
    mutated[0] = {**mutated[0], "final_solution": {"changed": True}}
    assert (
        exp.compute_reproducibility_checksum(
            selected_task_ids=artifact["selected_task_ids"],
            model_specs=artifact["model_specs"],
            per_task_results=mutated,
        )
        != artifact["reproducibility_checksum"]
    )


def test_req_verify_2934_prefix_audit_activates_only_with_frontier_evidence(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2934-5: prefix-bound availability follows local frontier evidence."""

    exp2926_path = _write_exp2926_fixture(tmp_path, n_rows=12, include_logprobs=True)
    calls: list[dict[str, Any]] = []
    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            exp2926_path=exp2926_path,
            selected_count=12,
            started_at=20.0,
            clock=_clock(21.0),
        ),
        cached_pair_provider=_cached_pair(calls),
    )

    assert artifact["prefix_bound_available"] is True
    assert artifact["prefix_bound_summary"]["frontier_rows"] == 12
    assert artifact["prefix_bound_summary"]["audited_tasks"] == 12
    assert artifact["prefix_bound_summary"]["prefix_violations"] == 3
    assert artifact["prefix_bound_summary"]["constraint"] == (
        "first_non_ws_token_must_open_json_object"
    )


def test_req_verify_2934_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-2934: helper edge cases stay deterministic and mandated."""

    def cache_raises(**_kwargs: Any) -> None:
        raise RuntimeError("cache unavailable")

    inherited_payload = {
        "model_specs": [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": MANDATED,
                "gpu": 1,
                "model_path": "/tmp/gemma.gguf",
            }
        ]
    }
    inherited, error = exp.resolve_model_specs(
        inherited_payload,
        cached_pair_provider=cache_raises,
    )
    fallback, fallback_error = exp.resolve_model_specs(
        {"model_specs": []},
        cached_pair_provider=lambda **_: [],
    )

    assert inherited == inherited_payload["model_specs"]
    assert error == "RuntimeError: cache unavailable"
    assert fallback_error is None
    assert {spec["hf_id"] for spec in fallback} == set(base.MANDATED_MODEL_IDS)

    with pytest.raises(ValueError, match="between 12 and 20"):
        exp.select_task_rows({"per_task_results": []}, 11)

    source_path = tmp_path / "results" / exp.EXP2926_FILENAME
    source_path.parent.mkdir(parents=True)
    missing_raw = exp.load_raw_payload(
        {"task_id": "missing", "raw_response_sha256": "abc"},
        source_path,
    )
    relative_raw_path = tmp_path / "raw" / "relative.json"
    relative_raw_path.parent.mkdir(parents=True)
    relative_raw_path.write_text(
        json.dumps({"raw_response": "{}", "frontier": []}),
        encoding="utf-8",
    )
    relative_raw = exp.load_raw_payload(
        {"raw_response_path": "raw/relative.json"},
        source_path,
    )

    assert missing_raw["task_id"] == "missing"
    assert relative_raw["raw_response"] == "{}"
    assert exp._resolve_raw_path(None, source_path) is None
    assert exp._resolve_raw_path(Path(exp2926.REPO_ROOT / "results").name, source_path)

    models = exp.resolve_models_used(
        {"models_used": []},
        [{"model_hf_id": MANDATED}, {"model_hf_id": MANDATED}, {"model_hf_id": "legacy"}],
    )
    assert models == [MANDATED]
