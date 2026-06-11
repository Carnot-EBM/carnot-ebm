"""Tests for Exp 4037 stronger-base collection and diagnosis.

Spec refs: REQ-VERIFY-4037, SCENARIO-VERIFY-4037.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import exp4037_decentralization_stronger_base_collect as collect


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_artifact(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4036_decentralization_stronger_base_build",
        "schema": "carnot.experiment_4036_decentralization_stronger_base_build.v1",
        "honest_verdict": (
            "success: decentralization_stronger_base_runner_launched_Gemma4-31B-it"
            if ready
            else "blocked_stronger_base_not_cached"
        ),
        "runner_ready": ready,
        "stronger_base_model": "Gemma4-31B-it" if ready else "none",
        "smoke_passed": ready,
        "launched_pid": 123 if ready else 0,
        "preconditions_checked": [{"resource": "stronger_base_gguf_cached", "available": ready}],
        "inference_substrate": "live_llm_inference",
        "duration_s": 1.0,
        "build_artifact_path": str(tmp_path / "build.json"),
        "model_specs": {
            "generator_model": "Gemma4-31B-it" if ready else "none",
            "generator_hf_id": "unsloth/gemma-4-31B-it-GGUF" if ready else "none",
            "generator_gguf_path": "/cache/gemma31.gguf" if ready else "none",
        },
        "random_seed": 12345,
        "reproducibility_checksum": "buildchecksum",
    }


def _task(task: str, perfect: bool, seconds: float = 10.0) -> dict[str, Any]:
    return {
        "task": task,
        "demo_perfect": perfect,
        "best_of_n_demo_perfect": perfect,
        "n_demo_perfect_samples": 1 if perfect else 0,
        "local_seconds": seconds,
        "gated_top2": perfect,
    }


def _raw_artifact(per_task: list[dict[str, Any]], *, pass2: float = 0.5) -> dict[str, Any]:
    coverage = round(
        sum(1 for row in per_task if row.get("best_of_n_demo_perfect")) / max(1, len(per_task)),
        4,
    )
    local_seconds = float(sum(float(row.get("local_seconds", 0.0)) for row in per_task))
    return {
        "experiment": "experiment_4037_decentralization_stronger_base_raw",
        "schema": "carnot.experiment_4037_decentralization_stronger_base_raw.v1",
        "honest_verdict": "complete: decentralization_stronger_base_cov0.5_pass20.5_below_codex",
        "runner_ready": True,
        "stronger_base_model": "Gemma4-31B-it",
        "best_of_n_coverage": coverage,
        "local_demo_perfect_coverage_bestofn": coverage,
        "k_samples_per_task": 8,
        "gated_pass_at_2": pass2,
        "local_gated_pass2": pass2,
        "local_seconds": local_seconds,
        "cost_local_seconds": round(local_seconds / max(1, len(per_task)), 2),
        "cost_codex_seconds_ref": 46.24,
        "per_task": per_task,
        "per_task_sample_summary": [
            {
                "task": row["task"],
                "n_demo_perfect": 1 if row.get("best_of_n_demo_perfect") else 0,
                "local_seconds": row.get("local_seconds", 0.0),
            }
            for row in per_task
        ],
        "missing_verifier_gaps": [
            row["task"] for row in per_task if not row.get("best_of_n_demo_perfect")
        ],
        "model_specs": {
            "generator_model": "Gemma4-31B-it",
            "generator_hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "generator_gguf_path": "/cache/gemma31.gguf",
            "verifier": "model-free GAP-4 verifier primitives reused unchanged",
        },
        "random_seed": 12345,
        "reproducibility_checksum": "rawchecksum",
        "preconditions_checked": [{"resource": "stronger_base_gguf_cached", "available": True}],
        "inference_substrate": "live_llm_inference",
        "duration_s": 120.0,
        "launched_pid": 0,
        "verifier_side_unchanged": True,
    }


def test_req_4037_spec_declared() -> None:
    # REQ-VERIFY-4037: OpenSpec declares the collector before implementation.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4037" in spec
    assert "SCENARIO-VERIFY-4037" in spec
    assert "exp4037_decentralization_stronger_base_collect.py" in spec
    assert "blocked_build_runner_not_ready" in spec


def test_blocked_build_writes_required_terminal_fields(tmp_path: Path) -> None:
    # REQ-VERIFY-4037: a closed build gate stops collection honestly.
    build_path = tmp_path / "build.json"
    output_path = tmp_path / "final.json"
    _write_json(build_path, _build_artifact(tmp_path, ready=False))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=tmp_path / "missing_raw.json",
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "missing_checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=output_path,
        poll_budget_s=0.0,
    )

    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_build_runner_not_ready"
    assert artifact["local_support_diagnosis"] == "absent"
    assert artifact["inference_substrate"] == collect.INFERENCE_SUBSTRATE
    assert output_path.exists()


def test_partial_checkpoint_after_poll_budget_is_not_fabricated(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4037: absent raw output becomes a partial terminal artifact.
    build_path = tmp_path / "build.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(
        checkpoint_path,
        {
            "schema": "carnot.experiment_4012_gap4_local_best_of_n.checkpoint.v1",
            "k_samples_per_task": 8,
            "local_model_used": "Gemma4-31B-it",
            "tasks": {
                "A": [{"demo_perfect": False, "local_s": 12.0}],
                "B": [{"demo_perfect": True, "local_s": 8.0}],
            },
        },
    )

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=tmp_path / "raw.json",
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=checkpoint_path,
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        poll_budget_s=0.0,
    )

    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: decentralization_stronger_base_partial_2_tasks"
    assert artifact["raw_complete"] is False
    assert artifact["partial_task_count"] == 2
    assert artifact["stronger_base_demo_perfect_coverage"] == pytest.approx(0.5)
    assert artifact["missing_verifier_gaps"] == ["A"]


def test_poll_can_collect_raw_that_appears_during_budget(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4037: polling validates a raw artifact that appears before timeout.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))

    def write_raw_after_first_sleep(_seconds: float) -> None:
        _write_json(raw_path, _raw_artifact([_task("A", True), _task("B", True)], pass2=0.7))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        poll_budget_s=1.0,
        poll_interval_s=1.0,
        sleep_fn=write_raw_after_first_sleep,
    )

    assert artifact["raw_complete"] is True
    assert artifact["local_support_diagnosis"] == "latent"
    assert artifact["honest_verdict"].endswith("_latent_distill_viable")


def test_complete_raw_with_lift_diagnoses_latent_support(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4037: positive coverage CI chooses the latent/distill branch.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    output_path = tmp_path / "final.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(raw_path, _raw_artifact([_task("A", True), _task("B", True)], pass2=0.7))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=output_path,
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: decentralization_stronger_base_cov_1_latent_distill_viable"
    )
    assert artifact["stronger_base_demo_perfect_coverage"] == pytest.approx(1.0)
    assert artifact["coverage_delta_vs_12b"] == pytest.approx(0.7419)
    assert artifact["bootstrap_ci95"] == [0.7419, 0.7419]
    assert artifact["pass2_comparison"]["vs_exp4012_12b_gated_pass2"] == pytest.approx(0.2484)
    assert artifact["local_seconds_per_task"] == pytest.approx(10.0)
    assert artifact["codex_seconds_per_task_reference"] == pytest.approx(46.24)


def test_complete_raw_without_lift_diagnoses_absent_support(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4037: flat or lower coverage chooses the absent/leash branch.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(raw_path, _raw_artifact([_task("A", False), _task("B", False)], pass2=0.4))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    assert artifact["honest_verdict"] == (
        "complete: decentralization_stronger_base_cov_0_absent_leash_holds"
    )
    assert artifact["local_support_diagnosis"] == "absent"
    assert artifact["coverage_delta_vs_12b"] == pytest.approx(-0.2581)
    assert artifact["missing_verifier_gaps"] == ["A", "B"]


def test_helper_edges_cover_malformed_inputs_and_empty_statistics(tmp_path: Path) -> None:
    # REQ-VERIFY-4037: malformed cached artifacts remain non-terminal evidence.
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    payload, error = collect._read_json(malformed)
    assert payload is None
    assert error and error.startswith("malformed:")

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    payload, error = collect._read_json(scalar)
    assert payload is None
    assert error and error.endswith("top_level_not_object")

    log_path = tmp_path / "run.log"
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    assert collect._tail_text(log_path, max_lines=2) == ["two", "three"]

    baseline = tmp_path / "baseline.json"
    _write_json(
        baseline,
        {
            "local_demo_perfect_coverage_bestofn": 0.3,
            "local_gated_pass2": 0.4,
            "cost_codex_seconds_ref": 47.0,
            "references": {"oracle_pass2": 0.7, "codex_gated_pass2": 0.6},
        },
    )
    refs = collect._reference_values(baseline)
    assert refs == {
        "coverage_12b": 0.3,
        "pass2_12b": 0.4,
        "oracle_pass2": 0.7,
        "codex_pass2": 0.6,
        "codex_seconds": 47.0,
    }

    assert collect._coverage_from_indicators([]) == 0.0
    assert collect._percentile([], 0.5) == 0.0
    assert collect.bootstrap_delta_ci95([], 0.2581) == [0.0, 0.0]
    assert collect._seconds_per_task([]) == 0.0

    bad_checkpoint = tmp_path / "checkpoint.json"
    _write_json(bad_checkpoint, {"tasks": []})
    assert collect._partial_rows_from_checkpoint(bad_checkpoint) == []


def test_raw_payload_validation_errors_are_reported(tmp_path: Path) -> None:
    # REQ-VERIFY-4037: raw artifacts must pass the existing Exp 4037 schema gate.
    raw_path = tmp_path / "raw.json"
    _write_json(raw_path, {"honest_verdict": "complete: missing_fields"})
    payload, error = collect._complete_raw_payload(raw_path)
    assert payload is None
    assert error and error.startswith("raw_schema_invalid:")

    raw = _raw_artifact([_task("A", True)])
    raw["runner_ready"] = False
    _write_json(raw_path, raw)
    payload, error = collect._complete_raw_payload(raw_path)
    assert payload is None
    assert error == "raw_runner_not_ready"

    raw = _raw_artifact([_task("A", True)])
    raw["honest_verdict"] = "blocked_llama_cpp_unavailable"
    _write_json(raw_path, raw)
    payload, error = collect._complete_raw_payload(raw_path)
    assert payload is None
    assert error == "raw_blocked"

    raw = _raw_artifact([_task("A", True)])
    raw["per_task"] = []
    _write_json(raw_path, raw)
    payload, error = collect._complete_raw_payload(raw_path)
    assert payload is None
    assert error == "raw_has_no_per_task_rows"


def test_complete_raw_without_missing_gap_list_derives_gaps(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4037: missing-verifier gaps can be derived from per-task rows.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))
    raw = _raw_artifact([_task("A", False), _task("B", True)], pass2=0.5)
    del raw["missing_verifier_gaps"]
    _write_json(raw_path, raw)

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    assert artifact["missing_verifier_gaps"] == ["A"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("stronger_base_demo_perfect_coverage", "1.0", "bare float"),
        ("coverage_delta_vs_12b", "0.1", "bare float"),
        ("bootstrap_ci95", [0.1], "2-element"),
        ("local_support_diagnosis", "partial", "latent or absent"),
        ("local_seconds_per_task", "10", "bare float"),
        ("model_specs", [], "must be a dict"),
        ("random_seed", True, "bare int"),
        ("reproducibility_checksum", 123, "must be a string"),
        ("missing_verifier_gaps", {}, "must be a list"),
        ("inference_substrate", "live_llm_inference", "cached candidates"),
    ],
)
def test_validate_final_artifact_rejects_bad_schema(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    # REQ-VERIFY-4037: final artifacts expose typed fields for downstream audit.
    artifact = collect.blocked_build_artifact(
        build_payload=_build_artifact(tmp_path, ready=False),
        output_path=tmp_path / "final.json",
        started_s=0.0,
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        collect.validate_final_artifact(artifact)


def test_validate_final_artifact_rejects_missing_field() -> None:
    # REQ-VERIFY-4037: missing collector fields are never considered terminal.
    with pytest.raises(ValueError, match="missing required field"):
        collect.validate_final_artifact({})
