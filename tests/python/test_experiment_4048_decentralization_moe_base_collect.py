"""Tests for Exp 4048 MoE-base collection and diagnosis.

Spec refs: REQ-VERIFY-4048, SCENARIO-VERIFY-4048.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import exp4048_decentralization_moe_base_collect as collect


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_artifact(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4047_decentralization_moe_base_build",
        "schema": "carnot.experiment_4047_decentralization_moe_base_build.v1",
        "honest_verdict": (
            "success: decentralization_moe_base_runner_launched_qwen35moe"
            if ready
            else "blocked_moe_base_not_cached"
        ),
        "runner_ready": ready,
        "moe_base_model": "Qwen3.6-35B-A3B" if ready else "none",
        "smoke_per_task_seconds": 10.0 if ready else 0.0,
        "smoke_passed": ready,
        "launched_pid": 123 if ready else 0,
        "preconditions_checked": [{"resource": "moe_base_gguf_cached", "available": ready}],
        "inference_substrate": "live_llm_inference",
        "duration_s": 1.0,
        "build_artifact_path": str(tmp_path / "build.json"),
        "model_specs": {
            "generator_model": "Qwen3.6-35B-A3B" if ready else "none",
            "generator_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF" if ready else "none",
            "generator_gguf_path": "/cache/qwen.gguf" if ready else "none",
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
        "experiment": "experiment_4048_decentralization_moe_base_raw",
        "schema": "carnot.experiment_4048_decentralization_moe_base_raw.v1",
        "honest_verdict": "complete: decentralization_moe_base_cov0.5_pass20.5_absent_or_flat",
        "runner_ready": True,
        "moe_base_model": "Qwen3.6-35B-A3B",
        "best_of_n_coverage": coverage,
        "local_demo_perfect_coverage_bestofn": coverage,
        "k_samples_per_task": 8,
        "gated_pass_at_2": pass2,
        "local_gated_pass2": pass2,
        "local_seconds": local_seconds,
        "cost_local_seconds": round(local_seconds / max(1, len(per_task)), 2),
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
            "generator_model": "Qwen3.6-35B-A3B",
            "generator_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "generator_gguf_path": "/cache/qwen.gguf",
            "verifier": "model-free GAP-4 verifier primitives reused unchanged",
        },
        "random_seed": 12345,
        "reproducibility_checksum": "rawchecksum",
        "preconditions_checked": [{"resource": "moe_base_gguf_cached", "available": True}],
        "inference_substrate": "live_llm_inference",
        "duration_s": 120.0,
        "launched_pid": 0,
        "verifier_side_unchanged": True,
    }


def _baseline(path: Path, *, coverage: float = 0.2581, oracle: float = 0.6129) -> None:
    _write_json(
        path,
        {
            "local_demo_perfect_coverage_bestofn": coverage,
            "local_gated_pass2": 0.4516,
            "cost_codex_seconds_ref": 46.24,
            "oracle_coverage": oracle,
            "references": {"oracle_pass2": oracle, "codex_gated_pass2": 0.5806},
        },
    )


def test_req_4048_spec_declared() -> None:
    # REQ-VERIFY-4048: OpenSpec declares the collector before implementation.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4048" in spec
    assert "SCENARIO-VERIFY-4048" in spec
    assert "exp4048_decentralization_moe_base_collect.py" in spec
    assert "decentralization_moe_base_partial_<n>_tasks_retire" in spec


def test_blocked_build_writes_required_terminal_fields(tmp_path: Path) -> None:
    # REQ-VERIFY-4048: a closed build gate stops collection honestly.
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
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=0.0,
    )

    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_build_runner_not_ready"
    assert artifact["local_support_diagnosis"] == "uninformative"
    assert artifact["n_tasks_scored"] == 0
    assert artifact["inference_substrate"] == collect.INFERENCE_SUBSTRATE
    assert output_path.exists()


def test_partial_checkpoint_under_task_floor_retires_and_logs_gap(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4048: absent raw output with <30 scored tasks is terminal retirement.
    build_path = tmp_path / "build.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(
        checkpoint_path,
        {
            "schema": "carnot.experiment_4048_decentralization_moe_base_raw.checkpoint.v1",
            "k_samples_per_task": 8,
            "local_model_used": "Qwen3.6-35B-A3B",
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
        gaps_path=gaps_path,
        poll_budget_s=0.0,
    )

    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: decentralization_moe_base_partial_2_tasks_retire"
    assert artifact["raw_complete"] is False
    assert artifact["n_tasks_scored"] == 2
    assert artifact["moe_base_demo_perfect_coverage"] == pytest.approx(0.5)
    assert artifact["local_support_diagnosis"] == "uninformative"
    assert artifact["missing_verifier_gaps"] == ["A"]
    assert "GAP-DECENTRALIZATION-MOE-BASE-4048" in gaps_path.read_text(encoding="utf-8")
    assert collect.record_verifier_gaps(gaps_path, artifact) is False


def test_poll_can_collect_raw_that_appears_during_budget(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4048: polling validates a raw artifact that appears before timeout.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))

    def write_raw_after_first_sleep(_seconds: float) -> None:
        rows = [_task(f"T{i}", True) for i in range(30)]
        _write_json(raw_path, _raw_artifact(rows, pass2=0.7))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=1.0,
        poll_interval_s=1.0,
        sleep_fn=write_raw_after_first_sleep,
        n_bootstrap=64,
    )

    assert artifact["raw_complete"] is True
    assert artifact["local_support_diagnosis"] == "latent"
    assert artifact["honest_verdict"].endswith("_latent_distill_viable")
    assert artifact["n_tasks_scored"] == 30


def test_complete_raw_with_lift_diagnoses_latent_support(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4048: positive coverage CI chooses the latent/distill branch.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    baseline_path = tmp_path / "baseline.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(raw_path, _raw_artifact([_task(f"T{i}", True) for i in range(31)], pass2=0.7))
    _baseline(baseline_path)

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=baseline_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    collect.validate_final_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: decentralization_moe_base_cov_1_latent_distill_viable"
    )
    assert artifact["moe_base_demo_perfect_coverage"] == pytest.approx(1.0)
    assert artifact["coverage_delta_vs_12b"] == pytest.approx(0.7419)
    assert artifact["bootstrap_ci95"] == [0.7419, 0.7419]
    assert artifact["oracle_coverage"] == pytest.approx(0.6129)
    assert artifact["pass2_comparison"]["vs_exp4012_12b_gated_pass2"] == pytest.approx(0.2484)
    assert artifact["local_seconds_per_task"] == pytest.approx(10.0)
    assert artifact["codex_seconds_per_task_reference"] == pytest.approx(46.24)


def test_complete_raw_without_lift_diagnoses_absent_support(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4048: flat or lower coverage chooses the absent/leash branch.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(raw_path, _raw_artifact([_task(f"T{i}", False) for i in range(31)], pass2=0.4))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    assert artifact["honest_verdict"] == "complete: decentralization_moe_base_cov_0_absent_leash_holds"
    assert artifact["local_support_diagnosis"] == "absent"
    assert artifact["coverage_delta_vs_12b"] == pytest.approx(-0.2581)
    assert artifact["missing_verifier_gaps"] == [f"T{i}" for i in range(31)]


def test_complete_raw_under_task_floor_is_retirement_not_diagnosis(tmp_path: Path) -> None:
    # REQ-VERIFY-4048: fewer than 30 raw rows is a non-measurement retirement trigger.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(raw_path, _raw_artifact([_task(f"T{i}", True) for i in range(29)], pass2=0.9))

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    assert artifact["honest_verdict"] == "complete: decentralization_moe_base_partial_29_tasks_retire"
    assert artifact["local_support_diagnosis"] == "uninformative"
    assert artifact["raw_complete"] is True


def test_saturated_oracle_makes_complete_measurement_uninformative(tmp_path: Path) -> None:
    # REQ-VERIFY-4048: a saturated pool positive control blocks latent/absent interpretation.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    baseline_path = tmp_path / "baseline.json"
    _write_json(build_path, _build_artifact(tmp_path))
    _write_json(raw_path, _raw_artifact([_task(f"T{i}", True) for i in range(31)], pass2=0.7))
    _baseline(baseline_path, coverage=0.2581, oracle=0.2581)

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=baseline_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    assert artifact["local_support_diagnosis"] == "uninformative"
    assert artifact["honest_verdict"] == (
        "complete: decentralization_moe_base_cov_1_uninformative_saturated_pool"
    )


def test_helper_edges_cover_malformed_inputs_and_empty_statistics(tmp_path: Path) -> None:
    # REQ-VERIFY-4048: malformed cached artifacts remain non-terminal evidence.
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
            "per_task": [{"oracle_hit": True}, {"oracle_hit": False}],
            "references": {"codex_gated_pass2": 0.6},
        },
    )
    refs = collect._reference_values(baseline)
    assert refs == {
        "coverage_12b": 0.3,
        "pass2_12b": 0.4,
        "oracle_coverage": 0.5,
        "codex_pass2": 0.6,
        "codex_seconds": 47.0,
    }
    assert collect._oracle_from_baseline({"references": {"oracle_pass2": 0.7}}) == 0.7
    assert collect._oracle_from_baseline({"references": []}) == collect.DEFAULT_ORACLE_COVERAGE

    non_dict_refs = tmp_path / "non_dict_refs.json"
    _write_json(
        non_dict_refs,
        {
            "local_demo_perfect_coverage_bestofn": 0.31,
            "local_gated_pass2": 0.41,
            "cost_codex_seconds_ref": 48.0,
            "oracle_coverage": 0.62,
            "references": [],
        },
    )
    assert collect._reference_values(non_dict_refs)["codex_pass2"] == collect.DEFAULT_CODEX_PASS2

    assert collect._coverage_from_indicators([]) == 0.0
    assert collect._percentile([], 0.5) == 0.0
    assert collect.bootstrap_delta_ci95([], 0.2581) == [0.0, 0.0]
    assert collect._seconds_per_task([]) == 0.0
    assert collect._diagnosis_from_ci([0.1, 0.2], raw_complete=True, oracle_saturated=False) == "latent"

    bad_checkpoint = tmp_path / "checkpoint.json"
    _write_json(bad_checkpoint, {"tasks": []})
    assert collect._partial_rows_from_checkpoint(bad_checkpoint) == []


def test_raw_payload_validation_errors_are_reported(tmp_path: Path) -> None:
    # REQ-VERIFY-4048: raw artifacts must pass the existing Exp 4048 schema gate.
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
    # SCENARIO-VERIFY-4048: missing-verifier gaps can be derived from per-task rows.
    build_path = tmp_path / "build.json"
    raw_path = tmp_path / "raw.json"
    _write_json(build_path, _build_artifact(tmp_path))
    raw = _raw_artifact([_task("A", False), *[_task(f"T{i}", True) for i in range(30)]], pass2=0.5)
    del raw["missing_verifier_gaps"]
    _write_json(raw_path, raw)

    artifact = collect.run_collection(
        build_path=build_path,
        raw_path=raw_path,
        baseline_path=tmp_path / "missing_4012.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        log_path=tmp_path / "missing.log",
        output_path=tmp_path / "final.json",
        gaps_path=tmp_path / "gaps.md",
        poll_budget_s=0.0,
        n_bootstrap=64,
    )

    assert artifact["missing_verifier_gaps"] == ["A"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("moe_base_demo_perfect_coverage", "1.0", "bare float"),
        ("coverage_delta_vs_12b", "0.1", "bare float"),
        ("bootstrap_ci95", [0.1], "2-element"),
        ("n_tasks_scored", True, "bare int"),
        ("oracle_coverage", "0.6", "bare float"),
        ("local_support_diagnosis", "partial", "latent, absent, or uninformative"),
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
    # REQ-VERIFY-4048: final artifacts expose typed fields for downstream audit.
    artifact = collect.blocked_build_artifact(
        build_payload=_build_artifact(tmp_path, ready=False),
        output_path=tmp_path / "final.json",
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        collect.validate_final_artifact(artifact)


def test_validate_final_artifact_rejects_missing_field() -> None:
    # REQ-VERIFY-4048: missing collector fields are never considered terminal.
    with pytest.raises(ValueError, match="missing required field"):
        collect.validate_final_artifact({})
