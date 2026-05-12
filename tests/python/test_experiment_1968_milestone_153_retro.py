"""Tests for the Exp 1968 milestone .153 retrospective.

Spec traces: REQ-REPORT-009, SCENARIO-REPORT-006.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1968_milestone_153_retro as exp1968


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _minimal_completed(exp_id: int) -> dict:
    return {"status": "complete", "honest_verdict": f"complete: exp{exp_id}_done"}


def _minimal_blocked(exp_id: int) -> dict:
    return {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "duration_s": 0.0,
    }


def _sample_artifacts() -> dict[int, dict]:
    """Return a realistic set of .153 artifacts mirroring the actual run."""
    return {
        1956: {
            "status": "complete",
            "honest_verdict": "complete: nco_negative_constraints_ready",
        },
        1957: {
            "status": "complete",
            "honest_verdict": "Successfully sealed structural budget.",
        },
        1958: {
            "status": "success",
            "honest_verdict": (
                "Successfully prototyped GCoT branching with Carnot "
                "partial-trace energy ranking and backtracking."
            ),
        },
        1959: _minimal_blocked(1959),
        1960: {
            "experiment_id": 1960,
            "spec_refs": ["REQ-SAMPLE-1960"],
            "kl_divergence": 6.931783676147461,
            "verdict": "OK",
            # Note: no status/honest_verdict — counts as failed
        },
        1961: {
            "status": "complete",
            "honest_verdict": "igd_mixed_sampler_benchmark_complete",
        },
        1962: {
            "status": "complete",
            "metrics": {
                "acceleration_factor": 469.46,
                "semantic_retention_verified": True,
            },
        },
        1963: _minimal_blocked(1963),
        1964: _minimal_blocked(1964),
        1965: _minimal_blocked(1965),
        1966: {"_missing": True},
        1967: {
            "schema": "carnot.milestone_pre_retro_audit.v1",
            "milestone": 153,
            "missing_files": ["experiment_1966"],
            "violated_gates": 4,
            "honest_verdict": "Audit failed: missing files or violated gates found.",
        },
    }


def _write_artifacts(results_dir: Path, artifacts: dict[int, dict]) -> None:
    for exp_id, payload in artifacts.items():
        filename = exp1968.EXPERIMENT_FILES[exp_id]
        if not payload.get("_missing"):
            (results_dir / filename).write_text(
                json.dumps(payload), encoding="utf-8"
            )


# ---------------------------------------------------------------------------
# Unit tests — REQ-REPORT-009
# ---------------------------------------------------------------------------

def test_load_artifacts_returns_missing_sentinel_for_absent_files(tmp_path: Path):
    """load_artifacts must return {_missing: True} for every absent file."""
    # Empty results dir — all files absent.
    artifacts = exp1968.load_artifacts(tmp_path)

    assert len(artifacts) == len(exp1968.EXPERIMENT_FILES)
    for exp_id in exp1968.EXPERIMENT_FILES:
        assert artifacts[exp_id].get("_missing") is True


def test_load_artifacts_reads_present_files(tmp_path: Path):
    """load_artifacts reads JSON correctly for files that exist."""
    payload = {"status": "complete", "honest_verdict": "complete: done"}
    (tmp_path / exp1968.EXPERIMENT_FILES[1956]).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    artifacts = exp1968.load_artifacts(tmp_path)

    assert artifacts[1956]["status"] == "complete"
    assert artifacts[1957].get("_missing") is True  # not written


def test_record_honest_verdicts_maps_all_experiments():
    """record_honest_verdicts must return an entry for every .153 experiment."""
    artifacts = _sample_artifacts()
    verdicts = exp1968.record_honest_verdicts(artifacts)

    assert len(verdicts) == len(exp1968.EXPERIMENT_FILES)
    assert verdicts["exp1956"].startswith("complete")
    assert verdicts["exp1966"] == "MISSING"


def test_classify_tasks_correct_partition():
    """classify_tasks must correctly sort into completed / blocked / failed."""
    artifacts = _sample_artifacts()
    completed, blocked, failed = exp1968.classify_tasks(artifacts)

    # exp1956, 1957, 1958, 1961, 1962 are complete/success
    assert 1956 in completed
    assert 1957 in completed
    assert 1958 in completed
    assert 1961 in completed
    assert 1962 in completed

    # exp1959, 1963, 1964, 1965 are blocked
    assert 1959 in blocked
    assert 1963 in blocked
    assert 1964 in blocked
    assert 1965 in blocked

    # exp1960 (no status), exp1966 (missing), exp1967 (pre-retro audit — failed)
    assert 1960 in failed
    assert 1966 in failed


def test_evaluate_criteria_flow_kl_fails_and_core_tasks_pass():
    """evaluate_criteria: core tasks pass; flow KL fails; discipline block passes."""
    artifacts = _sample_artifacts()
    criteria = exp1968.evaluate_criteria(artifacts)

    assert criteria["nco_negative_constraints_shipped"] is True
    assert criteria["truncproof_ll1_shipped"] is True
    assert criteria["gcot_branching_shipped"] is True
    assert criteria["igd_sampler_shipped"] is True
    assert criteria["ni_sampling_shipped"] is True
    assert criteria["flow_sampler_kl_below_threshold"] is False  # KL = 6.93
    assert criteria["continual_routing_blocked_correctly_by_discipline"] is True
    assert criteria["pre_retro_audit_ran"] is True
    assert criteria["gate_contract_gap_surfaced"] is True


def test_evaluate_criteria_flow_kl_passes_when_within_threshold():
    """evaluate_criteria: flow KL criterion flips True when KL < 1.0."""
    artifacts = _sample_artifacts()
    artifacts[1960] = {"kl_divergence": 0.05, "status": "complete"}
    criteria = exp1968.evaluate_criteria(artifacts)
    assert criteria["flow_sampler_kl_below_threshold"] is True


def test_build_artifact_required_schema_fields():
    """build_artifact must include all required schema fields."""
    artifacts = _sample_artifacts()
    artifact = exp1968.build_artifact(artifacts)

    required = {
        "experiment_id",
        "schema",
        "milestone",
        "run_date",
        "status",
        "completed_task_count",
        "blocked_task_count",
        "failed_task_count",
        "completed_experiments",
        "blocked_experiments",
        "failed_experiments",
        "criteria_met",
        "criteria_total",
        "criteria_results",
        "experiment_honest_verdicts",
        "notable_successes",
        "bottlenecks_identified",
        "gate_contract_gap_note",
        "recommendations",
        "retro_complete",
        "honest_verdict",
    }
    assert required.issubset(artifact)


def test_build_artifact_honest_verdict_has_terminal_prefix():
    """honest_verdict must start with 'complete:' per Verdict Terminal-Prefix Discipline."""
    artifact = exp1968.build_artifact(_sample_artifacts())
    hv = artifact["honest_verdict"]
    assert hv.startswith("complete:") or hv.startswith("complete_"), (
        f"honest_verdict missing terminal prefix: {hv!r}"
    )


def test_build_artifact_counts_match():
    """completed + blocked + failed counts must cover all 12 task slots."""
    artifact = exp1968.build_artifact(_sample_artifacts())
    total = (
        artifact["completed_task_count"]
        + artifact["blocked_task_count"]
        + artifact["failed_task_count"]
    )
    assert total == len(exp1968.EXPERIMENT_FILES)


def test_build_artifact_retro_complete_true():
    artifacts = _sample_artifacts()
    artifact = exp1968.build_artifact(artifacts)
    assert artifact["retro_complete"] is True
    assert artifact["status"] == "complete"


def test_build_artifact_bottlenecks_mention_flow_kl_and_exp1966():
    """Bottleneck list must call out the flow KL failure and missing exp1966."""
    artifact = exp1968.build_artifact(_sample_artifacts())
    bottleneck_text = " ".join(artifact["bottlenecks_identified"])
    assert "6.9" in bottleneck_text or "kl" in bottleneck_text.lower()
    assert "1966" in bottleneck_text


def test_build_artifact_recommendations_mention_gate_fix_and_flow_rerun():
    """Recommendations must include gate-field fix and flow sampler rerun."""
    artifact = exp1968.build_artifact(_sample_artifacts())
    rec_text = " ".join(artifact["recommendations"]).lower()
    assert "success" in rec_text  # gate-field fix
    assert "flow" in rec_text     # flow sampler rerun


def test_float_value_safe_conversion():
    """_float_value must return the default on non-numeric inputs."""
    assert exp1968._float_value(3.14) == pytest.approx(3.14)
    assert exp1968._float_value("not-a-number", default=7.5) == pytest.approx(7.5)
    assert exp1968._float_value(None, default=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration test — SCENARIO-REPORT-006
# ---------------------------------------------------------------------------

def test_main_writes_valid_deliverable(tmp_path: Path):
    """main() must write a valid JSON retro artifact to --out path."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_artifacts(results_dir, _sample_artifacts())
    out_path = tmp_path / "experiment_1968_milestone_153_retro.json"

    code = exp1968.main(
        ["--results-dir", str(results_dir), "--out", str(out_path)]
    )

    assert code == 0
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["retro_complete"] is True
    assert artifact["milestone"] == "2026.05.153"
    assert artifact["honest_verdict"].startswith("complete")
    assert artifact["completed_task_count"] >= 5
    assert "flow_sampler_kl_below_threshold" in artifact["criteria_results"]
    assert artifact["criteria_results"]["flow_sampler_kl_below_threshold"] is False
