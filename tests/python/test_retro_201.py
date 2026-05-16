"""
Tests for the milestone 2026.05.201 retrospective generator.

REQ-REPORT-201: The retro generator MUST scan experiment_2001-2013 artifacts,
classify them as completed/blocked/failed, and emit a carnot.milestone_retro.v1
artifact with correct counts, verdicts, at least one recommendation, and
adversarial flags for suspicious artifacts.

SCENARIO-REPORT-201-A: Milestone .201 Retrospective Correctly Counts
Blocked (doomed-rerun) and Completed Tasks

SCENARIO-REPORT-201-B: Gate-Schema Mismatch Causes Blocked Cascade

SCENARIO-REPORT-201-C: Adversarial Flags Populated for Tautology and
Duration-Too-Short Artifacts
"""

import json
from pathlib import Path

from carnot.retro_201 import (
    generate_retro,
    _classify_artifact,
    _adversarial_flags,
    _DELIVERABLES,
)


# ─── unit: _classify_artifact ───────────────────────────────────────────────

def test_classify_blocked_via_honest_verdict():
    # REQ-REPORT-201: artifacts with "blocked" in honest_verdict are BLOCKED
    assert _classify_artifact({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"


def test_classify_blocked_via_status():
    assert _classify_artifact({"status": "blocked"}) == "blocked"


def test_classify_failed_via_verdict():
    assert _classify_artifact({"honest_verdict": "failed_error"}) == "failed"
    assert _classify_artifact({"honest_verdict": "missing artifacts"}) == "failed"


def test_classify_failed_via_status():
    assert _classify_artifact({"status": "failure"}) == "failed"
    assert _classify_artifact({"status": "error"}) == "failed"


def test_classify_completed_on_complete_verdict():
    assert _classify_artifact({"honest_verdict": "complete: done"}) == "completed"


def test_classify_completed_on_success_prefix():
    assert _classify_artifact({"honest_verdict": "success_csr_metric_evaluated"}) == "completed"


def test_classify_completed_on_success_status():
    assert _classify_artifact({"status": "success"}) == "completed"


def test_classify_completed_default():
    # When no negative signal exists, default to completed
    assert _classify_artifact({}) == "completed"


# ─── unit: _adversarial_flags ────────────────────────────────────────────────

def test_adversarial_flags_tautology():
    # SCENARIO-REPORT-201-C: bit-identical cpu vs mock_gpu is a TAUTOLOGY
    artifact = {
        "cpu_counts": [100, 200, 300],
        "mock_gpu_counts": [100, 200, 300],
        "divergence": 0.0,
    }
    flags = _adversarial_flags(2011, artifact)
    assert any("TAUTOLOGY" in f for f in flags)


def test_adversarial_flags_duration_too_short():
    # A task referencing CUDA but finishing in 0s should be flagged
    artifact = {
        "status": "success",
        "duration_s": 0.0,
        "cuda_device": "RTX3090",
    }
    flags = _adversarial_flags(2012, artifact)
    assert any("DURATION_TOO_SHORT" in f for f in flags)


def test_adversarial_flags_implausible_perfect_z3():
    # SCENARIO-REPORT-201-C: 100/100 Z3 in <1 s = implausible perfect
    artifact = {
        "status": "success",
        "n_puzzles_generated": 100,
        "n_puzzles_verified": 100,
        "duration_s": 0.07,
    }
    flags = _adversarial_flags(2005, artifact)
    assert any("IMPLAUSIBLE_PERFECT" in f for f in flags)


def test_adversarial_flags_clean_artifact():
    # A real benchmark artifact should generate no flags
    artifact = {
        "status": "success",
        "csr_value": 0.43,
        "energy_correlation": 0.94,
    }
    flags = _adversarial_flags(2003, artifact)
    assert flags == []


# ─── integration: generate_retro ─────────────────────────────────────────────

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_generate_retro_schema_fields(tmp_path: Path) -> None:
    # REQ-REPORT-201: output must use carnot.milestone_retro.v1 schema
    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.201"
    assert result["experiment_id"] == 2014
    assert result["status"] == "complete"
    assert result["retro_complete"] is True
    assert result["honest_verdict"].startswith("complete:")


def test_generate_retro_blocked_tasks(tmp_path: Path) -> None:
    # SCENARIO-REPORT-201-A: four tasks blocked by doomed-rerun
    # Must write files using the known deliverable names so the module finds them.
    for exp_id in [2001, 2002, 2004, 2007]:
        _write(
            tmp_path / _DELIVERABLES[exp_id],
            {
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "blocked_at_layer": "conductor_pre_gate",
            },
        )
    for exp_id in [2003, 2008]:
        _write(
            tmp_path / _DELIVERABLES[exp_id],
            {"status": "success", "honest_verdict": "complete: done"},
        )

    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["blocked_task_count"] == 4
    for exp_id in [2001, 2002, 2004, 2007]:
        assert exp_id in result["blocked_experiments"]
    assert 2003 in result["completed_experiments"]
    assert 2008 in result["completed_experiments"]


def test_generate_retro_missing_artifacts(tmp_path: Path) -> None:
    # Tasks with no JSON artifact are classified as MISSING (failed)
    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    # All 2001-2013 are missing → all failed
    assert result["failed_task_count"] == 13
    assert 2001 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp2001"] == "MISSING"


def test_generate_retro_unreadable_artifact(tmp_path: Path) -> None:
    broken = tmp_path / _DELIVERABLES[2005]
    broken.write_text("{ not valid json >>>")

    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 2005 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp2005"] == "UNREADABLE"


def test_generate_retro_adversarial_flags_populated(tmp_path: Path) -> None:
    # SCENARIO-REPORT-201-C: tautology artifact generates flag in output
    _write(
        tmp_path / _DELIVERABLES[2011],
        {
            "status": "complete",
            "cpu_counts": [100, 200],
            "mock_gpu_counts": [100, 200],
            "divergence": 0.0,
        },
    )

    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "exp2011" in result["adversarial_flags"]
    assert any("TAUTOLOGY" in f for f in result["adversarial_flags"]["exp2011"])


def test_generate_retro_bottlenecks_and_recommendations(tmp_path: Path) -> None:
    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) >= 3
    assert isinstance(result["bottlenecks_identified"], list)
    assert len(result["bottlenecks_identified"]) >= 2


def test_generate_retro_verdict_terminal_prefix(tmp_path: Path) -> None:
    # REQ-REPORT-201: honest_verdict must start with a terminal prefix
    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    verdict = result["honest_verdict"]
    valid_prefixes = ("complete:", "complete_", "success:", "success_", "passed:", "shipped:")
    assert any(verdict.startswith(p) for p in valid_prefixes), (
        f"honest_verdict does not start with a terminal prefix: {verdict!r}"
    )


def test_generate_retro_writes_file(tmp_path: Path) -> None:
    out = str(tmp_path / "experiment_2014_retro.json")
    generate_retro(out, results_dir=str(tmp_path))

    with open(out) as fh:
        on_disk = json.load(fh)

    assert on_disk["schema"] == "carnot.milestone_retro.v1"
    assert on_disk["experiment_id"] == 2014


def test_generate_retro_ignores_out_of_range(tmp_path: Path) -> None:
    # Experiments outside 2001-2013 must not be counted even if files exist.
    # Write some decoy files at exp numbers outside the range.
    _write(tmp_path / "experiment_2000_some_task.json", {"status": "success"})
    _write(tmp_path / "experiment_2014_other.json", {"status": "success"})
    # Also write one real in-range file so the test is non-trivial
    _write(tmp_path / _DELIVERABLES[2003], {"status": "success", "honest_verdict": "complete: ok"})

    out = str(tmp_path / "experiment_2014_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    all_counted = (
        result["completed_experiments"]
        + result["blocked_experiments"]
        + result["failed_experiments"]
    )
    assert 2000 not in all_counted
    assert 2014 not in all_counted
    assert 2003 in result["completed_experiments"]
