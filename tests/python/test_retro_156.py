"""
Tests for the milestone .156 retrospective generator.

REQ-REPORT-156: The retro generator MUST scan experiment_1996-2006 artifacts,
classify them as completed/blocked/failed, and emit a carnot.milestone_retro.v1
artifact with correct counts, verdicts, and at least one recommendation.

SCENARIO-REPORT-156-A: Typical .156 run with four blocked doomed-rerun artifacts
(exp1997, exp2001, exp2002, exp2003) and completed NSVIF/EBT/KAN artifacts
produces correct counts and non-empty recommendations.

SCENARIO-REPORT-156-B: Unreadable artifact files are counted as failed.

SCENARIO-REPORT-156-C: honest_verdict field takes priority over status for
classification.

SCENARIO-REPORT-156-D: NSVIF/Z3, COLD Decoding, and Tier 2 Memory analysis
fields are all present and non-empty in the output.
"""

import json
import os
from pathlib import Path

from carnot.retro_156 import generate_retro, _classify_artifact


# --- unit tests for _classify_artifact ---

def test_classify_blocked_via_honest_verdict():
    """REQ-REPORT-156: blocked_gate_check_failed honest_verdict classifies as blocked."""
    assert _classify_artifact({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"


def test_classify_blocked_via_status():
    """REQ-REPORT-156: status='blocked' classifies as blocked."""
    assert _classify_artifact({"status": "blocked"}) == "blocked"


def test_classify_failed_via_verdict():
    """REQ-REPORT-156: verdict containing 'fail' classifies as failed."""
    assert _classify_artifact({"honest_verdict": "failed_due_to_error"}) == "failed"


def test_classify_completed_on_complete_verdict():
    """REQ-REPORT-156: 'complete:' prefixed verdict without fail/blocked -> completed."""
    assert _classify_artifact({"honest_verdict": "complete: nsvif_z3_zero_fp"}) == "completed"


def test_classify_completed_on_success_status():
    """REQ-REPORT-156: status='success' classifies as completed."""
    assert _classify_artifact({"status": "success"}) == "completed"


def test_classify_completed_on_complete_status():
    """REQ-REPORT-156: status='complete' classifies as completed."""
    assert _classify_artifact({"status": "complete"}) == "completed"


def test_classify_empty_artifact():
    """REQ-REPORT-156: empty artifact defaults to completed (no negative signal)."""
    assert _classify_artifact({}) == "completed"


def test_classify_blocked_supersedes_fail_in_combined():
    """REQ-REPORT-156: 'blocked' takes priority over 'fail' in combined text."""
    assert _classify_artifact({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"


# --- helpers for integration tests ---

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


# --- integration tests for generate_retro ---

def test_generate_retro_schema_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-A: artifact has all required schema fields."""
    _write(tmp_path / "experiment_1996_nsvif.json", {
        "status": "complete",
        "honest_verdict": "complete: NSVIF/Z3 extractor zero false positives",
        "success": True,
    })
    _write(tmp_path / "experiment_2002_cold.json", {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
    })

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.156"
    assert result["experiment_id"] == 2007
    assert result["status"] == "complete"
    assert result["retro_complete"] is True
    assert result["honest_verdict"].startswith("complete:")


def test_generate_retro_counts_completed_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-A: completed/blocked/failed counts are correct."""
    _write(tmp_path / "experiment_1996_nsvif.json", {"status": "complete"})
    _write(tmp_path / "experiment_1998_gsm8k.json", {"status": "success"})
    _write(tmp_path / "experiment_1997_llm.json", {"honest_verdict": "blocked_gate_check_failed"})
    _write(tmp_path / "experiment_2002_cold.json", {"status": "blocked"})
    _write(tmp_path / "experiment_2003_memory.json", {"status": "blocked"})

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["completed_task_count"] == 2
    assert result["blocked_task_count"] == 3


def test_generate_retro_honest_verdict_priority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-C: honest_verdict takes priority over status in verdicts dict."""
    _write(tmp_path / "experiment_1996_nsvif.json", {
        "status": "complete",
        "honest_verdict": "complete: NSVIF/Z3 SMT extractor zero false positives",
    })

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["experiment_honest_verdicts"]["exp1996"] == (
        "complete: NSVIF/Z3 SMT extractor zero false positives"
    )


def test_generate_retro_recommendations_nonempty(tmp_path: Path) -> None:
    """REQ-REPORT-156: recommendations list MUST be non-empty."""
    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


def test_generate_retro_criteria_structure(tmp_path: Path) -> None:
    """REQ-REPORT-156: criteria_results, criteria_met, criteria_total are present."""
    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "criteria_results" in result
    assert "criteria_met" in result
    assert "criteria_total" in result
    assert result["criteria_met"] <= result["criteria_total"]


def test_generate_retro_writes_valid_json(tmp_path: Path) -> None:
    """REQ-REPORT-156: output file is written and parseable as valid JSON."""
    out = str(tmp_path / "experiment_2007_retro.json")
    generate_retro(out, results_dir=str(tmp_path))

    assert os.path.exists(out)
    with open(out) as fh:
        parsed = json.load(fh)
    assert parsed["schema"] == "carnot.milestone_retro.v1"


def test_generate_retro_ignores_out_of_range(tmp_path: Path) -> None:
    """REQ-REPORT-156: experiments outside [1996, 2006] are not counted."""
    # exp1995 is the .155 retro — must be ignored
    _write(tmp_path / "experiment_1995_retro.json", {"status": "complete"})
    # exp2007 is the .156 retro itself — must be ignored
    _write(tmp_path / "experiment_2007_retro.json", {"status": "complete"})

    out = str(tmp_path / "experiment_2007_milestone_156_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    all_counted = (
        result["completed_experiments"]
        + result["blocked_experiments"]
        + result["failed_experiments"]
    )
    assert 1995 not in all_counted
    assert 2007 not in all_counted


def test_generate_retro_unreadable_artifact_counted_as_failed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-B: unreadable JSON artifact is counted as failed."""
    broken = tmp_path / "experiment_1999_code.json"
    broken.write_text("{ not valid json >>>")

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 1999 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp1999"] == "UNREADABLE"


def test_generate_retro_nsvif_cold_memory_analysis_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-D: nsvif_cold_memory_analysis field is present and non-empty."""
    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "nsvif_cold_memory_analysis" in result
    analysis = result["nsvif_cold_memory_analysis"]
    assert "nsvif_z3_extractor" in analysis
    assert "cold_decoding" in analysis
    assert "tier2_constraint_memory" in analysis
    assert len(analysis["nsvif_z3_extractor"]) > 0
    assert len(analysis["cold_decoding"]) > 0
    assert len(analysis["tier2_constraint_memory"]) > 0


def test_generate_retro_gate_contract_gap_note_nonempty(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-D: gate_contract_gap_note is non-empty and mentions verdicts."""
    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "gate_contract_gap_note" in result
    note = result["gate_contract_gap_note"]
    assert len(note) > 0
    assert "terminal" in note.lower() or "prefix" in note.lower()


def test_generate_retro_trajectory_and_hardware_lessons(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-D: trajectory and hardware accounting lessons are non-empty."""
    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "trajectory_optimization_lessons" in result
    assert "hardware_accounting_lessons" in result
    assert len(result["trajectory_optimization_lessons"]) > 0
    assert len(result["hardware_accounting_lessons"]) > 0


def test_generate_retro_notable_successes_nonempty(tmp_path: Path) -> None:
    """REQ-REPORT-156: notable_successes list is non-empty."""
    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["notable_successes"], list)
    assert len(result["notable_successes"]) > 0


def test_generate_retro_failed_verdict_classifies_as_failed(tmp_path: Path) -> None:
    """REQ-REPORT-156: artifact with 'fail' in verdict is counted as failed."""
    _write(tmp_path / "experiment_2000_sade.json", {
        "honest_verdict": "failed: constraint layer not converging",
        "status": "failed",
    })

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 2000 in result["failed_experiments"]


def test_known_missing_experiment_counted_as_failed(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-156: an ID in _KNOWN_MISSING_EXP_IDS with no artifact is counted as failed."""
    import carnot.retro_156 as mod
    monkeypatch.setattr(mod, "_KNOWN_MISSING_EXP_IDS", [2005])

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 2005 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp2005"] == "MISSING"


def test_generate_retro_four_doomed_reruns_blocked_correctly(tmp_path: Path) -> None:
    """SCENARIO-REPORT-156-A: four doomed-rerun blocks (1997,2001,2002,2003) are blocked."""
    for exp_id in [1997, 2001, 2002, 2003]:
        _write(tmp_path / f"experiment_{exp_id}_x.json", {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        })

    out = str(tmp_path / "experiment_2007_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    for exp_id in [1997, 2001, 2002, 2003]:
        assert exp_id in result["blocked_experiments"]
