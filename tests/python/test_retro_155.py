"""
Tests for the milestone .155 retrospective generator.

REQ-REPORT-155: The retro generator MUST scan experiment_198x–199x artifacts,
classify them as completed/blocked/failed, and emit a carnot.milestone_retro.v1
artifact with correct counts, verdicts, and at least one recommendation.

SCENARIO-REPORT-155-A: Typical .155 run with blocked gate-check artifacts and
completed energy/continual-learning artifacts produces correct counts and
non-empty recommendations.

SCENARIO-REPORT-155-B: Missing artifacts (exp1987, exp1993) are counted as failed.

SCENARIO-REPORT-155-C: honest_verdict field takes priority over status for
classification.

SCENARIO-REPORT-155-D: Gate-contract gap lessons (exp1985, exp1992) and
hardware accounting lessons are included in the output.
"""

import json
import os
from pathlib import Path

from carnot.retro_155 import generate_retro, _classify_artifact


# --- unit tests for _classify_artifact ---

def test_classify_blocked_via_honest_verdict():
    """REQ-REPORT-155: blocked_gate_check_failed honest_verdict classifies as blocked."""
    assert _classify_artifact({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"


def test_classify_blocked_via_status():
    """REQ-REPORT-155: status='blocked' classifies as blocked."""
    assert _classify_artifact({"status": "blocked"}) == "blocked"


def test_classify_failed_via_verdict():
    """REQ-REPORT-155: verdict containing 'fail' classifies as failed."""
    assert _classify_artifact({"honest_verdict": "failed_due_to_error"}) == "failed"


def test_classify_completed_on_success_verdict():
    """REQ-REPORT-155: success verdict without fail/blocked keyword -> completed."""
    assert _classify_artifact({"honest_verdict": "complete: fr11_shipped"}) == "completed"


def test_classify_completed_on_status_success():
    """REQ-REPORT-155: status='success' classifies as completed."""
    assert _classify_artifact({"status": "success"}) == "completed"


def test_classify_empty_artifact():
    """REQ-REPORT-155: empty artifact defaults to completed (no negative signal)."""
    assert _classify_artifact({}) == "completed"


# --- integration tests for generate_retro ---

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_generate_retro_schema_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-155-A: artifact has all required schema fields."""
    _write(tmp_path / "experiment_1986_fr11.json", {
        "status": "success",
        "honest_verdict": "complete: fr11_shipped",
    })
    _write(tmp_path / "experiment_1991_curie.json", {
        "status": "success",
        "honest_verdict": "success: kl_0.0130_delta_0.1439",
        "acceptance_gate_passed": True,
    })
    _write(tmp_path / "experiment_1982_latent.json", {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
    })

    out = str(tmp_path / "experiment_1995_milestone_155_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.155"
    assert result["experiment_id"] == 1995
    assert result["status"] == "complete"
    assert result["retro_complete"] is True
    assert result["honest_verdict"].startswith("complete:")


def test_generate_retro_counts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-155-A: completed/blocked/failed counts are correct."""
    _write(tmp_path / "experiment_1986_fr11.json", {"status": "success"})
    _write(tmp_path / "experiment_1991_curie.json", {"status": "success"})
    _write(tmp_path / "experiment_1985_consformer.json", {"status": "complete"})
    _write(tmp_path / "experiment_1982_latent.json", {"honest_verdict": "blocked_gate_check_failed"})
    _write(tmp_path / "experiment_1983_egdec.json", {"status": "blocked"})
    _write(tmp_path / "experiment_1984_kanele.json", {"status": "blocked"})

    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["completed_task_count"] == 3
    assert result["blocked_task_count"] == 3
    # exp1987 and exp1993 always appear as missing/failed
    assert 1987 in result["failed_experiments"]
    assert 1993 in result["failed_experiments"]


def test_generate_retro_missing_experiments_counted_as_failed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-155-B: exp1987 and exp1993 are always counted as missing/failed."""
    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 1987 in result["failed_experiments"]
    assert 1993 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp1987"] == "MISSING"
    assert result["experiment_honest_verdicts"]["exp1993"] == "MISSING"


def test_generate_retro_honest_verdict_priority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-155-C: honest_verdict takes priority over status in verdicts dict."""
    _write(tmp_path / "experiment_1991_curie.json", {
        "status": "complete",
        "honest_verdict": "success: kl_0.0130_delta_0.1439",
    })

    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["experiment_honest_verdicts"]["exp1991"] == "success: kl_0.0130_delta_0.1439"


def test_generate_retro_recommendations_nonempty(tmp_path: Path) -> None:
    """REQ-REPORT-155: recommendations list MUST be non-empty."""
    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


def test_generate_retro_criteria_structure(tmp_path: Path) -> None:
    """REQ-REPORT-155: criteria_results, criteria_met, criteria_total are present."""
    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "criteria_results" in result
    assert "criteria_met" in result
    assert "criteria_total" in result
    assert result["criteria_met"] <= result["criteria_total"]


def test_generate_retro_writes_file(tmp_path: Path) -> None:
    """REQ-REPORT-155: output file is written and parseable as valid JSON."""
    out = str(tmp_path / "experiment_1995_milestone_155_retro.json")
    generate_retro(out, results_dir=str(tmp_path))

    assert os.path.exists(out)
    with open(out) as fh:
        parsed = json.load(fh)
    assert parsed["schema"] == "carnot.milestone_retro.v1"


def test_generate_retro_ignores_out_of_range(tmp_path: Path) -> None:
    """REQ-REPORT-155: experiments outside [1982, 1994] are not counted."""
    # exp1981 is the .154 retro, must be ignored
    _write(tmp_path / "experiment_1981_milestone_154_retro.json", {"status": "complete"})
    # exp1996 is beyond the range
    _write(tmp_path / "experiment_1996_future.json", {"status": "complete"})

    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    all_counted = (
        result["completed_experiments"]
        + result["blocked_experiments"]
        + result["failed_experiments"]
    )
    assert 1981 not in all_counted
    assert 1996 not in all_counted


def test_generate_retro_gate_contract_gap_note(tmp_path: Path) -> None:
    """SCENARIO-REPORT-155-D: gate_contract_gap_note is present and non-empty."""
    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "gate_contract_gap_note" in result
    assert len(result["gate_contract_gap_note"]) > 0
    assert "third" in result["gate_contract_gap_note"].lower()


def test_generate_retro_lessons_present(tmp_path: Path) -> None:
    """SCENARIO-REPORT-155-D: trajectory and hardware accounting lessons are present."""
    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "trajectory_optimization_lessons" in result
    assert "hardware_accounting_lessons" in result
    assert len(result["trajectory_optimization_lessons"]) > 0
    assert len(result["hardware_accounting_lessons"]) > 0


def test_generate_retro_notable_successes_present(tmp_path: Path) -> None:
    """REQ-REPORT-155: notable_successes list is non-empty."""
    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["notable_successes"], list)
    assert len(result["notable_successes"]) > 0


def test_generate_retro_unreadable_artifact_counted_as_failed(tmp_path: Path) -> None:
    """REQ-REPORT-155: unreadable JSON artifact (OSError path) is counted as failed."""
    broken = tmp_path / "experiment_1986_fr11.json"
    broken.write_text("{ not valid json >>>")

    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 1986 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp1986"] == "UNREADABLE"


def test_generate_retro_failed_classification_counted(tmp_path: Path) -> None:
    """REQ-REPORT-155: artifact with 'fail' in verdict is counted as failed."""
    _write(tmp_path / "experiment_1994_pre_retro.json", {
        "honest_verdict": "Audit failed: missing files or violated gates found.",
        "status": "failed",
    })

    out = str(tmp_path / "experiment_1995_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 1994 in result["failed_experiments"]
