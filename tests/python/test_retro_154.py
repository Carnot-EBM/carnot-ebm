"""
Tests for the milestone .154 retrospective generator.

REQ-REPORT-154: The retro generator MUST scan experiment_196x–198x artifacts,
classify them as completed/blocked/failed, and emit a carnot.milestone_retro.v1
artifact with correct counts, verdicts, and at least one recommendation.

SCENARIO-REPORT-154-A: Typical .154 run with blocked gate-check artifacts and
completed energy-solver artifacts produces correct counts and non-empty
recommendations.

SCENARIO-REPORT-154-B: Missing artifacts (exp1971, exp1979) are counted as failed.

SCENARIO-REPORT-154-C: Honest_verdict field takes priority over status for
classification.
"""

import json
import os
from pathlib import Path

from carnot.retro_154 import generate_retro, _classify_artifact


# --- unit tests for _classify_artifact ---

def test_classify_blocked_via_honest_verdict():
    """REQ-REPORT-154: blocked honest_verdict classifies as blocked."""
    assert _classify_artifact({"honest_verdict": "blocked_gate_check_failed"}) == "blocked"


def test_classify_blocked_via_status():
    """REQ-REPORT-154: status='blocked' classifies as blocked."""
    assert _classify_artifact({"status": "blocked"}) == "blocked"


def test_classify_failed_via_honest_verdict():
    """REQ-REPORT-154: verdict containing 'fail' classifies as failed."""
    assert _classify_artifact({"honest_verdict": "failed_due_to_error"}) == "failed"


def test_classify_completed_on_success_verdict():
    """REQ-REPORT-154: success verdict with no blocked/fail keyword -> completed."""
    assert _classify_artifact({"honest_verdict": "complete: runcsp_shipped"}) == "completed"


def test_classify_completed_on_status_complete():
    """REQ-REPORT-154: status='complete' without blocked/fail -> completed."""
    assert _classify_artifact({"status": "complete"}) == "completed"


def test_classify_empty_artifact():
    """REQ-REPORT-154: empty artifact defaults to completed (no negative signal)."""
    assert _classify_artifact({}) == "completed"


# --- integration tests for generate_retro ---

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_generate_retro_schema_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-154-A: artifact has required schema fields."""
    _write(tmp_path / "experiment_1972_run_csp.json", {"status": "complete", "honest_verdict": "complete_generalized"})
    _write(tmp_path / "experiment_1973_deepsade.json", {"status": "complete", "honest_verdict": "zero_false_accept_strictly_guaranteed"})
    _write(tmp_path / "experiment_1969_cold.json", {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"})

    out = str(tmp_path / "experiment_1981_milestone_154_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["schema"] == "carnot.milestone_retro.v1"
    assert result["milestone"] == "2026.05.154"
    assert result["experiment_id"] == 1981
    assert result["status"] == "complete"
    assert result["retro_complete"] is True
    assert result["honest_verdict"].startswith("complete:")


def test_generate_retro_counts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-154-A: completed/blocked/failed counts are correct."""
    _write(tmp_path / "experiment_1970_domino.json", {"status": "complete"})
    _write(tmp_path / "experiment_1972_runcsp.json", {"status": "complete"})
    _write(tmp_path / "experiment_1973_deepsade.json", {"status": "complete"})
    _write(tmp_path / "experiment_1969_cold.json", {"honest_verdict": "blocked_gate_check_failed"})
    _write(tmp_path / "experiment_1974_hw.json", {"status": "blocked"})

    out = str(tmp_path / "experiment_1981_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    # 3 completed in range, 2 blocked
    assert result["completed_task_count"] == 3
    assert result["blocked_task_count"] == 2
    # exp1971 and exp1979 always appear as missing/failed
    assert 1971 in result["failed_experiments"]
    assert 1979 in result["failed_experiments"]


def test_generate_retro_missing_experiments_counted_as_failed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-154-B: exp1971 and exp1979 are always counted as failed/missing."""
    # No files at all for 1971 or 1979
    out = str(tmp_path / "experiment_1981_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert 1971 in result["failed_experiments"]
    assert 1979 in result["failed_experiments"]
    assert result["experiment_honest_verdicts"]["exp1971"] == "MISSING"
    assert result["experiment_honest_verdicts"]["exp1979"] == "MISSING"


def test_generate_retro_honest_verdict_priority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-154-C: honest_verdict takes priority over status for verdicts dict."""
    _write(tmp_path / "experiment_1975_eorm.json", {
        "status": "complete",
        "honest_verdict": "eorm_verification_layer_reranks_accurately"
    })

    out = str(tmp_path / "experiment_1981_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert result["experiment_honest_verdicts"]["exp1975"] == "eorm_verification_layer_reranks_accurately"


def test_generate_retro_recommendations_nonempty(tmp_path: Path) -> None:
    """REQ-REPORT-154: recommendations list MUST be non-empty."""
    out = str(tmp_path / "experiment_1981_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


def test_generate_retro_criteria_structure(tmp_path: Path) -> None:
    """REQ-REPORT-154: criteria_results, criteria_met, criteria_total are present."""
    out = str(tmp_path / "experiment_1981_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    assert "criteria_results" in result
    assert "criteria_met" in result
    assert "criteria_total" in result
    assert result["criteria_met"] <= result["criteria_total"]


def test_generate_retro_writes_file(tmp_path: Path) -> None:
    """REQ-REPORT-154: output file is written and parseable."""
    out = str(tmp_path / "experiment_1981_milestone_154_retro.json")
    generate_retro(out, results_dir=str(tmp_path))

    assert os.path.exists(out)
    with open(out) as fh:
        parsed = json.load(fh)
    assert parsed["schema"] == "carnot.milestone_retro.v1"


def test_generate_retro_ignores_out_of_range(tmp_path: Path) -> None:
    """REQ-REPORT-154: experiments outside [1969,1980] are not counted."""
    # exp1968 is the .153 retro, should be ignored
    _write(tmp_path / "experiment_1968_milestone_153_retro.json", {"status": "complete"})
    # exp1982 is beyond the range
    _write(tmp_path / "experiment_1982_future.json", {"status": "complete"})

    out = str(tmp_path / "experiment_1981_retro.json")
    result = generate_retro(out, results_dir=str(tmp_path))

    # Only exp1971 and exp1979 appear (as MISSING)
    all_counted = (
        result["completed_experiments"]
        + result["blocked_experiments"]
        + result["failed_experiments"]
    )
    assert 1968 not in all_counted
    assert 1982 not in all_counted
