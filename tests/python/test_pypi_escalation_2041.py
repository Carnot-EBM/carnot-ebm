import pytest
from carnot.pypi_escalation import generate_escalation_report

def test_generate_escalation_report_contains_all_fields():
    report = generate_escalation_report(
        experiment_id=2041,
        run_date="2026-05-16T15:00:00Z",
        duration_s=25,
        random_seed=173241,
        preconditions_checked=["blocked_gh_cli_unavailable"],
        workflow_run_id=12345,
        workflow_run_status="completed",
        workflow_run_conclusion="failure",
        milestones_pending_count=10,
        re_trigger_attempted=False,
        re_trigger_outcome=None,
        pypi_url_reachable=True,
        external_install_verified=True,
        actionable_next_step="none",
        escalation_summary=None,
        honest_verdict="success: PyPI 0.1.0b1 reachable + install works"
    )
    
    assert report["schema"] == "carnot.pypi_escalation.v1"
    assert report["experiment"] == 2041
    assert report["run_date"] == "2026-05-16T15:00:00Z"
    assert report["duration_s"] == 25
    assert report["random_seed"] == 173241
    assert "reproducibility_checksum" in report
    assert report["preconditions_checked"] == ["blocked_gh_cli_unavailable"]
    
    specs = report["model_specs"]
    assert specs["workflow_file"] == ".github/workflows/publish-pypi.yml"
    assert specs["tag_checked"] == "v0.1.0b1"
    assert specs["milestones_pending"] == 10
    
    assert report["n_samples"] == 1
    assert report["n_samples_justification"] == "Status escalation; n=1 workflow."
    assert report["workflow_run_id"] == 12345
    assert report["workflow_run_status"] == "completed"
    assert report["workflow_run_conclusion"] == "failure"
    assert report["milestones_pending_count"] == 10
    assert report["re_trigger_attempted"] is False
    assert report["re_trigger_outcome"] is None
    assert report["pypi_url_reachable"] is True
    assert report["external_install_verified"] is True
    assert report["actionable_next_step"] == "none"
    assert report["escalation_summary"] is None
    assert report["acceptance_gate_passed"] is True
    assert report["acceptance_gate_criteria"] == "Status reported; escalation summary written if overdue; install verified if succeeded."
    assert report["methodology_note"] == "NO new tag push. workflow_dispatch re-trigger only if cancelled/expired."
    assert report["optimization_direction"] == "neither"
    assert report["honest_verdict"] == "success: PyPI 0.1.0b1 reachable + install works"
