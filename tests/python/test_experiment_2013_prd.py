from scripts.experiment_2013_prd import generate_reconciliation_summary

def test_generate_reconciliation_summary():
    """Verify that the JSON summary generator works as expected (REQ-EXP2013)."""
    summary = generate_reconciliation_summary()
    assert summary["experiment"] == 2013
    assert summary["schema"] == "carnot.experiment.v1"
    assert summary["status"] == "success"
    assert summary["docs_reconciled"] is True
