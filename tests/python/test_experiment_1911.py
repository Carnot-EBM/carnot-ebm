import json
import pytest
from run_experiment_1911 import run


def test_experiment_1911_schema(tmp_path):
    # Spec: REQ-PHASE4-CANONICAL-DECISION-1911, SCENARIO-DECISION-ARTIFACT-GENERATION-1911
    #
    # Per the Test-Run Record Integrity Discipline: this test must not write the real repo's
    # openspec/ops/results files (see run_experiment_1911.py's paper_path incident note --
    # calling run() with no args against the real repo produced 425 duplicate paper sections
    # before the paper_path write was made idempotent, and would still silently rewrite
    # run_date/duration_s in the real results artifact on every test invocation). Everything
    # writable is redirected to tmp_path; only the read-only precondition inputs under
    # results/ (exp1811/1745/1909, which this test does not itself produce) come from the
    # real repo.
    paper_path = tmp_path / "section-6-limitations.md"
    paper_path.write_text("")
    known_issues_path = tmp_path / "known-issues.md"
    known_issues_path.write_text("")
    result_path = tmp_path / "experiment_1911_phase4_canonical_decision.json"

    run(
        paper_path=str(paper_path),
        known_issues_path=str(known_issues_path),
        result_path=str(result_path),
    )

    assert result_path.exists()
    with open(result_path) as f:
        data = json.load(f)

    assert data["schema"] == "carnot.phase4_canonical_decision.v2"
    assert data["experiment"] == 1911
    assert data["honest_verdict"].startswith("success:")
    assert data["acceptance_gate_passed"] is True
    assert "preconditions_checked" in data
    assert "Fast-Slow Variant" in data["canonical_metric_named"]
