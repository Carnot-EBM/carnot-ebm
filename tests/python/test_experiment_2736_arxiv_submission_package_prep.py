"""Exp 2736: paper-v6 arXiv submission PACKAGE PREP (never submission itself).

Spec: REQ-REPORT-2736 / SCENARIO-REPORT-2736 -- same local-identifier convention the
sibling archive/report tests use (cf. REQ-REPORT-3583, REQ-REPORT-3611).

The package-prep workflow SHALL compile the paper, count pages and theory citations,
and emit an OPERATOR checklist whose upload step is explicitly reserved to the
operator, marking `submission_package_ready` only when every item passes. It SHALL
NEVER submit to arXiv (CLAUDE.md "Operator-Only External Publication"), which is why
the assertion below pins the literal OPERATOR-ONLY checklist line.
"""

import os
import json
from pathlib import Path

import pytest

from experiment_2736_arxiv_submission_package_prep import main

# Repo root, derived from this file's location rather than a hardcoded absolute
# path, so the test works from either of this environment's two working-directory
# aliases (one is a symlink to the other).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_experiment_2736(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SANDBOX THE SCRIPT'S WRITES -- see the long rationale in
    # tests/python/test_experiment_3611.py, which has the identical defect.
    # In short: `main()` resolves its inputs (docs/arxiv-paper/main.tex,
    # results/experiment_2729_paper_v6_theory_v3.json) and its output
    # (results/experiment_2736_...json) relative to the CURRENT WORKING
    # DIRECTORY, so running the test suite from the repo root silently
    # OVERWROTE a committed historical artifact with a fresh duration_s.
    # Running the suite is not running the experiment. chdir into tmp_path,
    # symlink the read-only inputs, and let the script write there instead.
    # Assertions are unchanged.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "docs").mkdir()
    os.symlink(_REPO_ROOT / "docs/arxiv-paper", tmp_path / "docs/arxiv-paper")
    os.symlink(
        _REPO_ROOT / "results/experiment_2729_paper_v6_theory_v3.json",
        tmp_path / "results/experiment_2729_paper_v6_theory_v3.json",
    )

    main()
    artifact_path = tmp_path / "results/experiment_2736_arxiv_submission_package_prep.json"
    assert artifact_path.exists()

    with open(artifact_path) as f:
        data = json.load(f)

    assert data["honest_verdict"].startswith("complete:") or data["honest_verdict"].startswith(
        "blocked_"
    )

    if data["honest_verdict"].startswith("complete:"):
        assert data["submission_package_ready"] is True
        assert data["pdf_compiles"] is True
        assert data["n_pages"] > 0
        assert data["n_theory_citations_present"] == 3
        assert len(data["operator_arxiv_checklist"]) > 0
        assert (
            "Step 4: Upload to arxiv.org (OPERATOR-ONLY per CLAUDE.md)"
            in data["operator_arxiv_checklist"]
        )
        assert data["duration_s"] >= 10.0
        assert len(data["preconditions_checked"]) >= 3
