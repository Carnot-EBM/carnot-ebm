"""Tests for the Exp 1269 paper-v6 critical-fixes v2 artifact.

Spec anchors:
  REQ-PUBLISH-013 -- terminal artifact for five paper critical claim classes.
  SCENARIO-PUBLISH-013 -- clean paper audit cites exp1256/1264/1265/1266.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import paper_v6_critical_fixes_v2 as exp1269


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
RESULTS = REPO_ROOT / "results"


def test_audit_flags_old_unsupported_claim_strings() -> None:
    """REQ-PUBLISH-013: the auditor recognizes every old claim class."""

    old_text = """
    The old paper claimed $13{,}061\\times$ FPGA speedup and FPGA KL=3.07.
    CPU_GIBBS_PER_SWEEP_NS = 1000.0 produced 15.6x from a hand constant.
    HardNet++ ran 76,130x faster than prompt repair on HumanEval latency.
    SOSKANEnergyV3 AUROC=0.3333 was presented without corpus context.
    """

    assert exp1269.audit_old_claims(old_text) == [
        "estimated_cpu_fpga_speedups",
        "kl_measurement_provenance",
        "hand_typed_cpu_constants",
        "apples_to_oranges_humaneval_latency",
        "sos_kan_auroc_ambiguity",
    ]


def test_real_paper_has_clean_old_claim_audit() -> None:
    """SCENARIO-PUBLISH-013: main.tex has no old unsupported claim strings."""

    tex = PAPER.read_text(encoding="utf-8")
    assert exp1269.audit_old_claims(tex) == []


def test_real_paper_cites_new_measured_artifacts() -> None:
    """REQ-PUBLISH-013: exp1256, exp1264, exp1265, and exp1266 are cited."""

    tex = PAPER.read_text(encoding="utf-8")
    assert exp1269.find_measured_artifacts_cited(tex) == [
        "exp1256",
        "exp1264",
        "exp1265",
        "exp1266",
    ]


def test_build_artifact_reports_complete_schema() -> None:
    """SCENARIO-PUBLISH-013: the artifact schema reports all five fixes."""

    artifact = exp1269.build_artifact(
        PAPER.read_text(encoding="utf-8"),
        results_dir=RESULTS,
        run_date="20260504",
    )

    assert artifact["experiment"] == "1269_paper_v6_critical_fixes_v2"
    assert artifact["schema"] == "paper_integrity_v2"
    assert artifact["run_date"] == "20260504"
    assert artifact["critical_issues_fixed"] == 5
    assert artifact["issues_fixed_list"] == exp1269.ISSUES_FIXED_LIST
    assert artifact["measured_artifacts_cited"] == ["exp1256", "exp1264", "exp1265", "exp1266"]
    assert artifact["old_claims_remaining"] == []
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"] == "paper_v6_critical_fixes_v2_complete"


def test_run_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-PUBLISH-013: run() writes the final JSON deliverable."""

    out_path = tmp_path / "experiment_1269_paper_v6_critical_fixes_v2.json"
    artifact = exp1269.run(paper_path=PAPER, results_dir=RESULTS, out_path=out_path)

    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["critical_issues_fixed"] == 5
    assert loaded["honest_verdict"] == "paper_v6_critical_fixes_v2_complete"
