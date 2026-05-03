"""Tests for scripts/experiment_1182_paper_v5_medium_low_issues_11_18.py.

All checks run against the real docs/arxiv-paper sources; no GPU required.

Spec coverage:
  REQ-PUBLISH-008 — Medium/low paper-v5 integrity fixes ISSUE-11 through ISSUE-18
  REQ-PUBLISH-009 — Paper Numerical Claim Audit
  SCENARIO-PUBLISH-008 — All medium/low fixes and claim audit verified
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1182_paper_v5_medium_low_issues_11_18 as exp1182


def _tex() -> str:
    return exp1182._load_paper()


def _bib() -> str:
    return exp1182._load_bib()


def _patched_deliverable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    deliverable = tmp_path / "experiment_1182_paper_v5_medium_low_issues_11_18.json"
    monkeypatch.setattr(exp1182, "_DELIVERABLE", deliverable)
    return deliverable


class TestIssue11ThroughIssue15:
    """REQ-PUBLISH-008 — Medium-severity paper text fixes are present."""

    def test_issue_11_thinkprm_names_exp1111_predecessor(self) -> None:
        """REQ-PUBLISH-008: ThinkPRM AUROC=0.9885 names exp1111 v1 and v2."""
        assert exp1182.check_issue_11_thinkprm(_tex())

    def test_issue_12_holdout_n_and_production_contradiction_stated(self) -> None:
        """REQ-PUBLISH-008: FoVer holdout claims disclose n=50 and exp1121 conflict."""
        assert exp1182.check_issue_12_holdout(_tex())

    def test_issue_13_nrgpt_disclosure_or_no_citation(self) -> None:
        """REQ-PUBLISH-008: NRGPT is either uncited or explicitly non-monotone."""
        resolved, status = exp1182.check_issue_13_nrgpt(_tex())
        assert resolved is True
        assert status in {"not_cited", "disclosed"}

    def test_issue_14_soskan_aurocs_have_corpus_and_n(self) -> None:
        """REQ-PUBLISH-008: SOS-KAN AUROC values are reconciled by corpus and sample size."""
        assert exp1182.check_issue_14_soskan_auroc(_tex())

    def test_issue_15_fig2_caption_has_binormal_caveat(self) -> None:
        """REQ-PUBLISH-008: Figure 2 caption states the binormal-fit limitation."""
        assert exp1182.check_issue_15_fig2_caveat(_tex())


class TestIssue16ThroughIssue18:
    """REQ-PUBLISH-008 — Low-severity bibliography, table, and hardware fixes are present."""

    def test_issue_16_bib_stub_audit(self) -> None:
        """REQ-PUBLISH-008: real papers have authors; non-paper Themesis cite is removed."""
        removed, ok = exp1182.check_issue_16_bibliography(_tex(), _bib())
        assert ok is True
        assert removed == 1

    def test_issue_17_k15_caption_tightened(self) -> None:
        """REQ-PUBLISH-008: k=15 row is explicitly theoretical, not empirical."""
        assert exp1182.check_issue_17_k15_caption(_tex())

    def test_issue_18_hardware_scope_added(self) -> None:
        """REQ-PUBLISH-008: hardware portability scope is limited to KV260 evidence."""
        assert exp1182.check_issue_18_hardware_scope(_tex())


class TestRunReturnsSchema:
    """SCENARIO-PUBLISH-008 — run() emits the required exp1182 artifact schema."""

    _REQUIRED_BOOL_FIELDS = [
        "issue_11_thinkprm_citation_fixed",
        "issue_12_holdout_n_stated",
        "issue_13_nrgpt_disclosure_added",
        "issue_14_soskan_auroc_reconciled",
        "issue_15_fig2_caveat_added",
        "issue_17_k15_caption_tightened",
        "issue_18_hardware_scope_added",
        "paper_claim_audit_script_active",
    ]

    def test_all_required_fields_present(self) -> None:
        payload = exp1182.run()
        for field in self._REQUIRED_BOOL_FIELDS:
            assert field in payload, f"Required field '{field}' missing from artifact"
        for field in [
            "issue_16_bib_stubs_removed",
            "paper_claim_audit_n_claims_total",
            "paper_claim_audit_n_verified",
            "paper_claim_audit_n_mismatches",
            "medium_low_issues_fixed",
            "honest_verdict",
        ]:
            assert field in payload, f"Required field '{field}' missing from artifact"

    def test_all_8_medium_low_issues_resolved(self) -> None:
        """SCENARIO-PUBLISH-008: all eight issue checks pass."""
        payload = exp1182.run()
        assert payload["medium_low_issues_fixed"] == 8
        assert payload["honest_verdict"] == "all_8_medium_low_resolved"

    def test_paper_claim_audit_is_green(self) -> None:
        """REQ-PUBLISH-009: exp1182 records a green claim-audit result."""
        payload = exp1182.run()
        assert payload["paper_claim_audit_script_active"] is True
        assert payload["paper_claim_audit_n_claims_total"] > 0
        assert payload["paper_claim_audit_n_verified"] > 0
        assert payload["paper_claim_audit_n_mismatches"] == 0


class TestDeliverableJson:
    """Deliverable JSON must be well formed and contain all required fields."""

    def test_main_writes_success_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SCENARIO-PUBLISH-008: main() writes the exp1182 deliverable."""
        deliverable = _patched_deliverable(monkeypatch, tmp_path)
        exp1182.main()

        loaded = json.loads(deliverable.read_text(encoding="utf-8"))
        assert loaded["experiment"] == 1182
        assert loaded["honest_verdict"] == "all_8_medium_low_resolved"
        assert json.loads(capsys.readouterr().out)["medium_low_issues_fixed"] == 8

    def test_main_exits_nonzero_on_partial_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """REQ-PUBLISH-008: main() refuses to bless partial remediation."""
        deliverable = _patched_deliverable(monkeypatch, tmp_path)
        monkeypatch.setattr(
            exp1182,
            "run",
            lambda: {"honest_verdict": "blocked", "medium_low_issues_fixed": 0},
        )

        with pytest.raises(SystemExit) as excinfo:
            exp1182.main()

        assert excinfo.value.code == 1
        assert json.loads(deliverable.read_text(encoding="utf-8"))["honest_verdict"] == "blocked"
        assert "not all fixes verified" in capsys.readouterr().err
