"""Tests for scripts/experiment_1181_paper_v5_high_issues_6_10.py.

All checks run against the real docs/arxiv-paper/main.tex; no GPU required.

Spec coverage:
  REQ-PUBLISH-007  — High-severity integrity fixes ISSUE-6 through ISSUE-10
  SCENARIO-PUBLISH-007 — All five high-severity fixes verified in main.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1181_paper_v5_high_issues_6_10 as _mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tex() -> str:
    return _mod._load_paper()


def _patched_deliverable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    deliverable = tmp_path / "experiment_1181_paper_v5_high_issues_6_10.json"
    monkeypatch.setattr(_mod, "_DELIVERABLE", deliverable)
    return deliverable


# ---------------------------------------------------------------------------
# ISSUE-6: GRPO confidence intervals
# ---------------------------------------------------------------------------


class TestIssue6GrpoCIs:
    """REQ-PUBLISH-007 — GRPO claims must include n= and 95% CI annotations."""

    def test_all_grpo_delta_claims_are_annotated(self) -> None:
        """REQ-PUBLISH-007: every GRPO delta claim must carry n= and 95% CI inline."""
        missing = _mod.find_unannotated_grpo_delta_claims(_tex())
        assert missing == [], (
            "ISSUE-6 not fixed everywhere: found GRPO delta claims without inline "
            f"sample size and CI annotations: {missing!r}"
        )

    def test_grpo_delta_checker_reports_missing_annotation(self) -> None:
        """REQ-PUBLISH-007: checker must fail a bare GRPO +8.51 pp claim."""
        missing = _mod.find_unannotated_grpo_delta_claims(
            "GRPO with ThinkPRM improved GSM8K by +8.51 pp."
        )
        assert missing == ["GRPO with ThinkPRM improved GSM8K by +8.51 pp."]

    def test_grpo_checker_ignores_grpo_without_delta_claim(self) -> None:
        """REQ-PUBLISH-007: checker should ignore GRPO mentions with no delta-pp claim."""
        assert _mod.find_unannotated_grpo_delta_claims("GRPO reward model pp context.") == []

    def test_ci_annotations_present(self) -> None:
        """main.tex must contain bracketed CI annotations next to GRPO delta claims."""
        tex = _tex()
        ci_added, _ = _mod.check_issue_6_grpo_cis(tex)
        assert ci_added, (
            "ISSUE-6 not fixed: GRPO delta claims in main.tex are missing "
            "Clopper-Pearson 95% CI annotations (n=25, [X%, Y%] format)."
        )

    def test_small_sample_caveat_present(self) -> None:
        """main.tex must contain the small-sample preliminary-indicator footnote."""
        tex = _tex()
        _, caveat_added = _mod.check_issue_6_grpo_cis(tex)
        assert caveat_added, (
            "ISSUE-6 not fully fixed: small-sample caveat footnote is missing. "
            "Readers need to know n=25-47 CIs are wide and results are preliminary."
        )


# ---------------------------------------------------------------------------
# ISSUE-7: HumanEval 0.0% harness failure
# ---------------------------------------------------------------------------


class TestIssue7HumanEval:
    """REQ-PUBLISH-007 — HumanEval 0.0% must be framed as a harness failure."""

    def test_humaneval_reframed_as_harness_failure(self) -> None:
        """main.tex must describe HumanEval 0.0% as a harness extraction failure."""
        tex = _tex()
        assert _mod.check_issue_7_humaneval(tex), (
            "ISSUE-7 not fixed: HumanEval 0.0% is not explained as a harness "
            "extraction failure and the +36 pp post-fix result is not framed correctly."
        )

    def test_humaneval_anomaly_appendix_present(self) -> None:
        """SCENARIO-PUBLISH-007: HumanEval must be in anomaly context, not headlines."""
        tex = _tex()
        assert r"\section{Harness and Measurement Anomalies}" in tex
        assert r"\label{app:harness-anomalies}" in tex


# ---------------------------------------------------------------------------
# ISSUE-8: alpha_t false rejection rate
# ---------------------------------------------------------------------------


class TestIssue8AlphaT:
    """REQ-PUBLISH-007 — 24/100 false rejection rate must accompany alpha_t=0.38."""

    def test_false_rejection_rate_disclosed(self) -> None:
        """main.tex must disclose that 24/100 accepted-correct responses were rejected."""
        tex = _tex()
        assert _mod.check_issue_8_alpha_t(tex), (
            "ISSUE-8 not fixed: the 24/100 false-rejection rate (ground-truth-correct "
            "responses that Carnot rejected) is not disclosed near alpha_t=0.38."
        )


# ---------------------------------------------------------------------------
# ISSUE-9: Phase-4 pilot trivial baseline caveat
# ---------------------------------------------------------------------------


class TestIssue9Phase4Baseline:
    """REQ-PUBLISH-007 — Phase-4 74.7% result must caveat the random-greedy baseline."""

    def test_baseline_caveat_and_forward_ref(self) -> None:
        """main.tex must note the baseline is intentionally weak and reference exp1189."""
        tex = _tex()
        assert _mod.check_issue_9_phase4_baseline(tex), (
            "ISSUE-9 not fixed: the 74.7% action-reduction claim does not note that "
            "the random-legal-greedy baseline is intentionally weak, or the exp1189 "
            "stronger-baseline forward reference is missing."
        )

    def test_phase4_uses_exp1165_artifact_values(self) -> None:
        """REQ-PUBLISH-007: Phase-4 caveat must preserve the checked-in exp1165 values."""
        compact = " ".join(_tex().split())
        assert "Phase~4: 6.30 mean steps vs greedy: 24.86 mean steps" in compact


# ---------------------------------------------------------------------------
# ISSUE-10: Seed IQ confirmed=false disclosure
# ---------------------------------------------------------------------------


class TestIssue10SeedIQ:
    """REQ-PUBLISH-007 — Seed IQ table row must footnote non-independent verification."""

    def test_seed_iq_footnote_present(self) -> None:
        """main.tex must footnote that Seed IQ was not independently re-fetched."""
        tex = _tex()
        assert _mod.check_issue_10_seed_iq(tex), (
            "ISSUE-10 not fixed: Seed IQ table row lacks the footnote disclosing that "
            "the score was not independently re-fetched (exp1166: "
            "seed_iq_score_confirmed=false)."
        )


# ---------------------------------------------------------------------------
# Integration: run() returns required schema
# ---------------------------------------------------------------------------


class TestRunReturnsSchema:
    """SCENARIO-PUBLISH-007 — run() emits the full required artifact schema."""

    _REQUIRED_BOOL_FIELDS = [
        "issue_6_grpo_cis_added",
        "issue_6_small_sample_caveat_added",
        "issue_7_humaneval_reframed",
        "issue_8_alpha_t_rejection_rate_added",
        "issue_9_phase4_baseline_caveat_added",
        "issue_10_seed_iq_footnote_added",
        "4_test_passes_high",
    ]

    def test_all_schema_fields_present(self) -> None:
        payload = _mod.run()
        for field in self._REQUIRED_BOOL_FIELDS:
            assert field in payload, f"Required field '{field}' missing from artifact"

    def test_high_severity_fixed_is_int(self) -> None:
        payload = _mod.run()
        assert isinstance(payload["high_severity_fixed"], int)

    def test_honest_verdict_valid_value(self) -> None:
        payload = _mod.run()
        allowed = {"all_5_high_resolved", "partial_fix", "blocked"}
        assert payload["honest_verdict"] in allowed, (
            f"honest_verdict={payload['honest_verdict']!r} not in {allowed}"
        )

    def test_all_5_issues_resolved(self) -> None:
        """SCENARIO-PUBLISH-007: when all fixes are applied, high_severity_fixed == 5."""
        payload = _mod.run()
        assert payload["high_severity_fixed"] == 5, (
            f"Expected high_severity_fixed=5, got {payload['high_severity_fixed']}. "
            "One or more ISSUE-6..10 fixes are not yet applied to main.tex."
        )

    def test_honest_verdict_all_resolved(self) -> None:
        payload = _mod.run()
        assert payload["honest_verdict"] == "all_5_high_resolved", (
            f"honest_verdict={payload['honest_verdict']!r}; expected 'all_5_high_resolved'"
        )


# ---------------------------------------------------------------------------
# Deliverable JSON round-trip
# ---------------------------------------------------------------------------


class TestDeliverableJson:
    """Deliverable JSON must be well-formed and contain all required fields."""

    def test_deliverable_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        deliverable = _patched_deliverable(monkeypatch, tmp_path)
        # Patch main to avoid sys.exit on partial fix
        import time

        t0 = time.monotonic()
        payload = _mod.run()
        payload["duration_s"] = round(time.monotonic() - t0, 3)
        deliverable.parent.mkdir(parents=True, exist_ok=True)
        deliverable.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        loaded = json.loads(deliverable.read_text())
        assert loaded["experiment"] == 1181
        assert "honest_verdict" in loaded
        assert "high_severity_fixed" in loaded

    def test_main_writes_success_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        deliverable = _patched_deliverable(monkeypatch, tmp_path)
        _mod.main()

        loaded = json.loads(deliverable.read_text())
        assert loaded["honest_verdict"] == "all_5_high_resolved"
        assert json.loads(capsys.readouterr().out)["high_severity_fixed"] == 5

    def test_main_exits_nonzero_on_partial_fix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        deliverable = _patched_deliverable(monkeypatch, tmp_path)
        monkeypatch.setattr(
            _mod, "run", lambda: {"honest_verdict": "blocked", "high_severity_fixed": 0}
        )

        with pytest.raises(SystemExit) as excinfo:
            _mod.main()

        assert excinfo.value.code == 1
        loaded = json.loads(deliverable.read_text())
        assert loaded["honest_verdict"] == "blocked"
        assert "not all fixes verified" in capsys.readouterr().err
