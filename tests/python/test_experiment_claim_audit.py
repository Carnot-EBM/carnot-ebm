"""Tests for scripts/experiment_claim_audit.py — the claim-refutation audit.

Spec refs: REQ-OPS-CLAIM-REFUTATION-6650 and its scenarios (see
openspec/capabilities/research-harnesses/spec.md): invented-evidence-is-voided,
absence-talk-is-not-voided, budget-writes-partial, flagged-is-skipped,
never-edits.

Mirrors the sibling audits' untested-by-design pattern for the thin LLM-CLI
subprocess wrapper (`_call` needs a live external CLI; it is exercised by use).
Everything else is pure logic and is covered here: verdict parsing, the
audit-integrity guard, budget handling, the flagged-artifact skip, condensation,
and the adversarial_verify delegation pre-pass. All reviewer calls are
monkeypatched; every write goes to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.experiment_claim_audit as eca


def _artifact(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def _review(verdict: str, evidence: str, refute: str = "an arm loss") -> str:
    return (
        f"## VERDICT\n{verdict}\n\n"
        "## THE HEADLINE CLAIM\nmethod beats baseline\n\n"
        f"## WHAT WOULD REFUTE IT\n{refute}\n\n"
        "## WAS THAT CHECKED\nno\n\n"
        f"## EVIDENCE\n{evidence}\n\n"
        "## RECOMMENDATION\nNARROW_CLAIM\n"
    )


class TestParsers:
    # REQ-OPS-CLAIM-REFUTATION-6650: verdicts come from a fixed enum.
    def test_parse_verdict_known(self) -> None:
        assert eca.parse_verdict(_review("CLAIM_OVERSTATED", "none")) == "CLAIM_OVERSTATED"

    def test_parse_verdict_unknown_token(self) -> None:
        assert eca.parse_verdict(_review("SOMETHING_ELSE", "none")) == "UNKNOWN"

    def test_parse_recommendation(self) -> None:
        assert eca.parse_recommendation(_review("NO_CLAIM", "none")) == "NARROW_CLAIM"
        assert eca.parse_recommendation("## RECOMMENDATION\nDANCE\n") == "NONE"


class TestIntegrityGuard:
    # SCENARIO-OPS-CLAIM-REFUTATION-6650-INVENTED-EVIDENCE-IS-VOIDED
    def test_invented_evidence_downgrades_flagged_verdict(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        art = _artifact(
            tmp_path,
            "experiment_9001_x.json",
            {"honest_verdict": "complete_positive: x beats y", "real_field": 1},
        )
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        monkeypatch.setattr(
            eca,
            "_call",
            lambda *a, **k: (True, _review("CLAIM_OVERSTATED", "`totally_invented_field`")),
        )
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        text = report_path.read_text()
        assert "CANNOT_DETERMINE" in text
        assert "Audit-integrity guard" in text
        assert "CLAIM_OVERSTATED**" not in text

    # SCENARIO-OPS-CLAIM-REFUTATION-6650-ABSENCE-TALK-IS-NOT-VOIDED
    def test_absence_talk_in_refutation_sections_is_exempt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        art = _artifact(
            tmp_path,
            "experiment_9002_x.json",
            {"honest_verdict": "complete_positive: x beats y", "real_field": 1},
        )
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        monkeypatch.setattr(
            eca,
            "_call",
            lambda *a, **k: (
                True,
                _review(
                    "CLAIM_OVERSTATED",
                    "`real_field`",
                    refute="a `missing_holdout_control` arm the artifact lacks",
                ),
            ),
        )
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        text = report_path.read_text()
        assert "**CLAIM_OVERSTATED**" in text
        assert "Audit-integrity guard" not in text

    def test_supported_verdict_is_never_swept(self, tmp_path: Path, monkeypatch) -> None:
        # The guard exists to stop false ACCUSATIONS; a passing verdict that
        # hallucinates evidence changes nothing an operator would act on.
        art = _artifact(tmp_path, "experiment_9003_x.json", {"honest_verdict": "complete: ok"})
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        monkeypatch.setattr(
            eca,
            "_call",
            lambda *a, **k: (True, _review("CLAIM_SUPPORTED", "`invented_but_harmless`")),
        )
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        assert "**CLAIM_SUPPORTED**" in report_path.read_text()

    def test_verify_quoted_evidence_token_fallback(self) -> None:
        # A span joining field and value can differ from the JSON dump byte-wise;
        # it counts as present when every long token appears in the body.
        body = '{"exact_success_rate": 1.0, "arm": "violation_count"}'
        rep = "## EVIDENCE\n`exact_success_rate = 1.0 for violation_count`\n"
        assert eca.verify_quoted_evidence(rep, body) == []
        rep_bad = "## EVIDENCE\n`exact_success_rate for phantom_metric`\n"
        assert eca.verify_quoted_evidence(rep_bad, body) != []


class TestBudget:
    # SCENARIO-OPS-CLAIM-REFUTATION-6650-BUDGET-WRITES-PARTIAL
    def test_budget_exhaustion_writes_partial_report(self, tmp_path: Path, monkeypatch) -> None:
        arts = [
            _artifact(tmp_path, f"experiment_910{i}_x.json", {"honest_verdict": "complete: ok"})
            for i in range(3)
        ]
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        calls: list[str] = []

        def fake_call(agent, model, prompt, body):
            calls.append(body)
            return True, _review("CLAIM_SUPPORTED", "none")

        monkeypatch.setattr(eca, "_call", fake_call)
        # Fake clock: deadline calc sees t=0 (deadline=10); the first loop check
        # sees t=7.5 (inside budget); the second sees t=15 (exhausted).
        ticks = iter([0.0, 7.5, 15.0, 22.5, 30.0])
        monkeypatch.setattr(eca, "_now", lambda: next(ticks))
        rc = eca.main(
            ["--budget-seconds", "10"] + [x for a in arts for x in ("--artifact", str(a))]
        )
        assert rc == 0
        assert len(calls) == 1
        text = report_path.read_text()
        assert "PARTIAL RUN" in text
        assert "1 of 3" in text
        assert text.count("NOT_REVIEWED_BUDGET") >= 2

    def test_zero_budget_disables_deadline(self, tmp_path: Path, monkeypatch) -> None:
        art = _artifact(tmp_path, "experiment_9110_x.json", {"honest_verdict": "complete: ok"})
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        monkeypatch.setattr(eca, "_call", lambda *a, **k: (True, _review("NO_CLAIM", "none")))
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        assert "PARTIAL RUN" not in report_path.read_text()


class TestSkipsAndReceipt:
    # SCENARIO-OPS-CLAIM-REFUTATION-6650-FLAGGED-IS-SKIPPED
    def test_flagged_adversarial_skipped_without_reviewer_call(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        art = _artifact(
            tmp_path,
            "experiment_9200_x.json",
            {"honest_verdict": "complete: fab", "flagged_adversarial": True},
        )
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        calls: list[str] = []
        monkeypatch.setattr(
            eca, "_call", lambda *a, **k: calls.append("x") or (True, _review("NO_CLAIM", "none"))
        )
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        assert calls == []
        assert "SKIPPED_ALREADY_FLAGGED" in report_path.read_text()

    def test_report_written_when_reviewer_fails(self, tmp_path: Path, monkeypatch) -> None:
        # The report is the RECEIPT: it must exist even when every call fails.
        art = _artifact(tmp_path, "experiment_9201_x.json", {"honest_verdict": "complete: ok"})
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        monkeypatch.setattr(eca, "_call", lambda *a, **k: (False, "boom"))
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        text = report_path.read_text()
        assert "CANNOT_DETERMINE" in text
        assert "reviewer call failed" in text

    # SCENARIO-OPS-CLAIM-REFUTATION-6650-NEVER-EDITS
    def test_never_edits_audited_artifact(self, tmp_path: Path, monkeypatch) -> None:
        art = _artifact(
            tmp_path,
            "experiment_9202_x.json",
            {"honest_verdict": "complete: ok", "rows": [1, 2, 3]},
        )
        before = art.read_bytes()
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        monkeypatch.setattr(
            eca, "_call", lambda *a, **k: (True, _review("CLAIM_SUPPORTED", "none"))
        )
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        assert art.read_bytes() == before


class TestPrepassDelegation:
    # REQ-OPS-CLAIM-REFUTATION-6650 rule 2: degenerate-corpus detection is
    # DELEGATED to adversarial_verify, not re-implemented.
    def test_prepass_surfaces_false_negative_risk(self) -> None:
        d = {
            "honest_verdict": "complete_no_improvement: reranker does not beat sc",
            "flip_count": 0,
        }
        notes = "\n".join(eca.prepass_notes(d))
        assert "FALSE_NEGATIVE_RISK" in notes

    def test_prepass_notes_reach_the_reviewer(self, tmp_path: Path, monkeypatch) -> None:
        art = _artifact(
            tmp_path,
            "experiment_9300_x.json",
            {
                "honest_verdict": "complete_no_improvement: does not beat baseline",
                "flip_count": 0,
            },
        )
        report_path = tmp_path / "report.md"
        monkeypatch.setattr(eca, "REPORT", report_path)
        seen: list[str] = []

        def fake_call(agent, model, prompt, body):
            seen.append(body)
            return True, _review("CLAIM_SUPPORTED", "none")

        monkeypatch.setattr(eca, "_call", fake_call)
        assert eca.main(["--artifact", str(art), "--budget-seconds", "0"]) == 0
        assert len(seen) == 1
        assert "MECHANICAL PRE-PASS NOTES" in seen[0]
        assert "FALSE_NEGATIVE_RISK" in seen[0]


class TestCondense:
    def test_condense_elides_long_lists_and_keeps_claim_fields(self) -> None:
        d = {
            "honest_verdict": "complete_positive: big",
            "per_unit_rows": [{"i": i} for i in range(500)],
        }
        out = eca.condense(d)
        assert out["honest_verdict"] == "complete_positive: big"
        assert len(out["per_unit_rows"]) == 4  # 3 head entries + elision marker
        assert "497 more entries elided" in out["per_unit_rows"][-1]

    def test_condense_leaves_short_lists_alone(self) -> None:
        d = {"rows": [1, 2, 3]}
        assert eca.condense(d) == {"rows": [1, 2, 3]}
