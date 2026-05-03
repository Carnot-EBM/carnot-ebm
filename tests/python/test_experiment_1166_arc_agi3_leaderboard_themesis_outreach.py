"""Tests for Exp 1166 ARC-AGI-3 positioning and Themesis outreach.

Spec traces: REQ-KONA-013, SCENARIO-KONA-013.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import experiment_1166_arc_agi3_leaderboard_themesis_outreach as exp1166  # noqa: E402


def _exp1165_payload() -> dict[str, object]:
    return {
        "action_count_ratio": 0.25341914722445696,
        "phase4_solved_rate": 1.0,
        "phase4_mean_action_count": 6.3,
        "baseline_mean_action_count": 24.86,
    }


def test_fetch_evidence_confirms_seed_iq_when_leaderboard_row_is_exposed() -> None:
    """REQ-KONA-013: current leaderboard data can independently confirm Seed IQ."""

    def fetch_text(url: str) -> str:
        if url == exp1166.LEADERBOARD_V3_DATA_URL:
            return json.dumps(
                {
                    "generatedAt": "2026-05-01T18:58:56.508Z",
                    "evaluations": [
                        {
                            "modelDisplayName": "Seed IQ",
                            "providerDisplayName": "Themesis",
                            "score": 1.0,
                        }
                    ],
                }
            )
        return "<html>ARC-AGI-3 Leaderboard</html>"

    evidence = exp1166.fetch_leaderboard_evidence(fetch_text)

    assert evidence.seed_iq_score_confirmed is True
    assert evidence.seed_iq_score == 1.0
    assert evidence.honest_verdict == "comparison_documented_email_drafted"
    assert evidence.source == "arcprize_v3_json"


def test_fetch_evidence_falls_back_when_seed_iq_row_is_not_visible() -> None:
    """REQ-KONA-013: missing Seed IQ rows use documented fallback values honestly."""

    def fetch_text(url: str) -> str:
        if url == exp1166.LEADERBOARD_V3_DATA_URL:
            return json.dumps(
                {
                    "generatedAt": "2026-05-01T18:58:56.508Z",
                    "evaluations": [{"modelDisplayName": "Anthropic Opus 4.6", "score": 0.0051}],
                }
            )
        return "<html>ARC-AGI-3 Leaderboard</html>"

    evidence = exp1166.fetch_leaderboard_evidence(fetch_text)

    assert evidence.seed_iq_score_confirmed is False
    assert evidence.seed_iq_score == 1.0
    assert "0.95" in evidence.note
    assert evidence.honest_verdict == "leaderboard_unavailable_email_drafted"


def test_fetch_evidence_can_confirm_from_html_or_survive_fetch_errors() -> None:
    """REQ-KONA-013: HTML confirmation and fetch failures are handled honestly."""

    def html_fetch(url: str) -> str:
        if url == exp1166.LEADERBOARD_URL:
            return "<html><tr><td>Seed IQ</td><td>score 0.95</td></tr></html>"
        return json.dumps({"generatedAt": "2026-05-01T18:58:56.508Z", "evaluations": []})

    html_evidence = exp1166.fetch_leaderboard_evidence(html_fetch)

    assert html_evidence.seed_iq_score_confirmed is True
    assert html_evidence.seed_iq_score == 0.95
    assert html_evidence.source == "arcprize_html"
    assert html_evidence.generated_at == "2026-05-01T18:58:56.508Z"

    def raising_fetch(_: str) -> str:
        raise OSError("offline")

    fallback = exp1166.fetch_leaderboard_evidence(raising_fetch)
    assert fallback.seed_iq_score_confirmed is False
    assert fallback.generated_at is None


def test_build_artifact_contains_comparison_table_and_operator_email() -> None:
    """SCENARIO-KONA-013: artifact includes the table and ready-for-review email."""

    evidence = exp1166.LeaderboardEvidence(
        seed_iq_score_confirmed=False,
        seed_iq_score=1.0,
        seed_iq_action_efficiency=exp1166.SEED_IQ_ACTION_EFFICIENCY,
        source="documented_fallback",
        note="Seed IQ 0.95 public demo independently documented; 1.00 row not visible.",
        generated_at=None,
        honest_verdict="leaderboard_unavailable_email_drafted",
    )

    artifact = exp1166.build_artifact(_exp1165_payload(), evidence)

    assert exp1166.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["carnot_phase4_action_count_ratio"] == pytest.approx(0.25341914722445696)
    assert artifact["carnot_phase4_solved_rate"] == 1.0
    assert artifact["seed_iq_score_confirmed"] is False
    assert artifact["honest_verdict"] == "leaderboard_unavailable_email_drafted"
    assert [row["system_name"] for row in artifact["leaderboard_comparison_table"]] == [
        "Seed IQ (Active Inference)",
        "Carnot Phase 4 pilot",
        "Frontier LLMs (autoregressive)",
    ]

    email = artifact["themesis_email_text"]
    assert "To: Denise Holt / Denis O. at Themesis" in email
    assert "From: Ian Blenke <ian@blenke.com>" in email
    assert "Subject: Carnot EBM + Active Inference" in email
    assert "Apache 2.0" in email
    assert "multi-vendor" in email
    assert "action_count_ratio=0.253419" in email
    assert "joint benchmark evaluation" in email
    assert "icblenke@gmail.com" not in email
    assert exp1166.count_email_words(email) < 300


def test_fetch_and_validation_helpers_cover_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KONA-013: validation failures are explicit rather than silent."""

    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b"leaderboard body"

    monkeypatch.setattr(
        exp1166.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse()
    )
    assert exp1166._default_fetch_text("https://example.invalid") == "leaderboard body"

    evidence = exp1166.LeaderboardEvidence(
        seed_iq_score_confirmed=True,
        seed_iq_score=1.0,
        seed_iq_action_efficiency=exp1166.SEED_IQ_ACTION_EFFICIENCY,
        source="test",
        note="test",
        generated_at=None,
        honest_verdict="comparison_documented_email_drafted",
    )
    with pytest.raises(KeyError, match="phase4_solved_rate"):
        exp1166.build_artifact({"action_count_ratio": 0.9}, evidence)

    monkeypatch.setattr(exp1166, "REQUIRED_ARTIFACT_FIELDS", {"missing_field"})
    with pytest.raises(AssertionError, match="missing required artifact fields"):
        exp1166.build_artifact(_exp1165_payload(), evidence)

    monkeypatch.setattr(exp1166, "REQUIRED_ARTIFACT_FIELDS", {"honest_verdict"})
    bad_verdict = exp1166.LeaderboardEvidence(
        seed_iq_score_confirmed=True,
        seed_iq_score=1.0,
        seed_iq_action_efficiency=exp1166.SEED_IQ_ACTION_EFFICIENCY,
        source="test",
        note="test",
        generated_at=None,
        honest_verdict="bad_verdict",
    )
    with pytest.raises(AssertionError, match="unsupported honest_verdict"):
        exp1166.build_artifact(_exp1165_payload(), bad_verdict)

    monkeypatch.setattr(
        exp1166,
        "draft_themesis_email",
        lambda *_args: "word " * 300,
    )
    with pytest.raises(AssertionError, match="under 300 words"):
        exp1166.build_artifact(_exp1165_payload(), evidence)


def test_run_experiment_writes_deliverable_from_exp1165(tmp_path: Path) -> None:
    """SCENARIO-KONA-013: runner writes the required Exp 1166 JSON fields."""

    exp1165_path = tmp_path / "experiment_1165.json"
    deliverable_path = tmp_path / "experiment_1166.json"
    exp1165_path.write_text(json.dumps(_exp1165_payload()), encoding="utf-8")

    def fetch_text(_: str) -> str:
        return ""

    artifact = exp1166.run_experiment(
        exp1165_path=exp1165_path,
        deliverable_path=deliverable_path,
        fetch_text=fetch_text,
    )

    written = json.loads(deliverable_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["themesis_email_drafted"] is True
    assert written["seed_iq_score"] == 1.0
    assert written["carnot_phase4_action_count_ratio"] == pytest.approx(0.25341914722445696)


def test_main_prints_artifact(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-KONA-013: CLI entrypoint prints the written artifact."""

    monkeypatch.setattr(exp1166, "run_experiment", lambda: {"experiment": 1166, "ok": True})

    assert exp1166.main() == 0
    assert '"experiment": 1166' in capsys.readouterr().out
