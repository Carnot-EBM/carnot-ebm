"""Artifact tests for REQ-ARC-6859 first-party receipt wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6859_first_party_tool_gap_receipt_wiring as exp


REPO = Path(__file__).resolve().parents[2]


def _router(root: Path, *, ready: int = 1, first_party_rows: list[dict] | None = None) -> None:
    path = root / exp.ROUTER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "carnot.experiment_6857.dynamic_live_arc_receipt_router.v1",
                "arc_receipt_router_complete_score": ready,
                "first_party_tool_gap_rows": first_party_rows or [],
                "solve_claim": False,
                "game_level_solve_count": 0,
            }
        ),
        encoding="utf-8",
    )


def _authentic_row() -> dict:
    return {
        "receipt_identity": "sha256:" + "b" * 64,
        "provenance_class": "authentic_live",
        "first_party": True,
        "live_reachable": True,
        "agent_visible": True,
        "response_used": True,
        "next_action_recorded": True,
        "exact_outcome": True,
        "valid_headroom": True,
        "join_complete": True,
        "quarantine_reason": None,
        "levels_before": 0,
        "levels_after": 1,
    }


def test_req_arc_6859_spec_declares_full_contract() -> None:
    """REQ-ARC-6859: the spec owns every required field and scenario."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-ARC-6859", 1)[1]
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for scenario in (
        "NO-GAP-AND-DEFAULT-OFF",
        "GAP-REJECTION-AND-TOOL-ERROR",
        "VISIBILITY-USE-AND-NEXT-ACTION",
        "EXACT-OUTCOME-AND-CAUSAL-ELIGIBILITY",
        "PERSISTENCE-RESTART-AND-DEDUPLICATION",
        "FIXTURE-REPLAY-AND-PROVENANCE",
        "BLOCKED-GATE-AND-NO-SOLVE",
    ):
        assert f"SCENARIO-ARC-6859-{scenario}" in section
    assert exp.INFERENCE_SUBSTRATE in section


def test_scenario_arc_6859_fixture_contract_is_ready_but_not_live_effect_eligible(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6859-FIXTURE-REPLAY-AND-PROVENANCE cannot claim live utility."""

    _router(tmp_path)
    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.25)

    assert exp.validate_artifact(artifact) == []
    assert artifact["tool_gap_receipt_contract_ready_score"] == 1
    assert artifact["tool_gap_live_effect_claim_eligible_score"] == 0
    assert artifact["default_off_verified"] is True
    assert artifact["solve_claimed"] is False
    assert artifact["game_level_solve_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == (
        "complete_first_party_tool_gap_receipt_contract_ready_no_live_effect_claim"
    )
    assert set(artifact["field_principles"]) == set(artifact)
    assert any(row["row_kind"] == "gap_chain" for row in artifact["rows"])
    assert any(row["row_kind"] == "hop_metrics" for row in artifact["rows"])
    classes = {
        row["provenance_class"]: row["row_count"] for row in artifact["provenance_class_rows"]
    }
    assert classes["fixture"] >= 1
    assert classes["authentic_live"] == 0
    assert classes["terminal_replay"] == 0


def test_scenario_arc_6859_authentic_exact_row_alone_opens_effect_eligibility(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6859-EXACT-OUTCOME-AND-CAUSAL-ELIGIBILITY excludes fixtures."""

    _router(tmp_path, first_party_rows=[_authentic_row()])
    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.25)

    assert artifact["tool_gap_receipt_contract_ready_score"] == 1
    assert artifact["tool_gap_live_effect_claim_eligible_score"] == 1
    assert artifact["verdict_class"] == "positive"
    assert artifact["solve_claimed"] is False
    assert any(
        row["provenance_class"] == "authentic_live" and row["causal_eligible"]
        for row in artifact["join_completeness_rows"]
    )


def test_scenario_arc_6859_proxy_and_reconstructed_rows_are_quarantined(tmp_path: Path) -> None:
    """SCENARIO-ARC-6859-FIXTURE-REPLAY-AND-PROVENANCE rejects proxy utility."""

    proxy = _authentic_row()
    proxy["provenance_class"] = "development_proxy"
    reconstructed = _authentic_row()
    reconstructed["receipt_identity"] = "sha256:" + "c" * 64
    reconstructed["provenance_class"] = "reconstructed"
    _router(tmp_path, first_party_rows=[proxy, reconstructed])

    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.25)
    assert artifact["tool_gap_live_effect_claim_eligible_score"] == 0
    replay = [
        row for row in artifact["join_completeness_rows"] if row["source"] == "terminal_replay"
    ]
    assert {row["quarantine_reason"] for row in replay} == {
        "development_proxy_not_live_eligible",
        "reconstructed_not_live_eligible",
    }


def test_scenario_arc_6859_blocked_precondition_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-6859-BLOCKED-GATE-AND-NO-SOLVE reports the observed gate."""

    _router(tmp_path, ready=0)
    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.25)

    assert artifact["tool_gap_receipt_contract_ready_score"] == 0
    assert artifact["tool_gap_live_effect_claim_eligible_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == ("complete_blocked_first_party_tool_gap_receipt_wiring")
    assert artifact["gate_check_summary"]["failed_check"] == ("arc_receipt_router_complete_score")
    assert artifact["gate_check_summary"]["observed"] == 0


def test_req_arc_6859_reproducibility_ignores_duration(tmp_path: Path) -> None:
    """REQ-ARC-6859: deterministic fixtures and terminal replay hash stably."""

    _router(tmp_path)
    first = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.1)
    second = exp.build_artifact(tmp_path, run_date="20260901", duration_s=9.9)
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]


def test_req_arc_6859_atomic_cli_write(tmp_path: Path) -> None:
    """REQ-ARC-6859: the requested command writes one valid stable artifact."""

    _router(tmp_path)
    output = tmp_path / "artifact.json"
    assert exp.main(["--date", "20260901", "--root", str(tmp_path), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(artifact) == []
    assert artifact["schema"] == exp.ARTIFACT_SCHEMA


@pytest.mark.parametrize("content", [None, "{broken", "[]"])
def test_req_arc_6859_router_reader_failures_are_blocked(
    tmp_path: Path,
    content: str | None,
) -> None:
    """REQ-ARC-6859: missing, invalid, and scalar router bytes fail closed."""

    path = tmp_path / exp.ROUTER_PATH
    if content is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.1)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == ("arc_receipt_router_complete_score")


def test_req_arc_6859_default_off_check_restores_existing_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-6859: the default-off fixture does not alter caller configuration."""

    configured = tmp_path / "configured.json"
    monkeypatch.setenv(exp.receipt.ENABLE_ENV, "1")
    monkeypatch.setenv(exp.receipt.PATH_ENV, str(configured))
    assert exp._configured_default_off_check() is True
    assert exp.os.environ[exp.receipt.ENABLE_ENV] == "1"
    assert exp.os.environ[exp.receipt.PATH_ENV] == str(configured)


def test_req_arc_6859_replay_reader_skips_wrong_shapes() -> None:
    """REQ-ARC-6859: malformed replay containers cannot invent receipt rows."""

    assert exp._normalize_replay_rows({"first_party_tool_gap_rows": {"bad": 1}}) == []
    rows = exp._normalize_replay_rows(
        {"first_party_tool_gap_rows": ["not-a-row", {"first_party": True}]}
    )
    assert len(rows) == 1
    assert rows[0]["provenance_class"] == "terminal_replay"


def test_req_arc_6859_partial_contract_verdict_is_row_supported(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-ARC-6859: a failed fixture is partial, not a fabricated ready score."""

    _router(tmp_path)
    monkeypatch.setattr(
        exp,
        "_exercise_fixtures",
        lambda: {
            "rows": [],
            "clean_no_gap": False,
            "action_identity_preserved": False,
            "restart": {},
            "deduplication": {},
        },
    )
    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.1)
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("complete_partial_")


def test_req_arc_6859_validator_reports_every_contract_violation(tmp_path: Path) -> None:
    """REQ-ARC-6859: validation names every malformed terminal field."""

    _router(tmp_path)
    artifact = exp.build_artifact(tmp_path, run_date="20260901", duration_s=0.1)
    artifact.pop("rows")
    artifact["schema"] = "wrong"
    artifact["inference_substrate"] = "wrong"
    artifact["solve_claimed"] = True
    artifact["game_level_solve_count"] = 1
    artifact["verifier_is_oracle"] = True
    artifact["verdict_class"] = "wrong"
    artifact["honest_verdict"] = "partial"
    artifact["field_principles"] = {}
    artifact["tool_gap_live_effect_claim_eligible_score"] = 1
    artifact["join_completeness_rows"] = []
    errors = exp.validate_artifact(artifact)
    assert "missing field: rows" in errors
    assert "wrong schema" in errors
    assert "wrong inference substrate" in errors
    assert "receipt wiring cannot claim a solve" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "invalid verdict class" in errors
    assert "honest_verdict is not terminal" in errors
    assert "field_principles do not cover every top-level field" in errors
    assert "live effect score lacks an authentic exact row" in errors


def test_req_arc_6859_cli_default_paths_bad_date_and_validation_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """REQ-ARC-6859: CLI defaults and all terminal error exits are covered."""

    _router(tmp_path)
    output = tmp_path / "default-output.json"
    monkeypatch.setattr(exp, "repo_root", lambda **kwargs: tmp_path)
    monkeypatch.setattr(exp, "results_path", lambda *args, **kwargs: output)
    assert exp.main(["--date", "20260901"]) == 0
    assert output.exists()
    with pytest.raises(SystemExit):
        exp.main(["--date", "2026-09-01"])
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["fixture-invalid"])
    with pytest.raises(SystemExit, match="fixture-invalid"):
        exp.main(["--date", "20260901"])
