"""Tests for REQ-REPORT-7126 and its ARC phase-forensics scenarios."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7126_v626_arc_loo_phase_receipts as exp
from scripts import adversarial_verify


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "openspec/capabilities/research-reporting/spec.md"


def _valid_receipts() -> list[dict[str, object]]:
    """Build complete receipts so each attack changes only one condition."""

    wall = datetime(2026, 9, 7, 20, 0, tzinfo=UTC)
    rows = []
    for index, phase in enumerate(exp.PHASES):
        start_ns = 1_000_000_000 + index * 2_000_000_000
        start_wall = wall + timedelta(seconds=index * 2)
        rows.append(
            exp.make_phase_receipt(
                phase=phase,
                monotonic_start_ns=start_ns,
                monotonic_end_ns=start_ns + 1_000_000_000,
                wall_clock_start=start_wall.isoformat().replace("+00:00", "Z"),
                wall_clock_end=(start_wall + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
                deadline=start_ns + 1_500_000_000,
                subprocess_pid=7000 + index,
                exit_state="completed",
                timeout_state="not_timed_out",
                evidence_hash=f"sha256:{index + 1:064x}",
            )
        )
    return rows


def test_req_report_7126_spec_precedes_implementation() -> None:
    """REQ-REPORT-7126 owns every artifact field and named scenario."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7126") :]
    for scenario in ("SOURCES", "TIMELINE", "RECEIPT", "ADVERSARIAL", "BUDGET", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7126-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_7126_receipt_accepts_complete_dual_clock_rows() -> None:
    """SCENARIO-REPORT-7126-RECEIPT accepts complete consistent receipts."""

    rows = _valid_receipts()
    assert exp.validate_phase_receipts(rows) == []
    schema = exp.phase_receipt_schema()
    assert schema["required_phases"] == list(exp.PHASES)
    assert schema["required_fields"] == list(exp.PHASE_RECEIPT_FIELDS)


def test_scenario_report_7126_adversarial_attacks_fail_closed() -> None:
    """SCENARIO-REPORT-7126-ADVERSARIAL rejects every required attack class."""

    matrix = exp.synthetic_attack_matrix(_valid_receipts())
    by_id = {row["attack_id"]: row for row in matrix}
    assert set(by_id) == set(exp.ATTACK_IDS)
    assert all(row["fail_closed"] is True for row in matrix)
    expected = {
        "missing_timestamps": "missing_timestamp",
        "contradictory_clocks": "contradictory_clocks",
        "absent_start_receipt": "absent_start_receipt",
        "absent_end_receipt": "absent_end_receipt",
        "negative_interval": "negative_interval",
        "duplicate_events": "duplicate_phase_event",
        "fabricated_phase_completion": "completion_without_complete_evidence",
    }
    for attack_id, reason in expected.items():
        assert reason in by_id[attack_id]["errors"]


def test_receipt_validator_names_deadline_identity_and_state_failures() -> None:
    """REQ-REPORT-7126 keeps malformed process and deadline facts visible."""

    rows = _valid_receipts()
    rows[0]["deadline"] = rows[0]["monotonic_end_ns"] - 1
    rows[0]["subprocess_pid"] = True
    rows[0]["exit_state"] = "invented"
    rows[0]["timeout_state"] = "invented"
    rows[0]["evidence_hash"] = "sha256:" + "0" * 64
    rows[1]["wall_clock_end"] = "2026-09-07T20:00:01Z"
    rows[2]["evidence_hash"] = "bad"
    rows[-1]["phase"] = "invented"
    errors = set(exp.validate_phase_receipts(rows))
    assert {
        "deadline_before_phase_end",
        "invalid_subprocess_pid",
        "invalid_exit_state",
        "invalid_timeout_state",
        "placeholder_evidence_hash",
        "invalid_evidence_hash",
        "unknown_phase",
        f"missing_phase:{exp.PHASES[-1]}",
    } <= errors

    missing = deepcopy(_valid_receipts())
    del missing[0]["deadline"]
    assert "missing_required_field:artifact_initialization:deadline" in exp.validate_phase_receipts(
        missing
    )


def test_scenario_report_7126_sources_parse_conductor_bounds_and_hashes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7126-SOURCES preserves minute precision as bounds."""

    path = tmp_path / "conductor-log.md"
    path.write_text(
        "\n".join(
            [
                "| 2026-09-07 19:02 UTC | Adapter-withheld ARC leave-one-game-out shard A | FAIL | artifact_verdict_not_terminal |",
                "| 2026-09-07 19:26 UTC | Adapter-withheld ARC leave-one-game-out shard A | FAIL | Codex CLI error: Wall-clock+idle timeout after 1266s (600s silence). Last output |",
                "| 2026-09-07 19:29 UTC | Adapter-withheld ARC leave-one-game-out shard A | SKIP | Pre-tests failing, self-heal failed |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = exp.parse_conductor_rows(path)
    assert [row["event"] for row in rows] == [
        "artifact_postflight_failure",
        "conductor_timeout",
        "task_exit",
    ]
    assert all(row["source_sha256"] == exp.sha256_file(path) for row in rows)
    assert all(row["timestamp_precision_s"] == 60 for row in rows)
    assert rows[1]["elapsed_s"] == 1266
    assert rows[1]["silence_s"] == 600
    assert rows[1]["timestamp_lower"] == "2026-09-07T19:26:00Z"
    assert rows[1]["timestamp_upper"] == "2026-09-07T19:26:59.999999Z"

    empty = tmp_path / "empty.md"
    empty.write_text("no matching rows\n", encoding="utf-8")
    assert exp.parse_conductor_rows(empty) == []


def test_filesystem_parser_preserves_missing_and_second_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7126 leaves a missing birth clock null instead of estimating it."""

    assert exp._parse_stat_timestamp("not-a-clock") is None
    assert exp._parse_stat_timestamp("2026-09-07 12:00:00 +0000") == (
        "2026-09-07T12:00:00Z",
        1.0,
        "2026-09-07 12:00:00 +0000",
    )
    path = tmp_path / "source"
    path.write_text("evidence", encoding="utf-8")
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="-\n2026-09-07 12:00:00 +0000"),
    )
    rows = exp.filesystem_timestamp_rows(path)
    assert rows[0]["event"] == "artifact_filesystem_birth_unavailable"
    assert rows[0]["timestamp"] is None
    assert rows[1]["timestamp_precision_s"] == 1.0


def test_scenario_report_7126_timeline_localizes_setup_without_estimates() -> None:
    """SCENARIO-REPORT-7126-TIMELINE keeps legacy markers separate from clocks."""

    artifact = exp.build_artifact(root=ROOT, run_date="20260907", duration_s=0.5)
    assert artifact["first_absent_start_receipt"] == {
        "phase": "setup",
        "receipt": "legacy_start_marker",
        "reason": "no_setup_start_marker_in_any_source",
    }
    assert [row["phase"] for row in artifact["phase_timing_rows"]] == list(exp.PHASES)
    assert len(artifact["phase_timing_rows"]) == 10
    setup = next(row for row in artifact["phase_timing_rows"] if row["phase"] == "setup")
    assert setup["start_receipt_present"] is False
    assert setup["start_timestamp"] is None
    assert setup["interval_kind"] == "observed_activity_without_phase_boundary"
    assert artifact["observed_stall_interval"]["duration_s"] == 600
    assert artifact["observed_stall_interval"]["start_lower"] == "2026-09-07T19:16:00Z"
    assert artifact["observed_stall_interval"]["end_upper"] == "2026-09-07T19:26:59.999999Z"
    assert any(row["gap"] == "phase_timing_rows_absent" for row in artifact["missing_receipt_rows"])
    assert any(row["event"] == "conductor_timeout" for row in artifact["source_timeline_rows"])
    assert any(row["event"] == "task_exit" for row in artifact["source_timeline_rows"])


def test_scenario_report_7126_budget_and_artifact_recompute() -> None:
    """SCENARIO-REPORT-7126-BUDGET and ARTIFACT recompute without value claims."""

    artifact = exp.build_artifact(root=ROOT, run_date="20260907", duration_s=0.5)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert sum(row["cap_s"] for row in artifact["phase_budget_rows"]) == 3600
    assert [artifact[name] for name in exp.CAP_FIELDS] == [300, 1500, 1500, 300]
    assert all(row["contract_for_future_run"] is True for row in artifact["phase_budget_rows"])
    assert artifact["arc_phase_receipt_contract_ready_score"] == 1
    assert artifact["value_measurement_run"] is False
    assert artifact["solve_claim_made"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive_")
    assert exp.validate_artifact(artifact) == []

    attacked = deepcopy(artifact)
    attacked["phase_timing_rows"][2]["start_receipt_present"] = True
    attacked["reproducibility_checksum"] = exp.artifact_checksum(attacked)
    assert "phase_timing_rows_mismatch" in exp.validate_artifact(attacked)

    malformed = deepcopy(artifact)
    malformed.pop("duration_s")
    malformed["field_principles"]["run_date"] = ""
    malformed["phase_budget_rows"] = []
    malformed["phase_receipt_schema"] = {}
    first_source = next(iter(malformed["source_artifact_hashes"]))
    malformed["source_artifact_hashes"][first_source] = "bad"
    errors = set(exp.validate_artifact(malformed))
    assert {
        "artifact_fields_mismatch",
        "field_principles_mismatch",
        "phase_budget_rows_mismatch",
        "phase_receipt_schema_mismatch",
        f"source_hash_invalid:{first_source}",
    } <= errors


def test_fixed_phase_caps_are_not_treated_as_measured_tautologies() -> None:
    """SCENARIO-REPORT-7126-BUDGET permits contractually symmetric hard caps."""

    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_tautology(
        {
            "setup_cap_s": 300,
            "finalization_cap_s": 300,
            "withheld_arm_cap_s": 1500,
            "control_arm_cap_s": 1500,
        },
        flags,
    )
    assert [flag for flag in flags if flag.severity == "critical"] == []


def test_blocked_artifact_names_missing_evidence_and_cli_writes(tmp_path: Path) -> None:
    """REQ-REPORT-7126 blocks only when no defensible timeline can be built."""

    existing_output = tmp_path / "existing.json"
    existing_output.write_text("{}\n", encoding="utf-8")
    blocked = exp.build_artifact(
        root=tmp_path,
        run_date="20260907",
        duration_s=0.1,
        output_path=existing_output,
    )
    assert blocked["verdict_class"] == "blocked"
    assert str(blocked["honest_verdict"]).startswith("blocked_")
    assert blocked["gate_check_summary"]["failed_check"] == "defensible_timeline"
    assert blocked["gate_check_summary"]["expected_value"] is True
    assert blocked["gate_check_summary"]["observed_value"] is False
    assert exp.validate_artifact(blocked) == []

    output = tmp_path / "artifact.json"
    assert exp.main(["--date", "20260907", "--root", str(ROOT), "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["run_date"] == "20260907"
    assert exp.validate_artifact(written) == []


def test_run_refuses_to_write_an_invalid_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7126 validates before publishing even for caller-owned paths."""

    output = tmp_path / "invalid.json"
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forged"])
    with pytest.raises(ValueError, match="forged"):
        exp.run(root=ROOT, run_date="20260907", output_path=output)
    assert not output.exists()
