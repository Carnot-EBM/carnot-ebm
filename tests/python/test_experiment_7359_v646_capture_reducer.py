"""Tests for REQ-REPORT-7359 and SCENARIO-REPORT-7359-*.

The fixtures are small byte records. They exercise the cold accounting path
without loading a model or changing the archived Exp7348 result.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7359_v646_capture_reducer as mod


ROOT = Path(__file__).resolve().parents[2]


def _schedule() -> list[dict[str, object]]:
    """Build two identifier twins and one independent cancelled call."""

    return [
        {
            "call_index": 0,
            "call_id": "call-0",
            "request_id": "request-original",
            "panel_id": "panel-0",
            "candidate_index": 0,
            "pair_side": "original",
            "prompt_sha256": "sha256:prompt-0",
            "public_request_sha256": "sha256:request-0",
        },
        {
            "call_index": 1,
            "call_id": "call-1",
            "request_id": "request-twin",
            "panel_id": "panel-0",
            "candidate_index": 0,
            "pair_side": "twin",
            "prompt_sha256": "sha256:prompt-1",
            "public_request_sha256": "sha256:request-1",
        },
        {
            "call_index": 2,
            "call_id": "call-2",
            "request_id": "request-cancelled",
            "panel_id": "panel-1",
            "candidate_index": 0,
            "pair_side": "original",
            "prompt_sha256": "sha256:prompt-2",
            "public_request_sha256": "sha256:request-2",
        },
    ]


def _calls(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach one response, one request error, and one cancellation."""

    states = ("response", "request_error", "cancelled")
    rows: list[dict[str, object]] = []
    for schedule_row, state in zip(schedule, states, strict=True):
        attempted = state != "cancelled"
        raw_reply = '{"ok":true}' if state == "response" else ""
        rows.append(
            {
                **schedule_row,
                "attempted": attempted,
                "terminal_state": state,
                "censored": state != "response",
                "raw_reply": raw_reply,
                "raw_reply_sha256": mod.sha256_text(raw_reply),
                "parse_status": "valid" if state == "response" else "invalid",
                "runtime_identity_receipt": {"pid": 17, "model_sha256": "sha256:model"},
            }
        )
    return rows


def _candidate_bytes(calls: list[dict[str, object]]) -> dict[str, bytes]:
    """Use the same stable bytes required from archived candidate files."""

    return {str(row["call_id"]): mod.canonical_bytes(row) + b"\n" for row in calls}


def _reduce_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, bytes],
    str,
]:
    """Return one complete schedule and its immutable evidence boundary."""

    schedule = _schedule()
    calls = _calls(schedule)
    return schedule, calls, _candidate_bytes(calls), mod.sha256_json(schedule)


def test_scenario_report_7359_failures_count_terminal_states_without_quality() -> None:
    """SCENARIO-REPORT-7359-FAILURES keeps all terminal states in coverage."""

    schedule, calls, candidates, schedule_hash = _reduce_fixture()
    evaluator_rows = [
        {"call_id": "call-0", "hidden_rule_accepted": False},
        {"call_id": "call-1", "hidden_rule_accepted": None},
        {"call_id": "call-2", "hidden_rule_accepted": None},
    ]
    reduced = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
        evaluator_rows=evaluator_rows,
    )

    assert reduced["errors"] == []
    assert reduced["capture_complete_score"] == 1
    assert reduced["reduced_budget"] == {
        "planned_units": 3,
        "attempted_units": 2,
        "completed_units": 1,
        "failed_units": 1,
        "cancelled_units": 1,
        "censored_units": 2,
    }
    assert len(reduced["call_rows"]) == 3
    assert reduced["identifier_twin_rows"][0]["original_call_id"] == "call-0"
    assert reduced["identifier_twin_rows"][0]["twin_call_id"] == "call-1"
    assert reduced["semantic_failure_counts"] == {
        "source_parse_failures": 2,
        "hidden_rule_failures": 1,
    }
    assert all(row["runtime_receipt_sha256"].startswith("sha256:") for row in reduced["call_rows"])
    assert mod.independent_reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    ) == {
        "capture_complete_score": 1,
        "errors": [],
        "reduced_budget": reduced["reduced_budget"],
    }


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing", "missing_call:call-2"),
        ("duplicate", "duplicate_call_id:call-0"),
        ("candidate_bytes", "candidate_bytes_mismatch:call-0"),
        ("request_id", "request_id_mismatch:call-0"),
        ("non_terminal", "terminal_state_invalid:call-0"),
        ("attempted_state", "attempted_state_mismatch:call-2"),
        ("partial_schedule", "schedule_sha256_mismatch"),
    ),
)
def test_scenario_report_7359_integrity_rejects_mutations(
    mutation: str, expected_error: str
) -> None:
    """SCENARIO-REPORT-7359-INTEGRITY rejects changed schedule or call evidence."""

    schedule, calls, candidates, schedule_hash = _reduce_fixture()
    if mutation == "missing":
        calls.pop()
        candidates.pop("call-2")
    elif mutation == "duplicate":
        calls.append(deepcopy(calls[0]))
    elif mutation == "candidate_bytes":
        candidates["call-0"] += b" "
    elif mutation == "request_id":
        calls[0]["request_id"] = "changed"
        candidates = _candidate_bytes(calls)
    elif mutation == "non_terminal":
        calls[0]["terminal_state"] = "in_flight"
        candidates = _candidate_bytes(calls)
    elif mutation == "attempted_state":
        calls[2]["attempted"] = True
        candidates = _candidate_bytes(calls)
    else:
        schedule.pop()

    reduced = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    independent = mod.independent_reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    assert expected_error in reduced["errors"]
    assert expected_error in independent["errors"]
    assert reduced["capture_complete_score"] == 0
    assert independent["capture_complete_score"] == 0


def test_scenario_report_7359_circularity_ignores_final_scores() -> None:
    """SCENARIO-REPORT-7359-CIRCULARITY excludes readiness from call accounting."""

    schedule, calls, candidates, schedule_hash = _reduce_fixture()
    first = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
        final_readiness_score=0,
    )
    second = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
        final_readiness_score=1,
    )
    assert first == second
    assert first["capture_complete_score"] == 1


def test_scenario_report_7359_integrity_names_remaining_identity_failures() -> None:
    """SCENARIO-REPORT-7359-INTEGRITY names each byte and identity fault."""

    schedule, calls, candidates, schedule_hash = _reduce_fixture()
    schedule[0]["call_id"] = None
    schedule[1]["call_id"] = "call-2"
    calls[0]["call_id"] = "unexpected"
    calls[0]["raw_reply_sha256"] = "sha256:changed"
    calls[0]["runtime_identity_receipt"] = None
    candidates.pop("call-1")
    reduced = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    independent = mod.independent_reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    assert {
        "schedule_call_id_missing",
        "duplicate_schedule_call_id:call-2",
        "unexpected_call:unexpected",
        "missing_call:None",
    } <= set(reduced["errors"])
    assert "schedule_call_id_missing" in independent["errors"]
    assert "duplicate_schedule_call_id:call-2" in independent["errors"]

    schedule, calls, candidates, schedule_hash = _reduce_fixture()
    calls[0]["raw_reply_sha256"] = "sha256:changed"
    calls[0]["runtime_identity_receipt"] = None
    candidates = _candidate_bytes(calls)
    reduced = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    assert "raw_reply_sha256_mismatch:call-0" in reduced["errors"]
    assert "runtime_receipt_missing:call-0" in reduced["errors"]

    schedule, calls, candidates, schedule_hash = _reduce_fixture()
    candidates.pop("call-0")
    reduced = mod.reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    independent = mod.independent_reduce_capture(
        schedule,
        calls,
        candidates,
        expected_schedule_sha256=schedule_hash,
    )
    assert "candidate_bytes_missing:call-0" in reduced["errors"]
    assert "candidate_bytes_missing:call-0" in independent["errors"]


def test_req_report_7359_hash_load_and_terminal_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-7359 keeps byte hashes, JSON failures, and receipt gates exact."""

    path = tmp_path / "value.json"
    path.write_text('{"value":1}\n', encoding="utf-8")
    assert mod.sha256_file(path) == mod.sha256_bytes(path.read_bytes())
    assert mod._load_object(path) == {"value": 1}
    path.write_text("not-json", encoding="utf-8")
    assert mod._load_object(path) == {}
    path.write_text("[]", encoding="utf-8")
    assert mod._load_object(path) == {}
    assert mod._load_object(tmp_path / "missing.json") == {}

    receipts = [
        {"name": name, "passed": True, "exit_code": 0}
        for name in (
            "independent_reducer",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        )
    ]
    assert mod._terminal_receipts_pass(receipts) is True
    receipts[0]["passed"] = False
    assert mod._terminal_receipts_pass(receipts) is False
    assert mod._date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(Exception, match="date must be"):
        mod._date_argument("20260916")


def test_scenario_report_7359_budget_diagnoses_historical_writer_reducer_gap() -> None:
    """SCENARIO-REPORT-7359-BUDGET records the exact Exp7348 mismatch."""

    original = json.loads(
        (ROOT / "results/experiment_7348_v645_plan_capture.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (ROOT / "results/raw/experiment_7348_v645_plan_capture/terminal_candidate.json").read_text(
            encoding="utf-8"
        )
    )
    diagnosis = mod.diagnose_historical_mismatches(original, candidate)

    budget = diagnosis["sample_size_budget_mismatch"]
    assert budget["expected_value"] == {
        "planned_units": 128,
        "attempted_units": 128,
        "completed_units": 128,
        "failed_units": 0,
        "cancelled_units": 0,
        "censored_units": 0,
    }
    assert budget["observed_value"] == original["sample_size_budget"]
    assert set(budget["observed_only_fields"]) == {
        "generation_timeout_s",
        "max_generated_tokens_per_unit",
        "model_load_timeout_s",
        "stopping_rule",
    }
    assert budget["writer"] == "experiment_7348_v645_plan_capture.run_experiment"
    assert budget["reducer"] == "experiment_7348_v645_plan_capture.independent_reduce"

    score = diagnosis["plan_capture_complete_score_mismatch"]
    assert score["expected_value"] == 1
    assert score["observed_value"] == 0
    assert score["candidate_observed_value"] == 1
    assert score["cause"] == "final readiness classification overwrote accounting completion"


def test_req_report_7359_precondition_gate_rejects_ineligible_producer() -> None:
    """REQ-REPORT-7359 rejects missing, partial, flagged, or unready Exp7358."""

    producer = json.loads(
        (ROOT / "results/experiment_7358_v646_validation_contract.json").read_text(encoding="utf-8")
    )
    assert all(row["passed"] for row in mod.validation_contract_gate_rows(producer))

    mutations = (
        ("status", "partial"),
        ("verdict_class", "partial"),
        ("flagged_adversarial", True),
        ("validation_contract_ready_score", 0),
    )
    for field, value in mutations:
        changed = deepcopy(producer)
        changed[field] = value
        checks = mod.validation_contract_gate_rows(changed)
        assert any(row["artifact_field"] == field and not row["passed"] for row in checks)

    changed = deepcopy(producer)
    changed["honest_verdict"] = "complete_quarantined_contract"
    checks = mod.validation_contract_gate_rows(changed)
    assert any(row["artifact_field"] == "quarantined" and not row["passed"] for row in checks)


def _real_reduction() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Cold-load the real archive for artifact validation without model work."""

    original = json.loads((ROOT / mod.HISTORICAL_RESULT_PATH).read_text(encoding="utf-8"))
    candidate = json.loads(
        (ROOT / mod.HISTORICAL_RAW_DIR / "terminal_candidate.json").read_text(encoding="utf-8")
    )
    schedule_manifest = json.loads((ROOT / mod.SCHEDULE_MANIFEST_PATH).read_text())
    candidate_manifest = json.loads((ROOT / mod.CANDIDATE_MANIFEST_PATH).read_text())
    evaluation = json.loads((ROOT / mod.EVALUATION_PATH).read_text())
    schedule = schedule_manifest["schedule"]
    calls = candidate_manifest["calls"]
    blobs = {
        str(row["call_id"]): (ROOT / path).read_bytes()
        for row, path in zip(schedule, mod._candidate_paths(schedule), strict=True)
    }
    reduced = mod.reduce_capture(
        schedule,
        calls,
        blobs,
        expected_schedule_sha256=schedule_manifest["schedule_sha256"],
        evaluator_rows=evaluation["rows"],
        cost_rows=original["generation_cost_rows"],
    )
    independent = mod.independent_reduce_capture(
        schedule,
        calls,
        blobs,
        expected_schedule_sha256=schedule_manifest["schedule_sha256"],
    )
    assert reduced["errors"] == []
    assert independent["errors"] == []
    return original, candidate, schedule_manifest, candidate_manifest, reduced, independent


def _validation_receipts() -> list[dict[str, object]]:
    """Build the minimal truthful shape consumed by the independent validator."""

    receipts: list[dict[str, object]] = []
    for name in mod.validation_scope.REQUIRED_CHECK_NAMES:
        row: dict[str, object] = {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        if name == "worktree_imports":
            row["resolved_imports"] = {
                "carnot.experiment_7359_v646_capture_reducer": str(ROOT / mod.MODULE_PATH)
            }
        receipts.append(row)
    receipts.extend(
        {"name": name, "passed": True, "exit_code": 0}
        for name in (
            "independent_reducer",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        )
    )
    return receipts


def _synthetic_terminal_artifact() -> dict[str, object]:
    """Build a full terminal record from real raw bytes before publication."""

    original, candidate, schedule_manifest, candidate_manifest, reduced, independent = (
        _real_reduction()
    )
    artifact = mod._build_artifact(
        started_at="2026-09-17T00:00:00Z",
        duration_s=1.0,
        spans=[],
        preconditions=[{"passed": True}],
        hashes={"AGENTS.md": mod.sha256_file(ROOT / "AGENTS.md")},
        original=original,
        candidate=candidate,
        schedule_manifest=schedule_manifest,
        candidate_manifest=candidate_manifest,
        reduction=reduced,
        independent=independent,
        validation_receipts=_validation_receipts(),
        affected_passed=True,
        terminal_passed=True,
        stage="terminal",
    )
    return artifact


def test_req_report_7359_validator_fails_closed_on_artifact_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7359 validates scores, sources, quarantine, and blocked shape."""

    assert mod.validate_artifact([]) == ["artifact_not_object"]
    artifact = _synthetic_terminal_artifact()
    assert mod.validate_artifact(artifact, repo_root=ROOT) == []

    changed = deepcopy(artifact)
    changed.update(
        {
            "schema": "wrong",
            "MODEL_SPECS": ["model"],
            "model_invoked": True,
            "invocation_counts": {},
            "inference_substrate_class": "wrong",
            "execution_venue": "board",
            "verdict_class": "wrong",
            "field_principles": {},
            "capture_reducer_ready_score": 0,
            "scientific_value_score": 1,
            "promotion_score": 1,
            "historical_correction": {},
            "source_artifact_hashes": {"AGENTS.md": "sha256:wrong"},
            "reproducibility_checksum": "sha256:wrong",
        }
    )
    errors = mod.validate_artifact(changed, repo_root=ROOT)
    assert {
        "identity_mismatch",
        "current_model_declaration_mismatch",
        "current_invocation_counts_nonzero",
        "substrate_class_mismatch",
        "execution_venue_mismatch",
        "verdict_class_invalid",
        "field_principles_incomplete",
        "capture_reducer_ready_score_mismatch",
        "accounting_promoted_as_science",
        "historical_quarantine_not_preserved",
        "source_hash_mismatch:AGENTS.md",
        "reproducibility_checksum_mismatch",
    } <= set(errors)

    failed_reduction = deepcopy(artifact)
    failed_reduction["rows"] = []
    failed_reduction["reproducibility_checksum"] = mod._artifact_checksum(failed_reduction)
    monkeypatch.setattr(
        mod,
        "_reload_reduction",
        lambda _artifact, _root: (
            {"errors": ["changed"], "capture_complete_score": 0},
            {"errors": ["changed"], "capture_complete_score": 0},
        ),
    )
    errors = mod.validate_artifact(failed_reduction, repo_root=ROOT)
    assert "archived_reduction_failed" in errors
    assert "stored_reduction_mismatch" in errors

    blocked = mod._blocked_artifact(
        started_at="2026-09-17T00:00:00Z",
        duration_s=0.1,
        spans=[],
        preconditions=[
            {
                "check": "missing",
                "upstream": "missing.json",
                "artifact_field": "bytes",
                "expected": "present",
                "observed": None,
                "passed": False,
            }
        ],
        hashes={},
    )
    assert mod.validate_artifact(blocked, repo_root=ROOT) == []
    blocked["rows"] = [{}]
    blocked["validation_receipts"] = [{}]
    blocked["gate_check_summary"]["first_failure"] = None
    blocked["capture_reducer_ready_score"] = 1
    blocked["reproducibility_checksum"] = mod._artifact_checksum(blocked)
    assert {
        "blocked_artifact_has_dependent_work",
        "blocked_gate_summary_missing",
        "blocked_ready_score_nonzero",
    } <= set(mod.validate_artifact(blocked, repo_root=ROOT))


def test_scenario_report_7359_replay_validates_checked_in_artifact() -> None:
    """SCENARIO-REPORT-7359-REPLAY cold-validates the separate correction record."""

    path = ROOT / mod.RESULT_PATH
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stored_hashes = stored.get("source_artifact_hashes", {})
    artifact = (
        stored
        if stored_hashes.get(mod.MODULE_PATH.as_posix()) == mod.sha256_file(ROOT / mod.MODULE_PATH)
        else _synthetic_terminal_artifact()
    )
    assert mod.validate_artifact(artifact, repo_root=ROOT) == []
    assert artifact["capture_reducer_ready_score"] == 1
    assert artifact["scientific_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["historical_correction"]["original_verdict_class"] == "disqualified"
    assert artifact["historical_correction"]["original_flagged_adversarial"] is True
    assert artifact["reduced_budget"]["planned_units"] == 128
    assert len(artifact["rows"]) == 128
    assert len(artifact["identifier_twin_rows"]) == 64
