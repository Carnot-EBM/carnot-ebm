"""Tests for the fresh plan capture evidence contract.

Spec refs: REQ-REPORT-7361 and SCENARIO-REPORT-7361-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7361_v646_fresh_plan_capture as capture
from carnot.experiment_7330_v644_public_learner import canonical_bytes, sha256_json


JsonDict = dict[str, Any]


def _request(request_id: str, names: tuple[str, ...]) -> JsonDict:
    return {
        "request_id": request_id,
        "version_token": f"opaque-{request_id}",
        "activities": list(names),
        "allowed_starts": {name: [0, 1, 2] for name in names},
        "durations": {name: 1 for name in names},
        "weights": {name: 1 for name in names},
        "horizon": 4,
        "public_revision": 0,
    }


def _public_manifest() -> JsonDict:
    panel = []
    for index in range(32):
        original = _request(f"original-{index:02d}", ("a", "b"))
        twin = _request(f"twin-{index:02d}", ("unit-1", "unit-2"))
        panel.append(
            {
                "panel_id": f"panel-{index:02d}",
                "stream_id": f"stream-{index // 4:02d}",
                "cohort": "stable_rules",
                "warmup": index % 4 < 2,
                "presentation_order": "original_first" if index % 2 == 0 else "twin_first",
                "renaming_map": {"a": "unit-1", "b": "unit-2"},
                "original": original,
                "twin": twin,
            }
        )
    return {
        "schema": "carnot.exp7360.public_manifest.v1",
        "development_seed": capture.RANDOM_SEED["development"],
        "evaluation_seed": capture.RANDOM_SEED["evaluation"],
        "resampling_seed": capture.RANDOM_SEED["resampling"],
        "development_canary": [_request(f"canary-{index:02d}", ("a", "b")) for index in range(4)],
        "live_proposal_panel": panel,
        "sealed_before_outcomes": True,
    }


def _eligible_producers() -> tuple[JsonDict, JsonDict]:
    reducer = {
        "experiment_id": "exp7359-capture-reducer",
        "milestone": capture.MILESTONE,
        "run_date": capture.RUN_DATE,
        "status": "complete_capture_reducer_null_science",
        "capture_reducer_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "honest_verdict": "complete_null_accounting_reducer_ready_no_current_science",
    }
    fixture = {
        "experiment_id": "exp7360-learning-fixture",
        "milestone": capture.MILESTONE,
        "run_date": capture.RUN_DATE,
        "status": "complete_learning_fixture_ready_value_not_evaluated",
        "learning_fixture_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "honest_verdict": "complete_null_learning_fixture_ready_value_not_evaluated",
    }
    return reducer, fixture


def _response_call(schedule_row: JsonDict, *, raw_reply: str | None = None) -> JsonDict:
    request = schedule_row["public_request"]
    reply = raw_reply or json.dumps(
        {
            "request_id": request["request_id"],
            "assignments": {name: 0 for name in request["activities"]},
        }
    )
    parsed = capture.canary_mod.decode_public_plan(reply, request)
    return {
        **deepcopy(schedule_row),
        "raw_reply": reply,
        "raw_reply_sha256": capture.sha256_text(reply),
        "raw_response": {"model": "Qwen3.8-27B-Q4_K_M.gguf"},
        "parse_status": parsed["parse_status"],
        "parse_errors": parsed["parse_errors"],
        "decoded_plan": parsed["plan"],
        "attempted": True,
        "terminal_state": "response",
        "error": None,
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "latency_s": 0.5,
        "finish_reason": "stop",
        "runtime_identity_receipt": {"lease_id": "lease:test"},
        "censored": False,
    }


def test_scenario_report_7361_gates_reject_ineligible_producer() -> None:
    """SCENARIO-REPORT-7361-GATES keeps exact producer failures visible."""

    reducer, fixture = _eligible_producers()
    assert all(row["passed"] for row in capture.dependency_gate_rows(reducer, fixture))

    changed = deepcopy(reducer)
    changed["capture_reducer_ready_score"] = 0
    rows = capture.dependency_gate_rows(changed, fixture)
    failure = next(row for row in rows if row["artifact_field"] == "capture_reducer_ready_score")
    assert failure["expected_value"] == 1
    assert failure["observed_value"] == 0
    assert failure["passed"] is False

    quarantined = deepcopy(fixture)
    quarantined["honest_verdict"] = "complete_quarantined"
    assert not all(row["passed"] for row in capture.dependency_gate_rows(reducer, quarantined))

    passing_summary = capture.gate_check_summary(capture.dependency_gate_rows(reducer, fixture))
    assert passing_summary["passed"] is True
    failed_summary = capture.gate_check_summary(rows)
    assert failed_summary["failed_check_count"] == 1
    assert failed_summary["artifact_field"] == "capture_reducer_ready_score"


def test_scenario_report_7361_schedule_freezes_four_plus_128_calls() -> None:
    """SCENARIO-REPORT-7361-SCHEDULE fixes every identity before outcomes."""

    manifest = _public_manifest()
    canary = capture.build_canary_schedule(manifest)
    evaluation = capture.build_evaluation_schedule(manifest)

    assert len(canary) == 4
    assert len(evaluation) == 128
    assert {row["max_generated_tokens"] for row in [*canary, *evaluation]} == {256}
    assert len({row["call_id"] for row in evaluation}) == 128
    assert set(row["candidate_index"] for row in evaluation) == {0, 1}
    assert [row["pair_side"] for row in evaluation[:4]] == [
        "original",
        "twin",
        "original",
        "twin",
    ]
    assert [row["pair_side"] for row in evaluation[4:8]] == [
        "twin",
        "original",
        "twin",
        "original",
    ]
    assert capture.schedule_errors(canary, evaluation, manifest) == []

    changed = deepcopy(evaluation)
    changed[0]["max_generated_tokens"] = 257
    assert "evaluation_schedule_rebuild_mismatch" in capture.schedule_errors(
        canary, changed, manifest
    )

    changed_canary = deepcopy(canary)
    changed_canary[0]["call_id"] = changed_canary[1]["call_id"]
    errors = capture.schedule_errors(changed_canary, evaluation, manifest)
    assert "canary_schedule_rebuild_mismatch" in errors
    assert "call_identity" in errors

    bad_token = deepcopy(evaluation)
    bad_token[0]["max_generated_tokens"] = 257
    assert "token_budget" in capture.schedule_errors(canary, bad_token, manifest)
    assert capture.schedule_errors(canary, evaluation, {}) == [
        "schedule_source_invalid:ValueError:development_canary_count"
    ]


def test_scenario_report_7361_schedule_rejects_bad_panel_shapes() -> None:
    """SCENARIO-REPORT-7361-SCHEDULE rejects malformed sealed inputs."""

    manifest = _public_manifest()
    with pytest.raises(ValueError, match="development_canary_count"):
        capture.build_canary_schedule({"development_canary": []})

    bad = deepcopy(manifest)
    bad["development_canary"][0] = "not-an-object"
    with pytest.raises(ValueError, match="development_canary_request"):
        capture.build_canary_schedule(bad)

    bad = deepcopy(manifest)
    bad["development_canary"][1]["request_id"] = bad["development_canary"][0]["request_id"]
    with pytest.raises(ValueError, match="development_canary_identity"):
        capture.build_canary_schedule(bad)

    with pytest.raises(ValueError, match="live_proposal_panel_count"):
        capture.build_evaluation_schedule({"live_proposal_panel": []})

    bad = deepcopy(manifest)
    bad["live_proposal_panel"][0] = "not-an-object"
    with pytest.raises(ValueError, match="live_proposal_pair"):
        capture.build_evaluation_schedule(bad)

    bad = deepcopy(manifest)
    bad["live_proposal_panel"][0].pop("original")
    with pytest.raises(ValueError, match="public_request"):
        capture.build_evaluation_schedule(bad)


def test_scenario_report_7361_fidelity_keeps_dimensions_separate() -> None:
    """SCENARIO-REPORT-7361-FIDELITY does not equate parsing with meaning."""

    request = _request("request-1", ("a", "b"))
    raw = '{"request_id":"request-1","assignments":{"b":1,"a":0}}'
    row = {"public_request": request, "raw_reply": raw, "parse_status": "valid"}
    evidence = capture.source_fidelity(row)
    assert evidence["parser_valid"] is True
    assert evidence["schema_valid"] is True
    assert evidence["entity_fidelity"] is True
    assert evidence["quantity_fidelity"] is True
    assert evidence["ordering_fidelity"] is False
    assert evidence["public_semantic_correct"] is False

    out_of_domain = '{"request_id":"request-1","assignments":{"a":9,"b":0}}'
    evidence = capture.source_fidelity(
        {"public_request": request, "raw_reply": out_of_domain, "parse_status": "invalid"}
    )
    assert evidence["entity_fidelity"] is True
    assert evidence["quantity_fidelity"] is False
    assert evidence["ordering_fidelity"] is True
    assert evidence["schema_valid"] is False


def test_scenario_report_7361_reduction_is_schedule_derived() -> None:
    """SCENARIO-REPORT-7361-REDUCTION separates completion from usefulness."""

    schedule = capture.build_evaluation_schedule(_public_manifest())
    calls = [_response_call(row) for row in schedule]
    calls[0] = _response_call(schedule[0], raw_reply="not-json")
    candidate_bytes = {str(row["call_id"]): canonical_bytes(row) + b"\n" for row in calls}
    reduced = capture.reduce_evaluation(
        schedule,
        calls,
        candidate_bytes,
        expected_schedule_sha256=sha256_json(schedule),
        evaluator_rows=[],
    )
    assert reduced["plan_capture_complete_score"] == 1
    assert reduced["usable_proposal_count"] == 127
    assert reduced["primary_independent_agreement"] is True
    assert reduced["reducer_errors"] == []

    censored = deepcopy(calls)
    censored[-1].update(
        {
            "attempted": False,
            "terminal_state": "cancelled",
            "censored": True,
            "raw_reply": "",
            "raw_reply_sha256": capture.sha256_text(""),
            "parse_status": "invalid",
            "decoded_plan": None,
        }
    )
    candidate_bytes[str(censored[-1]["call_id"])] = canonical_bytes(censored[-1]) + b"\n"
    reduced = capture.reduce_evaluation(
        schedule,
        censored,
        candidate_bytes,
        expected_schedule_sha256=sha256_json(schedule),
        evaluator_rows=[],
    )
    assert reduced["plan_capture_complete_score"] == 0
    assert reduced["sample_size_budget"]["cancelled_units"] == 1


def test_scenario_report_7361_terminal_validation_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7361-TERMINAL cold-checks raw evidence and checksums."""

    artifact = capture.base_artifact(capture.RUN_DATE, "2026-09-17T00:00:00Z", tmp_path)
    assert capture.validate_artifact(artifact, allow_preterminal=True) == []

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = [{"hf_id": "legacy/substitute"}]
    assert "model_identity_invalid" in capture.validate_artifact(changed, allow_preterminal=True)

    changed = deepcopy(artifact)
    changed["field_principles"].pop("call_manifest")
    assert "field_principles_mismatch" in capture.validate_artifact(changed, allow_preterminal=True)

    assert capture.validate_artifact([]) == ["artifact_not_object"]
    assert capture.validate_artifact({})[0].startswith("missing_required_field:")

    changed = deepcopy(artifact)
    changed["schema"] = "wrong"
    changed["milestone"] = "wrong"
    changed["verdict_class"] = "wrong"
    changed["execution_venue"] = "board"
    changed["reproducibility_checksum"] = "wrong"
    errors = capture.validate_artifact(changed, allow_preterminal=True)
    assert {
        "identity_invalid",
        "lifecycle_invalid",
        "verdict_class_invalid",
        "execution_venue_invalid",
        "reproducibility_checksum_mismatch",
    }.issubset(errors)

    changed = deepcopy(artifact)
    changed.update(
        {
            "status": "complete_fresh_plan_capture_disqualified",
            "model_invoked": True,
            "inference_substrate_class": "wrong",
            "verdict_class": "disqualified",
            "plan_capture_complete_score": 1,
        }
    )
    changed["reproducibility_checksum"] = capture.artifact_checksum(changed)
    monkeypatch.setattr(capture, "independent_reduce_artifact", lambda *_args, **_kwargs: ["cold"])
    errors = capture.validate_artifact(changed)
    assert "cold" in errors
    assert "substrate_class_invalid" in errors
    assert "failed_scores_invalid" in errors


def test_small_evidence_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-7361 covers exact hashes and receipt cardinality."""

    path = tmp_path / "bytes.txt"
    path.write_bytes(b"evidence")
    assert capture.sha256_file(path).startswith("sha256:")
    receipt = {"name": "one", "passed": True, "exit_code": 0}
    assert capture._receipts_pass([receipt], ["one"]) is True
    assert capture._receipts_pass([receipt, receipt], ["one"]) is False


def test_date_argument_and_thin_main_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7361 exposes a bounded validation-only CLI path."""

    with pytest.raises(Exception, match="date must be"):
        capture._date_argument("20260918")
    artifact = capture.base_artifact(capture.RUN_DATE, "2026-09-17T00:00:00Z", tmp_path)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert capture.main(["--date", capture.RUN_DATE, "--validate", str(path)]) == 0
    assert '"errors":[]' in capsys.readouterr().out
