"""Tests for the unchanged prospective proposal capture panel.

Spec refs: REQ-REPORT-7402 and SCENARIO-REPORT-7402-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7402_v649_proposal_capture as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


def _identity() -> dict[str, object]:
    """Represent one owned CUDA server without making a real model call."""

    return {
        "pid": 7402,
        "start_time_ticks": 649,
        "owned_by_task": True,
        "owned_by_current_uid": True,
        "command": ["llama-server", "--n-gpu-layers", "all"],
        "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256": "sha256:" + "a" * 64,
        "server_executable_sha256": "sha256:" + "b" * 64,
        "served_model": "Qwen3.8-27B-Q4_K_M.gguf",
        "gpu_uuid": "GPU-test",
        "gpu_index": 0,
        "lease_id": "lease-test",
        "cuda_provenance_ok": True,
        "task_owned_vram_mb": 16340,
        "quantization": "Q4_K_M",
    }


def _manifest() -> dict[str, object]:
    """Load the exact frozen protocol used by the capture."""

    return json.loads(mod.PROTOCOL_PATH.read_text(encoding="utf-8"))


def _response(raw_reply: str, *, error: str | None = None) -> dict[str, object]:
    """Build one native response shape for the production row builder."""

    return {
        "raw_reply": raw_reply,
        "raw_response": {
            "model": "Qwen3.8-27B-Q4_K_M.gguf",
            "choices": [{"message": {"content": raw_reply}}],
        },
        "prompt_tokens": 90,
        "completion_tokens": 8,
        "latency_s": 0.25,
        "finish_reason": "stop" if error is None else None,
        "error": error,
        "attempted": True,
        "terminal_state": "response" if error is None else "request_error",
    }


def _candidate_rows(replies: list[str] | None = None) -> list[dict[str, object]]:
    """Build all 64 candidate rows through the exact parser and solver."""

    schedule = mod.build_capture_schedule(_manifest())
    values = replies or ['{"assignments":[1,2]}'] * len(schedule)
    return [
        mod.build_candidate_row(item, _response(reply), _identity())
        for item, reply in zip(schedule, values, strict=True)
    ]


def _events(run_id: str = "run-test", owner_pid: int = 7402) -> list[dict[str, object]]:
    """Build one completed load and 64 completed generation event pairs."""

    rows: list[dict[str, object]] = []
    timestamp = 100
    calls = [("model-load", "model_load")]
    calls.extend((f"generation-{index}", "generation") for index in range(64))
    for call_id, operation in calls:
        for state in ("attempted", "completed"):
            rows.append(
                {
                    "scope": "current",
                    "transport": "owned_runtime",
                    "run_id": run_id,
                    "owner_pid": owner_pid,
                    "call_id": call_id,
                    "operation": operation,
                    "state": state,
                    "monotonic_ns": timestamp,
                }
            )
            timestamp += 1
    return rows


def _receipts() -> list[dict[str, object]]:
    """Return one passing receipt for each affected and terminal check."""

    return [
        {
            "name": name,
            "command_argv": [name],
            "command_environment": {},
            "exit_code": 0,
            "duration_s": 0.1,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "c" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (*REQUIRED_CHECK_NAMES, *mod.TERMINAL_CHECK_NAMES)
    ]


def _artifact(tmp_path: Path) -> dict[str, object]:
    """Build one complete fixture with exact raw candidate sidecars."""

    reduced = mod.reduce_panel(_candidate_rows())
    raw_rows: list[dict[str, object]] = []
    for index, row in enumerate(reduced["candidate_rows"]):
        path = tmp_path / f"call_{index:02d}.json"
        mod.atomic_json(path, row)
        raw_rows.append({"path": str(path), "sha256": mod.sha256_file(path)})
    artifact = mod.build_artifact_for_test(
        reduced=reduced,
        raw_rows=raw_rows,
        events=_events(),
        receipts=_receipts(),
    )
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def test_req_report_7402_authenticates_canary_and_frozen_protocol() -> None:
    """REQ-REPORT-7402 requires eligible canary bytes and the exact panel."""

    checks, context = mod.collect_preconditions(mod.REPO_ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert context["producer"]["qwen_assignment_transport_ready_score"] == 1
    assert context["producer_sha256"] == mod.sha256_file(mod.REPO_ROOT / mod.PRODUCER_PATH)
    assert context["protocol_sha256"] == mod.sha256_file(mod.REPO_ROOT / mod.PROTOCOL_PATH)
    assert len(context["schedule"]) == 64
    assert mod.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]

    producer = deepcopy(context["producer"])
    producer["flagged_adversarial"] = True
    failed = mod.producer_gate_rows(producer, root=mod.REPO_ROOT)
    assert [row["artifact_field"] for row in failed if not row["passed"]] == [
        "flagged_adversarial",
        "validate_artifact.errors",
    ]


def test_scenario_report_7402_gate_retains_missing_and_malformed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7402-GATE reports absent bytes and schedule errors."""

    assert mod.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("[", encoding="utf-8")
    assert mod.load_object(malformed) == {}

    def reject(_manifest: object) -> list[dict[str, object]]:
        raise ValueError("panel_drift")

    monkeypatch.setattr(mod, "build_capture_schedule", reject)
    checks, context = mod.collect_preconditions(mod.REPO_ROOT)
    panel = next(row for row in checks if row["check"] == "frozen_proposal_panel")
    assert panel["passed"] is False
    assert panel["observed_value"]["selection_error"] == "ValueError:panel_drift"
    assert context["schedule"] == []

    missing = mod.gate_row("missing", "upstream", "field", "==", 1, None)
    assert missing["passed"] is False
    assert mod.gate_check_summary([missing])["observed_value"] is None
    assert mod.compare(">=", 3, 3) is True
    with pytest.raises(ValueError, match="unsupported_operator"):
        mod.compare("!=", 1, 2)
    assert mod.gate_row("bad", "upstream", "field", "!=", 1, 2)["passed"] is False


def test_scenario_report_7402_panel_preserves_all_registered_questions() -> None:
    """SCENARIO-REPORT-7402-PANEL freezes order, prompts, splits, and seeds."""

    manifest = _manifest()
    schedule = mod.build_capture_schedule(manifest)
    assert len(schedule) == 64
    assert len({row["call_id"] for row in schedule}) == 64
    assert len({row["request_id"] for row in schedule}) == 32
    assert [row["call_index"] for row in schedule] == list(range(64))
    assert {row["candidate_index"] for row in schedule} == {0, 1}
    assert {row["split"] for row in schedule} == {"warm_up", "future", "version_change"}
    assert all(row["max_new_tokens"] == 256 for row in schedule)
    assert all(
        row["inference_seed"] == manifest["random_seed"]["live_protocol"] for row in schedule
    )
    assert schedule[0]["prompt"] == manifest["live_proposal_streams"][0]["requests"][0]["prompt"]

    changed = deepcopy(manifest)
    changed["live_proposal_streams"][0]["requests"][0]["proposal_count"] = 3
    with pytest.raises(ValueError, match="proposal_count"):
        mod.build_capture_schedule(changed)
    with pytest.raises(ValueError, match="stream_count"):
        mod.build_capture_schedule({"live_proposal_streams": []})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["random_seed"].pop("live_protocol"), "live_protocol_seed"),
        (lambda value: value["live_proposal_streams"].__setitem__(0, None), "stream_shape"),
        (
            lambda value: value["live_proposal_streams"][0].__setitem__("requests", []),
            "request_count",
        ),
        (
            lambda value: value["live_proposal_streams"][0].__setitem__("versions", None),
            "version_shape",
        ),
        (
            lambda value: value["live_proposal_streams"][0]["requests"].__setitem__(0, None),
            "request_shape",
        ),
        (
            lambda value: value["live_proposal_streams"][0]["requests"][0].__setitem__(
                "formula_source_hash", "sha256:wrong"
            ),
            "formula_identity",
        ),
        (
            lambda value: value["live_proposal_streams"][1]["requests"][0].__setitem__(
                "request_id", value["live_proposal_streams"][0]["requests"][0]["request_id"]
            ),
            "call_identity",
        ),
        (
            lambda value: value["live_proposal_streams"][0]["requests"][0].__setitem__(
                "split", "future"
            ),
            "request_schedule",
        ),
    ],
)
def test_scenario_report_7402_panel_rejects_each_shape_drift(
    mutation: object, message: str
) -> None:
    """SCENARIO-REPORT-7402-PANEL fails each closed schedule boundary."""

    manifest = _manifest()
    mutation(manifest)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        mod.build_capture_schedule(manifest)


def test_scenario_report_7402_selection_is_syntax_first_and_semantics_separate() -> None:
    """SCENARIO-REPORT-7402-SELECTION never selects by favorable SAT outcome."""

    replies = ['{"assignments":[1,1]}', '{"assignments":[1,2]}'] + ['{"assignments":[1,2]}'] * 62
    reduced = mod.reduce_panel(_candidate_rows(replies))
    first = reduced["request_dispositions"][0]
    assert first["selected_candidate_index"] == 0
    assert first["selection_status"] == "selected_first_schema_valid"
    assert first["selected_source_literal_fidelity"] is False
    assert reduced["selected_request_count"] == 32
    assert reduced["source_faithful_selected_request_count"] == 31
    assert reduced["candidate_capture_observed_score"] == 0
    assert len(reduced["candidate_rows"]) == 64
    assert sum(row["selected_for_request"] for row in reduced["candidate_rows"]) == 32

    no_parse = ["not-json", "still-not-json"] + ['{"assignments":[1,2]}'] * 62
    low = mod.reduce_panel(_candidate_rows(no_parse))
    assert low["request_dispositions"][0]["selection_status"] == "no_schema_valid_candidate"
    assert low["request_dispositions"][0]["selected_candidate_index"] is None
    assert low["selected_request_count"] == 31
    assert low["candidate_capture_observed_score"] == 0


def test_scenario_report_7402_candidate_rows_keep_failures_and_exact_bytes() -> None:
    """SCENARIO-REPORT-7402-SELECTION retains raw bytes and terminal failures."""

    schedule = mod.build_capture_schedule(_manifest())[0]
    row = mod.build_candidate_row(schedule, _response('{"assignments":[1,2]}'), _identity())
    assert row["raw_reply"] == '{"assignments":[1,2]}'
    assert row["raw_reply_sha256"].startswith("sha256:")
    assert row["raw_request_bytes_b64"]
    assert row["raw_response_bytes_b64"]
    assert row["schema_valid"] is True
    assert row["source_literal_fidelity"] is True
    assert isinstance(row["semantic_extendible"], bool)
    assert row["exact_solver_duration_ns"] >= 0

    failed = mod.build_candidate_row(schedule, _response("", error="timeout"), _identity())
    assert failed["attempted"] is True
    assert failed["terminal_state"] == "request_error"
    assert failed["censored"] is True
    assert failed["schema_valid"] is False
    assert failed["semantic_extendible"] is None


def test_scenario_report_7402_complexity_is_descriptive_and_unfiltered() -> None:
    """SCENARIO-REPORT-7402-COMPLEXITY emits one stratum row per candidate."""

    reduced = mod.reduce_panel(_candidate_rows())
    complexity = reduced["complexity_rows"]
    assert len(complexity) == 64
    assert {row["family"] for row in complexity} == {
        stream["family"] for stream in _manifest()["live_proposal_streams"]
    }
    assert {row["query_type"] for row in complexity} == {
        "warm_up",
        "future",
        "version_change",
    }
    assert all(row["original_variable_count"] in {8, 12, 24, 32} for row in complexity)
    assert all(row["original_clause_count"] > 0 for row in complexity)
    assert all(row["solver_effort_is_model_hardness"] is False for row in complexity)
    assert sum(row["completion_tokens"] for row in complexity) == 64 * 8


def test_scenario_report_7402_validation_plan_is_exact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7402-VALIDATION freezes the eight affected checks."""

    commands = mod.build_validation_plan(mod.REPO_ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(REQUIRED_CHECK_NAMES)
    assert mod.validate_validation_plan(mod.REPO_ROOT, commands) == []
    assert "full_python_suite" not in {row.name for row in commands}
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "-n" in focused.argv
    assert "--no-cov" in focused.argv
    coverage_report = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert dict(coverage_report.command_environment)["COVERAGE_FILE"].startswith(str(tmp_path))
    assert (tmp_path / "private/basetemp").is_dir()


def test_scenario_report_7402_gate_treats_device_count_as_a_minimum() -> None:
    """SCENARIO-REPORT-7402-GATE accepts two free GPUs for a one-slot minimum."""

    rows = mod.rtx3090_slot_gate_rows(query_ok=True, available_slot_count=2)
    assert all(row["passed"] for row in rows)
    assert rows[1]["operator"] == ">="
    assert rows[1]["expected_value"] == 1
    assert rows[1]["observed_value"] == 2
    assert not all(
        row["passed"] for row in mod.rtx3090_slot_gate_rows(query_ok=False, available_slot_count=0)
    )


def test_scenario_report_7402_artifact_reloads_raw_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7402-VALIDATION rejects raw, reduction, and count drift."""

    artifact = _artifact(tmp_path)
    assert mod.independent_reduce_artifact(artifact) == []
    assert mod.validate_artifact(artifact, require_terminal=True) == []

    changed = deepcopy(artifact)
    changed["selected_request_count"] = 0
    assert "selected_request_count_mismatch" in mod.independent_reduce_artifact(changed)

    changed = deepcopy(artifact)
    changed["invocation_counts"]["generation_calls_completed"] = 63
    assert "invocation_counts_mismatch" in mod.independent_reduce_artifact(changed)

    changed = deepcopy(artifact)
    changed["candidate_capture_complete_score"] = 0
    assert "candidate_capture_complete_score_mismatch" in mod.independent_reduce_artifact(changed)

    invalid = deepcopy(artifact)
    invalid["raw_call_rows"][0] = "bad"
    assert "raw_call_manifest_invalid:0" in mod.independent_reduce_artifact(invalid)

    missing = deepcopy(artifact)
    missing["raw_call_rows"][0]["path"] = str(tmp_path / "missing.json")
    assert "raw_call_path_missing:0" in mod.independent_reduce_artifact(missing)

    drift = deepcopy(artifact)
    drift["candidate_rows"][0]["decoded_assignments"] = [99]
    assert "raw_call_row_mismatch:0" in mod.independent_reduce_artifact(drift)

    bad_events = deepcopy(artifact)
    bad_events["current_invocation_events"] = "bad"
    assert mod.independent_reduce_artifact(bad_events) == ["current_invocation_events_invalid"]

    bad_owner = deepcopy(artifact)
    bad_owner["current_owner_pid"] = None
    assert mod.independent_reduce_artifact(bad_owner)[0].startswith(
        "current_event_reduction_failed:"
    )

    Path(artifact["raw_call_rows"][0]["path"]).write_text("{}", encoding="utf-8")
    assert "raw_call_hash_mismatch:0" in mod.independent_reduce_artifact(artifact)
    assert mod.independent_reduce_artifact({}) == ["raw_evidence_unavailable"]


def test_req_report_7402_blocked_and_closed_artifact_contract(tmp_path: Path) -> None:
    """REQ-REPORT-7402 keeps dates, receipts, scores, and blocked fields closed."""

    artifact = _artifact(tmp_path)
    assert mod.artifact_checksum(artifact) == artifact["reproducibility_checksum"]
    assert mod._date_argument("20260918") == "20260918"
    with pytest.raises(ValueError, match="date must be 20260918"):
        mod._date_argument("20260919")
    assert mod.receipts_pass(_receipts(), (*REQUIRED_CHECK_NAMES, *mod.TERMINAL_CHECK_NAMES))
    assert not mod.receipts_pass(_receipts()[:-1], mod.TERMINAL_CHECK_NAMES)
    assert all(value == 0 for value in mod.zero_counts().values())

    blocked = mod.build_blocked_artifact(
        [mod.gate_row("cache", "local_cache", "MODEL_SPECS", "==", "present", None)]
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_cache"
    assert blocked["gate_check_summary"]["observed_value"] is None
    assert blocked["model_invoked"] is False
    assert blocked["candidate_capture_complete_score"] == 0
    assert blocked["promotion_score"] == 0


def test_req_report_7402_validator_rejects_closed_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-7402 rejects identity, substrate, receipt, score, and checksum drift."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact([]) == ["artifact_not_object"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")

    cases = {
        "identity_invalid": {"schema": "wrong"},
        "model_contract_invalid": {"MODEL_SPECS": []},
        "execution_venue_invalid": {"execution_venue": "host_cpu"},
        "verdict_class_invalid": {"verdict_class": "success"},
        "substrate_class_invalid": {"inference_substrate_class": "model_full_generation"},
        "bounded_generation_duration_floor": {"duration_s": 9.9},
        "required_validation_receipts_invalid": {"validation_receipts": []},
        "positive_readiness_invalid": {"candidate_capture_complete_score": 0},
        "sample_budget_accounting_invalid": {
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "unstarted": 1,
            }
        },
    }
    for expected, updates in cases.items():
        changed = deepcopy(artifact)
        changed.update(updates)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    no_model = deepcopy(artifact)
    no_model["model_invoked"] = False
    no_model["verdict_class"] = "null"
    no_model["reproducibility_checksum"] = mod.artifact_checksum(no_model)
    assert "completed_artifact_without_model_attempt" in mod.validate_artifact(no_model)

    blocked = mod.build_blocked_artifact(
        [mod.gate_row("cache", "local_cache", "MODEL_SPECS", "==", "present", None)]
    )
    blocked["inference_substrate_class"] = "aggregation"
    blocked["candidate_capture_complete_score"] = 1
    blocked["reproducibility_checksum"] = "wrong"
    errors = mod.validate_artifact(blocked, require_terminal=False)
    assert "blocked_substrate_invalid" in errors
    assert "failed_readiness_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors
