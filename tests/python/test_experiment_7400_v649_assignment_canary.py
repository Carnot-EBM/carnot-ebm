"""Tests for the fresh bounded assignment canary.

Spec refs: REQ-REPORT-7400 and SCENARIO-REPORT-7400-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7372_v647_qwen_canary as prior_canary
from carnot import experiment_7400_v649_assignment_canary as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


def _identity() -> dict[str, object]:
    """Represent one task-owned native CUDA server for reducer tests."""

    return {
        "pid": 7400,
        "start_time_ticks": 649,
        "owned_by_task": True,
        "owned_by_current_uid": True,
        "command": ["llama-server", "--n-gpu-layers", "all"],
        "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256": "sha256:" + "a" * 64,
        "server_executable_sha256": "sha256:" + "b" * 64,
        "served_model": "Qwen3.8-27B-Q4_K_M.gguf",
        "gpu_uuid": "GPU-test",
        "gpu_index": 1,
        "lease_id": "lease-test",
        "cuda_provenance_ok": True,
        "task_owned_vram_mb": 16340,
        "quantization": "Q4_K_M",
    }


def _fixture() -> dict[str, object]:
    """Load the immutable development formulas used by the real canary."""

    return json.loads(mod.DEVELOPMENT_FORMULAS_PATH.read_text(encoding="utf-8"))


def _calls(replies: list[str] | None = None) -> list[dict[str, object]]:
    """Build four source-bound call rows through the shipped parser."""

    values = replies or [
        '{"assignments":[1,2]}',
        '{"assignments":[1,-2]}',
        '{"assignments":[1,3]}',
        '{"assignments":[-1,4]}',
    ]
    rows: list[dict[str, object]] = []
    for schedule, raw_reply in zip(mod.build_schedule(_fixture()), values, strict=True):
        rows.append(
            prior_canary.build_call_row(
                schedule,
                {
                    "raw_reply": raw_reply,
                    "raw_response": {
                        "model": "Qwen3.8-27B-Q4_K_M.gguf",
                        "choices": [{"message": {"content": raw_reply}}],
                    },
                    "prompt_tokens": 200,
                    "completion_tokens": 10,
                    "latency_s": 0.5,
                    "finish_reason": "stop",
                    "error": None,
                },
                _identity(),
            )
        )
    return rows


def _events(run_id: str = "run-test", owner_pid: int = 7400) -> list[dict[str, object]]:
    """Build one completed load and four completed generation event pairs."""

    rows: list[dict[str, object]] = []
    timestamp = 100
    for call_id, operation in [
        ("model-load", "model_load"),
        *((f"generation-{index}", "generation") for index in range(4)),
    ]:
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
    """Return one passing receipt per affected and terminal command."""

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
    """Build a complete deterministic artifact and its exact raw sidecars."""

    rows = _calls()
    raw_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        path = tmp_path / f"call_{index:02d}.json"
        mod.atomic_json(path, row)
        raw_rows.append({"path": str(path), "sha256": mod.sha256_file(path)})
    reduced = mod.reduce_transport(rows, _fixture()["development_formulas"])
    artifact = mod.build_artifact_for_test(
        rows=rows,
        raw_rows=raw_rows,
        events=_events(),
        receipts=_receipts(),
        reduced=reduced,
    )
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def test_req_report_7400_authenticates_exp7395_and_exact_code_hashes() -> None:
    """REQ-REPORT-7400 accepts only the eligible hash-bound reducer receipt."""

    checks, context = mod.collect_preconditions(mod.REPO_ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert context["producer"]["assignment_reducer_ready_score"] == 1
    assert len(context["schedule"]) == 4
    assert mod.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]

    producer = deepcopy(context["producer"])
    producer["flagged_adversarial"] = True
    rejected = mod.producer_gate_rows(producer, root=mod.REPO_ROOT)
    failed = [row for row in rejected if not row["passed"]]
    assert [row["artifact_field"] for row in failed] == ["flagged_adversarial"]
    summary = mod.gate_check_summary(rejected)
    assert summary["upstream"] == mod.PRODUCER_PATH.as_posix()
    assert summary["observed_value"] is True


def test_scenario_report_7400_gate_reports_missing_and_hash_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7400-GATE keeps missing and changed inputs explicit."""

    assert mod.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "bad.json"
    malformed.write_text("[", encoding="utf-8")
    assert mod.load_object(malformed) == {}

    producer = json.loads(mod.PRODUCER_PATH.read_text(encoding="utf-8"))
    producer["source_artifact_hashes"][mod.QUALIFIED_CODE_PATHS[0].as_posix()] = (
        "sha256:" + "0" * 64
    )
    failed = mod.producer_gate_rows(producer, root=mod.REPO_ROOT)
    drift = next(row for row in failed if row["check"].startswith("qualified_code_hash:"))
    assert drift["passed"] is False
    assert drift["operator"] == "=="

    missing = mod.gate_row("missing", "upstream.json", "field", "==", 1, None)
    assert missing["passed"] is False
    assert mod.compare(">=", 3, 3) is True
    with pytest.raises(ValueError, match="unsupported_operator"):
        mod.compare("!=", 1, 2)
    assert mod.gate_row("bad", "upstream", "field", "!=", 1, 2)["passed"] is False


def test_scenario_report_7400_gate_keeps_schedule_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7400-GATE preserves a malformed frozen schedule error."""

    def reject_schedule(_fixture: object) -> list[dict[str, object]]:
        raise ValueError("sealed_schedule_invalid")

    monkeypatch.setattr(mod, "build_schedule", reject_schedule)
    checks, context = mod.collect_preconditions(mod.REPO_ROOT)
    schedule = next(row for row in checks if row["check"] == "four_frozen_development_prompts")
    assert schedule["passed"] is False
    assert schedule["observed_value"]["selection_error"] == ("ValueError:sealed_schedule_invalid")
    assert context["schedule"] == []


def test_req_verify_7422_shared_capacity_gates_are_independent() -> None:
    """REQ-VERIFY-7422 checks query success and minimum capacity separately."""

    success = [{"returncode": 0}, {"returncode": 0}]
    failed = [{"returncode": 0}, {"returncode": 9}]

    two_free = mod.rtx3090_capacity_gate_rows(success, ["GPU-0", "GPU-1"])
    assert [row["check"] for row in two_free] == [
        "rtx3090_inventory_query_succeeded",
        "minimum_available_rtx3090_capacity",
    ]
    assert [row["operator"] for row in two_free] == ["==", ">="]
    assert [row["observed_value"] for row in two_free] == [True, 2]
    assert all(row["passed"] for row in two_free)

    no_free = mod.rtx3090_capacity_gate_rows(success, [])
    assert no_free[0]["passed"] is True
    assert no_free[1]["passed"] is False

    failed_query = mod.rtx3090_capacity_gate_rows(failed, ["GPU-0"])
    assert failed_query[0]["passed"] is False
    assert failed_query[1]["passed"] is True

    missing_query = mod.rtx3090_capacity_gate_rows([], ["GPU-0"])
    assert missing_query[0]["observed_value"] is False
    assert missing_query[0]["passed"] is False


def test_scenario_report_7400_transport_keeps_sat_separate() -> None:
    """SCENARIO-REPORT-7400-TRANSPORT counts faithful syntax, not SAT success."""

    rows = _calls()
    reduced = mod.reduce_transport(rows, _fixture()["development_formulas"])
    assert reduced["usable_proposal_count"] == 4
    assert reduced["source_literal_fidelity_count"] == 4
    assert reduced["qwen_assignment_transport_ready_score"] == 1
    assert reduced["sat_extendible_count"] < reduced["usable_proposal_count"]
    assert all(row["current_model_invocation"] for row in reduced["transport_rows"])
    assert all(not row["historical"] for row in reduced["transport_rows"])

    low_yield = _calls(["not-json", "not-json", '{"assignments":[1,3]}', "not-json"])
    low = mod.reduce_transport(low_yield, _fixture()["development_formulas"])
    assert low["usable_proposal_count"] == 1
    assert low["qwen_assignment_transport_ready_score"] == 0

    broken = deepcopy(rows)
    broken[0]["raw_reply_sha256"] = "sha256:" + "0" * 64
    invalid = mod.reduce_transport(broken, _fixture()["development_formulas"])
    assert invalid["qwen_assignment_transport_ready_score"] == 0
    assert "response_hash_mismatch:0" in invalid["errors"]


def test_scenario_report_7400_runtime_events_use_qualified_reducer() -> None:
    """SCENARIO-REPORT-7400-RUNTIME derives every current count from events."""

    reduced = mod.reduce_current_events(_events(), run_id="run-test", owner_pid=7400)
    assert reduced["model_invoked"] is True
    assert reduced["invocation_counts"]["model_loads_completed"] == 1
    assert reduced["invocation_counts"]["generation_calls_completed"] == 4
    assert reduced["event_sha256"].startswith("sha256:")

    unfinished = _events()[:-1]
    with pytest.raises(ValueError, match="unfinished_call:generation-3"):
        mod.reduce_current_events(unfinished, run_id="run-test", owner_pid=7400)


def test_scenario_report_7400_runtime_binds_all_layer_offload() -> None:
    """SCENARIO-REPORT-7400-RUNTIME requires owned CUDA and an all-layer command."""

    receipt = mod.build_offload_receipt(
        _identity(),
        {"provenance_ok": True, "cuda_log_evidence": True, "task_owned_vram_mb": 16340},
        model_block_count=65,
    )
    assert receipt["requested_gpu_layers"] == "all"
    assert receipt["actual_offloaded_layers"] == 65
    assert receipt["total_model_layers"] == 65
    assert receipt["all_layers_offloaded"] is True

    no_all = deepcopy(_identity())
    no_all["command"] = ["llama-server", "--n-gpu-layers", "32"]
    assert (
        mod.build_offload_receipt(no_all, {"provenance_ok": True}, 65)["all_layers_offloaded"]
        is False
    )
    missing_option = deepcopy(_identity())
    missing_option["command"] = ["llama-server"]
    assert mod.build_offload_receipt(missing_option, {}, 65)["requested_gpu_layers"] is None


def test_scenario_report_7400_validation_plan_is_exact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7400-VALIDATION freezes the eight Exp7358 checks."""

    commands = mod.build_validation_plan(mod.REPO_ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(REQUIRED_CHECK_NAMES)
    assert mod.validate_validation_plan(mod.REPO_ROOT, commands) == []
    assert "full_python_suite" not in {row.name for row in commands}
    coverage = next(row for row in commands if row.name == "changed_module_coverage")
    assert "-n" in coverage.argv
    assert "--no-cov" in coverage.argv
    assert (tmp_path / "private/basetemp").is_dir()


def test_scenario_report_7400_artifact_reloads_exact_raw_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7400-ARTIFACT rejects row, count, and checksum drift."""

    artifact = _artifact(tmp_path)
    assert mod.independent_reduce_artifact(artifact) == []
    assert mod.validate_artifact(artifact, require_terminal=True) == []

    changed = deepcopy(artifact)
    changed["usable_proposal_count"] = 0
    assert "usable_proposal_count_mismatch" in mod.independent_reduce_artifact(changed)

    changed = deepcopy(artifact)
    changed["invocation_counts"]["generation_calls_completed"] = 3
    assert "invocation_counts_mismatch" in mod.independent_reduce_artifact(changed)

    changed = deepcopy(artifact)
    changed["qwen_assignment_transport_ready_score"] = 0
    assert "qwen_assignment_transport_ready_score_mismatch" in mod.independent_reduce_artifact(
        changed
    )

    invalid_manifest = deepcopy(artifact)
    invalid_manifest["raw_call_rows"][0] = "bad"
    assert "raw_call_manifest_invalid:0" in mod.independent_reduce_artifact(invalid_manifest)

    missing_path = deepcopy(artifact)
    missing_path["raw_call_rows"][0]["path"] = str(tmp_path / "missing.json")
    assert "raw_call_path_missing:0" in mod.independent_reduce_artifact(missing_path)

    row_drift = deepcopy(artifact)
    row_drift["rows"][0]["decoded_assignments"] = [99]
    assert "raw_call_row_mismatch:0" in mod.independent_reduce_artifact(row_drift)

    missing_formulas = tmp_path / "empty-root"
    missing_formulas.mkdir()
    assert "development_formulas_unavailable" in mod.independent_reduce_artifact(
        artifact, root=missing_formulas
    )

    invalid_events = deepcopy(artifact)
    invalid_events["current_invocation_events"] = "bad"
    assert mod.independent_reduce_artifact(invalid_events) == ["current_invocation_events_invalid"]

    invalid_owner = deepcopy(artifact)
    invalid_owner["current_owner_pid"] = None
    assert mod.independent_reduce_artifact(invalid_owner)[0].startswith(
        "current_event_reduction_failed:"
    )

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = []
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "model_contract_invalid" in mod.validate_artifact(changed)

    Path(artifact["raw_call_rows"][0]["path"]).write_text("{}", encoding="utf-8")
    assert "raw_call_hash_mismatch:0" in mod.independent_reduce_artifact(artifact)
    assert mod.independent_reduce_artifact({}) == ["raw_evidence_unavailable"]


def test_req_report_7400_artifact_helpers_preserve_closed_contract(tmp_path: Path) -> None:
    """REQ-REPORT-7400 keeps checksum, date, receipt, and blocked shapes closed."""

    artifact = _artifact(tmp_path)
    checksum = artifact["reproducibility_checksum"]
    assert mod.artifact_checksum(artifact) == checksum
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
    assert blocked["promotion_score"] == 0


def test_req_report_7400_validator_rejects_each_closed_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-7400 rejects identity, class, timing, readiness, and receipt drift."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact([]) == ["artifact_not_object"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")

    cases = {
        "identity_invalid": {"schema": "wrong"},
        "execution_venue_invalid": {"execution_venue": "host_cpu"},
        "verdict_class_invalid": {"verdict_class": "success"},
        "substrate_class_invalid": {"inference_substrate_class": "model_full_generation"},
        "bounded_generation_duration_floor": {"duration_s": 9.9},
        "required_validation_receipts_invalid": {"validation_receipts": []},
        "positive_readiness_invalid": {"qwen_assignment_transport_ready_score": 0},
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
    blocked["qwen_assignment_transport_ready_score"] = 1
    blocked["reproducibility_checksum"] = "wrong"
    errors = mod.validate_artifact(blocked, require_terminal=False)
    assert "blocked_substrate_invalid" in errors
    assert "failed_readiness_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors
