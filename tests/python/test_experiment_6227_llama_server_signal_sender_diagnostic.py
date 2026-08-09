"""Tests for Exp6227 llama-server signal sender diagnostic.

Spec refs: REQ-INFRA-6227, SCENARIO-INFRA-6227-1,
SCENARIO-INFRA-6227-2, SCENARIO-INFRA-6227-3,
SCENARIO-INFRA-6227-4, SCENARIO-INFRA-6227-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import signal
from typing import Any

from carnot import experiment_6227_llama_server_signal_sender_diagnostic as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"
FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6227_llama_server_signal_sender_diagnostic.py "
    "-m pytest tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6227_llama_server_signal_sender_diagnostic.py "
    "--fail-under=100 --show-missing"
)
FULL_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py"
)
ARTIFACT_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6227_llama_server_signal_sender_diagnostic --date 20260809"
)
TEST_COMMANDS = [
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    FULL_COMMAND,
    SPEC_COMMAND,
    ARTIFACT_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _gguf(tmp_path: Path) -> Path:
    path = (
        tmp_path
        / "hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/rev-fixture"
        / "gemma-4-31B-it-Q4_K_M.gguf"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF fixture gemma 31b q4")
    return path


def _metadata(path: Path) -> mod.JsonDict:
    template = "fixture embedded chat template for " + path.name
    return {
        "chat_template_present": True,
        "chat_template_sha256": mod.sha256_text(template),
        "metadata_summary_sha256": mod.sha256_json({"tokenizer.chat_template": template}),
        "metadata_keys": ["tokenizer.chat_template"],
        "tokenizer_detail": "fixture embedded tokenizer",
    }


def _resolver(path: Path):
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert hf_id == mod.MODEL_SPECS[0]["hf_id"]
        assert preferred_quant == "Q4_K_M"
        return str(path)

    return resolve


def _gpu_snapshot(*, busy: bool = False) -> mod.JsonDict:
    apps = (
        [
            {
                "gpu_index": 1,
                "pid": 7777,
                "process_name": "external-owner",
                "used_memory_mb": 22000,
                "command": "python unrelated.py",
                "owned_by_task": False,
            }
        ]
        if busy
        else []
    )
    return {
        "ok": True,
        "label": "fixture",
        "gpu_count": 2,
        "devices": [
            {
                "index": 0,
                "name": "RTX 3090",
                "utilization_pct": 0,
                "memory_total_mb": 24576,
                "memory_used_mb": 4,
                "memory_free_mb": 24120,
            },
            {
                "index": 1,
                "name": "RTX 3090",
                "utilization_pct": 0 if not busy else 88,
                "memory_total_mb": 24576,
                "memory_used_mb": 4 if not busy else 22004,
                "memory_free_mb": 24120 if not busy else 2572,
            },
        ],
        "compute_apps": apps,
    }


class FakeRuntime:
    """REQ-INFRA-6227: deterministic adapter for bounded lifecycle tests."""

    def __init__(self, mode: str = "success", *, busy: bool = False) -> None:
        self.mode = mode
        self.busy = busy
        self.lifecycle_calls = 0
        self.child_tests = 0

    def gpu_snapshot(self, label: str) -> mod.JsonDict:
        row = _gpu_snapshot(busy=self.busy)
        row["label"] = label
        return row

    def llama_cpp_receipt(self) -> mod.JsonDict:
        return {
            "native_llama_server_path": "/tmp/llama-server",
            "native_llama_server_exists": True,
            "native_llama_server_version": "llama.cpp fixture cuda",
            "native_llama_server_cuda_build": True,
            "llama_cpp_python_gpu_offload": True,
            "no_autotokenizer_used": True,
        }

    def trace_tool_availability(self) -> mod.JsonDict:
        return {
            "tools": {
                "auditctl": {"available": True, "permitted": False, "reason": "permission_denied"},
                "bpftrace": {"available": False, "permitted": False, "reason": "not_found"},
                "strace": {"available": True, "permitted": True, "reason": "version_ok"},
            },
            "no_privilege_escalation_requested": True,
            "sender_pid_observable_without_extra_privilege": False,
        }

    def run_owned_lifecycle(
        self,
        gguf: mod.JsonDict,
        command: list[str],
        contract: mod.JsonDict,
        output_dir: Path,
    ) -> mod.JsonDict:
        self.lifecycle_calls += 1
        retry_attempts = 1 if self.mode == "retry" else 0
        token_returned = self.mode in {"success", "retry"}
        deadline = self.mode == "deadline"
        first_class = "connection_refused" if self.mode == "retry" else "no_failure"
        final_class = "deadline_expired" if deadline else "no_failure"
        token_hash = mod.sha256_bytes(b"A") if token_returned else None
        return {
            "owned_process_tree_snapshots": [
                {"label": "after_launch", "nodes": [{"pid": 622700, "start_time_ticks": 12345}]}
            ],
            "launch_receipts": {
                "command": command,
                "pid": 622700,
                "start_time_ticks": 12345,
                "session_id": 622700,
                "process_group_id": 622700,
                "cgroup": ["0::/fixture"],
                "deadline_s": contract["outer_deadline_s"],
            },
            "health_and_token_timeline": {
                "bounded": True,
                "retry_attempts": retry_attempts,
                "token_returned": token_returned,
                "embedded_chat_template_used": True,
                "request_endpoint": "/v1/chat/completions",
                "events": [
                    {"phase": "launch", "elapsed_s": 0.0, "classification": first_class},
                    {
                        "phase": "token" if token_returned else "deadline",
                        "elapsed_s": 0.5,
                        "classification": final_class,
                        "first_token_bytes_sha256": token_hash,
                    },
                ],
                "timeouts": {
                    "outer_deadline_s": contract["outer_deadline_s"],
                    "health_timeout_s": contract["health_timeout_s"],
                    "token_timeout_s": contract["token_timeout_s"],
                },
            },
            "signal_events": [
                {
                    "source": "prior_server_log",
                    "classification": "server_received_interrupt_sender_unknown",
                    "sender_pid": None,
                    "evidence": "Received second interrupt",
                }
            ],
            "server_exit": {
                "pid": 622700,
                "start_time_ticks": 12345,
                "exit_code": 0 if token_returned else None,
                "classification": final_class,
                "stderr_path": str(output_dir / "fixture.llama_server.log"),
                "stderr_sha256": mod.sha256_text("operator(): cleaning up before exit"),
                "stderr_tail": "operator(): cleaning up before exit",
                "received_interrupt_markers": 1,
            },
            "caller_wait": {
                "bounded_wait_observed": True,
                "wait_timeout_s": contract["token_timeout_s"],
                "thread_snapshots": [{"label": "before_request", "threads": ["MainThread"]}],
                "classifications": [final_class],
            },
        }

    def controlled_child_death_test(self, contract: mod.JsonDict) -> mod.JsonDict:
        self.child_tests += 1
        return {
            "passed": True,
            "classification": "controlled_owned_child_sigterm_bounded",
            "timeout_s": contract["cleanup_grace_s"],
            "cleanup_receipt": {
                "action": "terminated",
                "bounded": True,
                "signals_sent": [{"target": "process_group", "signal": "SIGTERM"}],
                "unrelated_process_kill_count_delta": 0,
            },
        }


class FakeProcessOps:
    """SCENARIO-INFRA-6227-4: process operations are injectable in tests."""

    def __init__(self, wait_result: str = "exited") -> None:
        self.wait_result = wait_result
        self.signals: list[dict[str, Any]] = []

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        self.signals.append(
            {
                "pid": pid,
                "signal": sig.name,
                "process_group": process_group,
            }
        )

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        assert pid == 100
        assert timeout_s > 0
        return self.wait_result


def test_req_infra_6227_spec_declares_signal_sender_contract() -> None:
    """REQ-INFRA-6227: OpenSpec names the model, fields, and scenarios."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6227") : text.index("REQ-INFRA-6210")]

    for marker in (
        "REQ-INFRA-6227",
        "SCENARIO-INFRA-6227-1",
        "SCENARIO-INFRA-6227-2",
        "SCENARIO-INFRA-6227-3",
        "SCENARIO-INFRA-6227-4",
        "SCENARIO-INFRA-6227-5",
        mod.MODEL_SPECS[0]["hf_id"],
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6227_complete_unlocalized_artifact_is_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6227-2: one owned lifecycle can be ready with unknown sender."""

    runtime = FakeRuntime()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        model_resolver=_resolver(_gguf(tmp_path)),
        metadata_reader=_metadata,
        runtime=runtime,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=True,
    )

    assert runtime.lifecycle_calls == 1
    assert runtime.child_tests == 1
    assert artifact["status"] == "complete_unlocalized"
    assert artifact["sender_identified_score"] == 0
    assert artifact["runtime_diagnostic_ready_score"] == 1
    assert artifact["unlocalized_reason"].startswith("sender not identified:")
    assert artifact["health_and_token_timeline"]["token_returned"] is True
    assert artifact["health_and_token_timeline"]["bounded"] is True
    assert artifact["health_and_token_timeline"]["embedded_chat_template_used"] is True
    assert artifact["health_and_token_timeline"]["request_endpoint"] == "/v1/chat/completions"
    assert artifact["exact_gguf_and_llama_cpp_receipts"]["gguf"]["no_autotokenizer_used"] is True
    assert artifact["exact_gguf_and_llama_cpp_receipts"]["command"][0] == "/tmp/llama-server"
    assert (
        str(artifact["exact_gguf_and_llama_cpp_receipts"]["gguf"]["model_path"])
        in (artifact["exact_gguf_and_llama_cpp_receipts"]["command"])
    )
    assert artifact["controlled_owned_child_death_test"]["passed"] is True
    assert artifact["unrelated_process_kill_count"] == 0
    assert type(artifact["unrelated_process_kill_count"]) is int
    assert artifact["gguf_mutation_count"] == 0
    assert type(artifact["gguf_mutation_count"]) is int
    assert artifact["protected_files_unchanged"]["scripts_research_conductor_py_untouched"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert mod.validate_artifact(artifact) == []


def test_scenario_infra_6227_prior_incidents_are_phase_separated(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6227-1: persisted logs do not collapse phases."""

    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "results/experiment_6212_three_family_gguf_runtime_recovery.json").write_text(
        '{"status":"complete_partial"}\n', encoding="utf-8"
    )
    (root / "ops/known-issues.md").write_text(
        "RemoteDisconnected after llama-server death\n"
        "caller hung on sigsuspend waiting on defunct llama-server\n",
        encoding="utf-8",
    )
    run_log = root / "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json"
    run_log.write_text('{"honest_verdict":"complete"}\n', encoding="utf-8")
    log_dir = tmp_path / "server_logs"
    log_dir.mkdir()
    (log_dir / "llama_server_p8940_1.log").write_text(
        "slot update\noperator(): cleaning up before exit\nReceived second interrupt\n",
        encoding="utf-8",
    )

    timeline = mod.reconstruct_prior_incidents(root=root, server_log_dir=log_dir)

    assert timeline["phase_counts"]["server_death"] >= 1
    assert timeline["phase_counts"]["connection_failure"] >= 1
    assert timeline["phase_counts"]["caller_wait_or_hang"] >= 1
    assert timeline["phase_counts"]["retry"] >= 1
    assert all(row["sha256"].startswith("sha256:") for row in timeline["paths"])
    assert timeline["timeline_hash"].startswith("sha256:")


def test_scenario_infra_6227_classification_contract_and_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6227-3: exact classes and artifact validation fail closed."""

    assert mod.classify_runtime_event(exception_name="ConnectionRefusedError") == (
        "connection_refused"
    )
    assert mod.classify_runtime_event(text="RemoteDisconnected: remote end closed") == (
        "server_died_mid_request"
    )
    assert mod.classify_runtime_event(exception_name="TimeoutError") == "request_timeout"
    assert mod.classify_runtime_event(deadline_expired=True) == "deadline_expired"
    assert mod.classify_runtime_event(text="caller wait exceeded") == "caller_wait_timeout"
    assert mod.classify_runtime_event(exit_code=-15) == "external_signal_sigterm"
    assert mod.classify_runtime_event(text="Received second interrupt") == (
        "server_received_interrupt_sender_unknown"
    )
    assert mod.classify_runtime_event("", 0) == "no_failure"
    assert mod.classify_runtime_event("odd stderr", 2) == "unclassified_runtime_failure"

    contract = mod.finite_wait_retry_cleanup_contract()
    assert contract["retry_budget"] == 1
    assert contract["pid_reuse_guard"] == "pid_start_time_ticks_and_uid"
    assert set(contract["classified_failures"]) >= {
        "connection_refused",
        "server_died_mid_request",
        "deadline_expired",
    }
    assert mod.controlled_child_not_run(contract)["classification"] == "not_run"
    localized = {
        "events": [
            {"source": "controlled_cleanup", "sender_pid": 1},
            {"source": "owned_cleanup", "sender_pid": 1111},
            {"source": "strace", "sender_pid": 2222},
        ]
    }
    assert (
        mod.sender_identified_score({"events": [{"source": "owned_cleanup", "sender_pid": 1111}]})
        == 0
    )
    assert (
        mod.sender_identified_score(
            {
                "events": [
                    {
                        "classification": "controlled_sigterm_from_current_process",
                        "sender_pid": 1111,
                    }
                ]
            }
        )
        == 0
    )
    assert mod.sender_identified_score(localized) == 1
    assert mod.unlocalized_reason({}, 1) == "sender identified by permitted tracing receipt"
    assert mod.status_for_artifact([], 1, 1, {}) == "complete_localized"
    assert mod.status_for_artifact([], 0, 0, {}) == "blocked"
    assert mod.honest_verdict({"status": "complete_localized"}).startswith("complete_localized:")
    taxonomy = mod.root_cause_taxonomy(
        {"phase_counts": {"server_death": 1, "caller_wait_or_hang": 0}},
        {
            "health_and_token_timeline": {"token_returned": False, "bounded": True},
            "server_exit": {"classification": "external_signal_sigterm"},
            "caller_wait": {"bounded_wait_observed": True},
        },
        localized,
        [],
    )
    assert "sender_localized" in taxonomy["categories"]

    artifact = mod.run(
        result_path=tmp_path / "artifact.json",
        model_resolver=_resolver(_gguf(tmp_path)),
        metadata_reader=_metadata,
        runtime=FakeRuntime(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=False,
    )
    assert artifact["field_provenance"]["status"][0] == "REQ-INFRA-6227"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    mutations = [
        (lambda item: item.pop("status"), "missing:status"),
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": True}), "verifier_is_oracle"),
        (
            lambda item: item.update({"unrelated_process_kill_count": False}),
            "unrelated_process_kill_count",
        ),
        (lambda item: item.update({"gguf_mutation_count": "0"}), "gguf_mutation_count"),
        (lambda item: item.update({"sender_identified_score": 2}), "sender_identified_score"),
        (
            lambda item: item.update({"runtime_diagnostic_ready_score": 0}),
            "runtime_diagnostic_ready_score",
        ),
        (lambda item: item["field_provenance"].pop("status"), "field_provenance:status"),
        (lambda item: item["field_principles"].pop("status"), "field_principles:status"),
        (lambda item: item.update({"honest_verdict": "maybe"}), "honest_verdict"),
        (
            lambda item: item.update({"reproducibility_checksum": "sha256:bad"}),
            "reproducibility_checksum",
        ),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        assert expected in mod.validate_artifact(candidate)


def test_scenario_infra_6227_pid_reuse_and_unowned_cleanup_are_refused() -> None:
    """SCENARIO-INFRA-6227-4: cleanup uses the recorded PID/start-time identity."""

    recorded = {
        "pid": 100,
        "start_time_ticks": 555,
        "uid": 1000,
        "pgid": 100,
        "owned_by_task": True,
    }
    current = {**recorded, "exists": True}
    ops = FakeProcessOps()
    receipt = mod.cleanup_recorded_identity(recorded, lambda _pid: current, ops, grace_s=0.1)
    assert receipt["action"] == "terminated"
    assert receipt["bounded"] is True
    assert ops.signals == [{"pid": 100, "signal": "SIGTERM", "process_group": True}]
    assert receipt["unrelated_process_kill_count_delta"] == 0

    reused_ops = FakeProcessOps()
    reused = {**current, "start_time_ticks": 556}
    refused = mod.cleanup_recorded_identity(recorded, lambda _pid: reused, reused_ops, grace_s=0.1)
    assert refused["action"] == "refused"
    assert refused["reason"] == "identity_mismatch"
    assert reused_ops.signals == []

    unowned_ops = FakeProcessOps()
    unowned_record = {**recorded, "owned_by_task": False}
    unowned = mod.cleanup_recorded_identity(
        unowned_record, lambda _pid: current, unowned_ops, grace_s=0.1
    )
    assert unowned["action"] == "refused"
    assert unowned["reason"] == "unowned_process"
    assert unowned_ops.signals == []

    already_gone = mod.cleanup_recorded_identity(
        recorded, lambda _pid: {"pid": 100, "exists": False}, FakeProcessOps(), grace_s=0.1
    )
    assert already_gone["action"] == "already_exited"


def test_scenario_infra_6227_dead_server_retry_and_deadline_are_bounded(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6227-4: retry and deadline cases stay finite."""

    retry_artifact = mod.run(
        result_path=tmp_path / "retry.json",
        model_resolver=_resolver(_gguf(tmp_path / "retry")),
        metadata_reader=_metadata,
        runtime=FakeRuntime(mode="retry"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=False,
    )
    assert retry_artifact["status"] == "complete_unlocalized"
    assert retry_artifact["health_and_token_timeline"]["retry_attempts"] == 1
    assert retry_artifact["health_and_token_timeline"]["events"][0]["classification"] == (
        "connection_refused"
    )
    assert mod.validate_artifact(retry_artifact) == []

    deadline_artifact = mod.run(
        result_path=tmp_path / "deadline.json",
        model_resolver=_resolver(_gguf(tmp_path / "deadline")),
        metadata_reader=_metadata,
        runtime=FakeRuntime(mode="deadline"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=False,
    )
    assert deadline_artifact["status"] == "failed_bounded"
    assert deadline_artifact["runtime_diagnostic_ready_score"] == 0
    assert deadline_artifact["health_and_token_timeline"]["events"][-1]["classification"] == (
        "deadline_expired"
    )
    assert (
        deadline_artifact["caller_thread_and_wait_state_receipts"]["bounded_wait_observed"] is True
    )
    assert mod.validate_artifact(deadline_artifact) == []


def test_req_infra_6227_model_precondition_blockers_and_proc_parsing(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-INFRA-6227: exact model and process identity helpers are mechanical."""

    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        model_resolver=lambda _hf_id, _preferred_quant: None,
        metadata_reader=_metadata,
        runtime=FakeRuntime(busy=True),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert "gemma4_31b_dense_gguf_not_cached" in blocked["preconditions_checked"]["blocked_reasons"]
    assert "external_gpu_owner_present" in blocked["preconditions_checked"]["blocked_reasons"]
    assert blocked["gpu_owner_receipts_before_during_after"]["admission"]["blocked_owner_pids"] == [
        7777
    ]
    assert blocked["health_and_token_timeline"]["token_returned"] is False
    assert mod.read_tail(tmp_path / "missing.log") == ""
    assert mod.snapshot_revision(Path("/flat/model.gguf")) == "local-flat-cache"
    assert (
        "dual_gpu_snapshot_unavailable"
        in mod.safe_gpu_admission({"ok": False, "devices": [], "compute_apps": []})[
            "blocked_reasons"
        ]
    )
    model_dir = tmp_path / "resolved-directory"
    model_dir.mkdir()
    directory_record = mod.resolve_exact_gguf(
        model_resolver=lambda _hf_id, _preferred_quant: str(model_dir),
        metadata_reader=lambda _path: {},
    )
    assert "gemma4_31b_dense_resolved_path_not_file" in directory_record["blocked_reasons"]
    assert "gemma4_31b_dense_embedded_chat_template_missing" in directory_record["blocked_reasons"]

    stat = "100 (name with space) S 9 100 100 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 98765 0 0"
    parsed = mod.parse_proc_stat(stat)
    assert parsed["pid"] == 100
    assert parsed["comm"] == "name with space"
    assert parsed["ppid"] == 9
    assert parsed["pgid"] == 100
    assert parsed["session_id"] == 100
    assert parsed["start_time_ticks"] == 98765

    assert mod.identity_matches(
        {"pid": 100, "start_time_ticks": 98765, "uid": 1000},
        {"pid": 100, "start_time_ticks": 98765, "uid": 1000, "exists": True},
    )
    assert not mod.identity_matches(
        {"pid": 100, "start_time_ticks": 98765, "uid": 1000},
        {"pid": 100, "start_time_ticks": 98766, "uid": 1000, "exists": True},
    )

    class MissingServerRuntime(FakeRuntime):
        def llama_cpp_receipt(self) -> mod.JsonDict:
            row = super().llama_cpp_receipt()
            row["native_llama_server_exists"] = False
            return row

    missing_server = mod.run(
        result_path=tmp_path / "missing-server.json",
        model_resolver=_resolver(_gguf(tmp_path / "missing-server")),
        metadata_reader=_metadata,
        runtime=MissingServerRuntime(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=False,
    )
    assert (
        "native_llama_server_missing" in missing_server["preconditions_checked"]["blocked_reasons"]
    )

    monkeypatch.setattr(mod, "_parent_writable", lambda _path: False)
    unwritable = mod.run(
        result_path=tmp_path / "unwritable.json",
        model_resolver=_resolver(_gguf(tmp_path / "unwritable")),
        metadata_reader=_metadata,
        runtime=FakeRuntime(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.227,
        write=False,
    )
    assert "output_parent_not_writable" in unwritable["preconditions_checked"]["blocked_reasons"]
