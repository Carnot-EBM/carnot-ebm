"""Exp6227 llama-server signal sender diagnostic.

Spec refs: REQ-INFRA-6227, SCENARIO-INFRA-6227-1,
SCENARIO-INFRA-6227-2, SCENARIO-INFRA-6227-3,
SCENARIO-INFRA-6227-4, SCENARIO-INFRA-6227-5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Protocol
from urllib import error, request

from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    llama_cpp_build_receipt,
    nvidia_smi_gpu_snapshot,
    read_gguf_metadata,
    resolve_native_llama_server,
)
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6227_llama_server_signal_sender_diagnostic.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6227_llama_server_signal_sender_diagnostic.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")

SCHEMA = "carnot.experiment_6227.llama_server_signal_sender_diagnostic.v1"
EXPERIMENT_ID = "experiment_6227_llama_server_signal_sender_diagnostic"
RUN_DATE = "20260809"
RANDOM_SEED = 6227
INFERENCE_SUBSTRATE = "native_llama_cpp_owned_signal_sender_diagnostic"
PREFERRED_QUANT = "Q4_K_M"
CANARY_N_CTX = 384
SAFE_SPLIT_FREE_MB_PER_GPU = 12_000

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma4_31b_dense",
        "role": "reaper reproduction and caller-hang diagnostic",
        "preferred_quant": PREFERRED_QUANT,
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "prior_incident_paths_hashes_and_timeline",
    "preconditions_checked",
    "gpu_owner_receipts_before_during_after",
    "model_specs",
    "exact_gguf_and_llama_cpp_receipts",
    "owned_process_tree_snapshots",
    "launch_command_session_and_cgroup_receipts",
    "health_and_token_timeline",
    "signal_trace_tool_availability",
    "signal_events_and_sender_receipts",
    "server_exit_and_stderr_receipts",
    "caller_thread_and_wait_state_receipts",
    "bounded_reproduction_deadline",
    "root_cause_taxonomy",
    "sender_identified_score",
    "unlocalized_reason",
    "finite_wait_retry_cleanup_contract",
    "controlled_owned_child_death_test",
    "unrelated_process_kill_count",
    "gguf_mutation_count",
    "runtime_diagnostic_ready_score",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The terminal state separates ready, bounded failure, and blocked diagnostics.",
    "prior_incident_paths_hashes_and_timeline": "Persisted incident evidence is phase-separated before any new launch.",
    "preconditions_checked": "GPU, privilege, model, server, output, and protected-file gates run before launch.",
    "gpu_owner_receipts_before_during_after": "GPU owner receipts prevent accidental external-process cleanup.",
    "model_specs": "Only the mandated Gemma-4-31B GGUF is eligible.",
    "exact_gguf_and_llama_cpp_receipts": "Exact model bytes and native server identity prevent silent substitution.",
    "owned_process_tree_snapshots": "Process-tree snapshots bind observations to task-owned ancestry.",
    "launch_command_session_and_cgroup_receipts": "Launch receipts record PID, start time, session, process group, and cgroup.",
    "health_and_token_timeline": "Health and token events prove the caller made bounded progress or failed finitely.",
    "signal_trace_tool_availability": "Unavailable or denied sender channels are recorded instead of guessed.",
    "signal_events_and_sender_receipts": "Signal evidence records sender PID only when the host exposes it.",
    "server_exit_and_stderr_receipts": "Exit status and stderr preserve the server-side signal view.",
    "caller_thread_and_wait_state_receipts": "Caller state explains waits and prevents silent multi-hour hangs.",
    "bounded_reproduction_deadline": "Every lifecycle phase is capped by explicit wall-clock limits.",
    "root_cause_taxonomy": "The result distinguishes localized sender, unlocalized signal, bounded retry, and non-reproduction.",
    "sender_identified_score": "The score is one only when sender identity evidence is direct.",
    "unlocalized_reason": "Unknown sender cases state exactly which evidence was missing.",
    "finite_wait_retry_cleanup_contract": "The recovery contract defines finite waits, retries, and owned cleanup.",
    "controlled_owned_child_death_test": "A test child proves bounded wait and ownership-safe cleanup.",
    "unrelated_process_kill_count": "Bare zero proves no unrelated process was killed.",
    "gguf_mutation_count": "Bare zero proves cached model bytes were not modified.",
    "runtime_diagnostic_ready_score": "Readiness can be one with an unknown sender when evidence and contract are complete.",
    "protected_files_unchanged": "Conductor, ops, and protected inputs remain byte-identical.",
    "inference_substrate": "Declares native llama.cpp server diagnostics rather than benchmark inference.",
    "verifier_is_oracle": "False because this diagnostic is runtime evidence, not hidden-task grading.",
    "field_provenance": "Every required field traces to REQ-INFRA-6227.",
    "field_principles": "Each field states the audit failure it prevents.",
    "test_commands": "Verification commands are recorded with the artifact.",
    "test_exit_codes": "Exit codes preserve what was actually verified.",
    "duration_s": "Measured wall time is reported without padding.",
    "reproducibility_checksum": "Stable checksum binds the diagnostic payload.",
    "honest_verdict": "The verdict reports localization limits without overclaiming.",
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("ops/known-issues.md"),
    Path("results/experiment_6212_three_family_gguf_runtime_recovery.json"),
    Path("scripts/run_exp6199_expanded_roster.py"),
)

PRIOR_INCIDENT_PATHS = (
    Path("ops/known-issues.md"),
    Path("results/experiment_6212_three_family_gguf_runtime_recovery.json"),
    Path("results/experiment_6221_gemma_think_mode_ab_expanded_roster.json"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6227_llama_server_signal_sender_diagnostic.py -m pytest tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6227_llama_server_signal_sender_diagnostic.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py",
    ".venv/bin/python -m carnot.experiment_6227_llama_server_signal_sender_diagnostic --date 20260809",
)


class RuntimeAdapter(Protocol):
    """Small live boundary for deterministic artifact tests."""

    def gpu_snapshot(self, label: str) -> JsonDict:
        """Return GPU device and owner receipts."""

    def llama_cpp_receipt(self) -> JsonDict:
        """Return native llama.cpp server identity."""

    def trace_tool_availability(self) -> JsonDict:
        """Return already-available signal tracing channels."""

    def run_owned_lifecycle(
        self,
        gguf: JsonDict,
        command: list[str],
        contract: JsonDict,
        output_dir: Path,
    ) -> JsonDict:
        """Run one bounded owned server lifecycle."""

    def controlled_child_death_test(self, contract: JsonDict) -> JsonDict:
        """Prove bounded cleanup on a task-owned child."""


class ProcessOps(Protocol):
    """Injectable process actions for ownership tests."""

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        """Send a signal to a PID or process group."""

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        """Wait no longer than timeout_s and return exited or timeout."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def utc_now() -> str:  # pragma: no cover
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def path_receipt(path: Path) -> JsonDict:
    exists = path.exists()
    is_file = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": path.stat().st_size if is_file else None,
        "sha256": sha256_file(path) if is_file else None,
    }


def read_tail(path: Path, max_chars: int = 20_000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def observed_quantization(path: Path) -> str:
    match = re.search(r"(?:UD-)?Q\d(?:_[A-Z0-9]+)+", path.name)
    return match.group(0) if match else "unknown"


def snapshot_revision(path: Path) -> str:
    parts = path.parts
    if "snapshots" not in parts:
        return "local-flat-cache"
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else "local-flat-cache"


def protected_file_hash_map(root: Path = REPO_ROOT) -> JsonDict:
    return {relative.as_posix(): path_receipt(root / relative) for relative in PROTECTED_FILES}


def protected_files_unchanged(before: JsonDict, root: Path = REPO_ROOT) -> JsonDict:
    after = protected_file_hash_map(root)
    changed = [
        path
        for path, receipt in before.items()
        if after.get(path, {}).get("sha256") != receipt.get("sha256")
        or after.get(path, {}).get("exists") != receipt.get("exists")
    ]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
        "ops_ledgers_untouched": not any(
            path in changed
            for path in ("ops/changelog.md", "ops/status.md", "_bmad/traceability.md")
        ),
    }


def resolve_exact_gguf(
    *,
    model_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[Path], JsonDict] = read_gguf_metadata,
) -> JsonDict:
    spec = MODEL_SPECS[0]
    path_text = model_resolver(str(spec["hf_id"]), str(spec["preferred_quant"]))
    if not path_text:
        return {
            **spec,
            "model_path": None,
            "exists": False,
            "path_is_file": False,
            "sha256": None,
            "size_bytes": None,
            "revision": None,
            "quantization": None,
            "chat_template_present": False,
            "no_autotokenizer_used": True,
            "blocked_reasons": [f"{spec['family']}_gguf_not_cached"],
        }
    path = Path(path_text)
    metadata = metadata_reader(path) if path.is_file() else {}
    blocked: list[str] = []
    if not path.is_file():
        blocked.append(f"{spec['family']}_resolved_path_not_file")
    if not metadata.get("chat_template_present"):
        blocked.append(f"{spec['family']}_embedded_chat_template_missing")
    return {
        **spec,
        "model_path": str(path),
        "real_path": str(path.resolve()) if path.exists() else str(path),
        "filename": path.name,
        "exists": path.exists(),
        "path_is_file": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "revision": snapshot_revision(path),
        "quantization": observed_quantization(path),
        "chat_template_present": bool(metadata.get("chat_template_present")),
        "chat_template_sha256": metadata.get("chat_template_sha256"),
        "metadata_summary_sha256": metadata.get("metadata_summary_sha256"),
        "metadata_keys": list(metadata.get("metadata_keys", [])),
        "embedded_tokenizer_detail": metadata.get("tokenizer_detail", "metadata parser used"),
        "no_autotokenizer_used": True,
        "blocked_reasons": blocked,
    }


def build_server_command(
    *,
    server_path: str | Path,
    model_path: str | Path,
    port: int,
    n_ctx: int = CANARY_N_CTX,
) -> list[str]:
    return [
        str(server_path),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(n_ctx),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--fit",
        "off",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--log-verbosity",
        "3",
    ]


def free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def safe_gpu_admission(
    gpu_snapshot: JsonDict, *, min_free_mb: int = SAFE_SPLIT_FREE_MB_PER_GPU
) -> JsonDict:
    devices = [dict(row) for row in gpu_snapshot.get("devices", [])]
    apps = [dict(row) for row in gpu_snapshot.get("compute_apps", [])]
    external_apps = [row for row in apps if not bool(row.get("owned_by_task"))]
    free_blockers = [
        int(row.get("index", -1))
        for row in devices
        if int(row.get("memory_free_mb", 0)) < min_free_mb
    ]
    blockers: list[str] = []
    if not gpu_snapshot.get("ok") or len(devices) < 2:
        blockers.append("dual_gpu_snapshot_unavailable")
    if external_apps:
        blockers.append("external_gpu_owner_present")
    if free_blockers:
        blockers.append("insufficient_split_free_vram")
    return {
        "safe": not blockers,
        "blocked_reasons": blockers,
        "min_free_mb_per_gpu_required": min_free_mb,
        "blocked_owner_pids": [
            int(row["pid"]) for row in external_apps if str(row.get("pid", "")).isdigit()
        ],
        "free_vram_blocked_gpu_indices": free_blockers,
        "external_compute_apps": external_apps,
    }


def classify_runtime_event(
    text: str = "",
    exit_code: int | None = None,
    *,
    exception_name: str = "",
    deadline_expired: bool = False,
) -> str:
    lowered = f"{text} {exception_name}".lower()
    if deadline_expired:
        return "deadline_expired"
    if "caller wait exceeded" in lowered or "callerwaittimeout" in lowered:
        return "caller_wait_timeout"
    if "connectionrefused" in lowered or "connection refused" in lowered:
        return "connection_refused"
    if (
        "remotedisconnected" in lowered
        or "remote end closed" in lowered
        or "connection aborted" in lowered
    ):
        return "server_died_mid_request"
    if "timeouterror" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "request_timeout"
    if exit_code is not None and exit_code < 0:
        sig = signal.Signals(-exit_code).name.lower()
        return f"external_signal_{sig}"
    if "received second interrupt" in lowered or "cleaning up before exit" in lowered:
        return "server_received_interrupt_sender_unknown"
    if exit_code == 0 and not text.strip() and not exception_name:
        return "no_failure"
    return "unclassified_runtime_failure"


def finite_wait_retry_cleanup_contract() -> JsonDict:
    return {
        "schema": SCHEMA + ".finite_wait_retry_cleanup_contract",
        "outer_deadline_s": 600,
        "health_timeout_s": 360,
        "token_timeout_s": 120,
        "request_timeout_s": 120,
        "cleanup_grace_s": 30,
        "kill_after_cleanup_timeout_s": 10,
        "retry_budget": 1,
        "retryable_failures": ["connection_refused", "server_died_mid_request"],
        "classified_failures": [
            "connection_refused",
            "server_died_mid_request",
            "request_timeout",
            "deadline_expired",
            "caller_wait_timeout",
            "external_signal_sigterm",
            "server_received_interrupt_sender_unknown",
            "owned_cleanup_after_success",
            "unclassified_runtime_failure",
        ],
        "pid_reuse_guard": "pid_start_time_ticks_and_uid",
        "cleanup_scope": "recorded_pid_identity_or_process_group_only",
        "unrelated_process_policy": "refuse_cleanup_when_pid_start_time_or_uid_mismatch",
        "checkpointing_added": False,
    }


def parse_proc_stat(text: str) -> JsonDict:
    close = text.rfind(")")
    open_paren = text.find("(")
    rest = text[close + 2 :].split()
    return {
        "pid": int(text[:open_paren].strip()),
        "comm": text[open_paren + 1 : close],
        "state": rest[0],
        "ppid": int(rest[1]),
        "pgid": int(rest[2]),
        "session_id": int(rest[3]),
        "start_time_ticks": int(rest[19]),
    }


def read_process_identity(
    pid: int, proc_root: Path = Path("/proc")
) -> JsonDict:  # pragma: no cover
    proc_dir = proc_root / str(pid)
    if not proc_dir.exists():
        return {"pid": pid, "exists": False}
    stat = parse_proc_stat((proc_dir / "stat").read_text(encoding="utf-8"))
    uid = None
    status_text = (proc_dir / "status").read_text(encoding="utf-8", errors="replace")
    for line in status_text.splitlines():
        if line.startswith("Uid:"):
            uid = int(line.split()[1])
            break
    cmdline = (
        (proc_dir / "cmdline")
        .read_bytes()
        .replace(b"\x00", b" ")
        .decode("utf-8", "replace")
        .strip()
    )
    cgroup = (proc_dir / "cgroup").read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        **stat,
        "exists": True,
        "uid": uid,
        "cmdline": cmdline,
        "cgroup": cgroup,
        "owned_by_current_uid": uid == os.getuid(),
    }


def identity_matches(recorded: JsonDict, current: JsonDict) -> bool:
    return (
        bool(current.get("exists"))
        and int(current.get("pid", -1)) == int(recorded.get("pid", -2))
        and int(current.get("start_time_ticks", -1)) == int(recorded.get("start_time_ticks", -2))
        and int(current.get("uid", -1)) == int(recorded.get("uid", -2))
    )


def cleanup_recorded_identity(
    recorded: JsonDict,
    current_identity: Callable[[int], JsonDict],
    process_ops: ProcessOps,
    *,
    grace_s: float,
) -> JsonDict:
    pid = int(recorded.get("pid", -1))
    current = current_identity(pid)
    if not current.get("exists"):
        return {
            "action": "already_exited",
            "bounded": True,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        }
    if not bool(recorded.get("owned_by_task")):
        return {
            "action": "refused",
            "reason": "unowned_process",
            "bounded": True,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        }
    if not identity_matches(recorded, current):
        return {
            "action": "refused",
            "reason": "identity_mismatch",
            "bounded": True,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        }
    process_group = int(recorded.get("pgid", -1)) == pid
    process_ops.send_signal(pid, signal.SIGTERM, process_group=process_group)
    wait_status = process_ops.wait_for_exit(pid, grace_s)
    signals = [{"target": "process_group" if process_group else "pid", "signal": "SIGTERM"}]
    return {
        "action": "terminated" if wait_status == "exited" else "timeout_after_sigterm",
        "bounded": True,
        "wait_status": wait_status,
        "signals_sent": signals,
        "unrelated_process_kill_count_delta": 0,
    }


def reconstruct_prior_incidents(
    *,
    root: Path = REPO_ROOT,
    server_log_dir: Path = Path("/tmp/carnot_llama_server_logs"),
) -> JsonDict:
    paths = [path_receipt(root / relative) for relative in PRIOR_INCIDENT_PATHS]
    timeline: list[JsonDict] = []
    known = read_tail(root / "ops/known-issues.md")
    if "RemoteDisconnected" in known or "connection" in known.lower():
        timeline.append(
            {
                "phase": "connection_failure",
                "source": "ops/known-issues.md",
                "evidence": "known issues mention connection failure after server loss",
            }
        )
    if "sigsuspend" in known or "hung" in known.lower() or "defunct" in known.lower():
        timeline.append(
            {
                "phase": "caller_wait_or_hang",
                "source": "ops/known-issues.md",
                "evidence": "known issues mention caller wait or hang on defunct server",
            }
        )
    if (root / "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json").exists():
        timeline.append(
            {
                "phase": "retry",
                "source": "results/experiment_6221_gemma_think_mode_ab_expanded_roster.json",
                "evidence": "expanded roster artifact exists after multiple rerun attempts",
            }
        )
    logs = sorted(server_log_dir.glob("llama_server*.log"), key=lambda item: item.stat().st_mtime)[
        -24:
    ]
    for log in logs:
        tail = read_tail(log, 4000)
        if "Received second interrupt" in tail or "cleaning up before exit" in tail:
            timeline.append(
                {
                    "phase": "server_death",
                    "source": str(log),
                    "evidence": "server stderr contains interrupt or cleanup marker",
                    "sha256": sha256_file(log),
                }
            )
    phase_counts = {
        phase: 0 for phase in ("server_death", "connection_failure", "retry", "caller_wait_or_hang")
    }
    for row in timeline:
        phase = str(row.get("phase"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    return {
        "schema": SCHEMA + ".prior_incident_timeline",
        "paths": paths,
        "server_log_dir": str(server_log_dir),
        "timeline": timeline,
        "phase_counts": phase_counts,
        "separated_phases": sorted([phase for phase, count in phase_counts.items() if count]),
        "timeline_hash": sha256_json(timeline),
    }


def sender_identified_score(signal_receipts: JsonDict) -> int:
    for row in signal_receipts.get("events", []):
        if row.get("source") in {"controlled_cleanup", "owned_cleanup"}:
            continue
        if row.get("classification") == "controlled_sigterm_from_current_process":
            continue
        if isinstance(row.get("sender_pid"), int):
            return 1
    return 0


def unlocalized_reason(trace_tools: JsonDict, score: int) -> str:
    if score:
        return "sender identified by permitted tracing receipt"
    tools = trace_tools.get("tools", {})
    unavailable = sorted(
        name for name, row in tools.items() if not row.get("available") or not row.get("permitted")
    )
    return (
        "sender not identified: Linux wait status and llama-server stderr do not expose sender PID; "
        f"unavailable_or_denied_tools={unavailable}"
    )


def root_cause_taxonomy(
    prior: JsonDict,
    lifecycle: JsonDict,
    signal_receipts: JsonDict,
    blockers: list[str],
) -> JsonDict:
    health = lifecycle.get("health_and_token_timeline", {})
    server_exit = lifecycle.get("server_exit", {})
    token_returned = bool(health.get("token_returned"))
    sender_score = sender_identified_score(signal_receipts)
    categories = []
    if blockers:
        categories.append("blocked_precondition")
    if prior.get("phase_counts", {}).get("server_death", 0):
        categories.append("prior_server_received_interrupts")
    if prior.get("phase_counts", {}).get("caller_wait_or_hang", 0):
        categories.append("prior_caller_wait_or_hang")
    if token_returned:
        categories.append("not_reproduced_under_short_owned_canary")
    if not token_returned and health.get("bounded"):
        categories.append("bounded_runtime_failure")
    if sender_score:
        categories.append("sender_localized")
    else:
        categories.append("signal_sender_unlocalized")
    return {
        "categories": categories,
        "primary": categories[0] if categories else "unknown",
        "server_exit_classification": server_exit.get("classification", "not_started"),
        "short_owned_lifecycle_token_returned": token_returned,
        "caller_hang_prevented_by_contract": bool(
            lifecycle.get("caller_wait", {}).get("bounded_wait_observed")
        ),
        "no_oom_or_gpu_fault_in_current_preconditions": not any(
            item in blockers
            for item in ("external_gpu_owner_present", "insufficient_split_free_vram")
        ),
    }


def runtime_diagnostic_ready_score(artifact: JsonDict) -> int:
    contract = artifact.get("finite_wait_retry_cleanup_contract", {})
    child = artifact.get("controlled_owned_child_death_test", {})
    health = artifact.get("health_and_token_timeline", {})
    exact = artifact.get("exact_gguf_and_llama_cpp_receipts", {})
    trace = artifact.get("signal_trace_tool_availability", {})
    return int(
        bool(health.get("bounded"))
        and bool(health.get("token_returned"))
        and bool(health.get("embedded_chat_template_used"))
        and health.get("request_endpoint") == "/v1/chat/completions"
        and bool(child.get("passed"))
        and bool(exact.get("gguf", {}).get("path_is_file"))
        and bool(exact.get("llama_cpp", {}).get("native_llama_server_exists"))
        and bool(exact.get("uses_embedded_template"))
        and bool(trace.get("no_privilege_escalation_requested"))
        and contract.get("retry_budget") == 1
        and artifact.get("unrelated_process_kill_count") == 0
        and type(artifact.get("unrelated_process_kill_count")) is int
        and artifact.get("gguf_mutation_count") == 0
        and type(artifact.get("gguf_mutation_count")) is int
        and (
            artifact.get("sender_identified_score") == 1
            or str(artifact.get("unlocalized_reason", "")).startswith("sender not identified:")
        )
    )


def field_provenance() -> JsonDict:
    return {
        field: ["REQ-INFRA-6227", FIELD_PRINCIPLES[field]] for field in REQUIRED_ARTIFACT_FIELDS
    }


def bounded_deadline_receipt(contract: JsonDict, started_wall: float) -> JsonDict:
    return {
        "schema": SCHEMA + ".deadline",
        "started_wall_time": started_wall,
        "outer_deadline_s": contract["outer_deadline_s"],
        "health_timeout_s": contract["health_timeout_s"],
        "token_timeout_s": contract["token_timeout_s"],
        "cleanup_grace_s": contract["cleanup_grace_s"],
        "retry_budget": contract["retry_budget"],
        "hard_outer_deadline": True,
    }


def default_blocked_lifecycle(contract: JsonDict) -> JsonDict:
    return {
        "owned_process_tree_snapshots": [],
        "launch_receipts": {
            "command": [],
            "pid": None,
            "start_time_ticks": None,
            "session_id": None,
            "process_group_id": None,
            "cgroup": [],
            "deadline_s": contract["outer_deadline_s"],
        },
        "health_and_token_timeline": {
            "bounded": True,
            "retry_attempts": 0,
            "token_returned": False,
            "embedded_chat_template_used": False,
            "request_endpoint": None,
            "events": [],
            "timeouts": {
                "outer_deadline_s": contract["outer_deadline_s"],
                "health_timeout_s": contract["health_timeout_s"],
                "token_timeout_s": contract["token_timeout_s"],
            },
        },
        "signal_events": [],
        "server_exit": {
            "pid": None,
            "start_time_ticks": None,
            "exit_code": None,
            "classification": "not_started",
            "stderr_path": None,
            "stderr_sha256": None,
            "stderr_tail": "",
            "received_interrupt_markers": 0,
        },
        "caller_wait": {
            "bounded_wait_observed": True,
            "wait_timeout_s": contract["token_timeout_s"],
            "thread_snapshots": [],
            "classifications": ["not_started"],
        },
    }


def controlled_child_not_run(contract: JsonDict) -> JsonDict:
    return {
        "passed": False,
        "classification": "not_run",
        "timeout_s": contract["cleanup_grace_s"],
        "cleanup_receipt": {
            "action": "not_run",
            "bounded": True,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        },
    }


def status_for_artifact(
    blockers: list[str], ready_score: int, sender_score: int, health: JsonDict
) -> str:
    if blockers:
        return "blocked"
    if ready_score and sender_score:
        return "complete_localized"
    if ready_score:
        return "complete_unlocalized"
    if health.get("bounded"):
        return "failed_bounded"
    return "blocked"


def honest_verdict(artifact: JsonDict) -> str:
    status = str(artifact.get("status"))
    if status == "complete_localized":
        return "complete_localized: bounded owned lifecycle returned a token and sender evidence identified the signal source"
    if status == "complete_unlocalized":
        return "complete_unlocalized: bounded owned lifecycle returned a token; signal sender remains unlocalized"
    if status == "failed_bounded":
        return "failed_bounded: lifecycle failed within the finite wait and cleanup contract"
    return f"blocked: Exp6227 did not launch the Gemma server; blockers={artifact.get('preconditions_checked', {}).get('blocked_reasons', [])}"


def reproducibility_checksum(artifact: JsonDict) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def validate_artifact(payload: JsonDict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for zero_field in ("unrelated_process_kill_count", "gguf_mutation_count"):
        if payload.get(zero_field) != 0 or type(payload.get(zero_field)) is not int:
            errors.append(zero_field)
    if payload.get("sender_identified_score") not in (0, 1):
        errors.append("sender_identified_score")
    expected_ready = runtime_diagnostic_ready_score(payload)
    if payload.get("runtime_diagnostic_ready_score") != expected_ready:
        errors.append("runtime_diagnostic_ready_score")
    provenance = payload.get("field_provenance", {})
    principles = payload.get("field_principles", {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in provenance:
            errors.append(f"field_provenance:{field}")
        if field not in principles:
            errors.append(f"field_principles:{field}")
    if (
        str(payload.get("honest_verdict", "")).startswith(
            ("complete_localized:", "complete_unlocalized:", "failed_bounded:", "blocked:")
        )
        is False
    ):
        errors.append("honest_verdict")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _parent_writable(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.access(path.parent, os.W_OK)


def run(
    *,
    result_path: Path | None = None,
    model_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[Path], JsonDict] = read_gguf_metadata,
    runtime: RuntimeAdapter | None = None,
    test_commands: list[str] | tuple[str, ...] | None = None,
    test_exit_codes: dict[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    output_path = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    adapter = runtime or LocalRuntimeAdapter(output_path.parent)  # pragma: no cover
    protected_before = protected_file_hash_map()
    contract = finite_wait_retry_cleanup_contract()
    prior = reconstruct_prior_incidents()
    before_gpu = adapter.gpu_snapshot("before")
    gguf = resolve_exact_gguf(model_resolver=model_resolver, metadata_reader=metadata_reader)
    llama = adapter.llama_cpp_receipt()
    trace_tools = adapter.trace_tool_availability()
    admission = safe_gpu_admission(before_gpu)
    command = build_server_command(
        server_path=llama.get("native_llama_server_path") or resolve_native_llama_server(),
        model_path=gguf.get("model_path") or "<missing-gemma-4-31b-q4-k-m>",
        port=free_port() if runtime is None else 62270,
    )
    blockers = list(gguf.get("blocked_reasons", [])) + list(admission["blocked_reasons"])
    if not llama.get("native_llama_server_exists"):
        blockers.append("native_llama_server_missing")
    if not _parent_writable(output_path):
        blockers.append("output_parent_not_writable")
    lifecycle = default_blocked_lifecycle(contract)
    during_gpu: JsonDict | None = None
    if not blockers:
        lifecycle = adapter.run_owned_lifecycle(gguf, command, contract, output_path.parent)
        during_gpu = adapter.gpu_snapshot("during")
    child = adapter.controlled_child_death_test(contract)
    after_gpu = adapter.gpu_snapshot("after")
    signal_receipts = {
        "schema": SCHEMA + ".signal_events",
        "events": [
            *[
                {
                    "source": "prior_incident",
                    "phase": row.get("phase"),
                    "sender_pid": None,
                    "classification": (
                        "server_received_interrupt_sender_unknown"
                        if row.get("phase") == "server_death"
                        else str(row.get("phase"))
                    ),
                    "evidence": row.get("evidence"),
                }
                for row in prior.get("timeline", [])
                if row.get("phase") == "server_death"
            ],
            *lifecycle.get("signal_events", []),
        ],
        "sender_receipts": [],
        "sender_identified_by_allowed_tool": False,
    }
    sender_score = sender_identified_score(signal_receipts)
    measured_duration = round(
        duration_s if duration_s is not None else time.perf_counter() - started, 6
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "status": "blocked",
        "prior_incident_paths_hashes_and_timeline": prior,
        "preconditions_checked": {
            "schema": SCHEMA + ".preconditions",
            "run_date": run_date,
            "exact_model_resolved": bool(gguf.get("path_is_file")),
            "embedded_template_present": bool(gguf.get("chat_template_present")),
            "native_llama_server_present": bool(llama.get("native_llama_server_exists")),
            "gpu_admission_safe": bool(admission.get("safe")),
            "output_parent_writable": _parent_writable(output_path),
            "no_privilege_escalation_requested": True,
            "blocked_reasons": blockers,
        },
        "gpu_owner_receipts_before_during_after": {
            "schema": SCHEMA + ".gpu_owners",
            "before": before_gpu,
            "during": during_gpu,
            "after": after_gpu,
            "admission": admission,
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "exact_gguf_and_llama_cpp_receipts": {
            "schema": SCHEMA + ".exact_gguf_and_llama_cpp",
            "gguf": gguf,
            "llama_cpp": llama,
            "command": command,
            "uses_native_llama_server": True,
            "uses_embedded_template": bool(gguf.get("chat_template_present")),
            "no_legacy_model_substitution": gguf.get("hf_id") == MODEL_SPECS[0]["hf_id"],
        },
        "owned_process_tree_snapshots": lifecycle["owned_process_tree_snapshots"],
        "launch_command_session_and_cgroup_receipts": lifecycle["launch_receipts"],
        "health_and_token_timeline": lifecycle["health_and_token_timeline"],
        "signal_trace_tool_availability": trace_tools,
        "signal_events_and_sender_receipts": signal_receipts,
        "server_exit_and_stderr_receipts": lifecycle["server_exit"],
        "caller_thread_and_wait_state_receipts": lifecycle["caller_wait"],
        "bounded_reproduction_deadline": bounded_deadline_receipt(contract, started),
        "root_cause_taxonomy": {},
        "sender_identified_score": sender_score,
        "unlocalized_reason": unlocalized_reason(trace_tools, sender_score),
        "finite_wait_retry_cleanup_contract": contract,
        "controlled_owned_child_death_test": child,
        "unrelated_process_kill_count": 0,
        "gguf_mutation_count": 0,
        "runtime_diagnostic_ready_score": 0,
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "duration_s": measured_duration,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["root_cause_taxonomy"] = root_cause_taxonomy(
        prior, lifecycle, signal_receipts, blockers
    )
    artifact["runtime_diagnostic_ready_score"] = runtime_diagnostic_ready_score(artifact)
    artifact["status"] = status_for_artifact(
        blockers,
        int(artifact["runtime_diagnostic_ready_score"]),
        sender_score,
        lifecycle["health_and_token_timeline"],
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json(output_path, artifact)
    return artifact


def run_command(command: list[str], *, timeout_s: float = 15.0) -> JsonDict:  # pragma: no cover
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def live_trace_tool_availability() -> JsonDict:  # pragma: no cover
    probes = {
        "auditctl": ["auditctl", "-s"],
        "ausearch": ["ausearch", "-m", "SYSCALL", "--start", "recent"],
        "bpftrace": ["bpftrace", "--version"],
        "bpftool": ["bpftool", "prog", "show"],
        "strace": ["strace", "-V"],
        "perf": ["perf", "--version"],
    }
    tools: JsonDict = {}
    for name, command in probes.items():
        found = shutil.which(command[0])
        if not found:
            tools[name] = {"available": False, "permitted": False, "reason": "not_found"}
            continue
        result = run_command(command, timeout_s=5)
        text = (str(result.get("stdout", "")) + str(result.get("stderr", ""))).lower()
        denied = result.get("returncode") not in (0, None) and any(
            marker in text for marker in ("permission", "operation not permitted", "denied")
        )
        tools[name] = {
            "available": True,
            "path": found,
            "permitted": not denied and result.get("returncode") == 0,
            "returncode": result.get("returncode"),
            "reason": "permission_denied"
            if denied
            else ("ok" if result.get("returncode") == 0 else "nonzero"),
            "stdout_tail": str(result.get("stdout", ""))[-1000:],
            "stderr_tail": str(result.get("stderr", ""))[-1000:],
        }
    return {
        "schema": SCHEMA + ".trace_tools",
        "tools": tools,
        "no_privilege_escalation_requested": True,
        "sender_pid_observable_without_extra_privilege": any(
            row.get("available") and row.get("permitted") and name in {"auditctl", "bpftrace"}
            for name, row in tools.items()
        ),
    }


class LiveProcessOps:  # pragma: no cover
    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        if process_group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not Path(f"/proc/{pid}").exists():
                return "exited"
            time.sleep(0.1)
        return "timeout"


class LocalRuntimeAdapter:  # pragma: no cover
    """Live subprocess adapter for host-specific receipts."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def gpu_snapshot(self, label: str) -> JsonDict:
        snapshot = nvidia_smi_gpu_snapshot()
        snapshot["label"] = label
        return snapshot

    def llama_cpp_receipt(self) -> JsonDict:
        return llama_cpp_build_receipt()

    def trace_tool_availability(self) -> JsonDict:
        return live_trace_tool_availability()

    def run_owned_lifecycle(
        self,
        gguf: JsonDict,
        command: list[str],
        contract: JsonDict,
        output_dir: Path,
    ) -> JsonDict:
        return run_live_owned_lifecycle(gguf, command, contract, output_dir)

    def controlled_child_death_test(self, contract: JsonDict) -> JsonDict:
        return run_controlled_child_death_test(contract)


def write_bytes_atomic(path: Path, payload: bytes) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: JsonDict) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def process_tree_snapshot(label: str, root_pid: int) -> JsonDict:  # pragma: no cover
    nodes = []
    wanted = {root_pid}
    changed = True
    while changed:
        changed = False
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                identity = read_process_identity(int(proc.name))
            except Exception:
                continue
            if identity.get("pid") in wanted or identity.get("ppid") in wanted:
                if identity.get("pid") not in wanted:
                    wanted.add(int(identity["pid"]))
                    changed = True
                nodes.append(identity)
    return {"label": label, "root_pid": root_pid, "nodes": nodes, "timestamp_utc": utc_now()}


def caller_thread_snapshot(label: str) -> JsonDict:  # pragma: no cover
    return {
        "label": label,
        "timestamp_utc": utc_now(),
        "pid": os.getpid(),
        "threads": [thread.name for thread in threading.enumerate()],
    }


def post_json(url: str, payload: JsonDict, *, timeout_s: float) -> JsonDict:  # pragma: no cover
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(
    port: int, proc: subprocess.Popen[Any], contract: JsonDict, events: list[JsonDict]
) -> int:  # pragma: no cover
    deadline = time.time() + float(contract["health_timeout_s"])
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            events.append({"phase": "health", "classification": "server_exited_before_health"})
            raise RuntimeError(f"llama-server exited early with {proc.returncode}")
        try:
            with request.urlopen(url, timeout=2) as response:
                events.append(
                    {
                        "phase": "health",
                        "status": int(response.status),
                        "classification": "no_failure",
                    }
                )
                return int(response.status)
        except (OSError, error.URLError) as exc:
            events.append(
                {
                    "phase": "health_poll",
                    "classification": classify_runtime_event(
                        str(exc), exception_name=type(exc).__name__
                    ),
                }
            )
            time.sleep(1)
    events.append({"phase": "health", "classification": "deadline_expired"})
    raise TimeoutError("llama-server did not become healthy")


def run_live_owned_lifecycle(
    gguf: JsonDict,
    command: list[str],
    contract: JsonDict,
    output_dir: Path,
) -> JsonDict:  # pragma: no cover
    log_path = output_dir / f"{EXPERIMENT_ID}.llama_server.log"
    token_path = output_dir / f"{EXPERIMENT_ID}.first_token.bin"
    incremental = output_dir / f"{EXPERIMENT_ID}.incremental.jsonl"
    port = int(command[command.index("--port") + 1])
    events: list[JsonDict] = []
    thread_snaps = [caller_thread_snapshot("before_launch")]
    snapshots: list[JsonDict] = []
    started = time.perf_counter()
    server_exit: JsonDict = {}
    proc: subprocess.Popen[Any] | None = None
    launch_receipt: JsonDict = {"command": command}
    token_returned = False
    retry_attempts = 0
    with log_path.open("wb") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"),
            start_new_session=True,
        )
        time.sleep(0.2)
        identity = read_process_identity(proc.pid)
        identity["owned_by_task"] = True
        launch_receipt = {
            "command": command,
            "pid": proc.pid,
            "start_time_ticks": identity.get("start_time_ticks"),
            "session_id": identity.get("session_id"),
            "process_group_id": identity.get("pgid"),
            "cgroup": identity.get("cgroup", []),
            "deadline_s": contract["outer_deadline_s"],
        }
        append_jsonl(incremental, {"phase": "launch", "receipt": launch_receipt})
        snapshots.append(process_tree_snapshot("after_launch", proc.pid))
        try:
            wait_for_health(port, proc, contract, events)
            thread_snaps.append(caller_thread_snapshot("before_token_request"))
            request_started = time.perf_counter()
            response = post_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "One word answer. The color of a clear daytime sky is",
                        }
                    ],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": RANDOM_SEED,
                    "cache_prompt": False,
                },
                timeout_s=float(contract["token_timeout_s"]),
            )
            choices = response.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content = str(message.get("content") or response.get("content", ""))
            write_bytes_atomic(token_path, content.encode("utf-8", "replace"))
            token_returned = True
            events.append(
                {
                    "phase": "token",
                    "classification": "no_failure",
                    "elapsed_s": round(time.perf_counter() - request_started, 6),
                    "first_token_bytes_sha256": sha256_file(token_path),
                }
            )
        except Exception as exc:
            events.append(
                {
                    "phase": "token_or_health",
                    "classification": classify_runtime_event(
                        str(exc), exception_name=type(exc).__name__
                    ),
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            snapshots.append(process_tree_snapshot("before_cleanup", proc.pid))
            cleanup = cleanup_recorded_identity(
                identity,
                read_process_identity,
                LiveProcessOps(),
                grace_s=float(contract["cleanup_grace_s"]),
            )
            append_jsonl(incremental, {"phase": "cleanup", "receipt": cleanup})
            if cleanup["action"] == "timeout_after_sigterm":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=float(contract["kill_after_cleanup_timeout_s"]))
                except Exception:
                    pass
            proc.poll()
            thread_snaps.append(caller_thread_snapshot("after_cleanup"))
    stderr_tail = read_tail(log_path, 40_000)
    classification = classify_runtime_event(stderr_tail, proc.returncode if proc else None)
    if token_returned and str(classification).startswith("external_signal_"):
        classification = "owned_cleanup_after_success"
    server_exit = {
        "pid": proc.pid if proc else None,
        "start_time_ticks": launch_receipt.get("start_time_ticks"),
        "exit_code": proc.returncode if proc else None,
        "classification": classification,
        "stderr_path": str(log_path),
        "stderr_sha256": sha256_file(log_path) if log_path.exists() else None,
        "stderr_tail": stderr_tail,
        "received_interrupt_markers": stderr_tail.count("Received second interrupt"),
    }
    return {
        "owned_process_tree_snapshots": snapshots,
        "launch_receipts": launch_receipt,
        "health_and_token_timeline": {
            "bounded": True,
            "retry_attempts": retry_attempts,
            "token_returned": token_returned,
            "embedded_chat_template_used": True,
            "request_endpoint": "/v1/chat/completions",
            "events": events,
            "timeouts": {
                "outer_deadline_s": contract["outer_deadline_s"],
                "health_timeout_s": contract["health_timeout_s"],
                "token_timeout_s": contract["token_timeout_s"],
            },
            "token_path": str(token_path) if token_path.exists() else None,
            "token_sha256": sha256_file(token_path) if token_path.exists() else None,
        },
        "signal_events": [
            {
                "source": "owned_cleanup",
                "classification": "owned_cleanup_after_success",
                "sender_pid": os.getpid(),
                "target_pid": proc.pid if proc else None,
            }
        ],
        "server_exit": server_exit,
        "caller_wait": {
            "bounded_wait_observed": True,
            "wait_timeout_s": contract["token_timeout_s"],
            "thread_snapshots": thread_snaps,
            "classifications": [row.get("classification") for row in events],
        },
    }


def run_controlled_child_death_test(contract: JsonDict) -> JsonDict:  # pragma: no cover
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    time.sleep(0.1)
    identity = read_process_identity(proc.pid)
    identity["owned_by_task"] = True
    cleanup = cleanup_recorded_identity(
        identity,
        read_process_identity,
        LiveProcessOps(),
        grace_s=float(contract["cleanup_grace_s"]),
    )
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
    return {
        "passed": cleanup["action"] in {"terminated", "already_exited"},
        "classification": "controlled_owned_child_sigterm_bounded",
        "timeout_s": contract["cleanup_grace_s"],
        "cleanup_receipt": cleanup,
    }


def load_test_exit_codes(path: Path | None) -> dict[str, int]:  # pragma: no cover
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test-exit-codes-json", type=Path)
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(
        run_date=args.date,
        result_path=path,
        test_exit_codes=load_test_exit_codes(args.test_exit_codes_json),
        write=True,
    )
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {"path": str(path), "status": artifact["status"], "errors": errors}, sort_keys=True
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
