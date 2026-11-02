"""Reusable native llama-server lifecycle supervisor.

Spec refs: REQ-INFRA-6228,
SCENARIO-INFRA-6228-DEAD-PORT-SLOW-LOAD-AND-EARLY-EXIT-ARE-BOUNDED,
SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS,
SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS.

The conductor has its own orchestration rules. This module is smaller: it only
owns one native `llama-server` process at a time and records enough identity to
clean it up without touching unrelated work.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Protocol
from urllib import error, request


JsonDict = dict[str, Any]
SCHEMA = "carnot.inference.llama_server_supervisor.v1"


class ProcessOps(Protocol):
    """Small process boundary used by cleanup tests and live cleanup."""

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        """Send a signal to one process or to its process group."""

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        """Wait for bounded process exit and return a short status."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def base64_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def command_hash(command: list[str] | tuple[str, ...]) -> str:
    return sha256_text(canonical_json(list(command)))


def utc_now() -> str:  # pragma: no cover
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def supervisor_contract(
    *,
    outer_deadline_s: float = 900.0,
    health_timeout_s: float = 360.0,
    token_timeout_s: float = 120.0,
    cleanup_grace_s: float = 30.0,
    kill_after_cleanup_timeout_s: float = 10.0,
    retry_budget: int = 1,
    endurance_interval_s: float = 12.0,
    endurance_sample_count: int = 3,
) -> JsonDict:
    return {
        "schema": SCHEMA + ".contract",
        "outer_deadline_s": float(outer_deadline_s),
        "health_timeout_s": float(health_timeout_s),
        "token_timeout_s": float(token_timeout_s),
        "cleanup_grace_s": float(cleanup_grace_s),
        "kill_after_cleanup_timeout_s": float(kill_after_cleanup_timeout_s),
        "retry_budget": int(retry_budget),
        "retryable_failures": ["connection_refused", "early_exit", "server_died_mid_request"],
        "endurance_interval_s": float(endurance_interval_s),
        "endurance_sample_count": int(endurance_sample_count),
        "pid_reuse_guard": "pid_start_time_uid_command_hash_process_group_parent_identity",
        "cleanup_scope": "recorded_pid_or_process_group_only",
        "unrelated_process_policy": "refuse_cleanup_when_identity_or_owner_changes",
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
    if "retry budget exhausted" in lowered:
        return "retry_exhausted"
    if "connectionrefused" in lowered or "connection refused" in lowered:
        return "connection_refused"
    if "server_exited_before_health" in lowered or "exited early" in lowered:
        return "early_exit"
    if "remotedisconnected" in lowered or "remote end closed" in lowered:
        return "server_died_mid_request"
    if "timeout" in lowered or "timed out" in lowered:
        return "request_timeout"
    if exit_code is not None and exit_code < 0:
        return f"external_signal_{signal.Signals(-exit_code).name.lower()}"
    if exit_code == 0 and not text.strip() and not exception_name:
        return "no_failure"
    return "unclassified_runtime_failure"


def should_retry(classification: str, attempt_index: int, contract: JsonDict) -> bool:
    return (
        classification in set(contract.get("retryable_failures", []))
        and int(attempt_index) < int(contract.get("retry_budget", 0))
    )


def parse_proc_stat(text: str) -> JsonDict:
    open_paren = text.find("(")
    close_paren = text.rfind(")")
    rest = text[close_paren + 2 :].split()
    return {
        "pid": int(text[:open_paren].strip()),
        "comm": text[open_paren + 1 : close_paren],
        "state": rest[0],
        "ppid": int(rest[1]),
        "process_group_id": int(rest[2]),
        "session_id": int(rest[3]),
        "start_time_ticks": int(rest[19]),
    }


def read_process_identity(
    pid: int,
    proc_root: Path = Path("/proc"),
    *,
    include_parent: bool = True,
) -> JsonDict:  # pragma: no cover
    proc_dir = proc_root / str(pid)
    if not proc_dir.exists():
        return {"pid": pid, "exists": False}
    stat = parse_proc_stat((proc_dir / "stat").read_text(encoding="utf-8"))
    status_text = (proc_dir / "status").read_text(encoding="utf-8", errors="replace")
    uid = None
    for line in status_text.splitlines():
        if line.startswith("Uid:"):
            uid = int(line.split()[1])
            break
    raw_argv = (proc_dir / "cmdline").read_bytes().split(b"\x00")
    argv = [item.decode("utf-8", "replace") for item in raw_argv if item]
    parent_identity = None
    if include_parent and stat["ppid"] > 0:
        parent = read_process_identity(stat["ppid"], proc_root, include_parent=False)
        parent_identity = {
            "pid": parent.get("pid"),
            "start_time_ticks": parent.get("start_time_ticks"),
        }
    return {
        **stat,
        "exists": True,
        "uid": uid,
        "command": argv,
        "command_hash": command_hash(argv),
        "parent_identity": parent_identity,
        "owned_by_current_uid": uid == os.getuid(),
    }


def identity_matches(recorded: JsonDict, current: JsonDict) -> bool:
    recorded_parent = recorded.get("parent_identity") or {}
    current_parent = current.get("parent_identity") or {}
    return (
        bool(current.get("exists"))
        and int(current.get("pid", -1)) == int(recorded.get("pid", -2))
        and int(current.get("start_time_ticks", -1)) == int(recorded.get("start_time_ticks", -2))
        and int(current.get("uid", -1)) == int(recorded.get("uid", -2))
        and str(current.get("command_hash")) == str(recorded.get("command_hash"))
        and int(current.get("process_group_id", -1)) == int(recorded.get("process_group_id", -2))
        and int(current_parent.get("pid", -1)) == int(recorded_parent.get("pid", -2))
        and int(current_parent.get("start_time_ticks", -1))
        == int(recorded_parent.get("start_time_ticks", -2))
    )


def cleanup_recorded_identity(
    recorded: JsonDict,
    current_identity: Callable[[int], JsonDict],
    process_ops: ProcessOps,
    *,
    contract: JsonDict,
) -> JsonDict:
    pid = int(recorded.get("pid", -1))
    current = current_identity(pid)
    if not current.get("exists"):
        return {
            "action": "already_exited",
            "bounded": True,
            "leak_free": True,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        }
    if not bool(recorded.get("owned_by_task")):
        return {
            "action": "refused",
            "reason": "unowned_process",
            "bounded": True,
            "leak_free": False,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        }
    if not identity_matches(recorded, current):
        return {
            "action": "refused",
            "reason": "identity_mismatch",
            "bounded": True,
            "leak_free": False,
            "signals_sent": [],
            "unrelated_process_kill_count_delta": 0,
        }
    process_group = int(recorded.get("process_group_id", -1)) == pid
    process_ops.send_signal(pid, signal.SIGTERM, process_group=process_group)
    wait_status = process_ops.wait_for_exit(pid, float(contract["cleanup_grace_s"]))
    signals = [{"target": "process_group" if process_group else "pid", "signal": "SIGTERM"}]
    if wait_status in {"exited", "exited_zombie"}:
        return {
            "action": "terminated",
            "bounded": True,
            "leak_free": True,
            "wait_status": wait_status,
            "signals_sent": signals,
            "unrelated_process_kill_count_delta": 0,
        }
    process_ops.send_signal(pid, signal.SIGKILL, process_group=process_group)
    kill_wait = process_ops.wait_for_exit(pid, float(contract["kill_after_cleanup_timeout_s"]))
    signals.append({"target": "process_group" if process_group else "pid", "signal": "SIGKILL"})
    leak_free = kill_wait in {"exited", "exited_zombie"}
    return {
        "action": "force_killed" if leak_free else "cleanup_leak",
        "bounded": True,
        "leak_free": leak_free,
        "wait_status": wait_status,
        "kill_wait_status": kill_wait,
        "signals_sent": signals,
        "unrelated_process_kill_count_delta": 0,
    }


def parse_cuda_placement(
    family: str,
    hf_id: str,
    server_log: str,
    gpu_intervals: list[JsonDict],
    *,
    owned_pids: set[int],
) -> JsonDict:
    layer_match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", server_log)
    offloaded = int(layer_match.group(1)) if layer_match else 0
    total = int(layer_match.group(2)) if layer_match else 0
    lowered = server_log.lower()
    tensor_or_buffer = any(
        marker in lowered
        for marker in ("cuda0 buffer", "cuda1 buffer", "compute buffer", "ggml_cuda", "cuda found")
    )
    owned_vram = False
    matching_pids: list[int] = []
    for interval in gpu_intervals:
        for app in interval.get("compute_apps", []):
            pid = int(app.get("pid", -1))
            if bool(app.get("owned_by_task")) and pid in owned_pids and int(app.get("used_memory_mb", 0)) > 0:
                owned_vram = True
                matching_pids.append(pid)
    log_evidence = offloaded > 0 or tensor_or_buffer
    return {
        "schema": SCHEMA + ".cuda_placement",
        "family": family,
        "hf_id": hf_id,
        "cuda_layers_offloaded": offloaded,
        "total_layers": total,
        "cuda_tensor_or_buffer_confirmed": tensor_or_buffer,
        "log_cuda_evidence_present": log_evidence,
        "gpu_interval_owned_vram_confirmed": owned_vram,
        "owned_gpu_pids": sorted(set(matching_pids)),
        "cuda_placement_confirmed": log_evidence and owned_vram,
        "evidence": layer_match.group(0) if layer_match else ("cuda tensor/buffer marker" if tensor_or_buffer else ""),
    }


def raw_token_receipt(
    token_bytes: bytes,
    *,
    latency_s: float,
    path: str | Path,
    sample_index: int,
) -> JsonDict:
    return {
        "sample_index": int(sample_index),
        "raw_token_bytes_b64": base64_bytes(token_bytes),
        "raw_token_bytes_sha256": sha256_bytes(token_bytes),
        "latency_s": float(latency_s),
        "path": str(path),
    }


def summarize_repeated_tokens(samples: list[JsonDict], *, min_samples: int) -> JsonDict:
    hashes = [str(row.get("raw_token_bytes_sha256")) for row in samples]
    latencies = [float(row.get("latency_s", 0.0)) for row in samples]
    unique_hashes = sorted(set(hashes))
    return {
        "schema": SCHEMA + ".repeated_tokens",
        "samples": samples,
        "sample_count": len(samples),
        "unique_raw_token_hashes": unique_hashes,
        "deterministic_repeated_output": len(samples) >= int(min_samples) and len(unique_hashes) == 1,
        "min_latency_s": min(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
    }


def family_runtime_ready(receipt: JsonDict) -> bool:
    return (
        bool(receipt.get("ownership", {}).get("owned_process"))
        and bool(receipt.get("cuda", {}).get("cuda_placement_confirmed"))
        and bool(receipt.get("tokens", {}).get("deterministic_repeated_output"))
        and int(receipt.get("tokens", {}).get("sample_count", 0)) >= 3
        and bool(receipt.get("endurance", {}).get("passed"))
        and int(receipt.get("endurance", {}).get("health_sample_count", 0)) >= 3
        and bool(receipt.get("recovery", {}).get("controlled_failure_bounded"))
        and bool(receipt.get("recovery", {}).get("recovery_success"))
        and bool(receipt.get("leak_check", {}).get("leak_free"))
    )


class LiveProcessOps:  # pragma: no cover
    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        if process_group:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            proc_dir = Path(f"/proc/{pid}")
            if not proc_dir.exists():
                return "exited"
            try:
                if parse_proc_stat((proc_dir / "stat").read_text(encoding="utf-8"))["state"] == "Z":
                    return "exited_zombie"
            except Exception:
                pass
            time.sleep(0.1)
        return "timeout"


def write_bytes_atomic(path: Path, payload: bytes) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


class NativeLlamaServerSupervisor:  # pragma: no cover
    """Own one live native `llama-server` process under a finite contract."""

    def __init__(self, command: list[str], output_dir: Path, contract: JsonDict) -> None:
        self.command = command
        self.output_dir = output_dir
        self.contract = contract
        self.proc: subprocess.Popen[Any] | None = None
        self.identity: JsonDict | None = None
        self.log_path = output_dir / f"llama_server_{abs(hash(command_hash(command)))}.log"

    def launch(self) -> JsonDict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_handle = self.log_path.open("ab")
        self.proc = subprocess.Popen(
            self.command,
            cwd=Path.cwd(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")),
            start_new_session=True,
        )
        time.sleep(0.2)
        identity = read_process_identity(self.proc.pid)
        identity["owned_by_task"] = True
        identity["command"] = list(self.command)
        identity["command_hash"] = command_hash(self.command)
        self.identity = identity
        return identity

    def wait_for_health(self) -> JsonDict:
        port = int(self.command[self.command.index("--port") + 1])
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + float(self.contract["health_timeout_s"])
        samples = []
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                return {"ok": False, "samples": samples, "classification": "early_exit"}
            try:
                started = time.perf_counter()
                with request.urlopen(url, timeout=2) as response:
                    samples.append(
                        {
                            "status": int(response.status),
                            "elapsed_s": round(time.perf_counter() - started, 6),
                            "classification": "no_failure",
                        }
                    )
                    return {"ok": True, "samples": samples, "classification": "no_failure"}
            except (OSError, error.URLError) as exc:
                samples.append(
                    {
                        "status": None,
                        "classification": classify_runtime_event(
                            str(exc), exception_name=type(exc).__name__
                        ),
                    }
                )
                time.sleep(1)
        return {"ok": False, "samples": samples, "classification": "deadline_expired"}

    def request_token(self, payload: JsonDict, token_path: Path, sample_index: int) -> JsonDict:
        port = int(self.command[self.command.index("--port") + 1])
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with request.urlopen(req, timeout=float(self.contract["token_timeout_s"])) as response:
            body = json.loads(response.read().decode("utf-8"))
        choices = body.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = str(message.get("content") or body.get("content", ""))
        token_bytes = content.encode("utf-8", "replace")
        write_bytes_atomic(token_path, token_bytes)
        return raw_token_receipt(
            token_bytes,
            latency_s=round(time.perf_counter() - started, 6),
            path=token_path,
            sample_index=sample_index,
        )

    def cleanup(self) -> JsonDict:
        if not self.identity:
            return {"action": "not_started", "bounded": True, "leak_free": True}
        return cleanup_recorded_identity(
            self.identity,
            read_process_identity,
            LiveProcessOps(),
            contract=self.contract,
        )

    def stderr_tail(self, limit: int = 40_000) -> str:
        if not self.log_path.is_file():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")[-limit:]

