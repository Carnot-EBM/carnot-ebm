"""Owner-scoped lifecycle support for one local llama.cpp process.

Spec refs: REQ-INFERENCE-6850 and SCENARIO-INFERENCE-6850-PROCESS-IDENTITY,
SCENARIO-INFERENCE-6850-PORT-AND-ORPHAN, and
SCENARIO-INFERENCE-6850-TEARDOWN.

PID numbers can be reused after a process exits. Cleanup therefore compares the
Linux start time and the full launch identity before it sends a signal. This
keeps an unrelated server safe even when it later receives the same PID.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any, Protocol
from urllib import error, request

from carnot.inference.llama_server_supervisor import (
    LiveProcessOps,
    command_hash,
    read_process_identity,
)


JsonDict = dict[str, Any]
SCHEMA = "carnot.inference.llama_cpp_process.v1"


class ProcessOps(Protocol):
    """Small signal boundary that makes destructive behavior testable."""

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        """Send one signal to the recorded PID or its private process group."""

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        """Wait for a bounded exit and return a stable status string."""


def ownership_token_digest(token: str) -> str:
    """Hash the secret token so public receipts never reveal cleanup authority."""

    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def process_contract(*, cleanup_grace_s: float = 30.0, kill_timeout_s: float = 10.0) -> JsonDict:
    """Return the finite cleanup contract shared by live code and tests."""

    return {
        "schema": SCHEMA + ".contract",
        "cleanup_grace_s": float(cleanup_grace_s),
        "kill_timeout_s": float(kill_timeout_s),
        "identity_fields": [
            "pid",
            "start_time_ticks",
            "uid",
            "command_hash",
            "process_group_id",
            "owner_pid",
            "owner_start_time_ticks",
            "ownership_token_digest",
        ],
        "cleanup_scope": "matching_owned_pid_or_private_process_group_only",
    }


def ownership_errors(
    recorded: Mapping[str, Any], current: Mapping[str, Any], *, token: str
) -> list[str]:
    """Name every identity mismatch before cleanup gets signal authority."""

    errors: list[str] = []
    if recorded.get("owned_by_task") is not True:
        errors.append("owned_by_task")
    if ownership_token_digest(token) != recorded.get("ownership_token_digest"):
        errors.append("ownership_token_digest")
    for field in ("pid", "start_time_ticks", "uid", "command_hash", "process_group_id"):
        if current.get(field) != recorded.get(field):
            errors.append(field)
    parent = current.get("parent_identity")
    parent = parent if isinstance(parent, Mapping) else {}
    if parent.get("pid") != recorded.get("owner_pid"):
        errors.append("owner_pid")
    if parent.get("start_time_ticks") != recorded.get("owner_start_time_ticks"):
        errors.append("owner_start_time_ticks")
    return errors


def cleanup_owned_process(
    recorded: Mapping[str, Any],
    *,
    token: str,
    current_identity: Callable[[int], JsonDict],
    process_ops: ProcessOps,
    port_probe: Callable[[int], bool],
    contract: Mapping[str, Any],
) -> JsonDict:
    """Stop only a matching process and confirm that its port was released."""

    pid = int(recorded.get("pid", -1))
    port = int(recorded.get("port", -1))
    current = current_identity(pid)
    if current.get("exists") is not True:
        port_released = bool(port_probe(port))
        return {
            "action": "already_exited",
            "ownership_verified": False,
            "ownership_errors": [],
            "process_exit_confirmed": True,
            "port_release_confirmed": port_released,
            "signals_sent": [],
            "bounded": True,
            "leak_free": port_released,
            "unrelated_process_kill_count_delta": 0,
        }

    mismatches = ownership_errors(recorded, current, token=token)
    if mismatches:
        return {
            "action": "refused",
            "ownership_verified": False,
            "ownership_errors": mismatches,
            "process_exit_confirmed": False,
            "port_release_confirmed": bool(port_probe(port)),
            "signals_sent": [],
            "bounded": True,
            "leak_free": False,
            "unrelated_process_kill_count_delta": 0,
        }

    private_group = int(recorded.get("process_group_id", -1)) == pid
    process_ops.send_signal(pid, signal.SIGTERM, process_group=private_group)
    signals_sent = [{"target": "process_group" if private_group else "pid", "signal": "SIGTERM"}]
    wait_status = process_ops.wait_for_exit(pid, float(contract["cleanup_grace_s"]))
    action = "terminated"
    if wait_status not in {"exited", "exited_zombie"}:
        process_ops.send_signal(pid, signal.SIGKILL, process_group=private_group)
        signals_sent.append(
            {"target": "process_group" if private_group else "pid", "signal": "SIGKILL"}
        )
        wait_status = process_ops.wait_for_exit(pid, float(contract["kill_timeout_s"]))
        action = "force_killed" if wait_status in {"exited", "exited_zombie"} else "cleanup_leak"
    process_exited = wait_status in {"exited", "exited_zombie"}
    port_released = bool(port_probe(port))
    return {
        "action": action,
        "ownership_verified": True,
        "ownership_errors": [],
        "process_exit_confirmed": process_exited,
        "port_release_confirmed": port_released,
        "signals_sent": signals_sent,
        "bounded": True,
        "leak_free": process_exited and port_released,
        "wait_status": wait_status,
        "unrelated_process_kill_count_delta": 0,
    }


def prepare_owned_port(
    port: int,
    *,
    orphan_receipt: Mapping[str, Any] | None,
    token: str | None,
    current_identity: Callable[[int], JsonDict],
    process_ops: ProcessOps,
    port_probe: Callable[[int], bool],
    contract: Mapping[str, Any],
) -> JsonDict:
    """Admit a free port or reclaim a listener with full ownership proof."""

    if port_probe(int(port)):
        return {
            "ready": True,
            "port": int(port),
            "owned_orphan_recovered": False,
            "signals_sent": [],
        }
    if orphan_receipt is None or token is None:
        return {
            "ready": False,
            "port": int(port),
            "reason": "occupied_port_unowned",
            "owned_orphan_recovered": False,
            "signals_sent": [],
        }
    cleanup = cleanup_owned_process(
        orphan_receipt,
        token=token,
        current_identity=current_identity,
        process_ops=process_ops,
        port_probe=port_probe,
        contract=contract,
    )
    ready = cleanup.get("leak_free") is True and cleanup.get("port_release_confirmed") is True
    return {
        "ready": ready,
        "port": int(port),
        "reason": None if ready else "occupied_port_orphan_not_reclaimable",
        "owned_orphan_recovered": ready,
        "cleanup": cleanup,
        "signals_sent": list(cleanup.get("signals_sent") or []),
    }


def port_is_free(port: int) -> bool:  # pragma: no cover - host state varies.
    """Return true only when a loopback listener can bind the exact port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # A closed HTTP connection can leave a harmless TIME_WAIT socket. Reuse
        # it for this ownership probe while a live listener still fails bind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def write_owner_state(
    path: Path, receipt: Mapping[str, Any], token: str
) -> None:  # pragma: no cover
    """Store crash-recovery authority in a private file outside the artifact."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"receipt": dict(receipt), "ownership_token": token}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def read_owner_state(path: Path) -> JsonDict | None:  # pragma: no cover
    """Read private orphan state, or return none when no prior owner exists."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


class OwnedLlamaCppProcess:  # pragma: no cover - exercised by the live experiment.
    """Launch and own one scoring worker under a private process group."""

    def __init__(
        self,
        *,
        command: list[str],
        port: int,
        env: Mapping[str, str],
        log_path: Path,
        state_path: Path,
    ) -> None:
        self.command = list(command)
        self.port = int(port)
        self.env = dict(env)
        self.log_path = log_path
        self.state_path = state_path
        self.token = secrets.token_urlsafe(32)
        self.process: subprocess.Popen[Any] | None = None
        self.receipt: JsonDict | None = None
        self._log_handle: Any = None

    def launch(self) -> JsonDict:
        """Start the worker and capture its immutable Linux identity."""

        if not port_is_free(self.port):
            raise RuntimeError(f"occupied_port:{self.port}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        owner = read_process_identity(os.getpid())
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=self.env,
            start_new_session=True,
        )
        time.sleep(0.1)
        identity = read_process_identity(self.process.pid)
        if identity.get("exists") is not True:
            raise RuntimeError("worker_identity_missing")
        self.receipt = {
            **identity,
            "command": list(self.command),
            "command_hash": command_hash(self.command),
            "owner_pid": owner.get("pid"),
            "owner_start_time_ticks": owner.get("start_time_ticks"),
            "ownership_token_digest": ownership_token_digest(self.token),
            "owned_by_task": True,
            "port": self.port,
        }
        write_owner_state(self.state_path, self.receipt, self.token)
        return dict(self.receipt)

    def wait_for_health(self, timeout_s: float) -> JsonDict:
        """Wait for the worker after model load, bounded by the supplied limit."""

        deadline = time.monotonic() + float(timeout_s)
        samples: list[JsonDict] = []
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                return {"ok": False, "reason": "worker_exited", "samples": samples}
            try:
                with request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as response:
                    samples.append({"status": response.status})
                    return {"ok": response.status == 200, "samples": samples}
            except (OSError, error.URLError) as exc:
                samples.append({"status": None, "error": type(exc).__name__})
                time.sleep(1)
        return {"ok": False, "reason": "health_timeout", "samples": samples}

    def post_json(self, path: str, payload: Mapping[str, Any], timeout_s: float) -> JsonDict:
        """Send one bounded JSON request to the owned loopback worker."""

        req = request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=float(timeout_s)) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, Mapping):
            raise RuntimeError("worker_response_not_object")
        return dict(value)

    def cleanup(self) -> JsonDict:
        """Apply owner-scoped teardown and remove private state after success."""

        if self.receipt is None:
            receipt = {
                "action": "not_started",
                "ownership_verified": False,
                "process_exit_confirmed": self.process is None or self.process.poll() is not None,
                "port_release_confirmed": port_is_free(self.port),
                "leak_free": self.process is None or self.process.poll() is not None,
                "unrelated_process_kill_count_delta": 0,
            }
        else:
            receipt = cleanup_owned_process(
                self.receipt,
                token=self.token,
                current_identity=read_process_identity,
                process_ops=LiveProcessOps(),
                port_probe=port_is_free,
                contract=process_contract(),
            )
        if self.process is not None and receipt.get("process_exit_confirmed") is True:
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                receipt["process_exit_confirmed"] = False
        receipt["process_reaped"] = self.process is None or self.process.poll() is not None
        receipt["port_release_confirmed"] = port_is_free(self.port)
        receipt["leak_free"] = bool(
            receipt.get("process_exit_confirmed")
            and receipt["process_reaped"]
            and receipt["port_release_confirmed"]
        )
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if receipt.get("leak_free") is True:
            self.state_path.unlink(missing_ok=True)
        return receipt
