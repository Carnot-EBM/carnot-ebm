"""REQ-VERIFY-8608: the original bounded framed protocol with lean arithmetic.

Both arms enter this same lifecycle. Only the residual provider changes.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import selectors
import struct
import sys
import time
from typing import Any

import resident_energy as numeric

Json = dict[str, Any]
MAX_FRAME = 65536
MAX_REQUESTS = 1025


def sha256_file(path: Path) -> str:
    """Hash actual bytes before each request so configuration drift fails closed."""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(fd: int, timeout: float) -> bytes | None:
    """A fixed header and deadline bound allocation and incomplete-peer waiting."""
    deadline = time.monotonic() + timeout

    def exact(size: int, *, header: bool = False) -> bytes:
        data = b""
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while len(data) < size:
                if not selector.select(max(0, deadline - time.monotonic())):
                    raise TimeoutError("frame_deadline")
                part = os.read(fd, size - len(data))
                if not part:
                    if header and not data:
                        return b""
                    raise ValueError("truncated_frame")
                data += part
        return data

    header = exact(4, header=True)
    if not header:
        return None
    size = struct.unpack("!I", header)[0]
    if size > MAX_FRAME or size == 0:
        raise ValueError("oversized_or_empty_frame")
    return exact(size)


def write_frame(fd: int, value: Json, timeout: float = 15) -> None:
    """Small serial writes keep the queue at one request without response caching."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_FRAME:
        raise ValueError("oversized_response")
    frame = struct.pack("!I", len(payload)) + payload
    deadline = time.monotonic() + timeout
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_WRITE)
            while frame:
                if not selector.select(max(0, deadline - time.monotonic())):
                    raise TimeoutError("write_deadline")
                frame = frame[os.write(fd, frame[:4096]) :]
    finally:
        os.set_blocking(fd, blocking)


def envelope(identifier: str, digest: str, row: Json) -> Json:
    """Monotonic deadlines share the local host clock across the pipe boundary."""
    return dict(
        request_id=identifier,
        config_hash=digest,
        operation="infer",
        row=row,
        deadline_ns=time.monotonic_ns() + 15_000_000_000,
    )


def configuration(path: Path, digest: str, arm: str) -> tuple[Json, Any, Json | None]:
    """Read immutable configuration once and identify the extension actually imported."""
    if arm not in {"python", "native"} or sha256_file(path) != digest:
        raise ValueError("immutable_config")
    config = json.loads(path.read_bytes())
    native, loaded = None, None
    print(f"[exp8608] phase=before_configuration arm={arm}", file=sys.stderr, flush=True)
    if arm == "native":
        ref = config["extension"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("extension_hash")
        with redirect_stdout(sys.stderr):
            native = numeric.load_module(Path(ref["path"]), "_nine_input_8533")
            if not hasattr(native, "nine_input_residual_8533"):
                raise ValueError("extension_entrypoint")
        loaded = dict(path=native.__file__, sha256=sha256_file(Path(native.__file__)))
        if loaded != {k: ref[k] for k in ["path", "sha256"]}:
            raise ValueError("loaded_extension")
    print(f"[exp8608] phase=after_configuration arm={arm}", file=sys.stderr, flush=True)
    return config, native, loaded


def decide(value: Any, config: Json, digest: str, native: Any, seen: set[str]) -> Json:
    """Each rejection creates a fresh error with no successful payload attached."""
    identifier = value.get("request_id") if isinstance(value, dict) else None
    try:
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 64:
            raise ValueError("request_id")
        if identifier in seen:
            raise ValueError("duplicate_request_id")
        if value.get("config_hash") != digest:
            raise ValueError("config_hash")
        deadline = value.get("deadline_ns")
        if (
            type(deadline) is not int
            or not time.monotonic_ns() < deadline <= time.monotonic_ns() + 30_000_000_000
        ):
            raise ValueError("request_deadline")
        seen.add(identifier)
        if value.get("operation") == "shutdown":
            return dict(request_id=identifier, status="shutdown", config_hash=digest)
        if value.get("operation") != "infer":
            raise ValueError("operation")
        response, spans = numeric.evaluate(config["head"], value["row"], native)
        if time.monotonic_ns() >= deadline:
            raise ValueError("evaluation_deadline")
        return dict(
            request_id=identifier, status="ok", config_hash=digest, response=response, spans=spans
        )
    except (ValueError, KeyError, TypeError) as error:
        return dict(
            request_id=identifier
            if isinstance(identifier, str) and len(identifier) <= 64
            else None,
            status="error",
            error=str(error),
            config_hash=digest,
        )


def serve(path: Path, digest: str, arm: str, timeout: float = 15) -> int:
    """A session has one in-flight response and bounded frame, idle and request limits."""
    try:
        config, native, loaded = configuration(path, digest, arm)
        write_frame(
            1,
            dict(
                status="ready",
                pid=os.getpid(),
                config_hash=digest,
                loaded_extension=loaded,
                config_loads=1,
            ),
        )
        seen: set[str] = set()
        for index in range(MAX_REQUESTS):
            payload = read_frame(0, timeout)
            if payload is None:
                return 0
            if sha256_file(path) != digest:
                raise ValueError("configuration_changed")
            try:
                value = json.loads(payload)
            except (ValueError, UnicodeError):
                value = None
            response = decide(value, config, digest, native, seen)
            write_frame(1, response, timeout)
            if (index + 1) % 100 == 0:
                print(
                    f"[exp8608] completed={index + 1} pending={MAX_REQUESTS - index - 1}",
                    file=sys.stderr,
                    flush=True,
                )
            if response["status"] == "shutdown":
                return 0
        raise ValueError("request_limit")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ImportError) as error:
        print(f"[exp8608] phase=worker_error error={error}", file=sys.stderr, flush=True)
        return 1
