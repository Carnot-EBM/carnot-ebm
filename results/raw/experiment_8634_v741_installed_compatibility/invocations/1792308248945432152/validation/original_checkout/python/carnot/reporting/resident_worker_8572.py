"""REQ-VERIFY-8572: measured pipes retain the shipped arithmetic and errors."""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.reporting.complete_native_worker_8559 import batch
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.resident_worker_8571 import (
    envelope as envelope,
    read_frame as read_frame,
    write_frame as write_frame,
)
from carnot.verify import nine_input_8533 as numeric
from carnot.verify.native_nine_input_8533 import load

Json = dict[str, Any]
MAX_REQUESTS = 1025


def configuration(path: Path, digest: str, arm: str) -> tuple[Json, Any, Json | None]:
    """Read immutable configuration once and identify the extension actually imported."""
    if arm not in {"python", "native"} or sha256_file(path) != digest:
        raise ValueError("immutable_config")
    config = json.loads(path.read_bytes())
    native, loaded = None, None
    print(f"[exp8572] phase=before_configuration arm={arm}", file=sys.stderr, flush=True)
    if arm == "native":
        ref = config["extension"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("extension_hash")
        with redirect_stdout(sys.stderr):
            native = load(Path(ref["path"]))
        loaded = dict(path=native.__file__, sha256=sha256_file(Path(native.__file__)))
        if loaded != {k: ref[k] for k in ["path", "sha256"]}:
            raise ValueError("loaded_extension")
    print(f"[exp8572] phase=after_configuration arm={arm}", file=sys.stderr, flush=True)
    return config, native, loaded


def evaluate(head: Json, row: Json, native: Any) -> tuple[Json, list[Json]]:
    """Text requests use the complete adapter; finite controls keep original scaled operands."""
    if row.get("kind") != "numerical":
        if (
            type(row["q"]) not in {int, float}
            or not math.isfinite(row["q"])
            or not 0 <= row["q"] <= 1
        ):
            raise ValueError("probability_input")
        responses, spans = batch(head, [row], native)
        return json.loads(responses[0]["response"]), spans
    start = time.monotonic_ns()
    scaled: Any = np.asarray([row["scaled_features"]], dtype=float)
    theta = numeric.sparse.parameters(head) * row["decay_multiplier"]
    residual = np.asarray(
        numeric.sparse.basis(scaled) @ theta
        if native is None
        else native.nine_input_residual_8533(theta.tolist(), scaled.tolist()),
        dtype=float,
    )
    end = time.monotonic_ns()
    low, high = np.asarray(head["scaler"]["minimum"]), np.asarray(head["scaler"]["maximum"])
    raw = low + scaled * np.where(high > low, high - low, 1.0)
    probability = float(numeric.normalize(head, raw, residual)[0])
    fallback = native is not None and min(abs(probability - 0.2), abs(probability - 0.8)) <= 1e-10
    if fallback:
        reference = numeric.sparse.basis(scaled) @ theta
        probability = float(numeric.normalize(head, raw, reference)[0])
    response = dict(
        unit_id=row["unit_id"],
        probability=probability,
        residual=float(residual[0]),
        decision=numeric.action(probability),
        boundary_fallback=fallback,
    )
    return response, [
        dict(operation="binding_residual", start_ns=start, end_ns=end),
        dict(
            operation="normalization_and_python_fallback", start_ns=end, end_ns=time.monotonic_ns()
        ),
    ]


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
        response, spans = evaluate(config["head"], value["row"], native)
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
                    f"[exp8572] completed={index + 1} pending={MAX_REQUESTS - index - 1}",
                    file=sys.stderr,
                    flush=True,
                )
            if response["status"] == "shutdown":
                return 0
        raise ValueError("request_limit")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, ImportError) as error:
        print(f"[exp8572] phase=worker_error error={error}", file=sys.stderr, flush=True)
        return 1


def panel(path: Path, digest: str, arm: str, output: Path) -> int:
    """One process evaluates a complete panel without using a resident pipe."""
    try:
        config, native, loaded = configuration(path, digest, arm)
        ready = time.monotonic_ns()
        rows = []
        for index, row in enumerate(config["rows"]):
            response, spans = evaluate(config["head"], row, native)
            rows.append(dict(response=response, spans=spans))
            if index % 100 == 0:
                print(
                    f"[exp8572] phase=panel completed={index + 1} pending={len(config['rows']) - index - 1}",
                    flush=True,
                )
        atomic_json(output, dict(rows=rows, ready_ns=ready, loaded_extension=loaded))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return 1
