"""REQ-REPORT-8731: use resident arithmetic with the original lifecycle driver."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carnot.reporting.v748_collection_contract import namespace
from carnot.verify import compiled_sparse_8719 as original

Json = dict[str, Any]
numeric = original.numeric
DEFAULT_EXTENSION = original.DEFAULT_EXTENSION
engine = namespace(original, "resident_driver_8731")
instances: list[ResidentFixed] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual work counts keep native calls and child execution visible."""
    print(f"[exp8731] phase={phase} completed={completed} pending={pending}", flush=True)


def load(path: Path) -> Any:
    """The same binary must expose both original and resident entrypoints."""
    os.environ.setdefault("LLVM_PROFILE_FILE", "/var/tmp/resident8731-%p-%m.profraw")
    module = original.load(path)
    if not hasattr(module, "ResidentSparse8731"):
        raise ValueError("resident_entrypoint")
    return module


class ResidentFixed:
    """Only qualification diagnostics copy coefficients back after an update."""

    def __init__(self, extension: Any):
        initial = original.NativeFixed(extension)
        self.unit = initial.unit
        self.counts: dict[str, int] = initial.counts
        self.operations: dict[str, int] = initial.operations
        self.maximum: int = initial.maximum
        self.extension = extension
        self.calls: dict[str, int] = {}
        self.resident: Any = None
        self.transfers: dict[str, Json] = {}
        self.cached_residual = 0
        self.token = 0
        instances.append(self)

    def record(self, op: str, sent: int, received: int, copies: int = 0, upload: int = 0) -> None:
        """Count logical typed payload bytes, excluding Python allocator overhead."""
        row = self.transfers.setdefault(
            op,
            dict(
                operation=op,
                calls=0,
                input_bytes=0,
                output_bytes=0,
                extra_copy_bytes=0,
                coefficient_uploads=0,
            ),
        )
        for key, value in dict(
            calls=1,
            input_bytes=sent,
            output_bytes=received,
            extra_copy_bytes=copies,
            coefficient_uploads=upload,
        ).items():
            row[key] += value

    def bind(self, state: Json) -> None:
        """Import counters and coefficients once, including after a host checkpoint."""
        if self.resident is None:
            context = json.dumps(
                dict(counts=self.counts, operations=self.operations, maximum=self.maximum)
            )
            self.resident = self.extension.ResidentSparse8731(
                state["parameters"], state["decay_scale"], state["temperature"], context
            )
            self.record("import_checkpoint", 111 * 8 + len(context.encode()), 0, 109 * 8, 109)

    def q(self, value: float) -> int:
        """Keep finite checks and binary64 ties-to-even quantization on the host."""
        if self.resident is None:
            return int(original.NativeFixed.q(self, value))
        if not original.math.isfinite(value):
            raise ValueError("finite_operand")
        rounded = round(value * self.unit)
        clamped = min(2**31 - 1, max(-(2**31), rounded))
        self.resident.quantized(clamped, clamped != rounded)
        self.record("quantized", 5, 0)
        return int(clamped)

    def sync(self) -> None:
        """Counter reads are qualification diagnostics, outside the update boundary."""
        self.counts, self.operations, self.maximum = self.resident.telemetry(self.token)
        size = 8 + sum(
            len(k.encode()) + 8 for group in [self.counts, self.operations] for k in group
        )
        self.record("telemetry", 8, size, size)

    def call(self, operation: str, **operands: Any) -> Any:
        """The legacy driver requests snapshots solely to assert exact state parity."""
        self.calls[operation] = self.calls.get(operation, 0) + 1
        self.bind(operands) if self.resident is None else None
        if operation == "basis":
            ids, weights, self.cached_residual, self.token = self.resident.prepare(
                operands["knots"], operands["scaled"]
            )
            self.record("prepare", 25 * 8, 16 + len(ids) * 16, 25 * 8 + len(ids) * 16)
            return ids, weights
        if operation == "residual":
            return self.cached_residual
        if operation == "update":
            scale, _, writes = self.resident.apply(
                operands["probability"], operands["label"], self.token
            )
            self.record("apply", 24, 24, 109 * 8)
            parameters, _, self.counts, self.operations, self.maximum = self.resident.audit(
                self.token
            )
            self.record(
                "qualification_audit",
                8,
                110 * 8
                + 8
                + sum(len(k.encode()) + 8 for g in [self.counts, self.operations] for k in g),
                109 * 8,
            )
            return parameters, scale, writes
        if operation != "serialize":
            raise ValueError("resident_operation")
        result = self.resident.serialize(self.token)
        self.record("serialize_checkpoint", 8, len(result), len(result))
        return result


def probability(state: Json, raw: list[float], f: ResidentFixed) -> tuple[float, Json]:
    """Reuse host feature scaling, offset and logistic calculations unchanged."""
    f.bind(state)
    value: tuple[float, Json] = engine_probability(state, raw, f)
    f.sync()
    return value


def controls(extension: Any) -> list[Json]:
    """Resident error probes preserve both the Python outcome and its counters."""
    rows: list[Json] = original.controls(extension)
    for op, a, b in [
        ("ratio", -7, 2),
        ("ratio", -5, 2),
        ("ratio", 1, 0),
        ("product", 2**62, 2),
        ("sat", 2**31, 0),
    ]:
        f = ResidentFixed(extension)
        f.bind(
            numeric.encode(
                dict(parameters=[0.0] * 109, decay_scale=1.0, temperature=1.0, scaler={}), f
            )
        )
        py = numeric.Fixed(32, 24)
        py.counts, py.operations = f.counts.copy(), f.operations.copy()
        outcomes = []
        actions: list[Callable[[], Any]] = [
            lambda: f.resident.control(op, a, b),
            lambda: py.sat(a) if op == "sat" else getattr(py, op)(a, b),
        ]
        for action in actions:
            try:
                outcomes.append(dict(value=action(), error=None))
            except (ValueError, OverflowError) as error:
                outcomes.append(dict(value=None, error=type(error).__name__ + ":" + str(error)))
        f.sync()
        rows.append(
            dict(
                operation="resident_" + op,
                native=outcomes[0],
                python=outcomes[1],
                passed=outcomes[0] == outcomes[1]
                and f.counts == py.counts
                and f.operations == py.operations
                and f.maximum == py.maximum,
            )
        )
    return rows + feedback_controls(extension)


def feedback_controls(extension: Any) -> list[Json]:
    """A one-use preparation token rejects duplicate and superseded feedback."""
    f = ResidentFixed(extension)
    f.bind(
        numeric.encode(dict(parameters=[0.0] * 109, decay_scale=1.0, temperature=1.0, scaler={}), f)
    )
    rows = []
    _, _, _, current_token = f.resident.prepare([0] * 16, [0] * 9)
    f.resident.apply(0, 0, current_token)
    for kind, token, probability, label, expected in [
        ("duplicate", 1, 0, 0, "pending_probability"),
        ("late", 1, 0, 0, "late_probability"),
        ("rejected", 1, -1, 0, "supported_domain"),
    ]:
        if kind != "duplicate":
            _, _, _, current_token = f.resident.prepare([0] * 16, [0] * 9)
        before = f.resident.audit(current_token)
        observed = None
        try:
            f.resident.apply(probability, label, token)
        except ValueError as error:
            observed = str(error)
        rows.append(
            dict(
                control=kind,
                acknowledged=False,
                expected=expected,
                observed=observed,
                passed=observed == expected and f.resident.audit(current_token) == before,
            )
        )
    return rows


engine_probability = engine.probability
engine.NativeFixed, engine.probability, engine.progress = ResidentFixed, probability, progress


def drive(
    job: Json, folder: Path, start: int, stop: int, extension: Any, historical: Path | None = None
) -> Json:
    """The unchanged driver checks each state and owns all durable host writes."""
    instances.clear()
    value: Json = engine.drive(job, folder, start, stop, extension, historical)
    rows = list(instances[0].transfers.values())
    value.update(
        binding_call_rows=rows,
        transferred_byte_rows=rows,
        byte_accounting="logical typed field bytes; Python object allocator overhead unmeasured; audit copies are qualification-only",
    )
    original.atomic_json(folder / f"summary-{start:04}-{stop:04}.json", value)
    return value
