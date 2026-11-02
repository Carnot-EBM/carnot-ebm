"""REQ-REPORT-8719: compare native integers with the unchanged lifecycle emulator."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
from collections.abc import Callable
from typing import Any

from carnot.verify import spline_precision_8692 as numeric
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTENSION = os.environ.get(
    "CARNOT_8719_EXTENSION", str(ROOT / "target/debug/libcarnot_python.so")
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so the supervising process can detect a stalled child."""
    print(f"[exp8719] phase={phase} completed={completed} pending={pending}", flush=True)


def load(path: Path) -> Any:
    """Import the exact binary that supplies the opt-in arithmetic entrypoint."""
    progress("extension_load_before")
    spec = importlib.util.spec_from_file_location("_rust", path)
    if spec is None or spec.loader is None:
        raise ValueError("extension_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "fixed_sparse_8719"):
        raise ValueError("native_entrypoint")
    progress("extension_load_after")
    return module


__all__ = ["numeric"]


class NativeFixed:
    """Keep host quantization counters while each composite integer operation runs in Rust."""

    def __init__(self, extension: Any):
        self.unit = 1 << 24
        initial = numeric.Fixed(32, 24)
        self.counts: dict[str, int] = initial.counts
        self.operations: dict[str, int] = initial.operations
        self.maximum: int = 0
        self.extension = extension
        self.calls: dict[str, int] = {}

    def q(self, value: float) -> int:
        """Host ingress retains the emulator's finite binary64 quantization rule."""
        if not math.isfinite(value):
            raise ValueError("finite_operand")
        rounded = round(value * self.unit)
        self.operations["saturation_checks"] += 1
        self.counts["saturation"] += int(not -(1 << 31) <= rounded < (1 << 31))
        return min((1 << 31) - 1, max(-(1 << 31), rounded))

    def call(self, operation: str, **operands: Any) -> Any:
        """Transfer explicit operands and telemetry; count every crossing of the binding."""
        self.calls[operation] = self.calls.get(operation, 0) + 1
        response = json.loads(
            self.extension.fixed_sparse_8719(
                operation,
                json.dumps(
                    dict(
                        f=dict(
                            counts=self.counts, operations=self.operations, maximum=self.maximum
                        ),
                        **operands,
                    )
                ),
            )
        )
        self.counts, self.operations, self.maximum = (
            response["f"]["counts"],
            response["f"]["operations"],
            response["f"]["maximum"],
        )
        if response["error"] is not None:
            error_class = OverflowError if response["error"] == "signed64" else ValueError
            raise error_class(response["error"])
        return response["output"]


def probability(state: Json, raw: list[float], f: NativeFixed) -> tuple[float, Json]:
    """Leave feature scaling and nonlinear probability transforms on the host."""
    if len(raw) != 9 or not all(math.isfinite(v) for v in raw):
        raise ValueError("input_contract")
    low, high = state["scaler"]["minimum"], state["scaler"]["maximum"]
    scaled = [
        min(1.0, max(0.0, (x - a) / (b - a if b > a else 1.0)))
        for x, a, b in zip(raw, low, high, strict=True)
    ]
    ids, weights = f.call(
        "basis",
        knots=[f.q(v) for v in numeric.original.sparse.KNOTS],
        scaled=[f.q(v) for v in scaled],
    )
    residual = (
        f.call(
            "residual",
            parameters=state["parameters"],
            decay_scale=state["decay_scale"],
            temperature=state["temperature"],
            ids=ids,
            weights=weights,
        )
        / f.unit
    )
    q = min(1 - 1e-4, max(1e-4, raw[0]))
    z = (math.log(q / (1 - q)) + residual) / (state["temperature"] / f.unit)
    p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    return p, dict(ids=ids, weights=weights, residual=residual)


def step(state: Json, raw: list[float], label: int, f: NativeFixed) -> tuple[Json, Json]:
    """Admit labels on the host before delegating the sparse and dense writes."""
    if type(label) is not int or label not in (0, 1):
        raise ValueError("binary_label")
    p, receipt = probability(state, raw, f)
    parameters, scale, writes = f.call(
        "update",
        parameters=state["parameters"],
        decay_scale=state["decay_scale"],
        temperature=state["temperature"],
        probability=round(p * f.unit),
        label=label,
        ids=receipt["ids"],
        weights=receipt["weights"],
    )
    result = dict(state, parameters=parameters, decay_scale=scale)
    receipt.update(probability=p, local_writes=len(receipt["ids"]), renormalization_writes=writes)
    return result, receipt


def serialize(state: Json, f: NativeFixed) -> bytes:
    """Native little-endian coefficient order precedes the original host scaler bytes."""
    packed = f.call(
        "serialize",
        parameters=state["parameters"],
        decay_scale=state["decay_scale"],
        temperature=state["temperature"],
    )
    return bytes(packed) + struct.pack(
        "<18d", *state["scaler"]["minimum"], *state["scaler"]["maximum"]
    )


def controls(extension: Any) -> list[Json]:
    """Compare signed ties, saturation and widened rejection with the Python rules."""
    rows = []
    for op, a, b in [("ratio", a, 2) for a in [-7, -5, -3, -1, 1, 3, 5, 7]] + [
        ("ratio", 1, 0),
        ("ratio", 1, -1),
        ("product", 2**62, 2),
        ("product", -(2**63), 1),
        ("product", 2**63 - 1, 1),
        ("sat", 2**31, 0),
        ("sat", -(2**31) - 1, 0),
        ("mul", 2**24, 2**24),
        ("div", 1, 0),
        ("div", 2**24, 2**24),
    ]:
        native, python = NativeFixed(extension), numeric.Fixed(32, 24)
        outcomes = []
        actions: list[Callable[[], Any]] = [
            lambda: native.call(op, a=a, b=b),
            lambda: getattr(python, op)(a) if op == "sat" else getattr(python, op)(a, b),
        ]
        for action in actions:
            try:
                outcomes.append(dict(value=action(), error=None))
            except (OverflowError, ValueError) as error:
                outcomes.append(dict(value=None, error=type(error).__name__ + ":" + str(error)))
        rows.append(
            dict(
                operation=op,
                a=a,
                b=b,
                native=outcomes[0],
                python=outcomes[1],
                counters_identical=native.operations == python.operations
                and native.counts == python.counts
                and native.maximum == python.maximum,
                passed=outcomes[0] == outcomes[1]
                and native.operations == python.operations
                and native.counts == python.counts
                and native.maximum == python.maximum,
            )
        )
    host_controls: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        (
            "nan_ingress",
            lambda: NativeFixed(extension).q(float("nan")),
            lambda: numeric.Fixed(32, 24).q(float("nan")),
        ),
    ]
    for name, native_action, python_action in host_controls:
        errors: list[str | None] = []
        for action in [native_action, python_action]:
            try:
                action()
                errors.append(None)
            except ValueError as error:
                errors.append(str(error))
        rows.append(
            dict(
                operation=name,
                native_error=errors[0],
                python_error=errors[1],
                passed=errors[0] == errors[1] == "finite_operand",
            )
        )
    rejection_controls: list[tuple[str, Json]] = [
        ("basis", dict(knots=[0] * 16, scaled=[-1] * 9)),
        ("residual", dict(parameters=[0] * 109, decay_scale=1)),
        ("residual", dict(parameters=[0] * 109, temperature=1)),
        ("update", dict(parameters=[2**31] * 109)),
    ]
    for operation, operands in rejection_controls:
        try:
            NativeFixed(extension).call(operation, **operands)
            observed = None
        except ValueError as error:
            observed = str(error)
        rows.append(
            dict(
                operation="out_of_domain_" + operation,
                expected="supported_domain",
                observed=observed,
                passed=observed == "supported_domain",
                reference="qualified_exp8706_supported_domain_envelope",
            )
        )
    return rows


def drive(
    job: Json, folder: Path, start: int, stop: int, extension: Any, historical: Path | None = None
) -> Json:
    """Compare every committed update and persist only acknowledged native state."""
    folder.mkdir(parents=True, exist_ok=True)
    f, py = NativeFixed(extension), numeric.Fixed(32, 24)
    signature = canonical_hash(job)
    if start:
        checkpoint = json.loads((folder / "checkpoint.json").read_bytes())
        if checkpoint["next_index"] != start or checkpoint["job_hash"] != signature:
            raise ValueError("restart_identity")
        state, ack = checkpoint["state"], checkpoint["ack"]
        for engine in [f, py]:
            engine.counts, engine.operations, engine.maximum = (
                deepcopy(checkpoint["counts"]),
                deepcopy(checkpoint["operations"]),
                checkpoint["maximum"],
            )
        if serialize(state, f) != (folder / "checkpoint.bin").read_bytes():
            raise ValueError("restart_serialization")
    else:
        state = numeric.encode(job["head"], f)
        if job["boundary"]:
            state["decay_scale"] = f.unit // 256 + job["boundary_offset"]
        py.counts = deepcopy(f.counts)
        py.operations, py.maximum = deepcopy(f.operations), f.maximum
        ack = canonical_hash(dict(job_hash=signature, next_index=0, state=state))
    python = deepcopy(state)
    original = (
        [json.loads(line) for line in historical.read_text().splitlines()] if historical else []
    )
    rows_path = folder / f"rows-{start:04}-{stop:04}.jsonl"
    failed = 0
    progress("benchmark_before_" + folder.name, start, stop - start)
    with rows_path.open("w") as stream:
        for index in range(start, stop):
            raw = job["vectors"][index % len(job["vectors"])]["x"]
            label = job["labels"][index % len(job["labels"])]
            python, receipt_py = numeric.step(python, raw, label, py)
            probability_py, _ = numeric.probability(python, raw, py)
            state, receipt = step(state, raw, label, f)
            p, _ = probability(state, raw, f)
            ack = canonical_hash(dict(previous_ack=ack, next_index=index + 1, state=state))
            state_hash = canonical_hash(state)
            history_ok = not original or (
                original[index]["state_hash"] == state_hash
                and original[index]["ack"] == ack
                and original[index]["probability"] == p
                and original[index]["passed"]
            )
            passed = (
                state == python
                and f.counts == py.counts
                and f.operations == py.operations
                and f.maximum == py.maximum
                and p == probability_py
                and receipt["probability"] == receipt_py["probability"]
                and all(
                    receipt[k] == receipt_py[k]
                    for k in ["ids", "weights", "local_writes", "renormalization_writes"]
                )
                and history_ok
            )
            failed += int(not passed)
            row = dict(
                update_index=index + 1,
                state_hash=state_hash,
                ack=ack,
                probability=p,
                integer_state_identical=state == python,
                counters_identical=f.operations == py.operations and f.counts == py.counts,
                historical_identical=bool(history_ok),
                passed=bool(passed),
                local_writes=receipt["local_writes"],
                dense_writes=receipt["renormalization_writes"],
            )
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            if (index + 1) % 256 == 0 or index + 1 == stop:
                encoded = serialize(state, f)
                if encoded != numeric.serialize(python, py):
                    raise ValueError("serialized_order")
                checkpoint = dict(
                    job_hash=signature,
                    next_index=index + 1,
                    state=state,
                    ack=ack,
                    counts=f.counts,
                    operations=f.operations,
                    maximum=f.maximum,
                )
                with (folder / "checkpoint.bin").open("wb") as binary:
                    binary.write(encoded)
                    binary.flush()
                    os.fsync(binary.fileno())
                atomic_json(folder / "checkpoint.json", checkpoint)
                directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                os.fsync(directory)
                os.close(directory)
                stream.flush()
                os.fsync(stream.fileno())
                progress("checkpoint_" + folder.name, index + 1, stop - index - 1)
    result = dict(
        checkpoint=checkpoint,
        rows_path=str(rows_path),
        observed_updates=stop - start,
        failed_count=failed,
        passed=failed == 0,
        native_calls=f.calls,
        operation_partition_rows=[
            dict(
                operation=k,
                calls=v,
                coefficient_uploads=109 * v if k in ["residual", "update", "serialize"] else 0,
                coefficient_downloads=109 * v if k in ["update", "serialize"] else 0,
            )
            for k, v in sorted(f.calls.items())
        ],
        host_calls=dict(
            scaling=2 * (stop - start),
            sigmoid=2 * (stop - start),
            label_admission=stop - start,
            reference_snapshot_coefficient_copies=109 * (stop - start),
            durable_checkpoints=(stop - start + 255) // 256,
        ),
    )
    atomic_json(folder / f"summary-{start:04}-{stop:04}.json", result)
    progress("benchmark_after_" + folder.name, stop, 0)
    return result
