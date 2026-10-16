"""REQ-REPORT-8559: sequential clocks include public text and the full response.

The adapter uses the shipped dense bases and policy. Each operation owns a
separate span so a future hardware reader cannot count overlapping work twice.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from carnot.reporting import nine_input_qualification_8558 as qualified
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import nine_input_8533 as numeric

Json = dict[str, Any]


def batch(head: Json, rows: list[Json], native: Any = None) -> tuple[list[Json], list[Json]]:
    """Recompute every text feature before numerical work; cached vectors are unused."""
    began = time.monotonic_ns()
    spans: list[Json] = []

    def mark(name: str) -> None:
        """Adjacent boundaries make all operation spans disjoint by construction."""
        end = time.monotonic_ns()
        spans.append(
            dict(operation=name, start_ns=spans[-1]["end_ns"] if spans else began, end_ns=end)
        )

    features = [numeric.extract(r["public"])["values"] for r in rows]
    if any(x is None or r["q"] is None for r, x in zip(rows, features, strict=True)):
        raise ValueError("request_features")
    raw = np.asarray([[r["q"], *x] for r, x in zip(rows, features, strict=True)], dtype=float)
    mark("public_text_features")
    scaled, _ = numeric.sparse.scale(raw, head["scaler"])
    mark("scaling")
    q = np.clip(raw[:, 0], 1e-4, 1 - 1e-4)
    offset = np.log(q / (1 - q))
    mark("offset")
    theta = numeric.sparse.parameters(head)
    residual = np.asarray(
        numeric.sparse.basis(scaled) @ theta
        if native is None
        else native.nine_input_residual_8533(theta.tolist(), scaled.tolist()),
        dtype=float,
    )
    mark("binding_residual")
    z = (offset + residual) / head["temperature"]
    energies = np.column_stack((np.zeros(len(raw)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    probabilities = weights[:, 1] / weights.sum(axis=1)
    fallback: Any = np.zeros(len(rows), dtype=bool)
    if native is not None:
        fallback = (np.abs(probabilities - 0.2) <= 1e-10) | (np.abs(probabilities - 0.8) <= 1e-10)
        if np.any(fallback):
            probabilities[fallback] = numeric.sparse.predict(head, raw[fallback])
    mark("normalization_and_python_fallback")
    decisions = [numeric.action(float(p)) for p in probabilities]
    mark("typed_decision")
    results = []
    for row, p, decision, used in zip(rows, probabilities, decisions, fallback, strict=True):
        response = json.dumps(
            dict(
                unit_id=row["unit_id"],
                probability=float(p),
                decision=decision,
                boundary_fallback=bool(used),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        results.append(
            dict(
                unit_id=row["unit_id"],
                response=response,
                start_ns=began,
                end_ns=time.monotonic_ns(),
            )
        )
    mark("serialization")
    for result in results:
        result["latency_ns"] = result["end_ns"] - began
    return results, spans


def execute(bundle: Json, arm: str, size: int) -> Json:
    """The parent starts the complete clock before this fresh interpreter exists."""
    if arm not in {"python", "native"} or size not in {1, 8}:
        raise ValueError("arm_batch")
    start = time.monotonic_ns()
    native = None
    loaded = None
    if arm == "native":
        extension = bundle["extension"]
        if sha256_file(Path(extension["path"])) != extension["sha256"]:
            raise ValueError("extension_hash")
        native = qualified.legacy.build.load(Path(extension["path"]))
        loaded = dict(path=native.__file__, sha256=sha256_file(Path(native.__file__)))
        if loaded != {k: extension[k] for k in ["path", "sha256"]}:
            raise ValueError("loaded_extension_identity")
    head, requests = bundle["head"], bundle["requests"]
    rows: list[Json] = []
    spans: list[Json] = []
    ready = time.monotonic_ns()
    for index in range(0, len(requests), size):
        responses, operations = batch(head, requests[index : index + size], native)
        rows.extend(responses)
        spans.extend(dict(s, batch_index=index // size) for s in operations)
        if (index + size) % 16 == 0:
            print(
                f"[exp8559] phase=worker arm={arm} completed={len(rows)} pending={len(requests) - len(rows)}",
                flush=True,
            )
    end = time.monotonic_ns()
    return dict(
        start_ns=start,
        ready_ns=ready,
        end_ns=end,
        requests=rows,
        spans=spans,
        arm=arm,
        batch=size,
        loaded_extension=loaded,
        pid=os.getpid(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def worker(manifest: Path, output: Path, arm: str, size: int) -> int:
    """Evidence persistence happens after the final response clock and is excluded."""
    try:
        result = execute(json.loads(manifest.read_bytes()), arm, size)
        atomic_json(output, result)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return 1
