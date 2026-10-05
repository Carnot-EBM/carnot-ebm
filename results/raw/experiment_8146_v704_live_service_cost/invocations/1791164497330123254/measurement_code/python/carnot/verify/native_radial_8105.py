"""REQ-VERIFY-8105: independent equations and a narrow native service boundary.

Only arithmetic and durable center state enter Rust. Canonical threshold
fallback keeps numerical rounding from changing typed service actions.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore[import-untyped]

from carnot import experiment_7230_v636_native_belief as build
from carnot.experiment_8027_v695_native_update_cost import copy_extension as copy_extension
from carnot.experiment_8021_v695_typed_decision_test import action
from carnot.reporting.current_work_receipt import sha256_file

ROOT = Path(__file__).resolve().parents[3]
Array = NDArray[np.float64]
Json = dict[str, Any]


def extension() -> tuple[Any, Json]:
    """Bind readiness to loaded bytes, rather than to successful compilation."""
    began = time.monotonic()
    supplied = os.environ.get("CARNOT_8105_EXTENSION")
    if supplied:
        path, receipt = Path(supplied), {}
    else:
        print("[exp8105] before native build", flush=True)
        path, receipt = build.build_native_extension(ROOT)
        destination = ROOT / "target/experiment-8105-load" / path.name
        copy_extension(path, destination)
        path = destination
        print("[exp8105] after native build", flush=True)
    build_s = time.monotonic() - began
    print("[exp8105] before binding load", flush=True)
    module = build.load_native_extension(path)
    if not hasattr(module, "RustRadial8105"):
        raise ValueError("missing_radial_entrypoint")
    receipt.update(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        actual_loaded=True,
        build_duration_s=build_s,
        module_file=module.__file__,
    )
    print("[exp8105] after binding load", flush=True)
    return module, receipt


def reference(state: Json, values: Any, labels: Any) -> tuple[Array, Array, Array]:
    """Direct broadcast float64 equations do not reuse the Rust distance loop."""
    g = state["geometry"]
    x = np.asarray(values, dtype=np.float64)
    z = (x - np.asarray(g["mean"])) / np.asarray(g["std"])
    centers = np.asarray([c["x"] for c in state["centers"]])
    with np.errstate(over="ignore"):
        squared = np.sum(((z[:, None, :] - centers[None, :, :]) / g["sigma"]) ** 2, axis=2)
    phi = np.column_stack((np.ones(len(x)), np.exp(-0.5 * squared)))
    theta = np.asarray(state["coefficients"])
    p = np.asarray(expit(phi @ theta), dtype=np.float64)
    gradient = np.asarray(phi.T @ (p - np.asarray(labels)) / len(x) + 0.01 * theta)
    return phi, p, gradient


def service(state: Json, values: Any, native: Any = None) -> tuple[Array, list[str], list[bool]]:
    """Both arms resolve close cost ties using the same canonical float64 result."""
    x = np.asarray(values, dtype=np.float64)
    _, canonical, _ = reference(state, x, np.zeros(len(x)))
    p = canonical.copy() if native is None else np.asarray(native.predict(x.tolist()))
    fallback = (np.minimum(abs(p - 0.1), abs(p - 0.5)) <= 1e-8) | (
        np.minimum(abs(canonical - 0.1), abs(canonical - 0.5)) <= 1e-8
    )
    p[fallback] = canonical[fallback]
    return p, [action(float(v)) for v in p], fallback.tolist()
