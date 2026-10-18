"""REQ-VERIFY-8379: native arithmetic shares Python's deployed decision boundary.

The private build contains the repository's exact core and binding sources.
It avoids importing unrelated native services into this finite parity check.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.v709_execution import child
from carnot.verify import continuous_local_learning_8348 as optimizer
from carnot.verify import local_update_isolation_8306 as kernel
from carnot.verify import spline_table_fidelity_8352 as evaluator
from carnot.verify.threshold_guard_8362 import validate

ROOT = Path(__file__).resolve().parents[3]
Json = dict[str, Any]
SOURCES = [
    "crates/carnot-core/src/direct_spline_8379.rs",
    "crates/carnot-python/src/direct_spline_8379.rs",
]


def extension(private: Path, logs: Path, supplied: str | None = None) -> tuple[Any, Json]:
    """A loaded binary hash binds observations to actual Rust execution."""
    supplied = supplied or os.environ.get("CARNOT_8379_EXTENSION")
    receipts = []
    if supplied:
        path = Path(supplied)
    else:
        project = private / "native"
        (project / "core/src").mkdir(parents=True, exist_ok=True)
        (project / "core/tests").mkdir()
        (project / "binding/src").mkdir(parents=True)
        (project / "Cargo.toml").write_text(
            '[workspace]\nresolver="2"\nmembers=["core","binding"]\n'
        )
        (project / "core/Cargo.toml").write_text(
            '[package]\nname="carnot-core"\nversion="0.1.0"\nedition="2021"\n'
        )
        (project / "core/src/lib.rs").write_text("pub mod direct_spline_8379;\n")
        (project / "binding/Cargo.toml").write_text(
            '[package]\nname="carnot-direct-8379"\nversion="0.1.0"\nedition="2021"\n'
            '[lib]\nname="carnot_python"\ncrate-type=["cdylib"]\n'
            '[dependencies]\ncarnot-core={path="../core"}\n'
            'pyo3={version="=0.24.2",features=["extension-module"]}\n'
        )
        (project / "binding/src/lib.rs").write_text(
            "use pyo3::prelude::*;\nmod direct_spline_8379;\n"
            "#[pymodule]\nfn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {\n"
            "    direct_spline_8379::register(m)\n}\n"
        )
        shutil.copyfile(ROOT / SOURCES[0], project / "core/src/direct_spline_8379.rs")
        shutil.copyfile(ROOT / SOURCES[1], project / "binding/src/direct_spline_8379.rs")
        shutil.copyfile(
            ROOT / "crates/carnot-core/tests/direct_spline_8379.rs",
            project / "core/tests/direct_spline_8379.rs",
        )
        manifest = str(project / "Cargo.toml")
        os.environ["PYO3_PYTHON"] = str(ROOT / ".venv/bin/python")
        os.environ["TMPDIR"] = str(private)
        os.environ["RUSTFLAGS"] = "-C target-feature=-fma -C instrument-coverage"
        os.environ["LLVM_PROFILE_FILE"] = str(logs / "rust-%p-%m.profraw")
        commands = [
            ("cargo_fmt", ["cargo", "fmt", "--manifest-path", manifest, "--all", "--", "--check"]),
            (
                "cargo_check",
                ["cargo", "check", "--manifest-path", manifest, "--workspace", "--offline"],
            ),
            (
                "cargo_test",
                ["cargo", "test", "--manifest-path", manifest, "-p", "carnot-core", "--offline"],
            ),
            (
                "cargo_clippy",
                [
                    "cargo",
                    "clippy",
                    "--manifest-path",
                    manifest,
                    "--workspace",
                    "--all-targets",
                    "--offline",
                    "--",
                    "-D",
                    "warnings",
                ],
            ),
            (
                "cargo_build",
                ["cargo", "build", "--manifest-path", manifest, "--release", "--offline"],
            ),
        ]
        for name, argv in commands:
            receipts.append(child(name, argv, logs, deadline=360))
        if not all(r["passed"] for r in receipts):
            raise ValueError("native_build_failed")
        path = project / "target/release/libcarnot_python.so"
    print("[exp8379] before_actual_extension_import", flush=True)
    spec = importlib.util.spec_from_file_location("_rust", path)
    if spec is None or spec.loader is None:
        raise ValueError("extension_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not all(
        hasattr(module, key)
        for key in ("direct_design_8379", "direct_logits_8379", "direct_update_8379")
    ):
        raise ValueError("native_entrypoints")
    print("[exp8379] after_actual_extension_import", flush=True)
    return module, dict(
        path=str(path.absolute()),
        sha256=sha256_file(path),
        module_file=module.__file__,
        actual_loaded=True,
        receipts=receipts,
        fast_math=False,
        rustflags="-C target-feature=-fma -C instrument-coverage",
    )


def probabilities(head: Json, rows: list[list[float]], native: Any = None) -> NDArray[np.float64]:
    """Both arms call exactly the same SciPy ufunc and existing action thresholds."""
    validate(head)
    if not rows:
        raise ValueError("features")
    for row in rows:
        validate(head, np.asarray(row, dtype=np.float64))
    if native is None:
        return np.asarray(
            evaluator.direct(head, np.asarray(rows, dtype=np.float64)), dtype=np.float64
        )
    z = native.direct_logits_8379(head["coefficients"], rows, head["temperature"])
    return np.asarray(expit(np.asarray(z, dtype=np.float64)), dtype=np.float64)


def learn(head: Json, x: list[float], y: int, native: Any) -> list[float]:
    """Reuse the frozen Python optimizer residual before native local-only writes."""
    validate(head, np.asarray(x, dtype=np.float64))
    if type(y) is not int or y not in (0, 1):
        raise ValueError("label")
    residual = (float(optimizer.probability(head, x)) - y) / head["temperature"]
    if not math.isfinite(residual):
        raise ValueError("residual")
    result: list[float] = native.direct_update_8379(head["coefficients"], x, residual)
    return result


def coordinator(native: Any) -> Any:
    """Isolate numeric providers while reusing the unchanged Python durable writer."""
    from carnot.verify import direct_atomic_state_8376 as original

    spec = importlib.util.spec_from_file_location("_direct_coordinator_8379", original.__file__)
    if spec is None or spec.loader is None:
        raise ValueError("coordinator_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def direct(head: Json, x: NDArray[Any]) -> NDArray[np.float64]:
        return probabilities(head, x.tolist(), native)

    def update(head: Json, x: list[float] | None, y: int | None, arm: str) -> Json:
        result = optimizer.learn(head, x, y, arm)
        if x is not None and y is not None:
            c = learn(head, x, y, native)
            delta = [a - b for a, b in zip(c, head["coefficients"], strict=True)]
            result.update(
                coefficients=c,
                coefficient_delta=delta,
                changed=[i for i, v in enumerate(delta) if v],
            )
        return result

    module.evaluator = SimpleNamespace(direct=direct, kernel=kernel)
    module.optimizer = SimpleNamespace(learn=update)
    return module


def tracked(native: Any) -> tuple[Any, Json]:
    """Count logical numeric payload copies at each actual binding call."""
    counts: Json = dict(
        native_invocation_count=0,
        binding_copy_bytes=0,
        binding_conversion_bytes=0,
        native_call_counts={},
    )

    def values(value: Any) -> int:
        return sum(values(v) for v in value) if isinstance(value, list) else 1

    def invoke(name: str, *args: Any) -> Any:
        counts["native_invocation_count"] += 1
        counts["native_call_counts"][name] = counts["native_call_counts"].get(name, 0) + 1
        counts["binding_copy_bytes"] += 8 * sum(values(v) for v in args if isinstance(v, list))
        counts["binding_conversion_bytes"] += 8 * sum(values(v) for v in args)
        result = getattr(native, name)(*args)
        counts["binding_copy_bytes"] += 8 * values(result)
        counts["binding_conversion_bytes"] += 8 * values(result)
        return result

    wrapped = SimpleNamespace(
        direct_design_8379=lambda *a: invoke("direct_design_8379", *a),
        direct_logits_8379=lambda *a: invoke("direct_logits_8379", *a),
        direct_update_8379=lambda *a: invoke("direct_update_8379", *a),
    )
    return wrapped, counts
