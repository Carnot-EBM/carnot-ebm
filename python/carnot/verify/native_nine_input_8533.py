"""REQ-VERIFY-8533: an offline isolated build loads the exact shipped Rust sources."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.v709_execution import child

ROOT = Path(__file__).resolve().parents[3]
SOURCES = [
    "crates/carnot-core/src/nine_input_8533.rs",
    "crates/carnot-python/src/nine_input_8533.rs",
    "crates/carnot-core/tests/nine_input_8533.rs",
]
FLAGS = "-C target-feature=-fma -C instrument-coverage"
Json = dict[str, Any]


def load(path: Path) -> Any:
    """Import a real extension; a Python mirror cannot supply this entry point."""
    print("[exp8533] phase=before_extension_load completed=0 pending=1", flush=True)
    os.environ.setdefault("LLVM_PROFILE_FILE", str(path.parent / "rust-%p-%m.profraw"))
    spec = importlib.util.spec_from_file_location("_nine_input_8533", path)
    if spec is None or spec.loader is None:
        raise ValueError("extension_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "nine_input_residual_8533"):
        raise ValueError("extension_entrypoint")
    print("[exp8533] phase=after_extension_load completed=1 pending=0", flush=True)
    return module


def extension(private: Path, logs: Path) -> tuple[Any, Json]:
    """Pin cached PyO3, preserve compiled bytes, and retain every bounded build receipt."""
    supplied = os.environ.get("CARNOT_8533_EXTENSION")
    receipts: list[Json] = []
    if supplied:
        path = Path(supplied)
    else:
        project = private / "native"
        for directory in ["core/src", "core/tests", "binding/src"]:
            (project / directory).mkdir(parents=True, exist_ok=True)
        (project / "Cargo.toml").write_text(
            '[workspace]\nresolver="2"\nmembers=["core","binding"]\n'
        )
        (project / "core/Cargo.toml").write_text(
            '[package]\nname="carnot-core"\nversion="0.1.0"\nedition="2021"\n'
        )
        (project / "core/src/lib.rs").write_text("pub mod nine_input_8533;\n")
        (project / "binding/Cargo.toml").write_text(
            '[package]\nname="carnot-nine-input-8533"\nversion="0.1.0"\nedition="2021"\n'
            '[lib]\nname="nine_input_8533"\ncrate-type=["cdylib"]\n[dependencies]\n'
            'carnot-core={path="../core"}\npyo3={version="=0.24.2",features=["extension-module"]}\n'
        )
        (project / "binding/src/lib.rs").write_text(
            "use pyo3::prelude::*;\nmod nine_input_8533;\n#[pymodule]\n"
            "fn _nine_input_8533(m: &Bound<'_, PyModule>) -> PyResult<()> {\n"
            "    nine_input_8533::register(m)\n}\n"
        )
        for source, target in zip(
            SOURCES,
            [
                "core/src/nine_input_8533.rs",
                "binding/src/nine_input_8533.rs",
                "core/tests/nine_input_8533.rs",
            ],
            strict=True,
        ):
            shutil.copyfile(ROOT / source, project / target)
        logs.mkdir(parents=True, exist_ok=True)
        prefix = [
            "env",
            "RUSTFLAGS=" + FLAGS,
            "LLVM_PROFILE_FILE=" + str(logs / "rust-%p-%m.profraw"),
            "PYO3_PYTHON=" + str(ROOT / ".venv/bin/python"),
            "TMPDIR=" + str(private),
        ]
        manifest = ["--manifest-path", str(project / "Cargo.toml")]
        commands = [
            ("compiler_version", ["rustc", "--version", "--verbose"]),
            ("cargo_version", ["cargo", "--version"]),
            ("lock", ["cargo", "generate-lockfile", *manifest, "--offline"]),
            ("fmt", ["cargo", "fmt", *manifest, "--all", "--", "--check"]),
            ("test", ["cargo", "test", *manifest, "-p", "carnot-core", "--offline", "--locked"]),
            (
                "clippy",
                [
                    "cargo",
                    "clippy",
                    *manifest,
                    "--workspace",
                    "--all-targets",
                    "--offline",
                    "--locked",
                    "--",
                    "-D",
                    "warnings",
                ],
            ),
            ("build", ["cargo", "build", *manifest, "--release", "--offline", "--locked"]),
        ]
        for name, argv in commands:
            receipts.append(child("rust_" + name, prefix + argv, logs, deadline=240))
        if not all(r["passed"] for r in receipts):
            raise ValueError("native_build_failed")
        saved = logs.parent / "native_binary"
        saved.mkdir(parents=True, exist_ok=True)
        path = saved / "libnine_input_8533.so"
        shutil.copyfile(project / "target/release/libnine_input_8533.so", path)
        shutil.copyfile(project / "Cargo.lock", saved / "Cargo.lock")
        for executable in (project / "target/debug/deps").glob("nine_input_8533-*"):
            if executable.is_file() and executable.suffix != ".d":
                shutil.copyfile(executable, saved / executable.name)
    module = load(path)
    return module, dict(
        path=str(path.absolute()),
        module_file=module.__file__,
        sha256=sha256_file(path),
        actual_loaded=True,
        receipts=receipts,
        compiler_flags=FLAGS,
        fast_math=False,
        fma_contraction=False,
        sources=[dict(path=str(ROOT / p), sha256=sha256_file(ROOT / p)) for p in SOURCES],
    )
