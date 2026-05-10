"""Tests for Exp 1704 KV260 Potts Rust/PyO3 binding wiring."""

from __future__ import annotations

import json
from pathlib import Path


def test_exp1704_artifact_records_binding_schema_fields() -> None:
    """REQ-POTTS-008-5: artifact records Rust/PyO3 binding schema fields."""
    artifact = json.loads(Path("results/experiment_1704_kv260.json").read_text())

    assert artifact["schema"] == "kv260_potts_pyo3_binding_v1"
    assert artifact["kv260_potts_binding_ready"] is True
    assert artifact["pyo3_binding_ready"] is True
    assert artifact["rust_binding_path"] == "crates/carnot-webgpu-gateway/src/kv260_bindings.rs"
    assert artifact["python_binding_name"] == "RustKv260PottsSampler"
    assert artifact["driver_interface"] == "generic-uio mmap register window"
    assert artifact["register_map"]["status_done_mask"] == "0x00000004"
    assert "REQ-POTTS-008-3" in artifact["spec_traces"]
    assert "REQ-POTTS-008-4" in artifact["spec_traces"]
    assert "REQ-POTTS-008-5" in artifact["spec_traces"]


def test_kv260_rust_binding_source_has_required_register_contract() -> None:
    """REQ-POTTS-008-3: Rust source exposes the KV260 Potts register contract."""
    source = Path("crates/carnot-webgpu-gateway/src/kv260_bindings.rs").read_text()

    assert "pub const ADDR_CONTROL: u32 = 0x0000;" in source
    assert "pub const ADDR_STATUS: u32 = 0x0004;" in source
    assert "pub const ADDR_SPOUT_BASE: u32 = 0xA010;" in source
    assert "pub const STATUS_DONE_MASK: u32 = 1 << 2;" in source
    assert "pub trait Kv260RegisterIo" in source
    assert "pub struct UioKv260Driver" in source
    assert "sample_with_io" in source
    assert "validate_artifact_json" in source


def test_carnot_python_exposes_kv260_pyo3_binding() -> None:
    """REQ-POTTS-008-4: carnot-python wires the KV260 sampler into PyO3."""
    lib_rs = Path("crates/carnot-python/src/lib.rs").read_text()
    kv260_rs = Path("crates/carnot-python/src/kv260.rs").read_text()
    cargo_toml = Path("crates/carnot-python/Cargo.toml").read_text()
    compat = Path("python/carnot/_rust_compat.py").read_text()

    assert "mod kv260;" in lib_rs
    assert "kv260::register_kv260_module(m)?;" in lib_rs
    assert "carnot-webgpu-gateway" in cargo_toml
    assert "#[pyclass(name = \"RustKv260PottsSampler\")]" in kv260_rs
    assert "register_kv260_module" in kv260_rs
    assert "RustKv260PottsSampler" in compat
