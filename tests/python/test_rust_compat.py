"""Tests for carnot._rust_compat module.

Covers both the success path (Rust extension available) and the fallback path
(ImportError when Rust extension is not installed).

REQ-CROSS-001: Rust bindings are optional; pure-Python fallback must work.
SCENARIO-CROSS-001: When Rust extension is absent, RUST_AVAILABLE is False
and all Rust class references are None.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest import mock


def test_rust_available_when_extension_present() -> None:
    """When carnot._rust is importable, RUST_AVAILABLE should be True."""
    # REQ-CROSS-001, SCENARIO-CROSS-001
    import carnot._rust_compat as compat

    # In this test environment the Rust extension is built, so it should be True
    assert compat.RUST_AVAILABLE is True
    assert compat.RustIsingModel is not None
    assert compat.RustGibbsModel is not None
    assert compat.RustBoltzmannModel is not None
    assert compat.RustLangevinSampler is not None
    assert compat.RustHMCSampler is not None


def test_rust_fallback_when_extension_missing() -> None:
    """When carnot._rust import fails, RUST_AVAILABLE is False and classes are None."""
    # REQ-CROSS-001, SCENARIO-CROSS-001
    # Remove cached module so reimport executes the module-level code
    mod_name = "carnot._rust_compat"
    saved = sys.modules.pop(mod_name, None)

    try:
        # Patch the import so that `from carnot._rust import ...` raises ImportError
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "carnot._rust":
                raise ImportError("no rust extension")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_fake_import):
            mod = importlib.import_module(mod_name)

        assert mod.RUST_AVAILABLE is False
        assert mod.RustIsingModel is None
        assert mod.RustGibbsModel is None
        assert mod.RustBoltzmannModel is None
        assert mod.RustLangevinSampler is None
        assert mod.RustHMCSampler is None
    finally:
        # Restore original module so other tests are not affected
        sys.modules.pop(mod_name, None)
        if saved is not None:
            sys.modules[mod_name] = saved
