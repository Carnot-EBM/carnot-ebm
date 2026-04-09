"""Rust binding availability flag.

When the optional ``carnot._rust`` extension (built via maturin from the
``carnot-python`` crate) is installed, ``RUST_AVAILABLE`` is True and the
native Rust model/sampler classes are re-exported here for convenience.

When Rust bindings are absent (pure-Python install), ``RUST_AVAILABLE`` is
False and all Rust class references are None.  Callers should check
``RUST_AVAILABLE`` before attempting to use any ``Rust*`` class.
"""

from __future__ import annotations

RUST_AVAILABLE: bool
"""True when the compiled Rust extension ``carnot._rust`` is importable."""

try:
    from carnot._rust import (  # type: ignore[import-not-found]
        RustBoltzmannModel,
        RustGibbsModel,
        RustHMCSampler,
        RustIsingModel,
        RustLangevinSampler,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustIsingModel = None  # type: ignore[assignment,misc]
    RustGibbsModel = None  # type: ignore[assignment,misc]
    RustBoltzmannModel = None  # type: ignore[assignment,misc]
    RustLangevinSampler = None  # type: ignore[assignment,misc]
    RustHMCSampler = None  # type: ignore[assignment,misc]

__all__ = [
    "RUST_AVAILABLE",
    "RustBoltzmannModel",
    "RustGibbsModel",
    "RustHMCSampler",
    "RustIsingModel",
    "RustLangevinSampler",
]
