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
        RustAdaptiveStateKernel,
        RustBoltzmannModel,
        RustGibbsModel,
        RustHMCSampler,
        RustIsingModel,
        RustKv260PottsSampler,
        RustLangevinSampler,
        RustModeJumpConfig,
        RustModeJumpCore,
        RustModeJumpState,
        RustModeJumpStateMetadata,
        RustOneAxisTemperingConfig,
        RustOneAxisTemperingCore,
        RustOneAxisTemperingState,
        RustSafetyNetFeatureRequest,
        RustSafetyNetRouter,
        RustSafetyNetRoutingDecision,
        RustS2KANLayer,
        RustVerificationResult,
        RustVerifyPipeline,
        safety_net_route_bytes,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustAdaptiveStateKernel = None  # type: ignore[assignment,misc]
    RustIsingModel = None  # type: ignore[assignment,misc]
    RustGibbsModel = None  # type: ignore[assignment,misc]
    RustBoltzmannModel = None  # type: ignore[assignment,misc]
    RustKv260PottsSampler = None  # type: ignore[assignment,misc]
    RustLangevinSampler = None  # type: ignore[assignment,misc]
    RustHMCSampler = None  # type: ignore[assignment,misc]
    RustModeJumpConfig = None  # type: ignore[assignment,misc]
    RustModeJumpCore = None  # type: ignore[assignment,misc]
    RustModeJumpState = None  # type: ignore[assignment,misc]
    RustModeJumpStateMetadata = None  # type: ignore[assignment,misc]
    RustOneAxisTemperingConfig = None  # type: ignore[assignment,misc]
    RustOneAxisTemperingCore = None  # type: ignore[assignment,misc]
    RustOneAxisTemperingState = None  # type: ignore[assignment,misc]
    RustSafetyNetFeatureRequest = None  # type: ignore[assignment,misc]
    RustSafetyNetRouter = None  # type: ignore[assignment,misc]
    RustSafetyNetRoutingDecision = None  # type: ignore[assignment,misc]
    RustS2KANLayer = None  # type: ignore[assignment,misc]
    RustVerifyPipeline = None  # type: ignore[assignment,misc]
    RustVerificationResult = None  # type: ignore[assignment,misc]
    safety_net_route_bytes = None  # type: ignore[assignment,misc]

__all__ = [
    "RUST_AVAILABLE",
    "RustAdaptiveStateKernel",
    "RustBoltzmannModel",
    "RustGibbsModel",
    "RustHMCSampler",
    "RustIsingModel",
    "RustKv260PottsSampler",
    "RustLangevinSampler",
    "RustModeJumpConfig",
    "RustModeJumpCore",
    "RustModeJumpState",
    "RustModeJumpStateMetadata",
    "RustOneAxisTemperingConfig",
    "RustOneAxisTemperingCore",
    "RustOneAxisTemperingState",
    "RustSafetyNetFeatureRequest",
    "RustSafetyNetRouter",
    "RustSafetyNetRoutingDecision",
    "RustS2KANLayer",
    "RustVerificationResult",
    "RustVerifyPipeline",
    "safety_net_route_bytes",
]
