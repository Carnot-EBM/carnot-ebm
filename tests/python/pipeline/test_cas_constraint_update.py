"""Tests for CASConstraintUpdater — CAS bounded memory updates.

Spec: REQ-CAS-001, REQ-CAS-001-1, REQ-CAS-001-2, REQ-CAS-001-3,
      REQ-CAS-001-4, SCENARIO-CAS-001, SCENARIO-CAS-002
"""

from __future__ import annotations

import pytest

from carnot.pipeline.cas_constraint_update import CASConstraintUpdater
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fresh_library() -> ConstraintTemplateLibrary:
    """Return a ConstraintTemplateLibrary with built-in templates registered."""
    lib = ConstraintTemplateLibrary()
    lib.register_builtin_templates()
    return lib


# ---------------------------------------------------------------------------
# REQ-CAS-001: constructor validation
# ---------------------------------------------------------------------------


def test_init_valid_parameters() -> None:
    """CASConstraintUpdater accepts valid parameters.

    Spec: REQ-CAS-001
    """
    updater = CASConstraintUpdater(
        compress_factor=0.9,
        smooth_alpha=0.1,
        smooth_target=0.0,
        max_count=100.0,
    )
    assert updater.compress_factor == 0.9
    assert updater.smooth_alpha == 0.1
    assert updater.smooth_target == 0.0
    assert updater.max_count == 100.0


def test_init_rejects_compress_factor_out_of_range() -> None:
    """compress_factor outside (0, 1) raises ValueError.

    Spec: REQ-CAS-001-1
    """
    with pytest.raises(ValueError, match="compress_factor"):
        CASConstraintUpdater(compress_factor=0.0)
    with pytest.raises(ValueError, match="compress_factor"):
        CASConstraintUpdater(compress_factor=1.0)
    with pytest.raises(ValueError, match="compress_factor"):
        CASConstraintUpdater(compress_factor=1.5)


def test_init_rejects_smooth_alpha_out_of_range() -> None:
    """smooth_alpha outside [0, 1] raises ValueError.

    Spec: REQ-CAS-001-3
    """
    with pytest.raises(ValueError, match="smooth_alpha"):
        CASConstraintUpdater(smooth_alpha=-0.1)
    with pytest.raises(ValueError, match="smooth_alpha"):
        CASConstraintUpdater(smooth_alpha=1.1)


def test_init_rejects_nonpositive_max_count() -> None:
    """max_count <= 0 raises ValueError.

    Spec: REQ-CAS-001-3
    """
    with pytest.raises(ValueError, match="max_count"):
        CASConstraintUpdater(max_count=0.0)
    with pytest.raises(ValueError, match="max_count"):
        CASConstraintUpdater(max_count=-10.0)


# ---------------------------------------------------------------------------
# REQ-CAS-001-1: compress step
# ---------------------------------------------------------------------------


def test_compress_decays_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """compress() multiplies all observation counts by compress_factor.

    Spec: REQ-CAS-001-1
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 100
    lib._observations[("sign_check", "m1")] = 50

    updater = CASConstraintUpdater(compress_factor=0.8)
    updater.compress(lib)

    assert abs(lib._observations[("carry_check", "m1")] - 80.0) < 1e-9
    assert abs(lib._observations[("sign_check", "m1")] - 40.0) < 1e-9


def test_compress_repeated_decays_geometrically() -> None:
    """After k compress steps, count = initial * factor^k.

    Spec: REQ-CAS-001-1, SCENARIO-CAS-001
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 100

    updater = CASConstraintUpdater(compress_factor=0.9, max_count=200.0)
    for _ in range(10):
        updater.compress(lib)

    # 100 * 0.9^10 ≈ 34.87
    count = lib._observations[("carry_check", "m1")]
    assert count < 40.0, f"Expected count < 40 after 10 compress steps, got {count}"
    assert count > 30.0, f"Expected count > 30 after 10 compress steps, got {count}"


# ---------------------------------------------------------------------------
# REQ-CAS-001-2: add step
# ---------------------------------------------------------------------------


def test_add_incorporates_new_observations() -> None:
    """add() increases observation counts by the supplied values.

    Spec: REQ-CAS-001-2
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 3.0

    updater = CASConstraintUpdater()
    updater.add(lib, {("carry_check", "m1"): 4.0})

    assert abs(lib._observations[("carry_check", "m1")] - 7.0) < 1e-9


def test_add_creates_new_entry_when_absent() -> None:
    """add() creates a new observation entry when the key does not exist yet.

    Spec: REQ-CAS-001-2
    """
    lib = _fresh_library()
    updater = CASConstraintUpdater()
    updater.add(lib, {("sign_check", "modelX"): 6.0})

    assert ("sign_check", "modelX") in lib._observations
    assert abs(lib._observations[("sign_check", "modelX")] - 6.0) < 1e-9


# ---------------------------------------------------------------------------
# REQ-CAS-001-3: smooth step
# ---------------------------------------------------------------------------


def test_smooth_caps_at_max_count() -> None:
    """smooth() caps all counts at max_count.

    Spec: REQ-CAS-001-3
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 500.0

    updater = CASConstraintUpdater(smooth_alpha=0.0, max_count=50.0)
    updater.smooth(lib)

    assert lib._observations[("carry_check", "m1")] <= 50.0


def test_smooth_blends_toward_target() -> None:
    """smooth() blends count toward smooth_target by smooth_alpha.

    Spec: REQ-CAS-001-3
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 20.0

    updater = CASConstraintUpdater(smooth_alpha=0.5, smooth_target=0.0, max_count=100.0)
    updater.smooth(lib)

    # (1-0.5)*20 + 0.5*0 = 10.0
    assert abs(lib._observations[("carry_check", "m1")] - 10.0) < 1e-9


def test_smooth_never_produces_negative_count() -> None:
    """smooth() clamps results to 0 when the blend would go negative.

    Spec: REQ-CAS-001-3
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 0.0

    # smooth_target is negative — but counts must stay >= 0
    updater = CASConstraintUpdater(smooth_alpha=0.5, smooth_target=-20.0, max_count=100.0)
    updater.smooth(lib)

    assert lib._observations[("carry_check", "m1")] >= 0.0


# ---------------------------------------------------------------------------
# REQ-CAS-001-4: full cas_update
# ---------------------------------------------------------------------------


def test_cas_update_applies_compress_add_smooth_in_order() -> None:
    """cas_update() applies compress → add → smooth and returns updated mapping.

    Spec: REQ-CAS-001-4
    """
    lib = _fresh_library()
    lib._observations[("carry_check", "m1")] = 10.0

    updater = CASConstraintUpdater(
        compress_factor=0.5,
        smooth_alpha=0.0,  # no blend, just cap
        max_count=100.0,
    )
    result = updater.cas_update(lib, {("carry_check", "m1"): 5.0})

    # compress: 10 * 0.5 = 5.0
    # add:      5.0 + 5.0 = 10.0
    # smooth:   (1-0)*10 + 0*0 = 10.0 → capped at 100 → 10.0
    assert abs(result[("carry_check", "m1")] - 10.0) < 1e-9


def test_cas_update_returns_dict() -> None:
    """cas_update() returns a plain dict mapping (str, str) → float.

    Spec: REQ-CAS-001-4
    """
    lib = _fresh_library()
    updater = CASConstraintUpdater()
    result = updater.cas_update(lib, {("carry_check", "m1"): 3.0})

    assert isinstance(result, dict)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in result.keys())


def test_cas_update_bounded_after_many_steps() -> None:
    """After many CAS steps with large additions, counts stay <= max_count.

    Spec: REQ-CAS-001-3, REQ-CAS-001-4
    """
    lib = _fresh_library()
    updater = CASConstraintUpdater(
        compress_factor=0.9, smooth_alpha=0.05, max_count=50.0
    )

    for _ in range(100):
        updater.cas_update(lib, {("carry_check", "m1"): 20.0})

    count = lib._observations.get(("carry_check", "m1"), 0.0)
    assert count <= 50.0, f"Expected count <= 50.0 after 100 CAS steps, got {count}"


# ---------------------------------------------------------------------------
# SCENARIO-CAS-002: new observation survives above threshold
# ---------------------------------------------------------------------------


def test_new_observation_activates_template() -> None:
    """A new observation added via CAS remains above min_frequency.

    Spec: SCENARIO-CAS-002
    """
    lib = _fresh_library()  # carry_check min_frequency=5
    updater = CASConstraintUpdater(
        compress_factor=0.9,
        smooth_alpha=0.05,
        max_count=100.0,
    )
    # Add 10 observations for carry_check — should exceed min_frequency=5
    updater.cas_update(lib, {("carry_check", "m1"): 10.0})

    active = lib.get_active_templates("m1")
    active_keys = [t.pattern_key for t in active]
    assert "carry_check" in active_keys, (
        f"carry_check should be active after CAS update with count=10, "
        f"got active={active_keys}, "
        f"obs={lib._observations.get(('carry_check', 'm1'))}"
    )


def test_decayed_observation_deactivates_template() -> None:
    """After enough CAS compress steps, a count below min_frequency deactivates.

    Spec: SCENARIO-CAS-001
    """
    lib = _fresh_library()  # carry_check min_frequency=5
    lib._observations[("carry_check", "m1")] = 6.0  # just above threshold

    updater = CASConstraintUpdater(
        compress_factor=0.5, smooth_alpha=0.0, max_count=100.0
    )
    # After 2 compress-only steps: 6 * 0.5 * 0.5 = 1.5, below min_frequency=5
    updater.compress(lib)
    updater.compress(lib)

    active = lib.get_active_templates("m1")
    active_keys = [t.pattern_key for t in active]
    assert "carry_check" not in active_keys, (
        f"carry_check should be inactive after decay, got active={active_keys}"
    )
