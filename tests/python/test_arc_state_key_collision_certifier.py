"""REQ-ARC-ARM-6434: collision-certified state-key suffix behavior."""

from __future__ import annotations

import pytest

from carnot.agentic.arc_state_key_certifier import (
    HashSubstitutionError,
    StateKeyCollisionCertifier,
)


def _step(action: int, data: dict | None = None) -> dict:
    return {"action": action, "data": data}


def test_genuine_collision_records_minimal_certificate() -> None:
    """SCENARIO-ARC-ARM-6434-CERTIFICATE: alias evidence enables only the needed suffix."""
    certifier = StateKeyCollisionCertifier(enabled=True, max_suffix_k=3)

    root_key = certifier.state_key("frame:a", ("obs:a",), [])
    child_key = certifier.state_key(
        "frame:a",
        ("obs:a", "act:1", "obs:a"),
        [_step(1, {"x": 2, "y": 3})],
    )

    assert root_key == "frame:a"
    assert child_key.startswith("frame:a|certk:1:")
    rows = certifier.certificate_rows()
    assert len(rows) == 1
    assert rows[0]["base_key"] == "frame:a"
    assert rows[0]["minimal_suffix_k"] == 1
    assert len(rows[0]["observation_history_hashes"]) == 2
    assert rows[0]["alias_evidence"]["known_history_count"] == 2
    assert rows[0]["forbidden_inputs"] == []


def test_no_collision_keeps_base_keys() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: different base keys need no suffix."""
    certifier = StateKeyCollisionCertifier(enabled=True)

    assert certifier.state_key("frame:a", ("obs:a",), []) == "frame:a"
    assert certifier.state_key("frame:b", ("obs:b",), [_step(1)]) == "frame:b"

    assert certifier.certificate_rows() == []
    assert certifier.diagnostics()["accepted_certificate_count"] == 0


def test_identical_history_does_not_create_false_certificate() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: replaying the same live history is not an alias."""
    certifier = StateKeyCollisionCertifier(enabled=True)
    history = ("obs:a", "act:1", "obs:a")
    actions = [_step(1)]

    assert certifier.state_key("frame:a", history, actions) == "frame:a"
    assert certifier.state_key("frame:a", history, actions) == "frame:a"

    assert certifier.certificate_rows() == []


def test_monotone_hud_change_does_not_create_certificate() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: changing base keys are not collisions."""
    certifier = StateKeyCollisionCertifier(enabled=True)

    for idx in range(3):
        key = certifier.state_key(f"hud:{idx}", (f"obs:{idx}",), [_step(idx)])
        assert key == f"hud:{idx}"

    assert certifier.certificate_rows() == []


def test_reset_and_process_restart_forget_prior_aliases() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: certificates are per live run."""
    first = StateKeyCollisionCertifier(enabled=True)
    assert first.state_key("frame:a", ("obs:a",), []) == "frame:a"
    first.reset()
    assert first.state_key("frame:a", ("obs:a", "act:1", "obs:a"), [_step(1)]) == "frame:a"
    assert first.certificate_rows() == []

    restarted = StateKeyCollisionCertifier(enabled=True)
    assert restarted.state_key("frame:a", ("obs:a", "act:2", "obs:a"), [_step(2)]) == "frame:a"
    assert restarted.certificate_rows() == []


def test_hash_substitution_fails_closed_when_distinct_histories_share_digest() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: digest substitution cannot fake identity."""
    certifier = StateKeyCollisionCertifier(enabled=True, history_digest_func=lambda _h: "same")

    assert certifier.state_key("frame:a", ("obs:a",), []) == "frame:a"
    with pytest.raises(HashSubstitutionError):
        certifier.state_key("frame:a", ("obs:a", "act:1", "obs:a"), [_step(1)])

    assert certifier.diagnostics()["hash_substitution_detected"] is True
    assert certifier.certificate_rows() == []


def test_hash_instability_fails_closed_when_same_history_changes_digest() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: unstable hashes cannot create certificates."""
    digests = iter(["first", "second"])
    certifier = StateKeyCollisionCertifier(enabled=True, history_digest_func=lambda _h: next(digests))
    history = ("obs:a",)

    assert certifier.state_key("frame:a", history, []) == "frame:a"
    with pytest.raises(HashSubstitutionError):
        certifier.state_key("frame:a", history, [])

    assert certifier.diagnostics()["hash_instability_detected"] is True


def test_unseparable_suffix_refuses_certificate() -> None:
    """SCENARIO-ARC-ARM-6434-ATTACKS-FAIL-CLOSED: ambiguous suffixes do not certify."""
    certifier = StateKeyCollisionCertifier(enabled=True, max_suffix_k=1)

    assert certifier.state_key("frame:a", ("obs:a",), []) == "frame:a"
    assert certifier.state_key("frame:a", ("obs:a", "act:1", "obs:a"), [_step(1)]) != "frame:a"
    refused_key = certifier.state_key(
        "frame:a",
        ("obs:a", "act:1", "obs:a", "marker:distinct"),
        [_step(1)],
    )

    assert refused_key == "frame:a"
    assert certifier.diagnostics()["refused_certificate_count"] == 1
