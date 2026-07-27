"""The free-VRAM guard on the opt-in 3090 generator path must track the context-pool size.

REQ-ARC-WMTE-5996 / SCENARIO-ARC-WMTE-5996-THE-VRAM-GUARD-TRACKS-THE-POOL

WHY THIS FILE EXISTS (2026-07-27 adversarial review of commit 776161963).

That commit raised `LocalGGUFProposer.n_ctx` from 16384 to 81920 to fix the shared-context-pool
concurrency fault. Raising `n_ctx` raises the server's VRAM footprint (measured: 12095 -> ~13452
MiB per-PID on a 3090 with `--spec-type draft-mtp`). It did NOT raise
`_GENERATOR_CUDA_MIN_FREE_MB`, which was a hand-typed 13000 whose comment still read "loads
~11.5GB".

The consequence is a NEW silent-LLM-off path created by the fix itself. On the live conductor
path (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`), a card with somewhere between 13000 and ~13452 MiB free
PASSES the guard, so `_generator_server_and_env()` hands back the CUDA build pinned to it -- and
then cudaMalloc fails, the server exits, `_ensure_server()` burns its retry budget, and
`generate()` returns `(False, msg)`. The agent proceeds LLM-off while still reporting itself as
the LLM-on scored path. That is precisely the silent-degradation shape the n_ctx fix was shipped
to remove, reintroduced one level up.

Nothing in the tree pinned the constant to the footprint, so the two could drift apart silently --
and did, in the very commit that moved the footprint. These tests are that pin. They deliberately
assert a RELATIONSHIP (guard >= measured footprint + margin, guard moves with n_ctx) rather than a
magic number, so a future n_ctx change cannot satisfy them without also moving the guard.

Measurement provenance for the numbers asserted below: per-PID `nvidia-smi --query-compute-apps`
residency of the exact shipped launch (`-ngl 999 -c <ctx> --spec-type draft-mtp --model-draft
<same gguf> --cache-type-k q8_0 --cache-type-v q8_0`, Qwen3.5-9B-MTP Q4_K_M, RTX 3090). Two
independent observations of the 81920 config recorded 13452 MiB and 13518 MiB.
"""

from __future__ import annotations

import importlib
import os

import pytest

MOD = "carnot.agentic.arc_executable_world_model"

# Per-PID measured residency of the shipped mtp-ON launch, in MiB. The larger of the two recorded
# observations, so the assertion is against the worst measurement we actually hold.
MEASURED_81920_MTP_ON_MIB = 13518
MEASURED_16384_MTP_ON_MIB = 12095


@pytest.fixture()
def wm(monkeypatch):
    """Import the module fresh with a clean env so the n_ctx override cannot leak between tests."""
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    mod = importlib.import_module(MOD)
    return mod


def test_guard_exceeds_the_measured_footprint_at_the_shipped_n_ctx(wm) -> None:
    """The whole point: the guard must not admit a card that cannot hold the server.

    This is the assertion that fails if someone raises n_ctx again without touching the guard.
    """
    assert wm._default_induce_n_ctx() == 81920, (
        "the shipped context-pool size moved; re-measure the footprint before updating this test, "
        "do not just edit the constant"
    )
    guard = wm._generator_cuda_min_free_mb()
    assert guard > MEASURED_81920_MTP_ON_MIB, (
        f"free-VRAM guard {guard} MiB does not exceed the MEASURED {MEASURED_81920_MTP_ON_MIB} MiB "
        "footprint of the launch it guards -- a card between the two passes the guard and then "
        "cudaMalloc-fails, silently returning the agent to LLM-off"
    )


def test_guard_carries_real_margin_over_the_measured_footprint(wm) -> None:
    """A guard that merely equals the footprint admits a card with zero slack for driver
    overhead, allocator fragmentation, or a second transient process. Require a real margin."""
    guard = wm._generator_cuda_min_free_mb()
    margin = guard - MEASURED_81920_MTP_ON_MIB
    assert margin >= 1000, (
        f"only {margin} MiB of margin between the guard ({guard}) and the measured footprint "
        f"({MEASURED_81920_MTP_ON_MIB}); binding a card this tightly is how the 2026-07-21 "
        "self-heal-onto-a-full-card incident happened"
    )


def test_guard_tracks_the_env_override_rather_than_being_a_literal(wm, monkeypatch) -> None:
    """CARNOT_ARC_INDUCE_N_CTX is the documented tight-VRAM lever. If the guard were a literal it
    would stay put while the footprint it guards moved -- in BOTH directions."""
    baseline = wm._generator_cuda_min_free_mb()

    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "16384")
    lowered = wm._generator_cuda_min_free_mb()
    assert lowered < baseline, "guard did not fall when the operator lowered the context pool"
    assert lowered > MEASURED_16384_MTP_ON_MIB, (
        f"guard {lowered} MiB does not clear the measured {MEASURED_16384_MTP_ON_MIB} MiB "
        "footprint of the 16384 configuration"
    )

    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "163840")
    raised = wm._generator_cuda_min_free_mb()
    assert raised > baseline, "guard did not rise when the operator raised the context pool"


def test_guard_is_derived_from_the_same_slot_count_the_fault_analysis_used(wm) -> None:
    """The admission arithmetic (n_ctx >= K*(prompt+max_tokens)) and the VRAM arithmetic
    (slots * per-slot MiB) must both use llama.cpp's own no---parallel default of 4 slots. If one
    of them were to use a different K, the fix and its guard would be sized for different servers.
    """
    assert wm._LLAMA_SERVER_DEFAULT_SLOTS == 4
    predicted = (
        wm._VRAM_MTP_ON_INTERCEPT_MIB
        + wm._VRAM_MTP_ON_PER_CTX_MIB * 81920
        + wm._VRAM_PER_SLOT_MIB * wm._LLAMA_SERVER_DEFAULT_SLOTS
    )
    # The published envelope's own prediction must land within a few percent of what was measured,
    # otherwise the envelope this guard is built on does not describe the shipped launch.
    err = abs(predicted - MEASURED_81920_MTP_ON_MIB) / MEASURED_81920_MTP_ON_MIB
    assert err < 0.02, (
        f"envelope predicts {predicted:.0f} MiB but the shipped launch measured "
        f"{MEASURED_81920_MTP_ON_MIB} MiB ({err:.1%} error) -- the envelope no longer describes "
        "the launch, so a guard derived from it is not trustworthy"
    )


def test_the_stale_11_5gb_comment_is_gone(wm) -> None:
    """The literal's comment claimed the server 'loads ~11.5GB' -- true in 2026-06, false after
    the n_ctx fix. A stale comment next to a safety constant is how the constant stopped being
    re-derived. Assert the file no longer carries the claim."""
    src = open(wm.__file__, encoding="utf-8").read()
    assert "loads ~11.5GB on a 3090" not in src, (
        "the stale footprint comment is back; it describes a configuration this file no longer "
        "launches"
    )
    assert "_GENERATOR_CUDA_MIN_FREE_MB" not in src, (
        "the hand-typed guard literal is back; derive it from _default_induce_n_ctx() so the two "
        "cannot drift"
    )


def test_env_override_is_read_at_call_time_not_import_time(wm, monkeypatch) -> None:
    """`_ensure_server()` resolves the launch env at LAUNCH time deliberately, so the guard must
    too -- reading the override once at import would freeze the guard at whatever the environment
    happened to be when the module was first imported (in a long-lived conductor process, that is
    typically hours earlier)."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "32768")
    first = wm._generator_cuda_min_free_mb()
    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "65536")
    second = wm._generator_cuda_min_free_mb()
    assert second > first, "guard did not respond to an override changed after import"
    assert "CARNOT_ARC_INDUCE_N_CTX" in os.environ  # sanity: monkeypatch really set it
