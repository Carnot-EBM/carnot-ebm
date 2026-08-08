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

Measurement provenance for the ORIGINAL numbers: per-PID `nvidia-smi --query-compute-apps`
residency of the then-shipped launch (`-ngl 999 -c <ctx> --spec-type draft-mtp --model-draft
<same gguf> --cache-type-k q8_0 --cache-type-v q8_0`, Qwen3.5-9B-MTP Q4_K_M, RTX 3090). Two
independent observations of the 81920 config recorded 13452 MiB and 13518 MiB.

UPDATED 2026-07-28 FOR THE GENERATOR SWITCH. The operator re-pinned the ARC generator from
Qwen3.5-9B-MTP to gemma-4-31B-it. That is precisely the event this file was written to catch one
level up: the footprint moved (~13.5 GB -> ~23.9 GB at the same n_ctx, because the 31B's per-token
KV is ~2x and its weights ~3x) WITHOUT n_ctx changing at all. Left alone, the 9B-derived guard
would have admitted any card with ~14 GB free and then cudaMalloc-failed on an 18.3 GB model --
the same silent-LLM-off ending, reached by a model swap rather than a context change. So the
assertions below now compare against the CURRENT generator's measured residency (mtp OFF, since
gemma-4-31B declares no MTP heads); the 9B constants are preserved as historical provenance and
are deliberately no longer what the guard is checked against. A new lever also enters the
arithmetic: `CARNOT_ARC_FFN_CPU_LAYERS` moves the footprint by ~195 MiB per offloaded layer, and
a guard blind to it would refuse a card the configured server fits on (covered in
tests/python/test_arc_ffn_cpu_offload.py).
"""

from __future__ import annotations

import importlib
import os

import pytest

MOD = "carnot.agentic.arc_executable_world_model"

# HISTORICAL (Qwen3.5-9B-MTP era). Per-PID measured residency of the mtp-ON launch, in MiB; the
# larger of the two recorded observations, so the assertion was against the worst measurement held.
# Kept because these are the provenance of every pre-2026-07-28 VRAM number in this repo. They are
# NO LONGER what the guard is built from -- see the gemma constants below.
MEASURED_81920_MTP_ON_MIB = 13518
MEASURED_16384_MTP_ON_MIB = 12095

# CURRENT (gemma-4-31B-it Q4_K_M, the generator since the 2026-07-28 operator directive). Per-PID
# resident VRAM measured on an RTX 3090, mtp OFF (this model has no MTP heads at all), q8_0 KV,
# llama-server's default 4 slots, card index confirmed by joining PID -> GPU UUID -> index rather
# than trusting CUDA_VISIBLE_DEVICES.
MEASURED_81920_GEMMA31B_MIB = 23888
MEASURED_32768_GEMMA31B_MIB = 21416
# Freed VRAM per FFN layer pushed to system RAM via `-ot` (CARNOT_ARC_FFN_CPU_LAYERS). Measured at
# n_ctx 32768 over 0/12/24/40 CPU layers: 21416 / 19072 / 16728 / 13580 MiB.
MEASURED_FREED_PER_CPU_FFN_LAYER_MIB = 195.3

# NEW SHIPPED n_ctx (2026-08-08, REQ-ARC-WMTE-6227): _INDUCE_WORST_CASE_PROMPT_TOKENS moved
# 15767 -> 22352 (a re-measurement under current defaults -- k=all transitions, object table on
# -- found the old constant stale), which raises `_default_induce_n_ctx()` 81920 -> 106496.
#
# RECONSTRUCTED, NOT DIRECTLY MEASURED, and that gap is disclosed rather than papered over. A
# no-offload launch at n_ctx=106496 does NOT fit a 24576 MiB 3090 at all -- confirmed directly:
# `cudaMalloc failed: out of memory` on `ggml_backend_cuda_buffer_type_alloc_buffer: allocating
# 1026.80 MiB`, the compute-buffer reservation, immediately after model+KV load. So the no-offload
# figure below is RECONSTRUCTED from a real launch WITH FFN offload (12 CPU-FFN layers, the same
# `-ot` lever and per-layer credit `MEASURED_FREED_PER_CPU_FFN_LAYER_MIB` used throughout this
# file): measured per-PID residency 22780 MiB + 12*195.3 = 25123.6 MiB. That figure lands within
# 0.4 MiB (0.0016%) of the linear model's own prediction at 106496
# (`_VRAM_GEMMA31B_INTERCEPT_MIB + _VRAM_GEMMA31B_PER_CTX_MIB*106496 + _VRAM_PER_SLOT_MIB*4` =
# 25124.0), which is strong independent corroboration of the model's ctx-slope term this far past
# the two points (32768, 81920) it was originally fit to -- but it is still an offload-and-
# reconstruct measurement, not a direct one, because a direct one cannot be taken on this
# hardware. Rounded DOWN to the nearest whole MiB (never up, for a footprint threshold) to
# 25123.
MEASURED_106496_GEMMA31B_MIB = 25123


@pytest.fixture()
def wm(monkeypatch):
    """Pin the env to the NO-OFFLOAD configuration, which is what the constants below measure.

    Every threshold in this file is measured against a launch that keeps the whole model on the
    card. Three env vars can move the guard away from that configuration, so all three are cleared:

      * CARNOT_ARC_INDUCE_N_CTX     -- the documented tight-VRAM lever; sets the context pool.
      * CARNOT_ARC_FFN_CPU_LAYERS   -- since 2026-07-28 a second input to the guard; an ambient
        value silently lowers every threshold asserted here.
      * CARNOT_ARC_GENERATOR_CUDA_GPU -- the one that actually bit us, and the least obvious. It
        looks irrelevant to a VRAM *threshold*, but it is the trigger for the FFN auto-fit path:
        when it names a real card and CARNOT_ARC_FFN_CPU_LAYERS is unset, the guard sizes an
        offload for that card and SUBTRACTS the freed-VRAM credit. With it set to 0 the guard came
        out at 23630 MiB against a measured 23888 MiB footprint -- a margin of MINUS 258 -- and
        two tests here failed.

        THAT LOWER GUARD IS NOT A BUG. With FFN layers pushed to system RAM the on-card footprint
        genuinely is smaller, so a smaller guard is correct *for that configuration*. The defect
        was in this fixture: it left the configuration ambient, so the tests compared an
        auto-fit-lowered guard against a no-offload measurement -- two different launches. The
        assertions were never wrong; they were just not told which launch to assert about.

        This matters beyond one variable: the conductor's standing systemd drop-in sets
        CARNOT_ARC_GENERATOR_CUDA_GPU=0, so it is present in normal operation and leaks into any
        test process that inherits it.

    NOTE ON THE IMPORT: ``import_module`` returns the CACHED module if already imported, so this is
    emphatically not a fresh import -- the previous docstring claimed it was, which was untrue. It
    does not need to be: every variable above is read at CALL time inside
    ``_generator_cuda_min_free_mb()`` / ``_default_induce_n_ctx()`` / ``_default_ffn_cpu_layers()``,
    and the module computes nothing env-dependent at import time (verified against the module AST:
    no module-level assignment calls any of those helpers). If import-time env-dependent state is
    ever added, this fixture must switch to ``importlib.reload`` -- this note is here so whoever
    adds it knows to.
    """
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
    mod = importlib.import_module(MOD)
    return mod


def test_guard_exceeds_the_measured_footprint_at_the_shipped_n_ctx(wm) -> None:
    """The whole point: the guard must not admit a card that cannot hold the server.

    This is the assertion that fails if someone raises n_ctx again without touching the guard.
    """
    assert wm._default_induce_n_ctx() == 106496, (
        "the shipped context-pool size moved; re-measure the footprint before updating this test, "
        "do not just edit the constant"
    )
    guard = wm._generator_cuda_min_free_mb()
    assert guard > MEASURED_106496_GEMMA31B_MIB, (
        f"free-VRAM guard {guard} MiB does not exceed the MEASURED "
        f"{MEASURED_106496_GEMMA31B_MIB} MiB footprint of the launch it guards -- a card between "
        "the two passes the guard and then cudaMalloc-fails, silently returning the agent to "
        "LLM-off"
    )


def test_guard_carries_real_margin_over_the_measured_footprint(wm) -> None:
    """A guard that merely equals the footprint admits a card with zero slack for driver
    overhead, allocator fragmentation, or a second transient process. Require a real margin.

    Checked against MEASURED_106496_GEMMA31B_MIB, not the historical 81920 figure (2026-08-08,
    REQ-ARC-WMTE-6227): the guard this test reads is `_generator_cuda_min_free_mb()` at
    whatever `_default_induce_n_ctx()` CURRENTLY returns, so comparing it against a footprint
    measured for a DIFFERENT n_ctx would silently stop being apples-to-apples the moment the
    two diverged -- exactly the gap this file exists to close for the constant itself."""
    guard = wm._generator_cuda_min_free_mb()
    margin = guard - MEASURED_106496_GEMMA31B_MIB
    assert margin >= 1000, (
        f"only {margin} MiB of margin between the guard ({guard}) and the measured footprint "
        f"({MEASURED_106496_GEMMA31B_MIB}); binding a card this tightly is how the 2026-07-21 "
        "self-heal-onto-a-full-card incident happened"
    )


def test_the_autofit_guard_carries_real_margin_over_the_autofit_footprint(wm, monkeypatch) -> None:
    """The same safety property as the test above, but for the configuration that actually SHIPS.

    ADDED 2026-07-30 after review pointed out a real gap in the fixture fix. Clearing
    CARNOT_ARC_GENERATOR_CUDA_GPU was the correct way to make the no-offload assertions honest --
    they are measured against a no-offload launch -- but it left the *production* path less covered
    than the one that does not ship: the conductor's standing systemd drop-in SETS that variable, so
    the FFN auto-fit is what runs in normal operation, and nothing asserted that the auto-fit guard
    clears an auto-fit footprint with margin.

    That gap mattered because the two properties are genuinely different, and only one of them was
    covered anywhere:

      * ADMISSIBILITY, guard <= free_mb -- "we only claim a card we fit on". Already covered, in
        tests/python/test_arc_ffn_cpu_offload.py.
      * SAFETY MARGIN, guard >= footprint + margin -- "we do not bind a card with no slack". This
        is the property whose absence caused the original incident, and at the auto-fit layer count
        it was asserted nowhere. A guard that satisfied admissibility while sitting BELOW its own
        footprint would pass every existing test and still cudaMalloc-fail into silent LLM-off.

    Both are asserted here, so neither can be satisfied at the other's expense.

    THE FOOTPRINT IS DERIVED FROM THIS FILE'S MEASUREMENTS, NOT FROM THE MODULE'S PREDICTOR.
    Comparing the guard against `_predicted_generator_vram_mib` would be circular -- the guard is
    computed FROM that predictor, so the assertion would hold no matter how wrong the envelope was.
    Instead the expected footprint is rebuilt from the two independently measured constants at the
    top of this file: the measured 106496 no-offload residency, minus the measured per-layer
    credit.

    Subtracting a credit measured at n_ctx 32768 from a footprint measured at 106496 is sound
    because the `-ot` lever moves FFN *weight* tensors to system RAM, and weight size does not
    depend on the context-pool size (what n_ctx moves is the KV cache, which stays on the card).

    MOCKED, NOT SKIPPED, per CLAUDE.md: the auto-fit's only environmental input is
    `_cuda_gpu_free_mb`, so stubbing that one call exercises the whole decision on any machine,
    with or without a GPU.
    """
    # A 3090 with a few hundred MiB of driver/desktop overhead already resident -- i.e. the
    # realistic case where the no-offload guard (26623 MiB) does NOT fit but a small offload does.
    #
    # RAISED 24123 -> 24400 (2026-08-08, REQ-ARC-WMTE-6227). At the new n_ctx=106496, the max
    # auto-fit offload (_FFN_CPU_AUTOFIT_MAX_LAYERS=12) frees 12*195.3=2343.6 MiB, landing the
    # guard-inclusive footprint at 24280 MiB -- ABOVE the old 24123 free_mb, so at that value the
    # auto-fit correctly refuses to engage at all (0 layers: even its maximum offload cannot
    # satisfy the margin). 24400 is the smallest round free_mb, swept empirically against the real
    # module functions, where 12 layers both engages and satisfies the guard -- still a realistic
    # "few hundred MiB of overhead" reading on a 24576 MiB card.
    free_mb = 24400
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: free_mb)

    layers = wm._default_ffn_cpu_layers()
    assert layers > 0, (
        "auto-fit did not engage on a card the no-offload guard cannot fit; this test is then not "
        "exercising the production path it was written for"
    )
    assert layers <= wm._FFN_CPU_AUTOFIT_MAX_LAYERS

    guard = wm._generator_cuda_min_free_mb(layers)
    expected_footprint = (
        MEASURED_106496_GEMMA31B_MIB - layers * MEASURED_FREED_PER_CPU_FFN_LAYER_MIB
    )

    margin = guard - expected_footprint
    # `guard` is an int over a float footprint, so allow 1 MiB for truncation. Derived from the
    # module's own margin constant rather than a literal, so raising the required margin cannot
    # leave this test asserting the old, smaller one.
    assert margin >= wm._GENERATOR_CUDA_GUARD_MARGIN_MIB - 1, (
        f"the auto-fit guard ({guard} MiB at {layers} CPU-FFN layers) carries only {margin:.0f} MiB "
        f"over the measurement-derived footprint ({expected_footprint:.0f} MiB). This is the "
        "SHIPPED configuration -- the conductor's systemd drop-in sets "
        "CARNOT_ARC_GENERATOR_CUDA_GPU -- so a guard that binds this tightly here binds tightly in "
        "production, and cudaMalloc failure returns the agent to LLM-off silently."
    )
    assert guard <= free_mb, (
        f"auto-fit chose {layers} layers but the resulting guard ({guard} MiB) still exceeds the "
        f"{free_mb} MiB it was fitting to -- the card would be declined despite an offload having "
        "been selected for it, which is the auto-fit failing at its one job"
    )


def test_guard_tracks_the_env_override_rather_than_being_a_literal(wm, monkeypatch) -> None:
    """CARNOT_ARC_INDUCE_N_CTX is the documented tight-VRAM lever. If the guard were a literal it
    would stay put while the footprint it guards moved -- in BOTH directions."""
    baseline = wm._generator_cuda_min_free_mb()

    # 32768, not 16384: 32768 is a point we have DIRECTLY MEASURED for the current generator.
    # 16384 was only ever measured for the retired 9B, and asserting against a footprint from a
    # different model is how a guard stops guarding while its tests stay green.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "32768")
    lowered = wm._generator_cuda_min_free_mb()
    assert lowered < baseline, "guard did not fall when the operator lowered the context pool"
    assert lowered > MEASURED_32768_GEMMA31B_MIB, (
        f"guard {lowered} MiB does not clear the measured {MEASURED_32768_GEMMA31B_MIB} MiB "
        "footprint of the 32768 configuration"
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
        wm._VRAM_GEMMA31B_INTERCEPT_MIB
        + wm._VRAM_GEMMA31B_PER_CTX_MIB * 81920
        + wm._VRAM_PER_SLOT_MIB * wm._LLAMA_SERVER_DEFAULT_SLOTS
    )
    # The published envelope's own prediction must land within a few percent of what was measured,
    # otherwise the envelope this guard is built on does not describe the shipped launch.
    err = abs(predicted - MEASURED_81920_GEMMA31B_MIB) / MEASURED_81920_GEMMA31B_MIB
    assert err < 0.02, (
        f"envelope predicts {predicted:.0f} MiB but the shipped launch measured "
        f"{MEASURED_81920_GEMMA31B_MIB} MiB ({err:.1%} error) -- the envelope no longer describes "
        "the launch, so a guard derived from it is not trustworthy"
    )
    # ...and it must ALSO describe the other measured point, which is the only thing that makes a
    # two-point fit more than a restatement of one measurement.
    predicted_32k = (
        wm._VRAM_GEMMA31B_INTERCEPT_MIB
        + wm._VRAM_GEMMA31B_PER_CTX_MIB * 32768
        + wm._VRAM_PER_SLOT_MIB * wm._LLAMA_SERVER_DEFAULT_SLOTS
    )
    # The FFN-offload credit is a measurement too, and it belongs to the same envelope; pin it here
    # so the module constant and this file's recorded measurement cannot drift apart.
    assert wm._VRAM_PER_CPU_FFN_LAYER_MIB == MEASURED_FREED_PER_CPU_FFN_LAYER_MIB
    err_32k = abs(predicted_32k - MEASURED_32768_GEMMA31B_MIB) / MEASURED_32768_GEMMA31B_MIB
    assert err_32k < 0.02, (predicted_32k, MEASURED_32768_GEMMA31B_MIB)


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
