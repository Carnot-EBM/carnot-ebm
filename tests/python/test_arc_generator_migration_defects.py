"""The four gemma-4-31B migration defects, and the invariant that would have caught two of them.

REQ-ARC-WMTE-6034 / SCENARIO-ARC-WMTE-6034-MIGRATION-DEFECTS-STAY-FIXED

WHY THIS FILE EXISTS (operator review, 2026-07-28). The switch from Qwen3.5-9B-MTP to
gemma-4-31B-it left four measured defects on the live path. Each is silent -- none raises, none
logs an error, and every one of them ends with a run that reports itself as the LLM-on scored path
while being slower or degraded than intended:

  1. THE VRAM GUARD DID NOT KNOW ABOUT MTP. `_generator_cuda_min_free_mb()` budgeted for the main
     weights only. The MTP draft head is a real, n_ctx-SCALED cost (+862 MiB at n_ctx 32768,
     +1290 MiB at n_ctx 81920 -- measured per-PID on an RTX 3090), so with MTP on the guard
     under-predicted the shipped configuration by ~1290 MiB, admitted a card, and let the server
     cudaMalloc-fail. That is the exact silent-LLM-off fault the guard exists to prevent.

     THE MISSING INVARIANT, which is the real subject of this file: the guard broke TWICE, in
     OPPOSITE directions -- first at 14937 MiB (under-predicting the 31B by ~10.5 GB), then at
     25388 MiB (exceeding what a 3090 physically has). Both endings are identical: a silent LLM-off
     run. Nothing anywhere asserted the one property that rules out both -- THE DEMAND MUST NEVER
     EXCEED WHAT THE TARGET HARDWARE CAN SUPPLY. That assertion is `test_the_missing_invariant_*`
     below, and it is written as a property over a RANGE of free-VRAM readings and both MTP
     settings, not as a spot check on today's numbers.

  2. `kv_quant` DEFAULTED TO None (= f16), WHICH CANNOT LOAD THIS MODEL. Measured: f16 KV OOMs at
     n_ctx 32768 and SIGSEGVs at the shipped 81920. Every live site passed "q8_0" explicitly, so
     the broken default was reachable only by a caller who omitted the argument -- and handed
     exactly that caller an unloadable server.

  3. `repo_substr` DEFAULTED TO "gemma-4-12B-it" -- a THIRD model, neither the retired Qwen nor
     the directed 31B. A bare proposer silently produced induction numbers from a model the
     project does not run.

  4. `--model-draft` POINTED AT THE MAIN GGUF. This is the single most dangerous line in the
     migration, because it does not fail. Given a draft it cannot use, llama.cpp emits
         W llama_init_from_model: context type MTP requested but model doesn't contain MTP layers
         W common_speculative_init: no implementations specified for speculative decoding
     and then serves normally with speculation SILENTLY DISABLED -- /health 200, correct output,
     zero speedup. A misconfigured MTP is indistinguishable from a working one except by tok/s.
     gemma-4-31B's MTP head is a SEPARATE 491 MiB GGUF (`mtp-gemma-4-31B-it-Q8_0.gguf`, arch
     `gemma4-assistant`), so the draft must resolve to that file, and MTP must be dropped entirely
     when it is absent rather than falling back to the main weights.

MEASUREMENT PROVENANCE for every number asserted here: per-PID `nvidia-smi --query-compute-apps`
residency, PID -> GPU UUID -> index (never CUDA_VISIBLE_DEVICES), one server alone on an RTX 3090,
gemma-4-31B-it Q4_K_M + the real `mtp-gemma-4-31B-it-Q8_0.gguf` head, q8_0 KV, 2026-07-28.
"""

from __future__ import annotations

import importlib

import pytest

MOD = "carnot.agentic.arc_executable_world_model"

# An RTX 3090: 24576 MiB total, 24123 MiB free when genuinely idle (the driver/display holds the
# rest). 24123 is the number a real guard has to be satisfiable against, not 24576.
RTX3090_TOTAL_MIB = 24576
RTX3090_IDLE_FREE_MIB = 24123
# The scored Kaggle card, where no FFN offload should ever be needed.
KAGGLE_96G_FREE_MIB = 96 * 1024

# Measured MTP draft-head surcharge, at the two context sizes it was taken at.
MEASURED_HEAD_MIB_AT_32768 = 862  # 22264 (mtp on) - 21402 (mtp off)
MEASURED_HEAD_MIB_AT_81920 = 1290  # 23020 (mtp on) - 21730 (mtp off), both at 11 CPU-FFN layers


@pytest.fixture()
def wm(monkeypatch):
    """Import the module with every generator env knob cleared, so an ambient override in the
    developer's shell cannot mask a real regression (or manufacture one)."""
    for var in (
        "CARNOT_ARC_FFN_CPU_LAYERS",
        "CARNOT_ARC_GENERATOR_CUDA_GPU",
        "CARNOT_ARC_INDUCE_N_CTX",
        "CARNOT_ARC_INDUCE_MAX_TOKENS",
        "CARNOT_ARC_MTP",
        "CARNOT_ARC_MTP_GGUF_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    return importlib.import_module(MOD)


# ==============================================================================================
# DEFECT 1 + THE MISSING INVARIANT.
# ==============================================================================================


@pytest.mark.parametrize("mtp", [False, True])
def test_the_missing_invariant_demand_never_exceeds_what_the_hardware_supplies(
    wm, monkeypatch, mtp
) -> None:
    """THE ASSERTION NEITHER BROKEN VERSION OF THE GUARD EVER HAD.

    For any free-VRAM reading a real card can present, the auto-fit must either (a) return a layer
    count whose guard requirement is SATISFIED by that reading, or (b) refuse and say so. What it
    must never do is return a layer count that the guard then rejects -- that combination is how
    both historical breakages ended: the operator asks for a CUDA card, the guard declines it, and
    the generator silently falls back to a ~2 tok/s iGPU that cannot complete a single induce call.

    Written as a property over a RANGE and over BOTH mtp settings, deliberately. A spot check on
    today's constants is what let the guard move 14937 -> 25388 with its tests still green.
    """
    monkeypatch.setenv("CARNOT_ARC_MTP", "1" if mtp else "0")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: RTX3090_TOTAL_MIB)
    n_ctx = wm._default_induce_n_ctx()

    for free in range(18000, RTX3090_IDLE_FREE_MIB + 1, 250):
        monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx, _f=free: _f)
        layers = wm._default_ffn_cpu_layers()
        fitted = wm._ffn_cpu_layers_to_fit(free, n_ctx, mtp)
        if fitted < 0:
            # Branch (b): honest refusal. The auto-fit must NOT have offered a layer count.
            assert layers == 0, (
                f"free={free} MiB cannot host the generator at any capped offload, but auto-fit "
                f"still returned {layers} layers"
            )
            continue
        # Branch (a): if it offered a count, the guard must accept it.
        need = wm._generator_cuda_min_free_mb(layers, mtp)
        assert need <= free, (
            f"auto-fit picked {layers} CPU-FFN layers at free={free} MiB (mtp={mtp}) but the "
            f"guard then demands {need} MiB -- the card is declined and the generator falls back "
            "to the ~2 tok/s iGPU, which is the silent-LLM-off outage this invariant forbids"
        )


def test_the_shipped_local_default_is_refused_on_a_real_3090_at_the_corrected_n_ctx(
    wm, monkeypatch
) -> None:
    """The concrete instance of the invariant, on the hardware this project actually has.

    A 3090 idles at 24123 MiB free, NOT 24576. Asserting against the total is how a guard can be
    'satisfiable' in arithmetic and unsatisfiable in practice.

    SCOPED TO THE LOCAL DEFAULT (mtp OFF) as of 2026-07-28, third pass. This test used to be
    parametrized over mtp, asserting BOTH arms were satisfiable. The mtp=True arm no longer is, by
    design -- see `test_mtp_on_is_refused_locally_rather_than_auto_offloaded_into_a_timeout` below
    for the contract that replaced it.

    RETARGETED AGAIN 2026-08-08 (REQ-ARC-WMTE-6227). This test's ORIGINAL name and body (preserved
    in the docstring above, and in git history) asserted the mtp-OFF local default IS satisfiable
    on a 3090 -- and it was, at the THEN-shipped n_ctx=81920 (7 offloaded layers fit within the
    _FFN_CPU_AUTOFIT_MAX_LAYERS=12 cap). `_INDUCE_WORST_CASE_PROMPT_TOKENS` moved 15767 -> 22352
    (a re-measurement under current defaults -- see that constant's own comment), raising
    `_default_induce_n_ctx()` 81920 -> 106496. At the new n_ctx, mtp-OFF needs MORE than 12
    offloaded layers to fit a 24123 MiB-free 3090 -- it falls off the same cap the mtp-ON arm
    already fell off in the 2026-07-28 third pass, for the identical reason: the offload needed
    exceeds the cap `_default_ffn_cpu_layers()`'s own docstring names as the throughput-timeout
    threshold, so auto-fit correctly REFUSES rather than silently selecting a configuration slow
    enough to time out induction. This is not a regression this test should paper over -- it is
    the guard doing its job with a NOW-CORRECT worst-case figure, and the local-dev remedy is the
    documented `CARNOT_ARC_INDUCE_N_CTX` override, not raising the offload cap (which would
    reintroduce the timeout risk `_FFN_CPU_AUTOFIT_MAX_LAYERS` was measured to prevent).
    """
    monkeypatch.setenv("CARNOT_ARC_MTP", "0")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: RTX3090_IDLE_FREE_MIB)
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: RTX3090_TOTAL_MIB)
    wm.GENERATOR_SELECTION_LOG.clear()
    wm._GENERATOR_SELECTION_SEEN.clear()

    layers = wm._default_ffn_cpu_layers()
    assert layers == 0, (
        f"expected the auto-fit to REFUSE on a 3090 at the corrected n_ctx (needs more than "
        f"_FFN_CPU_AUTOFIT_MAX_LAYERS={wm._FFN_CPU_AUTOFIT_MAX_LAYERS} offloaded layers to fit); "
        f"got {layers} -- if this now passes with a nonzero layer count, either the worst-case "
        "prompt constant shrank again or the autofit cap moved, and this test's numbers need a "
        "fresh re-derivation, not a silent bump"
    )
    # ...and the refusal is AUDIBLE (mirrors the mtp-on refusal test below): the whole failure
    # class this module guards against is a degraded configuration that no channel reports.
    joined = "\n".join(wm.GENERATOR_SELECTION_LOG)
    assert "cannot fit the generator" in joined, (
        f"the refusal must be logged, not silent; got:\n{joined}"
    )


def test_mtp_on_costs_more_vram_than_mtp_off_which_is_why_the_local_default_is_off(wm) -> None:
    """The measured basis of the LOCAL default, asserted rather than asserted-in-prose.

    This is the CAP-INDEPENDENT form of the claim. It used to compare the two auto-fit LAYER
    COUNTS, which silently made it a test of `_FFN_CPU_AUTOFIT_MAX_LAYERS` as much as of the VRAM
    envelope: once the cap dropped below the layer count MTP-on needs, the mtp=True side returned
    the -1 sentinel and the comparison became meaningless. The underlying measured fact -- the
    draft head is a real, context-scaled VRAM cost -- is about the ENVELOPE, so assert it there.

    If this ever stops holding, the reasoning behind `ARC_LIVE_GENERATOR_MTP_DEFAULT = "0"` has
    changed and the constant should be revisited -- not silently left in place.
    """
    n_ctx = wm._default_induce_n_ctx()
    for layers in (0, 7, 12):
        assert wm._generator_cuda_min_free_mb(layers, True) > wm._generator_cuda_min_free_mb(
            layers, False
        ), f"MTP-on must require strictly more VRAM at {layers} offloaded layers"


def test_mtp_on_is_refused_locally_rather_than_auto_offloaded_into_a_timeout(
    wm, monkeypatch
) -> None:
    """MTP-on at the shipped n_ctx does NOT fit a 3090, and the auto-fit says so instead of
    offloading its way there.

    THE CONTRACT THIS PINS, which changed 2026-07-28 (third pass). At n_ctx 81920 on a 24123 MiB
    card, MTP-off needs 7 offloaded FFN layers and MTP-on needs 14. `_FFN_CPU_AUTOFIT_MAX_LAYERS`
    is 12, so MTP-on falls off the end and `_ffn_cpu_layers_to_fit` returns -1.

    THAT IS THE INTENDED ANSWER, not a capability regression, and the reason is measured rather
    than cautious: a 14-layer offload costs more decode throughput than MTP's 1.398x buys back
    (MTP-off @7 layers ~= 23.8 tok/s versus MTP-on @14 layers ~= 13.9 x 1.398 ~= 19.4 tok/s), so
    the configuration the old cap would have auto-selected was STRICTLY WORSE than the one it
    replaced. Refusing it loudly -- with a message naming `CARNOT_ARC_MTP=0` as the remedy -- is
    better than silently selecting a slower path, and better than the old cap of 24, which let the
    auto-fit walk past the ~12-layer threshold `_default_ffn_cpu_layers()`'s own docstring names as
    the point where induction starts timing out.

    The SCORED path is unaffected: see
    `test_the_scored_96gb_card_needs_no_offload_at_all_with_mtp_on`.

    UPDATED 2026-08-08 (REQ-ARC-WMTE-6227). `_default_induce_n_ctx()` moved 81920 -> 106496 (see
    `test_the_shipped_local_default_is_refused_on_a_real_3090_at_the_corrected_n_ctx` for why).
    MTP-off no longer fits in 7 layers at the new n_ctx either -- both arms now fall off the same
    12-layer cap, so both assertions read -1. The BOTH-REFUSED outcome and the reasoning above are
    otherwise unchanged: this is the guard correctly refusing a configuration slow enough to time
    out, not a capability regression to compensate for.
    """
    n_ctx = wm._default_induce_n_ctx()
    assert wm._ffn_cpu_layers_to_fit(RTX3090_IDLE_FREE_MIB, n_ctx, False) == -1, (
        "MTP-off no longer auto-fits a 3090 at the corrected n_ctx -- see this test's 2026-08-08 "
        "update note for why that is now the correct answer"
    )
    assert wm._ffn_cpu_layers_to_fit(RTX3090_IDLE_FREE_MIB, n_ctx, True) == -1, (
        "MTP-on must NOT auto-fit on a 3090 -- the offload it needs is a net throughput loss"
    )
    # ...and the refusal is AUDIBLE, not a silent 0. The whole failure class this module guards
    # against is a degraded configuration that no channel reports.
    monkeypatch.setenv("CARNOT_ARC_MTP", "1")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: RTX3090_IDLE_FREE_MIB)
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: RTX3090_TOTAL_MIB)
    wm.GENERATOR_SELECTION_LOG.clear()
    wm._GENERATOR_SELECTION_SEEN.clear()
    assert wm._default_ffn_cpu_layers() == 0
    joined = "\n".join(wm.GENERATOR_SELECTION_LOG)
    assert "cannot fit the generator" in joined and "CARNOT_ARC_MTP=0" in joined, (
        f"the refusal must name the remedy; got:\n{joined}"
    )


def test_the_autofit_cap_agrees_with_the_threshold_its_own_docstring_names(wm) -> None:
    """`_FFN_CPU_AUTOFIT_MAX_LAYERS` must not exceed the ~12-layer timeout threshold that
    `_default_ffn_cpu_layers()`'s measured table describes.

    The cap was 24 while the docstring said "treat anything past ~12 layers as likely to push real
    induction into timeout" -- so the auto-fit could select 23 or 24 layers, straight through the
    threshold its own justification names, into a region where nothing has measured the 4-slot
    decode rate the live path actually runs at. A cap and the prose that justifies it must not
    contradict each other; this asserts they do not.
    """
    assert wm._FFN_CPU_AUTOFIT_MAX_LAYERS <= 12
    assert "past ~12 layers" in (wm._default_ffn_cpu_layers.__doc__ or "")


def test_the_scored_96gb_card_needs_no_offload_at_all_with_mtp_on(wm, monkeypatch) -> None:
    """The other half of "correct on BOTH": the reason the SCORED default differs from the local
    one is that the 96 GB card hosts the MTP-on server outright, so the throughput cost that makes
    MTP a net loss locally simply does not exist there."""
    monkeypatch.setenv("CARNOT_ARC_MTP", "1")
    n_ctx = wm._default_induce_n_ctx()
    assert wm._ffn_cpu_layers_to_fit(KAGGLE_96G_FREE_MIB, n_ctx, True) == 0
    assert wm._generator_cuda_min_free_mb(0, True) < KAGGLE_96G_FREE_MIB


def test_the_mtp_head_cost_scales_with_context_and_is_not_a_flat_constant(wm) -> None:
    """A FLAT head constant is the trap this envelope was one edit away from.

    The head is a 491 MiB file but carries KV proportional to the pool: 862 MiB at n_ctx 32768 and
    1290 MiB at 81920. Taking either single reading as a constant under-predicts the other by
    ~428 MiB at the shipped context -- enough to admit a card that then cudaMalloc-fails.
    """
    for n_ctx, measured in (
        (32768, MEASURED_HEAD_MIB_AT_32768),
        (81920, MEASURED_HEAD_MIB_AT_81920),
    ):
        head = wm._predicted_generator_vram_mib(n_ctx, 0, True) - wm._predicted_generator_vram_mib(
            n_ctx, 0, False
        )
        assert abs(head - measured) <= 2, (n_ctx, head, measured)
    assert wm._VRAM_MTP_HEAD_PER_CTX_MIB > 0, (
        "the head surcharge has been flattened into a constant; it scales with n_ctx (measured "
        "862 MiB at 32768 vs 1290 MiB at 81920) and a flat value under-predicts the shipped 81920"
    )


def test_the_envelope_predicts_a_configuration_it_was_never_fitted_on(wm) -> None:
    """Two-point fits are weak evidence. This checks the envelope against a THIRD, differently
    shaped measurement -- n_ctx 81920 with 11 offloaded FFN layers, a count chosen by the auto-fit
    rather than by the original sweep. Predicted 21740 MiB, measured 21730 MiB."""
    predicted = wm._predicted_generator_vram_mib(81920, 11, False)
    measured = wm._VRAM_GEMMA31B_11LAYER_81920_CHECK_MIB
    assert abs(predicted - measured) / measured < 0.01, (predicted, measured)


# ==============================================================================================
# DEFECTS 2 + 3: dataclass defaults that shipped a model/config nobody chose.
# ==============================================================================================


def test_kv_quant_default_is_q8_0_not_none(wm) -> None:
    """f16 KV (kv_quant=None) CANNOT LOAD this model: OOM at n_ctx 32768, SIGSEGV at the shipped
    81920. A default that produces an unloadable server is not a conservative default."""
    assert wm.LocalGGUFProposer().kv_quant == "q8_0"
    assert wm.LocalGGUFProposer(repo_substr="x").kv_quant == "q8_0"
    # An explicit override still wins -- this is a default, not a hardcode.
    assert wm.LocalGGUFProposer(kv_quant=None).kv_quant is None


def test_repo_substr_default_is_the_pinned_31b_not_a_third_model(wm) -> None:
    """The default was `gemma-4-12B-it` -- neither the retired Qwen nor the directed 31B."""
    assert wm.LocalGGUFProposer().repo_substr == wm.ARC_LIVE_GENERATOR_REPO_SUBSTR
    assert "31B" in wm.LocalGGUFProposer().repo_substr
    for retired in ("Qwen3.5-9B", "Qwen3.6-27B", "gemma-4-12B"):
        assert retired not in wm.LocalGGUFProposer().repo_substr
    assert wm.LocalGGUFProposer(repo_substr="explicit-other").repo_substr == "explicit-other"


# ==============================================================================================
# DEFECT 4: the draft head, and the silent-degradation configuration.
# ==============================================================================================


def _launch_argv(wm, monkeypatch, tmp_path, *, mtp: bool, head: str | None):
    """Drive the REAL `_ensure_server()` with a fake Popen and return (argv, proposer).

    Everything faked is an external dependency (binary on disk, GGUF on disk, subprocess, health
    poll). The argv construction under test is the real code path.
    """
    fake_server = tmp_path / "llama-server"
    fake_server.write_text("#!/bin/sh\nexit 0\n")
    fake_main = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    fake_main.write_bytes(b"\0")

    monkeypatch.setattr(
        wm,
        "_generator_server_and_env",
        # Accepts BOTH threaded arguments. `_ensure_server()` passes ffn_cpu_layers AND mtp so the
        # guard budgets for the configuration that will really launch; a stub with a narrower
        # signature would silently stop exercising that wiring.
        lambda _ffn_cpu_layers=None, _mtp=None: (fake_server, None),
    )
    monkeypatch.setattr(wm, "_resolve_gguf", lambda _s: str(fake_main))
    monkeypatch.setattr(wm, "_resolve_mtp_head", lambda *_a, **_k: head)

    captured: dict = {}

    class _FakeProc:
        pid = 4321

    def _fake_popen(args, **_kw):
        captured["argv"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(wm.subprocess, "Popen", _fake_popen)
    prop = wm.LocalGGUFProposer(n_ctx=32768, mtp=mtp)
    calls = {"n": 0}

    def _healthy():
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(prop, "_healthy", _healthy)
    assert prop._ensure_server() is True
    return captured["argv"], prop


def test_model_draft_resolves_to_the_head_never_the_main_gguf(wm, monkeypatch, tmp_path) -> None:
    """THE DEFECT, DIRECTLY. `--model-draft <main gguf>` is accepted by llama.cpp, warned about,
    and then served with speculation silently disabled -- healthy server, correct output, zero
    speedup. It must never be emitted."""
    head = tmp_path / wm.ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME
    head.write_bytes(b"\0")
    argv, prop = _launch_argv(wm, monkeypatch, tmp_path, mtp=True, head=str(head))

    assert "--spec-type" in argv and argv[argv.index("--spec-type") + 1] == "draft-mtp"
    draft = argv[argv.index("--model-draft") + 1]
    assert draft == str(head)
    main = argv[argv.index("-m") + 1]
    assert draft != main, "--model-draft points at the main weights: the silent-degradation config"
    assert wm._is_mtp_head_file(str(draft).split("/")[-1])
    assert prop.last_mtp_draft_path == str(head)
    assert prop.mtp_disabled_reason == ""


def test_mtp_is_dropped_loudly_when_the_head_is_absent(wm, monkeypatch, tmp_path) -> None:
    """Head missing must mean NO speculative flags -- not the main weights as a stand-in.

    Dropping the flags costs the ~1.4x speedup. Passing a bogus draft costs the same speedup AND
    leaves a run that believes it had MTP, which is strictly worse: the first is recoverable from
    the log, the second is only detectable by measuring tok/s against a control."""
    argv, prop = _launch_argv(wm, monkeypatch, tmp_path, mtp=True, head=None)
    assert "--spec-type" not in argv
    assert "--model-draft" not in argv
    assert prop.last_mtp_draft_path == ""
    assert prop.mtp_disabled_reason, "MTP was dropped with no recorded reason -- silently"
    assert "CARNOT_ARC_MTP_GGUF_PATH" in prop.mtp_disabled_reason


def test_a_head_shaped_like_the_main_model_is_rejected_rather_than_drafted(
    wm, monkeypatch, tmp_path
) -> None:
    """Defence in depth: even if a resolver returns the MAIN weights as the 'head', the launch
    must refuse it. The name test is what distinguishes the two files -- nothing else does."""
    decoy = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    decoy.write_bytes(b"\0")
    argv, prop = _launch_argv(wm, monkeypatch, tmp_path, mtp=True, head=str(decoy))
    assert "--model-draft" not in argv
    assert prop.mtp_disabled_reason


def test_mtp_off_emits_no_speculative_flags_at_all(wm, monkeypatch, tmp_path) -> None:
    argv, prop = _launch_argv(wm, monkeypatch, tmp_path, mtp=False, head=None)
    assert "--spec-type" not in argv and "--model-draft" not in argv
    assert prop.mtp_disabled_reason == ""


def test_resolve_gguf_never_returns_the_draft_head_as_the_main_model(wm, monkeypatch) -> None:
    """The head is 491 MiB and would LOAD, SERVE, and answer nonsense. It used to be excluded only
    by alphabetical luck ('gemma-...' sorts before 'mtp-...') and by living in an `MTP/`
    subdirectory the non-recursive glob could not see -- both accidents, not decisions."""
    assert wm._is_mtp_head_file("mtp-gemma-4-31B-it-Q8_0.gguf")
    assert not wm._is_mtp_head_file("gemma-4-31B-it-Q4_K_M.gguf")
    resolved = wm._resolve_gguf(wm.ARC_LIVE_GENERATOR_REPO_SUBSTR)
    if resolved:  # only assert when the model is actually cached on this machine
        assert not wm._is_mtp_head_file(resolved.split("/")[-1])


def test_the_head_constants_describe_the_real_file(wm) -> None:
    """`gemma4-assistant` is read from the head GGUF's own header, not assumed. If the filename or
    architecture ever changes upstream, the resolver and the kernel's name filter both break, so
    the constants are pinned here."""
    assert wm.ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME == "mtp-gemma-4-31B-it-Q8_0.gguf"
    assert wm.ARC_LIVE_GENERATOR_MTP_HEAD_SUBSTR in wm.ARC_LIVE_GENERATOR_MTP_HEAD_FILENAME
    assert wm.ARC_LIVE_GENERATOR_MTP_HEAD_ARCH == "gemma4-assistant"


# ==============================================================================================
# The local/scored MTP split, and the Kaggle wiring.
# ==============================================================================================


def test_local_and_scored_mtp_defaults_are_distinct_named_constants(wm) -> None:
    """They are two different hardware answers to the same question. Collapsing them into one is
    how a submission silently ships the slower configuration: nothing reports it, the run just
    takes ~1.4x longer per induction."""
    assert wm.ARC_LIVE_GENERATOR_MTP_DEFAULT == "0"  # 24 GB dev card: offload cost > MTP gain
    assert wm.ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT == "1"  # 96 GB scored card: pure ~1.4x win


def test_mtp_default_on_reads_the_env_against_the_local_constant(wm, monkeypatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_MTP", raising=False)
    assert wm._mtp_default_on() is (wm.ARC_LIVE_GENERATOR_MTP_DEFAULT != "0")
    monkeypatch.setenv("CARNOT_ARC_MTP", "1")
    assert wm._mtp_default_on() is True
    monkeypatch.setenv("CARNOT_ARC_MTP", "0")
    assert wm._mtp_default_on() is False


def test_the_mtp_head_dataset_is_attached_by_the_kernel() -> None:
    """TASK 3. The head is a SECOND Kaggle dataset because it is a second file. If it drops out of
    `dataset_sources` the kernel finds no head, correctly disables MTP, and the scored run is
    ~1.4x slower with nothing but one log line to show for it."""
    import json
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    meta = json.loads(
        (repo / "scripts" / "kaggle" / "submission_kernel" / "kernel-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    sources = meta["dataset_sources"]
    assert "iancblenke/carnot-gemma4-31b-mtp-head" in sources, sources
    assert "iancblenke/carnot-gemma4-31b-it-gguf" in sources, sources
    # Both must be present: the head alone is not a generator, and the weights alone are not MTP.
    assert len({s for s in sources if "gemma4-31b" in s}) == 2, sources


def test_the_kernel_disambiguates_the_two_ggufs_by_name_not_by_rglob_order() -> None:
    """Both attached datasets mount under /kaggle/input and both contain a `*.gguf`. The previous
    filter (`"gemma-4-31B" in name or "Q4_K_M" in name`) MATCHED THE HEAD TOO -- its filename is
    `mtp-gemma-4-31B-it-Q8_0.gguf` -- so the main model was chosen by rglob order between two
    matching files. Binding the 491 MB head as the generator loads and serves and answers nonsense.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    src = (repo / "scripts" / "kaggle" / "submission_kernel" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "_mains = [" in src and "_heads = [" in src, (
        "the kernel no longer separates main-model candidates from MTP-head candidates"
    )
    assert '--model-draft", str(gguf)' not in src and "--model-draft', str(gguf)" not in src, (
        "the kernel probe drafts against the MAIN gguf -- the configuration in which llama.cpp "
        "silently disables speculation while still reporting a healthy server"
    )
    assert "adding speculative implementation 'draft-mtp'" in src, (
        "the kernel probe no longer asserts the POSITIVE MTP marker; a healthy server is NOT "
        "evidence that speculative decoding is engaged"
    )


# ==============================================================================================
# THIRD-PASS FINDINGS (2026-07-28). Defects found reviewing the fixes for the four above.
# ==============================================================================================


def test_head_detection_matches_the_prefix_not_the_substring(wm) -> None:
    """`_is_mtp_head_file` must key on the `mtp-` FILENAME PREFIX, not on "mtp-" anywhere.

    THE DEFECT THIS PINS. The first implementation was `"mtp-" in name.lower()`, which is a
    false-positive generator on a naming convention this project actually uses:
    `Qwen3.5-9B-MTP-Q4_K_M.gguf` is MAIN WEIGHTS whose name contains "MTP-". Two silent
    consequences followed:

      * `_resolve_gguf` excludes anything this predicate calls a head, so for that repo it
        excluded the ONLY candidate and returned None -- the documented "the retired 9B remains a
        legitimate CARNOT_ARC_GGUF_PATH override for a 16GB-class box" escape hatch could not
        resolve its own weights from cache.
      * Any caller asking "is this a self-drafting MTP build?" got the exact inverse of the truth.

    Upstream's convention is a prefix (`MTP/mtp-gemma-4-31B-it-Q8_0.gguf`), so the prefix is both
    the precise test and the documented one.
    """
    assert wm._is_mtp_head_file("mtp-gemma-4-31B-it-Q8_0.gguf") is True
    assert wm._is_mtp_head_file("MTP/mtp-gemma-4-31B-it-Q8_0.gguf") is True
    assert wm._is_mtp_head_file("gemma-4-31B-it-Q4_K_M.gguf") is False
    # The regression case: main weights whose name CONTAINS "MTP-" but does not start with it.
    assert wm._is_mtp_head_file("Qwen3.5-9B-MTP-Q4_K_M.gguf") is False
    assert wm._is_mtp_head_file("Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf") is False


def test_proposer_fits_the_offload_to_its_own_mtp_not_the_environment_default(
    wm, monkeypatch
) -> None:
    """`LocalGGUFProposer.__post_init__` must size `ffn_cpu_layers` from `self.mtp`.

    THE DEFECT THIS PINS. `ffn_cpu_layers` was a `default_factory`, and a dataclass evaluates
    default factories with NO access to sibling fields -- so the auto-fit sized the offload against
    the ENVIRONMENT's mtp answer while the server launches with the CONSTRUCTOR's. Eight-plus
    harnesses construct `LocalGGUFProposer(mtp=...)` explicitly, so the two disagreed routinely,
    and the result was a guard validating a configuration the server was not about to run --
    CUDA declined, iGPU fallback at ~2 tok/s, induce timeout, silent LLM-off.

    FREE-VRAM READING RAISED 24123 (RTX3090_IDLE_FREE_MIB) -> 24576 (RTX3090_TOTAL_MIB), 2026-08-08
    (REQ-ARC-WMTE-6227). `_default_induce_n_ctx()` moved 81920 -> 106496 (see the shipped-local-
    default test above), and at RTX3090_IDLE_FREE_MIB BOTH mtp arms now refuse (both return 0/-1
    layers, per `test_mtp_on_is_refused_locally_rather_than_auto_offloaded_into_a_timeout`'s own
    2026-08-08 update) -- which would make this test's own-mtp-vs-env-mtp comparison vacuous, since
    "sized from the env's mtp" and "sized from this proposer's own mtp" produce the SAME (refused)
    answer when neither can fit at all. This test's PURPOSE is narrower than realistic-idle-VRAM
    coverage (that is what the shipped-local-default test is for): it only needs ONE free-VRAM
    reading where the two mtp arms still land on DIFFERENT layer counts, so the property under
    test (own-mtp, not env-mtp) stays demonstrable. RTX3090_TOTAL_MIB is that reading here (11
    layers vs refused, swept empirically against the real module functions) -- a generous,
    already-established constant in this file, not a fabricated one.
    """
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setenv("CARNOT_ARC_MTP", "0")  # environment says OFF
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: RTX3090_TOTAL_MIB)
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: RTX3090_TOTAL_MIB)

    # A proposer that agrees with the environment gets the environment's fit.
    env_fit = wm.LocalGGUFProposer(mtp=False).ffn_cpu_layers
    assert env_fit > 0, (
        f"sanity: this test needs the mtp=False arm to actually fit (nonzero layers) at "
        f"RTX3090_TOTAL_MIB free, or it cannot demonstrate the own-mtp-vs-env-mtp property; got "
        f"{env_fit}"
    )
    # A proposer that DISAGREES must be fitted for ITS OWN value, and must say so.
    p = wm.LocalGGUFProposer(mtp=True)
    assert p.ffn_cpu_layers != env_fit, (
        "the offload was sized from the env default, not from this proposer's mtp"
    )
    assert "differs from the environment default" in p.ffn_cpu_layers_refit_note


def test_an_explicit_ffn_cpu_layers_is_never_overridden(wm, monkeypatch) -> None:
    """A caller who NAMES a layer count has stated a fact about the launch they want.

    The sentinel exists precisely so the auto-fit can be distinguished from an explicit 0; silently
    overriding a named value would be the same class of error in the other direction, and the tests
    that pin exact argv for a given layer count depend on it surviving.
    """
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: RTX3090_IDLE_FREE_MIB)
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: RTX3090_TOTAL_MIB)
    for explicit in (0, 3, 12):
        assert wm.LocalGGUFProposer(ffn_cpu_layers=explicit).ffn_cpu_layers == explicit


def test_mtp_engagement_is_read_from_stderr_not_inferred_from_a_healthy_server(
    wm, tmp_path
) -> None:
    """`_verify_mtp_engaged` must decide on the POSITIVE marker, never on server health.

    THE DEFECT THIS PINS. This module states the doctrine in capitals -- "never conclude MTP is
    enabled from a healthy server or an absent error" -- and then did not apply it to its own
    server. The only place the marker was ever grepped was the Kaggle kernel's pre-flight probe: a
    different process, on a different port, torn down before the agent starts. Inside the proposer,
    `last_mtp_draft_path` recorded what was PASSED to `--model-draft`, which is a fact about our
    argv, not about whether the runtime used it.

    Three-valued on purpose: None ("could not determine" -- stderr was not captured) must not
    collapse into False, and above all an unreadable log must never read as success.
    """
    p = wm.LocalGGUFProposer(mtp=True, ffn_cpu_layers=0)
    p.last_mtp_draft_path = "/fake/mtp-gemma-4-31B-it-Q8_0.gguf"

    # ENGAGED: the real positive marker from a successful launch.
    log = tmp_path / "ok.log"
    log.write_text(
        "I srv    load_model: loading draft model '/fake/mtp-gemma-4-31B-it-Q8_0.gguf'\n"
        "I common_speculative_impl_draft_mtp: adding speculative implementation 'draft-mtp'\n"
        "I srv    load_model: speculative decoding context initialized\n"
    )
    p._stderr_log_path = log
    p._verify_mtp_engaged()
    assert p.last_mtp_engaged is True

    # SILENTLY DISABLED: the server is healthy, the marker is absent, the warnings are present.
    log2 = tmp_path / "degraded.log"
    log2.write_text(
        "W llama_init_from_model: context type MTP requested but model doesn't contain MTP layers\n"
        "W common_speculative_init: no implementations specified for speculative decoding\n"
        "I main: server is listening on 127.0.0.1:8919\n"
    )
    p._stderr_log_path = log2
    p._verify_mtp_engaged()
    assert p.last_mtp_engaged is False
    assert "NOT engaged" in p.mtp_engaged_evidence

    # UNDETERMINED: no log at all is NOT success.
    p._stderr_log_path = tmp_path / "does-not-exist.log"
    p._verify_mtp_engaged()
    assert p.last_mtp_engaged is None
    assert "CANNOT be determined" in p.mtp_engaged_evidence


def test_liveness_witness_publishes_mtp_engagement_separately_from_the_request(
    wm, tmp_path, monkeypatch
) -> None:
    """The witness must distinguish "we asked for MTP" from "the runtime did MTP".

    Before this, an artifact could describe a fully MTP-on run that had speculation silently
    disabled: every other field on the witness is identical either way.
    """
    monkeypatch.setattr(wm.LocalGGUFProposer, "_healthy", lambda _self: False)
    p = wm.LocalGGUFProposer(mtp=True, ffn_cpu_layers=0)
    p.last_mtp_draft_path = "/fake/mtp-gemma-4-31B-it-Q8_0.gguf"
    p.last_mtp_engaged = False
    w = p.liveness_witness()
    assert w["generator_mtp_requested"] is True
    assert w["generator_mtp_engaged"] is False
    assert w["generator_mtp_draft_path"].endswith("mtp-gemma-4-31B-it-Q8_0.gguf")
