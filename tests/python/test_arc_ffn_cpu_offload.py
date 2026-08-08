"""The opt-in dense-FFN-to-system-RAM offload knob (`CARNOT_ARC_FFN_CPU_LAYERS`).

REQ-ARC-WMTE-6020 / SCENARIO-ARC-WMTE-6020-FFN-OFFLOAD-REACHES-THE-SERVER

WHY THIS FILE EXISTS. The operator directive of 2026-07-28 asked for the FFN weights to be
offloadable to system RAM so an 18.3 GB gemma-4-31B generator can share a 24 GB local card with
the rest of the live agent. The dangerous outcome for a feature like this is not that it fails --
it is that it SILENTLY DOES NOTHING:

  * `--cpu-moe` / `--n-cpu-moe` are the flags that look right, and they are accepted by
    llama-server, and on this DENSE model they match zero tensors. The GGUF contains
    `blk.<i>.ffn_{gate,up,down}.weight` and no `ffn_*_exps` at all. A run with `-cmoe` would come
    up healthy, generate correctly, and free exactly 0 MiB.
  * An `-ot` regex written as `blk\\.[0-9]+\\.ffn_...` matches EVERY block, so asking for 12 CPU
    layers would offload all 60 and collapse throughput by ~13x for reasons nothing reports.
  * A flag that is built but never appended to the Popen argv is indistinguishable, from outside
    the process, from one that works.

So these tests assert the three things that can actually be wrong: the default path gains NO
argument, the regex names the REAL tensors and only the requested indices, and the flag reaches
the exact argv handed to subprocess.Popen.

Measurement provenance for the numbers referenced here (RTX 3090, gemma-4-31B-it Q4_K_M, n_ctx
32768, q8_0 KV, per-PID `nvidia-smi --query-compute-apps` residency joined PID -> GPU UUID ->
index): 0/12/24/40 CPU FFN layers -> 21416 / 19072 / 16728 / 13580 MiB, decode 36.14 / 15.17 /
9.81 / 6.33 tok/s. Freed VRAM is linear at ~195 MiB per layer; throughput is emphatically not.
"""

from __future__ import annotations

import importlib

import pytest

MOD = "carnot.agentic.arc_executable_world_model"


@pytest.fixture()
def wm(monkeypatch):
    """Import the module with a clean env so one test's override cannot leak into another.

    `CARNOT_ARC_GENERATOR_CUDA_GPU` is cleared too, and that one is not cosmetic: since the
    2026-07-28 auto-fit fix, an UNSET `CARNOT_ARC_FFN_CPU_LAYERS` no longer implies 0 -- it implies
    "0 unless a local CUDA card was opted into, in which case fit to it". The conductor's standing
    systemd drop-in sets that variable, so a suite run under the conductor's environment would
    otherwise see `_default_ffn_cpu_layers()` return a card-dependent number and the
    default-is-off tests would fail for a reason that has nothing to do with what they test.
    """
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
    return importlib.import_module(MOD)


# --------------------------------------------------------------------------------------------
# The regex: real tensor names, only the requested blocks.
# --------------------------------------------------------------------------------------------


def _pattern_only(ot_value: str) -> str:
    """llama.cpp's `-ot` argument is `<tensor-name-regex>=<buffer-type>`. Everything before the
    final `=` is the regex; `CPU` after it is the destination buffer. Tests that want to MATCH
    tensor names have to strip the destination first, or every match fails for the wrong reason."""
    pattern, _, dest = ot_value.rpartition("=")
    assert dest == "CPU", ot_value
    return pattern


def test_regex_targets_the_real_dense_ffn_tensor_names(wm) -> None:
    """Read out of the actual GGUF tensor table, not guessed."""
    rx = wm._ffn_cpu_override_regex(2)
    assert rx.endswith("=CPU"), rx
    pattern = _pattern_only(rx)
    # The three dense FFN tensors of a gemma-4 block, as an alternation.
    assert "ffn_(gate|up|down)" in pattern, pattern
    # The MoE tensor name must NOT appear: this model is dense, and pretending otherwise is how
    # `-cmoe` becomes a silent no-op.
    assert "exps" not in pattern


def test_regex_matches_only_the_requested_block_indices(wm) -> None:
    """The failure this guards: a numeric-range-looking pattern that offloads the whole model.

    llama.cpp's override matcher is a plain regex with no numeric-range support, so `[0-9]+`
    would match block 59 as happily as block 0.
    """
    import re

    rx = re.compile(_pattern_only(wm._ffn_cpu_override_regex(3)) + "$")
    for i in (0, 1, 2):
        for t in ("ffn_gate", "ffn_up", "ffn_down"):
            assert rx.match(f"blk.{i}.{t}.weight"), (i, t)
    for i in (3, 12, 30, 59):
        assert not rx.match(f"blk.{i}.ffn_gate.weight"), i
    # block 12 must not be swept in by the alternation branch for block 1
    assert not rx.match("blk.12.ffn_up.weight")
    # attention/norm tensors of an OFFLOADED block must stay on the GPU -- only FFN moves
    assert not rx.match("blk.0.attn_q.weight")
    assert not rx.match("blk.0.ffn_norm.weight")


def test_regex_is_empty_when_offload_is_off(wm) -> None:
    """An empty pattern is how the caller knows to append NOTHING (see the argv test below)."""
    assert wm._ffn_cpu_override_regex(0) == ""
    assert wm._ffn_cpu_override_regex(-4) == ""


# --------------------------------------------------------------------------------------------
# The env knob.
# --------------------------------------------------------------------------------------------


def test_default_is_off(wm, monkeypatch) -> None:
    """Unset env must mean byte-identical behaviour to before this knob existed."""
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    assert wm._default_ffn_cpu_layers() == 0
    assert wm.LocalGGUFProposer(repo_substr="x").ffn_cpu_layers == 0


def test_env_knob_is_read(wm, monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", "12")
    assert wm._default_ffn_cpu_layers() == 12
    assert wm.LocalGGUFProposer(repo_substr="x").ffn_cpu_layers == 12


@pytest.mark.parametrize("bad", ["twelve", "", "  ", "12.5"])
def test_a_malformed_value_degrades_to_off_rather_than_crashing_the_live_path(
    wm, monkeypatch, bad
) -> None:
    """A typo must not take down the generator. It falls back to 0 -- and because the launch argv
    is recorded, the absent `-ot` is visible rather than mysterious."""
    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", bad)
    assert wm._default_ffn_cpu_layers() == 0


# --------------------------------------------------------------------------------------------
# The part that actually matters: does the flag REACH the server process?
# --------------------------------------------------------------------------------------------


def _launch_and_capture_argv(wm, monkeypatch, tmp_path, *, ffn_cpu_layers: int) -> list[str]:
    """Drive the real `_ensure_server()` with a fake Popen and return the argv it built.

    Everything faked here is an EXTERNAL dependency (the binary on disk, the GGUF on disk, the
    subprocess, the health poll). The argv construction under test is the real code path.
    """
    fake_server = tmp_path / "llama-server"
    fake_server.write_text("#!/bin/sh\nexit 0\n")
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"\0")

    monkeypatch.setattr(
        wm,
        "_generator_server_and_env",
        # Accepts the ffn_cpu_layers argument the real function now takes: `_ensure_server()`
        # threads the ACTUAL layer count through so the VRAM guard budgets for the offload the
        # server will really launch with (see `_generator_cuda_min_free_mb`). A 0-arg stub here
        # would silently under-test that wiring by never exercising the call shape.
        # Takes BOTH threaded arguments as of 2026-07-28: `_ensure_server()` passes the actual
        # ffn_cpu_layers AND the actual `mtp` so the VRAM guard budgets for the configuration the
        # server will really launch with. The MTP draft head is a real, n_ctx-scaled cost
        # (+1290 MiB at the shipped n_ctx 81920), so a guard blind to it validates a different
        # server than the one about to start -- the same class of gap the ffn_cpu_layers thread
        # closed. A narrower stub here would silently stop exercising that wiring.
        lambda _ffn_cpu_layers=None, _mtp=None: (fake_server, None),
    )
    monkeypatch.setattr(wm, "_resolve_gguf", lambda _s: str(fake_gguf))

    captured: dict[str, list[str]] = {}

    class _FakeProc:
        pid = 1234

    def _fake_popen(args, **_kw):
        captured["argv"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(wm.subprocess, "Popen", _fake_popen)

    prop = wm.LocalGGUFProposer(
        repo_substr="gemma-4-31B-it", n_ctx=32768, kv_quant="q8_0", ffn_cpu_layers=ffn_cpu_layers
    )
    # First poll must say "no server yet" so we take the launch path; the next says "up" so the
    # 90-attempt wait loop exits immediately instead of sleeping 180 real seconds.
    calls = {"n": 0}

    def _healthy():
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(prop, "_healthy", _healthy)

    assert prop._ensure_server() is True
    assert "argv" in captured, "_ensure_server never reached subprocess.Popen"
    # The instance must record what it launched -- that record is what an artifact can cite.
    assert list(prop.last_launch_argv) == captured["argv"]
    return captured["argv"]


def test_default_launch_argv_contains_no_override_tensor_argument(wm, monkeypatch, tmp_path):
    """The opt-in contract: with the knob off, the argv is exactly what it was before."""
    argv = _launch_and_capture_argv(wm, monkeypatch, tmp_path, ffn_cpu_layers=0)
    assert "-ot" not in argv
    assert "--override-tensor" not in argv
    assert not wm.LocalGGUFProposer(repo_substr="x").last_ffn_cpu_override


def test_offload_flag_reaches_the_launch_argv_with_the_right_regex(wm, monkeypatch, tmp_path):
    """THE test. A flag that is accepted and ignored is worse than no flag."""
    argv = _launch_and_capture_argv(wm, monkeypatch, tmp_path, ffn_cpu_layers=12)
    assert "-ot" in argv, argv
    value = argv[argv.index("-ot") + 1]
    assert value == wm._ffn_cpu_override_regex(12)
    assert "blk\\.(0|1|2|3|4|5|6|7|8|9|10|11)\\.ffn_(gate|up|down)\\.weight=CPU" == value
    # and NOT the MoE flags, which this dense model would silently ignore
    assert "-cmoe" not in argv and "--cpu-moe" not in argv and "--n-cpu-moe" not in argv


def test_the_env_knob_alone_is_enough_to_get_the_flag_onto_the_argv(wm, monkeypatch, tmp_path):
    """The operator sets an env var, not a constructor argument. Prove that path end-to-end."""
    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", "24")
    fake_server = tmp_path / "llama-server"
    fake_server.write_text("x")
    fake_gguf = tmp_path / "m.gguf"
    fake_gguf.write_bytes(b"\0")
    monkeypatch.setattr(
        wm,
        "_generator_server_and_env",
        # Accepts the ffn_cpu_layers argument the real function now takes: `_ensure_server()`
        # threads the ACTUAL layer count through so the VRAM guard budgets for the offload the
        # server will really launch with (see `_generator_cuda_min_free_mb`). A 0-arg stub here
        # would silently under-test that wiring by never exercising the call shape.
        # Takes BOTH threaded arguments as of 2026-07-28: `_ensure_server()` passes the actual
        # ffn_cpu_layers AND the actual `mtp` so the VRAM guard budgets for the configuration the
        # server will really launch with. The MTP draft head is a real, n_ctx-scaled cost
        # (+1290 MiB at the shipped n_ctx 81920), so a guard blind to it validates a different
        # server than the one about to start -- the same class of gap the ffn_cpu_layers thread
        # closed. A narrower stub here would silently stop exercising that wiring.
        lambda _ffn_cpu_layers=None, _mtp=None: (fake_server, None),
    )
    monkeypatch.setattr(wm, "_resolve_gguf", lambda _s: str(fake_gguf))
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        wm.subprocess,
        "Popen",
        lambda args, **_k: (captured.__setitem__("a", list(args)), type("P", (), {"pid": 1})())[1],
    )
    prop = wm.LocalGGUFProposer(repo_substr="gemma-4-31B-it")  # no explicit ffn_cpu_layers
    calls = {"n": 0}
    monkeypatch.setattr(
        prop, "_healthy", lambda: (calls.__setitem__("n", calls["n"] + 1), calls["n"] > 1)[1]
    )
    assert prop._ensure_server() is True
    assert "-ot" in captured["a"]
    assert captured["a"][captured["a"].index("-ot") + 1] == wm._ffn_cpu_override_regex(24)


# --------------------------------------------------------------------------------------------
# The guard must know about the lever, or it declines a card the server would have fitted on.
# --------------------------------------------------------------------------------------------


def test_vram_guard_credits_the_offloaded_layers(wm, monkeypatch) -> None:
    """~195 MiB per layer, measured. A guard blind to the knob is a guard that refuses the fix."""
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    base = wm._generator_cuda_min_free_mb()
    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", "12")
    with_offload = wm._generator_cuda_min_free_mb()
    assert with_offload < base
    freed = base - with_offload
    # 12 * 195.3 = 2343.6; measured freed VRAM at 12 layers was exactly 2344 MiB.
    assert 2300 <= freed <= 2400, freed


def test_guard_at_defaults_exceeds_a_3090_which_is_the_honest_answer(wm, monkeypatch) -> None:
    """THE ARITHMETIC IS RIGHT; THE CONCLUSION DRAWN FROM IT WAS WRONG, AND THIS TEST HAS BEEN
    RETARGETED ACCORDINGLY.

    What is still true, and still asserted below: with NO offload, gemma-4-31B at n_ctx 81920
    resides at 23888 MiB and the guard therefore demands more than a 24576 MiB 3090 has in total.
    That is a measurement and a future 'fix' that quietly lowers the guard to make the card pass
    must still argue with it.

    What this test USED to conclude -- that declining the card was 'the honest answer' and an
    acceptable DEFAULT -- rested on an unmeasured premise: that the iGPU HIP fallback was a
    functional, if slower, place to land. It is not. Measured 2026-07-28, same methodology as this
    file's CUDA table: ~2 tok/s decode for this model, against `max_tokens=4096` and a 600 s
    induce timeout. Every induce call times out, `generate()` returns `(False, msg)`, and the agent
    runs LLM-OFF while reporting itself LLM-on. So the old default was an outage, and this test --
    by asserting the outage was correct -- would have made repairing it look like a regression.

    The retargeted contract is the one that actually matters to a caller: WITH THE SHIPPED
    DEFAULTS AND A LOCAL CUDA OPT-IN, THE GENERATOR MUST END UP SOMEWHERE USABLE. The no-offload
    arithmetic is kept as a component assertion, not as the definition of correct behaviour.

    RETARGETED AGAIN 2026-08-08 (REQ-ARC-WMTE-6227). `_INDUCE_WORST_CASE_PROMPT_TOKENS` moved
    15767 -> 22352 (a re-measurement under current defaults -- k=all transitions, object table on
    -- found the old constant stale; see that constant's own comment), raising
    `_default_induce_n_ctx()` 81920 -> 106496. At the new n_ctx, the max auto-fit offload (12
    layers, `_FFN_CPU_AUTOFIT_MAX_LAYERS`) no longer brings the footprint under a realistic-idle
    3090's free VRAM -- confirmed directly (a no-offload launch at 106496 does not even fit a
    24576 MiB card at all: `cudaMalloc failed: out of memory` on the compute-buffer reservation,
    not a graceful decline). Raising the offload cap is NOT the fix: `_FFN_CPU_AUTOFIT_MAX_LAYERS`
    was calibrated against a decode-throughput/induce-timeout tradeoff, and a BIGGER worst-case
    prompt makes that tradeoff WORSE at more offloaded layers, not better -- so a cap raise would
    reintroduce the exact `test_mtp_on_is_refused_locally_rather_than_auto_offloaded_into_a_timeout`
    -style timeout risk (in `test_arc_generator_migration_defects.py`) for the mtp-OFF arm too.

    So THIS test's "must end up somewhere usable AT THE SHIPPED DEFAULTS" contract is retargeted
    once more: the shipped DEFAULT n_ctx correctly REFUSES a realistic-idle 3090 for the absolute
    worst-case prompt (a genuine, measured physical constraint, not a regression to compensate
    for), and the property that must still hold is that the DOCUMENTED escape hatch --
    `CARNOT_ARC_INDUCE_N_CTX` -- actually restores usability for a local developer who accepts a
    smaller worst-case-prompt safety margin. That is what "must end up somewhere usable" now
    means: not at ANY n_ctx unconditionally, but reachable via the lever the module's own
    docstring already names for exactly this box class.
    """
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    # Component fact (unchanged, still measured): with zero offload the requirement exceeds a 3090.
    assert wm._generator_cuda_min_free_mb(0) > 24576
    # ...and the documented escape hatch actually escapes: 12 offloaded layers brings the
    # requirement under an idle 3090's free memory -- AT THE OLD n_ctx (81920) this constant's
    # own historical basis. The default n_ctx no longer satisfies this at all (see below), which
    # is exactly the regression this component assertion exists to catch if reintroduced silently.
    with monkeypatch.context() as m:
        m.setenv("CARNOT_ARC_INDUCE_N_CTX", "81920")
        assert wm._generator_cuda_min_free_mb(12) < 24576

    # THE SHIPPED-DEFAULT CONTRACT (2026-08-08: the corrected worst case genuinely refuses this
    # card, and that refusal must be loud, not silent).
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: 24123)
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: 24576)
    wm.GENERATOR_SELECTION_LOG.clear()
    wm._GENERATOR_SELECTION_SEEN.clear()
    layers = wm._default_ffn_cpu_layers()
    assert layers == 0, (
        "the shipped default must REFUSE (not silently under-offload) on a realistic-idle 3090 "
        f"at the corrected worst-case n_ctx; got {layers} nonzero layers -- if this now fits, "
        "either the worst-case prompt constant shrank again or the autofit cap moved, and this "
        "test's numbers need a fresh re-derivation"
    )
    joined = "\n".join(wm.GENERATOR_SELECTION_LOG)
    assert "cannot fit the generator" in joined, f"the refusal must be logged; got:\n{joined}"

    # THE ESCAPE HATCH CONTRACT: a local developer who explicitly accepts a smaller worst-case
    # safety margin (CARNOT_ARC_INDUCE_N_CTX=81920, this constant's own historical value) DOES get
    # a usable card -- "must end up somewhere usable" still holds, reachable via the documented
    # lever, which is the property this test was originally written to protect.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_N_CTX", "81920")
    layers = wm._default_ffn_cpu_layers()
    assert 0 < layers <= wm._FFN_CPU_AUTOFIT_MAX_LAYERS, (
        "the documented CARNOT_ARC_INDUCE_N_CTX escape hatch must restore a usable local card; "
        f"got {layers} layers at n_ctx=81920"
    )
    assert wm._generator_cuda_min_free_mb(layers) <= 24123, (
        "auto-fit picked a layer count that still does not satisfy the guard it was fitted "
        "against -- the card would be declined and the iGPU outage would recur"
    )
    # ...and the end-to-end placement decision agrees: we get the CUDA build, pinned to the card.
    server, env = wm._generator_server_and_env(layers)
    assert server.name == "llama-server"
    assert env is not None and env.get("CUDA_VISIBLE_DEVICES") == "0", (
        "guard passed but the CUDA build was not selected -- the two halves disagree"
    )


def test_a_card_too_full_for_any_capped_offload_is_declined_loudly_not_silently(
    wm, monkeypatch
) -> None:
    """The other branch of the fatal finding. When the card genuinely cannot host the generator
    (a conductor job is holding it), auto-fit must NOT keep offloading until the CUDA path is
    slower than the fallback -- it must give up at the cap. And the give-up must be VISIBLE:
    the pre-fix code fell through to the iGPU with zero output, so a run that lost its card was
    indistinguishable from one that never asked for it.
    """
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    monkeypatch.setattr(wm, "_cuda_gpu_free_mb", lambda _idx: 670)  # a busy 3090
    monkeypatch.setattr(wm, "_cuda_gpu_total_mb", lambda _idx: 24576)
    # A fresh log + dedupe set, so this assertion reads THIS call's output and not a cached line
    # from an earlier test in the same process.
    monkeypatch.setattr(wm, "GENERATOR_SELECTION_LOG", [])
    monkeypatch.setattr(wm, "_GENERATOR_SELECTION_SEEN", set())

    assert wm._default_ffn_cpu_layers() == 0, "must not offload past the throughput cap"
    server, env = wm._generator_server_and_env(0)
    assert env is None, "a card with 670 MiB free must not be bound"
    assert server.name == "llama-server"  # the HIP fallback binary, same filename
    joined = "\n".join(wm.GENERATOR_SELECTION_LOG)
    assert "DECLINED" in joined, f"the refusal was silent again:\n{joined}"
    assert "FALLING BACK" in joined, f"the fallback was silent again:\n{joined}"
    # The reader must be told the fallback cannot do the job, not merely that it happened.
    assert "tok/s" in joined and "LLM-OFF" in joined, joined
