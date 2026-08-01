"""Layer-splitting the live ARC generator across multiple CUDA cards.

Spec coverage: REQ-ARC-WMTE-5717

Origin: 2026-07-31 operator directive "wire the split into the live agent config", after a
five-arm sweep on 2x RTX 3090 (results/outer_loop_arc_gpu_layer_split_sweep_20260731.json)
measured layer-splitting as FREE: +1.0% decode at n_ctx 32768 and +0.6% at the shipped 81920,
while cutting peak per-card residency 47.2% / 45.9%.

WHAT THE SPLIT ACTUALLY BUYS, measured through the live `_ensure_server()` path at the shipped
n_ctx 81920 (not extrapolated from the sweep harness):

    single card (CARNOT_ARC_GENERATOR_CUDA_GPU=1)   20.47 tok/s decode, 288 prefill, ffn_cpu=7
    layer split (CARNOT_ARC_GENERATOR_CUDA_GPU=0,1) 38.83 tok/s decode, 908 prefill, ffn_cpu=0
    -> +89.7% decode, +215% prefill

A CORRECTION THIS FILE EXISTS TO PRESERVE. The first version of this change was justified by the
claim that the single-card guard is unsatisfiable (`_generator_cuda_min_free_mb()` = 25388 MiB
vs a 3090's 24576 total), so the generator "falls through to the ~2 tok/s iGPU build and the
agent runs LLM-off". That is WRONG, and the counterfactual run disproved it: the single-card
path does not fail over at all. The auto-fit re-reads free VRAM and spills FFN layers to system
RAM until the model fits -- landing on ffn_cpu_layers=7 -- and the launch SUCCEEDS on the CUDA
build. Nothing errors. That is exactly why the cost was invisible: the only symptom is roughly
half the throughput.

So the split avoids a forced CPU offload; it does not rescue a launch that would otherwise not
happen. Both configurations run. One runs degraded.

WHAT THE LEVER ACTUALLY IS. Not `-sm layer` -- that is llama.cpp's DEFAULT. It is GPU
VISIBILITY: `CUDA_VISIBLE_DEVICES=str(idx)` pinned the server to one card, so it could never
split whatever flags it was given. The flags are passed anyway so the argv documents itself.
"""

import pytest

from carnot.agentic import arc_executable_world_model as wm


class TestParsing:
    """CARNOT_ARC_GENERATOR_CUDA_GPU gained a list form; the scalar form must not shift."""

    def test_single_index_unchanged(self) -> None:
        assert wm._parse_generator_cuda_gpus("1") == [1]

    def test_comma_list(self) -> None:
        assert wm._parse_generator_cuda_gpus("0,1") == [0, 1]

    def test_whitespace_tolerated(self) -> None:
        assert wm._parse_generator_cuda_gpus(" 0 , 1 ") == [0, 1]

    @pytest.mark.parametrize("raw", ["", "abc", "0,abc", "-1", "0,0", "1,1"])
    def test_malformed_yields_empty_not_a_guess(self, raw: str) -> None:
        """A typo must reach the caller's refusal path, never bind an arbitrary card.

        Duplicates ("0,0") are rejected rather than deduped: the operator asked for a two-card
        split and would silently get a one-card pin sized for two, which is the unsatisfiable
        arithmetic this change exists to avoid.
        """
        assert wm._parse_generator_cuda_gpus(raw) == []


class TestSplitArgsAreDerivedFromTheLaunchEnv:
    """The `-ts` ratio must describe the cards the process can actually see."""

    def test_two_cards_get_even_split_flags(self) -> None:
        assert wm._split_args_for_env({"CUDA_VISIBLE_DEVICES": "0,1"}) == [
            "-sm",
            "layer",
            "-ts",
            "1,1",
        ]

    def test_four_cards_get_four_ratios(self) -> None:
        args = wm._split_args_for_env({"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        assert args[-1] == "1,1,1,1"

    def test_ratio_is_positional_not_physical_index(self) -> None:
        """`-ts` counts VISIBLE devices. Cards "1,2" still get "1,1", not "1,1,1"."""
        assert wm._split_args_for_env({"CUDA_VISIBLE_DEVICES": "1,2"})[-1] == "1,1"

    def test_single_card_gets_nothing(self) -> None:
        """Split flags on one visible device are meaningless; the argv stays as it was."""
        assert wm._split_args_for_env({"CUDA_VISIBLE_DEVICES": "1"}) == []

    def test_scored_path_is_left_alone(self) -> None:
        """env=None is the Kaggle path: ambient env, all devices visible.

        llama.cpp already layer-splits by default there. Passing an explicit `-ts` would be
        guessing at a placement never measured on L4s, so this deliberately returns [].
        """
        assert wm._split_args_for_env(None) == []
        assert wm._split_args_for_env({}) == []


class TestTheArithmeticThatMotivatedThis:
    """One card cannot hold the shipped config WITHOUT offload; two cards can."""

    def test_single_card_requirement_exceeds_a_3090(self) -> None:
        """The fact that forces the auto-fit's CPU offload on a single card.

        This does NOT mean the single-card launch fails -- see the module docstring's
        correction; the auto-fit spills FFN layers until it fits and succeeds. It means the
        single-card path cannot run at ffn_cpu_layers=0, which is where the throughput goes.

        If a future n_ctx or quant change brings this back under 24576, the auto-fit stops
        firing, the single-card path stops being degraded, and the split's ~90% advantage should
        be re-measured rather than assumed -- hence a test rather than a comment.
        """
        assert wm._generator_cuda_min_free_mb() > 24576

    def test_split_fits_without_offload_where_one_card_does_not(self) -> None:
        """THE test. Two cards clear a per-card bar that one card cannot clear un-offloaded."""
        need = wm._generator_cuda_min_free_mb()
        per_card = int(need * wm._SPLIT_AGGREGATE_OVERHEAD / 2) + 1
        assert need > 24576, "single-card need should exceed a 3090, forcing the auto-fit"
        assert per_card < 24576, "two-way split should fit on a 3090 with no offload"

    def test_overhead_multiplier_is_above_one(self) -> None:
        """Splitting must never be modelled as SAVING aggregate memory.

        Measured: aggregate rose x1.048 at 32768 and x1.074 at 81920. A multiplier <= 1.0 would
        under-budget every card and reintroduce the cudaMalloc-then-silent-LLM-off failure.
        """
        assert wm._SPLIT_AGGREGATE_OVERHEAD > 1.0


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """A HOME containing both llama.cpp builds where the module actually looks for them.

    The module computes `Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin"`. An
    earlier draft of these tests created the binary at `tmp_path/build/bin`, so `cuda.exists()`
    was False and the function fell through to the iGPU path -- the test failed for a reason
    that had nothing to do with splitting. Centralised here so the layout is stated once.
    """
    base = tmp_path / ".cache" / "llama.cpp-master"
    cuda = base / "build" / "bin" / "llama-server"
    hip = base / "build-hip" / "bin" / "llama-server"
    for exe in (cuda, hip):
        exe.parent.mkdir(parents=True)
        exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(wm.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
    return cuda, hip


class TestPartialHeadroomIsARefusal:
    """A partially-available split must not be PARTIALLY APPLIED.

    Note the boundary against `TestRefusedSplitDegradesToOneCardNotToTheIGpu` below, since an
    earlier draft of this docstring asserted the opposite of what that class now requires.
    Two distinct claims:

      * this class -- the SPLIT itself is all-or-nothing. No comma-list env, no `-ts` sized for
        cards that were refused, no narrower split (fewer cards means a HIGHER per-card bar,
        which is the arithmetic just refused).
      * that class -- once the split is refused, the FALLBACK is a plain single-card pin, not
        the iGPU.

    Refusing to split and degrading to one card are the same decision seen from two sides.
    """

    def test_split_refused_when_one_card_is_busy(self, monkeypatch, fake_home) -> None:
        """A busy second card must not yield a comma-list env sized for two cards."""
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", lambda idx, _mb: idx == 0)

        _server, env = wm._generator_server_and_env()
        assert env is None or "," not in (env.get("CUDA_VISIBLE_DEVICES") or ""), (
            "a partially-available split must not bind a comma list"
        )

    def test_both_cards_free_yields_a_comma_list_env(self, monkeypatch, fake_home) -> None:
        cuda, _hip = fake_home
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", lambda _idx, _mb: True)

        server, env = wm._generator_server_and_env()
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert server == cuda
        # and the flags derived from that env describe the same two cards
        assert wm._split_args_for_env(env) == ["-sm", "layer", "-ts", "1,1"]

    def test_single_card_form_still_pins_one_card(self, monkeypatch, fake_home) -> None:
        """The historical scalar form must be byte-for-byte unaffected by this change."""
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", lambda _idx, _mb: True)

        _server, env = wm._generator_server_and_env()
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert wm._split_args_for_env(env) == []

    def test_per_card_bar_is_lower_than_the_single_card_bar(self, monkeypatch, fake_home) -> None:
        """The split must be admitted on a SMALLER per-card number, or it buys nothing.

        Captures the requirement actually passed to the guard, rather than trusting that the
        division happened. A regression that passed `need_total` per card would still produce a
        comma-list env and pass every other test in this file.
        """
        seen: list[int] = []
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        monkeypatch.setattr(
            wm, "_cuda_gpu_has_headroom", lambda _idx, mb: (seen.append(mb), True)[1]
        )
        wm._generator_server_and_env()
        assert seen, "guard was never consulted"
        assert all(mb < wm._generator_cuda_min_free_mb() for mb in seen), (
            f"per-card requirement {seen} should be below the single-card "
            f"{wm._generator_cuda_min_free_mb()}"
        )


class TestScoredPathUnaffected:
    """CARNOT_LLAMA_SERVER short-circuits before any of this."""

    def test_explicit_server_still_returns_none_env(self, monkeypatch, tmp_path) -> None:
        exe = tmp_path / "llama-server"
        exe.write_text("#!/bin/sh\n")
        monkeypatch.setenv("CARNOT_LLAMA_SERVER", str(exe))
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        server, env = wm._generator_server_and_env()
        assert server == exe
        assert env is None, "the scored path must keep inheriting the ambient environment"
        assert wm._split_args_for_env(env) == []


class TestRefusedSplitDegradesToOneCardNotToTheIGpu:
    """Ordering must be split -> single card -> iGPU. Never split -> iGPU.

    Caught by reading the control flow rather than by a failing test, before shipping. The
    first version set `idx = -1` whenever more than one card was requested, so a REFUSED split
    skipped the single-card branch entirely and fell through to the HIP build at ~2 tok/s. That
    would have made `CARNOT_ARC_GENERATOR_CUDA_GPU="0,1"` strictly MORE fragile than `"1"`: any
    transient contention on either card would cost the whole generator (agent runs LLM-off)
    instead of degrading to the ~20 tok/s single-card auto-fit path.
    """

    def test_busy_second_card_still_yields_a_working_single_card_pin(
        self, monkeypatch, fake_home
    ) -> None:
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        # gpu0 clears the single-card bar; gpu1 clears nothing.
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", lambda idx, _mb: idx == 0)

        server, env = wm._generator_server_and_env()
        assert env is not None, "a refused split must NOT fall through to the iGPU"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert server.name == "llama-server" and "build-hip" not in str(server)
        assert wm._split_args_for_env(env) == [], "degraded pin must not carry split flags"

    def test_degrades_to_the_second_card_when_the_first_is_busy(
        self, monkeypatch, fake_home
    ) -> None:
        """The fallback scans the requested list; it is not hardcoded to idxs[0]."""
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", lambda idx, _mb: idx == 1)
        _server, env = wm._generator_server_and_env()
        assert env is not None and env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_no_card_available_falls_through_to_the_igpu(self, monkeypatch, fake_home) -> None:
        """With genuinely nothing free, the iGPU fallback is still the correct last resort."""
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", lambda _idx, _mb: False)
        _server, env = wm._generator_server_and_env()
        assert env is None

    def test_headroom_is_evaluated_once_per_card(self, monkeypatch, fake_home) -> None:
        """`_cuda_gpu_has_headroom` retries with sleeps; a check-then-recheck doubles that.

        It could also disagree with itself on a card whose VRAM is in flux, admitting a card on
        one reading and sizing the launch from another.
        """
        calls: list[int] = []

        def spy(idx, _mb):
            calls.append(idx)
            return idx == 0

        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
        monkeypatch.setattr(wm, "_cuda_gpu_has_headroom", spy)
        wm._generator_server_and_env()
        # split attempt probes each card once; the degrade scan then finds gpu0 on its first try
        assert calls.count(0) <= 2, f"gpu0 probed {calls.count(0)}x: {calls}"
