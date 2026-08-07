"""Tests for the CARNOT_ARC_GENERATOR_CUDA_GPU free-VRAM retry fix.

Spec coverage: REQ-ARC-WMTE-5769, SCENARIO-ARC-WMTE-5769-TRANSIENT-LOW-VRAM-SURVIVES-RETRY

Also covers the 2026-08-07 opt-in `CARNOT_ARC_GENERATOR_REQUIRE_CUDA` hard-stop
(`TestGeneratorRequireCuda`) -- see `GeneratorCudaRequiredError`'s docstring in
`arc_executable_world_model.py` for the exp6199 incident that motivated it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import carnot.agentic.arc_executable_world_model as mod


class TestCudaGpuHasHeadroom:
    """REQ-ARC-WMTE-5769: the retry survives a transient low-VRAM reading but still yields
    to a genuinely busy card."""

    def test_returns_true_immediately_when_headroom_available(self, monkeypatch):
        calls = []

        def _fake_free(idx):
            calls.append(idx)
            return 20000

        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", _fake_free)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        assert mod._cuda_gpu_has_headroom(1, 13000) is True
        assert calls == [1]  # no retries needed -- first reading already passed

    def test_survives_a_transient_low_reading(self, monkeypatch):
        """The exact bug this fixes: a just-crashed process's VRAM hasn't been reclaimed
        yet on the first check, but clears up by a later retry."""
        readings = iter([2000, 3000, 21000])  # transiently low, then reclaimed
        slept = []

        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: next(readings))
        monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))

        assert mod._cuda_gpu_has_headroom(1, 13000) is True
        assert slept == [
            mod._GENERATOR_CUDA_FREE_RETRY_DELAY_S,
            mod._GENERATOR_CUDA_FREE_RETRY_DELAY_S,
        ]

    def test_yields_when_card_is_genuinely_busy_the_whole_window(self, monkeypatch):
        """Not a relaxation of the guard: a card that stays busy for the ENTIRE retry
        budget still falls back, exactly as before this fix."""
        calls = []

        def _fake_free(idx):
            calls.append(idx)
            return 500  # never enough, every attempt

        slept = []
        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", _fake_free)
        monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))

        assert mod._cuda_gpu_has_headroom(1, 13000) is False
        assert len(calls) == mod._GENERATOR_CUDA_FREE_RETRY_ATTEMPTS
        # sleeps between attempts, not after the last one
        assert len(slept) == mod._GENERATOR_CUDA_FREE_RETRY_ATTEMPTS - 1


class TestGeneratorServerAndEnvUsesRetry:
    """REQ-ARC-WMTE-5769: _generator_server_and_env routes through the retrying check,
    not a bare single _cuda_gpu_free_mb call."""

    def _fake_cuda_and_hip_exist(self, monkeypatch, cuda_exists=True, hip_exists=True):
        real_exists = Path.exists

        def _fake_exists(self):
            s = str(self)
            if s.endswith("build/bin/llama-server"):
                return cuda_exists
            if s.endswith("build-hip/bin/llama-server"):
                return hip_exists
            return real_exists(self)

        monkeypatch.setattr(mod.Path, "exists", _fake_exists)

    def test_picks_cuda_after_a_transient_low_reading_recovers(self, monkeypatch):
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        self._fake_cuda_and_hip_exist(monkeypatch)

        # The recovered reading must clear the CURRENT requirement, and that requirement moved on
        # 2026-07-28: the generator is now an 18.3 GB gemma-4-31B rather than a 5.9 GB Qwen3.5-9B,
        # so `_generator_cuda_min_free_mb()` returns ~24 GB, not the ~13 GB this test was written
        # against. 21000 would now be a genuine refusal, and the test would be asserting that a
        # card we cannot fit on gets bound. The point of the test is the RETRY -- that a transient
        # low reading does not permanently lose the card -- so the low reading stays low and the
        # recovered one is raised to something that actually fits.
        #
        # `_cuda_gpu_total_mb` is stubbed because the real card (24576 MiB) is smaller than the
        # no-offload requirement (25388 MiB), which correctly short-circuits the retry entirely --
        # a different code path from the one under test here.
        readings = iter([1000, 24400])
        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: next(readings))
        monkeypatch.setattr(mod, "_cuda_gpu_total_mb", lambda idx: 49152)
        # 12 CPU-FFN layers is the documented escape hatch that makes this generator fit a 24 GB
        # card at all (25388 -> 23044 MiB required). Pinned explicitly rather than left to the
        # auto-fit, because the auto-fit reads live free VRAM and would make this test's outcome
        # depend on which card happens to be idle when the suite runs.
        monkeypatch.setattr(mod, "_default_ffn_cpu_layers", lambda: 12)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        server, env = mod._generator_server_and_env()

        assert str(server).endswith("build/bin/llama-server")
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_falls_back_to_hip_when_card_stays_busy(self, monkeypatch):
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        self._fake_cuda_and_hip_exist(monkeypatch)

        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: 500)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        server, env = mod._generator_server_and_env()

        assert str(server).endswith("build-hip/bin/llama-server")
        assert env is None


class TestGeneratorRequireCuda:
    """2026-08-07: opt-in `CARNOT_ARC_GENERATOR_REQUIRE_CUDA=1` refuses the iGPU HIP fallback
    instead of silently returning it, so a CUDA-substrate measurement (exp6199's think-mode A/B)
    fails loudly rather than corrupting its own result. Default (unset) behavior must be provably
    unchanged -- the first test below is byte-identical in setup to
    `test_falls_back_to_hip_when_card_stays_busy` above, just without the env var set."""

    def _fake_cuda_and_hip_exist(self, monkeypatch, cuda_exists=True, hip_exists=True):
        real_exists = Path.exists

        def _fake_exists(self):
            s = str(self)
            if s.endswith("build/bin/llama-server"):
                return cuda_exists
            if s.endswith("build-hip/bin/llama-server"):
                return hip_exists
            return real_exists(self)

        monkeypatch.setattr(mod.Path, "exists", _fake_exists)

    def test_default_unset_still_falls_back_to_hip(self, monkeypatch):
        """Regression guard: adding the opt-in must not change the pre-existing default."""
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.delenv("CARNOT_ARC_GENERATOR_REQUIRE_CUDA", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        self._fake_cuda_and_hip_exist(monkeypatch)
        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: 500)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        server, env = mod._generator_server_and_env()

        assert str(server).endswith("build-hip/bin/llama-server")
        assert env is None

    def test_require_cuda_raises_instead_of_falling_back(self, monkeypatch):
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_REQUIRE_CUDA", "1")
        self._fake_cuda_and_hip_exist(monkeypatch)
        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: 500)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        with pytest.raises(mod.GeneratorCudaRequiredError, match="CARNOT_ARC_GENERATOR_CUDA_GPU"):
            mod._generator_server_and_env()

    def test_require_cuda_raises_when_cuda_binary_missing(self, monkeypatch):
        """gpu requested but the CUDA build itself is absent -- the split/single-card block never
        even runs, so the raise must live OUTSIDE that block to still catch this case."""
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_REQUIRE_CUDA", "1")
        self._fake_cuda_and_hip_exist(monkeypatch, cuda_exists=False)

        with pytest.raises(mod.GeneratorCudaRequiredError):
            mod._generator_server_and_env()

    def test_require_cuda_does_not_fire_when_no_cuda_gpu_requested(self, monkeypatch):
        """The flag only guards an EXPLICIT CUDA request. With no pin requested at all, priority-3
        default (iGPU HIP, no logging, no exception) is completely unaffected."""
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_REQUIRE_CUDA", "1")
        self._fake_cuda_and_hip_exist(monkeypatch)

        server, env = mod._generator_server_and_env()

        assert str(server).endswith("build-hip/bin/llama-server")
        assert env is None

    def test_require_cuda_does_not_fire_on_success(self, monkeypatch):
        """The flag must never interfere with a launch that actually succeeds on CUDA."""
        monkeypatch.delenv("CARNOT_LLAMA_SERVER", raising=False)
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
        monkeypatch.setenv("CARNOT_ARC_GENERATOR_REQUIRE_CUDA", "1")
        self._fake_cuda_and_hip_exist(monkeypatch)

        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: 24400)
        monkeypatch.setattr(mod, "_cuda_gpu_total_mb", lambda idx: 49152)
        monkeypatch.setattr(mod, "_default_ffn_cpu_layers", lambda: 12)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        server, env = mod._generator_server_and_env()

        assert str(server).endswith("build/bin/llama-server")
        assert env is not None
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
