"""Tests for the CARNOT_ARC_GENERATOR_CUDA_GPU free-VRAM retry fix.

Spec coverage: REQ-ARC-WMTE-5769, SCENARIO-ARC-WMTE-5769-TRANSIENT-LOW-VRAM-SURVIVES-RETRY
"""

from __future__ import annotations

from pathlib import Path

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

        readings = iter([1000, 21000])
        monkeypatch.setattr(mod, "_cuda_gpu_free_mb", lambda idx: next(readings))
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
