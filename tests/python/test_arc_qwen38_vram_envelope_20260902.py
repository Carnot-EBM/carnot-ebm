"""REQ-ARC-WMTE-6880: the VRAM envelope matches the generator the guard actually launches.

Origin: 2026-09-02. `_predicted_generator_vram_mib` still used the gemma-4-31B envelope for the
Qwen3.8-27B pin. It over-predicted `-c 98304` by ~4.4 GB, which forced an 11-layer FFN offload
(decode 28 -> 13.1 tok/s) and pushed think-mode induces past the 2400s timeout -- the direct
blocker for the induce-truncation fix and the tool A/B.

The nine points below are per-PID residencies measured 2026-09-02 on an RTX 3090 (q8_0 KV,
Qwen3.8-27B Q4_K_M, mtp off; `nvidia-smi --query-compute-apps` joined by PID). Launch-to-launch
scatter is ~±150 MiB -- a 0-layer launch measured BELOW a 1-layer one -- so the assertions bound
residuals rather than demand exactness, and the per-layer credit is a bounded estimate.

Spec refs: REQ-ARC-WMTE-6880, SCENARIO-ARC-WMTE-6880-A, SCENARIO-ARC-WMTE-6880-B.
"""

from __future__ import annotations

import importlib

import pytest

MOD = "carnot.agentic.arc_executable_world_model"

# (n_ctx, ffn_cpu_layers, slots, measured MiB)
MEASURED_POINTS = (
    (49152, 1, 1, 18183),
    (73728, 1, 1, 18858),
    (98304, 1, 1, 19794),
    (49152, 0, 1, 18030),
    (49152, 1, 4, 18426),
    (73728, 1, 4, 19416),
    (98304, 11, 4, 18700),
    (98304, 0, 4, 20352),
    (106496, 0, 4, 20664),
)
RESIDUAL_BOUND_MIB = 600  # SCENARIO-6880-A: well inside the 1500 MiB guard margin


@pytest.fixture()
def wm(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
    monkeypatch.delenv("CARNOT_ARC_LLAMA_SERVER_SLOTS", raising=False)
    return importlib.import_module(MOD)


def test_envelope_tracks_every_measured_point(wm, monkeypatch) -> None:
    """SCENARIO-6880-A. Both directions bounded: an under-prediction admits a card the server
    then cudaMalloc-fails on; a large over-prediction is the 2.1x-decode-tax bug this refit
    removes. The gemma envelope fails this test at seven of nine points."""
    for n_ctx, layers, slots, measured in MEASURED_POINTS:
        monkeypatch.setenv("CARNOT_ARC_LLAMA_SERVER_SLOTS", str(slots))
        predicted = wm._predicted_generator_vram_mib(n_ctx, layers, mtp=False)
        delta = predicted - measured
        assert abs(delta) <= RESIDUAL_BOUND_MIB, (
            f"envelope off by {delta:+.0f} MiB at n_ctx={n_ctx} layers={layers} slots={slots} "
            f"(predicted {predicted:.0f}, measured {measured})"
        )


def test_envelope_never_underpredicts_a_guard_shape_point(wm, monkeypatch) -> None:
    """The guard's launch shape is 4 slots. Under-prediction there is the unsafe direction
    (admit, then cudaMalloc-fail); the fit is anchored so every 4-slot point is covered."""
    for n_ctx, layers, slots, measured in MEASURED_POINTS:
        if slots != 4:
            continue
        monkeypatch.setenv("CARNOT_ARC_LLAMA_SERVER_SLOTS", "4")
        predicted = wm._predicted_generator_vram_mib(n_ctx, layers, mtp=False)
        assert predicted >= measured, (
            f"guard-shape under-prediction at n_ctx={n_ctx} layers={layers}: "
            f"predicted {predicted:.0f} < measured {measured}"
        )


def test_default_shape_admits_a_free_3090_without_offload(wm) -> None:
    """SCENARIO-6880-B. The point of the refit: a free 24 GB card hosts the default launch with
    ZERO offload layers (the gemma envelope demanded 11 and halved decode), and the directly
    measured footprint of that exact launch sits at least 1000 MiB under the guard."""
    free_3090_mib = 24120  # measured free on an idle card, driver overhead already out
    guard = wm._generator_cuda_min_free_mb(ffn_cpu_layers=0)
    assert guard <= free_3090_mib, (
        f"guard {guard} MiB declines a FREE 3090 at the default n_ctx -- the offload tax is back"
    )
    measured_default_shape = 20664  # -c 106496, 4 slots, 0 layers, direct 2026-09-02 measurement
    assert guard - measured_default_shape >= 1000, (
        f"only {guard - measured_default_shape} MiB between guard ({guard}) and the measured "
        f"footprint ({measured_default_shape}) -- no real safety margin"
    )
