"""Tests for the reusable spectral k-block sampler and Rust boundary.

Spec refs: REQ-SAMPLER-6612,
SCENARIO-SAMPLER-6612-REUSABLE-PARTITION-AND-TRANSITION,
REQ-RUSTPY-6612, SCENARIO-RUSTPY-6612-MATCHED-CHAIN-PARITY.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from carnot.samplers.backend import SamplerBackend, get_backend
from carnot.samplers.spectral_k_block import (
    SpectralKBlockBackend,
    build_random_blocks,
    build_spectral_blocks,
    ising_energy,
    run_python_chain,
    run_rust_chain,
)
from carnot.samplers import spectral_k_block as sampler_mod


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    couplings = np.array(
        [
            [0.0, 0.8, -0.35, 0.0],
            [0.8, 0.0, 0.45, -0.25],
            [-0.35, 0.45, 0.0, 0.7],
            [0.0, -0.25, 0.7, 0.0],
        ],
        dtype=np.float64,
    )
    return couplings, np.array([0.1, -0.05, 0.0, 0.08], dtype=np.float64)


def test_scenario_sampler_6612_partitions_are_complete_distinct_and_stable() -> None:
    """SCENARIO-SAMPLER-6612-REUSABLE-PARTITION-AND-TRANSITION."""

    couplings, _ = _fixture()
    spectral = build_spectral_blocks(couplings, block_size=2)
    replay = build_spectral_blocks(couplings, block_size=2)
    random = build_random_blocks(4, block_size=2, seed=6612, forbidden_hash=spectral.sha256)

    for partition in (spectral, random):
        assert sorted(spin for block in partition.blocks for spin in block) == list(range(4))
        assert partition.block_sizes == (2, 2)
        assert partition.setup_time_s > 0.0
        assert partition.sha256.startswith("sha256:")
    assert spectral.blocks == replay.blocks
    assert spectral.sha256 == replay.sha256
    assert random.sha256 != spectral.sha256
    assert "experiment_" not in inspect.getsourcefile(build_spectral_blocks)


def test_req_sampler_6612_python_chain_replays_and_charges_spins() -> None:
    """REQ-SAMPLER-6612-TRANSITION charges burn-in and all updated spins."""

    couplings, fields = _fixture()
    blocks = build_spectral_blocks(couplings, block_size=2).blocks
    initial = np.array([1, -1, 1, -1], dtype=np.int8)
    first = run_python_chain(
        couplings,
        fields,
        0.9,
        blocks,
        initial,
        seed=6612,
        burn_in=7,
        retained_samples=32,
    )
    replay = run_python_chain(
        couplings,
        fields,
        0.9,
        blocks,
        initial,
        seed=6612,
        burn_in=7,
        retained_samples=32,
    )

    assert np.array_equal(first.samples, replay.samples)
    assert first.sample_sha256 == replay.sample_sha256
    assert first.transitions == 39
    assert first.spins_updated == 78
    assert first.rng_final_state == replay.rng_final_state
    assert first.sample_time_s > 0.0
    assert np.isfinite(ising_energy(first.samples[0], couplings, fields))


def test_scenario_rustpy_6612_matched_chain_is_exact() -> None:
    """SCENARIO-RUSTPY-6612-MATCHED-CHAIN-PARITY checks exact seeded replay."""

    couplings, fields = _fixture()
    blocks = build_spectral_blocks(couplings, block_size=2).blocks
    initial = np.array([1, -1, 1, -1], dtype=np.int8)
    kwargs = {
        "seed": 6612,
        "burn_in": 7,
        "retained_samples": 64,
    }
    python = run_python_chain(couplings, fields, 0.9, blocks, initial, **kwargs)
    rust = run_rust_chain(couplings, fields, 0.9, blocks, initial, **kwargs)

    assert np.array_equal(python.samples, rust.samples)
    assert python.sample_sha256 == rust.sample_sha256
    assert python.final_state.tolist() == rust.final_state.tolist()
    assert python.rng_final_state == rust.rng_final_state
    assert python.transitions == rust.transitions
    assert python.spins_updated == rust.spins_updated


def test_req_sampler_6612_backend_uses_public_interface_and_no_silent_fallback() -> None:
    """REQ-SAMPLER-6612-PARITY exposes Rust through SamplerBackend."""

    couplings, fields = _fixture()
    blocks = build_spectral_blocks(couplings, block_size=2).blocks
    backend = SpectralKBlockBackend(seed=91, engine="python")
    samples = backend.sample(
        fields,
        couplings,
        12,
        {
            "temperature": 0.9,
            "blocks": blocks,
            "burn_in": 3,
            "initial_state": [1, -1, 1, -1],
        },
    )

    assert isinstance(backend, SamplerBackend)
    assert backend.backend_name == "spectral_k_block_python"
    assert samples.shape == (12, 4)
    assert samples.dtype == np.bool_
    assert isinstance(get_backend("spectral_k_block"), SpectralKBlockBackend)
    assert backend.set_constraints(None) is None
    assert backend.dual_update_step(0.1) is None

    minimized = backend.minimize_energy(fields, couplings, 3, 2, beta=1.0 / 0.9)
    assert minimized.shape == (3, 4)
    default_partition = backend.sample(
        fields,
        couplings,
        2,
        {"temperature": 0.9, "initial_state": [1, -1, 1, -1]},
    )
    assert default_partition.shape == (2, 4)

    blocked = SpectralKBlockBackend(
        engine="rust", rust_module_loader=lambda: (_ for _ in ()).throw(ImportError("missing"))
    )
    with pytest.raises(RuntimeError, match="Rust spectral k-block binding unavailable"):
        blocked.sample(
            fields,
            couplings,
            2,
            {
                "temperature": 0.9,
                "blocks": blocks,
                "initial_state": [1, -1, 1, -1],
            },
        )

    invalid_engine = SpectralKBlockBackend(engine="other")
    with pytest.raises(ValueError, match="engine"):
        invalid_engine.sample(
            fields,
            couplings,
            1,
            {
                "temperature": 0.9,
                "blocks": blocks,
                "initial_state": [1, -1, 1, -1],
            },
        )


@pytest.mark.parametrize(
    ("couplings", "fields", "temperature", "blocks", "initial", "message"),
    [
        (np.zeros((2, 3)), np.zeros(2), 1.0, ((0,), (1,)), [1, -1], "square"),
        (
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.zeros(2),
            1.0,
            ((0,), (1,)),
            [1, -1],
            "symmetric",
        ),
        (np.zeros((2, 2)), np.zeros(1), 1.0, ((0,), (1,)), [1, -1], "fields"),
        (np.zeros((2, 2)), np.zeros(2), 0.0, ((0,), (1,)), [1, -1], "temperature"),
        (np.zeros((2, 2)), np.zeros(2), 1.0, ((0,), (0,)), [1, -1], "partition"),
        (np.zeros((2, 2)), np.zeros(2), 1.0, ((0,), (1,)), [1, 0], "spins"),
    ],
)
def test_req_rustpy_6612_invalid_descriptors_fail_closed(
    couplings: np.ndarray,
    fields: np.ndarray,
    temperature: float,
    blocks: tuple[tuple[int, ...], ...],
    initial: list[int],
    message: str,
) -> None:
    """REQ-RUSTPY-6612 rejects malformed inputs before a transition."""

    with pytest.raises(ValueError, match=message):
        run_python_chain(
            couplings,
            fields,
            temperature,
            blocks,
            np.asarray(initial),
            seed=1,
            burn_in=0,
            retained_samples=1,
        )


def test_req_sampler_6612_all_input_guards_are_exercised() -> None:
    """REQ-SAMPLER-6612-ATTACKS covers every bounded input guard."""

    couplings, fields = _fixture()
    with pytest.raises(ValueError, match="unsigned 64"):
        run_python_chain(
            couplings,
            fields,
            0.9,
            ((0,), (1,), (2,), (3,)),
            [1, 1, 1, 1],
            seed=-1,
            burn_in=0,
            retained_samples=1,
        )
    with pytest.raises(ValueError, match="finite"):
        build_spectral_blocks(np.array([[0.0, np.nan], [np.nan, 0.0]]), 1)
    with pytest.raises(ValueError, match="block_size"):
        build_spectral_blocks(couplings, 0)
    with pytest.raises(ValueError, match="n_spins"):
        build_random_blocks(0, 1, 1)
    with pytest.raises(ValueError, match="block_size"):
        build_random_blocks(4, 0, 1)
    one_hash = sampler_mod._sha256_json(((0,),))
    with pytest.raises(ValueError, match="identical"):
        build_random_blocks(1, 1, 1, forbidden_hash=one_hash)
    with pytest.raises(ValueError, match="spins"):
        ising_energy([1, 0, 1, -1], couplings, fields)
    with pytest.raises(ValueError, match="burn_in"):
        run_python_chain(
            couplings,
            fields,
            0.9,
            ((0,), (1,), (2,), (3,)),
            [1, 1, 1, 1],
            seed=1,
            burn_in=-1,
            retained_samples=1,
        )
    with pytest.raises(ValueError, match="retained_samples"):
        run_python_chain(
            couplings,
            fields,
            0.9,
            ((0,), (1,), (2,), (3,)),
            [1, 1, 1, 1],
            seed=1,
            burn_in=0,
            retained_samples=0,
        )
    with pytest.raises(ValueError, match="partition blocks"):
        run_python_chain(
            couplings,
            fields,
            0.9,
            ((), (0, 1, 2, 3)),
            [1, 1, 1, 1],
            seed=1,
            burn_in=0,
            retained_samples=1,
        )
    with pytest.raises(ValueError, match="fields must be finite"):
        run_python_chain(
            couplings,
            [0.0, np.nan, 0.0, 0.0],
            0.9,
            ((0,), (1,), (2,), (3,)),
            [1, 1, 1, 1],
            seed=1,
            burn_in=0,
            retained_samples=1,
        )
    with pytest.raises(ValueError, match="spins"):
        run_rust_chain(
            couplings,
            fields,
            0.9,
            ((0,), (1,), (2,), (3,)),
            [1, 1, 1, 0],
            seed=1,
            burn_in=0,
            retained_samples=1,
        )
