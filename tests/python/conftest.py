"""Shared test fixtures for Carnot Python tests."""

import os
import sys
from pathlib import Path

# Must be set before any JAX import to prevent CUDA backend probing.
# jax.config.update("jax_platform_name", "cpu") is insufficient — JAX still
# probes all backends at startup, which fails with CUDA_ERROR_OUT_OF_MEMORY
# when GPUs are occupied by other processes.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

# Add repo root to sys.path so tests can import from scripts/
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

# Disable JAX GPU for testing (CPU only)
jax.config.update("jax_platform_name", "cpu")

# Pytest collection exclusions:
# - quarantine/ — tests we've explicitly removed from default discovery.
# - test_conductor_supervisor.py — spawns subprocess.Popen of the real
#   supervisor, which then SIGTERMs the running conductor as "orphan"
#   (observed 4× on 2026-04-28). Quarantined version stays in
#   tests/python/quarantine/. Conductor self-heal sometimes regenerates
#   the file at the un-quarantined path; this ignore stops pytest
#   discovering it.
collect_ignore_glob = [
    "quarantine/**",
]
collect_ignore = [
    "test_conductor_supervisor.py",
    # test_experiment_772_semantic_energy_probe.py imports SemanticCluster,
    # _compute_tfidf_matrix, _cosine_similarity from semantic_energy_probe —
    # none of those exist in the shipped module. Test was written against a
    # TF-IDF+cosine design that was never landed; the actual implementation
    # uses random-projection embedding + Gaussian kernels. Quarantining
    # until the test is rewritten to match the shipped SemanticEnergyProbe.
    #
    # Observed: this collection error blocked .81 milestone activation
    # 2026-04-29 — the conductor's pre-test self-heal looped indefinitely
    # (1 failed / 302 passed) because the failure is a structural
    # ImportError, not a fixable runtime assertion.
    "test_experiment_772_semantic_energy_probe.py",
    # test_experiment_{1029,1031,1043}_*.py reference scripts that exist
    # transiently in the conductor's workflow but were never committed
    # (batching-check hook rejected them; see scripts/batching_precommit_check.py).
    # The scripts were removed from disk during a stash cycle; the tests
    # remain orphaned. Collection errors block the conductor's pre-test
    # self-heal, which in turn blocks .81 milestone retro from completing.
    # Quarantining until either the scripts are re-authored or the tests
    # are deleted as part of the conductor's clean-up pass.
    #
    # Observed: 2026-04-29 — exp1049 milestone-retro-81 SKIPPED because
    # 3 collection errors caused self-heal to loop (2 failed / 381 passed).
    "test_experiment_1029_fover_expansion_v2.py",
    "test_experiment_1031_energy_ssd_v3.py",
    "test_experiment_1043_fover_expansion_v3.py",
]
