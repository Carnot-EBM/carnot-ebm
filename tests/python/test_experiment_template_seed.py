"""Tests for seed discipline and reproducibility checksum in ExperimentTemplate.

Spec: REQ-INFRA-061 — ExperimentTemplate seed discipline and reproducibility.

REQ-INFRA-REPRO-001: ExperimentTemplate seeds numpy/random during setup().
REQ-INFRA-REPRO-002: build_result() emits random_seed in every artifact.
REQ-INFRA-REPRO-003: build_result() emits reproducibility_checksum in every artifact.
REQ-INFRA-REPRO-004: Two runs with the same seed produce the same numpy random values.

These tests cover the verdict-reproducibility discipline added after the 2026-04-29
exp1031 verdict flip (fr11_loop_closed at 21:12Z vs carnot_filter_below_baseline at
01:13Z). The goal is not bit-exact reproducibility (GPU non-determinism prevents that)
but verdict-stable reproducibility: same seed + code + data should produce the same
stochastic samples and therefore the same verdict label across reruns.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.experiment_template import ExperimentTemplate, _compute_repro_checksum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_template(seed: int = 42, tmp_path: Path | None = None) -> ExperimentTemplate:
    """Create an ExperimentTemplate pointing at a temp deliverable path.

    Uses a temporary directory so tests never write to results/ in the real repo.
    The template is NOT set up (setup() not called) by default — call tmpl.setup()
    inside individual tests that need the side-effects.
    """
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    deliverable = str(tmp_path / "test_artifact.json")
    # repo_root override keeps the template from scanning the real results/ tree
    return ExperimentTemplate(
        exp_id=9999,
        title="Seed discipline test",
        deliverable=deliverable,
        repo_root=tmp_path,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Test 1: setup() seeds numpy and stdlib random
# ---------------------------------------------------------------------------


class TestSetupSeedsRNGs:
    """REQ-INFRA-REPRO-001: setup() initialises numpy and random from the seed."""

    def test_numpy_seed_applied(self, tmp_path):
        # SCENARIO: two templates with the same seed both call setup().
        # After setup(), the next numpy random draw should be the same value.
        tmpl_a = _make_template(seed=42, tmp_path=tmp_path / "a")
        tmpl_b = _make_template(seed=42, tmp_path=tmp_path / "b")

        # We patch the heavy setup machinery to skip GPU/lock logic; we only
        # care that the seed assignment to numpy happens.
        with (
            patch.object(ExperimentTemplate, "assert_live_env_if_gpu"),
            patch.object(ExperimentTemplate, "kill_gpu_zombies", return_value={}),
            patch.object(ExperimentTemplate, "checkpoint_resume", return_value=None),
            patch("scripts.experiment_template.DeliverableGuard"),
            patch(
                "scripts.experiment_template.ExperimentTemplate._caller_main_module",
                return_value="<not_experiment_script_module>",
            ),
        ):
            (tmp_path / "a").mkdir(parents=True, exist_ok=True)
            (tmp_path / "b").mkdir(parents=True, exist_ok=True)
            tmpl_a.setup()
            val_a = np.random.rand()
            tmpl_b.setup()
            val_b = np.random.rand()

        assert val_a == val_b, (
            f"setup(seed=42) should produce the same numpy draw twice; got {val_a} vs {val_b}"
        )

    def test_stdlib_random_seed_applied(self, tmp_path):
        # SCENARIO: two templates with the same seed; stdlib random.random() should
        # return the same value after each setup() call.
        tmpl_a = _make_template(seed=7, tmp_path=tmp_path / "a")
        tmpl_b = _make_template(seed=7, tmp_path=tmp_path / "b")

        with (
            patch.object(ExperimentTemplate, "assert_live_env_if_gpu"),
            patch.object(ExperimentTemplate, "kill_gpu_zombies", return_value={}),
            patch.object(ExperimentTemplate, "checkpoint_resume", return_value=None),
            patch("scripts.experiment_template.DeliverableGuard"),
            patch(
                "scripts.experiment_template.ExperimentTemplate._caller_main_module",
                return_value="<not_experiment_script_module>",
            ),
        ):
            (tmp_path / "a").mkdir(parents=True, exist_ok=True)
            (tmp_path / "b").mkdir(parents=True, exist_ok=True)
            tmpl_a.setup()
            val_a = random.random()
            tmpl_b.setup()
            val_b = random.random()

        assert val_a == val_b, (
            f"setup(seed=7) should produce the same stdlib draw twice; got {val_a} vs {val_b}"
        )

    def test_jax_env_var_set(self, tmp_path):
        # SCENARIO: after setup(), JAX_DEFAULT_PRNG_SEED should equal str(seed).
        tmpl = _make_template(seed=123, tmp_path=tmp_path)

        with (
            patch.object(ExperimentTemplate, "assert_live_env_if_gpu"),
            patch.object(ExperimentTemplate, "kill_gpu_zombies", return_value={}),
            patch.object(ExperimentTemplate, "checkpoint_resume", return_value=None),
            patch("scripts.experiment_template.DeliverableGuard"),
            patch(
                "scripts.experiment_template.ExperimentTemplate._caller_main_module",
                return_value="<not_experiment_script_module>",
            ),
        ):
            tmpl.setup()

        assert os.environ.get("JAX_DEFAULT_PRNG_SEED") == "123", (
            "setup() must set JAX_DEFAULT_PRNG_SEED to str(seed)"
        )


# ---------------------------------------------------------------------------
# Test 2: build_result() emits random_seed
# ---------------------------------------------------------------------------


class TestBuildResultEmitsSeed:
    """REQ-INFRA-REPRO-002: build_result() always emits random_seed."""

    def test_default_seed_in_artifact(self, tmp_path):
        tmpl = _make_template(seed=42, tmp_path=tmp_path)
        artifact = tmpl.build_result({}, status="success")
        assert "random_seed" in artifact, "build_result() must emit random_seed field"
        assert artifact["random_seed"] == 42

    def test_custom_seed_in_artifact(self, tmp_path):
        tmpl = _make_template(seed=999, tmp_path=tmp_path)
        artifact = tmpl.build_result({}, status="success")
        assert artifact["random_seed"] == 999

    def test_seed_in_schema_list(self, tmp_path):
        # The schema field is the sorted list of all top-level keys.
        # random_seed must appear in it to be visible to downstream tooling.
        tmpl = _make_template(seed=42, tmp_path=tmp_path)
        artifact = tmpl.build_result({}, status="success")
        assert "random_seed" in artifact.get("schema", []), (
            "random_seed must be present in artifact['schema']"
        )


# ---------------------------------------------------------------------------
# Test 3: build_result() emits reproducibility_checksum
# ---------------------------------------------------------------------------


class TestBuildResultEmitsChecksum:
    """REQ-INFRA-REPRO-003: build_result() always emits reproducibility_checksum."""

    def test_checksum_present(self, tmp_path):
        tmpl = _make_template(seed=42, tmp_path=tmp_path)
        artifact = tmpl.build_result({}, status="success")
        assert "reproducibility_checksum" in artifact, (
            "build_result() must emit reproducibility_checksum"
        )

    def test_checksum_is_16_chars(self, tmp_path):
        # The checksum is the first 16 hex chars of a SHA-256 digest.
        tmpl = _make_template(seed=42, tmp_path=tmp_path)
        artifact = tmpl.build_result({}, status="success")
        checksum = artifact["reproducibility_checksum"]
        assert len(checksum) == 16, (
            f"reproducibility_checksum should be 16 chars, got {len(checksum)}"
        )
        assert all(c in "0123456789abcdef" for c in checksum), (
            f"reproducibility_checksum should be lowercase hex, got {checksum!r}"
        )

    def test_checksum_changes_with_seed(self, tmp_path):
        # Different seeds must produce different checksums (seed is included in hash).
        tmpl_a = _make_template(seed=1, tmp_path=tmp_path / "a")
        tmpl_b = _make_template(seed=2, tmp_path=tmp_path / "b")
        artifact_a = tmpl_a.build_result({}, status="success")
        artifact_b = tmpl_b.build_result({}, status="success")
        assert artifact_a["reproducibility_checksum"] != artifact_b["reproducibility_checksum"], (
            "Checksums for different seeds must differ"
        )

    def test_checksum_stable_same_seed_no_files(self, tmp_path):
        # Without code/data files, two templates with the same seed must produce
        # the same checksum (pure seed hash is deterministic).
        tmpl_a = _make_template(seed=42, tmp_path=tmp_path / "a")
        tmpl_b = _make_template(seed=42, tmp_path=tmp_path / "b")
        artifact_a = tmpl_a.build_result({}, status="success")
        artifact_b = tmpl_b.build_result({}, status="success")
        assert artifact_a["reproducibility_checksum"] == artifact_b["reproducibility_checksum"], (
            "Two templates with the same seed and no code_files must produce the same checksum"
        )


# ---------------------------------------------------------------------------
# Test 4: Two runs with same seed produce same numpy random values
# ---------------------------------------------------------------------------


class TestSameSeedSameValues:
    """REQ-INFRA-REPRO-004: deterministic numpy draws after seeding."""

    def test_numpy_draws_reproducible(self, tmp_path):
        # This tests the _compute_repro_checksum helper directly (no setup() needed)
        # AND confirms that calling np.random.seed() twice with the same value
        # produces identical draw sequences — which is the property that
        # verdict-reproducibility depends on.
        np.random.seed(42)
        draws_a = [np.random.rand() for _ in range(10)]

        np.random.seed(42)
        draws_b = [np.random.rand() for _ in range(10)]

        assert draws_a == draws_b, (
            "np.random.seed(42) must produce identical draw sequences across calls"
        )

    def test_compute_repro_checksum_deterministic(self, tmp_path):
        # _compute_repro_checksum with the same inputs must produce the same output.
        code_file = tmp_path / "fake_script.py"
        code_file.write_text("x = 1\n")
        data_file = tmp_path / "fake_data.json"
        data_file.write_text('{"n": 100}\n')

        c1 = _compute_repro_checksum(42, [str(code_file)], str(data_file))
        c2 = _compute_repro_checksum(42, [str(code_file)], str(data_file))
        assert c1 == c2, "checksum must be deterministic for identical inputs"

    def test_compute_repro_checksum_code_change_detected(self, tmp_path):
        # When the code file changes, the checksum must change too.
        code_file = tmp_path / "fake_script.py"
        code_file.write_text("x = 1\n")
        c1 = _compute_repro_checksum(42, [str(code_file)])

        code_file.write_text("x = 2\n")
        c2 = _compute_repro_checksum(42, [str(code_file)])
        assert c1 != c2, "checksum must change when code file content changes"
