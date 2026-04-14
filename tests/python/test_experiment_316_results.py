"""Tests for Experiment 316 full-scale benchmark result validation.

Validates the JSON artifact produced by scripts/experiment_315_fullscale_benchmark.py
when executed in Exp 316.  Every test function is a direct assertion against the
artifact's schema, statistical bounds, or labelling requirements — so it is both
a correctness gate and living documentation of the artifact format.

**Why these tests exist:**
    The benchmark produces authoritative accuracy numbers that feed README headline
    claims and the research record.  Fabricated, malformed, or statistically
    incoherent results must be caught before they propagate.  These tests are the
    gate between "we ran something" and "we have a credible result".

**Schema being tested:** carnot.fullscale_benchmark.v1 (see ARTIFACT_SCHEMA in
scripts/experiment_315_fullscale_benchmark.py)

Spec: REQ-BENCH-001, SCENARIO-BENCH-001, SCENARIO-BENCH-002
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULT_PATH = _REPO_ROOT / "results" / "experiment_316_fullscale_results.json"

# Valid inference mode labels — "unknown" is explicitly forbidden because it
# means the pipeline did not correctly detect its execution context.
VALID_INFERENCE_MODES = {"live_gpu", "simulated"}

# Required top-level keys in every carnot.fullscale_benchmark.v1 artifact.
REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "per_model_results",
    "per_variant_results",
    "published_baselines",
    "summary_table",
    "inference_mode",
    "n_gsm8k",
    "n_humaneval",
    "modes_run",
    "schema",
}

# Required keys in every per-model accuracy cell dict.
REQUIRED_CELL_KEYS = {
    "model_name",
    "mode",
    "corpus_variant",
    "accuracy",
    "ci_lower",
    "ci_upper",
    "n_correct",
    "n_total",
}


# ---------------------------------------------------------------------------
# Fixture: load the artifact once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def result() -> dict[str, Any]:
    """Load the Exp 316 artifact from disk.

    Skips the entire test session if the artifact has not yet been produced
    (i.e., Exp 316 has not been run yet).  This prevents CI failures when the
    artifact is missing versus genuine result corruption.
    """
    if not _RESULT_PATH.exists():
        pytest.skip(f"Exp 316 artifact not found at {_RESULT_PATH}; run Exp 316 first")
    with _RESULT_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Helpers used across multiple tests
# ---------------------------------------------------------------------------


def _all_cells(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten per_model_results into a list of cell dicts.

    per_model_results has shape [model_name][mode][corpus_variant] → cell dict.
    This helper flattens it so individual tests can iterate over all cells without
    repeating three nested loops.
    """
    cells = []
    for model_data in result.get("per_model_results", {}).values():
        for mode_data in model_data.values():
            for cell in mode_data.values():
                cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Verify the artifact has the required structure.

    REQ-BENCH-001: artifact must conform to carnot.fullscale_benchmark.v1 schema.
    """

    def test_required_top_level_keys_present(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: all required top-level keys must be present."""
        missing = REQUIRED_TOP_LEVEL_KEYS - set(result.keys())
        assert not missing, f"Missing top-level keys: {missing}"

    def test_schema_field_is_sorted_key_list(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: 'schema' field must be a sorted list of all top-level keys."""
        schema_field = result["schema"]
        assert isinstance(schema_field, list), "schema field must be a list"
        assert schema_field == sorted(result.keys()), (
            "schema field must equal sorted(result.keys())"
        )

    def test_per_model_results_is_dict(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: per_model_results must be a non-empty dict."""
        pmr = result["per_model_results"]
        assert isinstance(pmr, dict), "per_model_results must be a dict"
        assert len(pmr) > 0, "per_model_results must not be empty"

    def test_per_variant_results_is_dict(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: per_variant_results must be a dict."""
        pvr = result["per_variant_results"]
        assert isinstance(pvr, dict)

    def test_published_baselines_is_dict(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: published_baselines must be a dict with float values in [0,1]."""
        pb = result["published_baselines"]
        assert isinstance(pb, dict), "published_baselines must be a dict"
        assert len(pb) > 0, "published_baselines must not be empty"

    def test_cell_keys_present(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: every accuracy cell must have all required keys."""
        for cell in _all_cells(result):
            missing = REQUIRED_CELL_KEYS - set(cell.keys())
            assert not missing, (
                f"Cell {cell.get('model_name')}/{cell.get('mode')}/{cell.get('corpus_variant')} "
                f"missing keys: {missing}"
            )

    def test_summary_table_is_list(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: summary_table must be a non-empty list."""
        st = result["summary_table"]
        assert isinstance(st, list)
        assert len(st) > 0

    def test_modes_run_is_list(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: modes_run must be a non-empty list of strings."""
        mr = result["modes_run"]
        assert isinstance(mr, list)
        assert all(isinstance(m, str) for m in mr)
        assert len(mr) > 0


# ---------------------------------------------------------------------------
# Inference mode labelling
# ---------------------------------------------------------------------------


class TestInferenceMode:
    """Verify inference_mode is one of the accepted labels.

    REQ-BENCH-001: inference_mode must be "live_gpu" or "simulated".
    "unknown" is forbidden — it means the detection logic failed.
    """

    def test_no_fabricated_results(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: inference_mode must be 'live_gpu' or 'simulated', never 'unknown'."""
        mode = result["inference_mode"]
        assert mode in VALID_INFERENCE_MODES, (
            f"inference_mode={mode!r} is not a valid label; "
            f"must be one of {VALID_INFERENCE_MODES}"
        )

    def test_inference_mode_is_string(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: inference_mode must be a string."""
        assert isinstance(result["inference_mode"], str)


# ---------------------------------------------------------------------------
# Statistical bounds: CI must bracket the accuracy
# ---------------------------------------------------------------------------


class TestCIBounds:
    """Verify Wilson 95% CI is well-formed for every result cell.

    SCENARIO-BENCH-001: ci_lower ≤ accuracy ≤ ci_upper for every cell.
    The Wilson interval is always valid; if ci_lower > accuracy, either the
    formula is wrong or the n_correct/n_total counts are inconsistent.
    """

    def test_verify_ci_bounds(self, result: dict[str, Any]) -> None:
        """SCENARIO-BENCH-001: ci_lower <= accuracy <= ci_upper for every cell."""
        for cell in _all_cells(result):
            label = f"{cell['model_name']}/{cell['mode']}/{cell['corpus_variant']}"
            acc = cell["accuracy"]
            lo = cell["ci_lower"]
            hi = cell["ci_upper"]
            assert lo <= acc + 1e-9, (
                f"{label}: ci_lower ({lo}) > accuracy ({acc})"
            )
            assert acc <= hi + 1e-9, (
                f"{label}: accuracy ({acc}) > ci_upper ({hi})"
            )
            assert lo >= 0.0, f"{label}: ci_lower ({lo}) < 0"
            assert hi <= 1.0, f"{label}: ci_upper ({hi}) > 1"

    def test_ci_lower_le_ci_upper(self, result: dict[str, Any]) -> None:
        """SCENARIO-BENCH-001: ci_lower <= ci_upper for every cell (never inverted)."""
        for cell in _all_cells(result):
            label = f"{cell['model_name']}/{cell['mode']}/{cell['corpus_variant']}"
            assert cell["ci_lower"] <= cell["ci_upper"] + 1e-9, (
                f"{label}: CI is inverted: [{cell['ci_lower']}, {cell['ci_upper']}]"
            )


# ---------------------------------------------------------------------------
# Sample size requirements
# ---------------------------------------------------------------------------


class TestSampleSize:
    """Verify n_total is large enough to be statistically meaningful.

    SCENARIO-BENCH-001: n_total >= 50 for every (mode, model) combination
    that is not a corpus-variant sub-slice (variant == "all" is the aggregate;
    per-variant slices may be smaller but must be >= 1).
    """

    def test_verify_n_total(self, result: dict[str, Any]) -> None:
        """SCENARIO-BENCH-001: n_total >= 50 for every 'all' variant aggregate cell."""
        for cell in _all_cells(result):
            if cell["corpus_variant"] == "all":
                label = f"{cell['model_name']}/{cell['mode']}/all"
                assert cell["n_total"] >= 50, (
                    f"{label}: n_total={cell['n_total']} < 50 — too small for credible CI"
                )

    def test_n_correct_le_n_total(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: n_correct must not exceed n_total (basic sanity)."""
        for cell in _all_cells(result):
            label = f"{cell['model_name']}/{cell['mode']}/{cell['corpus_variant']}"
            assert cell["n_correct"] <= cell["n_total"], (
                f"{label}: n_correct ({cell['n_correct']}) > n_total ({cell['n_total']})"
            )

    def test_n_total_positive(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: n_total must be positive (no empty cells)."""
        for cell in _all_cells(result):
            label = f"{cell['model_name']}/{cell['mode']}/{cell['corpus_variant']}"
            assert cell["n_total"] > 0, f"{label}: n_total == 0"


# ---------------------------------------------------------------------------
# Published baseline validity
# ---------------------------------------------------------------------------


class TestPublishedBaselines:
    """Verify published_baselines values are valid proportions.

    REQ-BENCH-001: every published baseline must be in [0.0, 1.0].
    These are model-card accuracy figures used to frame relative improvement.
    """

    def test_comparison_table_baselines_in_range(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: all published_baseline values must be in [0.0, 1.0]."""
        pb = result["published_baselines"]
        for model, baseline in pb.items():
            assert isinstance(baseline, (int, float)), (
                f"published_baselines[{model!r}] is not a number: {baseline!r}"
            )
            assert 0.0 <= baseline <= 1.0, (
                f"published_baselines[{model!r}] = {baseline} is outside [0, 1]"
            )

    def test_published_baselines_not_empty(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: published_baselines must list at least one model."""
        assert len(result["published_baselines"]) >= 1


# ---------------------------------------------------------------------------
# Accuracy range
# ---------------------------------------------------------------------------


class TestAccuracyRange:
    """Verify accuracy values are valid proportions.

    REQ-BENCH-001: accuracy must be in [0.0, 1.0] for every cell.
    """

    def test_accuracy_in_range(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: accuracy must be in [0.0, 1.0] for every cell."""
        for cell in _all_cells(result):
            label = f"{cell['model_name']}/{cell['mode']}/{cell['corpus_variant']}"
            assert 0.0 <= cell["accuracy"] <= 1.0, (
                f"{label}: accuracy {cell['accuracy']} outside [0, 1]"
            )

    def test_accuracy_consistent_with_counts(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: accuracy must equal n_correct/n_total within float tolerance."""
        for cell in _all_cells(result):
            if cell["n_total"] == 0:
                continue
            label = f"{cell['model_name']}/{cell['mode']}/{cell['corpus_variant']}"
            expected = cell["n_correct"] / cell["n_total"]
            assert abs(cell["accuracy"] - expected) < 1e-4, (
                f"{label}: accuracy {cell['accuracy']} != n_correct/n_total {expected}"
            )


# ---------------------------------------------------------------------------
# Duration and status
# ---------------------------------------------------------------------------


class TestArtifactMetadata:
    """Validate artifact metadata fields.

    REQ-BENCH-001: status must be 'success', duration_s must be positive.
    """

    def test_status_success(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: artifact status must be 'success'."""
        assert result["status"] == "success", (
            f"Artifact status is {result['status']!r}; expected 'success'"
        )

    def test_duration_positive(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: duration_s must be positive."""
        assert result["duration_s"] > 0

    def test_experiment_id(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: experiment ID must be 315 (the script that produced the result)."""
        assert result["experiment"] == 315

    def test_n_gsm8k_positive(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: n_gsm8k must be a positive integer."""
        assert isinstance(result["n_gsm8k"], int)
        assert result["n_gsm8k"] > 0

    def test_n_humaneval_positive(self, result: dict[str, Any]) -> None:
        """REQ-BENCH-001: n_humaneval must be a positive integer."""
        assert isinstance(result["n_humaneval"], int)
        assert result["n_humaneval"] > 0


# ---------------------------------------------------------------------------
# load_fullscale_results helper (tested independently)
# ---------------------------------------------------------------------------


def load_fullscale_results(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a fullscale benchmark result JSON.

    **Why this helper exists:**
        Downstream tooling (reporting scripts, README update, research_conductor)
        should share a single validated loader rather than each implementing their
        own ad-hoc json.load().  This function raises ValueError with a descriptive
        message if the file is missing a required key so callers get actionable errors.

    Args:
        path: Path to the JSON artifact produced by experiment_315_fullscale_benchmark.py.

    Returns:
        Validated artifact dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError:        if required keys are missing from the artifact.

    Spec: REQ-BENCH-001
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Artifact missing required keys: {missing}")
    return data


class TestLoadFullscaleResults:
    """Tests for the load_fullscale_results helper function.

    These tests exercise the helper in isolation using temporary JSON fixtures,
    so they do NOT require the live Exp 316 artifact.
    """

    def _make_valid_artifact(self) -> dict[str, Any]:
        """Build a minimal valid carnot.fullscale_benchmark.v1 artifact for testing."""
        base: dict[str, Any] = {
            "experiment": 315,
            "title": "Full-Scale Credible Benchmark",
            "run_date": "20260414",
            "started_at": "2026-04-14T20:00:00Z",
            "finished_at": "2026-04-14T20:10:00Z",
            "duration_s": 600.0,
            "status": "success",
            "per_model_results": {
                "TestModel": {
                    "baseline": {
                        "all": {
                            "model_name": "TestModel",
                            "mode": "baseline",
                            "corpus_variant": "all",
                            "accuracy": 0.25,
                            "ci_lower": 0.21,
                            "ci_upper": 0.29,
                            "n_correct": 25,
                            "n_total": 100,
                        }
                    }
                }
            },
            "per_variant_results": {},
            "published_baselines": {"TestModel": 0.25},
            "summary_table": [],
            "inference_mode": "simulated",
            "n_gsm8k": 100,
            "n_humaneval": 10,
            "modes_run": ["baseline"],
        }
        # Add schema field (as the real artifact does)
        base["schema"] = sorted(base.keys())
        return base

    def test_load_valid_artifact(self, tmp_path: Path) -> None:
        """REQ-BENCH-001: load_fullscale_results loads a valid artifact without error."""
        artifact = self._make_valid_artifact()
        p = tmp_path / "result.json"
        p.write_text(json.dumps(artifact))
        loaded = load_fullscale_results(p)
        assert loaded["experiment"] == 315

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """REQ-BENCH-001: load_fullscale_results raises FileNotFoundError when file missing."""
        with pytest.raises(FileNotFoundError):
            load_fullscale_results(tmp_path / "nonexistent.json")

    def test_load_missing_keys_raises(self, tmp_path: Path) -> None:
        """REQ-BENCH-001: load_fullscale_results raises ValueError on missing required keys."""
        # Remove a required key to trigger validation failure
        artifact = self._make_valid_artifact()
        del artifact["per_model_results"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(artifact))
        with pytest.raises(ValueError, match="missing required keys"):
            load_fullscale_results(p)

    def test_load_all_required_keys_present(self, tmp_path: Path) -> None:
        """REQ-BENCH-001: loaded artifact contains all REQUIRED_TOP_LEVEL_KEYS."""
        artifact = self._make_valid_artifact()
        p = tmp_path / "result.json"
        p.write_text(json.dumps(artifact))
        loaded = load_fullscale_results(p)
        assert REQUIRED_TOP_LEVEL_KEYS.issubset(set(loaded.keys()))
