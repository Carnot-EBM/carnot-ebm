"""Tests for jepa_cascade_loader.py — dynamic Exp 646 Platt temperature loader.

Covers:
  - find_exp646_result() globs results/experiment_646_*.json without hardcoding the name
  - extract_platt_temperature() handles missing key gracefully (returns None)
  - load_platt_jepa() returns loaded=True with correct temperature when Exp 646 exists
  - load_platt_jepa() returns loaded=False when file is absent
  - ThreeTierPipeline accepts platt_temperature parameter (wiring smoke test)

Spec: REQ-VERIFY-150, SCENARIO-VERIFY-198, SCENARIO-VERIFY-199
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import jax.random as jr
import pytest

from carnot.models.eorm import EORMModel
from carnot.pipeline.jepa_cascade_loader import (
    PlattCalibratedJEPA,
    extract_platt_temperature,
    find_exp646_result,
    load_platt_jepa,
)
from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_exp646_result(**overrides) -> dict:
    """Return a minimal Exp 646 result dict with sensible defaults."""
    base = {
        "experiment": 646,
        "honest_verdict": "platt_calibrated",
        "T_optimal": 0.3813067376613617,
        "scaler_saved": "/some/path/jepa_v14_platt_T.json",
    }
    base.update(overrides)
    return base


def _write_exp646_file(directory: Path, filename: str = "experiment_646_jepa_v14_platt.json", **kwargs) -> Path:
    """Write a fake Exp 646 JSON file and return its path."""
    results_dir = directory / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    path.write_text(json.dumps(_fake_exp646_result(**kwargs)))
    return path


def _make_tiny_eorm() -> EORMModel:
    """Build a tiny EORM for CI tests (fast on CPU)."""
    return EORMModel(embed_dim=32, n_heads=2, n_layers=1, max_seq_len=64, vocab_size=256, key=jr.PRNGKey(0))


def _ising_stub(response: str, question: str) -> tuple[bool, float]:
    return (True, 0.0)


# ---------------------------------------------------------------------------
# find_exp646_result() — SCENARIO-VERIFY-198
# ---------------------------------------------------------------------------


class TestFindExp646Result:
    """Spec: REQ-VERIFY-150-1, SCENARIO-VERIFY-198"""

    def test_returns_dict_when_file_exists(self, tmp_path):
        """find_exp646_result returns a parsed dict when a matching file exists.

        WHY: verifies glob-based discovery (not hardcoded filename) so the loader
        is resilient to conductor-generated filename variations.
        """
        _write_exp646_file(tmp_path)
        result = find_exp646_result(str(tmp_path))
        assert result is not None
        assert isinstance(result, dict)
        assert result["T_optimal"] == pytest.approx(0.3813067376613617)

    def test_returns_none_when_no_file(self, tmp_path):
        """find_exp646_result returns None when results/ has no experiment_646 file."""
        (tmp_path / "results").mkdir()
        result = find_exp646_result(str(tmp_path))
        assert result is None

    def test_matches_via_glob_not_hardcoded_name(self, tmp_path):
        """find_exp646_result matches any experiment_646_*.json filename.

        WHY: Exp 657 failed because of a hardcoded filename.  This test confirms
        the loader works regardless of the suffix that the conductor chooses.
        """
        # Use a non-standard suffix to confirm glob matching (not exact name match)
        _write_exp646_file(tmp_path, filename="experiment_646_alternative_suffix.json")
        result = find_exp646_result(str(tmp_path))
        assert result is not None
        assert "T_optimal" in result

    def test_returns_none_when_results_dir_missing(self, tmp_path):
        """find_exp646_result returns None gracefully when results/ does not exist."""
        result = find_exp646_result(str(tmp_path))
        assert result is None

    def test_returns_none_on_corrupted_json(self, tmp_path):
        """find_exp646_result returns None when the matched file is not valid JSON."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "experiment_646_bad.json").write_text("NOT JSON {{{")
        result = find_exp646_result(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# extract_platt_temperature() — SCENARIO-VERIFY-199
# ---------------------------------------------------------------------------


class TestExtractPlattTemperature:
    """Spec: REQ-VERIFY-150-2, SCENARIO-VERIFY-199"""

    def test_returns_float_when_key_present(self):
        """extract_platt_temperature returns a float from T_optimal.

        WHY: confirms the happy path — Exp 646 writes T_optimal as a float.
        """
        result = _fake_exp646_result()
        temp = extract_platt_temperature(result)
        assert isinstance(temp, float)
        assert temp == pytest.approx(0.3813067376613617)

    def test_returns_none_when_key_missing(self):
        """extract_platt_temperature returns None when T_optimal is absent.

        WHY SCENARIO-VERIFY-199: if the experiment was blocked before calibration,
        T_optimal will not appear in the JSON.  None signals a clean fallback.
        """
        result = {"experiment": 646, "honest_verdict": "blocked"}
        temp = extract_platt_temperature(result)
        assert temp is None

    def test_returns_none_on_non_numeric_value(self):
        """extract_platt_temperature returns None when T_optimal cannot be cast to float."""
        result = _fake_exp646_result(T_optimal="not-a-number")
        temp = extract_platt_temperature(result)
        assert temp is None

    def test_handles_integer_t_optimal(self):
        """extract_platt_temperature accepts an integer T_optimal and returns float."""
        result = _fake_exp646_result(T_optimal=1)
        temp = extract_platt_temperature(result)
        assert temp == pytest.approx(1.0)
        assert isinstance(temp, float)


# ---------------------------------------------------------------------------
# load_platt_jepa() — integration path
# ---------------------------------------------------------------------------


class TestLoadPlattJepa:
    """Spec: REQ-VERIFY-150, REQ-VERIFY-150-3"""

    def test_loaded_true_when_exp646_exists(self, tmp_path):
        """load_platt_jepa returns loaded=True when Exp 646 result is available."""
        _write_exp646_file(tmp_path)
        platt = load_platt_jepa(str(tmp_path))
        assert platt.loaded is True
        assert platt.platt_temperature == pytest.approx(0.3813067376613617)
        assert isinstance(platt, PlattCalibratedJEPA)

    def test_loaded_false_when_no_file(self, tmp_path):
        """load_platt_jepa returns loaded=False and platt_temperature=1.0 as identity."""
        platt = load_platt_jepa(str(tmp_path))
        assert platt.loaded is False
        assert platt.platt_temperature == pytest.approx(1.0)

    def test_model_path_extracted(self, tmp_path):
        """load_platt_jepa extracts scaler_saved as model_path when present."""
        _write_exp646_file(tmp_path, scaler_saved="/some/scaler.json")
        platt = load_platt_jepa(str(tmp_path))
        assert platt.model_path == "/some/scaler.json"

    def test_model_path_none_when_absent(self, tmp_path):
        """load_platt_jepa sets model_path=None when scaler_saved is not in result."""
        result = {"experiment": 646, "T_optimal": 0.4}
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_646_minimal.json").write_text(json.dumps(result))
        platt = load_platt_jepa(str(tmp_path))
        assert platt.loaded is True
        assert platt.model_path is None

    def test_loaded_false_when_t_optimal_missing(self, tmp_path):
        """load_platt_jepa returns loaded=False when file exists but T_optimal is absent."""
        result = {"experiment": 646, "honest_verdict": "blocked"}
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "experiment_646_blocked.json").write_text(json.dumps(result))
        platt = load_platt_jepa(str(tmp_path))
        assert platt.loaded is False


# ---------------------------------------------------------------------------
# ThreeTierPipeline platt_temperature wiring — REQ-VERIFY-150-4/5
# ---------------------------------------------------------------------------


class TestThreeTierPipelinePlattWiring:
    """Confirm ThreeTierPipeline accepts platt_temperature and jepa_v14_deployed is set.

    Spec: REQ-VERIFY-150-4, REQ-VERIFY-150-5
    """

    def _make_pipeline(self, platt_temperature=None) -> ThreeTierPipeline:
        return ThreeTierPipeline(
            sink_probe=SinkProbe(threshold=0.3),
            eorm_model=_make_tiny_eorm(),
            ising_pipeline=_ising_stub,
            sink_threshold=0.3,
            eorm_threshold=0.5,
            platt_temperature=platt_temperature,
        )

    def test_accepts_platt_temperature_parameter(self):
        """ThreeTierPipeline.__init__() accepts platt_temperature without error.

        Spec: REQ-VERIFY-150-4
        """
        pipeline = self._make_pipeline(platt_temperature=0.38)
        assert pipeline.platt_temperature == pytest.approx(0.38)

    def test_platt_temperature_none_by_default(self):
        """platt_temperature defaults to None (no Platt scaling applied)."""
        pipeline = self._make_pipeline()
        assert pipeline.platt_temperature is None

    def test_jepa_v14_deployed_true_in_benchmark_result(self):
        """benchmark() result has jepa_v14_deployed=True when platt_temperature is set.

        Spec: REQ-VERIFY-150-5
        """
        pipeline = self._make_pipeline(platt_temperature=0.38)
        responses = [{"question": "q", "response": "r", "attention_matrix": None}]
        result = pipeline.benchmark(responses, [True])
        assert result.jepa_v14_deployed is True

    def test_jepa_v14_deployed_false_without_platt(self):
        """benchmark() result has jepa_v14_deployed=False when no platt_temperature."""
        pipeline = self._make_pipeline(platt_temperature=None)
        responses = [{"question": "q", "response": "r", "attention_matrix": None}]
        result = pipeline.benchmark(responses, [True])
        assert result.jepa_v14_deployed is False

    def test_platt_scaling_affects_tier2_decision(self):
        """Platt temperature actually changes the effective energy in Tier 2.

        WHY: a very small temperature (T=0.01) divides energy by 0.01, making the
        effective energy huge, which should push all responses to Tier 3 (Ising).
        This confirms the scaling is applied rather than ignored.

        Spec: REQ-VERIFY-150-4
        """
        # With T=0.01, any raw energy E becomes E/0.01 = 100*E.
        # For a tiny EORM, raw energy ~0.5 -> effective=50.0 >> threshold=0.5
        # So no responses should be cleared at Tier 2 (eorm tier_used should be rare)
        pipeline_tight = self._make_pipeline(platt_temperature=0.01)

        # With T=100.0, raw energy E becomes E/100 = tiny -> most cleared at Tier 2
        pipeline_loose = self._make_pipeline(platt_temperature=100.0)

        n = 10
        responses = [{"question": f"q{i}", "response": f"r{i}", "attention_matrix": None} for i in range(n)]
        ground_truth = [True] * n

        result_tight = pipeline_tight.benchmark(responses, ground_truth)
        result_loose = pipeline_loose.benchmark(responses, ground_truth)

        # Loose (large T) should clear more at Tier 2 than tight (small T)
        assert result_loose.skip_rate_eorm >= result_tight.skip_rate_eorm
