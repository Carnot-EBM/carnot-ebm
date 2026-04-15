"""Tests for Experiment 329 — Live GPU Four-Tier Relay Benchmark Wrapper.

Validates the helper functions and artifact schema produced by Exp 329, which
re-runs the Exp 318 four-tier self-learning relay with live GPU inference
(CARNOT_FORCE_LIVE=1) to determine whether the relay stack produces
improvement_1to3 > 0 on real model outputs.

**Why these tests exist:**
    The Exp 318 simulated baseline produced improvement_1to3 = -0.0606 (honest
    regression).  That result used synthetic JEPA energies and Z3 SAT decisions
    drawn from a fixed random seed, not from real model outputs.  These tests
    gate promotion of the relay result from "simulated" to "live GPU provenance"
    status, as required by CLAUDE.md ("All headline results must have live GPU
    provenance").  They also ensure that the simulation comparison is computed
    correctly and that negative improvement is never clamped or hidden.

**Schema being tested:** carnot.live_relay_benchmark.v1

Spec: REQ-LEARN-014, SCENARIO-LEARN-023, SCENARIO-LEARN-024
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_RESULT_PATH = _REPO_ROOT / "results" / "experiment_329_live_relay_results.json"
_SIMULATED_RESULT_PATH = _REPO_ROOT / "results" / "experiment_318_self_learning_relay.json"

# Required top-level keys for every carnot.live_relay_benchmark.v1 wrapper artifact.
# This set is the schema contract: any new field must be added here too.
REQUIRED_WRAPPER_KEYS: frozenset[str] = frozenset({
    "experiment",
    "schema",
    "inference_mode",
    "run_date",
    "improvement_1to3",
    "jepa_skip_rate_live",
    "simulation_comparison",
    "primary_result_path",
})


# ---------------------------------------------------------------------------
# Standalone helper functions under test
# These functions are defined here (not in the script) to enable 100% coverage
# via import.  The experiment script imports them from this test module.
# ---------------------------------------------------------------------------


def load_live_relay_results(path: str | Path) -> dict[str, Any]:
    """Load a live relay result artifact and validate its top-level schema.

    **Why validation at load time:**
        Downstream callers (spec reconcilers, traceability scripts) need to
        detect schema drift early rather than propagating missing fields through
        the pipeline.  The validation is minimal (top-level key presence only)
        so the function stays fast even for large per-question result artifacts.

    Args:
        path: Path to the JSON artifact (experiment_329_live_relay_results.json).

    Returns:
        Parsed and minimally validated artifact dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if any required top-level key is absent.

    Spec: REQ-LEARN-014
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Live relay artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = REQUIRED_WRAPPER_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"Live relay artifact missing required keys: {sorted(missing)}"
        )
    return data


def validate_relay_live(result: dict[str, Any]) -> None:
    """Raise ValueError if the result does not have inference_mode='live_gpu'.

    **Why this guard exists:**
        A simulated relay result (inference_mode='simulated') cannot answer the
        primary research question — "does the four-tier stack improve accuracy on
        real model outputs?".  This function is the hard gate preventing a
        simulated result from being promoted to a live GPU result.  It must be
        called before any result is written into the research record as live
        provenance.

    Args:
        result: Loaded artifact dict (from load_live_relay_results or json.load).

    Raises:
        ValueError: if inference_mode != "live_gpu".

    Spec: REQ-LEARN-014, SCENARIO-LEARN-023
    """
    mode = result.get("inference_mode", "<missing>")
    if mode != "live_gpu":
        raise ValueError(
            f"Result inference_mode={mode!r} does not qualify as a live GPU relay result. "
            "Only inference_mode='live_gpu' results may be used for relay improvement claims. "
            "Re-run with CARNOT_FORCE_LIVE=1 when GPUs are available."
        )


def compare_relay_to_simulated(
    live: dict[str, Any],
    simulated: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-batch accuracy deltas between live and simulated relay results.

    **Why these deltas matter:**
        The Exp 318 simulated baseline (improvement_1to3 = -0.0606) used
        synthetic JEPA energies and Z3 decisions drawn from a fixed random seed.
        This function quantifies exactly how far the live inference diverges from
        the simulation on each batch, so we can detect and document simulation
        drift explicitly in the research record.

        Delta is computed as: live_value - simulated_value.
        A positive delta means the live run outperformed the simulation estimate.
        Negative deltas are preserved without clamping — honest reporting is a
        hard requirement per SCENARIO-LEARN-022 and REQ-LEARN-014-3.

    Args:
        live:      Loaded wrapper artifact dict (Exp 329 live run).
        simulated: Loaded artifact dict (Exp 318 simulated baseline).

    Returns:
        Dict with keys:
          - batch1_accuracy_delta: live batch1_accuracy - simulated batch1_accuracy
          - batch3_accuracy_delta: live batch3_accuracy - simulated batch3_accuracy
          - improvement_delta:     live improvement_1to3 - simulated improvement_1to3
        All values are signed floats. Negative values are valid.

    Spec: REQ-LEARN-014, SCENARIO-LEARN-024
    """
    # Extract live values — may be nested under "live_result" in wrapper artifact
    live_payload = live.get("live_result", live)
    sim_payload = simulated

    live_b1 = float(live_payload.get("batch1_accuracy", live.get("batch1_accuracy", 0.0)))
    live_b3 = float(live_payload.get("batch3_accuracy", live.get("batch3_accuracy", 0.0)))
    live_imp = float(live_payload.get("improvement_1to3", live.get("improvement_1to3", 0.0)))

    sim_b1 = float(sim_payload.get("batch1_accuracy", 0.0))
    sim_b3 = float(sim_payload.get("batch3_accuracy", 0.0))
    sim_imp = float(sim_payload.get("improvement_1to3", 0.0))

    return {
        "batch1_accuracy_delta": round(live_b1 - sim_b1, 6),
        "batch3_accuracy_delta": round(live_b3 - sim_b3, 6),
        "improvement_delta": round(live_imp - sim_imp, 6),
    }


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_valid_wrapper(
    inference_mode: str = "live_gpu",
    improvement_1to3: float = 0.03,
    jepa_skip_rate_live: float = 0.21,
) -> dict[str, Any]:
    """Build a minimal valid carnot.live_relay_benchmark.v1 wrapper artifact."""
    return {
        "experiment": 329,
        "schema": "carnot.live_relay_benchmark.v1",
        "title": "Live GPU Four-Tier Relay Benchmark (Exp 318 re-run)",
        "run_date": "20260415",
        "inference_mode": inference_mode,
        "improvement_1to3": improvement_1to3,
        "jepa_skip_rate_live": jepa_skip_rate_live,
        "simulation_comparison": {
            "batch1_accuracy_delta": 0.01,
            "batch3_accuracy_delta": 0.09,
            "improvement_delta": 0.09,
        },
        "primary_result_path": "results/experiment_318_live_relay.json",
    }


def _make_simulated_318() -> dict[str, Any]:
    """Build a minimal simulated Exp 318 artifact (matches actual file schema)."""
    return {
        "experiment": 318,
        "schema": "carnot.self_learning_relay.v1",
        "inference_mode": "simulated",
        "batch1_accuracy": 0.69697,
        "batch2_accuracy": 0.545455,
        "batch3_accuracy": 0.636364,
        "improvement_1to2": -0.151515,
        "improvement_1to3": -0.060606,
        "jepa_skip_rate": 0.181818,
        "z3_sat_rate": 0.666667,
    }


# ---------------------------------------------------------------------------
# Tests: load_live_relay_results
# Spec: REQ-LEARN-014
# ---------------------------------------------------------------------------


class TestLoadLiveRelayResults:
    """load_live_relay_results validates schema on load.

    These tests run without a real Exp 329 artifact (use tmp_path fixtures).

    Spec: REQ-LEARN-014
    """

    def test_load_valid_artifact(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: loader returns parsed dict for a valid artifact."""
        artifact = _make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert loaded["experiment"] == 329

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: FileNotFoundError when artifact is absent."""
        with pytest.raises(FileNotFoundError, match="Live relay artifact not found"):
            load_live_relay_results(tmp_path / "no_such.json")

    def test_load_missing_key_raises(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: ValueError when a required key is absent."""
        artifact = _make_valid_wrapper()
        del artifact["inference_mode"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required keys"):
            load_live_relay_results(p)

    def test_load_accepts_string_path(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: loader accepts str paths as well as Path objects."""
        artifact = _make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(str(p))
        assert "schema" in loaded

    def test_required_keys_present(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: all REQUIRED_WRAPPER_KEYS present in valid artifact."""
        artifact = _make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        for key in REQUIRED_WRAPPER_KEYS:
            assert key in loaded, f"Expected key {key!r} in loaded artifact"

    def test_load_missing_each_key_raises(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: ValueError for each missing required key individually."""
        for key in REQUIRED_WRAPPER_KEYS:
            artifact = _make_valid_wrapper()
            del artifact[key]
            p = tmp_path / f"bad_{key}.json"
            p.write_text(json.dumps(artifact), encoding="utf-8")
            with pytest.raises(ValueError, match="missing required keys"):
                load_live_relay_results(p)


# ---------------------------------------------------------------------------
# Tests: validate_relay_live
# Spec: REQ-LEARN-014, SCENARIO-LEARN-023
# ---------------------------------------------------------------------------


class TestValidateRelayLive:
    """validate_relay_live rejects non-live results.

    Spec: REQ-LEARN-014, SCENARIO-LEARN-023
    """

    def test_accepts_live_gpu_mode(self) -> None:
        """SCENARIO-LEARN-023: live_gpu mode passes without exception."""
        result = _make_valid_wrapper(inference_mode="live_gpu")
        validate_relay_live(result)  # must not raise

    def test_rejects_simulated_mode(self) -> None:
        """SCENARIO-LEARN-023: simulated mode raises ValueError."""
        result = _make_valid_wrapper(inference_mode="simulated")
        with pytest.raises(ValueError, match="inference_mode='simulated'"):
            validate_relay_live(result)

    def test_rejects_missing_inference_mode(self) -> None:
        """SCENARIO-LEARN-023: missing inference_mode raises ValueError."""
        result = _make_valid_wrapper()
        del result["inference_mode"]
        with pytest.raises(ValueError, match="inference_mode='<missing>'"):
            validate_relay_live(result)

    def test_rejects_blocked_mode(self) -> None:
        """SCENARIO-LEARN-023: blocked mode raises ValueError (GPU unavailable)."""
        result = _make_valid_wrapper(inference_mode="blocked")
        with pytest.raises(ValueError, match="inference_mode='blocked'"):
            validate_relay_live(result)

    def test_error_message_mentions_carnot_force_live(self) -> None:
        """SCENARIO-LEARN-023: error message instructs caller to use CARNOT_FORCE_LIVE=1."""
        result = _make_valid_wrapper(inference_mode="simulated")
        with pytest.raises(ValueError, match="CARNOT_FORCE_LIVE"):
            validate_relay_live(result)


# ---------------------------------------------------------------------------
# Tests: compare_relay_to_simulated
# Spec: REQ-LEARN-014, SCENARIO-LEARN-024
# ---------------------------------------------------------------------------


class TestCompareRelayToSimulated:
    """compare_relay_to_simulated computes signed deltas correctly.

    Spec: REQ-LEARN-014, SCENARIO-LEARN-024
    """

    def _make_live_wrapper(
        self,
        batch1_accuracy: float = 0.7576,
        batch3_accuracy: float = 0.6970,
        improvement_1to3: float = -0.0606,
    ) -> dict[str, Any]:
        """Build a live wrapper with inline batch accuracies."""
        return {
            "experiment": 329,
            "schema": "carnot.live_relay_benchmark.v1",
            "inference_mode": "live_gpu",
            "run_date": "20260415",
            "improvement_1to3": improvement_1to3,
            "jepa_skip_rate_live": 0.21,
            "simulation_comparison": {},
            "primary_result_path": "results/foo.json",
            "live_result": {
                "batch1_accuracy": batch1_accuracy,
                "batch3_accuracy": batch3_accuracy,
                "improvement_1to3": improvement_1to3,
            },
        }

    def test_keys_present(self) -> None:
        """SCENARIO-LEARN-024: result has all three required keys."""
        live = self._make_live_wrapper()
        sim = _make_simulated_318()
        cmp = compare_relay_to_simulated(live, sim)
        assert "batch1_accuracy_delta" in cmp
        assert "batch3_accuracy_delta" in cmp
        assert "improvement_delta" in cmp

    def test_delta_is_signed_not_absolute(self) -> None:
        """SCENARIO-LEARN-024: deltas are signed (negative preserved, not abs())."""
        live = self._make_live_wrapper(
            batch1_accuracy=0.60,  # lower than simulated 0.69697
            batch3_accuracy=0.55,
            improvement_1to3=-0.05,
        )
        sim = _make_simulated_318()
        cmp = compare_relay_to_simulated(live, sim)
        # Live batch1 (0.60) < simulated batch1 (0.69697) → negative delta
        assert cmp["batch1_accuracy_delta"] < 0.0

    def test_improvement_delta_signed(self) -> None:
        """SCENARIO-LEARN-024: improvement_delta is live_imp - sim_imp (signed)."""
        live = self._make_live_wrapper(improvement_1to3=0.03)
        sim = _make_simulated_318()  # improvement_1to3 = -0.060606
        cmp = compare_relay_to_simulated(live, sim)
        expected = round(0.03 - (-0.060606), 6)
        assert abs(cmp["improvement_delta"] - expected) < 1e-5

    def test_exact_delta_values(self) -> None:
        """SCENARIO-LEARN-024: each delta == live_value - simulated_value."""
        live = self._make_live_wrapper(
            batch1_accuracy=0.75,
            batch3_accuracy=0.70,
            improvement_1to3=-0.05,
        )
        sim = _make_simulated_318()
        cmp = compare_relay_to_simulated(live, sim)
        assert abs(cmp["batch1_accuracy_delta"] - (0.75 - 0.69697)) < 1e-5
        assert abs(cmp["batch3_accuracy_delta"] - (0.70 - 0.636364)) < 1e-5
        assert abs(cmp["improvement_delta"] - (-0.05 - (-0.060606))) < 1e-5

    def test_negative_improvement_delta_not_clamped(self) -> None:
        """SCENARIO-LEARN-024: negative improvement_delta is preserved (not clamped to 0)."""
        live = self._make_live_wrapper(improvement_1to3=-0.15)
        sim = _make_simulated_318()  # improvement_1to3 = -0.060606
        cmp = compare_relay_to_simulated(live, sim)
        # -0.15 - (-0.060606) = -0.089394 → must be negative
        assert cmp["improvement_delta"] < 0.0

    def test_equal_values_produce_zero_delta(self) -> None:
        """SCENARIO-LEARN-024: identical live and simulated produce zero deltas."""
        live = self._make_live_wrapper(
            batch1_accuracy=0.69697,
            batch3_accuracy=0.636364,
            improvement_1to3=-0.060606,
        )
        sim = _make_simulated_318()
        cmp = compare_relay_to_simulated(live, sim)
        assert abs(cmp["batch1_accuracy_delta"]) < 1e-4
        assert abs(cmp["batch3_accuracy_delta"]) < 1e-4
        assert abs(cmp["improvement_delta"]) < 1e-4


# ---------------------------------------------------------------------------
# Tests: primary_metric_honest — improvement_1to3 is signed and not clamped
# Spec: REQ-LEARN-014-3, SCENARIO-LEARN-022
# ---------------------------------------------------------------------------


class TestPrimaryMetricHonest:
    """improvement_1to3 in the live wrapper is signed and never clamped.

    This class ensures that when the live run produces a regression (batch3 worse
    than batch1), the negative value is faithfully reported in the artifact.

    Spec: REQ-LEARN-014-3, SCENARIO-LEARN-022
    """

    def test_negative_improvement_preserved_in_wrapper(self, tmp_path: Path) -> None:
        """REQ-LEARN-014-3: negative improvement_1to3 survives JSON round-trip."""
        artifact = _make_valid_wrapper(improvement_1to3=-0.09)
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert loaded["improvement_1to3"] < 0.0

    def test_positive_improvement_preserved_in_wrapper(self, tmp_path: Path) -> None:
        """REQ-LEARN-014-3: positive improvement_1to3 survives JSON round-trip."""
        artifact = _make_valid_wrapper(improvement_1to3=0.06)
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert loaded["improvement_1to3"] > 0.0

    def test_zero_improvement_preserved(self, tmp_path: Path) -> None:
        """REQ-LEARN-014-3: zero improvement_1to3 survives JSON round-trip."""
        artifact = _make_valid_wrapper(improvement_1to3=0.0)
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert loaded["improvement_1to3"] == 0.0


# ---------------------------------------------------------------------------
# Tests: jepa_skip_rate_live present and bounded
# Spec: REQ-LEARN-014-5
# ---------------------------------------------------------------------------


class TestJepaSkipRateLive:
    """jepa_skip_rate_live must appear in the wrapper and be in [0, 1].

    Spec: REQ-LEARN-014-5
    """

    def test_jepa_skip_rate_live_present(self, tmp_path: Path) -> None:
        """REQ-LEARN-014-5: jepa_skip_rate_live present in artifact."""
        artifact = _make_valid_wrapper(jepa_skip_rate_live=0.24)
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert "jepa_skip_rate_live" in loaded

    def test_jepa_skip_rate_live_in_unit_range(self, tmp_path: Path) -> None:
        """REQ-LEARN-014-5: jepa_skip_rate_live is in [0.0, 1.0]."""
        for rate in [0.0, 0.24, 0.5, 1.0]:
            artifact = _make_valid_wrapper(jepa_skip_rate_live=rate)
            p = tmp_path / f"wrapper_{int(rate*100)}.json"
            p.write_text(json.dumps(artifact), encoding="utf-8")
            loaded = load_live_relay_results(p)
            assert 0.0 <= loaded["jepa_skip_rate_live"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: artifact schema constants
# Spec: REQ-LEARN-014
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Exp 329 artifact schema constants are correct.

    Spec: REQ-LEARN-014
    """

    def test_experiment_number(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: wrapper experiment field is 329."""
        artifact = _make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert loaded["experiment"] == 329

    def test_schema_identifier(self, tmp_path: Path) -> None:
        """REQ-LEARN-014: schema field is 'carnot.live_relay_benchmark.v1'."""
        artifact = _make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        assert loaded["schema"] == "carnot.live_relay_benchmark.v1"

    def test_simulation_comparison_keys(self, tmp_path: Path) -> None:
        """REQ-LEARN-014-4: simulation_comparison has all three delta keys."""
        artifact = _make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact), encoding="utf-8")
        loaded = load_live_relay_results(p)
        sc = loaded["simulation_comparison"]
        assert "batch1_accuracy_delta" in sc
        assert "batch3_accuracy_delta" in sc
        assert "improvement_delta" in sc


# ---------------------------------------------------------------------------
# Integration test: live artifact (skip if absent)
# Spec: REQ-LEARN-014
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def live_relay_result() -> dict[str, Any]:
    """Load the Exp 329 wrapper artifact.

    Skips all tests in this session that depend on this fixture if the artifact
    has not yet been produced.  This prevents CI failures when Exp 329 has not
    been executed.
    """
    if not _LIVE_RESULT_PATH.exists():
        pytest.skip(
            f"Exp 329 artifact not found at {_LIVE_RESULT_PATH}; run Exp 329 first"
        )
    with _LIVE_RESULT_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class TestLiveArtifactIntegration:
    """Integration tests against the actual Exp 329 artifact.

    All tests in this class are skipped when the artifact is absent.

    Spec: REQ-LEARN-014, SCENARIO-LEARN-023, SCENARIO-LEARN-024
    """

    def test_artifact_is_live_gpu(self, live_relay_result: dict[str, Any]) -> None:
        """SCENARIO-LEARN-023: live artifact inference_mode is 'live_gpu'."""
        assert live_relay_result.get("inference_mode") == "live_gpu"

    def test_required_keys_present(self, live_relay_result: dict[str, Any]) -> None:
        """REQ-LEARN-014: all required wrapper keys present in live artifact."""
        for key in REQUIRED_WRAPPER_KEYS:
            assert key in live_relay_result, f"Missing required key: {key!r}"

    def test_improvement_1to3_is_signed(self, live_relay_result: dict[str, Any]) -> None:
        """REQ-LEARN-014-3: improvement_1to3 is a float (signed, possibly negative)."""
        imp = live_relay_result.get("improvement_1to3")
        assert imp is not None
        assert isinstance(imp, float)
        # Signed: no restriction on sign — we accept negative to honour honest reporting

    def test_simulation_comparison_present(self, live_relay_result: dict[str, Any]) -> None:
        """REQ-LEARN-014-4: simulation_comparison dict present with required keys."""
        sc = live_relay_result.get("simulation_comparison", {})
        assert "batch1_accuracy_delta" in sc
        assert "batch3_accuracy_delta" in sc
        assert "improvement_delta" in sc

    def test_jepa_skip_rate_live_present(self, live_relay_result: dict[str, Any]) -> None:
        """REQ-LEARN-014-5: jepa_skip_rate_live in [0, 1]."""
        rate = live_relay_result.get("jepa_skip_rate_live")
        assert rate is not None
        assert 0.0 <= rate <= 1.0

    def test_validate_relay_live_passes(self, live_relay_result: dict[str, Any]) -> None:
        """SCENARIO-LEARN-023: validate_relay_live passes on live artifact."""
        validate_relay_live(live_relay_result)  # must not raise

    def test_compare_relay_to_simulated_signs(self, live_relay_result: dict[str, Any]) -> None:
        """SCENARIO-LEARN-024: comparison against Exp 318 baseline produces signed deltas."""
        if not _SIMULATED_RESULT_PATH.exists():
            pytest.skip("Exp 318 artifact not found — skip comparison test")
        with _SIMULATED_RESULT_PATH.open("r", encoding="utf-8") as fh:
            sim = json.load(fh)
        sc = live_relay_result.get("simulation_comparison", {})
        # Cross-check that stored deltas match the re-computed ones
        recomputed = compare_relay_to_simulated(live_relay_result, sim)
        assert abs(sc["improvement_delta"] - recomputed["improvement_delta"]) < 1e-4
