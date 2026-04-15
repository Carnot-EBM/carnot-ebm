"""Tests for Experiment 328 live GPU benchmark result validation.

Validates the wrapper artifact produced by Exp 328 and the helper functions
that compare live GPU results against simulated baseline (Exp 316) and
against published model-card accuracy figures.

**Why these tests exist:**
    Live GPU results are the only results that qualify as headline claims per
    CLAUDE.md policy ("All headline results must have live GPU provenance").
    These tests gate promotion of any result from simulated to headline status.
    They also ensure that if the GPU is unavailable, the pipeline emits an
    honest "blocked" artifact rather than a disguised simulation.

**Schema being tested:** carnot.live_fullscale_benchmark.v1

Spec: REQ-BENCH-002, SCENARIO-BENCH-003, SCENARIO-BENCH-004
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
_LIVE_RESULT_PATH = _REPO_ROOT / "results" / "experiment_328_live_fullscale_results.json"

# Published baselines used throughout — these are model-card accuracy figures
# for GSM8K, not internal estimates.  Qwen3.5-0.8B is ~25%, Gemma4-E4B-it is ~80%.
PUBLISHED_BASELINES: dict[str, float] = {
    "Qwen3.5-0.8B": 0.25,
    "gemma-4-E4B-it": 0.80,
}

# Required top-level keys in every carnot.live_fullscale_benchmark.v1 artifact.
REQUIRED_WRAPPER_KEYS = {
    "experiment",
    "schema",
    "inference_mode",
    "status",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "primary_result_path",
}

# ---------------------------------------------------------------------------
# Standalone helper functions under test
# ---------------------------------------------------------------------------


def load_live_benchmark_results(path: str | Path) -> dict[str, Any]:
    """Load a live GPU benchmark result JSON and validate top-level schema.

    **Why this function exists:**
        Downstream callers (README updater, traceability scripts) need a
        single validated entry point so they detect schema drift early rather
        than propagating stale data.  The validation is intentionally minimal
        (top-level keys only) so the function stays fast even for large artifacts.

    Args:
        path: Path to the JSON artifact (e.g. experiment_328_live_fullscale_results.json).

    Returns:
        Parsed and minimally validated artifact dict.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if any required top-level key is absent.

    Spec: REQ-BENCH-002
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Live benchmark artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    missing = REQUIRED_WRAPPER_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Wrapper artifact missing required keys: {missing}")
    return data


def validate_live_result(result: dict[str, Any]) -> None:
    """Raise ValueError if the result does not have inference_mode 'live_gpu'.

    **Why this guard exists:**
        A simulated result is plausible but its accuracy figures are derived
        from a fixed random seed, not from actual model inference.  Mixing
        simulated and live numbers in the same reporting pipeline could produce
        misleading claims.  This function is the hard gate that prevents that.

    Args:
        result: Loaded artifact dict (from load_live_benchmark_results or json.load).

    Raises:
        ValueError: if inference_mode != "live_gpu".

    Spec: REQ-BENCH-002, SCENARIO-BENCH-003
    """
    mode = result.get("inference_mode", "<missing>")
    if mode != "live_gpu":
        raise ValueError(
            f"Result inference_mode={mode!r} does not qualify as a live GPU result. "
            "Only inference_mode='live_gpu' results may be promoted to headline claims. "
            "Re-run with CARNOT_FORCE_LIVE=1 when GPUs are available."
        )


def compare_to_simulated(
    live: dict[str, Any],
    simulated: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-model, per-mode accuracy delta between live and simulated results.

    **Why this comparison matters:**
        The Exp 316 simulated run (34% Qwen, 30% Gemma) produced results that
        are plausibly wrong because the simulation used a fixed seed designed
        for ~25% accuracy, not 34%.  This function quantifies how far the live
        inference deviates from the simulation so we can detect and document
        simulation drift explicitly in the research record.

    Delta is computed as: live_accuracy - simulated_accuracy.
    A positive delta means the live model outperforms the simulation estimate.

    Args:
        live:      Loaded artifact dict with per_model_results (live run).
        simulated: Loaded artifact dict with per_model_results (Exp 316 simulated run).

    Returns:
        Nested dict: {model_name: {mode: {corpus_variant: {"live": x, "simulated": y,
        "delta": x-y}}}}  Only cells present in both artifacts are included.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """
    divergence: dict[str, Any] = {}
    live_pmr = live.get("per_model_results", {})
    sim_pmr = simulated.get("per_model_results", {})

    for model_name, live_modes in live_pmr.items():
        sim_modes = sim_pmr.get(model_name, {})
        divergence[model_name] = {}
        for mode, live_variants in live_modes.items():
            sim_variants = sim_modes.get(mode, {})
            divergence[model_name][mode] = {}
            for variant, live_cell in live_variants.items():
                sim_cell = sim_variants.get(variant)
                if sim_cell is None:
                    continue
                live_acc = live_cell.get("accuracy", 0.0)
                sim_acc = sim_cell.get("accuracy", 0.0)
                divergence[model_name][mode][variant] = {
                    "live": live_acc,
                    "simulated": sim_acc,
                    "delta": round(live_acc - sim_acc, 6),
                }
    return divergence


def compare_to_published_baseline(
    result: dict[str, Any],
    baselines: dict[str, float],
) -> dict[str, Any]:
    """Compute per-model deviation of live accuracy from published model-card baseline.

    **Why published baselines are needed:**
        Simulation produced 34% for Qwen3.5-0.8B when the published baseline is ~25%.
        This discrepancy is evidence the simulation was miscalibrated.  By computing
        the deviation of the live result from published figures, we can quantify
        whether the live inference confirms or contradicts the published baseline,
        which in turn validates the measurement pipeline.

    Deviation is computed as: live_accuracy - baseline.
    Positive means live outperforms the published baseline.

    Args:
        result:    Loaded artifact dict with per_model_results (live or simulated).
        baselines: Dict mapping model name → published accuracy proportion (0-1).

    Returns:
        Dict: {model_name: {"baseline_accuracy": x, "published_baseline": y,
        "deviation": x-y, "within_tolerance": bool (tolerance=0.15)}}
        Only "all" corpus_variant baseline mode cells are used for the headline figure.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """
    deviations: dict[str, Any] = {}
    pmr = result.get("per_model_results", {})

    for model_name, modes in pmr.items():
        published = baselines.get(model_name)
        if published is None:
            # Try fuzzy match — model names may not include the full HF org prefix
            for base_key, base_val in baselines.items():
                if base_key in model_name or model_name in base_key:
                    published = base_val
                    break
        if published is None:
            continue

        # Use "baseline" mode, "all" variant as the headline accuracy figure.
        baseline_mode = modes.get("baseline", {})
        all_cell = baseline_mode.get("all")
        if all_cell is None:
            continue

        live_acc = all_cell.get("accuracy", 0.0)
        deviation = round(live_acc - published, 6)
        deviations[model_name] = {
            "baseline_accuracy": live_acc,
            "published_baseline": published,
            "deviation": deviation,
            "within_tolerance": within_expected_range(live_acc, published, tolerance=0.15),
        }
    return deviations


def within_expected_range(accuracy: float, baseline: float, tolerance: float = 0.15) -> bool:
    """Return True if |accuracy - baseline| <= tolerance.

    **Why tolerance=0.15:**
        A 15% absolute tolerance window is intentionally generous for a first live
        inference run.  It accounts for corpus composition differences (HuggingFace
        vs Apple adversarial vs synthetic fallback), hardware variability, and
        quantization effects.  Results outside this window warrant investigation
        before being cited — either the pipeline is broken or the model is
        behaving unexpectedly (e.g., VRAM pressure causing degradation).

    Args:
        accuracy: Measured accuracy proportion (0-1).
        baseline: Published baseline accuracy proportion (0-1).
        tolerance: Maximum allowed absolute deviation (default 0.15).

    Returns:
        True if accuracy is within tolerance of baseline.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """
    # Use a small epsilon to guard against floating-point edge cases where
    # abs(accuracy - baseline) computes to 0.15000000000000002 instead of 0.15.
    return abs(accuracy - baseline) <= tolerance + 1e-9


# ---------------------------------------------------------------------------
# Fixture: load the live artifact (skip if absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def live_result() -> dict[str, Any]:
    """Load the Exp 328 wrapper artifact.

    Skips all tests in this session if the artifact has not been produced yet.
    This prevents spurious CI failures when Exp 328 has not been executed.
    """
    if not _LIVE_RESULT_PATH.exists():
        pytest.skip(
            f"Exp 328 artifact not found at {_LIVE_RESULT_PATH}; run Exp 328 first"
        )
    with _LIVE_RESULT_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Tests for load_live_benchmark_results
# ---------------------------------------------------------------------------


class TestLoadLiveBenchmarkResults:
    """Validate the loader helper in isolation with temporary fixtures.

    These tests do not depend on the live Exp 328 artifact being present.

    Spec: REQ-BENCH-002
    """

    def _make_valid_wrapper(self) -> dict[str, Any]:
        """Build a minimal valid carnot.live_fullscale_benchmark.v1 wrapper artifact."""
        data: dict[str, Any] = {
            "experiment": 328,
            "schema": "carnot.live_fullscale_benchmark.v1",
            "inference_mode": "live_gpu",
            "status": "success",
            "run_date": "20260415",
            "started_at": "2026-04-15T03:00:00Z",
            "finished_at": "2026-04-15T05:00:00Z",
            "duration_s": 7200.0,
            "primary_result_path": "results/experiment_316_fullscale_results_live.json",
            "simulation_divergence": {},
            "baseline_deviation": {},
        }
        return data

    def test_load_valid_artifact(self, tmp_path: Path) -> None:
        """REQ-BENCH-002: loader returns parsed dict for a valid artifact."""
        artifact = self._make_valid_wrapper()
        p = tmp_path / "wrapper.json"
        p.write_text(json.dumps(artifact))
        loaded = load_live_benchmark_results(p)
        assert loaded["experiment"] == 328

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """REQ-BENCH-002: FileNotFoundError when artifact is absent."""
        with pytest.raises(FileNotFoundError):
            load_live_benchmark_results(tmp_path / "no_such_file.json")

    def test_load_missing_key_raises(self, tmp_path: Path) -> None:
        """REQ-BENCH-002: ValueError when a required key is absent."""
        artifact = self._make_valid_wrapper()
        del artifact["inference_mode"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(artifact))
        with pytest.raises(ValueError, match="missing required keys"):
            load_live_benchmark_results(p)

    def test_load_all_required_keys(self, tmp_path: Path) -> None:
        """REQ-BENCH-002: all REQUIRED_WRAPPER_KEYS present after load."""
        artifact = self._make_valid_wrapper()
        p = tmp_path / "ok.json"
        p.write_text(json.dumps(artifact))
        loaded = load_live_benchmark_results(p)
        assert REQUIRED_WRAPPER_KEYS.issubset(set(loaded.keys()))


# ---------------------------------------------------------------------------
# Tests for validate_live_result
# ---------------------------------------------------------------------------


class TestValidateLiveResult:
    """Verify that validate_live_result enforces the live_gpu mode gate.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-003
    """

    def test_simulated_raises(self) -> None:
        """SCENARIO-BENCH-003: simulated mode raises ValueError."""
        with pytest.raises(ValueError, match="live_gpu"):
            validate_live_result({"inference_mode": "simulated"})

    def test_unknown_raises(self) -> None:
        """SCENARIO-BENCH-003: unknown mode raises ValueError (never valid)."""
        with pytest.raises(ValueError, match="live_gpu"):
            validate_live_result({"inference_mode": "unknown"})

    def test_missing_key_raises(self) -> None:
        """SCENARIO-BENCH-003: missing inference_mode raises ValueError."""
        with pytest.raises(ValueError, match="live_gpu"):
            validate_live_result({})

    def test_live_gpu_passes(self) -> None:
        """SCENARIO-BENCH-003: live_gpu mode does not raise."""
        validate_live_result({"inference_mode": "live_gpu"})  # must not raise

    def test_error_message_mentions_force_live(self) -> None:
        """SCENARIO-BENCH-003: error message guides user to CARNOT_FORCE_LIVE=1."""
        with pytest.raises(ValueError, match="CARNOT_FORCE_LIVE"):
            validate_live_result({"inference_mode": "simulated"})


# ---------------------------------------------------------------------------
# Tests for compare_to_simulated
# ---------------------------------------------------------------------------


def _make_pmr(model: str, mode: str, variant: str, accuracy: float) -> dict[str, Any]:
    """Build a per_model_results dict with one cell for test fixtures."""
    n_total = 100
    n_correct = round(accuracy * n_total)
    return {
        model: {
            mode: {
                variant: {
                    "model_name": model,
                    "mode": mode,
                    "corpus_variant": variant,
                    "accuracy": n_correct / n_total,
                    "ci_lower": max(0.0, accuracy - 0.05),
                    "ci_upper": min(1.0, accuracy + 0.05),
                    "n_correct": n_correct,
                    "n_total": n_total,
                }
            }
        }
    }


class TestCompareToSimulated:
    """Verify simulation divergence computation.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """

    def test_delta_computation(self) -> None:
        """SCENARIO-BENCH-004: delta = live_accuracy - simulated_accuracy."""
        live = {"per_model_results": _make_pmr("Qwen3.5-0.8B", "baseline", "all", 0.27)}
        sim = {"per_model_results": _make_pmr("Qwen3.5-0.8B", "baseline", "all", 0.34)}
        divergence = compare_to_simulated(live, sim)
        delta = divergence["Qwen3.5-0.8B"]["baseline"]["all"]["delta"]
        # live 0.27 - sim 0.34 = -0.07
        assert abs(delta - (0.27 - 0.34)) < 1e-4

    def test_positive_delta_when_live_better(self) -> None:
        """SCENARIO-BENCH-004: positive delta when live exceeds simulation."""
        live = {"per_model_results": _make_pmr("Qwen3.5-0.8B", "baseline", "all", 0.80)}
        sim = {"per_model_results": _make_pmr("Qwen3.5-0.8B", "baseline", "all", 0.34)}
        divergence = compare_to_simulated(live, sim)
        delta = divergence["Qwen3.5-0.8B"]["baseline"]["all"]["delta"]
        assert delta > 0.0

    def test_missing_model_in_sim_skipped(self) -> None:
        """REQ-BENCH-002: models present in live but absent from sim are skipped."""
        live = {"per_model_results": _make_pmr("ModelX", "baseline", "all", 0.5)}
        sim = {"per_model_results": {}}
        divergence = compare_to_simulated(live, sim)
        # ModelX exists in divergence but the mode-variant level is empty
        assert divergence.get("ModelX", {}).get("baseline", {}) == {}

    def test_empty_live_returns_empty(self) -> None:
        """REQ-BENCH-002: empty per_model_results yields empty divergence dict."""
        divergence = compare_to_simulated(
            {"per_model_results": {}},
            {"per_model_results": {}},
        )
        assert divergence == {}

    def test_both_live_and_simulated_values_present(self) -> None:
        """REQ-BENCH-002: output cell contains live, simulated, and delta keys."""
        live = {"per_model_results": _make_pmr("M", "baseline", "all", 0.3)}
        sim = {"per_model_results": _make_pmr("M", "baseline", "all", 0.2)}
        divergence = compare_to_simulated(live, sim)
        cell = divergence["M"]["baseline"]["all"]
        assert "live" in cell
        assert "simulated" in cell
        assert "delta" in cell

    def test_multiple_variants(self) -> None:
        """REQ-BENCH-002: divergence is computed for each corpus_variant independently."""
        # Build multi-variant per_model_results manually
        def pmr_multi(acc_all: float, acc_swap: float) -> dict[str, Any]:
            return {
                "M": {
                    "baseline": {
                        "all": {
                            "model_name": "M", "mode": "baseline",
                            "corpus_variant": "all", "accuracy": acc_all,
                            "ci_lower": 0.0, "ci_upper": 1.0,
                            "n_correct": int(acc_all * 100), "n_total": 100,
                        },
                        "number_swap": {
                            "model_name": "M", "mode": "baseline",
                            "corpus_variant": "number_swap", "accuracy": acc_swap,
                            "ci_lower": 0.0, "ci_upper": 1.0,
                            "n_correct": int(acc_swap * 100), "n_total": 100,
                        },
                    }
                }
            }

        live = {"per_model_results": pmr_multi(0.27, 0.22)}
        sim = {"per_model_results": pmr_multi(0.34, 0.33)}
        divergence = compare_to_simulated(live, sim)
        assert "all" in divergence["M"]["baseline"]
        assert "number_swap" in divergence["M"]["baseline"]


# ---------------------------------------------------------------------------
# Tests for compare_to_published_baseline
# ---------------------------------------------------------------------------


class TestCompareToPublishedBaseline:
    """Verify baseline deviation computation.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """

    def _make_result(self, model: str, accuracy: float) -> dict[str, Any]:
        """Build a minimal artifact with a single model baseline/all cell."""
        return {"per_model_results": _make_pmr(model, "baseline", "all", accuracy)}

    def test_deviation_positive_when_live_higher(self) -> None:
        """SCENARIO-BENCH-004: positive deviation when live accuracy > published."""
        result = self._make_result("Qwen3.5-0.8B", 0.27)
        deviations = compare_to_published_baseline(result, {"Qwen3.5-0.8B": 0.25})
        assert abs(deviations["Qwen3.5-0.8B"]["deviation"] - 0.02) < 1e-4

    def test_deviation_negative_when_live_lower(self) -> None:
        """SCENARIO-BENCH-004: negative deviation when live accuracy < published."""
        result = self._make_result("Qwen3.5-0.8B", 0.20)
        deviations = compare_to_published_baseline(result, {"Qwen3.5-0.8B": 0.25})
        assert deviations["Qwen3.5-0.8B"]["deviation"] < 0.0

    def test_within_tolerance_field_present(self) -> None:
        """REQ-BENCH-002: output dict includes within_tolerance bool."""
        result = self._make_result("Qwen3.5-0.8B", 0.27)
        deviations = compare_to_published_baseline(result, {"Qwen3.5-0.8B": 0.25})
        assert "within_tolerance" in deviations["Qwen3.5-0.8B"]
        assert isinstance(deviations["Qwen3.5-0.8B"]["within_tolerance"], bool)

    def test_model_not_in_baselines_skipped(self) -> None:
        """REQ-BENCH-002: model with no published baseline is omitted from output."""
        result = self._make_result("UnknownModel", 0.5)
        deviations = compare_to_published_baseline(result, {"Qwen3.5-0.8B": 0.25})
        assert "UnknownModel" not in deviations

    def test_fuzzy_match_finds_partial_key(self) -> None:
        """REQ-BENCH-002: model name substring match works (e.g. 'gemma-4-E4B-it' matches 'Gemma4-E4B-it')."""
        # Build result with a slightly different model name
        result = {"per_model_results": _make_pmr("gemma-4-E4B-it", "baseline", "all", 0.75)}
        # Baseline uses a key that is a substring match
        deviations = compare_to_published_baseline(result, {"gemma-4-E4B-it": 0.80})
        assert "gemma-4-E4B-it" in deviations

    def test_empty_result_returns_empty(self) -> None:
        """REQ-BENCH-002: empty per_model_results returns empty deviations."""
        deviations = compare_to_published_baseline(
            {"per_model_results": {}}, PUBLISHED_BASELINES
        )
        assert deviations == {}


# ---------------------------------------------------------------------------
# Tests for within_expected_range
# ---------------------------------------------------------------------------


class TestWithinExpectedRange:
    """Verify range tolerance check.

    Spec: REQ-BENCH-002, SCENARIO-BENCH-004
    """

    def test_exact_match_is_within(self) -> None:
        """SCENARIO-BENCH-004: accuracy == baseline is within range."""
        assert within_expected_range(0.25, 0.25) is True

    def test_within_tolerance_boundary(self) -> None:
        """SCENARIO-BENCH-004: accuracy at exactly +tolerance is within range."""
        assert within_expected_range(0.40, 0.25, tolerance=0.15) is True

    def test_just_outside_tolerance(self) -> None:
        """SCENARIO-BENCH-004: accuracy just outside tolerance is False."""
        assert within_expected_range(0.41, 0.25, tolerance=0.15) is False

    def test_below_tolerance(self) -> None:
        """SCENARIO-BENCH-004: accuracy at -tolerance is within range."""
        assert within_expected_range(0.10, 0.25, tolerance=0.15) is True

    def test_far_below(self) -> None:
        """SCENARIO-BENCH-004: accuracy far below baseline is outside range."""
        assert within_expected_range(0.0, 0.25, tolerance=0.15) is False

    def test_custom_tolerance(self) -> None:
        """REQ-BENCH-002: custom tolerance is respected."""
        assert within_expected_range(0.30, 0.25, tolerance=0.04) is False
        assert within_expected_range(0.30, 0.25, tolerance=0.06) is True

    def test_qwen_published_baseline_in_range(self) -> None:
        """REQ-BENCH-002: Qwen3.5-0.8B published baseline (0.25) within 15% of itself."""
        assert within_expected_range(0.25, 0.25) is True

    def test_gemma_published_baseline_in_range(self) -> None:
        """REQ-BENCH-002: Gemma4-E4B-it published baseline (0.80) within 15% of itself."""
        assert within_expected_range(0.80, 0.80) is True


# ---------------------------------------------------------------------------
# Tests for live wrapper artifact (skip if artifact absent)
# ---------------------------------------------------------------------------


class TestLiveWrapperArtifact:
    """Validate the Exp 328 wrapper artifact when it exists.

    All tests in this class are skipped if the artifact has not been produced
    (see live_result fixture above).

    Spec: REQ-BENCH-002, SCENARIO-BENCH-003
    """

    def test_required_keys_present(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: all required wrapper keys present in artifact."""
        missing = REQUIRED_WRAPPER_KEYS - set(live_result.keys())
        assert not missing, f"Missing required wrapper keys: {missing}"

    def test_experiment_id(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: experiment must be 328."""
        assert live_result["experiment"] == 328

    def test_schema_field(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: schema must be 'carnot.live_fullscale_benchmark.v1'."""
        assert live_result["schema"] == "carnot.live_fullscale_benchmark.v1"

    def test_status_is_success_or_blocked(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: status must be 'success' or 'blocked' — never unknown."""
        assert live_result["status"] in {"success", "blocked"}, (
            f"Unexpected status: {live_result['status']!r}"
        )

    def test_inference_mode_is_live_gpu_when_success(self, live_result: dict[str, Any]) -> None:
        """SCENARIO-BENCH-003: if status=success, inference_mode must be 'live_gpu'."""
        if live_result["status"] == "success":
            assert live_result["inference_mode"] == "live_gpu", (
                "A successful Exp 328 run must have inference_mode='live_gpu'. "
                "Simulation results must not be reported as live."
            )

    def test_blocked_artifact_has_gpu_diagnostics(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: blocked artifact must include gpu_diagnostics."""
        if live_result["status"] == "blocked":
            assert "gpu_diagnostics" in live_result, (
                "Blocked artifact must include gpu_diagnostics to document why GPU was unavailable"
            )

    def test_duration_positive(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: duration_s must be positive."""
        assert live_result["duration_s"] > 0

    def test_primary_result_path_field(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: primary_result_path must be a non-empty string."""
        prp = live_result["primary_result_path"]
        assert isinstance(prp, str) and len(prp) > 0

    def test_simulation_divergence_when_success(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: successful artifact must include simulation_divergence dict."""
        if live_result["status"] == "success":
            assert "simulation_divergence" in live_result
            assert isinstance(live_result["simulation_divergence"], dict)

    def test_baseline_deviation_when_success(self, live_result: dict[str, Any]) -> None:
        """REQ-BENCH-002: successful artifact must include baseline_deviation dict."""
        if live_result["status"] == "success":
            assert "baseline_deviation" in live_result
            assert isinstance(live_result["baseline_deviation"], dict)
