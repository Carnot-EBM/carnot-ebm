"""Tests for scripts/experiment_833_constraint_delta_root_cause.py.

Traces to: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-060

**What we test:**
    - _instrument_store() correctly counts store() calls (H1 detection).
    - _instrument_store() correctly counts retrieve() calls and records norms.
    - _instrument_injector() correctly counts compute_energy_with_external_field calls.
    - _instrument_injector() correctly counts project_to_spin_bias calls (legacy proxy).
    - compute_honest_verdict() maps counter combinations to correct verdict strings.
    - build_fix_recommendation() returns non-empty strings for all verdicts.
    - _map_verdict_to_hypothesis() maps verdicts to H1/H2/H3 labels correctly.
    - run_diagnosis() returns a dict with all required schema fields.
    - run_diagnosis() produces honest_verdict="write_path_missing" on the live pipeline
      (confirming H1: store() is never called by verify()).
    - The deliverable JSON exists and contains all required fields after main() runs.

All tests run on CPU only — no GPU or live LLM model required.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_833_constraint_delta_root_cause import (
    TEST_CASES,
    GROUND_TRUTH_CORRECT,
    _instrument_store,
    _instrument_injector,
    compute_honest_verdict,
    build_fix_recommendation,
    _map_verdict_to_hypothesis,
    run_diagnosis,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _empty_counters() -> dict[str, Any]:
    """Return a fresh counter dict matching the experiment's schema."""
    return {
        "n_store_write_calls": 0,
        "n_store_retrieve_calls": 0,
        "retrieved_vector_norms": [],
        "n_external_field_calls": 0,
        "n_legacy_energy_calls": 0,
    }


class _FakeStoreSPO:
    """Minimal SPO stub so _counted_retrieve can access .embedding."""

    def __init__(self, norm: float = 1.0) -> None:
        dim = 4
        val = norm / math.sqrt(dim)
        self.embedding = [val, val, val, val]
        self.source_violation_type = "carry"


# ---------------------------------------------------------------------------
# REQ-LEARN-048: _instrument_store counts store() and retrieve() calls
# ---------------------------------------------------------------------------


class TestInstrumentStore:
    """REQ-LEARN-048: Instrumentation must correctly track store/retrieve calls."""

    def test_store_call_counted(self) -> None:
        """_instrument_store increments n_store_write_calls on each store() call.

        Spec: REQ-LEARN-048, SCENARIO-LEARN-060
        """
        store_mock = MagicMock()
        store_mock.store = MagicMock()
        store_mock.retrieve = MagicMock(return_value=[])

        counters = _empty_counters()
        _instrument_store(store_mock, counters)

        # Call the patched store() twice.
        fake_spo = MagicMock()
        store_mock.store(fake_spo)
        store_mock.store(fake_spo)

        assert counters["n_store_write_calls"] == 2

    def test_retrieve_call_counted(self) -> None:
        """_instrument_store increments n_store_retrieve_calls on each retrieve() call.

        Spec: REQ-LEARN-048
        """
        store_mock = MagicMock()
        store_mock.store = MagicMock()
        store_mock.retrieve = MagicMock(return_value=[])

        counters = _empty_counters()
        _instrument_store(store_mock, counters)

        store_mock.retrieve("query", 3)
        store_mock.retrieve("query2", 3)

        assert counters["n_store_retrieve_calls"] == 2

    def test_retrieve_records_vector_norms(self) -> None:
        """_instrument_store appends retrieved embedding norms to retrieved_vector_norms.

        Spec: REQ-LEARN-048
        """
        spo_with_norm_1 = _FakeStoreSPO(norm=1.0)

        store_mock = MagicMock()
        store_mock.store = MagicMock()
        store_mock.retrieve = MagicMock(return_value=[spo_with_norm_1])

        counters = _empty_counters()
        _instrument_store(store_mock, counters)

        store_mock.retrieve("query", 1)

        assert len(counters["retrieved_vector_norms"]) == 1
        recorded_norm = counters["retrieved_vector_norms"][0]
        # norm of [0.5, 0.5, 0.5, 0.5] = sqrt(4 * 0.25) = 1.0
        assert abs(recorded_norm - 1.0) < 1e-6

    def test_retrieve_with_none_embedding_skipped(self) -> None:
        """_instrument_store skips SPO entries with None embeddings when computing norm.

        Spec: REQ-LEARN-048
        """
        spo_none = MagicMock()
        spo_none.embedding = None

        store_mock = MagicMock()
        store_mock.store = MagicMock()
        store_mock.retrieve = MagicMock(return_value=[spo_none])

        counters = _empty_counters()
        _instrument_store(store_mock, counters)

        store_mock.retrieve("query", 1)

        # No norms recorded because embedding was None.
        assert counters["retrieved_vector_norms"] == []

    def test_original_store_still_called(self) -> None:
        """_instrument_store wraps store() but still calls the underlying method.

        Spec: REQ-LEARN-048
        """
        underlying_calls = []

        def _real_store(spo: Any) -> None:
            underlying_calls.append(spo)

        store_mock = MagicMock()
        store_mock.store = _real_store
        store_mock.retrieve = MagicMock(return_value=[])

        counters = _empty_counters()
        _instrument_store(store_mock, counters)

        fake_spo = object()
        store_mock.store(fake_spo)

        assert underlying_calls == [fake_spo]

    def test_original_retrieve_still_called(self) -> None:
        """_instrument_store wraps retrieve() but still calls the underlying method.

        Spec: REQ-LEARN-048
        """
        underlying_calls = []

        def _real_retrieve(query: str, top_k: int = 3) -> list[Any]:
            underlying_calls.append((query, top_k))
            return []

        store_mock = MagicMock()
        store_mock.store = MagicMock()
        store_mock.retrieve = _real_retrieve

        counters = _empty_counters()
        _instrument_store(store_mock, counters)

        store_mock.retrieve("hello", 5)

        assert underlying_calls == [("hello", 5)]


# ---------------------------------------------------------------------------
# REQ-LEARN-049: _instrument_injector counts external_field and legacy calls
# ---------------------------------------------------------------------------


class TestInstrumentInjector:
    """REQ-LEARN-049: Instrumentation must correctly track injector method calls."""

    def test_external_field_call_counted(self) -> None:
        """_instrument_injector increments n_external_field_calls.

        Spec: REQ-LEARN-049
        """
        injector_mock = MagicMock()
        injector_mock.compute_energy_with_external_field = MagicMock(return_value=0.0)
        injector_mock.project_to_spin_bias = MagicMock(return_value=[])

        counters = _empty_counters()
        _instrument_injector(injector_mock, counters)

        injector_mock.compute_energy_with_external_field(None, None, [])
        injector_mock.compute_energy_with_external_field(None, None, [])

        assert counters["n_external_field_calls"] == 2

    def test_legacy_energy_call_counted(self) -> None:
        """_instrument_injector increments n_legacy_energy_calls on project_to_spin_bias.

        Spec: REQ-LEARN-049
        """
        injector_mock = MagicMock()
        injector_mock.compute_energy_with_external_field = MagicMock(return_value=0.0)
        injector_mock.project_to_spin_bias = MagicMock(return_value=[])

        counters = _empty_counters()
        _instrument_injector(injector_mock, counters)

        injector_mock.project_to_spin_bias([])
        injector_mock.project_to_spin_bias([])
        injector_mock.project_to_spin_bias([])

        assert counters["n_legacy_energy_calls"] == 3

    def test_original_ext_field_still_called(self) -> None:
        """_instrument_injector wraps compute_energy_with_external_field but calls underlying.

        Spec: REQ-LEARN-049
        """
        called_with = []

        def _real_ext(J: Any, spins: Any, embs: Any) -> float:
            called_with.append((J, spins, embs))
            return 42.0

        injector_mock = MagicMock()
        injector_mock.compute_energy_with_external_field = _real_ext
        injector_mock.project_to_spin_bias = MagicMock(return_value=[])

        counters = _empty_counters()
        _instrument_injector(injector_mock, counters)

        result = injector_mock.compute_energy_with_external_field("J", "s", "e")

        assert result == 42.0
        assert called_with == [("J", "s", "e")]


# ---------------------------------------------------------------------------
# compute_honest_verdict — SCENARIO-LEARN-060 decision tree
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """SCENARIO-LEARN-060: compute_honest_verdict maps counter patterns correctly."""

    def test_write_path_missing_when_no_writes(self) -> None:
        """H1 verdict when n_store_write_calls == 0.

        Spec: REQ-LEARN-048, SCENARIO-LEARN-060
        """
        counters = _empty_counters()
        counters["n_store_write_calls"] = 0
        counters["n_store_retrieve_calls"] = 5
        counters["n_legacy_energy_calls"] = 5
        assert compute_honest_verdict(counters) == "write_path_missing"

    def test_retrieval_returns_zeros_when_low_norm(self) -> None:
        """H2 verdict when writes > 0 but mean vector norm < 0.01.

        Spec: REQ-LEARN-048, SCENARIO-LEARN-060
        """
        counters = _empty_counters()
        counters["n_store_write_calls"] = 3
        counters["n_store_retrieve_calls"] = 5
        counters["retrieved_vector_norms"] = [0.001, 0.002, 0.0005]
        assert compute_honest_verdict(counters) == "retrieval_returns_zeros"

    def test_external_field_not_called_when_legacy_only(self) -> None:
        """H3 verdict when legacy calls > 0 and external_field_calls == 0.

        Spec: REQ-LEARN-049, SCENARIO-LEARN-060
        """
        counters = _empty_counters()
        counters["n_store_write_calls"] = 3
        counters["n_store_retrieve_calls"] = 5
        counters["retrieved_vector_norms"] = [0.5, 0.6, 0.7]
        counters["n_external_field_calls"] = 0
        counters["n_legacy_energy_calls"] = 5
        assert compute_honest_verdict(counters) == "external_field_not_called"

    def test_pipeline_wiring_correct_when_all_paths_active(self) -> None:
        """pipeline_wiring_correct when all paths are active and norms are sane.

        Spec: SCENARIO-LEARN-060
        """
        counters = _empty_counters()
        counters["n_store_write_calls"] = 3
        counters["n_store_retrieve_calls"] = 5
        counters["retrieved_vector_norms"] = [0.5, 0.6]
        counters["n_external_field_calls"] = 5
        counters["n_legacy_energy_calls"] = 5
        assert compute_honest_verdict(counters) == "pipeline_wiring_correct"

    def test_diagnosis_inconclusive_on_ambiguous_state(self) -> None:
        """diagnosis_inconclusive when no branch matches.

        Spec: SCENARIO-LEARN-060
        """
        counters = _empty_counters()
        counters["n_store_write_calls"] = 3
        counters["n_store_retrieve_calls"] = 0
        counters["retrieved_vector_norms"] = [0.5]
        counters["n_external_field_calls"] = 0
        counters["n_legacy_energy_calls"] = 0
        assert compute_honest_verdict(counters) == "diagnosis_inconclusive"


# ---------------------------------------------------------------------------
# build_fix_recommendation — non-empty strings for all verdicts
# ---------------------------------------------------------------------------


class TestBuildFixRecommendation:
    """build_fix_recommendation returns non-empty strings for all five verdicts."""

    @pytest.mark.parametrize(
        "verdict",
        [
            "write_path_missing",
            "retrieval_returns_zeros",
            "external_field_not_called",
            "pipeline_wiring_correct",
            "diagnosis_inconclusive",
        ],
    )
    def test_non_empty_for_all_verdicts(self, verdict: str) -> None:
        """build_fix_recommendation returns a non-empty string for every known verdict.

        Spec: REQ-LEARN-048, REQ-LEARN-049
        """
        rec = build_fix_recommendation(verdict)
        assert isinstance(rec, str)
        assert len(rec) > 10

    def test_unknown_verdict_handled(self) -> None:
        """build_fix_recommendation does not crash for an unknown verdict."""
        rec = build_fix_recommendation("totally_unknown_verdict_xyz")
        assert isinstance(rec, str)
        assert len(rec) > 0


# ---------------------------------------------------------------------------
# _map_verdict_to_hypothesis — label mapping
# ---------------------------------------------------------------------------


class TestMapVerdictToHypothesis:
    """_map_verdict_to_hypothesis maps verdict strings to hypothesis labels."""

    def test_h1_for_write_path_missing(self) -> None:
        """write_path_missing maps to H1."""
        assert _map_verdict_to_hypothesis("write_path_missing") == "H1"

    def test_h2_for_retrieval_zeros(self) -> None:
        """retrieval_returns_zeros maps to H2."""
        assert _map_verdict_to_hypothesis("retrieval_returns_zeros") == "H2"

    def test_h3_for_external_field_not_called(self) -> None:
        """external_field_not_called maps to H3."""
        assert _map_verdict_to_hypothesis("external_field_not_called") == "H3"

    def test_none_for_pipeline_correct(self) -> None:
        """pipeline_wiring_correct maps to none_all_correct."""
        assert _map_verdict_to_hypothesis("pipeline_wiring_correct") == "none_all_correct"

    def test_unknown_for_inconclusive(self) -> None:
        """diagnosis_inconclusive maps to unknown."""
        assert _map_verdict_to_hypothesis("diagnosis_inconclusive") == "unknown"


# ---------------------------------------------------------------------------
# run_diagnosis — integration test against the real pipeline
# ---------------------------------------------------------------------------


class TestRunDiagnosis:
    """REQ-LEARN-048 + REQ-LEARN-049: run_diagnosis() finds the actual pipeline bug."""

    def test_run_diagnosis_returns_required_fields(self) -> None:
        """run_diagnosis() returns a dict with all required diagnostic schema fields.

        Spec: REQ-LEARN-048, REQ-LEARN-049
        """
        result = run_diagnosis()
        required = [
            "hypothesis_confirmed",
            "n_store_write_calls",
            "n_store_retrieve_calls",
            "mean_retrieved_vector_norm",
            "n_external_field_calls",
            "n_legacy_energy_calls",
            "root_cause",
            "fix_recommendation",
            "honest_verdict",
        ]
        for field in required:
            assert field in result, f"Missing required field: {field}"

    def test_run_diagnosis_confirms_h1_write_path_missing(self) -> None:
        """run_diagnosis() confirms H1: store() is never called by verify().

        The live VerifyRepairPipeline.verify() calls retrieve() but never store().
        After running 5 test cases, n_store_write_calls MUST be 0.
        This confirms the write path is structurally absent (REQ-LEARN-048 violation).

        Spec: REQ-LEARN-048, SCENARIO-LEARN-060
        """
        result = run_diagnosis()
        assert result["n_store_write_calls"] == 0, (
            "Expected store() to never be called (H1 diagnosis), "
            f"but got n_store_write_calls={result['n_store_write_calls']}"
        )
        assert result["honest_verdict"] == "write_path_missing"
        assert result["hypothesis_confirmed"] == "H1"

    def test_run_diagnosis_retrieve_is_called(self) -> None:
        """verify() does call retrieve() — retrieval path is wired, just not store().

        Spec: REQ-LEARN-048
        """
        result = run_diagnosis()
        # retrieve() is called once per verify() call = 5 calls minimum.
        assert result["n_store_retrieve_calls"] >= len(TEST_CASES)

    def test_fix_recommendation_is_non_empty(self) -> None:
        """run_diagnosis() always returns a non-empty fix_recommendation.

        Spec: REQ-LEARN-048, REQ-LEARN-049
        """
        result = run_diagnosis()
        assert isinstance(result["fix_recommendation"], str)
        assert len(result["fix_recommendation"]) > 10

    def test_verify_results_has_one_entry_per_test_case(self) -> None:
        """run_diagnosis() returns one verify_result entry per test case.

        Spec: REQ-LEARN-048
        """
        result = run_diagnosis()
        assert len(result["verify_results"]) == len(TEST_CASES)

    def test_embedding_mode_field_present(self) -> None:
        """run_diagnosis() records the embedding_mode (ci_hash or sentence_transformer).

        Spec: REQ-LEARN-048
        """
        result = run_diagnosis()
        assert result["embedding_mode"] in ("ci_hash", "sentence_transformer")

    def test_synthetic_test_cases_have_expected_shape(self) -> None:
        """TEST_CASES is a list of (question, response) 2-tuples.

        Spec: SCENARIO-LEARN-060
        """
        assert len(TEST_CASES) == 5
        for q, r in TEST_CASES:
            assert isinstance(q, str) and len(q) > 0
            assert isinstance(r, str) and len(r) > 0

    def test_ground_truth_correct_has_expected_length(self) -> None:
        """GROUND_TRUTH_CORRECT has one entry per TEST_CASES entry.

        Spec: SCENARIO-LEARN-060
        """
        assert len(GROUND_TRUTH_CORRECT) == len(TEST_CASES)
        for val in GROUND_TRUTH_CORRECT:
            assert isinstance(val, bool)


# ---------------------------------------------------------------------------
# Deliverable JSON validation
# ---------------------------------------------------------------------------


class TestDeliverableJson:
    """Deliverable JSON exists after main() and contains all required schema fields."""

    def test_deliverable_exists_and_has_required_fields(self) -> None:
        """Run main() and verify the deliverable JSON has all required schema fields.

        Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-060
        """
        import subprocess
        import os

        result = subprocess.run(
            [sys.executable, "scripts/experiment_833_constraint_delta_root_cause.py"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        # Allow non-zero exit only if the deliverable was still written.
        deliverable = _REPO / "results" / "experiment_833_constraint_delta_root_cause.json"
        assert deliverable.exists(), (
            f"Deliverable not found after main().\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

        with deliverable.open() as fh:
            artifact = json.load(fh)

        required_schema_fields = [
            "experiment",
            "title",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "schema",
            "honest_verdict",
            "hypothesis_confirmed",
            "n_store_write_calls",
            "n_store_retrieve_calls",
            "mean_retrieved_vector_norm",
            "n_external_field_calls",
            "n_legacy_energy_calls",
            "root_cause",
            "fix_recommendation",
        ]
        for field in required_schema_fields:
            assert field in artifact, f"Missing field '{field}' in deliverable JSON"

        assert artifact["experiment"] == 833
        assert artifact["status"] == "success"
        assert artifact["honest_verdict"] == "write_path_missing"
