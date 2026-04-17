"""Tests for Exp 442: FOVER live CoT annotation.

Coverage targets
----------------
LiveFOVERResult:
  - Instantiation with all required fields
  - labeling_rate computed correctly (n_labeled / n_steps_found)
  - labeling_rate=0.0 when n_steps_found==0

build_live_fover_artifact():
  - source='live', n_labeled>=20 → honest_verdict='real_data_labeled'
  - source='live', n_labeled<20  → honest_verdict='real_data_insufficient'
  - source='live', n_labeled==20 → honest_verdict='real_data_labeled' (boundary)
  - source='live', n_labeled==19 → honest_verdict='real_data_insufficient' (boundary)
  - source='synthetic'           → honest_verdict='synthetic_fallback' (any n_labeled)
  - schema == 'carnot.fover_live.v1' for all verdicts
  - all count fields round-trip correctly
  - labeling_rate round-trips correctly

run_experiment() (scripts/experiment_442_fover_live_annotation.py):
  - CI mode (CARNOT_FORCE_LIVE=0): honest_verdict in VALID_VERDICTS
  - all REQUIRED_RESULT_FIELDS present in artifact
  - artifact written to disk
  - env_autofix block embedded in artifact
  - fover_live block present in artifact with schema='carnot.fover_live.v1'

main():
  - calls run_experiment() inside ExperimentTimeoutWatchdog(442, ...)
  - watchdog called with experiment_id=442

Spec: REQ-LEARN-035, SCENARIO-LEARN-062, SCENARIO-LEARN-063
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from carnot.pipeline.fover_live import (  # noqa: E402
    LiveFOVERResult,
    build_live_fover_artifact,
)
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402

_VALID_VERDICTS = {"real_data_labeled", "real_data_insufficient", "synthetic_fallback"}


# ---------------------------------------------------------------------------
# LiveFOVERResult
# ---------------------------------------------------------------------------


class TestLiveFOVERResult:
    # REQ-LEARN-035

    def _make(self, **kwargs) -> LiveFOVERResult:
        defaults = dict(
            n_responses=100,
            n_steps_found=400,
            n_labeled=80,
            n_correct=50,
            n_incorrect=30,
            n_not_verifiable=320,
            labeling_rate=0.2,
            source="live",
            honest_verdict="real_data_labeled",
        )
        defaults.update(kwargs)
        return LiveFOVERResult(**defaults)

    def test_instantiation_all_fields(self):
        r = self._make()
        assert r.n_responses == 100
        assert r.n_steps_found == 400
        assert r.n_labeled == 80
        assert r.n_correct == 50
        assert r.n_incorrect == 30
        assert r.n_not_verifiable == 320
        assert r.labeling_rate == 0.2
        assert r.source == "live"
        assert r.honest_verdict == "real_data_labeled"

    def test_labeling_rate_fraction(self):
        # n_labeled / n_steps_found
        r = self._make(n_steps_found=200, n_labeled=50, labeling_rate=50 / 200)
        assert abs(r.labeling_rate - 0.25) < 1e-9

    def test_labeling_rate_zero_when_no_steps(self):
        r = self._make(n_steps_found=0, n_labeled=0, labeling_rate=0.0)
        assert r.labeling_rate == 0.0

    def test_source_synthetic(self):
        r = self._make(source="synthetic", honest_verdict="synthetic_fallback")
        assert r.source == "synthetic"

    def test_all_correct_no_incorrect(self):
        r = self._make(n_labeled=10, n_correct=10, n_incorrect=0)
        assert r.n_correct == 10
        assert r.n_incorrect == 0

    def test_zero_labeled(self):
        r = self._make(n_labeled=0, n_correct=0, n_incorrect=0, labeling_rate=0.0)
        assert r.n_labeled == 0


# ---------------------------------------------------------------------------
# build_live_fover_artifact
# ---------------------------------------------------------------------------


def _make_result(source: str, n_labeled: int) -> LiveFOVERResult:
    return LiveFOVERResult(
        n_responses=50,
        n_steps_found=200,
        n_labeled=n_labeled,
        n_correct=n_labeled,
        n_incorrect=0,
        n_not_verifiable=200 - n_labeled,
        labeling_rate=n_labeled / 200 if 200 > 0 else 0.0,
        source=source,  # type: ignore[arg-type]
        honest_verdict="",
    )


class TestBuildLiveFOVERArtifact:
    # SCENARIO-LEARN-062

    def test_live_sufficient_verdict(self):
        art = build_live_fover_artifact(_make_result("live", 20))
        assert art["honest_verdict"] == "real_data_labeled"

    def test_live_insufficient_verdict(self):
        art = build_live_fover_artifact(_make_result("live", 19))
        assert art["honest_verdict"] == "real_data_insufficient"

    def test_live_boundary_exactly_20(self):
        art = build_live_fover_artifact(_make_result("live", 20))
        assert art["honest_verdict"] == "real_data_labeled"

    def test_live_boundary_19(self):
        art = build_live_fover_artifact(_make_result("live", 19))
        assert art["honest_verdict"] == "real_data_insufficient"

    def test_live_zero_labeled(self):
        art = build_live_fover_artifact(_make_result("live", 0))
        assert art["honest_verdict"] == "real_data_insufficient"

    def test_live_large_labeled(self):
        art = build_live_fover_artifact(_make_result("live", 500))
        assert art["honest_verdict"] == "real_data_labeled"

    # SCENARIO-LEARN-063

    def test_synthetic_fallback(self):
        art = build_live_fover_artifact(_make_result("synthetic", 0))
        assert art["honest_verdict"] == "synthetic_fallback"

    def test_synthetic_with_labeled_still_fallback(self):
        # Even with many labeled pairs, synthetic source → fallback
        art = build_live_fover_artifact(_make_result("synthetic", 100))
        assert art["honest_verdict"] == "synthetic_fallback"

    def test_schema_live(self):
        art = build_live_fover_artifact(_make_result("live", 25))
        assert art["schema"] == "carnot.fover_live.v1"

    def test_schema_synthetic(self):
        art = build_live_fover_artifact(_make_result("synthetic", 0))
        assert art["schema"] == "carnot.fover_live.v1"

    def test_count_fields_round_trip(self):
        r = _make_result("live", 30)
        art = build_live_fover_artifact(r)
        assert art["n_responses"] == r.n_responses
        assert art["n_steps_found"] == r.n_steps_found
        assert art["n_labeled"] == r.n_labeled
        assert art["n_correct"] == r.n_correct
        assert art["n_incorrect"] == r.n_incorrect
        assert art["n_not_verifiable"] == r.n_not_verifiable

    def test_labeling_rate_round_trip(self):
        r = LiveFOVERResult(
            n_responses=10,
            n_steps_found=100,
            n_labeled=25,
            n_correct=15,
            n_incorrect=10,
            n_not_verifiable=75,
            labeling_rate=0.25,
            source="live",
            honest_verdict="",
        )
        art = build_live_fover_artifact(r)
        assert abs(art["labeling_rate"] - 0.25) < 1e-9

    def test_source_round_trip_live(self):
        art = build_live_fover_artifact(_make_result("live", 30))
        assert art["source"] == "live"

    def test_source_round_trip_synthetic(self):
        art = build_live_fover_artifact(_make_result("synthetic", 30))
        assert art["source"] == "synthetic"


# ---------------------------------------------------------------------------
# run_experiment (scripts/experiment_442_fover_live_annotation.py)
# ---------------------------------------------------------------------------


def _import_exp442():
    import importlib
    import importlib.util

    script = _REPO_ROOT / "scripts" / "experiment_442_fover_live_annotation.py"
    spec = importlib.util.spec_from_file_location("experiment_442", script)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class TestRunExperiment442:
    # CI mode: CARNOT_FORCE_LIVE=0 so live GPU is skipped

    def test_ci_mode_valid_verdict(self, tmp_path):
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()
            with patch.object(
                mod, "DELIVERABLE", str(tmp_path / "result.json")
            ), patch.object(
                mod, "LABELED_STEPS_PATH", str(tmp_path / "labeled.json")
            ):
                artifact = mod.run_experiment()
        assert artifact["fover_live"]["honest_verdict"] in _VALID_VERDICTS

    def test_required_result_fields_present(self, tmp_path):
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()
            with patch.object(
                mod, "DELIVERABLE", str(tmp_path / "result.json")
            ), patch.object(
                mod, "LABELED_STEPS_PATH", str(tmp_path / "labeled.json")
            ):
                artifact = mod.run_experiment()
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_artifact_written_to_disk(self, tmp_path):
        out = tmp_path / "result.json"
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()
            with patch.object(mod, "DELIVERABLE", str(out)), patch.object(
                mod, "LABELED_STEPS_PATH", str(tmp_path / "labeled.json")
            ):
                mod.run_experiment()
        assert out.exists()
        data = json.loads(out.read_text())
        assert "honest_verdict" in data

    def test_env_autofix_block_embedded(self, tmp_path):
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()
            with patch.object(
                mod, "DELIVERABLE", str(tmp_path / "result.json")
            ), patch.object(
                mod, "LABELED_STEPS_PATH", str(tmp_path / "labeled.json")
            ):
                artifact = mod.run_experiment()
        assert "env_autofix" in artifact

    def test_fover_live_block_present(self, tmp_path):
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()
            with patch.object(
                mod, "DELIVERABLE", str(tmp_path / "result.json")
            ), patch.object(
                mod, "LABELED_STEPS_PATH", str(tmp_path / "labeled.json")
            ):
                artifact = mod.run_experiment()
        assert "fover_live" in artifact
        assert artifact["fover_live"]["schema"] == "carnot.fover_live.v1"

    def test_labeled_steps_file_written(self, tmp_path):
        labeled = tmp_path / "labeled.json"
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()
            with patch.object(
                mod, "DELIVERABLE", str(tmp_path / "result.json")
            ), patch.object(mod, "LABELED_STEPS_PATH", str(labeled)):
                mod.run_experiment()
        assert labeled.exists()
        pairs = json.loads(labeled.read_text())
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# main() — watchdog wiring
# ---------------------------------------------------------------------------


class TestMain442:
    def test_watchdog_called_with_exp_442(self, tmp_path):
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()

        watchdog_calls = []

        class FakeWatchdog:
            def __init__(self, exp_id, timeout_minutes, result_path=None):
                watchdog_calls.append(exp_id)
                self.exp_id = exp_id

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}), patch.object(
            mod, "ExperimentTimeoutWatchdog", FakeWatchdog
        ), patch.object(
            mod, "DELIVERABLE", str(tmp_path / "r.json")
        ), patch.object(
            mod, "LABELED_STEPS_PATH", str(tmp_path / "l.json")
        ):
            mod.main()

        assert 442 in watchdog_calls

    def test_main_calls_run_experiment(self, tmp_path):
        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}):
            mod = _import_exp442()

        called = []

        def fake_run():
            called.append(True)
            return {
                "experiment": 442,
                "title": "t",
                "run_date": "d",
                "started_at": "s",
                "finished_at": "f",
                "duration_s": 0.0,
                "status": "success",
                "schema": "carnot.fover_live.v1",
                "honest_verdict": "synthetic_fallback",
                "env_autofix": {},
                "fover_live": {"schema": "carnot.fover_live.v1", "honest_verdict": "synthetic_fallback"},
            }

        class FakeWatchdog:
            def __init__(self, *a, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        with patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}), patch.object(
            mod, "run_experiment", fake_run
        ), patch.object(
            mod, "ExperimentTimeoutWatchdog", FakeWatchdog
        ), patch.object(
            mod, "DELIVERABLE", str(tmp_path / "r.json")
        ), patch.object(
            mod, "LABELED_STEPS_PATH", str(tmp_path / "l.json")
        ):
            mod.main()

        assert called
