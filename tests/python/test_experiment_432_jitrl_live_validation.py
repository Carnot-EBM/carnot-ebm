"""Tests for scripts/experiment_432_jitrl_live_validation.py and JitRLConstraintMemory.

100% coverage for:
  - load_live_violations(): parse Exp 427 result for violations + outcomes
  - build_jitrl_validation_artifact(): schema, fp_reduction_pct, honest_verdict
  - JitRLConstraintMemory: record(), threshold(), to_dict()
  - _generate_synthetic_violations(): synthetic fallback
  - _compute_fp_rate(): baseline vs JitRL-gated FP rate
  - main(): full integration via mocked IO

Spec: REQ-LEARN-034,
      SCENARIO-LEARN-060, SCENARIO-LEARN-061 (Exp 432)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_exp432():
    """Load experiment_432 module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "experiment_432",
        _REPO_ROOT / "scripts" / "experiment_432_jitrl_live_validation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    with patch("carnot.pipeline.env_autofix.apply_env_autofix", return_value=None):
        spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# JitRLConstraintMemory tests
# ---------------------------------------------------------------------------


class TestJitRLConstraintMemory:
    """Tests for carnot.pipeline.jitrl_memory.JitRLConstraintMemory."""

    def _make(self, base=0.5, lr=0.02):
        from carnot.pipeline.jitrl_memory import JitRLConstraintMemory

        return JitRLConstraintMemory(base_threshold=base, lr=lr)

    def test_initial_threshold_is_base(self):
        # SCENARIO-LEARN-061 related: unseen domain returns base_threshold
        mem = self._make(base=0.5)
        assert mem.threshold("arithmetic") == 0.5

    def test_record_fp_raises_threshold(self):
        # SCENARIO-LEARN-060
        mem = self._make(base=0.5, lr=0.02)
        mem.record("rate_problems", violation_energy=0.6, was_fp=True)
        assert abs(mem.threshold("rate_problems") - 0.52) < 1e-9

    def test_record_fp_appends_history(self):
        # SCENARIO-LEARN-060: history has length 1 after one record
        mem = self._make()
        mem.record("rate_problems", violation_energy=0.6, was_fp=True)
        assert len(mem.history) == 1
        assert mem.history[0].domain == "rate_problems"
        assert mem.history[0].was_fp is True

    def test_record_tp_above_threshold_lowers_threshold(self):
        # True positive above threshold → lower threshold
        mem = self._make(base=0.5, lr=0.02)
        mem.record("arithmetic", violation_energy=0.7, was_fp=False)
        assert abs(mem.threshold("arithmetic") - 0.48) < 1e-9

    def test_record_tp_below_threshold_no_change(self):
        # True positive below threshold → no change (energy not > threshold)
        mem = self._make(base=0.5, lr=0.02)
        mem.record("arithmetic", violation_energy=0.3, was_fp=False)
        assert mem.threshold("arithmetic") == 0.5

    def test_threshold_clamped_at_upper(self):
        # Many FPs should not push threshold above 0.95
        mem = self._make(base=0.94, lr=0.02)
        for _ in range(10):
            mem.record("domain_a", violation_energy=0.9, was_fp=True)
        assert mem.threshold("domain_a") <= 0.95

    def test_threshold_clamped_at_lower(self):
        # Many TPs should not push threshold below 0.05
        mem = self._make(base=0.06, lr=0.02)
        for _ in range(10):
            mem.record("domain_b", violation_energy=0.9, was_fp=False)
        assert mem.threshold("domain_b") >= 0.05

    def test_independent_domains(self):
        # Each domain adapts independently
        mem = self._make(base=0.5, lr=0.02)
        mem.record("rate_problems", violation_energy=0.6, was_fp=True)
        mem.record("arithmetic", violation_energy=0.8, was_fp=False)
        assert mem.threshold("rate_problems") > 0.5
        assert mem.threshold("arithmetic") < 0.5

    def test_to_dict_keys(self):
        mem = self._make(base=0.5, lr=0.02)
        mem.record("arithmetic", violation_energy=0.7, was_fp=False)
        d = mem.to_dict()
        assert d["base_threshold"] == 0.5
        assert d["lr"] == 0.02
        assert "arithmetic" in d["thresholds"]
        assert d["n_records"] == 1

    def test_to_dict_empty(self):
        mem = self._make()
        d = mem.to_dict()
        assert d["thresholds"] == {}
        assert d["n_records"] == 0


# ---------------------------------------------------------------------------
# load_live_violations tests
# ---------------------------------------------------------------------------


class TestLoadLiveViolations:
    """Tests for load_live_violations()."""

    def test_missing_file_returns_empty(self, tmp_path):
        mod = _load_exp432()
        result = mod.load_live_violations(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json")
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert result == []

    def test_non_success_status_returns_empty(self, tmp_path):
        p = tmp_path / "exp427.json"
        p.write_text(json.dumps({"status": "scaffolding_only", "questions": []}))
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert result == []

    def test_success_status_extracts_violations(self, tmp_path):
        p = tmp_path / "exp427.json"
        data = {
            "status": "success",
            "questions": [
                {
                    "violations": [
                        {
                            "domain": "arithmetic",
                            "violation_energy": 0.6,
                            "outcome": "fixed",
                        },
                        {
                            "domain": "rate_problems",
                            "violation_energy": 0.7,
                            "outcome": "false_positive",
                        },
                    ]
                }
            ],
        }
        p.write_text(json.dumps(data))
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert len(result) == 2
        assert result[0]["domain"] == "arithmetic"
        assert result[0]["was_fp"] is False
        assert result[1]["was_fp"] is True

    def test_unknown_outcome_filtered(self, tmp_path):
        p = tmp_path / "exp427.json"
        data = {
            "status": "success",
            "questions": [
                {
                    "violations": [
                        {"domain": "arithmetic", "violation_energy": 0.5, "outcome": "unknown"},
                        {"domain": "arithmetic", "violation_energy": 0.5, "outcome": "fixed"},
                    ]
                }
            ],
        }
        p.write_text(json.dumps(data))
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert len(result) == 1

    def test_live_status_also_accepted(self, tmp_path):
        # status='live' is also a valid real-data indicator
        p = tmp_path / "exp427.json"
        data = {
            "status": "live",
            "questions": [
                {"violations": [{"domain": "arithmetic", "violation_energy": 0.4, "outcome": "not_fixed"}]}
            ],
        }
        p.write_text(json.dumps(data))
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert len(result) == 1

    def test_question_with_no_violations_key(self, tmp_path):
        p = tmp_path / "exp427.json"
        data = {"status": "success", "questions": [{"text": "no violations key"}]}
        p.write_text(json.dumps(data))
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert result == []

    def test_default_domain_used_when_missing(self, tmp_path):
        p = tmp_path / "exp427.json"
        data = {
            "status": "success",
            "questions": [
                {"violations": [{"violation_energy": 0.5, "outcome": "fixed"}]}
            ],
        }
        p.write_text(json.dumps(data))
        mod = _load_exp432()
        result = mod.load_live_violations(str(p))
        assert result[0]["domain"] == "arithmetic"


# ---------------------------------------------------------------------------
# build_jitrl_validation_artifact tests
# ---------------------------------------------------------------------------


class TestBuildJitrlValidationArtifact:
    """Tests for build_jitrl_validation_artifact()."""

    def setup_method(self):
        self.mod = _load_exp432()

    def test_schema_field(self):
        art = self.mod.build_jitrl_validation_artifact(0.3, 0.2, 50, "live")
        assert art["schema"] == "carnot.jitrl_validation.v1"

    def test_live_fp_reduction_verdict(self):
        # SCENARIO-LEARN-061: before > after → live_fp_reduction
        art = self.mod.build_jitrl_validation_artifact(0.3, 0.2, 50, "live")
        assert abs(art["fp_reduction_pct"] - 33.3333) < 0.01
        assert art["honest_verdict"] == "live_fp_reduction"

    def test_live_no_reduction_verdict(self):
        # after >= before → live_no_reduction
        art = self.mod.build_jitrl_validation_artifact(0.2, 0.3, 50, "live")
        assert art["fp_reduction_pct"] < 0
        assert art["honest_verdict"] == "live_no_reduction"

    def test_live_zero_reduction_verdict(self):
        # exactly equal → live_no_reduction (not > 0)
        art = self.mod.build_jitrl_validation_artifact(0.2, 0.2, 50, "live")
        assert art["fp_reduction_pct"] == 0.0
        assert art["honest_verdict"] == "live_no_reduction"

    def test_synthetic_fallback_verdict(self):
        # SCENARIO-LEARN-061: source='synthetic' always → synthetic_fallback
        art = self.mod.build_jitrl_validation_artifact(0.0, 0.0, 50, "synthetic")
        assert art["honest_verdict"] == "synthetic_fallback"
        assert art["fp_reduction_pct"] == 0.0

    def test_zero_before_fp_returns_zero_reduction(self):
        # before_fp == 0 → fp_reduction_pct = 0 (avoid division by zero)
        art = self.mod.build_jitrl_validation_artifact(0.0, 0.1, 50, "live")
        assert art["fp_reduction_pct"] == 0.0

    def test_n_questions_field(self):
        art = self.mod.build_jitrl_validation_artifact(0.5, 0.3, 42, "live")
        assert art["n_questions"] == 42

    def test_source_field_preserved(self):
        art = self.mod.build_jitrl_validation_artifact(0.5, 0.3, 10, "live")
        assert art["source"] == "live"


# ---------------------------------------------------------------------------
# _generate_synthetic_violations tests
# ---------------------------------------------------------------------------


class TestGenerateSyntheticViolations:
    """Tests for _generate_synthetic_violations()."""

    def setup_method(self):
        self.mod = _load_exp432()

    def test_returns_n_records(self):
        records = self.mod._generate_synthetic_violations(100)
        assert len(records) == 100

    def test_each_record_has_required_keys(self):
        records = self.mod._generate_synthetic_violations(10)
        for r in records:
            assert "domain" in r
            assert "violation_energy" in r
            assert "outcome" in r
            assert "was_fp" in r

    def test_was_fp_matches_outcome(self):
        records = self.mod._generate_synthetic_violations(50)
        for r in records:
            if r["was_fp"]:
                assert r["outcome"] == "false_positive"
            else:
                assert r["outcome"] in ("fixed", "not_fixed")

    def test_domains_are_known_values(self):
        records = self.mod._generate_synthetic_violations(50)
        domains = {r["domain"] for r in records}
        assert domains <= {"arithmetic", "rate_problems"}


# ---------------------------------------------------------------------------
# _compute_fp_rate tests
# ---------------------------------------------------------------------------


class TestComputeFpRate:
    """Tests for _compute_fp_rate()."""

    def setup_method(self):
        self.mod = _load_exp432()

    def _make_record(self, domain, energy, was_fp):
        return {"domain": domain, "violation_energy": energy, "was_fp": was_fp}

    def test_no_jitrl_all_records_fired(self):
        records = [
            self._make_record("arithmetic", 0.6, True),
            self._make_record("arithmetic", 0.4, False),
        ]
        rate = self.mod._compute_fp_rate(records, memory=None)
        assert rate == 0.5

    def test_jitrl_suppresses_low_energy_records(self):
        from carnot.pipeline.jitrl_memory import JitRLConstraintMemory

        mem = JitRLConstraintMemory(base_threshold=0.5, lr=0.02)
        records = [
            self._make_record("arithmetic", 0.3, True),   # below threshold → suppressed
            self._make_record("arithmetic", 0.7, True),   # above threshold → fired
        ]
        rate = self.mod._compute_fp_rate(records, memory=mem)
        # Only second record fires; it's a FP → rate = 1.0
        assert rate == 1.0

    def test_empty_records_returns_zero(self):
        rate = self.mod._compute_fp_rate([], memory=None)
        assert rate == 0.0

    def test_all_suppressed_returns_zero(self):
        from carnot.pipeline.jitrl_memory import JitRLConstraintMemory

        mem = JitRLConstraintMemory(base_threshold=0.9, lr=0.02)
        records = [self._make_record("arithmetic", 0.3, True)]
        rate = self.mod._compute_fp_rate(records, memory=mem)
        assert rate == 0.0

    def test_no_fp_in_fired_records_returns_zero(self):
        records = [
            self._make_record("arithmetic", 0.6, False),
            self._make_record("arithmetic", 0.7, False),
        ]
        rate = self.mod._compute_fp_rate(records, memory=None)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# main() integration tests
# ---------------------------------------------------------------------------


class TestMain:
    """Integration tests for main()."""

    def setup_method(self):
        self.mod = _load_exp432()

    def test_main_writes_json_with_correct_schema(self, tmp_path):
        output = tmp_path / "results" / "experiment_432_jitrl_live_validation.json"

        with (
            patch.object(self.mod, "_EXP_427_PATH", tmp_path / "nofile.json"),
            patch.object(self.mod, "_REPO_ROOT", tmp_path),
            patch.object(self.mod, "_OUTPUT_PATH", "results/experiment_432_jitrl_live_validation.json"),
        ):
            self.mod.main()

        data = json.loads(output.read_text())
        assert data["schema"] == "carnot.jitrl_validation.v1"
        assert data["experiment"] == 432
        assert "honest_verdict" in data
        assert data["honest_verdict"] == "synthetic_fallback"

    def test_main_uses_live_data_when_available(self, tmp_path):
        # Build a fake Exp 427 result with 100 questions each having 1 violation
        questions = []
        for i in range(100):
            questions.append(
                {
                    "violations": [
                        {
                            "domain": "arithmetic",
                            "violation_energy": 0.6,
                            "outcome": "fixed" if i % 3 != 0 else "false_positive",
                        }
                    ]
                }
            )
        exp427 = tmp_path / "exp427.json"
        exp427.write_text(json.dumps({"status": "success", "questions": questions}))

        output = tmp_path / "results" / "experiment_432_jitrl_live_validation.json"

        with (
            patch.object(self.mod, "_EXP_427_PATH", exp427),
            patch.object(self.mod, "_REPO_ROOT", tmp_path),
            patch.object(self.mod, "_OUTPUT_PATH", "results/experiment_432_jitrl_live_validation.json"),
        ):
            self.mod.main()

        data = json.loads(output.read_text())
        assert data["source"] == "live"
        assert data["honest_verdict"] in ("live_fp_reduction", "live_no_reduction")

    def test_main_synthetic_fallback_when_exp427_partial(self, tmp_path):
        exp427 = tmp_path / "exp427.json"
        exp427.write_text(json.dumps({"status": "scaffolding_only"}))

        output = tmp_path / "results" / "experiment_432_jitrl_live_validation.json"

        with (
            patch.object(self.mod, "_EXP_427_PATH", exp427),
            patch.object(self.mod, "_REPO_ROOT", tmp_path),
            patch.object(self.mod, "_OUTPUT_PATH", "results/experiment_432_jitrl_live_validation.json"),
        ):
            self.mod.main()

        data = json.loads(output.read_text())
        assert data["source"] == "synthetic"
        assert data["honest_verdict"] == "synthetic_fallback"

    def test_main_includes_jitrl_state(self, tmp_path):
        output = tmp_path / "results" / "experiment_432_jitrl_live_validation.json"

        with (
            patch.object(self.mod, "_EXP_427_PATH", tmp_path / "nofile.json"),
            patch.object(self.mod, "_REPO_ROOT", tmp_path),
            patch.object(self.mod, "_OUTPUT_PATH", "results/experiment_432_jitrl_live_validation.json"),
        ):
            self.mod.main()

        data = json.loads(output.read_text())
        assert "jitrl_state" in data
        assert "thresholds" in data["jitrl_state"]
        assert data["warmup_n"] == 50
