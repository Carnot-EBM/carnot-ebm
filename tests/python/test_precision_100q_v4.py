"""Tests for Precision100qV4Result and CoTPairCollector (Exp 476).

Every test traces to a spec requirement or scenario:
    REQ-BENCH-025, REQ-BENCH-027
    SCENARIO-BENCH-044, SCENARIO-BENCH-045, SCENARIO-BENCH-046
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.precision_100q_v4_result import CoTPairCollector, Precision100qV4Result


# ---------------------------------------------------------------------------
# Precision100qV4Result tests
# ---------------------------------------------------------------------------


class TestPrecision100qV4Result:
    """REQ-BENCH-025, SCENARIO-BENCH-044"""

    def _make(self, pre=0.60, post=0.65, n=100) -> Precision100qV4Result:
        return Precision100qV4Result(
            model_id="TestModel",
            pre_accuracy=pre,
            post_accuracy=post,
            n=n,
            extractor_used="vericot,vprm",
            inference_mode="live_gpu",
        )

    def test_signed_improvement_positive(self):
        r = self._make(pre=0.60, post=0.65)
        assert abs(r.signed_improvement - 0.05) < 1e-9

    def test_signed_improvement_negative(self):
        r = self._make(pre=0.70, post=0.65)
        assert r.signed_improvement < 0

    def test_signed_improvement_zero(self):
        r = self._make(pre=0.65, post=0.65)
        assert r.signed_improvement == 0.0

    def test_is_positive_true(self):
        r = self._make(pre=0.60, post=0.65)
        assert r.is_positive is True

    def test_is_positive_false_when_zero(self):
        r = self._make(pre=0.65, post=0.65)
        assert r.is_positive is False

    def test_is_positive_false_when_negative(self):
        r = self._make(pre=0.70, post=0.65)
        assert r.is_positive is False

    def test_ci_95_wilson_returns_tuple(self):
        # REQ-BENCH-025: Wilson CI must be a tuple of two floats
        r = self._make()
        lo, hi = r.ci_95_wilson
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_95_wilson_ordered(self):
        r = self._make()
        lo, hi = r.ci_95_wilson
        assert lo < hi

    def test_ci_95_wilson_within_unit_interval(self):
        r = self._make()
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_ci_95_wilson_width_lt_010_at_n100(self):
        # REQ-BENCH-025: at n=100 the CI width must be < 0.10
        r = self._make(post=0.05, n=100)
        lo, hi = r.ci_95_wilson
        assert (hi - lo) < 0.10

    def test_ci_95_wilson_at_extreme_zero(self):
        # Wilson CI must not produce negative probabilities at p=0
        r = self._make(pre=0.0, post=0.0, n=100)
        lo, hi = r.ci_95_wilson
        assert lo >= 0.0
        assert hi >= 0.0

    def test_ci_95_wilson_at_extreme_one(self):
        # Wilson CI must not exceed 1.0 at p=1
        r = self._make(pre=1.0, post=1.0, n=100)
        lo, hi = r.ci_95_wilson
        assert lo <= 1.0
        assert hi <= 1.0

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        assert "model_id" in d
        assert "pre_accuracy" in d
        assert "post_accuracy" in d
        assert "n" in d
        assert "extractor_used" in d
        assert "inference_mode" in d
        assert "signed_improvement" in d
        assert "ci_95_wilson" in d
        assert "is_positive" in d

    def test_to_dict_ci_95_wilson_is_list(self):
        # JSON serialisability requires a list, not a tuple
        r = self._make()
        d = r.to_dict()
        assert isinstance(d["ci_95_wilson"], list)
        assert len(d["ci_95_wilson"]) == 2

    def test_to_dict_is_json_serializable(self):
        r = self._make()
        json.dumps(r.to_dict())  # must not raise

    def test_to_dict_values_match_properties(self):
        r = self._make(pre=0.60, post=0.65, n=100)
        d = r.to_dict()
        assert d["signed_improvement"] == pytest.approx(r.signed_improvement)
        assert d["is_positive"] == r.is_positive
        lo, hi = r.ci_95_wilson
        assert d["ci_95_wilson"][0] == pytest.approx(lo)
        assert d["ci_95_wilson"][1] == pytest.approx(hi)

    def test_n_guards_against_zero(self):
        # n=0 should not raise ZeroDivisionError (guarded by max(n,1))
        r = Precision100qV4Result(
            model_id="m", pre_accuracy=0.5, post_accuracy=0.5,
            n=0, extractor_used="none", inference_mode="synthetic",
        )
        lo, hi = r.ci_95_wilson  # must not raise
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# CoTPairCollector tests
# ---------------------------------------------------------------------------


class TestCoTPairCollector:
    """REQ-BENCH-027, SCENARIO-BENCH-046"""

    def test_flush_writes_valid_json(self, tmp_path):
        out = str(tmp_path / "pairs.json")
        col = CoTPairCollector(output_path=out)
        col.add("Model", "q1", "cot1", True)
        col.add("Model", "q2", "cot2", False)
        n = col.flush()
        assert n == 2
        data = json.loads(Path(out).read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_flush_returns_pair_count(self, tmp_path):
        out = str(tmp_path / "pairs.json")
        col = CoTPairCollector(output_path=out)
        for i in range(5):
            col.add("M", f"q{i}", f"cot{i}", i % 2 == 0)
        assert col.flush() == 5

    def test_flush_zero_pairs(self, tmp_path):
        out = str(tmp_path / "empty.json")
        col = CoTPairCollector(output_path=out)
        n = col.flush()
        assert n == 0
        data = json.loads(Path(out).read_text())
        assert data == []

    def test_flush_pair_fields(self, tmp_path):
        out = str(tmp_path / "pairs.json")
        col = CoTPairCollector(output_path=out)
        col.add("Gemma4-E4B-it", "What is 2+2?", "2+2=4. #### 4", True)
        col.flush()
        data = json.loads(Path(out).read_text())
        pair = data[0]
        assert pair["model"] == "Gemma4-E4B-it"
        assert pair["question"] == "What is 2+2?"
        assert pair["cot_text"] == "2+2=4. #### 4"
        assert pair["correct"] is True

    def test_flush_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "deep" / "nested" / "pairs.json")
        col = CoTPairCollector(output_path=out)
        col.add("M", "q", "cot", False)
        col.flush()
        assert Path(out).exists()

    def test_flush_atomic_tmp_removed(self, tmp_path):
        out = str(tmp_path / "pairs.json")
        col = CoTPairCollector(output_path=out)
        col.add("M", "q", "c", True)
        col.flush()
        # After flush, .tmp file must be gone (replaced)
        assert not Path(str(out) + ".tmp").exists()

    def test_add_does_not_write_immediately(self, tmp_path):
        out = str(tmp_path / "pairs.json")
        col = CoTPairCollector(output_path=out)
        col.add("M", "q", "c", True)
        assert not Path(out).exists()
