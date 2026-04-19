"""Tests for Live200qV2Result and CoTPairCollector (Exp 478).

Spec: REQ-BENCH-028, REQ-BENCH-029, REQ-BENCH-030,
      SCENARIO-BENCH-047, SCENARIO-BENCH-048, SCENARIO-BENCH-049
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.cot_pair_collector import CoTPairCollector
from carnot.pipeline.live_200q_v2_result import Live200qV2Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(pre=0.70, post=0.75, n=200, extractor="VeriCoT+VPRM+CRANE", mode="live_gpu"):
    return Live200qV2Result(
        model_id="TestModel",
        pre_acc=pre,
        post_acc=post,
        n=n,
        extractor_name=extractor,
        inference_mode=mode,
    )


# ---------------------------------------------------------------------------
# signed_improvement
# ---------------------------------------------------------------------------


class TestSignedImprovement:
    """signed_improvement = post - pre, unclamped."""

    def test_positive(self):
        r = _make(pre=0.70, post=0.75)
        assert abs(r.signed_improvement - 0.05) < 1e-9

    def test_negative(self):
        # Pipeline degraded accuracy — must NOT clamp to 0
        r = _make(pre=0.80, post=0.70)
        assert abs(r.signed_improvement - (-0.10)) < 1e-9

    def test_zero(self):
        r = _make(pre=0.70, post=0.70)
        assert r.signed_improvement == 0.0


# ---------------------------------------------------------------------------
# ci_95_wilson — Wilson score interval on post_acc
# ---------------------------------------------------------------------------


class TestCI95Wilson:
    """REQ-BENCH-030, SCENARIO-BENCH-047: Wilson CI at n=200 has width < 0.07 near p=0.05."""

    def test_width_lt_007_at_n200_p005(self):
        # SCENARIO-BENCH-047: at p=0.05, n=200, full width < 0.07
        r = _make(post=0.05, n=200)
        lo, hi = r.ci_95_wilson
        width = hi - lo
        assert width < 0.07, f"CI width={width:.4f} >= 0.07 at post=0.05, n=200"

    def test_within_zero_one(self):
        for post in (0.0, 0.05, 0.5, 0.95, 1.0):
            r = _make(post=post, n=200)
            lo, hi = r.ci_95_wilson
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0
            assert lo <= hi

    def test_narrows_with_more_samples(self):
        r100 = _make(post=0.5, n=100)
        r200 = _make(post=0.5, n=200)
        lo100, hi100 = r100.ci_95_wilson
        lo200, hi200 = r200.ci_95_wilson
        assert (hi200 - lo200) < (hi100 - lo100)

    def test_at_post_zero(self):
        r = _make(post=0.0, n=200)
        lo, hi = r.ci_95_wilson
        assert lo < 1e-9
        assert hi > 0.0

    def test_at_post_one(self):
        r = _make(post=1.0, n=200)
        lo, hi = r.ci_95_wilson
        assert hi == 1.0
        assert lo < 1.0


# ---------------------------------------------------------------------------
# is_statistically_positive — Wald CI on the improvement
# ---------------------------------------------------------------------------


class TestIsStatisticallyPositive:
    """REQ-BENCH-030, SCENARIO-BENCH-049: uses Wald CI for the improvement delta."""

    def test_false_for_small_improvement_at_n200(self):
        # SCENARIO-BENCH-049: improvement=0.02 at n=200 is NOT significant
        # pre=0.70, post=0.72 → SE ≈ 0.045 → lower ≈ 0.02 - 0.089 = -0.069 < 0
        r = _make(pre=0.70, post=0.72, n=200)
        assert r.signed_improvement == pytest.approx(0.02, abs=1e-9)
        assert r.is_statistically_positive is False

    def test_true_for_large_improvement(self):
        # A 20pp improvement at n=200 should be clearly significant
        # pre=0.50, post=0.70 → SE = sqrt(0.25/200 + 0.21/200) ≈ 0.048
        # lower = 0.20 - 1.96*0.048 = 0.20 - 0.094 = 0.106 > 0 → True
        r = _make(pre=0.50, post=0.70, n=200)
        assert r.is_statistically_positive is True

    def test_false_when_improvement_is_zero(self):
        r = _make(pre=0.70, post=0.70, n=200)
        assert r.is_statistically_positive is False

    def test_false_when_improvement_is_negative(self):
        r = _make(pre=0.70, post=0.60, n=200)
        assert r.is_statistically_positive is False

    def test_consistent_with_wald_formula(self):
        import math
        z95 = 1.959963984540054
        r = _make(pre=0.60, post=0.80, n=200)
        se = math.sqrt(0.60 * 0.40 / 200 + 0.80 * 0.20 / 200)
        expected_lower = 0.20 - z95 * se
        assert r.is_statistically_positive == (expected_lower > 0.0)


# ---------------------------------------------------------------------------
# cot_pairs_file
# ---------------------------------------------------------------------------


class TestCotPairsFile:
    """REQ-BENCH-028: cot_pairs_file field stores path, not the pairs themselves."""

    def test_default_none(self):
        r = _make()
        assert r.cot_pairs_file is None

    def test_custom_path(self):
        r = Live200qV2Result(
            model_id="M",
            pre_acc=0.5,
            post_acc=0.6,
            n=200,
            extractor_name="VeriCoT+VPRM+CRANE",
            inference_mode="live_gpu",
            cot_pairs_file="results/exp478_cot_pairs.json",
        )
        assert r.cot_pairs_file == "results/exp478_cot_pairs.json"


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    """Serialization completeness for Live200qV2Result."""

    def test_required_keys_present(self):
        r = _make()
        d = r.to_dict()
        expected = {
            "model_id", "pre_acc", "post_acc", "n", "extractor_name",
            "inference_mode", "signed_improvement", "ci_95_wilson",
            "is_statistically_positive", "cot_pairs_file",
        }
        assert expected.issubset(d.keys())

    def test_ci_95_wilson_is_list_of_two_floats(self):
        r = _make()
        d = r.to_dict()
        assert isinstance(d["ci_95_wilson"], list)
        assert len(d["ci_95_wilson"]) == 2
        lo, hi = d["ci_95_wilson"]
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_cot_pairs_file_in_dict(self):
        r = _make()
        d = r.to_dict()
        assert "cot_pairs_file" in d
        assert d["cot_pairs_file"] is None

    def test_signed_improvement_in_dict(self):
        r = _make(pre=0.60, post=0.70)
        d = r.to_dict()
        assert abs(d["signed_improvement"] - 0.10) < 1e-9


# ---------------------------------------------------------------------------
# CoTPairCollector
# ---------------------------------------------------------------------------


class TestCoTPairCollector:
    """SCENARIO-BENCH-046: CoTPairCollector flushes pairs atomically."""

    def test_empty_flush_writes_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pairs.json")
            c = CoTPairCollector(path)
            n = c.flush()
            assert n == 0
            data = json.loads(Path(path).read_text())
            assert data == []

    def test_flush_writes_correct_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pairs.json")
            c = CoTPairCollector(path)
            c.add("Gemma4", "Q1", "Step 1 ... #### 5", True)
            c.add("Qwen", "Q2", "Step 1 ... #### 10", False)
            n = c.flush()
            assert n == 2
            data = json.loads(Path(path).read_text())
            assert len(data) == 2
            assert data[0]["model"] == "Gemma4"
            assert data[0]["question"] == "Q1"
            assert data[0]["cot_text"] == "Step 1 ... #### 5"
            assert data[0]["correct"] is True
            assert data[1]["correct"] is False

    def test_flush_returns_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pairs.json")
            c = CoTPairCollector(path)
            for i in range(5):
                c.add("M", f"Q{i}", f"cot{i}", True)
            assert c.flush() == 5

    def test_len_before_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pairs.json")
            c = CoTPairCollector(path)
            assert len(c) == 0
            c.add("M", "Q", "cot", True)
            assert len(c) == 1

    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dir" / "pairs.json")
            c = CoTPairCollector(path)
            c.add("M", "Q", "cot", True)
            c.flush()
            assert Path(path).exists()

    def test_atomic_write_no_tmp_left(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pairs.json")
            c = CoTPairCollector(path)
            c.add("M", "Q", "cot", True)
            c.flush()
            tmp = str(Path(path).with_suffix(".tmp"))
            assert not Path(tmp).exists()
            assert Path(path).exists()

    def test_flush_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "pairs.json")
            c = CoTPairCollector(path)
            c.add("M", "Q", "cot", True)
            c.flush()
            data = json.loads(Path(path).read_text())
            assert isinstance(data, list)
