"""Tests for ThinkProbeV2 and ThinkProbeV2Result.

100% coverage target.

Spec: REQ-PROBE-005, REQ-PROBE-006, REQ-PROBE-007
SCENARIO-PROBE-010, SCENARIO-PROBE-011, SCENARIO-PROBE-012
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.think_probe_v2 import ThinkProbeV2, ThinkProbeV2Result


# ---------------------------------------------------------------------------
# ThinkProbeV2Result unit tests
# ---------------------------------------------------------------------------


class TestThinkProbeV2ResultProperties:
    """Test all properties of ThinkProbeV2Result (REQ-PROBE-006)."""

    def test_is_partial_true_when_fewer_completed(self):
        # REQ-PROBE-006: is_partial is True when n_completed < n_total
        r = ThinkProbeV2Result(n_completed=30, n_total=50, results=[], status="partial")
        assert r.is_partial is True

    def test_is_partial_false_when_complete(self):
        # REQ-PROBE-006: is_partial is False when n_completed == n_total
        r = ThinkProbeV2Result(n_completed=50, n_total=50, results=[], status="complete")
        assert r.is_partial is False

    def test_is_partial_false_when_both_zero(self):
        # Edge case: n_completed == n_total == 0 means the run was given no questions
        r = ThinkProbeV2Result(n_completed=0, n_total=0, results=[], status="complete")
        assert r.is_partial is False

    def test_completion_fraction_full_run(self):
        r = ThinkProbeV2Result(n_completed=50, n_total=50, results=[], status="complete")
        assert r.completion_fraction == 1.0

    def test_completion_fraction_partial(self):
        r = ThinkProbeV2Result(n_completed=30, n_total=50, results=[], status="partial")
        assert r.completion_fraction == pytest.approx(0.6)

    def test_completion_fraction_zero_when_none_completed(self):
        r = ThinkProbeV2Result(n_completed=0, n_total=50, results=[], status="empty")
        assert r.completion_fraction == 0.0

    def test_completion_fraction_zero_total(self):
        # n_total == 0: guard against division by zero
        r = ThinkProbeV2Result(n_completed=0, n_total=0, results=[], status="complete")
        assert r.completion_fraction == 0.0

    def test_honest_verdict_complete(self):
        # REQ-PROBE-006: 'complete' when n_completed == n_total
        r = ThinkProbeV2Result(n_completed=50, n_total=50, results=[], status="complete")
        assert r.honest_verdict == "complete"

    def test_honest_verdict_partial_30_of_50(self):
        # REQ-PROBE-006: 'partial_N_of_M' when partial
        r = ThinkProbeV2Result(n_completed=30, n_total=50, results=[], status="partial")
        assert r.honest_verdict == "partial_30_of_50"

    def test_honest_verdict_timeout_no_data(self):
        # REQ-PROBE-006: 'timeout_no_data' when n_completed == 0 but n_total > 0
        r = ThinkProbeV2Result(n_completed=0, n_total=50, results=[], status="empty")
        assert r.honest_verdict == "timeout_no_data"

    def test_honest_verdict_partial_1_of_3(self):
        r = ThinkProbeV2Result(n_completed=1, n_total=3, results=[], status="partial")
        assert r.honest_verdict == "partial_1_of_3"

    def test_status_field_defaults_to_complete(self):
        r = ThinkProbeV2Result(n_completed=5, n_total=5, results=[])
        assert r.status == "complete"


# ---------------------------------------------------------------------------
# ThinkProbeV2.run() — complete run
# ---------------------------------------------------------------------------


class TestThinkProbeV2RunComplete:
    """Test complete (no-timeout) run behaviour (REQ-PROBE-005, SCENARIO-PROBE-010)."""

    def _fast_inference(self, question: str) -> str:
        return f"answer_for_{question}"

    def test_complete_run_50_questions(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        questions = [f"q{i}" for i in range(50)]
        result = probe.run(questions, self._fast_inference)

        assert result.n_completed == 50
        assert result.n_total == 50
        assert result.status == "complete"
        assert result.honest_verdict == "complete"
        assert result.is_partial is False
        assert result.completion_fraction == 1.0

    def test_results_contain_all_questions(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        questions = [f"q{i}" for i in range(5)]
        result = probe.run(questions, self._fast_inference)

        assert len(result.results) == 5
        for i, entry in enumerate(result.results):
            assert entry["question_index"] == i
            assert entry["question"] == f"q{i}"
            assert entry["response"] == f"answer_for_q{i}"

    def test_empty_question_list(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        result = probe.run([], self._fast_inference)

        assert result.n_completed == 0
        assert result.n_total == 0
        assert result.status == "complete"
        assert result.completion_fraction == 0.0

    def test_single_question(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        result = probe.run(["only question"], self._fast_inference)
        assert result.n_completed == 1
        assert result.n_total == 1
        assert result.status == "complete"


# ---------------------------------------------------------------------------
# ThinkProbeV2.run() — partial / timeout
# ---------------------------------------------------------------------------


class TestThinkProbeV2RunPartial:
    """Test partial-verdict behaviour on budget expiry (REQ-PROBE-006, SCENARIO-PROBE-011)."""

    def test_budget_expired_before_start_emits_partial(self, tmp_path):
        # SCENARIO-PROBE-011: sub-second budget with slow inference → partial result
        call_count = {"n": 0}

        def slow_inference(q: str) -> str:
            call_count["n"] += 1
            time.sleep(0.5)
            return "answer"

        # budget_minutes = 0.001 → 0.06 s; slow_inference takes 0.5 s per question
        probe = ThinkProbeV2(
            budget_minutes=0.001, checkpoint_interval=10, checkpoint_dir=tmp_path
        )
        questions = [f"q{i}" for i in range(10)]
        result = probe.run(questions, slow_inference)

        # The run should NOT raise
        assert isinstance(result, ThinkProbeV2Result)
        # With a 0.06 s budget and 0.5 s per question, at most 1 question can complete
        assert result.n_completed < 10
        assert result.is_partial is True

    def test_partial_result_status_set_correctly(self, tmp_path):
        # When some but not all questions complete, status should be 'partial'
        completed = {"n": 0}

        def controlled_inference(q: str) -> str:
            completed["n"] += 1
            return "ans"

        # Give enough budget for the fast mock but set n_total to 100
        probe = ThinkProbeV2(
            budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path
        )

        # Simulate partial by exhausting budget at the loop level
        # We monkeypatch time.monotonic to expire budget after 3 questions
        real_monotonic = time.monotonic
        call_seq = [0.0, 0.0, 0.1, 0.2, 9999.0]  # 5th call returns past budget
        seq_iter = iter(call_seq)

        def fake_monotonic():
            try:
                return next(seq_iter)
            except StopIteration:
                return 9999.0

        with patch("carnot.pipeline.think_probe_v2.time.monotonic", side_effect=fake_monotonic):
            probe2 = ThinkProbeV2(
                budget_minutes=0.001,  # 0.06 s budget
                checkpoint_interval=10,
                checkpoint_dir=tmp_path,
            )
            questions = [f"q{i}" for i in range(20)]
            result = probe2.run(questions, lambda q: "ans")

        assert result.is_partial is True
        assert result.status in ("partial", "empty")

    def test_timeout_no_data_when_nothing_completes(self, tmp_path):
        # When budget expires before any question returns, honest_verdict='timeout_no_data'
        def instant_inference(q: str) -> str:
            return "ans"

        # Simulate: budget already expired on first elapsed check
        call_count = [0]

        def fake_monotonic():
            call_count[0] += 1
            # First call returns t_start, second call returns t_start + budget + 1
            if call_count[0] <= 1:
                return 0.0
            return 9999.0

        with patch("carnot.pipeline.think_probe_v2.time.monotonic", side_effect=fake_monotonic):
            probe = ThinkProbeV2(
                budget_minutes=0.001,
                checkpoint_interval=10,
                checkpoint_dir=tmp_path,
            )
            result = probe.run(["q0", "q1", "q2"], instant_inference)

        assert result.n_completed == 0
        assert result.honest_verdict == "timeout_no_data"
        assert result.status == "empty"


# ---------------------------------------------------------------------------
# ThinkProbeV2._checkpoint() — incremental checkpointing
# ---------------------------------------------------------------------------


class TestThinkProbeV2Checkpoint:
    """Test checkpoint writes (REQ-PROBE-007, SCENARIO-PROBE-012)."""

    def test_checkpoint_written_every_10_questions(self, tmp_path):
        # SCENARIO-PROBE-012: checkpoint called at step 10, 20, 30 for 30 questions
        checkpoint_steps = []
        original_checkpoint = ThinkProbeV2._checkpoint

        def recording_checkpoint(self_ref, results_so_far, step):
            checkpoint_steps.append(step)
            original_checkpoint(self_ref, results_so_far, step)

        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)

        with patch.object(ThinkProbeV2, "_checkpoint", recording_checkpoint):
            probe.run([f"q{i}" for i in range(30)], lambda q: "ans")

        assert checkpoint_steps == [10, 20, 30]

    def test_checkpoint_file_exists_after_run(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=5, checkpoint_dir=tmp_path)
        probe.run([f"q{i}" for i in range(10)], lambda q: "ans")

        ckpt = tmp_path / "checkpoint.json"
        assert ckpt.exists()

    def test_checkpoint_file_contains_results(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=5, checkpoint_dir=tmp_path)
        probe.run([f"q{i}" for i in range(10)], lambda q: f"ans_{q}")

        ckpt = tmp_path / "checkpoint.json"
        data = json.loads(ckpt.read_text())
        assert data["step"] == 10
        assert data["n_completed"] == 10
        assert len(data["results"]) == 10

    def test_checkpoint_not_written_when_interval_not_reached(self, tmp_path):
        # 3 questions with interval=10 → no checkpoint written
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        probe.run([f"q{i}" for i in range(3)], lambda q: "ans")

        ckpt = tmp_path / "checkpoint.json"
        assert not ckpt.exists()

    def test_checkpoint_creates_directory_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        probe = ThinkProbeV2(
            budget_minutes=60, checkpoint_interval=5, checkpoint_dir=nested
        )
        probe.run([f"q{i}" for i in range(10)], lambda q: "ans")

        assert (nested / "checkpoint.json").exists()

    def test_checkpoint_direct_call(self, tmp_path):
        # Direct unit test of _checkpoint (covers the atomic write path)
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        results = [{"question_index": i, "question": f"q{i}", "response": "a"} for i in range(5)]
        probe._checkpoint(results, step=5)

        ckpt = tmp_path / "checkpoint.json"
        assert ckpt.exists()
        data = json.loads(ckpt.read_text())
        assert data["step"] == 5
        assert data["n_completed"] == 5
        assert "saved_at" in data


# ---------------------------------------------------------------------------
# ThinkProbeV2._run_one() — per-question timeout
# ---------------------------------------------------------------------------


class TestThinkProbeV2RunOne:
    """Test the per-question timeout wrapper."""

    def test_fast_inference_returns_response(self, tmp_path):
        probe = ThinkProbeV2(checkpoint_dir=tmp_path)
        result = probe._run_one("question", lambda q: "response", timeout_s=10.0)
        assert result == "response"

    def test_timeout_returns_empty_string(self, tmp_path):
        def slow(q: str) -> str:
            time.sleep(5)
            return "never"

        probe = ThinkProbeV2(checkpoint_dir=tmp_path)
        result = probe._run_one("q", slow, timeout_s=0.05)
        assert result == ""

    def test_exception_returns_empty_string(self, tmp_path):
        def raises(q: str) -> str:
            raise ValueError("boom")

        probe = ThinkProbeV2(checkpoint_dir=tmp_path)
        result = probe._run_one("q", raises, timeout_s=10.0)
        assert result == ""


# ---------------------------------------------------------------------------
# ThinkProbeV2 default checkpoint_dir behaviour
# ---------------------------------------------------------------------------


class TestThinkProbeV2DefaultCheckpointDir:
    """Test that default checkpoint_dir works (covers _DEFAULT_CKPT_DIR path)."""

    def test_default_dir_attribute_set(self):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10)
        # Default dir should end with 'experiment_455'
        assert probe._ckpt_dir.name == "experiment_455"

    def test_custom_dir_respected(self, tmp_path):
        probe = ThinkProbeV2(budget_minutes=60, checkpoint_interval=10, checkpoint_dir=tmp_path)
        assert probe._ckpt_dir == tmp_path


# ---------------------------------------------------------------------------
# ThinkProbeV2 result 'partial' honest_verdict format
# ---------------------------------------------------------------------------


class TestHonestVerdictFormat:
    def test_partial_verdict_uses_n_of_m_format(self):
        for n, m in [(1, 10), (7, 20), (49, 50)]:
            r = ThinkProbeV2Result(n_completed=n, n_total=m, results=[], status="partial")
            assert r.honest_verdict == f"partial_{n}_of_{m}"
