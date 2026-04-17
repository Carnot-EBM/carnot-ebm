"""Tests for LongRunBenchmarkExecutor, LongRunBenchmarkResult, BenchmarkBatch, get_batch_size.

Spec: REQ-INFRA-027, REQ-INFRA-028,
      SCENARIO-INFRA-034, SCENARIO-INFRA-035, SCENARIO-INFRA-036
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from carnot.pipeline.long_run_executor import (
    BenchmarkBatch,
    LongRunBenchmarkExecutor,
    LongRunBenchmarkResult,
    get_batch_size,
)


# ---------------------------------------------------------------------------
# get_batch_size — REQ-INFRA-028, SCENARIO-INFRA-036
# ---------------------------------------------------------------------------


def test_get_batch_size_default(monkeypatch):
    """SCENARIO-INFRA-036: default batch size is 50 when env var absent."""
    monkeypatch.delenv("CARNOT_BENCH_BATCH_SIZE", raising=False)
    assert get_batch_size() == 50


def test_get_batch_size_from_env(monkeypatch):
    """SCENARIO-INFRA-036: env var overrides default."""
    monkeypatch.setenv("CARNOT_BENCH_BATCH_SIZE", "25")
    assert get_batch_size() == 25


def test_get_batch_size_empty_env(monkeypatch):
    """SCENARIO-INFRA-036: empty env var falls back to default 50."""
    monkeypatch.setenv("CARNOT_BENCH_BATCH_SIZE", "")
    assert get_batch_size() == 50


# ---------------------------------------------------------------------------
# BenchmarkBatch dataclass
# ---------------------------------------------------------------------------


def test_benchmark_batch_defaults():
    """BenchmarkBatch default status is 'pending' and results is None."""
    batch = BenchmarkBatch(batch_id=0, start_idx=0, end_idx=5, questions=list(range(5)))
    assert batch.status == "pending"
    assert batch.results is None


def test_benchmark_batch_explicit_fields():
    """BenchmarkBatch stores all explicit fields correctly."""
    batch = BenchmarkBatch(
        batch_id=2,
        start_idx=100,
        end_idx=120,
        questions=list(range(20)),
        results=list(range(20)),
        status="complete",
    )
    assert batch.batch_id == 2
    assert batch.start_idx == 100
    assert batch.end_idx == 120
    assert len(batch.questions) == 20
    assert len(batch.results) == 20
    assert batch.status == "complete"


# ---------------------------------------------------------------------------
# LongRunBenchmarkResult dataclass
# ---------------------------------------------------------------------------


def test_long_run_benchmark_result_fields():
    """LongRunBenchmarkResult stores all fields."""
    result = LongRunBenchmarkResult(
        total_questions=120,
        batch_size=50,
        n_batches=3,
        completed_batches=2,
        all_results=list(range(100)),
        honest_verdict="partial_2_of_3",
    )
    assert result.total_questions == 120
    assert result.batch_size == 50
    assert result.n_batches == 3
    assert result.completed_batches == 2
    assert len(result.all_results) == 100
    assert result.honest_verdict == "partial_2_of_3"


# ---------------------------------------------------------------------------
# partition — SCENARIO-INFRA-034
# ---------------------------------------------------------------------------


def test_partition_120_questions_into_3_batches():
    """SCENARIO-INFRA-034: 120 questions, batch_size=50 → [50, 50, 20]."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    questions = list(range(120))
    batches = executor.partition(questions)

    assert len(batches) == 3

    assert batches[0].batch_id == 0
    assert batches[0].start_idx == 0
    assert batches[0].end_idx == 50
    assert len(batches[0].questions) == 50

    assert batches[1].batch_id == 1
    assert batches[1].start_idx == 50
    assert batches[1].end_idx == 100
    assert len(batches[1].questions) == 50

    assert batches[2].batch_id == 2
    assert batches[2].start_idx == 100
    assert batches[2].end_idx == 120
    assert len(batches[2].questions) == 20


def test_partition_all_status_pending():
    """partition() returns batches all with status='pending'."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    batches = executor.partition(list(range(120)))
    for b in batches:
        assert b.status == "pending"
        assert b.results is None


def test_partition_exact_multiple():
    """partition() with len(questions) == batch_size produces 1 batch."""
    executor = LongRunBenchmarkExecutor(batch_size=10)
    batches = executor.partition(list(range(10)))
    assert len(batches) == 1
    assert batches[0].start_idx == 0
    assert batches[0].end_idx == 10


def test_partition_empty_list():
    """partition() of empty list returns empty list."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    batches = executor.partition([])
    assert batches == []


def test_partition_questions_are_slices():
    """partition() assigns correct question slices to each batch."""
    questions = [f"q{i}" for i in range(7)]
    executor = LongRunBenchmarkExecutor(batch_size=3)
    batches = executor.partition(questions)
    assert batches[0].questions == ["q0", "q1", "q2"]
    assert batches[1].questions == ["q3", "q4", "q5"]
    assert batches[2].questions == ["q6"]


# ---------------------------------------------------------------------------
# save_batch / load_batch — REQ-INFRA-027
# ---------------------------------------------------------------------------


def test_save_and_load_batch_roundtrip():
    """save_batch / load_batch round-trip preserves all fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=tmpdir)
        batch = BenchmarkBatch(
            batch_id=1,
            start_idx=50,
            end_idx=100,
            questions=list(range(50)),
            results=list(range(50, 100)),
            status="complete",
        )
        path = executor.save_batch(batch, prefix="test")
        assert os.path.exists(path)

        loaded = executor.load_batch(path)
        assert loaded is not None
        assert loaded.batch_id == 1
        assert loaded.start_idx == 50
        assert loaded.end_idx == 100
        assert loaded.questions == list(range(50))
        assert loaded.results == list(range(50, 100))
        assert loaded.status == "complete"


def test_load_batch_missing_file():
    """load_batch returns None for a non-existent path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=tmpdir)
        result = executor.load_batch(os.path.join(tmpdir, "nonexistent.json"))
        assert result is None


def test_load_batch_corrupt_file():
    """load_batch returns None for a corrupt JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.json")
        with open(path, "w") as f:
            f.write("NOT VALID JSON {{{")
        executor = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=tmpdir)
        result = executor.load_batch(path)
        assert result is None


def test_save_batch_creates_checkpoint_dir():
    """save_batch creates checkpoint_dir if it does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "nested", "ckpt")
        executor = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=ckpt_dir)
        batch = BenchmarkBatch(
            batch_id=0, start_idx=0, end_idx=2, questions=["a", "b"],
            results=["ra", "rb"], status="complete",
        )
        path = executor.save_batch(batch, prefix="x")
        assert os.path.exists(path)


def test_save_batch_filename_includes_batch_id():
    """save_batch filename encodes the batch_id for easy glob discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = LongRunBenchmarkExecutor(batch_size=50, checkpoint_dir=tmpdir)
        batch = BenchmarkBatch(batch_id=7, start_idx=0, end_idx=1, questions=["q"])
        batch.results = ["r"]
        batch.status = "complete"
        path = executor.save_batch(batch, prefix="exp437")
        assert "batch_0007" in path


# ---------------------------------------------------------------------------
# run_batch — REQ-INFRA-027
# ---------------------------------------------------------------------------


def test_run_batch_success():
    """run_batch runs all questions and sets status='complete'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = LongRunBenchmarkExecutor(batch_size=5, checkpoint_dir=tmpdir)
        batch = BenchmarkBatch(
            batch_id=0, start_idx=0, end_idx=5, questions=list(range(5))
        )

        def inference_fn(q):
            return q * 10

        completed = executor.run_batch(batch, inference_fn, watchdog_timeout_minutes=1)
        assert completed.status == "complete"
        assert completed.results == [0, 10, 20, 30, 40]


def test_run_batch_inference_fn_receives_questions():
    """run_batch passes each question to inference_fn in order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = LongRunBenchmarkExecutor(batch_size=3, checkpoint_dir=tmpdir)
        received = []

        def capture(q):
            received.append(q)
            return f"ans_{q}"

        batch = BenchmarkBatch(
            batch_id=0, start_idx=0, end_idx=3, questions=["a", "b", "c"]
        )
        executor.run_batch(batch, capture, watchdog_timeout_minutes=1)
        assert received == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# assemble — SCENARIO-INFRA-035, SCENARIO-INFRA-036
# ---------------------------------------------------------------------------


def test_assemble_all_complete():
    """SCENARIO-INFRA-036: all batches complete → honest_verdict='complete'."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    batches = [
        BenchmarkBatch(0, 0, 50, list(range(50)), results=list(range(50)), status="complete"),
        BenchmarkBatch(1, 50, 100, list(range(50)), results=list(range(50)), status="complete"),
        BenchmarkBatch(2, 100, 120, list(range(20)), results=list(range(20)), status="complete"),
    ]
    result = executor.assemble(batches)
    assert result.honest_verdict == "complete"
    assert result.completed_batches == 3
    assert result.n_batches == 3
    assert len(result.all_results) == 120


def test_assemble_partial_1_of_3():
    """SCENARIO-INFRA-035: only batch 0 complete → 'partial_1_of_3'."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    batches = [
        BenchmarkBatch(0, 0, 50, list(range(50)), results=list(range(50)), status="complete"),
        BenchmarkBatch(1, 50, 100, list(range(50)), results=None, status="pending"),
        BenchmarkBatch(2, 100, 120, list(range(20)), results=None, status="pending"),
    ]
    result = executor.assemble(batches)
    assert result.honest_verdict == "partial_1_of_3"
    assert result.completed_batches == 1
    assert result.n_batches == 3
    assert len(result.all_results) == 50


def test_assemble_partial_2_of_3():
    """SCENARIO-INFRA-035: 2 of 3 batches complete → 'partial_2_of_3'."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    batches = [
        BenchmarkBatch(0, 0, 50, list(range(50)), results=list(range(50)), status="complete"),
        BenchmarkBatch(1, 50, 100, list(range(50)), results=list(range(50)), status="complete"),
        BenchmarkBatch(2, 100, 120, list(range(20)), results=None, status="timed_out"),
    ]
    result = executor.assemble(batches)
    assert result.honest_verdict == "partial_2_of_3"
    assert result.completed_batches == 2


def test_assemble_empty_batches():
    """assemble of empty list produces complete verdict with zero questions."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    result = executor.assemble([])
    assert result.honest_verdict == "complete"
    assert result.n_batches == 0
    assert result.completed_batches == 0
    assert result.total_questions == 0
    assert result.all_results == []


def test_assemble_total_questions():
    """assemble correctly sums total_questions across all batches."""
    executor = LongRunBenchmarkExecutor(batch_size=50)
    batches = [
        BenchmarkBatch(0, 0, 50, list(range(50)), results=list(range(50)), status="complete"),
        BenchmarkBatch(1, 50, 70, list(range(20)), results=None, status="pending"),
    ]
    result = executor.assemble(batches)
    assert result.total_questions == 70


def test_assemble_preserves_batch_size():
    """assemble carries the executor's batch_size into the result."""
    executor = LongRunBenchmarkExecutor(batch_size=25)
    batches = [
        BenchmarkBatch(0, 0, 25, list(range(25)), results=list(range(25)), status="complete"),
    ]
    result = executor.assemble(batches)
    assert result.batch_size == 25


# ---------------------------------------------------------------------------
# Export from carnot.pipeline
# ---------------------------------------------------------------------------


def test_pipeline_init_exports():
    """All four symbols are exported from carnot.pipeline."""
    import carnot.pipeline as p

    assert hasattr(p, "BenchmarkBatch")
    assert hasattr(p, "LongRunBenchmarkExecutor")
    assert hasattr(p, "LongRunBenchmarkResult")
    assert hasattr(p, "get_batch_size")
