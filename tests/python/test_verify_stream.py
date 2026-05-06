"""Tests for streaming verification over candidate pools.

Spec coverage: REQ-VERIFY-1411, REQ-VERIFY-1412, REQ-VERIFY-1413,
               SCENARIO-VERIFY-1411
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from carnot.pipeline.verdict_record import VerdictRecord
from carnot.pipeline.verify_stream import collect_verify_stream, verify_stream


def _record(energy: float) -> VerdictRecord:
    verdict = "pass" if energy == 0.0 else "fail"
    return VerdictRecord(
        verdict=verdict,
        energy=energy,
        calibrated_confidence=1.0 if verdict == "pass" else 0.0,
        producing_tier=3,
        tier_reached=3,
        rationale="synthetic verifier",
        budget_ms_consumed=1.0,
    )


class SyntheticSyncPipeline:
    """Small synchronous verifier used to prove thread-offloaded stream behavior."""

    def __init__(self, energies: dict[str, float], delays: dict[str, float] | None = None) -> None:
        self.energies = energies
        self.delays = delays or {}
        self.scored: list[str] = []

    def verify_record(
        self, question: str, response: str, domain: str | None = None
    ) -> VerdictRecord:
        del question, domain
        time.sleep(self.delays.get(response, 0.0))
        self.scored.append(response)
        return _record(self.energies[response])


class SyntheticAsyncPipeline:
    """Async verifier used to assert pending workers are cancelled cleanly."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancelled: list[str] = []

    async def verify_record(
        self,
        question: str,
        response: str,
        domain: str | None = None,
    ) -> VerdictRecord:
        del question, domain
        self.started.append(response)
        try:
            await asyncio.sleep(0.01 if response == "fast" else 60.0)
        except asyncio.CancelledError:
            self.cancelled.append(response)
            raise
        return _record(0.0 if response == "fast" else 2.0)


async def _collect_records(stream: Any) -> list[VerdictRecord]:
    records: list[VerdictRecord] = []
    async for record in stream:
        records.append(record)
    return records


def test_verify_stream_emits_in_completion_order() -> None:
    """REQ-VERIFY-1411: async iterator emits verdicts as workers complete."""
    pipeline = SyntheticSyncPipeline(
        energies={"slow": 1.0, "fast": 0.0},
        delays={"slow": 0.05, "fast": 0.0},
    )
    candidates = [
        {"id": "candidate-slow", "question": "q", "answer": "slow"},
        {"id": "candidate-fast", "question": "q", "answer": "fast"},
    ]

    records = asyncio.run(
        _collect_records(
            verify_stream(
                candidates,
                pipeline=pipeline,
                max_concurrency=2,
            )
        )
    )

    assert [record.extras["candidate_id"] for record in records] == [
        "candidate-fast",
        "candidate-slow",
    ]
    assert [record.extras["stream_index"] for record in records] == [0, 1]
    assert records[0].to_dict()["extras"]["candidate_id"] == "candidate-fast"


def test_verify_stream_top_k_margin_stops_before_full_pool() -> None:
    """REQ-VERIFY-1412: decisive top-k margin stops scheduling remaining candidates."""
    pipeline = SyntheticSyncPipeline(energies={"best": 0.0, "runner_up": 2.0, "never_scored": 3.0})
    candidates = [
        {"id": "best", "question": "q", "answer": "best"},
        {"id": "runner-up", "question": "q", "answer": "runner_up"},
        {"id": "never", "question": "q", "answer": "never_scored"},
    ]

    records = asyncio.run(
        _collect_records(
            verify_stream(
                candidates,
                pipeline=pipeline,
                max_concurrency=1,
                top_k=1,
                early_stop_margin=1.0,
            )
        )
    )

    assert [record.extras["candidate_id"] for record in records] == ["best", "runner-up"]
    assert pipeline.scored == ["best", "runner_up"]
    stream_end = records[-1].extras["stream_end"]
    assert stream_end["event"] == "stream_end"
    assert stream_end["stopped_early"] is True
    assert stream_end["residual_candidates_unscored"] == 1


def test_verify_stream_cancels_pending_workers_on_consumer_close() -> None:
    """REQ-VERIFY-1412: consumer disconnect cancels outstanding async workers."""
    pipeline = SyntheticAsyncPipeline()
    candidates = [
        {"id": "fast", "question": "q", "answer": "fast"},
        {"id": "slow", "question": "q", "answer": "slow"},
    ]

    async def exercise_close() -> VerdictRecord:
        stream = verify_stream(candidates, pipeline=pipeline, max_concurrency=2)
        first = await anext(stream)
        await stream.aclose()
        await asyncio.sleep(0)
        return first

    first_record = asyncio.run(exercise_close())

    assert first_record.extras["candidate_id"] == "fast"
    assert "slow" in pipeline.started
    assert "slow" in pipeline.cancelled


def test_collect_verify_stream_returns_event_summary() -> None:
    """REQ-VERIFY-1413: collected stream output has verdict events and stream_end."""
    pipeline = SyntheticSyncPipeline(energies={"best": 0.0, "runner_up": 2.0, "never_scored": 3.0})
    candidates = [
        {"id": "best", "question": "q", "answer": "best"},
        {"id": "runner-up", "question": "q", "answer": "runner_up"},
        {"id": "never", "question": "q", "answer": "never_scored"},
    ]

    payload = asyncio.run(
        collect_verify_stream(
            candidates,
            pipeline=pipeline,
            max_concurrency=1,
            top_k=1,
            early_stop_margin=1.0,
        )
    )

    assert [event["event"] for event in payload["events"]] == ["verdict", "verdict"]
    assert payload["events"][0]["record"]["extras"]["candidate_id"] == "best"
    assert payload["stream_end"]["event"] == "stream_end"
    assert payload["stream_end"]["stopped_early"] is True
    assert payload["stream_end"]["residual_candidates_unscored"] == 1
