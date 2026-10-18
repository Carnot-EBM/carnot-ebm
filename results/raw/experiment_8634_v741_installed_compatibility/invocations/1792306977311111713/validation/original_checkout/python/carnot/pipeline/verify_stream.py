"""Streaming verification over candidate response pools.

Spec: REQ-VERIFY-1411, REQ-VERIFY-1412, REQ-VERIFY-1413,
SCENARIO-VERIFY-1411
"""

from __future__ import annotations

import asyncio
import inspect
import math
import queue
import threading
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from carnot.pipeline.verdict_record import VerdictRecord


@dataclass(frozen=True)
class VerifyStreamCandidate:
    """Candidate response accepted by the streaming verifier.

    Spec: REQ-VERIFY-1411
    """

    id: str
    question: str
    answer: str
    domain: str | None = None


CandidateLike = VerifyStreamCandidate | Mapping[str, Any]


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_positive_float(value: float | int | None, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0.0 or math.isnan(parsed):
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _coerce_candidate(candidate: CandidateLike, index: int) -> VerifyStreamCandidate:
    if isinstance(candidate, VerifyStreamCandidate):
        if not candidate.id:
            raise ValueError("candidate id must be non-empty")
        return candidate

    if not isinstance(candidate, Mapping):
        raise ValueError(f"candidate {index} must be a mapping or VerifyStreamCandidate")

    candidate_id = candidate.get("id", candidate.get("candidate_id"))
    question = candidate.get("question")
    answer = candidate.get("answer", candidate.get("response"))
    domain = candidate.get("domain")

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"candidate {index} must include a non-empty string id")
    if not isinstance(question, str):
        raise ValueError(f"candidate {index} must include a string question")
    if not isinstance(answer, str):
        raise ValueError(f"candidate {index} must include a string answer or response")
    if domain is not None and not isinstance(domain, str):
        raise ValueError(f"candidate {index} domain must be a string when provided")

    return VerifyStreamCandidate(
        id=candidate_id,
        question=question,
        answer=answer,
        domain=domain,
    )


def _coerce_candidates(candidates: Iterable[CandidateLike]) -> list[VerifyStreamCandidate]:
    return [_coerce_candidate(candidate, index) for index, candidate in enumerate(candidates)]


def _energy_sort_key(record: VerdictRecord) -> float:
    energy = float(record.energy)
    return energy if math.isfinite(energy) else float("inf")


def _rank_for_candidate(
    completed: list[tuple[VerifyStreamCandidate, VerdictRecord]],
    candidate_id: str,
) -> int:
    ranked = sorted(completed, key=lambda item: (_energy_sort_key(item[1]), item[0].id))
    for rank, (candidate, _record) in enumerate(ranked, start=1):
        if candidate.id == candidate_id:
            return rank
    return len(ranked)


def _stream_end_payload(
    *,
    stopped_early: bool,
    stop_reason: str,
    total_candidates: int,
    emitted_count: int,
    scored_count: int,
    residual_candidates_unscored: int,
    top_k: int | None,
    early_stop_margin: float | None,
    observed_margin: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "stream_end",
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "total_candidates": total_candidates,
        "emitted_count": emitted_count,
        "scored_count": scored_count,
        "residual_candidates_unscored": residual_candidates_unscored,
        "top_k": top_k,
        "early_stop_margin": early_stop_margin,
    }
    if observed_margin is not None:
        payload["observed_margin"] = observed_margin
    return payload


def _maybe_early_stop_payload(
    completed: list[tuple[VerifyStreamCandidate, VerdictRecord]],
    *,
    total_candidates: int,
    emitted_count_after_current: int,
    top_k: int | None,
    early_stop_margin: float | None,
) -> dict[str, Any] | None:
    if top_k is None or early_stop_margin is None:
        return None
    if len(completed) < top_k + 1:
        return None

    ranked = sorted(completed, key=lambda item: (_energy_sort_key(item[1]), item[0].id))
    kth_energy = _energy_sort_key(ranked[top_k - 1][1])
    next_energy = _energy_sort_key(ranked[top_k][1])
    observed_margin = next_energy - kth_energy
    if observed_margin < early_stop_margin:
        return None

    return _stream_end_payload(
        stopped_early=True,
        stop_reason="early_stop_margin",
        total_candidates=total_candidates,
        emitted_count=emitted_count_after_current,
        scored_count=len(completed),
        residual_candidates_unscored=max(0, total_candidates - len(completed)),
        top_k=top_k,
        early_stop_margin=early_stop_margin,
        observed_margin=observed_margin,
    )


def _with_stream_extras(record: VerdictRecord, extras: dict[str, Any]) -> VerdictRecord:
    merged_extras = {**record.extras, **extras}
    return VerdictRecord(
        verdict=record.verdict,
        energy=record.energy,
        calibrated_confidence=record.calibrated_confidence,
        producing_tier=record.producing_tier,
        tier_reached=record.tier_reached,
        rationale=record.rationale,
        budget_ms_consumed=record.budget_ms_consumed,
        repairs_applied=list(record.repairs_applied),
        extras=merged_extras,
    )


async def _call_verify_record(
    pipeline: Any,
    candidate: VerifyStreamCandidate,
    *,
    default_domain: str | None,
) -> VerdictRecord:
    verify_record = getattr(pipeline, "verify_record", None)
    if verify_record is None:
        raise TypeError("pipeline must expose verify_record(question, response, domain=None)")

    domain = candidate.domain or default_domain
    if inspect.iscoroutinefunction(verify_record):
        record = await verify_record(candidate.question, candidate.answer, domain=domain)
    else:
        record = await _run_sync_in_daemon_thread(
            verify_record,
            candidate.question,
            candidate.answer,
            domain=domain,
        )

    if not isinstance(record, VerdictRecord):
        raise TypeError("pipeline.verify_record() must return VerdictRecord")
    return record


async def _run_sync_in_daemon_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run sync verifier code outside the event loop.

    This intentionally avoids ``asyncio.to_thread`` because the repository's
    sandbox can run the worker thread but fail to wake the selector after the
    thread completes. Polling a one-item queue keeps the async stream portable.
    """
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def runner() -> None:
        try:
            result_queue.put((True, fn(*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put((False, exc))

    threading.Thread(target=runner, daemon=True).start()
    while True:
        try:
            ok, value = result_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.001)
            continue
        if ok:
            return value
        raise value


async def _verify_one(
    pipeline: Any,
    candidate: VerifyStreamCandidate,
    *,
    default_domain: str | None,
    budget_ms_per_candidate: float | None,
) -> VerdictRecord:
    started_at = time.monotonic()
    try:
        call = _call_verify_record(pipeline, candidate, default_domain=default_domain)
        if budget_ms_per_candidate is None:
            return await call
        return await asyncio.wait_for(call, timeout=budget_ms_per_candidate / 1000.0)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        budget_ms = (time.monotonic() - started_at) * 1000.0
        return VerdictRecord(
            verdict="abstain",
            energy=float("inf"),
            calibrated_confidence=0.0,
            producing_tier=3,
            tier_reached=3,
            rationale=f"candidate verification exceeded {budget_ms_per_candidate:.1f}ms",
            budget_ms_consumed=budget_ms,
            extras={"error": True, "error_code": "CANDIDATE_TIMEOUT"},
        )
    except Exception as exc:
        budget_ms = (time.monotonic() - started_at) * 1000.0
        return VerdictRecord(
            verdict="abstain",
            energy=float("inf"),
            calibrated_confidence=0.0,
            producing_tier=3,
            tier_reached=3,
            rationale=f"candidate verification failed: {exc}",
            budget_ms_consumed=budget_ms,
            extras={"error": True, "error_code": "CANDIDATE_ERROR", "detail": str(exc)},
        )


async def verify_stream(
    candidates: Iterable[CandidateLike],
    *,
    pipeline: Any | None = None,
    domain: str | None = None,
    budget_ms_total: float | None = None,
    budget_ms_per_candidate: float | None = None,
    top_k: int | None = None,
    early_stop_margin: float | None = None,
    max_concurrency: int = 4,
) -> AsyncIterator[VerdictRecord]:
    """Yield structured verdicts as candidate verification completes.

    Spec: REQ-VERIFY-1411, REQ-VERIFY-1412
    """
    candidate_pool = _coerce_candidates(candidates)
    max_concurrency = _validate_positive_int(max_concurrency, "max_concurrency")
    budget_ms_total = _validate_positive_float(budget_ms_total, "budget_ms_total")
    budget_ms_per_candidate = _validate_positive_float(
        budget_ms_per_candidate,
        "budget_ms_per_candidate",
    )
    if top_k is not None:
        top_k = _validate_positive_int(top_k, "top_k")
        if top_k >= len(candidate_pool) and early_stop_margin is not None:
            raise ValueError("top_k must be smaller than the candidate pool for early stopping")
    if early_stop_margin is not None:
        early_stop_margin = float(early_stop_margin)
        if early_stop_margin < 0.0 or math.isnan(early_stop_margin):
            raise ValueError("early_stop_margin must be non-negative")
        if top_k is None:
            raise ValueError("early_stop_margin requires top_k")

    if pipeline is None:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(model=None, domains=[domain] if domain else None)

    deadline = None
    if budget_ms_total is not None:
        deadline = time.monotonic() + budget_ms_total / 1000.0

    pending: dict[asyncio.Task[VerdictRecord], VerifyStreamCandidate] = {}
    completed: list[tuple[VerifyStreamCandidate, VerdictRecord]] = []
    next_index = 0
    emitted_count = 0

    def schedule_ready() -> None:
        nonlocal next_index
        while next_index < len(candidate_pool) and len(pending) < max_concurrency:
            if deadline is not None and time.monotonic() >= deadline:
                return
            candidate = candidate_pool[next_index]
            next_index += 1
            task = asyncio.create_task(
                _verify_one(
                    pipeline,
                    candidate,
                    default_domain=domain,
                    budget_ms_per_candidate=budget_ms_per_candidate,
                )
            )
            pending[task] = candidate

    try:
        schedule_ready()
        stopped = False
        while pending:
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout == 0.0:
                    break

            done, _not_done = await asyncio.wait(
                pending,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break

            for task in done:
                candidate = pending.pop(task)
                record = task.result()
                completed.append((candidate, record))
                emitted_count_after_current = emitted_count + 1
                stream_end = _maybe_early_stop_payload(
                    completed,
                    total_candidates=len(candidate_pool),
                    emitted_count_after_current=emitted_count_after_current,
                    top_k=top_k,
                    early_stop_margin=early_stop_margin,
                )
                extras: dict[str, Any] = {
                    "candidate_id": candidate.id,
                    "stream_index": emitted_count,
                    "stream_rank": _rank_for_candidate(completed, candidate.id),
                    "top_k": top_k,
                    "early_stop_margin": early_stop_margin,
                }
                if stream_end is not None:
                    extras["stream_end"] = stream_end
                    stopped = True

                emitted_count = emitted_count_after_current
                yield _with_stream_extras(record, extras)

                if stopped:
                    break

            if stopped:
                break
            schedule_ready()
    finally:
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


async def collect_verify_stream(
    candidates: Iterable[CandidateLike],
    *,
    pipeline: Any | None = None,
    domain: str | None = None,
    budget_ms_total: float | None = None,
    budget_ms_per_candidate: float | None = None,
    top_k: int | None = None,
    early_stop_margin: float | None = None,
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Collect streaming verdicts into an MCP-friendly event payload.

    Spec: REQ-VERIFY-1413
    """
    candidate_pool = _coerce_candidates(candidates)
    events: list[dict[str, Any]] = []
    stream_end: dict[str, Any] | None = None

    async for record in verify_stream(
        candidate_pool,
        pipeline=pipeline,
        domain=domain,
        budget_ms_total=budget_ms_total,
        budget_ms_per_candidate=budget_ms_per_candidate,
        top_k=top_k,
        early_stop_margin=early_stop_margin,
        max_concurrency=max_concurrency,
    ):
        record_dict = record.to_dict()
        events.append({"event": "verdict", "record": record_dict})
        maybe_stream_end = record_dict.get("extras", {}).get("stream_end")
        if isinstance(maybe_stream_end, dict):
            stream_end = maybe_stream_end

    if stream_end is None:
        stop_reason = "exhausted" if len(events) == len(candidate_pool) else "budget_exhausted"
        stream_end = _stream_end_payload(
            stopped_early=False,
            stop_reason=stop_reason,
            total_candidates=len(candidate_pool),
            emitted_count=len(events),
            scored_count=len(events),
            residual_candidates_unscored=max(0, len(candidate_pool) - len(events)),
            top_k=top_k,
            early_stop_margin=early_stop_margin,
        )

    return {"events": events, "stream_end": stream_end}
