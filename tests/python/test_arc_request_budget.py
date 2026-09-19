"""Live request-budget invariants for REQ-ARC-WMTE-7411."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
from pathlib import Path

import pytest

from carnot.agentic import arc_induction_tool_loop as tool_loop
from carnot.agentic import arc_executable_world_model as world_model
from carnot.agentic.arc_inference_boundary import boundary_call_for_proposer
from carnot.agentic.arc_request_budget import (
    EpisodeDeadlineExceeded,
    EpisodeRequestBudget,
    LateRequestCompletion,
    RequestAlreadyCompleted,
    RequestBudgetError,
    RequestBudgetExhausted,
    attach_request_budget,
    request_budget_scope,
)


def _proposer() -> world_model.LocalGGUFProposer:
    proposer = world_model.LocalGGUFProposer(
        model_path="/fixtures/fixture.gguf",
        model_repository="fixture/model",
        model_filename="fixture.gguf",
        model_revision="a" * 40,
        ffn_cpu_layers=0,
        mtp=False,
    )
    proposer._proc = type("Proc", (), {"pid": 7411})()
    return proposer


def test_success_failure_retry_and_exhaustion_share_two_slots() -> None:
    """SCENARIO-ARC-WMTE-7411-ATOMIC-RESERVATION covers parser retries."""
    budget = EpisodeRequestBudget("episode-a", limit=2, deadline_s=60.0)
    first = budget.reserve(branch="primary", request_id="primary-0")
    first.complete()
    retry = budget.reserve(branch="parser_retry", request_id="retry-0")
    retry.fail(ValueError("invalid envelope"))

    with pytest.raises(RequestBudgetExhausted):
        budget.reserve(branch="repair", request_id="repair-0")

    receipt = budget.receipt()
    assert receipt["attempted"] == 2
    assert receipt["completed"] == 1
    assert receipt["failed"] == 1
    assert receipt["cancelled"] == receipt["in_flight"] == 0
    assert receipt["accounting_valid"] is True
    assert [row["branch"] for row in receipt["callback_rows"]] == ["primary", "parser_retry"]


def test_cancellation_discards_late_completion_before_commit() -> None:
    """SCENARIO-ARC-WMTE-7411-CANCELLATION prevents a late write."""
    budget = EpisodeRequestBudget("episode-late", limit=2, deadline_s=60.0)
    reservation = budget.reserve(branch="refinement", request_id="slow")
    committed: list[str] = []
    budget.cancel("episode_closed")

    with pytest.raises(LateRequestCompletion):
        reservation.complete()
        committed.append("memory")

    receipt = budget.receipt()
    assert committed == []
    assert receipt["cancelled"] == 1
    assert receipt["late_completions_discarded"] == 1
    assert receipt["late_write_violations"] == 0
    assert receipt["callback_rows"][0]["disposition"] == "cancelled"


def test_monotonic_deadline_cancels_inflight_and_refuses_dispatch() -> None:
    """SCENARIO-ARC-WMTE-7411-CANCELLATION uses one monotonic deadline."""
    ticks = iter((100, 101, 111, 112, 113, 114, 115))
    budget = EpisodeRequestBudget(
        "episode-deadline", limit=2, deadline_s=10.0, clock_ns=lambda: next(ticks)
    )
    budget.reserve(branch="primary", request_id="in-flight")
    with pytest.raises(EpisodeDeadlineExceeded):
        budget.reserve(branch="supervisor", request_id="too-late")

    receipt = budget.receipt()
    assert receipt["attempted"] == 1
    assert receipt["cancelled"] == 1
    assert receipt["deadline_exceeded"] is True
    assert receipt["deadline_violations"] == 0


def test_concurrent_and_nested_callbacks_cannot_overreserve() -> None:
    """SCENARIO-ARC-WMTE-7411-ATOMIC-RESERVATION covers callback races."""
    budget = EpisodeRequestBudget("episode-race", limit=2, deadline_s=60.0)
    barrier = threading.Barrier(8)

    def callback(index: int) -> str:
        barrier.wait()
        try:
            reservation = budget.reserve(branch="concurrent", request_id=f"race-{index}")
        except RequestBudgetExhausted:
            return "refused"
        reservation.complete()
        return "dispatched"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(callback, range(8)))

    assert outcomes.count("dispatched") == 2
    assert budget.receipt()["attempted"] == 2

    nested = EpisodeRequestBudget("episode-nested", limit=2, deadline_s=60.0)
    outer = nested.reserve(branch="primary", request_id="outer")
    inner = nested.reserve(branch="nested", request_id="inner")
    with pytest.raises(RequestBudgetExhausted):
        nested.reserve(branch="nested", request_id="nested-overflow")
    inner.complete()
    outer.complete()
    assert nested.receipt()["accounting_valid"] is True


def test_cold_restart_replay_does_not_duplicate_completed_request() -> None:
    """SCENARIO-ARC-WMTE-7411-RESTART preserves completed identities."""
    first = EpisodeRequestBudget("episode-restart", limit=2, deadline_s=60.0)
    reservation = first.reserve(branch="primary", request_id="stable-request")
    reservation.complete()
    durable = first.receipt()

    restarted = EpisodeRequestBudget.from_receipt(durable, deadline_s=60.0)
    with pytest.raises(RequestAlreadyCompleted):
        restarted.reserve(branch="primary", request_id="stable-request")
    second = restarted.reserve(branch="repair", request_id="new-request")
    second.complete()

    receipt = restarted.receipt()
    assert receipt["attempted"] == 2
    assert len(receipt["callback_rows"]) == 2
    assert receipt["replayed_completed_refusals"] == 1


def test_actual_http_boundary_reserves_before_dispatch_and_default_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7411 enforces the opt-in budget at the HTTP boundary."""
    proposer = _proposer()
    budget = EpisodeRequestBudget("episode-http", limit=2, deadline_s=60.0)
    attach_request_budget(proposer, budget)
    dispatches: list[str] = []

    def scripted_open(*args: object, **kwargs: object) -> io.BytesIO:
        dispatches.append("sent")
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())

    monkeypatch.setattr(tool_loop.urllib.request, "urlopen", scripted_open)
    for index, branch in enumerate(("primary", "repair")):
        with request_budget_scope(branch, request_id=f"http-{index}"):
            tool_loop._post_chat(proposer, [], turn=index, timeout_s=1, selfparse=True)
    with request_budget_scope("supervisor", request_id="http-2"):
        with pytest.raises(RequestBudgetExhausted):
            tool_loop._post_chat(proposer, [], turn=2, timeout_s=1, selfparse=True)

    assert len(dispatches) == 2
    assert budget.receipt()["attempted"] == 2

    unguarded = _proposer()
    for index in range(3):
        tool_loop._post_chat(unguarded, [], turn=index, timeout_s=1, selfparse=True)
    assert len(dispatches) == 5


def test_boundary_timeout_consumes_slot_and_late_boundary_completion_raises() -> None:
    """SCENARIO-ARC-WMTE-7411-CANCELLATION covers terminal boundary states."""
    proposer = _proposer()
    budget = EpisodeRequestBudget("episode-boundary", limit=2, deadline_s=60.0)
    attach_request_budget(proposer, budget)

    with request_budget_scope("repair", request_id="timeout"):
        failed = boundary_call_for_proposer(proposer, "generation")
    failed.fail(TimeoutError("timed out"))

    with request_budget_scope("refinement", request_id="late"):
        late = boundary_call_for_proposer(proposer, "generation")
    budget.cancel("episode_closed")
    with pytest.raises(LateRequestCompletion):
        late.complete(usable=True)

    receipt = budget.receipt()
    assert receipt["attempted"] == 2
    assert receipt["cancelled"] == 2
    assert receipt["accounting_valid"] is True


def test_budget_input_validation_and_receipt_copy_are_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7411 rejects malformed limits and restart receipts."""
    with pytest.raises(ValueError, match="limit"):
        EpisodeRequestBudget("bad", limit=0, deadline_s=1)
    with pytest.raises(ValueError, match="deadline"):
        EpisodeRequestBudget("bad", limit=1, deadline_s=0)
    with pytest.raises(ValueError, match="episode"):
        EpisodeRequestBudget("", limit=1, deadline_s=1)

    budget = EpisodeRequestBudget("copy", limit=1, deadline_s=1)
    row = budget.reserve(branch="primary", request_id="one")
    row.complete()
    receipt = budget.receipt()
    receipt["callback_rows"][0]["branch"] = "mutated"
    assert budget.receipt()["callback_rows"][0]["branch"] == "primary"

    malformed = budget.receipt()
    malformed["callback_rows"][0]["disposition"] = "in_flight"
    with pytest.raises(ValueError, match="restart"):
        EpisodeRequestBudget.from_receipt(malformed, deadline_s=1)

    # The fixture path is only a type-stable sink for coverage tools.
    assert tmp_path.is_dir()


def test_restart_receipts_reject_missing_oversized_and_duplicate_rows() -> None:
    """SCENARIO-ARC-WMTE-7411-RESTART rejects ambiguous durable identity."""
    with pytest.raises(ValueError, match="callback_rows"):
        EpisodeRequestBudget.from_receipt({}, deadline_s=1)

    oversized = {
        "episode_id": "oversized",
        "limit": 1,
        "callback_rows": [
            {"request_id": "one", "disposition": "completed"},
            {"request_id": "two", "disposition": "failed"},
        ],
    }
    with pytest.raises(ValueError, match="exceeds"):
        EpisodeRequestBudget.from_receipt(oversized, deadline_s=1)

    duplicate = {
        "episode_id": "duplicate",
        "limit": 2,
        "callback_rows": [
            {"request_id": "same", "disposition": "completed"},
            {"request_id": "same", "disposition": "failed"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        EpisodeRequestBudget.from_receipt(duplicate, deadline_s=1)


def test_closed_duplicate_and_already_terminal_paths_fail_closed() -> None:
    """REQ-ARC-WMTE-7411 keeps closed and duplicate identities terminal."""
    duplicate = EpisodeRequestBudget("duplicate", limit=2, deadline_s=60)
    reservation = duplicate.reserve(branch="primary", request_id="same")
    with pytest.raises(RequestBudgetExhausted, match="already reserved"):
        duplicate.reserve(branch="repair", request_id="same")
    reservation.fail("parser failed")
    reservation.fail("ignored second failure")
    with pytest.raises(RequestBudgetError, match="already terminal"):
        reservation.complete()

    closed = EpisodeRequestBudget("closed", limit=2, deadline_s=60)
    closed.cancel("operator_cancelled")
    with pytest.raises(RequestBudgetExhausted, match="episode closed"):
        closed.reserve(branch="supervisor", request_id="after-close")


def test_completion_and_failure_after_deadline_are_cancelled() -> None:
    """SCENARIO-ARC-WMTE-7411-CANCELLATION closes both terminal paths."""
    completion_ticks = iter((10.0, 10.5, 12.0, 12.0))
    completion = EpisodeRequestBudget(
        "completion-deadline", limit=1, deadline_s=1, clock_ns=lambda: next(completion_ticks)
    )
    late = completion.reserve(branch="primary", request_id="late-complete")
    with pytest.raises(LateRequestCompletion):
        late.complete()
    assert completion.receipt()["deadline_exceeded"] is True

    failure_ticks = iter((20.0, 20.5, 22.0, 22.0))
    failure = EpisodeRequestBudget(
        "failure-deadline", limit=1, deadline_s=1, clock_ns=lambda: next(failure_ticks)
    )
    failed = failure.reserve(branch="primary", request_id="late-fail")
    failed.fail(ValueError("late parser failure"))
    receipt = failure.receipt()
    assert receipt["deadline_exceeded"] is True
    assert receipt["callback_rows"][0]["disposition"] == "cancelled"
