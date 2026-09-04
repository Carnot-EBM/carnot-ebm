"""Spec: REQ-INFRA-6840, SCENARIO-INFRA-6840-B

The dashboard names the GPUs a job actually occupies, read from the driver.

INCIDENT 2026-09-04, twice in one session. An outer-loop session reported a GGUF run as
"GPU 1 only" because it had SET `CARNOT_ARC_GENERATOR_CUDA_GPU=1`. That variable selects a
preferred device and does not mask the others, and Qwen3.8-27B at `n_ctx=98304` exceeds one
24 GB card, so llama.cpp split it across both. The first misreport was corrected by hand in
commit bd08d8d071. The prose written afterwards did not stop the second one, which is why this
is a check instead of a rule.

A multi-card run is outside the conductor-owns-GPU-0 allocation, so it is named loudly rather
than listed quietly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import outer_loop_dashboard as dash  # noqa: E402

_UUID_ROWS = "0, GPU-aaaa\n1, GPU-bbbb"


def _fake_run(monkeypatch: pytest.MonkeyPatch, app_rows: str) -> None:
    """Answer the two nvidia-smi queries and nothing else.

    The join is the part that broke in the incident, so both halves are faked rather than
    stubbing the function under test.
    """

    def run(*args: str) -> str:
        joined = " ".join(args)
        if "--query-gpu=index,uuid" in joined:
            return _UUID_ROWS
        if "--query-compute-apps" in joined:
            return app_rows
        return ""

    monkeypatch.setattr(dash, "_run", run)


def test_a_worker_on_two_cards_reports_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact incident shape: one llama-server holding memory on both cards."""
    _fake_run(monkeypatch, "3115288, GPU-aaaa\n3115288, GPU-bbbb")
    assert dash.gpu_indices_for(3115288) == [0, 1]


def test_a_worker_on_one_card_reports_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run(monkeypatch, "3115288, GPU-bbbb")
    assert dash.gpu_indices_for(3115288) == [1]


def test_another_process_gpu_is_not_attributed_to_this_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conductor's own GPU-0 work must not be reported as the outer loop's."""
    _fake_run(monkeypatch, "999, GPU-aaaa\n3115288, GPU-bbbb")
    assert dash.gpu_indices_for(3115288) == [1]


def test_an_unknown_uuid_is_dropped_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run(monkeypatch, "3115288, GPU-zzzz")
    assert dash.gpu_indices_for(3115288) == []


def test_a_process_holding_nothing_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run(monkeypatch, "")
    assert dash.gpu_indices_for(3115288) == []


def test_two_cards_are_named_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quiet list is what let the first misreport pass a human reader."""
    assert dash.gpu_span_label([0, 1]) == " SPANS GPU 0,1"


def test_one_card_is_named_plainly() -> None:
    assert dash.gpu_span_label([1]) == " GPU 1"


def test_no_card_adds_nothing() -> None:
    assert dash.gpu_span_label([]) == ""
