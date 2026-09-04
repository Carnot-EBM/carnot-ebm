"""The non-privileged sampler records what a silent gap needs (REQ-CONDUCTOR-WCHAN-1).

SCENARIO-CONDUCTOR-WCHAN-1-SAMPLE: one record carries ts, wchan, State line, children.
SCENARIO-CONDUCTOR-WCHAN-1-NEVER-SKIPS: unreadable proc entries become error markers,
never missing lines — a gap in the sample file is the ambiguity the sampler removes.

Origin: two ~60-minute conductor gaps closed with "no cause named" because kernel
state was only ever observed AFTER recovery, and py-spy needs ptrace this
environment lacks (known-issues 2026-09-03).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "conductor_heartbeat_sampler",
    Path(__file__).resolve().parents[2] / "scripts" / "conductor_heartbeat_sampler.py",
)
sampler = importlib.util.module_from_spec(_SPEC)
sys.modules["conductor_heartbeat_sampler"] = sampler
_SPEC.loader.exec_module(sampler)


def _fake_proc(tmp_path: Path, pid: int, *, wchan: str, state_line: str) -> Path:
    proc = tmp_path / "proc"
    (proc / str(pid)).mkdir(parents=True)
    (proc / str(pid) / "wchan").write_text(wchan)
    (proc / str(pid) / "status").write_text(f"Name:\tpython\n{state_line}\nPid:\t{pid}\n")
    return proc


def test_sample_carries_all_four_signals(tmp_path: Path) -> None:
    """SCENARIO-CONDUCTOR-WCHAN-1-SAMPLE."""

    proc = _fake_proc(tmp_path, 4242, wchan="do_select", state_line="State:\tS (sleeping)")
    sample = sampler.take_sample(4242, proc_root=proc, count_children=lambda pid: 3)

    assert sample["pid"] == 4242
    assert sample["wchan"] == "do_select"
    assert sample["state"] == "S (sleeping)"
    assert sample["children"] == 3
    assert sample["ts"].endswith("Z")


def test_unreadable_proc_becomes_marker_not_gap(tmp_path: Path) -> None:
    """SCENARIO-CONDUCTOR-WCHAN-1-NEVER-SKIPS: a vanished pid still yields a record."""

    proc = tmp_path / "proc"  # no <pid> directory at all
    proc.mkdir()
    sample = sampler.take_sample(999, proc_root=proc, count_children=lambda pid: None)

    assert sample["wchan"].startswith("unreadable:")
    assert sample["state"].startswith("unreadable:")
    assert sample["children"] == "pgrep_failed"


def test_append_writes_one_jsonl_line_per_sample(tmp_path: Path) -> None:
    """Samples land as daily JSONL outside the repo; each call appends exactly one line."""

    out = tmp_path / "samples"
    p1 = sampler.append_sample({"ts": "t1", "pid": 1}, out)
    p2 = sampler.append_sample({"ts": "t2", "pid": 1}, out)

    assert p1 == p2
    lines = [json.loads(line) for line in p1.read_text().splitlines()]
    assert [row["ts"] for row in lines] == ["t1", "t2"]


def test_run_once_records_conductor_down_as_data(tmp_path: Path, monkeypatch) -> None:
    """A stopped conductor is a record ('down at this minute'), not a silent no-op."""

    monkeypatch.setattr(sampler, "conductor_pid", lambda: None)
    out = tmp_path / "samples"
    assert sampler.run_once(out) == 0
    row = json.loads(next(out.glob("*.jsonl")).read_text().splitlines()[0])
    assert row["pid"] is None
    assert "not running" in row["note"]
