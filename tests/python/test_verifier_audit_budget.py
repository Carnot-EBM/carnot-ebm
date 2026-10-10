"""REQ-OPS-AUDIT-REVIEWER-1 (budget scenario): the verifier audit keeps partial work and rotates.

Incident (2026-10-09): the milestone-close verifier audit ran 20 reviewer calls of about 40 s each
under a 900 s kill. It wrote its report only at the end, so a slow run produced nothing. It also
audited the same first 20 of 389 files every time. These tests run the real main() against a
temporary verifier directory with a fake reviewer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import verifier_authenticity_audit as vaa  # noqa: E402

GOOD = "## VERDICT\nAUTHENTIC\n\n## RECOMMENDATION\nKEEP\n"
BODY = "# verifier\n" + "x = 1\n" * 40  # over the 100-character floor


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """A fake project root with 5 verifier files; report and rotation paths inside it."""
    verify = tmp_path / "python" / "carnot" / "verify"
    verify.mkdir(parents=True)
    for name in ("a", "b", "c", "d", "e"):
        (verify / f"{name}.py").write_text(BODY + f"# file-{name}\n")
    monkeypatch.setattr(vaa, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(vaa, "VERIFY_DIR", verify)
    monkeypatch.setattr(vaa, "REPORT_PATH", tmp_path / "ops" / "report.md")
    monkeypatch.setattr(vaa, "ROTATION_STATE", tmp_path / "ops" / "rot.json")
    return tmp_path


def _run(monkeypatch, argv, replies):
    """Run main() with a reviewer that returns `replies` in order and records file order."""
    calls: list[str] = []
    queue = list(replies)

    def fake(prompt, body, model=None):
        calls.append(body)
        return queue.pop(0) if queue else (True, GOOD)

    monkeypatch.setattr(vaa, "call_codex", fake)
    monkeypatch.setattr(sys, "argv", ["verifier_authenticity_audit.py", *argv])
    assert vaa.main() == 0
    return calls


def _offset(sb: Path) -> int:
    return json.loads((sb / "ops" / "rot.json").read_text())["offset"]


def test_rotation_walks_the_whole_list_instead_of_the_head(sandbox, monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-BUDGET: successive limited runs cover every file."""
    seen: list[int] = []
    order: list[str] = []
    for _ in range(3):  # 5 files, 2 per run: 3 runs wrap around
        calls = _run(monkeypatch, ["--model", "codex", "--limit", "2"], [])
        order += [c.rsplit("# file-", 1)[1].strip() for c in calls]
        seen.append(_offset(sandbox))
    assert seen == [2, 4, 1]  # (0+2), (2+2), (4+2) % 5
    assert order == ["a", "b", "c", "d", "e", "a"]  # the head is not re-audited every time


def test_budget_stops_starting_files_and_marks_the_report_partial(sandbox, monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-BUDGET: a spent budget keeps the work done so far."""
    clock = iter([0.0, 1.0, 999.0, 999.0, 999.0, 999.0])  # deadline set, file 1 starts, then late
    monkeypatch.setattr(vaa, "_now", lambda: next(clock))
    calls = _run(monkeypatch, ["--model", "codex", "--limit", "4", "--budget-seconds", "10"], [])
    assert len(calls) == 1  # one file reviewed, then the budget ran out
    report = (sandbox / "ops" / "report.md").read_text()
    assert "PARTIAL RUN" in report and "Scanned 1 of 4" in report
    assert _offset(sandbox) == 1  # moved by the one file actually reviewed, not by 4


def test_no_budget_means_no_partial_marker(sandbox, monkeypatch):
    _run(monkeypatch, ["--model", "codex", "--limit", "2"], [])
    assert "PARTIAL RUN" not in (sandbox / "ops" / "report.md").read_text()


def test_a_failed_first_call_still_moves_rotation_by_one(sandbox, monkeypatch):
    """One bad file must not freeze rotation, and it must show as UNKNOWN in the report."""
    _run(monkeypatch, ["--model", "codex", "--limit", "3"], [(False, "boom"), (True, GOOD)])
    assert _offset(sandbox) == 1
    assert "audit call failed: boom" in (sandbox / "ops" / "report.md").read_text()


def test_a_failure_in_the_middle_is_retried_next_run(sandbox, monkeypatch):
    _run(monkeypatch, ["--model", "codex", "--limit", "3"], [(True, GOOD), (False, "boom")])
    assert _offset(sandbox) == 1  # only the unbroken run of reviewed files counts


def test_state_is_not_written_when_the_report_write_fails(sandbox, monkeypatch):
    """A run that dies before the report exists must not count its files as covered."""
    real = Path.write_text

    def boom(self, *a, **k):
        if self == vaa.REPORT_PATH:
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", boom)
    monkeypatch.setattr(vaa, "call_codex", lambda *a, **k: (True, GOOD))
    monkeypatch.setattr(sys, "argv", ["x", "--model", "codex", "--limit", "2"])
    with pytest.raises(OSError):
        vaa.main()
    assert not (sandbox / "ops" / "rot.json").exists()


def test_single_file_runs_do_not_touch_rotation(sandbox, monkeypatch):
    target = str(sandbox / "python" / "carnot" / "verify" / "c.py")
    _run(monkeypatch, ["--model", "codex", "--file", target, "--limit", "1"], [])
    assert not (sandbox / "ops" / "rot.json").exists()


def test_changed_file_list_resets_the_offset(sandbox, monkeypatch):
    _run(monkeypatch, ["--model", "codex", "--limit", "2"], [])
    assert _offset(sandbox) == 2
    (sandbox / "python" / "carnot" / "verify" / "f.py").write_text(BODY)  # list changed
    calls = _run(monkeypatch, ["--model", "codex", "--limit", "1"], [])
    assert len(calls) == 1 and _offset(sandbox) == 1  # restarted from the top


def test_resolve_rotation_offset_is_fail_safe():
    sig = "5:abc"
    assert vaa.resolve_rotation_offset(None, sig) == 0
    assert vaa.resolve_rotation_offset({"units_signature": "other", "offset": 3}, sig) == 0
    assert vaa.resolve_rotation_offset({"units_signature": sig, "offset": "x"}, sig) == 0
    assert vaa.resolve_rotation_offset({"units_signature": sig, "offset": 3}, sig) == 3


@pytest.mark.parametrize(
    "results,expected",
    [([], 0), ([True, True], 2), ([False], 1), ([False, True], 1), ([True, False, True], 1)],
)
def test_advance_for(results, expected):
    assert vaa.advance_for(results) == expected


def test_conductor_passes_a_budget_that_fits_inside_its_kill_timeout():
    """The budget plus the worst single call must stay under the kill timeout the conductor sets."""
    src = (REPO / "scripts" / "research_conductor.py").read_text()
    start = src.index('"verifier-authenticity-audit"')
    block = src[start : start + 1800]
    budget = int(block.split('"--budget-seconds",')[1].split('"')[1])
    timeout = int(block.split("timeout=")[1].split(",")[0])
    assert budget + 300 < timeout
