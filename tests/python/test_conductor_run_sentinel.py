"""Tests for scripts/conductor_run_sentinel.py — the in-flight run sentinel.

REQ-CONDUCTOR-SENTINEL-1: read live-run validity signals while the run is
alive, via the SAME row evaluator the post-hoc gate uses, plus the
llama-server stderr log; never kill the run.
REQ-CONDUCTOR-SENTINEL-2: GPU resource health from /proc and nvidia-smi,
failing closed (a check that cannot run says so).
REQ-CONDUCTOR-SENTINEL-3: durable, deduplicated, receipted escalation.

All filesystem interaction goes through tmp_path: fake /proc trees, fake out
files, fake server logs, and explicit conductor-log/known-issues/state paths.
No test touches tracked state.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SENTINEL = _REPO / "scripts" / "conductor_run_sentinel.py"


def _load():
    spec = importlib.util.spec_from_file_location("conductor_run_sentinel", _SENTINEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load()


# ---------------------------------------------------------------------------
# fixtures: fake /proc entries and harness rows
# ---------------------------------------------------------------------------


def _mk_proc(proc_root: Path, pid: int, cmdline: list[str], environ=None, ppid=None, cgroup=None):
    d = proc_root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in cmdline) + b"\0")
    if environ is not None:
        (d / "environ").write_bytes(
            b"\0".join(f"{k}={v}".encode() for k, v in environ.items()) + b"\0"
        )
    if ppid is not None:
        (d / "status").write_text(f"Name:\tx\nPPid:\t{ppid}\n")
    if cgroup is not None:
        (d / "cgroup").write_text(cgroup)
    return d


def _invalid_llm_on_row() -> dict:
    """The 2026-08-22 incident shape: dead generator, zero completions."""
    return {
        "llm_enabled": True,
        "llm_tier_operational": True,
        "generator_healthy_after": False,
        "server_storm_suspected": False,
        "llm_on_row_valid": False,
        "llm": {"responses": 0, "calls": 3, "errors": 3, "content_failures": 0},
    }


def _valid_llm_on_row() -> dict:
    return {
        "llm_enabled": True,
        "llm_tier_operational": True,
        "generator_healthy_after": True,
        "server_storm_suspected": False,
        "llm_on_row_valid": True,
        "llm": {"responses": 5, "calls": 6, "errors": 1, "content_failures": 0},
    }


def _llm_off_row() -> dict:
    return {"llm_enabled": False, "llm_on_row_valid": True}


_INCIDENT_LOG_LINES = (
    "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 629440768\n"
    "graph_reserve: failed to allocate compute buffers\n"
    "failed to create context with model 'Qwen3.8-27B-Q4_K_M.gguf'\n"
)


def _healthy_gpu_runner(args):
    if "--query-gpu=index,uuid,memory.used,memory.total" in args[0]:
        return "0, GPU-aaa, 4, 24576\n1, GPU-bbb, 21237, 24576\n"
    return "GPU-bbb, 1583902, 20686\nGPU-bbb, 1550132, 536\n"


def _pin_ok():
    return "Qwen3.8-27B", None


def _scan(tmp_path: Path, proc_root: Path, **kw):
    """run_scan against a fake proc tree with all outputs under tmp_path."""
    defaults = dict(
        proc_root=proc_root,
        gpu_runner=_healthy_gpu_runner,
        pin_loader=_pin_ok,
        conductor_log=tmp_path / "conductor-log.md",
        known_issues=tmp_path / "known-issues.md",
        state_path=tmp_path / "state.json",
        # Hermetic: never let the shared /tmp of a busy box leak real
        # server logs into a test's findings.
        default_log_dir=tmp_path / "no_default_logs",
    )
    defaults.update(kw)
    return S.run_scan(**defaults)


def _codes(summary) -> list[str]:
    return [f["code"] for f in summary["findings"]]


def _mk_run(tmp_path: Path, rows) -> Path:
    """Fake proc tree holding one live harness run writing `rows`."""
    proc_root = tmp_path / "proc"
    out = tmp_path / "rows.json"
    out.write_text(json.dumps({"rows": rows}))
    slogs = tmp_path / "slogs"
    _mk_proc(
        proc_root,
        4242,
        ["python", "scripts/arc_scored_path_lever_harness.py", "--out", str(out), "--port", "8995"],
        environ={"CARNOT_ARC_SERVER_LOG_DIR": str(slogs)},
    )
    return proc_root


# ---------------------------------------------------------------------------
# REQ-CONDUCTOR-SENTINEL-1 — row validity + server log, in flight
# ---------------------------------------------------------------------------


def test_incident_replay_critical_escalation(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-1-INCIDENT-REPLAY: three invalid rows plus
    the allocation failure in the server log -> CRITICAL escalations, durable
    records, and no signal sent to any process."""
    proc_root = _mk_run(tmp_path, [_invalid_llm_on_row() for _ in range(3)])
    log_dir = tmp_path / "slogs" / "carnot_llama_server_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "llama_server_p8995_1.log").write_text(_INCIDENT_LOG_LINES)

    summary = _scan(tmp_path, proc_root)

    by_code = {f["code"]: f for f in summary["findings"]}
    assert by_code["CONSECUTIVE_INVALID_LLM_ON_ROWS"]["severity"] == "CRITICAL"
    assert by_code["SERVER_LOG_FAILURE"]["severity"] == "CRITICAL"
    assert "failed to allocate" in by_code["SERVER_LOG_FAILURE"]["detail"]
    log_text = (tmp_path / "conductor-log.md").read_text()
    assert "OPERATOR-ATTENTION: CONSECUTIVE_INVALID_LLM_ON_R" in log_text
    assert "BLOCK" in log_text
    issues = (tmp_path / "known-issues.md").read_text()
    assert "OPERATOR-ATTENTION" in issues and "SERVER_LOG_FAILURE" in issues


def test_sentinel_has_no_kill_path():
    """The sentinel never signals a process — enforced at the source level so
    a future edit that adds a kill cannot pass silently."""
    source = _SENTINEL.read_text()
    for token in ("os.kill", "killpg", "send_signal", "SIGKILL", "SIGTERM", ".terminate("):
        assert token not in source, f"kill primitive {token!r} found in sentinel"


def test_single_invalid_row_no_streak_escalation(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-1-SINGLE-INVALID-ROW: one bad row between
    valid rows is a tolerated blip, not an escalation."""
    proc_root = _mk_run(tmp_path, [_invalid_llm_on_row(), _valid_llm_on_row()])
    summary = _scan(tmp_path, proc_root)
    assert "CONSECUTIVE_INVALID_LLM_ON_ROWS" not in _codes(summary)


def test_llm_off_rows_do_not_mask_the_streak(tmp_path):
    """A --both run interleaves llm-off arms; the streak counts over the
    LLM-on subsequence only."""
    rows = [_invalid_llm_on_row(), _llm_off_row(), _invalid_llm_on_row()]
    proc_root = _mk_run(tmp_path, rows)
    summary = _scan(tmp_path, proc_root)
    assert "CONSECUTIVE_INVALID_LLM_ON_ROWS" in _codes(summary)


def test_contained_streak_is_warn_not_critical(tmp_path):
    """A streak inside a run that also has valid rows is WARN; only a run
    whose EVERY llm-on row is invalid (the incident shape) is CRITICAL."""
    rows = [_valid_llm_on_row(), _invalid_llm_on_row(), _invalid_llm_on_row()]
    proc_root = _mk_run(tmp_path, rows)
    summary = _scan(tmp_path, proc_root)
    finding = next(f for f in summary["findings"] if f["code"] == "CONSECUTIVE_INVALID_LLM_ON_ROWS")
    assert finding["severity"] == "WARN"


def test_midwrite_race_is_skipped(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-1-MIDWRITE-RACE: unparseable + fresh mtime
    is the harness's whole-file rewrite, not a finding — and the skip lands
    in the STATE FILE, not only on stdout."""
    proc_root = _mk_run(tmp_path, [])
    (tmp_path / "rows.json").write_text('{"rows": [truncated')
    summary = _scan(tmp_path, proc_root)
    assert "OUT_FILE_STALE_UNPARSEABLE" not in _codes(summary)
    assert any("mid-write" in n for n in summary["notes"])
    state = json.loads((tmp_path / "state.json").read_text())
    assert any("mid-write" in n for n in state["last_scan_notes"])


def test_nested_rows_shape_is_not_invisible(tmp_path):
    """Row discovery must not be narrower than the lint's row concept:
    a per-cell nested corpus shape (`cells[i].row`, the 2026-07 corpora)
    reaches the streak detector through the lint's own walk_rows."""
    doc = {"cells": [{"row": _invalid_llm_on_row()}, {"row": _invalid_llm_on_row()}]}
    proc_root = tmp_path / "proc"
    out = tmp_path / "rows.json"
    out.write_text(json.dumps(doc))
    _mk_proc(
        proc_root,
        4242,
        ["python", "scripts/arc_probe_harness.py", "--out", str(out), "--port", "8995"],
    )
    summary = _scan(tmp_path, proc_root)
    assert "CONSECUTIVE_INVALID_LLM_ON_ROWS" in _codes(summary)


def test_missing_out_file_is_noted_not_silent(tmp_path):
    """Absent is not zero: a run whose out file never appears leaves a
    durable trace in the state file."""
    proc_root = tmp_path / "proc"
    _mk_proc(
        proc_root,
        4242,
        ["python", "scripts/arc_x.py", "--out", str(tmp_path / "never.json"), "--port", "1"],
    )
    summary = _scan(tmp_path, proc_root)
    assert any("missing" in n for n in summary["notes"])
    state = json.loads((tmp_path / "state.json").read_text())
    assert any("missing" in n for n in state["last_scan_notes"])


def test_relative_out_resolves_against_run_cwd(tmp_path):
    """A relative --out means relative to the RUN's cwd (/proc/<pid>/cwd),
    not the sentinel's own cwd."""
    proc_root = tmp_path / "proc"
    run_cwd = tmp_path / "rundir"
    run_cwd.mkdir()
    (run_cwd / "rows.json").write_text(
        json.dumps({"rows": [_invalid_llm_on_row(), _invalid_llm_on_row()]})
    )
    d = _mk_proc(
        proc_root, 4242, ["python", "scripts/arc_x.py", "--out", "rows.json", "--port", "1"]
    )
    os.symlink(run_cwd, d / "cwd")
    summary = _scan(tmp_path, proc_root)
    assert "CONSECUTIVE_INVALID_LLM_ON_ROWS" in _codes(summary)


def test_stale_unparseable_is_a_finding(tmp_path):
    proc_root = _mk_run(tmp_path, [])
    out = tmp_path / "rows.json"
    out.write_text('{"rows": [truncated')
    old = time.time() - 3600
    os.utime(out, (old, old))
    summary = _scan(tmp_path, proc_root)
    assert "OUT_FILE_STALE_UNPARSEABLE" in _codes(summary)


def test_witness_missing_row_is_warn(tmp_path):
    """Absent is not valid: an LLM-on row with no liveness witness draws a
    WARN, never an implicit pass."""
    proc_root = _mk_run(tmp_path, [{"llm_enabled": True}, {"llm_enabled": True}])
    summary = _scan(tmp_path, proc_root)
    finding = next(f for f in summary["findings"] if f["code"] == "ROW_WITNESS_MISSING")
    assert finding["severity"] == "WARN"
    assert "2/2" in finding["detail"]


def test_row_evaluator_is_the_shared_lint(tmp_path):
    """REQ-CONDUCTOR-SENTINEL-1 rule 2: validity comes from the liveness
    lint's check_row, not a second pattern list. Mutating the lint's verdict
    must change the sentinel's verdict."""
    lint = S.load_liveness_lint()
    findings = S.evaluate_rows([_invalid_llm_on_row()] * 2, lint)
    assert any(f["code"] == "CONSECUTIVE_INVALID_LLM_ON_ROWS" for f in findings)

    class _MutedLint:
        FAIL_CODES = lint.FAIL_CODES
        _is_row = staticmethod(lint._is_row)
        _claims_llm_on = staticmethod(lint._claims_llm_on)

        @staticmethod
        def check_row(row):
            return []  # a lint that never fails a row

    assert S.evaluate_rows([_invalid_llm_on_row()] * 2, _MutedLint()) == []


# ---------------------------------------------------------------------------
# REQ-CONDUCTOR-SENTINEL-2 — GPU / resource health
# ---------------------------------------------------------------------------


def test_stranded_fragment_finding(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-2-STRANDED-FRAGMENT: the incident's
    3,744 MiB fragment with no owning compute app."""
    snapshot = {
        "gpus": [{"index": 0, "uuid": "GPU-aaa", "used_mib": 3744, "total_mib": 24576}],
        "apps": [],
    }
    findings = S.evaluate_gpu_snapshot(snapshot)
    assert findings and findings[0]["code"] == "STRANDED_VRAM"
    assert "3744 MiB" in findings[0]["detail"]
    assert findings[0]["gpu_index"] == 0


def test_healthy_box_no_gpu_findings(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-2-HEALTHY-BOX: the live box as measured
    2026-08-22 — 4 and 15 MiB unaccounted — draws nothing."""
    snapshot = {
        "gpus": [
            {"index": 0, "uuid": "GPU-aaa", "used_mib": 4, "total_mib": 24576},
            {"index": 1, "uuid": "GPU-bbb", "used_mib": 21237, "total_mib": 24576},
        ],
        "apps": [
            {"uuid": "GPU-bbb", "pid": 1583902, "used_mib": 20686},
            {"uuid": "GPU-bbb", "pid": 1550132, "used_mib": 536},
        ],
    }
    assert S.evaluate_gpu_snapshot(snapshot) == []


def test_orphan_llama_server_warn(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-2-ORPHAN: reparented to init, user slice,
    nothing referencing its port."""
    proc_root = tmp_path / "proc"
    _mk_proc(
        proc_root,
        999,
        ["/opt/bin/llama-server", "-m", "/models/Qwen3.8-27B-Q4.gguf", "--port", "8777"],
        ppid=1,
        cgroup="0::/user.slice/user-1000.slice/user@1000.service/app.slice/x.scope\n",
    )
    servers = S.discover_llama_servers(proc_root)
    findings = S.evaluate_llama_servers(servers, "Qwen3.8-27B", proc_root)
    assert [f["code"] for f in findings] == ["ORPHANED_LLAMA_SERVER"]
    assert findings[0]["severity"] == "WARN"


def test_reparented_but_referenced_is_not_orphan(tmp_path):
    """The live-box calibration case: the A/B server outlives its launcher
    shell while the harness still holds --port 8995. Not an orphan."""
    proc_root = tmp_path / "proc"
    _mk_proc(
        proc_root,
        999,
        ["/opt/bin/llama-server", "-m", "/models/Qwen3.8-27B-Q4.gguf", "--port", "8995"],
        ppid=1,
        cgroup="0::/user.slice/user-1000.slice/user@1000.service/tmux-spawn.scope\n",
    )
    _mk_proc(
        proc_root,
        1000,
        ["python", "scripts/arc_scored_path_lever_harness.py", "--port", "8995"],
    )
    servers = S.discover_llama_servers(proc_root)
    findings = S.evaluate_llama_servers(servers, "Qwen3.8-27B", proc_root)
    assert findings == []


def test_system_slice_server_is_not_orphan(tmp_path):
    proc_root = tmp_path / "proc"
    _mk_proc(
        proc_root,
        999,
        ["/opt/bin/llama-server", "-m", "/models/Qwen3.8-27B-Q4.gguf", "--port", "8777"],
        ppid=1,
        cgroup="0::/system.slice/llama.service\n",
    )
    servers = S.discover_llama_servers(proc_root)
    assert S.evaluate_llama_servers(servers, "Qwen3.8-27B", proc_root) == []


def test_wrong_model_loaded_warn(tmp_path):
    proc_root = tmp_path / "proc"
    _mk_proc(
        proc_root,
        999,
        ["/opt/bin/llama-server", "-m", "/models/Qwen3.5-9B-Q4.gguf", "--port", "8995"],
        ppid=1,
        cgroup="0::/user.slice/x.scope\n",
    )
    _mk_proc(proc_root, 1000, ["python", "x.py", "--port", "8995"])
    servers = S.discover_llama_servers(proc_root)
    findings = S.evaluate_llama_servers(servers, "Qwen3.8-27B", proc_root)
    assert [f["code"] for f in findings] == ["WRONG_MODEL_LOADED"]
    assert "Qwen3.5-9B-Q4.gguf" in findings[0]["detail"]


def test_gpu_check_unavailable_is_a_finding(tmp_path):
    """Fail closed and say so: nvidia-smi failure is UNKNOWN, never clean."""
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    summary = _scan(tmp_path, proc_root, gpu_runner=lambda args: None)
    assert "GPU_CHECK_UNAVAILABLE" in _codes(summary)


def test_pin_check_unavailable_is_a_finding(tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    summary = _scan(tmp_path, proc_root, pin_loader=lambda: (None, "ImportError: boom"))
    assert "PIN_CHECK_UNAVAILABLE" in _codes(summary)


# ---------------------------------------------------------------------------
# REQ-CONDUCTOR-SENTINEL-3 — durable, deduplicated, receipted escalation
# ---------------------------------------------------------------------------


def _one_finding():
    return [
        (
            "pid 1 /x/rows.json",
            {"code": "CONSECUTIVE_INVALID_LLM_ON_ROWS", "severity": "WARN", "detail": "d"},
        )
    ]


def test_dedupe_across_scans(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-3-DEDUPE: the same finding twice writes
    exactly one conductor-log row."""
    paths = dict(
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
    )
    first = S.escalate(_one_finding(), **paths)
    second = S.escalate(_one_finding(), **paths)
    assert first["written"] == 1 and second["written"] == 0 and second["deduplicated"] == 1
    rows = (tmp_path / "log.md").read_text().count("CONSECUTIVE_INVALID_LLM_ON_R")
    assert rows == 1


def test_receipt_always_advances(tmp_path):
    """SCENARIO-CONDUCTOR-SENTINEL-3-RECEIPT: a clean scan still writes
    last_scan_utc — the sentinel's own receipt."""
    state_path = tmp_path / "state.json"
    S.escalate(
        [],
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=state_path,
    )
    state = json.loads(state_path.read_text())
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", state["last_scan_utc"])
    assert not (tmp_path / "log.md").exists()  # no findings -> no log rows


def test_critical_writes_known_issue_warn_does_not(tmp_path):
    paths = dict(
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
    )
    warn = [("s1", {"code": "STRANDED_VRAM", "severity": "WARN", "detail": "d"})]
    crit = [("s2", {"code": "SERVER_LOG_FAILURE", "severity": "CRITICAL", "detail": "d"})]
    S.escalate(warn, **paths)
    assert not (tmp_path / "ki.md").exists()
    S.escalate(crit, **paths)
    text = (tmp_path / "ki.md").read_text()
    assert "## OPERATOR-ATTENTION" in text and "SERVER_LOG_FAILURE" in text


def test_log_row_format_matches_log_step(tmp_path):
    """The escalation row must parse under the conductor's own table format,
    so one parser reads one log."""
    paths = dict(
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=tmp_path / "state.json",
    )
    S.escalate(_one_finding(), **paths)
    lines = (tmp_path / "log.md").read_text().splitlines()
    row = lines[-1]
    assert re.fullmatch(
        r"\| \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC \| OPERATOR-ATTENTION: [^|]{1,50} \| WARN \| .{1,80} \|",
        row,
    )


def test_corrupt_state_recovers_and_still_escalates(tmp_path):
    """A torn or corrupt state file must not silence the sentinel."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{not json")
    summary = S.escalate(
        _one_finding(),
        conductor_log=tmp_path / "log.md",
        known_issues=tmp_path / "ki.md",
        state_path=state_path,
    )
    assert summary["written"] == 1
    assert json.loads(state_path.read_text())["last_scan_utc"]
