"""REQ-REPORT-9181-V782: unchanged flags and immutable child observations."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any

import pytest

from scripts import adversarial_verify as av

ROOT = Path(__file__).resolve().parents[2]
BASELINE = (
    ROOT / "results/raw/experiment_9181_v782_causal_validation/baseline/adversarial_verify.py"
)


def payload() -> dict[str, Any]:
    """A small fixed artifact exposes repeated scans without an expensive search."""
    return dict(
        schema="arc_go_explore",
        arms_non_degenerate=True,
        null_delta_methodology_note="Same fixed input and independent controls",
        positive_control_passed=True,
        first_win_left=0.123456789,
        first_win_right=0.123456789,
        metric_a=0.7654321,
        metric_b=0.7654321,
        metric_c=0.3456789,
        metric_d=0.4567891,
        nested=[dict(field="structural") for _ in range(40)],
    )


@pytest.mark.parametrize("kind", ["arc", "ordinary", "declared", "excluded"])
def test_exact_original_flags_and_one_scan(kind: str, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9181-V782: preserve details, severity and pair ordering."""
    value = payload()
    if kind == "ordinary":
        value["schema"] = "ordinary"
    elif kind == "declared":
        value.update(metric_delta=0, metric_positive_control_passed=True)
    elif kind == "excluded":
        value = dict(random_seed=12, experiment_id=12)
    old = runpy.run_path(str(BASELINE))
    expected: list[Any] = []
    old["check_tautology"](deepcopy(value), expected)
    calls = []
    original = av._is_arc_generation_or_exploration_artifact

    def classify(d: dict[str, Any]) -> bool:
        calls.append(1)
        return original(d)

    monkeypatch.setattr(av, "_is_arc_generation_or_exploration_artifact", classify)
    actual: list[Any] = []
    av.check_tautology(value, actual)
    assert [f.to_dict() for f in actual] == [f.to_dict() for f in expected]
    assert len(calls) <= 1


def test_direct_helper_and_mutation_between_calls() -> None:
    """REQ-REPORT-9181-V782: no result survives a synchronous checker invocation."""
    value = payload()
    assert av._declared_arc_nondegenerate_firstwin_null_descriptor(
        value, "first_win_left", "first_win_right"
    )
    assert (
        av._declared_arc_nondegenerate_firstwin_null_descriptor(value, "metric_a", "metric_b")
        is None
    )
    before: list[Any] = []
    av.check_tautology(value, before)
    value["arms_non_degenerate"] = False
    after: list[Any] = []
    av.check_tautology(value, after)
    assert [f.to_dict() for f in before] != [f.to_dict() for f in after]


def test_normal_reader_does_not_enter_hashing_wrapper(monkeypatch: Any) -> None:
    """REQ-REPORT-9181-V782: normal validation must no longer hash per numeric pair."""
    from carnot.reporting import forward_evidence_runner_9111 as reader
    from carnot.reporting import validation_reuse_9167 as reuse

    monkeypatch.setattr(reuse, "canonical_hash", lambda v: pytest.fail("per_pair_hash"))
    value = dict(payload(), rows=[{}], qid_rows=[dict(denominator=3)])
    assert all(r["passed"] for r in reader.structural_controls(value))


def test_reducer_block_and_rehashed_readiness(tmp_path: Path) -> None:
    """REQ-REPORT-9181-V782: external absence blocks and rehashed promotion is refused."""
    from carnot.reporting import causal_validation_9181 as r

    work = r.prepare(tmp_path / "missing", tmp_path / "raw", "20261117")
    r.immutable(tmp_path / "raw/work.json", work)
    value = r.build(work)
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] == "missing_file"
    path = tmp_path / (r.NAME + ".json")
    r.immutable(path, value)
    assert r.replay(path)["passed"]
    value["causal_fix_ready_score"] = 1
    value["reproducibility_checksum"] = r.canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    path.chmod(0o600)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(path)


def test_probe_arms_have_parity_and_changed_counts() -> None:
    """REQ-REPORT-9181-V782: the correction changes actual work, not just a wrapper."""
    from carnot.reporting.validation_diagnostics_9181 import probe

    before, wrapper, after = [
        probe(arm, BASELINE) for arm in ["original", "v781_wrapper", "changed"]
    ]
    assert before["flags"] == wrapper["flags"] == after["flags"]
    assert before["full_scans"] > 1
    assert wrapper["hashes"] > before["numeric_pairs"]
    assert after["full_scans"] == 1 and after["hashes"] == 0


def test_exclusive_attempts_deadline_and_sanitized_startup(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9181-V782: real child operations survive removal of PYTHONPATH."""
    from carnot.reporting import validation_diagnostics_9181 as d

    argv = [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        "/tmp",
        sys.executable,
        str(ROOT / "scripts/experiments/experiment_9181_v782_causal_validation.py"),
        "--probe",
        "changed",
        "--probe-output",
        str(tmp_path / "probe.json"),
    ]
    receipt = d.run(
        "sanitized",
        argv,
        tmp_path,
        time.monotonic() + 30,
        extra_env={"CARNOT_9181_TRACE": str(tmp_path / "trace")},
    )
    assert receipt["passed"] and receipt["normal_exit"]
    assert Path(receipt["stdout_path"]).stat().st_mode & 0o222 == 0
    events = [
        json.loads(line)
        for p in (tmp_path / "trace").glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert events[0]["pid"] == receipt["pid"] and events[0]["ppid"] == os.getpid()
    assert events[0]["pythonpath"] is None and events[0]["argv"]
    assert events[-1]["counts"]["check_tautology"] == 1
    first = Path(receipt["stdout_path"]).read_bytes()
    second = d.run(
        "sanitized", [sys.executable, "-c", "print('second')"], tmp_path, time.monotonic() + 10
    )
    assert second["stdout_path"] != receipt["stdout_path"]
    assert Path(receipt["stdout_path"]).read_bytes() == first
    expired = d.run(
        "expired",
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path,
        time.monotonic() + 0.1,
    )
    assert expired["timed_out"] and not expired["passed"]


@pytest.fixture(scope="module")
def observed(tmp_path_factory: Any) -> dict[str, Any]:
    """SCENARIO-REPORT-9181-V782: keep real diagnostic children alive through their fixture."""
    from carnot.reporting import causal_validation_9181 as r

    raw = tmp_path_factory.mktemp("v782_diagnostic")
    work = r.prepare(r.ROOT, raw, "20261117")
    assert all(c["passed"] for c in work["checks"])
    r.diagnose(work, time.monotonic() + 900)
    return work


def test_real_diagnostics_and_refusals(observed: dict[str, Any], tmp_path: Path) -> None:
    """REQ-REPORT-9181-V782: actual sanitized traces cannot substitute for normal checks."""
    from carnot.reporting import causal_validation_9181 as r

    work = deepcopy(observed)
    assert len(work["structural_scan_rows"]) == 4
    assert not r.qualified(work)
    for key, changed in [("diagnostic_elapsed_s", 901), ("structural_scan_rows", [])]:
        altered = deepcopy(work)
        altered[key] = changed
        assert not r.qualified(altered)
    altered = deepcopy(work)
    altered["structural_scan_rows"][2]["full_scans"] = 2
    assert not r.qualified(altered)
    altered = deepcopy(work)
    altered["diagnostic_attempt_rows"][0]["passed"] = False
    assert not r.qualified(altered)
    altered = deepcopy(work)
    altered["structural_scan_rows"][0]["measured_duration_s"] += 1
    assert not r.qualified(altered)
    altered = deepcopy(work)
    altered["diagnostic_attempt_rows"][0]["trace_references"] = []
    assert not r.qualified(altered)
    altered = deepcopy(work)
    events = [
        json.loads(line)
        for line in r.bound(work["diagnostic_attempt_rows"][0]["trace_references"][0])
        .read_text()
        .splitlines()
    ]
    events[0]["pythonpath"] = "hidden_child"
    trace = tmp_path / "bad_trace.jsonl"
    trace.write_text("\n".join(json.dumps(e) for e in events))
    altered["diagnostic_attempt_rows"][0]["trace_references"] = [r.ref(trace)]
    assert not r.qualified(altered)


def test_real_verification_receipts_and_full_coverage(
    observed: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-REPORT-9181-V782: run real assertions, then reject changed commands or coverage."""
    from carnot.reporting import causal_validation_9181 as r

    source = tmp_path / "test_units.py"
    source.write_text(
        "import pytest\n@pytest.mark.parametrize('i',range(7))\ndef test_rehashed_primitive_tampering(i):\n assert i in range(7)\n"
    )
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "OWNED", [source.name])
    monkeypatch.setattr(r, "CHANGED", [])
    argv = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(tmp_path / "coverage.ini"),
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(tmp_path),
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        str(source),
    ]
    spec = dict(name="real", argv=argv)
    monkeypatch.setattr(r, "commands", lambda raw: [spec])
    work = dict(deepcopy(observed), raw=str(tmp_path), commands=[spec], validation_receipts=[])
    r.verify(work, time.monotonic() + 60)
    events = [
        json.loads(line)
        for line in r.bound(work["validation_receipts"][0]["events_reference"])
        .read_text()
        .splitlines()
    ]
    work["expected_nodes"] = work["expected_target_nodes"] = sorted({e["nodeid"] for e in events})
    assert r.qualified(work)
    for mutation in ["name", "argv", "count", "target", "phases", "coverage"]:
        changed = deepcopy(work)
        if mutation in {"name", "argv"}:
            changed["validation_receipts"][0][mutation] = "changed"
        elif mutation == "count":
            changed["commands"] = []
        elif mutation == "target":
            changed["expected_target_nodes"] = []
        elif mutation == "phases":
            changed["expected_nodes"].append("missing_assertion")
        else:
            changed["coverage"]["files"][0]["missing"] = [1]
        assert not r.qualified(changed)
    value = r.build(work)
    assert value["causal_fix_ready_score"] == 1
    changed = deepcopy(work)
    receipt = changed["validation_receipts"][0]
    saved = json.loads(r.bound(receipt["receipt_reference"]).read_bytes())
    saved["name"] = receipt["name"] = "rehashed_wrong_command"
    receipt["receipt_reference"] = r.immutable(tmp_path / "changed_receipt.json", saved)
    assert not r.qualified(changed)
    r.verify(deepcopy(work), time.monotonic() - 1)
    work["validation_receipts"][0]["passed"] = False
    assert r.build(work)["verdict_class"] == "disqualified"


def test_private_cli_success_refusal_tamper_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9181-V782: real CLIs retain isolation and enforce terminal evidence."""
    from carnot.reporting import causal_validation_9181 as r

    work = r.prepare(tmp_path / "absent", tmp_path / "raw", "20261117")
    workpath = tmp_path / "raw/work.json"
    r.immutable(workpath, work)
    output = tmp_path / (r.NAME + ".json")
    env = {"CARNOT_9181_TRACE": str(tmp_path / "cli_trace")}
    cases = [
        ("success", ["--from-work", str(workpath), "--output", str(output)], 0),
        ("cold", ["--replay", str(output)], 0),
        ("date_refusal", ["--date", "bad"], 1),
        ("missing", ["--replay", str(tmp_path / "absent.json")], 1),
        ("external", ["--from-work", str(workpath)], 1),
    ]
    for name, args, expected in cases:
        receipt = r.run(
            name,
            r.foreign_argv(r.CLI, *args),
            tmp_path / "logs",
            time.monotonic() + 60,
            expected=expected,
            extra_env=env,
        )
        assert receipt["passed"]
    value = json.loads(output.read_bytes())
    value["completed_count"] += 1
    value["reproducibility_checksum"] = r.canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    output.write_text(json.dumps(value))
    assert r.run(
        "tamper",
        r.foreign_argv(r.CLI, "--replay", str(output)),
        tmp_path / "logs",
        time.monotonic() + 60,
        expected=1,
    )["passed"]


def test_normal_main_paths(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9181-V782: normal execution retains its checks before checked publication."""
    from carnot.reporting import causal_validation_9181 as r

    assert r.main(["--date", "bad"]) == 1
    assert r.main(["--replay", str(tmp_path / "missing")]) == 1
    real_prepare = r.prepare
    diagnoses = []
    verified = []

    def prepare(root: Path, raw: Path, date: str) -> dict[str, Any]:
        work = real_prepare(tmp_path / "missing", raw, date)
        work["checks"] = []
        return work

    monkeypatch.setattr(r, "prepare", prepare)
    monkeypatch.setattr(r, "diagnose", lambda *a: diagnoses.append(a))
    monkeypatch.setattr(r, "verify", lambda *a: verified.append(a))
    monkeypatch.setattr(r, "run", lambda *a, **k: dict(passed=False))
    published = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, check: published.append(value) or {}
    )
    assert r.main(["--output", str(tmp_path / (r.NAME + ".json"))]) == 0
    assert len(diagnoses) == len(verified) == 1
    assert published[0]["causal_fix_ready_score"] == 0
    monkeypatch.setattr(r, "prepare", real_prepare)
    assert (
        r.main(
            [
                "--input-root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "second" / (r.NAME + ".json")),
            ]
        )
        == 0
    )
    assert published[-1]["verdict_class"] == "blocked"


def test_real_reader_private_cli_preserves_all_assertions(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9181-V782: original private reader success and rehashed refusals still pass."""
    from carnot.reporting import forward_evidence_runner_9111 as reader

    monkeypatch.setenv("CARNOT_9181_TRACE", str(tmp_path / "traces"))
    assert all(r["passed"] for r in reader.cli_controls(tmp_path))
    events = [
        json.loads(line)
        for path in (tmp_path / "traces").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    starts = [e for e in events if e["event"] == "start"]
    ends = [e for e in events if e["event"] == "summary"]
    assert len(starts) == len(ends) >= 5
    assert all(e["pid"] and e["ppid"] and e["argv"] for e in starts)
    sanitized = [e for e in starts if e["argv"][0].endswith(reader.NAME + ".py")]
    assert len(sanitized) == 5 and all(e["pythonpath"] is None for e in sanitized)
    assert all(e["ppid"] in {os.getpid(), *(s["pid"] for s in starts)} for e in starts)
    assert any(e["counts"].get("_verify_artifact_impl", 0) > 0 for e in ends)


def test_expired_supervision_handles_process_disappearance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-REPORT-9181-V782: process metadata races cannot leave an expired child alive."""
    from carnot.reporting import validation_diagnostics_9181 as d

    def disappeared(pid: int) -> Any:
        raise d.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(d.psutil, "Process", disappeared)
    receipt = d.run(
        "raced",
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path,
        time.monotonic() + 0.1,
    )
    assert receipt["timed_out"]


def test_timeout_reaps_child_in_another_session(tmp_path: Path) -> None:
    """REQ-REPORT-9181-V782: the common deadline reaches nested process groups."""
    from carnot.reporting import validation_diagnostics_9181 as d

    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True); print(p.pid,flush=True); time.sleep(20)"
    receipt = d.run("descendant", [sys.executable, "-c", code], tmp_path, time.monotonic() + 3)
    assert receipt["timed_out"]
    pid = int(Path(receipt["stdout_path"]).read_text())
    assert not d.psutil.pid_exists(pid) or d.psutil.Process(pid).status() == d.psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("observed", [0, True, "missing_field"])
def test_new_authority_gate_zero_boolean_and_missing(
    tmp_path: Path, monkeypatch: Any, observed: Any
) -> None:
    """REQ-REPORT-9181-V782: external gate operands remain typed and explicit."""
    from carnot.reporting import causal_validation_9181 as r

    path = tmp_path / "upstream.json"
    path.write_text(json.dumps({} if observed == "missing_field" else dict(ready=observed)))
    monkeypatch.setattr(r, "INPUTS", {path.name: (r.ref(path)["sha256"], "ready", 1)})
    work = r.prepare(tmp_path, tmp_path / "raw", "20261117")
    failed = [c for c in work["checks"] if c["artifact_field"] == "ready"]
    assert failed[0]["observed"] == observed and not failed[0]["passed"]


def test_monitor_callbacks_preserve_measured_counts(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9181-V782: monitor callbacks are checked directly as well as in real children."""
    from types import SimpleNamespace
    from carnot.reporting import validation_diagnostics_9181 as d

    callbacks: dict[int, Any] = {}
    finished = []
    monitor = SimpleNamespace(
        DISABLE=object(),
        events=SimpleNamespace(PY_START=1, PY_RETURN=2),
        use_tool_id=lambda *a: None,
        free_tool_id=lambda *a: None,
        set_events=lambda *a: None,
        set_local_events=lambda *a: None,
        register_callback=lambda tool, event, callback: callbacks.__setitem__(event, callback),
    )
    monkeypatch.setattr(
        d,
        "sys",
        SimpleNamespace(monitoring=monitor, argv=["callback_control"], _getframe=sys._getframe),
    )
    monkeypatch.setattr(d, "atexit", SimpleNamespace(register=finished.append))
    monkeypatch.setenv("CARNOT_9181_TRACE", str(tmp_path))
    d.startup()

    def check_tautology() -> None:
        code = sys._getframe().f_code
        callbacks[1](code, 0)
        callbacks[2](code, 0, None)

    check_tautology()
    assert (
        callbacks[1](test_monitor_callbacks_preserve_measured_counts.__code__, 0) is monitor.DISABLE
    )
    finished[0]()
    events = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert events[-1]["counts"]["check_tautology"] == 1
    assert events[-1]["elapsed_ns"]["check_tautology"] >= 0


def test_coverage_owns_only_new_shared_statements() -> None:
    """REQ-REPORT-9181-V782: retained source diffs prevent claiming coverage of old checker code."""
    from coverage import Coverage
    from carnot.reporting import causal_validation_9181 as r

    tracer = Coverage.current() or Coverage(config_file=False)
    rows = r.coverage_rows(tracer)
    assert [row["path"] for row in rows] == r.OWNED + r.CHANGED
    shared = next(row for row in rows if row["path"] == "scripts/adversarial_verify.py")
    assert 0 < len(shared["statements"]) < 20
    assert all(not row["excluded"] for row in rows)
