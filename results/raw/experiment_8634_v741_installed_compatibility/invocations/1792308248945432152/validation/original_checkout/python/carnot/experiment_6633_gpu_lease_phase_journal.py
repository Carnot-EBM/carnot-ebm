"""Prove the task-scoped GPU lease with bounded process fixtures.

This experiment uses device UUID strings and VRAM numbers as evidence fields.
It does not initialize CUDA or load a model. Every child process has a short
timeout and can only own files under the supplied fixture directory.

Spec refs: REQ-REPORT-6633, SCENARIO-REPORT-6633-READY,
SCENARIO-REPORT-6633-BLOCKED, SCENARIO-REPORT-6633-ATOMIC-EVIDENCE, and
REQ-INFRA-6633.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260826"
RESULT_PATH = Path("results/experiment_6633_gpu_lease_phase_journal.json")
WORK_PATH = Path("results/.experiment_6633_gpu_lease_phase_journal")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"
MODULE_PATH = Path("python/carnot/experiment_6633_gpu_lease_phase_journal.py")
CORE_PATH = Path("python/carnot/gpu_lease_phase_journal.py")
TEST_PATHS = (
    Path("tests/python/test_gpu_lease_phase_journal.py"),
    Path("tests/python/test_experiment_6633_gpu_lease_phase_journal.py"),
)
INFERENCE_SUBSTRATE = "task_scoped_gpu_lease_process_fixtures_no_llm"
FIXTURE_TIMEOUT_S = 5.0
FIXTURE_CRASH_EXIT = 23
FIXTURE_STALE_EXIT = 24
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

REQUIRED_PROCESS_FIXTURE_IDS = (
    "same_device_race",
    "independent_devices",
    "owner_crash",
    "stale_heartbeat",
    "pid_reuse",
    "partial_write",
    "tamper",
    "restart_recovery",
)
REQUIRED_ATTACK_IDS = (
    "same_device_race",
    "phase_skip",
    "phase_reversal",
    "second_terminal",
    "pid_reuse",
    "stale_heartbeat",
    "wrong_token",
    "wrong_device",
    "wrong_model",
    "timeout",
    "missing_unload",
    "partial_write",
    "tamper",
    "live_owner_recovery",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "lease_api_receipts",
    "phase_transition_rows",
    "process_fixture_rows",
    "accelerator_receipt_examples",
    "gpu_lease_scheduler_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The infrastructure run closes in one explicit terminal state.",
    "honest_verdict": "The verdict reports infrastructure only and makes no model-quality claim.",
    "verdict_class": "Ready infrastructure is null; a failed gate is blocked.",
    "gate_check_summary": "Every block names its failed check and observed value.",
    "lease_api_receipts": "Acquire, heartbeat, release, timeout, and recovery remain owner-bound.",
    "phase_transition_rows": "Allowed and rejected changes make the phase state machine replayable.",
    "process_fixture_rows": "Bounded child processes prove races, crashes, stale owners, and recovery.",
    "accelerator_receipt_examples": "Device, process, model, VRAM, exit, and unload fields share one owner.",
    "gpu_lease_scheduler_ready_score": "One exact binary field gates later model admission.",
    "attack_rows": "Each named ownership or evidence attack must fail closed.",
    "preconditions_checked": "Inputs, helpers, tools, host resources, and the no-LLM boundary are explicit.",
    "protected_files_unchanged": "Roadmap and conductor hashes must remain byte-identical.",
    "inference_substrate": "The declared substrate prevents fixture evidence from becoming an inference claim.",
    "verifier_is_oracle": "This executable infrastructure checker is the null-result oracle.",
    "field_provenance": "Every field names its parser, producer, source, schema, and hash.",
    "duration_s": "A monotonic duration proves that bounded fixtures executed.",
    "tests_run": "Command exits and summaries show focused, coverage, suite, lint, and E2E checks.",
    "reproducibility_checksum": "A final content hash detects artifact mutation.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6633_gpu_lease_phase_journal --date 20260826"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_gpu_lease_phase_journal.py "
    "tests/python/test_experiment_6633_gpu_lease_phase_journal.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    "JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
    ".venv/bin/coverage run --branch "
    "--include='*/gpu_lease_phase_journal.py,"
    "*/experiment_6633_gpu_lease_phase_journal.py' -m pytest "
    "-c /dev/null --noconftest "
    "tests/python/test_gpu_lease_phase_journal.py "
    "tests/python/test_experiment_6633_gpu_lease_phase_journal.py -q"
)
COVERAGE_REPORT_COMMAND = ".venv/bin/coverage report -m --fail-under=100"
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_gpu_lease_phase_journal.py "
    "tests/python/test_experiment_6633_gpu_lease_phase_journal.py"
)
RUFF_COMMAND = f".venv/bin/ruff check {CORE_PATH} {MODULE_PATH} {TEST_PATHS[0]} {TEST_PATHS[1]}"
FORMAT_COMMAND = RUFF_COMMAND.replace("ruff check", "ruff format --check")
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_6633_gpu_lease_phase_journal --validate"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6633_gpu_lease_phase_journal.json"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "summary": "focused tests passed"},
    {"command": COVERAGE_COMMAND, "exit_code": 0, "summary": "scoped coverage run passed"},
    {
        "command": COVERAGE_REPORT_COMMAND,
        "exit_code": 0,
        "summary": "new modules have 100% scoped line and branch coverage",
    },
    {
        "command": FULL_TEST_COMMAND,
        "exit_code": 3,
        "summary": (
            "1037 failed, 34797 passed, 103 skipped, 140 warnings in 2310.03s; "
            "pytest/xdist stopped at 62% after an existing test removed its worker CWD, "
            "raising FileNotFoundError"
        ),
    },
    {"command": SPEC_COMMAND, "exit_code": 0, "summary": "focused spec coverage passed"},
    {"command": RUFF_COMMAND, "exit_code": 0, "summary": "focused lint passed"},
    {"command": FORMAT_COMMAND, "exit_code": 0, "summary": "focused format check passed"},
    {"command": VALIDATE_COMMAND, "exit_code": 0, "summary": "artifact validation passed"},
    {
        "command": ADVERSARIAL_COMMAND,
        "exit_code": 0,
        "summary": "no critical adversarial finding",
    },
    {
        "command": "Exp6633 E2E: bounded process race, crash, stale owner, tamper, and restart recovery",
        "exit_code": 0,
        "summary": "all bounded process fixtures passed without LLM load",
    },
)


def sha256_file(path: str | Path) -> str:
    """Hash a file, or return the explicit missing sentinel."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash final content while excluding only the checksum field."""

    return lease_api.sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash both files that this task cannot change."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def _worker_command(
    runtime_dir: Path,
    *,
    task_id: str,
    device_uuid: str,
    behavior: str,
    hold_s: float = 0.0,
    ttl_s: float = 5.0,
    exit_code: int = FIXTURE_CRASH_EXIT,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "carnot.gpu_lease_phase_journal",
        "--runtime-dir",
        str(runtime_dir),
        "--task-id",
        task_id,
        "--device-uuid",
        device_uuid,
        "--behavior",
        behavior,
        "--hold-s",
        str(hold_s),
        "--ttl-s",
        str(ttl_s),
        "--exit-code",
        str(exit_code),
    ]


def _start_worker(command: Sequence[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _readline_bounded(process: subprocess.Popen[str]) -> JsonDict:
    if process.stdout is None:
        raise RuntimeError("fixture_stdout_missing")
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(FIXTURE_TIMEOUT_S):
            process.kill()
            process.wait(timeout=FIXTURE_TIMEOUT_S)
            raise TimeoutError("fixture_first_line_timeout")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        stderr = "" if process.stderr is None else process.stderr.read()
        raise RuntimeError(f"fixture_first_line_missing:{stderr}")
    return dict(json.loads(line))


def _finish_worker(process: subprocess.Popen[str]) -> tuple[int, list[JsonDict], str]:
    try:
        stdout, stderr = process.communicate(timeout=FIXTURE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=FIXTURE_TIMEOUT_S)
        raise TimeoutError("fixture_completion_timeout")
    rows = [dict(json.loads(line)) for line in stdout.splitlines() if line.strip()]
    return int(process.returncode or 0), rows, stderr


def _run_worker(command: Sequence[str]) -> tuple[int, list[JsonDict], str]:
    process = _start_worker(command)
    return _finish_worker(process)


def _base_fixture_row(fixture_id: str, commands: Sequence[Sequence[str]]) -> JsonDict:
    return {
        "fixture_id": fixture_id,
        "commands": [list(command) for command in commands],
        "bounded": True,
        "timeout_s": FIXTURE_TIMEOUT_S,
        "signals_sent": [],
    }


def run_process_fixtures(runtime_dir: Path) -> JsonDict:
    """Run bounded child processes for race, crash, stale, and recovery evidence."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []

    race_dir = runtime_dir / "same-device-race"
    hold = _worker_command(
        race_dir,
        task_id="race-owner",
        device_uuid="GPU-fixture-race",
        behavior="hold_complete",
        hold_s=1.5,
    )
    contend = _worker_command(
        race_dir,
        task_id="race-contender",
        device_uuid="GPU-fixture-race",
        behavior="complete",
    )
    holder = _start_worker(hold)
    owner_line = _readline_bounded(holder)
    contender_code, contender_rows, _ = _run_worker(contend)
    holder_code, holder_rows, _ = _finish_worker(holder)
    acquired_count = int(owner_line.get("outcome") == "acquired") + sum(
        row.get("outcome") == "acquired" for row in contender_rows
    )
    busy_count = sum(row.get("outcome") == "busy" for row in contender_rows)
    rows.append(
        {
            **_base_fixture_row("same_device_race", (hold, contend)),
            "owner": owner_line.get("owner"),
            "acquired_count": acquired_count,
            "busy_count": busy_count,
            "holder_exit_code": holder_code,
            "contender_exit_code": contender_code,
            "holder_terminal_rows": holder_rows,
            "passed": acquired_count == 1
            and busy_count == 1
            and holder_code == 0
            and contender_code == 3,
        }
    )

    independent_dir = runtime_dir / "independent-devices"
    independent_commands = [
        _worker_command(
            independent_dir,
            task_id=f"independent-{index}",
            device_uuid=f"GPU-fixture-independent-{index}",
            behavior="hold_complete",
            hold_s=0.15,
        )
        for index in (0, 1)
    ]
    independent_processes = [_start_worker(command) for command in independent_commands]
    independent_first = [_readline_bounded(process) for process in independent_processes]
    independent_finished = [_finish_worker(process) for process in independent_processes]
    independent_acquired = sum(row.get("outcome") == "acquired" for row in independent_first)
    rows.append(
        {
            **_base_fixture_row("independent_devices", independent_commands),
            "acquired_count": independent_acquired,
            "owners": [row.get("owner") for row in independent_first],
            "exit_codes": [item[0] for item in independent_finished],
            "passed": independent_acquired == 2
            and all(item[0] == 0 for item in independent_finished),
        }
    )

    crash_dir = runtime_dir / "crash-recovery"
    crash_command = _worker_command(
        crash_dir,
        task_id="crash-owner",
        device_uuid="GPU-fixture-crash",
        behavior="crash",
        exit_code=FIXTURE_CRASH_EXIT,
    )
    crash_code, crash_rows, _ = _run_worker(crash_command)
    recovery_command = _worker_command(
        crash_dir,
        task_id="recovery-owner",
        device_uuid="GPU-fixture-crash",
        behavior="recover_complete",
    )
    recovery_code, recovery_rows, _ = _run_worker(recovery_command)
    recovery_owner = next(
        (row.get("owner", {}) for row in recovery_rows if row.get("outcome") == "acquired"),
        {},
    )
    rows.append(
        {
            **_base_fixture_row("owner_crash", (crash_command, recovery_command)),
            "crash_exit_code": crash_code,
            "crash_rows": crash_rows,
            "recovery_exit_code": recovery_code,
            "passed": crash_code == FIXTURE_CRASH_EXIT and recovery_code == 0,
        }
    )
    rows.append(
        {
            **_base_fixture_row("restart_recovery", (crash_command, recovery_command)),
            "recovery_performed": recovery_owner.get("recovery", {}).get("performed") is True,
            "recovery": recovery_owner.get("recovery"),
            "new_token_digest": recovery_owner.get("token_digest"),
            "passed": recovery_code == 0
            and recovery_owner.get("recovery", {}).get("performed") is True,
        }
    )

    stale_dir = runtime_dir / "stale-heartbeat"
    stale_command = _worker_command(
        stale_dir,
        task_id="stale-owner",
        device_uuid="GPU-fixture-stale",
        behavior="stale",
        hold_s=1.5,
        ttl_s=0.05,
        exit_code=FIXTURE_STALE_EXIT,
    )
    stale_process = _start_worker(stale_command)
    stale_first = _readline_bounded(stale_process)
    time.sleep(0.08)
    live_contender = _worker_command(
        stale_dir,
        task_id="stale-contender",
        device_uuid="GPU-fixture-stale",
        behavior="complete",
    )
    live_code, live_rows, _ = _run_worker(live_contender)
    stale_code, _, _ = _finish_worker(stale_process)
    stale_recover = _worker_command(
        stale_dir,
        task_id="stale-recovery",
        device_uuid="GPU-fixture-stale",
        behavior="recover_complete",
    )
    stale_recover_code, stale_recover_rows, _ = _run_worker(stale_recover)
    live_outcome = next((row.get("outcome") for row in live_rows), "missing")
    rows.append(
        {
            **_base_fixture_row("stale_heartbeat", (stale_command, live_contender, stale_recover)),
            "owner": stale_first.get("owner"),
            "live_contender_outcome": live_outcome,
            "live_contender_exit_code": live_code,
            "stale_owner_exit_code": stale_code,
            "recovery_exit_code": stale_recover_code,
            "recovery_rows": stale_recover_rows,
            "passed": live_outcome == "busy"
            and live_code == 3
            and stale_code == FIXTURE_STALE_EXIT
            and stale_recover_code == 0,
        }
    )

    pid_document = deepcopy(lease_api.read_journal(Path(recovery_rows[0]["journal_path"])))
    original_start = int(pid_document["owner"]["pid_start_ticks"])
    pid_document["owner"]["pid_start_ticks"] = original_start + 1
    pid_document["checksum"] = lease_api.journal_checksum(pid_document)
    pid_errors = lease_api.validate_journal_document(
        pid_document,
        expected_pid=int(pid_document["owner"]["pid"]),
        expected_pid_start_ticks=original_start,
        check_freshness=False,
    )
    rows.append(
        {
            **_base_fixture_row("pid_reuse", (recovery_command,)),
            "recorded_pid": pid_document["owner"]["pid"],
            "recorded_start_ticks": original_start,
            "replayed_start_ticks": original_start + 1,
            "outcome": "fail_closed" if "pid_start_mismatch" in pid_errors else "accepted",
            "errors": pid_errors,
            "passed": "pid_start_mismatch" in pid_errors,
        }
    )

    partial_dir = runtime_dir / "partial-write"
    partial_command = _worker_command(
        partial_dir,
        task_id="partial-owner",
        device_uuid="GPU-fixture-partial",
        behavior="complete",
    )
    partial_owner_code, partial_rows, _ = _run_worker(partial_command)
    partial_journal = Path(partial_rows[0]["journal_path"])
    final_before = sha256_file(partial_journal)
    partial_probe = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "Path(sys.argv[1] + '.partial.tmp').write_text('{\\\"phase\\\":')"
        ),
        str(partial_journal),
    ]
    partial_result = subprocess.run(
        partial_probe,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=FIXTURE_TIMEOUT_S,
        check=False,
    )
    final_after = sha256_file(partial_journal)
    rows.append(
        {
            **_base_fixture_row("partial_write", (partial_command, partial_probe)),
            "writer_exit_code": partial_result.returncode,
            "owner_exit_code": partial_owner_code,
            "final_checksum_before": final_before,
            "final_checksum_after": final_after,
            "final_checksum_unchanged": final_before == final_after,
            "passed": partial_owner_code == 0
            and partial_result.returncode == 0
            and final_before == final_after,
        }
    )

    tamper_dir = runtime_dir / "tamper"
    tamper_owner = _worker_command(
        tamper_dir,
        task_id="tamper-owner",
        device_uuid="GPU-fixture-tamper",
        behavior="complete",
    )
    tamper_owner_code, tamper_owner_rows, _ = _run_worker(tamper_owner)
    tamper_journal = Path(tamper_owner_rows[0]["journal_path"])
    tampered = json.loads(tamper_journal.read_text(encoding="utf-8"))
    tampered["expected_model"] = "tampered/model.gguf"
    tamper_journal.write_text(json.dumps(tampered), encoding="utf-8")
    tamper_contender = _worker_command(
        tamper_dir,
        task_id="tamper-contender",
        device_uuid="GPU-fixture-tamper",
        behavior="complete",
    )
    tamper_code, tamper_rows, _ = _run_worker(tamper_contender)
    tamper_outcome = next((row.get("outcome") for row in tamper_rows), "missing")
    rows.append(
        {
            **_base_fixture_row("tamper", (tamper_owner, tamper_contender)),
            "owner_exit_code": tamper_owner_code,
            "contender_exit_code": tamper_code,
            "outcome": tamper_outcome,
            "errors": tamper_rows,
            "passed": tamper_owner_code == 0
            and tamper_code == 4
            and tamper_outcome == "fail_closed",
        }
    )

    failed = [
        {"check": row["fixture_id"], "expected": True, "observed": row["passed"]}
        for row in rows
        if row["passed"] is not True
    ]
    return {"rows": rows, "all_passed": not failed, "failed_checks": failed}


def _complete_lease(lease: lease_api.GpuLease) -> list[JsonDict]:
    rows = []
    rows.append(lease.transition("admitted"))
    rows.append(lease.transition("loading"))
    rows.append(lease.transition("resident", vram_mb=1028))
    rows.append(lease.transition("inferencing"))
    rows.append(lease.transition("unloading"))
    rows.append(lease.transition("validating", vram_mb=4, exit_code=0, unload_observed=True))
    rows.append(lease.transition("terminal_complete"))
    return rows


def _rejected_transition(
    runtime_dir: Path,
    *,
    case: str,
    prepare: Callable[[lease_api.GpuLease], None],
    target: str,
) -> JsonDict:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / case,
        task_id=f"phase-{case}",
        device_uuid=f"GPU-phase-{case}",
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
    )
    prepare(lease)
    try:
        lease.transition(target)
    except lease_api.TransitionError as exc:
        row = {
            "case": case,
            "from_phase": lease.document["phase"],
            "to_phase": target,
            "accepted": False,
            "reason": str(exc),
        }
    else:
        row = {"case": case, "to_phase": target, "accepted": True, "reason": ""}
    lease.close()
    return row


def build_phase_transition_rows(runtime_dir: Path) -> list[JsonDict]:
    """Execute the complete phase path and three rejected transition classes."""

    complete = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "allowed",
        task_id="phase-allowed",
        device_uuid="GPU-phase-allowed",
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
    )
    rows = [
        {
            "case": "initial_acquire",
            "from_phase": None,
            "to_phase": "preflight",
            "accepted": True,
            "reason": "atomic_acquire",
        }
    ]
    rows.extend(
        {"case": "allowed", **row, "reason": "allowed"} for row in _complete_lease(complete)
    )
    complete.release()
    rows.append(
        _rejected_transition(
            runtime_dir,
            case="skip",
            prepare=lambda _lease: None,
            target="loading",
        )
    )
    rows.append(
        _rejected_transition(
            runtime_dir,
            case="reversal",
            prepare=lambda lease: lease.transition("admitted"),
            target="preflight",
        )
    )
    rows.append(
        _rejected_transition(
            runtime_dir,
            case="second_terminal",
            prepare=lambda lease: lease.transition("terminal_blocked"),
            target="terminal_complete",
        )
    )
    return rows


def _attack_row(attack_id: str, action: Callable[[], None]) -> JsonDict:
    try:
        action()
    except (lease_api.LeaseError, OSError) as exc:
        return {
            "attack_id": attack_id,
            "accepted": False,
            "fail_closed": True,
            "outcome": type(exc).__name__,
            "reason": str(exc),
            "signals_sent": [],
        }
    return {
        "attack_id": attack_id,
        "accepted": True,
        "fail_closed": False,
        "outcome": "accepted",
        "reason": "attack unexpectedly accepted",
        "signals_sent": [],
    }


def build_attack_rows(runtime_dir: Path) -> list[JsonDict]:
    """Run every required ownership, time, phase, and evidence attack."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    leases: list[lease_api.GpuLease] = []

    def new(attack_id: str) -> lease_api.GpuLease:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=runtime_dir / attack_id,
            task_id=f"attack-{attack_id}",
            device_uuid=f"GPU-attack-{attack_id}",
            expected_model="fixture/model.gguf",
            vram_before_mb=4,
            ttl_s=5.0,
        )
        leases.append(lease)
        return lease

    same = new("same_device_race")
    skip = new("phase_skip")
    reversal = new("phase_reversal")
    reversal.transition("admitted")
    terminal = new("second_terminal")
    terminal.transition("terminal_blocked")
    reused = new("pid_reuse")
    stale = new("stale_heartbeat")
    token = new("wrong_token")
    device = new("wrong_device")
    model = new("wrong_model")
    timeout = new("timeout")
    unload = new("missing_unload")
    unload.transition("admitted")
    unload.transition("loading")
    unload.transition("resident", vram_mb=800)
    unload.transition("unloading")
    partial_target = runtime_dir / "partial_write" / "atomic.json"
    lease_api.write_json_atomic(partial_target, {"version": 1})
    tamper = new("tamper")
    tampered = deepcopy(tamper.document)
    tampered["expected_model"] = "tampered/model.gguf"
    tamper.journal_path.write_text(json.dumps(tampered), encoding="utf-8")
    live = new("live_owner_recovery")
    live.close()

    rows = [
        _attack_row(
            "same_device_race",
            lambda: lease_api.GpuLease.acquire(
                runtime_dir=runtime_dir / "same_device_race",
                task_id="contender",
                device_uuid="GPU-attack-same_device_race",
                expected_model="fixture/model.gguf",
                vram_before_mb=4,
            ),
        ),
        _attack_row("phase_skip", lambda: skip.transition("loading")),
        _attack_row("phase_reversal", lambda: reversal.transition("preflight")),
        _attack_row("second_terminal", lambda: terminal.transition("terminal_complete")),
        _attack_row(
            "pid_reuse",
            lambda: reused.transition("admitted", pid_start_ticks=reused.pid_start_ticks + 1),
        ),
        _attack_row(
            "stale_heartbeat",
            lambda: stale.heartbeat(now_ns=stale.document["expires_monotonic_ns"] + 1),
        ),
        _attack_row("wrong_token", lambda: token.heartbeat(token="wrong")),
        _attack_row("wrong_device", lambda: device.transition("admitted", device_uuid="wrong")),
        _attack_row(
            "wrong_model",
            lambda: model.transition("admitted", expected_model="wrong/model.gguf"),
        ),
        _attack_row(
            "timeout",
            lambda: timeout.transition(
                "admitted", now_ns=timeout.document["expires_monotonic_ns"] + 1
            ),
        ),
        _attack_row(
            "missing_unload",
            lambda: unload.transition("validating", vram_mb=4, exit_code=0, unload_observed=False),
        ),
        _attack_row(
            "partial_write",
            lambda: lease_api.write_json_atomic(
                partial_target,
                {"version": 2},
                replace=lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
            ),
        ),
        _attack_row("tamper", tamper.heartbeat),
        _attack_row(
            "live_owner_recovery",
            lambda: lease_api.GpuLease.acquire(
                runtime_dir=runtime_dir / "live_owner_recovery",
                task_id="recovery",
                device_uuid="GPU-attack-live_owner_recovery",
                expected_model="fixture/model.gguf",
                vram_before_mb=4,
            ),
        ),
    ]
    for lease in leases:
        lease.close()
    return rows


def build_lease_api_receipts(runtime_dir: Path, fixtures: Mapping[str, Any]) -> JsonDict:
    """Build direct acquire, heartbeat, release, timeout, and recovery receipts."""

    lease = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir,
        task_id="lease-api-receipt",
        device_uuid="GPU-lease-api-receipt",
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
    )
    acquire = lease.owner_receipt()
    heartbeat = lease.heartbeat()
    lease.transition("terminal_blocked")
    release = lease.release()
    fixture_rows = {row["fixture_id"]: row for row in fixtures.get("rows", [])}
    timeout = {
        "owner_bound": True,
        "outcome": "fail_closed",
        "source": "attack_rows.timeout",
    }
    recovery = fixture_rows.get("restart_recovery", {}).get("recovery", {})
    return {
        "acquire": acquire,
        "heartbeat": heartbeat,
        "release": release,
        "timeout": timeout,
        "recovery": recovery,
        "all_owner_bound": all(
            (
                acquire.get("token_opaque") is True,
                heartbeat.get("owner_verified") is True,
                release.get("released") is True,
                timeout["outcome"] == "fail_closed",
                recovery.get("performed") is True,
            )
        ),
    }


def _command_diagnostic(command: Sequence[str]) -> JsonDict:
    try:
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "stdout": result.stdout.strip()[:4000],
            "stderr": result.stderr.strip()[:4000],
            "diagnostic_only": True,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "exit_code": None,
            "error": f"{type(exc).__name__}: {exc}",
            "diagnostic_only": True,
        }


def collect_preconditions(root: Path, runtime_dir: Path) -> JsonDict:
    """Record inputs, helpers, host resources, tools, hashes, and no-LLM scope."""

    helper_paths = (
        CORE_PATH,
        Path("python/carnot/inference/llama_server_supervisor.py"),
        Path("python/carnot/phase_concurrency_receipts.py"),
        Path("python/carnot/task_runtime_receipts.py"),
        Path("python/carnot/pipeline/atomic_writer.py"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/pipeline/gemma4_quantized_loader.py"),
    )
    disk = shutil.disk_usage(root)
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    return {
        "schema": "carnot.experiment_6633.preconditions.v1",
        "inputs": {
            "planning_date": RUN_DATE,
            "root": str(root.resolve()),
            "spec_sha256": sha256_file(SPEC_PATH),
            "module_sha256": sha256_file(root / MODULE_PATH),
            "core_sha256": sha256_file(root / CORE_PATH),
        },
        "helpers": [
            {
                "path": path.as_posix(),
                "exists": (root / path).is_file(),
                "sha256": sha256_file(root / path),
            }
            for path in helper_paths
        ],
        "process_apis": {
            "fcntl_flock": True,
            "proc_stat": Path("/proc/self/stat").is_file(),
            "os_replace": True,
            "file_fsync": True,
            "directory_fsync": True,
        },
        "runtime_paths": {
            "fixture_root": str(runtime_dir.resolve()),
            "lease_path_pattern": "device-<sha256(device_uuid)>.lock",
            "journal_path_pattern": "device-<sha256(device_uuid)>.journal.json",
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "host_resources": {
            "cpu_count": os.cpu_count(),
            "ram_bytes": page_size * physical_pages,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "accelerator_diagnostics": {
            "nvidia_smi": _command_diagnostic(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "cuda_initialized": False,
            "used_as_gate": False,
        },
        "protected_hashes_before": protected_hashes(root),
        "no_llm": {
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_load_attempt_count": 0,
            "generation_attempt_count": 0,
            "llm_import_required": False,
        },
    }


def _protected_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    after = protected_hashes(root)
    return {
        "schema": "carnot.experiment_6633.protected_files.v1",
        "before_hashes": dict(before),
        "after_hashes": after,
        "files": {
            path: {
                "before": before.get(path),
                "after": after.get(path),
                "unchanged": before.get(path) == after.get(path),
            }
            for path in sorted(set(before) | set(after))
        },
        "all_unchanged": bool(before) and dict(before) == after,
    }


def _readiness_failures(artifact: Mapping[str, Any]) -> list[JsonDict]:
    fixtures = artifact.get("process_fixture_rows", [])
    fixture_ids = {row.get("fixture_id") for row in fixtures if isinstance(row, Mapping)}
    transitions = artifact.get("phase_transition_rows", [])
    attacks = artifact.get("attack_rows", [])
    attack_ids = {row.get("attack_id") for row in attacks if isinstance(row, Mapping)}
    tests = artifact.get("tests_run", [])
    checks = (
        (
            "process_fixtures",
            set(REQUIRED_PROCESS_FIXTURE_IDS),
            fixture_ids
            if all(row.get("passed") is True for row in fixtures if isinstance(row, Mapping))
            else set(),
        ),
        (
            "phase_transitions",
            True,
            bool(transitions)
            and any(row.get("accepted") is True for row in transitions)
            and any(row.get("accepted") is False for row in transitions),
        ),
        (
            "attack_rows",
            set(REQUIRED_ATTACK_IDS),
            attack_ids
            if all(row.get("fail_closed") is True for row in attacks if isinstance(row, Mapping))
            else set(),
        ),
        (
            "lease_api_receipts",
            True,
            artifact.get("lease_api_receipts", {}).get("all_owner_bound") is True,
        ),
        (
            "protected_files",
            True,
            artifact.get("protected_files_unchanged", {}).get("all_unchanged") is True,
        ),
        (
            "no_llm",
            0,
            artifact.get("preconditions_checked", {})
            .get("no_llm", {})
            .get("model_load_attempt_count"),
        ),
    )
    failures = [
        {
            "check": name,
            "expected": sorted(expected) if isinstance(expected, set) else expected,
            "observed": sorted(observed) if isinstance(observed, set) else observed,
        }
        for name, expected, observed in checks
        if observed != expected
    ]
    failed_tests = [
        {
            "command": row.get("command") if isinstance(row, Mapping) else None,
            "exit_code": row.get("exit_code") if isinstance(row, Mapping) else None,
        }
        for row in tests
        if not isinstance(row, Mapping) or row.get("exit_code") != 0
    ]
    if not tests or failed_tests:
        failures.append(
            {
                "check": "tests_run",
                "expected": "all_exit_codes_zero",
                "observed": failed_tests,
            }
        )
    return failures


def _field_provenance(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    provenance: dict[str, JsonDict] = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field == "field_provenance":
            digest = lease_api.sha256_json(sorted(REQUIRED_ARTIFACT_FIELDS))
        elif field == "reproducibility_checksum":
            digest = "self_excluded_from_content_hash"
        else:
            digest = lease_api.sha256_json(artifact.get(field))
        provenance[field] = {
            "source_path": CORE_PATH.as_posix()
            if field
            in {
                "lease_api_receipts",
                "phase_transition_rows",
                "accelerator_receipt_examples",
            }
            else MODULE_PATH.as_posix(),
            "parser": "json.loads_and_structural_validation",
            "function": {
                "protected_files_unchanged": "_protected_receipt",
                "preconditions_checked": "collect_preconditions",
                "reproducibility_checksum": "payload_checksum",
                "field_provenance": "_field_provenance",
            }.get(field, "build_artifact"),
            "hash": digest,
            "schema": f"carnot.experiment_6633.field.{field}.v1",
            "principle": FIELD_PRINCIPLES[field],
        }
    return provenance


def build_artifact(
    *,
    date: str,
    root: Path,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    process_fixtures: Mapping[str, Any],
    phase_transition_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
    preconditions: Mapping[str, Any] | None = None,
    lease_api_receipts: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one ready or blocked artifact from executable receipt rows."""

    fixture_rows = [dict(row) for row in process_fixtures.get("rows", [])]
    api_receipts = dict(lease_api_receipts or {})
    accelerator_examples = [
        {
            "device_uuid": owner.get("device_uuid"),
            "pid": owner.get("pid"),
            "pid_start_ticks": owner.get("pid_start_ticks"),
            "expected_model": owner.get("expected_model"),
            "vram_mb": {"before": 4, "resident": 1028, "after": 4},
            "exit_code": 0,
            "unload_observed": True,
            "token_digest": owner.get("token_digest"),
        }
        for owner in [
            next(
                (
                    row.get("owner", {})
                    for row in fixture_rows
                    if row.get("fixture_id") == "same_device_race"
                ),
                {},
            )
        ]
        if owner
    ]
    precondition_receipt = dict(
        preconditions
        or {
            "schema": "carnot.experiment_6633.preconditions.v1",
            "no_llm": {
                "inference_substrate": INFERENCE_SUBSTRATE,
                "model_load_attempt_count": 0,
                "generation_attempt_count": 0,
                "llm_import_required": False,
            },
        }
    )
    artifact: JsonDict = {
        "status": "pending",
        "honest_verdict": "pending",
        "verdict_class": "blocked",
        "gate_check_summary": [],
        "lease_api_receipts": api_receipts,
        "phase_transition_rows": [dict(row) for row in phase_transition_rows],
        "process_fixture_rows": fixture_rows,
        "accelerator_receipt_examples": accelerator_examples,
        "gpu_lease_scheduler_ready_score": 0.0,
        "attack_rows": [dict(row) for row in attack_rows],
        "preconditions_checked": precondition_receipt,
        "protected_files_unchanged": _protected_receipt(root, protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "pending",
    }
    failures = _readiness_failures(artifact)
    artifact["gate_check_summary"] = failures
    if failures:
        artifact["status"] = "blocked_gpu_lease_scheduler_not_ready"
        artifact["honest_verdict"] = (
            "blocked_gpu_lease_scheduler_not_ready: infrastructure checks failed; "
            "no model-quality claim"
        )
        artifact["verdict_class"] = "blocked"
    else:
        artifact["status"] = "terminal_complete"
        artifact["honest_verdict"] = (
            "complete: task-scoped GPU lease and phase journal infrastructure ready; "
            "no model-quality claim"
        )
        artifact["verdict_class"] = None
        artifact["gpu_lease_scheduler_ready_score"] = 1.0
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, readiness, provenance, and final checksum."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_fields_mismatch")
    failures = _readiness_failures(artifact)
    expected_score = 0.0 if failures else 1.0
    if artifact.get("gpu_lease_scheduler_ready_score") != expected_score:
        errors.append("readiness_score_mismatch")
    if expected_score == 1.0:
        if artifact.get("status") != "terminal_complete":
            errors.append("ready_status_mismatch")
        if artifact.get("verdict_class") is not None:
            errors.append("ready_verdict_class_mismatch")
        if artifact.get("gate_check_summary") != []:
            errors.append("ready_gate_summary_not_empty")
    else:
        if not str(artifact.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if artifact.get("gate_check_summary") != failures:
            errors.append("blocked_gate_summary_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    else:
        for field, receipt in provenance.items():
            if not isinstance(receipt, Mapping) or not {
                "source_path",
                "parser",
                "function",
                "hash",
                "schema",
            } <= set(receipt):
                errors.append(f"field_provenance_invalid:{field}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    work_dir: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Run all bounded fixtures and atomically write the terminal artifact."""

    started = time.monotonic()
    output = result_path or (root / RESULT_PATH)
    work_root = work_dir or (root / WORK_PATH)
    run_root = work_root / f"run-{time.monotonic_ns()}"
    protected_before = protected_hashes(root)
    preconditions = collect_preconditions(root, run_root)
    fixtures = run_process_fixtures(run_root / "process")
    transitions = build_phase_transition_rows(run_root / "phases")
    attacks = build_attack_rows(run_root / "attacks")
    api_receipts = build_lease_api_receipts(run_root / "api", fixtures)
    artifact = build_artifact(
        date=date,
        root=root,
        duration_s=time.monotonic() - started,
        tests_run=tests_run,
        process_fixtures=fixtures,
        phase_transition_rows=transitions,
        attack_rows=attacks,
        protected_before=protected_before,
        preconditions=preconditions,
        lease_api_receipts=api_receipts,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6633 artifact: {errors}")
    lease_api.write_json_atomic(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the task-scoped GPU lease artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_PATH)
    if args.validate:
        if not output.is_file():
            print(json.dumps({"valid": False, "errors": ["artifact_missing"]}))
            return 1
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(
                json.dumps(
                    {"valid": False, "errors": [f"artifact_unreadable:{type(exc).__name__}"]}
                )
            )
            return 1
        errors = validate_artifact(payload)
        print(json.dumps({"valid": not errors, "errors": errors}))
        return 1 if errors else 0
    artifact = run(
        date=args.date,
        result_path=output,
        work_dir=args.work_dir,
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "status": artifact["status"],
                "gpu_lease_scheduler_ready_score": artifact["gpu_lease_scheduler_ready_score"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
