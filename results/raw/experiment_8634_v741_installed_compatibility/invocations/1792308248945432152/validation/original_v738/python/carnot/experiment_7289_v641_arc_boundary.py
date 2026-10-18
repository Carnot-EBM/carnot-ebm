"""Diagnose and repair ARC inference accounting at process boundaries.

This experiment invokes no model. It uses CPU child processes to prove that a
parent can recover model-shaped lifecycle events after its child disappears.
Historical model facts remain in hashed sidecars and never enter current-task
inference metadata.

Spec: REQ-ARC-WMTE-7289 and SCENARIO-ARC-WMTE-7289-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch

from carnot.agentic.arc_inference_boundary import (
    BOUNDARY_LEDGER_ENV,
    InvocationBoundaryLedger,
    reduce_boundary_events,
)

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7289-arc-boundary"
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
SCHEMA = "carnot.experiment_7289.v641.arc_boundary.v1"
RANDOM_SEED = {"development": 7_289_202_609_14, "independent_evaluation": 17_289_202_609_14}

MODULE_PATH = Path("python/carnot/experiment_7289_v641_arc_boundary.py")
BOUNDARY_MODULE_PATH = Path("python/carnot/agentic/arc_inference_boundary.py")
WORLD_MODEL_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
TOOL_LOOP_PATH = Path("python/carnot/agentic/arc_induction_tool_loop.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7289_v641_arc_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_7289_v641_arc_boundary.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
RESULT_PATH = Path("results/experiment_7289_v641_arc_boundary.json")
RAW_DIR = Path("results/raw/experiment_7289_v641_arc_boundary")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7289_v641_arc_boundary.json")
HISTORICAL_PATH = Path("results/experiment_7280_v640_arc_live.json")
HISTORICAL_RAW_DIR = Path("results/raw/experiment_7280_v640_arc_live")
HISTORICAL_CHECKPOINT = Path("results/checkpoints/experiment_7280_v640_arc_live.json")

MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    Path("python/carnot/experiment_7280_v640_arc_live.py"),
    Path("python/carnot/experiment_7276_v640_arc_identity.py"),
    Path("python/carnot/experiment_7263_v639_arc_live.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    HISTORICAL_PATH,
    Path("tests/python/test_arc_eval_provenance_contract_20260905.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    SPEC_PATH,
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Full generation60s, bounded10s, load-only2s, or actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual GPU/native/device execution separately.",
    "duration_s": "Measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Every comparative unit/arm/seed with metric, cost, error, abstention and censoring; no aggregate-only claim.",
    "sample_size_budget": "Planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness/value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked_* verdict names upstream/check, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_ or complete:; external absence starts blocked_; state the actual finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed efficacy gates forbid positive. Only own unfinished work is partial; unchanged external failure is terminal blocked.",
    "validation_receipts": "Command, exit code, elapsed time and log hash; preserve actual failures.",
    "arc_boundary_ready_score": "One only when the real live caller uses the tested lossless boundary ledger and identity checks.",
    "boundary_event_rows": "Each injected lifecycle event, persisted evidence, counter reduction and expected result.",
    "historical_timeout_diagnosis": "Observed last event and unresolved cause, linked to immutable Exp7280 evidence.",
    "live_handoff": "Actual caller/arguments, changed code hashes, cleanup ownership and existing budget constants.",
    "quarantine_preserved": "Confirm historical Exp7280 remains unchanged and excluded from scientific claims.",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Seal all fields except the field that carries the seal."""
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(_canonical_bytes(payload))


def atomic_write(path: Path, value: Any) -> None:
    """Publish JSON through a same-directory rename after bytes reach storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _progress(  # pragma: no cover - exercised by the CLI experiment runner
    started: float, phase: str, event: str, **fields: Any
) -> None:
    payload = {
        "experiment": EXPERIMENT_ID,
        "phase": phase,
        "event": event,
        "elapsed_s": round(time.monotonic() - started, 6),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _fixture_identity(*, valid: bool = True) -> JsonDict:
    return {
        "model_repository": "historical-fixture/model",
        "model_filename": "fixture.gguf",
        "model_revision": "a" * 40 if valid else "",
        "model_path": "/injected-history/fixture.gguf",
        "model_hash": "sha256:" + "b" * 64,
    }


def _fixture_child(  # pragma: no cover - exercised in owned fixture subprocesses
    case: str, ledger_path: Path, marker_path: Path | None
) -> int:
    """Emit model-shaped history without invoking a model in this experiment."""
    print(json.dumps({"phase": "fixture_child", "event": "begin", "case": case}), flush=True)
    ledger = InvocationBoundaryLedger(ledger_path)
    if case == "pre_load_failure":
        call = ledger.begin("model_load", _fixture_identity(valid=False), call_id="pre-load")
        call.fail("injected pre-load failure")
    elif case == "load_no_generation":
        call = ledger.begin("model_load", _fixture_identity(), call_id="load-only")
        call.child_started(os.getpid())
        call.complete()
    elif case == "generation_timeout":
        load = ledger.begin("model_load", _fixture_identity(), call_id="timeout-load")
        load.child_started(os.getpid())
        load.complete()
        ledger.begin(
            "generation",
            _fixture_identity(),
            child_pid=os.getpid(),
            call_id="timeout-generation",
        )
        print(json.dumps({"phase": "generation", "event": "in_flight"}), flush=True)
        signal.pause()
    elif case == "generation_unusable":
        load = ledger.begin("model_load", _fixture_identity(), call_id="unusable-load")
        load.child_started(os.getpid())
        load.complete()
        generation = ledger.begin(
            "generation",
            _fixture_identity(),
            child_pid=os.getpid(),
            call_id="unusable-generation",
        )
        generation.complete(usable=False)
    elif case == "duplicate_events":
        generation = ledger.begin(
            "generation",
            _fixture_identity(),
            child_pid=os.getpid(),
            call_id="duplicate-generation",
        )
        generation.complete(usable=False)
        events = ledger.read_events()
        ledger.append_event(events[0])
        ledger.append_event(events[1])
    elif case == "orphan_cleanup":
        load = ledger.begin("model_load", _fixture_identity(), call_id="cleanup-load")
        load.child_started(os.getpid())
        load.complete()
        grandchild = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
        if marker_path is not None:
            atomic_write(marker_path, {"owned_child_pids": [grandchild.pid]})

        def terminate_owned_child(_signum: int, _frame: Any) -> None:
            grandchild.terminate()
            try:
                grandchild.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                grandchild.kill()
                grandchild.wait(timeout=0.5)
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, terminate_owned_child)
        print(
            json.dumps(
                {"phase": "orphan_cleanup", "event": "child_started", "pid": grandchild.pid}
            ),
            flush=True,
        )
        signal.pause()
    else:
        raise ValueError(f"unknown fixture case: {case}")
    print(json.dumps({"phase": "fixture_child", "event": "end", "case": case}), flush=True)
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (OSError, IndexError):  # pragma: no cover - procfs disappearance race
        return False
    return state != "Z"


def _run_fixture(case: str, directory: Path) -> JsonDict:
    ledger_path = directory / f"{case}.jsonl"
    marker_path = directory / f"{case}_owned_children.json"
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / WRAPPER_PATH),
        "--role",
        "fixture-child",
        "--case",
        case,
        "--ledger-path",
        str(ledger_path),
        "--marker-path",
        str(marker_path),
    ]
    started = time.monotonic()
    print(json.dumps({"phase": "fixture", "event": "before_subprocess", "case": case}), flush=True)
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
    if case in {"generation_timeout", "orphan_cleanup"}:
        ready_deadline = time.monotonic() + 5.0
        while process.poll() is None and time.monotonic() < ready_deadline:
            if case == "orphan_cleanup" and marker_path.is_file():
                break
            if case == "generation_timeout" and ledger_path.is_file():
                current = reduce_boundary_events(
                    InvocationBoundaryLedger(ledger_path).read_events()
                )
                current_counts = current.get("invocation_counts") or {}
                if current_counts.get("generation_calls_attempted") == 1:
                    break
            time.sleep(0.01)
        timeout_s = 0.1
    else:
        timeout_s = 5.0
    timed_out = False
    signals_sent: list[str] = []
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        signals_sent.append("SIGTERM:owned_process_group")
        try:
            process.wait(timeout=0.75)
        except subprocess.TimeoutExpired:  # pragma: no cover - hostile-child fallback
            os.killpg(process.pid, signal.SIGKILL)
            signals_sent.append("SIGKILL:owned_process_group")
            process.wait(timeout=0.75)
    owned_pids: list[int] = []
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        owned_pids = [int(pid) for pid in marker.get("owned_child_pids", [])]
    deadline = time.monotonic() + 1.0
    while any(_pid_alive(pid) for pid in owned_pids) and time.monotonic() < deadline:
        time.sleep(0.01)  # pragma: no cover - timing-dependent cleanup poll
    alive = process.poll() is None or any(_pid_alive(pid) for pid in owned_pids)
    ledger = InvocationBoundaryLedger(ledger_path)
    events = ledger.read_events()
    reduced = reduce_boundary_events(events)
    print(
        json.dumps(
            {
                "phase": "fixture",
                "event": "after_subprocess",
                "case": case,
                "timed_out": timed_out,
                "elapsed_s": round(time.monotonic() - started, 6),
            }
        ),
        flush=True,
    )
    return {
        "case": case,
        "command": command,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "signals_sent": signals_sent,
        "duration_s": round(time.monotonic() - started, 6),
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256_file(ledger_path),
        "event_count": len(events),
        "reduction": reduced,
        "owned_child_pids": owned_pids,
        "owned_child_alive_after_cleanup": alive,
    }


def _fixture_passed(row: Mapping[str, Any]) -> tuple[bool, JsonDict, JsonDict, bool]:
    case = str(row["case"])
    reduced = row["reduction"]
    counts = reduced.get("invocation_counts")
    expected: JsonDict
    observed: JsonDict = dict(counts or {})
    identity_rejection = False
    if case == "pre_load_failure":
        expected = {"disqualified": True, "model_revision_missing": True}
        identity_rejection = any(
            str(error).startswith("model_revision_missing:") for error in reduced.get("errors", [])
        )
        passed = reduced.get("disqualified") is True and identity_rejection
        observed = {
            "disqualified": reduced.get("disqualified"),
            "errors": reduced.get("errors"),
        }
    elif case == "load_no_generation":
        expected = {"model_loads_completed": 1, "generation_calls_attempted": 0}
        passed = bool(
            counts
            and counts["model_loads_completed"] == 1
            and counts["generation_calls_attempted"] == 0
            and reduced.get("inference_substrate") == "model_load_no_generation"
        )
    elif case == "generation_timeout":
        expected = {"generation_calls_attempted": 1, "generation_calls_completed": 0}
        passed = bool(
            row.get("timed_out")
            and counts
            and counts["generation_calls_attempted"] == 1
            and counts["generation_calls_completed"] == 0
            and counts["generation_calls_in_flight"] == 1
        )
    elif case == "generation_unusable":
        expected = {"generation_calls_completed": 1, "usable_answers": 0}
        passed = bool(
            counts and counts["generation_calls_completed"] == 1 and counts["usable_answers"] == 0
        )
    elif case == "duplicate_events":
        expected = {"generation_calls_attempted": 1, "duplicate_event_count": 2}
        passed = bool(
            counts
            and counts["generation_calls_attempted"] == 1
            and reduced.get("duplicate_event_count") == 2
        )
        observed["duplicate_event_count"] = reduced.get("duplicate_event_count")
    else:
        expected = {"owned_child_alive_after_cleanup": False}
        passed = (
            row.get("timed_out") is True and row.get("owned_child_alive_after_cleanup") is False
        )
        observed = {"owned_child_alive_after_cleanup": row.get("owned_child_alive_after_cleanup")}
    passed = passed and row.get("owned_child_alive_after_cleanup") is False
    return passed, expected, observed, identity_rejection


def run_cpu_boundary_panel(workdir: Path) -> JsonDict:
    """Run six bounded subprocess controls and reduce only their sidecars."""
    workdir.mkdir(parents=True, exist_ok=True)
    cases = (
        "pre_load_failure",
        "load_no_generation",
        "generation_timeout",
        "generation_unusable",
        "duplicate_events",
        "orphan_cleanup",
    )
    rows: list[JsonDict] = []
    for index, case in enumerate(cases, start=1):
        raw = _run_fixture(case, workdir)
        passed, expected, observed, identity_rejection = _fixture_passed(raw)
        rows.append(
            {
                "case": case,
                "seed": RANDOM_SEED["development"] + index,
                "metric": int(passed),
                "expected": expected,
                "observed": observed,
                "passed": passed,
                "cost": {
                    "wall_s": raw["duration_s"],
                    "current_model_loads": 0,
                    "current_generations": 0,
                },
                "error": None if passed else raw["reduction"].get("errors"),
                "abstention": False,
                "censored": bool(raw["timed_out"]),
                "ledger_path": raw["ledger_path"],
                "ledger_sha256": raw["ledger_sha256"],
                "persisted_event_count": raw["event_count"],
                "counter_reduction": raw["reduction"],
                "signals_sent": raw["signals_sent"],
                "owned_child_pids": raw["owned_child_pids"],
                "owned_child_alive_after_cleanup": raw["owned_child_alive_after_cleanup"],
                "identity_rejection_preserved": identity_rejection,
                "historical_fixture_only": True,
            }
        )
    return {"rows": rows, "all_controls_passed": all(row["passed"] for row in rows)}


class _FakeProcess:
    pid = 7289


def exercise_live_caller_seams(workdir: Path) -> JsonDict:
    """Drive actual load and selfparse seams with CPU-only injected transports."""
    from carnot.agentic import arc_executable_world_model as world_model
    from carnot.agentic import arc_induction_tool_loop as tool_loop

    workdir.mkdir(parents=True, exist_ok=True)
    model_path = workdir / "fixture.gguf"
    server_path = workdir / "llama-server"
    model_path.write_bytes(b"injected model identity only\n")
    server_path.write_bytes(b"injected executable identity only\n")
    load_path = workdir / "actual_load_seam.jsonl"
    old_ledger = os.environ.get(BOUNDARY_LEDGER_ENV)
    os.environ[BOUNDARY_LEDGER_ENV] = str(load_path)
    try:
        proposer = world_model.LocalGGUFProposer(
            model_path=str(model_path),
            model_repository="fixture/model",
            model_filename=model_path.name,
            ffn_cpu_layers=0,
            mtp=False,
            timeout=1,
        )
        health = iter((False, True))
        with (
            patch.object(proposer, "_healthy", side_effect=lambda: next(health)),
            patch.object(
                world_model, "_generator_server_and_env", return_value=(server_path, None)
            ),
            patch.object(world_model, "_kv_quant_for_launch", return_value=None),
            patch.object(world_model, "_llama_server_parallel_launch", return_value=None),
            patch.object(world_model, "_split_args_for_env", return_value=[]),
            patch.object(world_model.subprocess, "Popen", return_value=_FakeProcess()),
            patch(
                "carnot.agentic.arc_eval_provenance.huggingface_snapshot_revision",
                return_value="c" * 40,
            ),
            patch("carnot.agentic.arc_eval_provenance.process_start_tick", return_value=99),
            patch.object(proposer, "_verify_mtp_engaged", return_value=None),
        ):
            load_ok = proposer._ensure_server()
        load_reduction = reduce_boundary_events(InvocationBoundaryLedger(load_path).read_events())

        chat_path = workdir / "actual_selfparse_seam.jsonl"
        os.environ[BOUNDARY_LEDGER_ENV] = str(chat_path)
        chat_proposer = world_model.LocalGGUFProposer(
            model_path=str(model_path),
            model_repository="fixture/model",
            model_filename=model_path.name,
            model_revision="c" * 40,
            ffn_cpu_layers=0,
            mtp=False,
        )
        chat_proposer._proc = _FakeProcess()
        response = {
            "choices": [
                {
                    "message": {"content": "<tool_call>fixture</tool_call>"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
        }
        with patch.object(
            tool_loop.urllib.request,
            "urlopen",
            return_value=io.BytesIO(json.dumps(response).encode()),
        ):
            tool_loop._post_chat(
                chat_proposer,
                [{"role": "user", "content": "fixture"}],
                turn=0,
                timeout_s=1,
                selfparse=True,
            )
        chat_reduction = reduce_boundary_events(InvocationBoundaryLedger(chat_path).read_events())

        rejected_path = workdir / "identity_rejection.jsonl"
        InvocationBoundaryLedger(rejected_path).begin(
            "model_load", _fixture_identity(valid=False), call_id="identity-rejected"
        )
        rejected = reduce_boundary_events(InvocationBoundaryLedger(rejected_path).read_events())
    finally:
        if old_ledger is None:
            os.environ.pop(BOUNDARY_LEDGER_ENV, None)
        else:
            os.environ[BOUNDARY_LEDGER_ENV] = old_ledger
    return {
        "load_boundary_reachable": bool(
            load_ok
            and load_reduction.get("invocation_counts", {}).get("model_loads_completed") == 1
        ),
        "selfparse_generation_boundary_reachable": bool(
            chat_reduction.get("invocation_counts", {}).get("generation_calls_completed") == 1
        ),
        "identity_rejection_preserved": bool(
            rejected.get("disqualified")
            and any(
                str(error).startswith("model_revision_missing:")
                for error in rejected.get("errors", [])
            )
        ),
        "load_ledger_sha256": _sha256_file(load_path),
        "selfparse_ledger_sha256": _sha256_file(chat_path),
        "identity_rejection_ledger_sha256": _sha256_file(rejected_path),
        "current_model_calls": deepcopy(ZERO_INVOCATION_COUNTS),
    }


def _recursive_hashes(directory: Path) -> JsonDict:
    return {
        str(path.relative_to(REPO_ROOT)): {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def freeze_historical_exp7280(root: Path, destination: Path) -> JsonDict:
    """Hash the immutable result and the separately mutable present raw tree."""
    terminal_path = root / HISTORICAL_PATH
    raw_dir = root / HISTORICAL_RAW_DIR
    checkpoint_path = root / HISTORICAL_CHECKPOINT
    terminal_bytes = terminal_path.read_bytes()
    terminal = json.loads(terminal_bytes)
    present_raw_hashes = _recursive_hashes(raw_dir)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    episode_path = raw_dir / "episode_rows.json"
    episode_payload = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_rows = episode_payload.get("rows", [])
    request_rows = [
        request
        for row in episode_rows
        if isinstance(row, Mapping)
        for request in row.get("raw_request_manifest", [])
        if isinstance(request, Mapping)
    ]
    artifact_raw_hashes = {
        key: value
        for key, value in terminal.get("source_artifact_hashes", {}).items()
        if str(key).startswith(str(HISTORICAL_RAW_DIR))
    }
    criticals = [
        "INFERENCE_PROVENANCE_CONTRADICTION",
        "SUBSTRATE_CLASS_MISMATCH",
    ]
    validation_text = "\n".join(
        str(row.get("output_tail") or "") for row in terminal.get("validation_receipts", [])
    )
    quarantine_observed = all(item in validation_text for item in criticals)
    server_pid = int(checkpoint.get("server_pid") or 0)
    receipt = {
        "schema": "carnot.exp7280_historical_freeze.v1",
        "captured_at_utc": _utc_now(),
        "terminal_artifact_path": str(HISTORICAL_PATH),
        "terminal_artifact_sha256": _sha256_bytes(terminal_bytes),
        "terminal_artifact_size_bytes": len(terminal_bytes),
        "terminal_observations": {
            "experiment_id": terminal.get("experiment_id"),
            "status": terminal.get("status"),
            "verdict_class": terminal.get("verdict_class"),
            "honest_verdict": terminal.get("honest_verdict"),
            "started_at_utc": terminal.get("started_at_utc"),
            "ended_at_utc": terminal.get("ended_at_utc"),
            "duration_s": terminal.get("duration_s"),
            "model_invoked": terminal.get("model_invoked"),
            "model_loads_attempted": terminal.get("invocation_counts", {}).get(
                "model_loads_attempted"
            ),
            "model_loads_completed": terminal.get("invocation_counts", {}).get(
                "model_loads_completed"
            ),
            "generation_calls_attempted": terminal.get("invocation_counts", {}).get(
                "generation_calls_attempted"
            ),
            "all_rows_censored": all(row.get("censored") for row in terminal.get("rows", [])),
            "artifact_raw_hashes": artifact_raw_hashes,
        },
        "present_raw_hashes": present_raw_hashes,
        "present_checkpoint": {
            "path": str(HISTORICAL_CHECKPOINT),
            "sha256": _sha256_file(checkpoint_path),
            "value": checkpoint,
        },
        "present_raw_observations": {
            "last_completed_phase": checkpoint.get("stage"),
            "completed_units": checkpoint.get("completed_units"),
            "total_units": checkpoint.get("total_units"),
            "model_loaded": checkpoint.get("model_loaded"),
            "server_pid": server_pid or None,
            "server_pid_alive_at_freeze": _pid_alive(server_pid) if server_pid else False,
            "episode_dispositions": [row.get("disposition") for row in episode_rows],
            "generation_calls_attempted": sum(
                int(row.get("generation_calls_attempted") or 0)
                for row in episode_rows
                if isinstance(row, Mapping)
            ),
            "generation_calls_completed": sum(
                int(row.get("generation_calls_completed") or 0)
                for row in episode_rows
                if isinstance(row, Mapping)
            ),
            "last_request_started_at_utc": request_rows[-1].get("started_at_utc")
            if request_rows
            else None,
            "last_request_transport_completed": request_rows[-1].get("transport_completed")
            if request_rows
            else None,
            "live_session_present": (raw_dir / "live_session.json").is_file(),
        },
        "diagnosis": {
            "observed_failures": [
                "terminal artifact says one completed load while model_invoked is false",
                "terminal artifact has four synthetic censored rows with no request lineage",
                "present checkpoint is nonterminal after one later episode",
                "present raw rows contain two attempted and zero completed generations",
                "present raw tree does not contain live_session.json",
                "terminal artifact raw hashes and present raw bytes describe different executions",
            ],
            "observed_active_call": "not_authenticatable",
            "observed_last_event": "later checkpoint stage=episodes completed_units=1",
            "hypotheses": [
                "the owning harness was externally cancelled after its child stopped updating the checkpoint",
                "the child or parent exceeded a timeout outside the persisted episode accounting boundary",
            ],
            "cause": "unresolved_external_timeout_boundary",
        },
        "quarantine": {
            "preserved": quarantine_observed,
            "critical_findings": criticals,
            "scientific_claims_included": False,
            "source_artifact_rewritten": False,
        },
    }
    atomic_write(destination, receipt)
    return receipt


def _gate(criterion: str, expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def reduce_panel(panel: Mapping[str, Any]) -> JsonDict:
    rows = [dict(row) for row in panel.get("rows", []) if isinstance(row, Mapping)]
    cases = {str(row.get("case")) for row in rows}
    expected = {
        "pre_load_failure",
        "load_no_generation",
        "generation_timeout",
        "generation_unusable",
        "duplicate_events",
        "orphan_cleanup",
    }
    return {
        "planned_units": 6,
        "attempted_units": len(rows),
        "completed_units": sum(bool(row.get("passed")) for row in rows),
        "censored_units": sum(bool(row.get("censored")) for row in rows),
        "case_set_complete": cases == expected,
        "no_surviving_owned_child": all(
            row.get("owned_child_alive_after_cleanup") is False for row in rows
        ),
        "identity_rejection_preserved": any(
            row.get("case") == "pre_load_failure"
            and row.get("identity_rejection_preserved") is True
            for row in rows
        ),
        "all_controls_passed": len(rows) == 6 and all(row.get("passed") is True for row in rows),
    }


def independent_reduce(path: Path) -> JsonDict:
    """Recompute readiness inputs from raw fixture rows only."""
    return {
        **reduce_panel(json.loads(path.read_text(encoding="utf-8"))),
        "arc_boundary_ready_score": int(
            reduce_panel(json.loads(path.read_text(encoding="utf-8")))["all_controls_passed"]
        ),
    }


def build_live_handoff(root: Path, caller_reachability: Mapping[str, Any]) -> JsonDict:
    handoff = {
        "entrypoint": "scripts/experiments/experiment_7290_v641_arc_selfparse.py",
        "arguments": ["--date", RUN_DATE],
        "entrypoint_state": "declared_future_output_not_a_precondition",
        "environment": {
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            BOUNDARY_LEDGER_ENV: (
                "results/raw/experiment_7290_v641_arc_selfparse/inference_boundary_events.jsonl"
            ),
        },
        "load_caller": "carnot.agentic.arc_executable_world_model.LocalGGUFProposer._ensure_server",
        "generation_caller": "carnot.agentic.arc_induction_tool_loop._post_chat",
        "caller_reachability": deepcopy(dict(caller_reachability)),
        "cleanup_ownership": "the wrapper owns and terminates only its start_new_session process group",
        "existing_budget_constants": {
            "session_limit_s": 3000,
            "model_load_limit_s": 600,
            "actions": 192,
            "generation_calls": 2,
            "generated_tokens": 4096,
        },
        "changed_code_hashes": {
            str(path): _sha256_file(root / path)
            for path in (
                BOUNDARY_MODULE_PATH,
                WORLD_MODEL_PATH,
                TOOL_LOOP_PATH,
                MODULE_PATH,
                WRAPPER_PATH,
            )
        },
    }
    handoff["handoff_sha256"] = _sha256_bytes(_canonical_bytes(handoff))
    return handoff


def _principles_for(fields: Sequence[str]) -> JsonDict:
    return {
        field: REQUIRED_FIELD_PRINCIPLES.get(
            field, "Retain this supporting value as ordinary machine-readable evidence."
        )
        for field in fields
    }


def _terminal_boundary_row(row: Mapping[str, Any]) -> JsonDict:
    """Project injected model-shaped evidence to a hashed historical sidecar reference."""
    case = str(row.get("case"))
    expected_results = {
        "pre_load_failure": "invalid_identity_disqualified_after_persisted_failure",
        "load_no_generation": "completed_load_without_generation",
        "generation_timeout": "attempt_retained_without_false_completion",
        "generation_unusable": "completed_transport_marked_unusable",
        "duplicate_events": "exact_duplicates_reduced_idempotently",
        "orphan_cleanup": "owned_process_group_terminated",
    }
    reduction = row.get("counter_reduction", {})
    call_rows = reduction.get("call_rows", []) if isinstance(reduction, Mapping) else []
    return {
        "case": case,
        "seed": row.get("seed"),
        "metric": row.get("metric"),
        "cost": {
            "wall_s": row.get("cost", {}).get("wall_s"),
            "cpu_fixture_units": 1,
        },
        "error": deepcopy(row.get("error")),
        "abstention": row.get("abstention"),
        "censored": row.get("censored"),
        "expected_result": expected_results.get(case, "unknown_fixture_must_fail_closed"),
        "observed_result": (
            expected_results.get(case) if row.get("passed") is True else "expectation_not_met"
        ),
        "passed": row.get("passed"),
        "persisted_evidence": {
            "provenance_scope": "historical",
            "sidecar_path": row.get("ledger_path"),
            "sidecar_sha256": row.get("ledger_sha256"),
            "event_count": row.get("persisted_event_count"),
        },
        "counter_reduction": {
            "provenance_scope": "historical",
            "activity_known": reduction.get("activity_known")
            if isinstance(reduction, Mapping)
            else None,
            "disqualified": reduction.get("disqualified")
            if isinstance(reduction, Mapping)
            else True,
            "duplicate_event_count": reduction.get("duplicate_event_count")
            if isinstance(reduction, Mapping)
            else None,
            "terminal_states": sorted(
                str(call.get("terminal_state")) for call in call_rows if isinstance(call, Mapping)
            ),
            "exact_expectation_matched": row.get("passed") is True,
        },
        "signals_sent": deepcopy(row.get("signals_sent", [])),
        "owned_child_pids": deepcopy(row.get("owned_child_pids", [])),
        "owned_child_alive_after_cleanup": row.get("owned_child_alive_after_cleanup"),
        "identity_rejection_preserved": row.get("identity_rejection_preserved"),
        "historical_fixture_only": True,
        "provenance_scope": "historical",
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    panel: Mapping[str, Any],
    historical: Mapping[str, Any],
    live_handoff: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    reduction = reduce_panel(panel)
    reachability = live_handoff.get("caller_reachability", {})
    validation_passed = bool(validation_receipts) and all(
        row.get("passed") is True for row in validation_receipts
    )
    ready = bool(
        reduction["all_controls_passed"]
        and reduction["no_surviving_owned_child"]
        and reduction["identity_rejection_preserved"]
        and reachability.get("load_boundary_reachable") is True
        and reachability.get("selfparse_generation_boundary_reachable") is True
        and reachability.get("identity_rejection_preserved") is True
        and historical.get("quarantine", {}).get("preserved") is True
        and validation_passed
    )
    gates = [
        _gate(
            "six_lifecycle_controls",
            True,
            reduction["all_controls_passed"],
            reduction["all_controls_passed"],
            "Every named lifecycle control must pass from persisted rows.",
        ),
        _gate(
            "no_surviving_owned_child",
            True,
            reduction["no_surviving_owned_child"],
            reduction["no_surviving_owned_child"],
            "Cancellation is complete only after task-owned children stop.",
        ),
        _gate(
            "identity_rejection_preserved",
            True,
            bool(
                reduction["identity_rejection_preserved"]
                and reachability.get("identity_rejection_preserved")
            ),
            bool(
                reduction["identity_rejection_preserved"]
                and reachability.get("identity_rejection_preserved")
            ),
            "Lifecycle evidence cannot weaken strict model identity.",
        ),
        _gate(
            "actual_load_and_selfparse_callers_reachable",
            True,
            {
                "load": reachability.get("load_boundary_reachable"),
                "selfparse": reachability.get("selfparse_generation_boundary_reachable"),
            },
            bool(
                reachability.get("load_boundary_reachable")
                and reachability.get("selfparse_generation_boundary_reachable")
            ),
            "Readiness requires the production call sites, not only the ledger class.",
        ),
        _gate(
            "historical_quarantine_preserved",
            True,
            historical.get("quarantine", {}).get("preserved"),
            historical.get("quarantine", {}).get("preserved") is True,
            "A provenance repair does not rehabilitate historical science.",
        ),
        _gate(
            "scoped_validation_passed",
            True,
            validation_passed,
            validation_passed,
            "Publication retains every actual command result.",
        ),
    ]
    rows = [
        _terminal_boundary_row(row) for row in panel.get("rows", []) if isinstance(row, Mapping)
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "field_principles": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned_units": 6,
            "attempted_units": reduction["attempted_units"],
            "completed_units": reduction["completed_units"],
            "censored_units": reduction["censored_units"],
            "stopping_rule": "run each frozen CPU lifecycle control once; timeout fixtures remain censored",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": [],
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_circular_positive_arc_boundary_ready_historical_timeout_unresolved"
            if ready
            else "complete_disqualified_arc_boundary_validation_or_control_failed"
        ),
        "verdict_class": "circular_positive" if ready else "disqualified",
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "arc_boundary_ready_score": int(ready),
        "boundary_event_rows": rows,
        "historical_timeout_diagnosis": deepcopy(dict(historical.get("diagnosis", {}))),
        "live_handoff": deepcopy(dict(live_handoff)),
        "quarantine_preserved": historical.get("quarantine", {}).get("preserved") is True,
        "current_scientific_claim": None,
        "historical_scientific_claim_included": False,
        "official_score": None,
    }
    artifact["field_principles"] = _principles_for(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Cold-check terminal identity, current inference scope, rows, and seal."""
    artifact = (
        json.loads(value.read_text(encoding="utf-8"))
        if isinstance(value, Path)
        else deepcopy(dict(value))
    )
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_or_experiment_identity_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("milestone_or_date_mismatch")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_not_terminal")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("MODEL_SPECS") != []:
        errors.append("current_model_specs_must_be_empty")
    if artifact.get("model_invoked") is not False:
        errors.append("current_model_invoked_must_be_false")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_must_be_zero")
    if artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator":
        errors.append("current_inference_substrate_invalid")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("current_inference_substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("verifier_is_oracle") is True and artifact.get("verdict_class") == "positive":
        errors.append("oracle_forbids_positive")
    if artifact.get("official_score") is not None:
        errors.append("official_score_must_be_unset")
    reduction = reduce_panel({"rows": artifact.get("rows", [])})
    if len(artifact.get("rows", [])) != 6 or not reduction["case_set_complete"]:
        errors.append("boundary_rows_incomplete")
    if artifact.get("boundary_event_rows") != artifact.get("rows"):
        errors.append("boundary_event_rows_mismatch")
    expected_ready = int(
        reduction["all_controls_passed"]
        and reduction["no_surviving_owned_child"]
        and reduction["identity_rejection_preserved"]
        and all(row.get("passed") is True for row in artifact.get("acceptance_gate_results", []))
    )
    if artifact.get("arc_boundary_ready_score") != expected_ready:
        errors.append("arc_boundary_ready_score_inconsistent")
    if expected_ready and artifact.get("verdict_class") != "circular_positive":
        errors.append("ready_verdict_class_invalid")
    if not expected_ready and artifact.get("verdict_class") == "circular_positive":
        errors.append("failed_gate_verdict_class_invalid")
    if artifact.get("quarantine_preserved") is not True:
        errors.append("historical_quarantine_not_preserved")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration_invalid")
    spans = artifact.get("phase_spans", [])
    if any(float(row.get("duration_s") or 0) < 0 for row in spans):
        errors.append("phase_span_invalid")
    if sum(float(row.get("duration_s") or 0) for row in spans) > float(duration or 0) + 0.01:
        errors.append("phase_spans_exceed_duration")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    rows: list[JsonDict] = []
    hashes: JsonDict = {}
    for path in INPUT_PATHS:
        target = root / path
        exists = target.is_file()
        digest = _sha256_file(target) if exists else None
        rows.append(
            {
                "check": "required_input",
                "upstream": str(path),
                "field": "regular_file",
                "expected": True,
                "observed": exists,
                "passed": exists,
                "sha256": digest,
            }
        )
        if digest:
            hashes[str(path)] = {
                "sha256": digest,
                "quarantined": path == HISTORICAL_PATH,
                "retired": False,
                "authority": "historical_diagnostic_only" if path == HISTORICAL_PATH else "input",
            }
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    rows.append(
        {
            "check": "driving_capability",
            "upstream": str(SPEC_PATH),
            "field": "REQ-ARC-WMTE-7289",
            "expected": True,
            "observed": "REQ-ARC-WMTE-7289" in spec_text,
            "passed": "REQ-ARC-WMTE-7289" in spec_text,
            "sha256": _sha256_file(root / SPEC_PATH) if (root / SPEC_PATH).is_file() else None,
        }
    )
    for path in (root / RESULT_PATH, root / CHECKPOINT_PATH, root / RAW_DIR):
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK)
        rows.append(
            {
                "check": "declared_output_path",
                "upstream": str(path.relative_to(root)),
                "field": "parent_writable",
                "expected": True,
                "observed": writable,
                "passed": writable,
                "sha256": _sha256_bytes(str(path).encode()),
            }
        )
    return rows, hashes


def build_validation_commands(raw_rows: Path, candidate: Path) -> list[JsonDict]:
    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    return [
        {
            "name": "focused_exp7289",
            "command": [
                pytest,
                "-o",
                "addopts=",
                str(TEST_PATH),
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7289-focused",
            ],
            "timeout_s": 300,
        },
        {
            "name": "affected_arc_eval_provenance",
            "command": [
                pytest,
                "-o",
                "addopts=",
                "tests/python/test_arc_eval_provenance_contract_20260905.py",
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7289-provenance",
            ],
            "timeout_s": 600,
        },
        {
            "name": "e2e_009_cross_call_persistence",
            "command": [
                pytest,
                "-o",
                "addopts=",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7289-e2e009",
            ],
            "timeout_s": 600,
        },
        {
            "name": "e2e_010_transport_and_cancellation",
            "command": [
                pytest,
                "-o",
                "addopts=",
                "tests/python/test_arc_tool_grammar_transport.py",
                str(TEST_PATH),
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7289-e2e010",
            ],
            "timeout_s": 600,
        },
        {
            "name": "full_python_suite",
            "command": [
                pytest,
                "-o",
                "addopts=",
                "tests/python",
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7289-full",
            ],
            "timeout_s": 1800,
        },
        {
            "name": "scoped_coverage_run",
            "command": [
                coverage,
                "run",
                "--data-file=/tmp/exp7289.coverage",
                "--include=*/arc_inference_boundary.py,*/experiment_7289_v641_arc_boundary.py",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                str(TEST_PATH),
                "-q",
                "-n",
                "0",
                "--basetemp=/tmp/exp7289-coverage",
            ],
            "timeout_s": 600,
        },
        {
            "name": "scoped_coverage_report",
            "command": [
                coverage,
                "report",
                "--data-file=/tmp/exp7289.coverage",
                "--include=*/arc_inference_boundary.py,*/experiment_7289_v641_arc_boundary.py",
                "--show-missing",
                "--fail-under=100",
            ],
            "timeout_s": 300,
        },
        {
            "name": "ruff_check",
            "command": [
                ruff,
                "check",
                str(BOUNDARY_MODULE_PATH),
                str(MODULE_PATH),
                str(WORLD_MODEL_PATH),
                str(TOOL_LOOP_PATH),
                str(WRAPPER_PATH),
                str(TEST_PATH),
            ],
            "timeout_s": 300,
        },
        {
            "name": "ruff_format",
            "command": [
                ruff,
                "format",
                "--check",
                str(BOUNDARY_MODULE_PATH),
                str(MODULE_PATH),
                str(WORLD_MODEL_PATH),
                str(TOOL_LOOP_PATH),
                str(WRAPPER_PATH),
                str(TEST_PATH),
            ],
            "timeout_s": 300,
        },
        {
            "name": "changed_module_mypy",
            "command": [
                mypy,
                str(BOUNDARY_MODULE_PATH),
                str(MODULE_PATH),
                str(WORLD_MODEL_PATH),
                str(TOOL_LOOP_PATH),
            ],
            "timeout_s": 600,
        },
        {
            "name": "scoped_spec_coverage",
            "command": [
                python,
                "-u",
                "scripts/check_spec_coverage.py",
                str(TEST_PATH),
                "tests/python/test_arc_eval_provenance_contract_20260905.py",
                "tests/python/test_arc_induction_state_persistence.py",
                "tests/python/test_arc_tool_grammar_transport.py",
            ],
            "timeout_s": 300,
        },
        {
            "name": "independent_raw_row_reducer",
            "command": [python, "-u", str(REPO_ROOT / WRAPPER_PATH), "--reduce-raw", str(raw_rows)],
            "timeout_s": 300,
        },
        {
            "name": "terminal_candidate_adversarial_verify",
            "command": [python, "-u", "scripts/adversarial_verify.py", str(candidate)],
            "timeout_s": 300,
        },
        {
            "name": "terminal_candidate_row_consistency",
            "command": [python, "-u", "scripts/verdict_row_consistency_lint.py", str(candidate)],
            "timeout_s": 300,
        },
    ]


def _run_validation_rows(
    commands: Sequence[Mapping[str, Any]], raw_dir: Path, started: float
) -> list[JsonDict]:  # pragma: no cover - subprocess integration.
    from carnot.experiment_7246_v638_source_map import _run_streaming_command

    directory = raw_dir / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for index, item in enumerate(commands):
        name = str(item["name"])
        command = [str(part) for part in item["command"]]
        _progress(started, "validation", "before_subprocess", name=name, completed_units=index)
        result = _run_streaming_command(
            command,
            cwd=REPO_ROOT,
            timeout_s=float(item["timeout_s"]),
            heartbeat_s=45,
            operation=f"exp7289:{name}",
        )
        output = str(result.get("output") or "")
        log_path = directory / f"{index:02d}_{name}.log"
        log_path.write_text(output, encoding="utf-8")
        row = {
            "name": name,
            "command": " ".join(command),
            "exit_code": int(result["exit_code"]),
            "expected_exit_code": 0,
            "passed": int(result["exit_code"]) == 0,
            "timed_out": bool(result.get("timed_out")),
            "duration_s": float(result["duration_s"]),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "log_sha256": _sha256_file(log_path),
            "output_tail": output[-4000:],
        }
        rows.append(row)
        _progress(
            started,
            "validation",
            "after_subprocess",
            name=name,
            exit_code=row["exit_code"],
            completed_units=len(rows),
        )
    return rows


def run_experiment(run_date: str) -> JsonDict:  # pragma: no cover - end-to-end publication.
    started = time.monotonic()
    started_at = _utc_now()
    _progress(started, "startup", "entrypoint_and_paths_authenticated")
    phase_spans: list[JsonDict] = []

    phase = time.monotonic()
    _progress(started, "preconditions", "begin")
    preconditions, source_hashes = collect_preconditions(REPO_ROOT)
    phase_spans.append({"phase": "preconditions", "duration_s": time.monotonic() - phase})
    _progress(
        started,
        "preconditions",
        "end",
        passed=all(row["passed"] for row in preconditions),
        completed_units=len(preconditions),
    )
    if run_date != RUN_DATE:
        raise ValueError(f"--date must be {RUN_DATE}")
    if not all(row["passed"] for row in preconditions):
        raise RuntimeError("required input or output path failed before evidence collection")

    raw_dir = REPO_ROOT / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {"status": "running", "stage": "historical_freeze", "completed_units": 0},
    )
    phase = time.monotonic()
    _progress(started, "historical_freeze", "before_benchmark")
    historical_path = raw_dir / "historical_exp7280_freeze.json"
    historical = freeze_historical_exp7280(REPO_ROOT, historical_path)
    source_hashes[str(historical_path.relative_to(REPO_ROOT))] = {
        "sha256": _sha256_file(historical_path),
        "experiment_id": "exp7280-arc-live",
        "terminal_class": "null",
        "quarantined": True,
        "retired": False,
    }
    phase_spans.append({"phase": "historical_freeze", "duration_s": time.monotonic() - phase})
    _progress(started, "historical_freeze", "after_benchmark", completed_units=1)

    phase = time.monotonic()
    _progress(started, "cpu_boundary_panel", "before_benchmark", total_units=6)
    panel = run_cpu_boundary_panel(raw_dir / "fixtures")
    raw_rows_path = raw_dir / "boundary_event_rows.json"
    atomic_write(raw_rows_path, panel)
    phase_spans.append({"phase": "cpu_boundary_panel", "duration_s": time.monotonic() - phase})
    _progress(started, "cpu_boundary_panel", "after_benchmark", completed_units=6)

    phase = time.monotonic()
    _progress(started, "caller_reachability", "before_benchmark")
    seams = exercise_live_caller_seams(raw_dir / "caller_reachability")
    handoff = build_live_handoff(REPO_ROOT, seams)
    handoff_path = raw_dir / "live_handoff.json"
    atomic_write(handoff_path, handoff)
    source_hashes[str(raw_rows_path.relative_to(REPO_ROOT))] = {
        "sha256": _sha256_file(raw_rows_path),
        "quarantined": False,
        "retired": False,
    }
    source_hashes[str(handoff_path.relative_to(REPO_ROOT))] = {
        "sha256": _sha256_file(handoff_path),
        "quarantined": False,
        "retired": False,
    }
    phase_spans.append({"phase": "caller_reachability", "duration_s": time.monotonic() - phase})
    _progress(started, "caller_reachability", "after_benchmark", completed_units=2)

    candidate_path = raw_dir / "terminal_candidate.json"
    preliminary = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        panel=panel,
        historical=historical,
        live_handoff=handoff,
        validation_receipts=[],
    )
    atomic_write(candidate_path, preliminary)
    commands = build_validation_commands(raw_rows_path, candidate_path)
    validation = _run_validation_rows(commands[:-2], raw_dir, started)
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        panel=panel,
        historical=historical,
        live_handoff=handoff,
        validation_receipts=validation,
    )
    atomic_write(candidate_path, candidate)
    validation.extend(_run_validation_rows(commands[-2:], raw_dir, started))
    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
        preconditions_checked=preconditions,
        source_artifact_hashes=source_hashes,
        panel=panel,
        historical=historical,
        live_handoff=handoff,
        validation_receipts=validation,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact failed validation: {errors}")
    atomic_write(candidate_path, artifact)
    atomic_write(REPO_ROOT / RESULT_PATH, artifact)
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {
            "status": "complete",
            "stage": "terminal_published",
            "completed_units": 6,
            "result_path": str(RESULT_PATH),
        },
    )
    _progress(
        started,
        "publication",
        "terminal_artifact_atomically_written",
        path=str(RESULT_PATH),
        arc_boundary_ready_score=artifact["arc_boundary_ready_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--role", choices=("experiment", "fixture-child"), default="experiment")
    parser.add_argument("--case")
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--marker-path", type=Path)
    parser.add_argument("--reduce-raw", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin dispatch.
    print(
        json.dumps({"experiment": EXPERIMENT_ID, "phase": "startup", "event": "entrypoint"}),
        flush=True,
    )
    args = parse_args(argv)
    if args.reduce_raw is not None:
        print(json.dumps(independent_reduce(args.reduce_raw), sort_keys=True), flush=True)
        return 0
    if args.role == "fixture-child":
        if not args.case or args.ledger_path is None:
            raise ValueError("fixture child requires --case and --ledger-path")
        return _fixture_child(args.case, args.ledger_path, args.marker_path)
    run_experiment(args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
