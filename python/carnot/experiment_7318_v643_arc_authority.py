"""Repair the owner-issued ARC episode-authority handoff without model work.

Spec refs: REQ-ARC-WMTE-7318 and SCENARIO-ARC-WMTE-7318-*.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

import numpy as np
import yaml

from carnot import gpu_lease_phase_journal as lease_api


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7318-arc-authority"
MILESTONE = "2026.09.643"
RUN_DATE = "20260915"
SCHEMA = "carnot.experiment_7318.v643.arc_authority.v1"
RESULT_PATH = Path("results/experiment_7318_v643_arc_authority.json")
RAW_DIR = Path("results/raw/experiment_7318_v643_arc_authority")
MODULE_PATH = Path("python/carnot/experiment_7318_v643_arc_authority.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7318_v643_arc_authority.py")
TEST_PATH = Path("tests/python/test_experiment_7318_v643_arc_authority.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
AUTHENTICATED_INPUTS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7305_v642_arc_selfparse.py"),
    Path("python/carnot/experiment_7280_v640_arc_live.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/experiment_6973_lease_aware_gguf_runtime.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    Path("results/experiment_7304_v642_arc_receipt.json"),
    Path("results/experiment_7305_v642_arc_selfparse.json"),
    Path("tests/python/test_arc_eval_provenance_contract_20260905.py"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
MODEL_SPECS: list[JsonDict] = []
SUBSTRATE = "cpu_exact_solver_or_simulator"

MODEL_IDENTITY: JsonDict = {
    "hf_id": "unsloth/Qwen3.8-27B-GGUF",
    "model_path": "/fixture/unsloth-Qwen3.8-27B-Q4_K_M.gguf",
    "model_hash": "sha256:" + "7" * 64,
    "quantization": "Q4_K_M",
}
RESOURCE_BOUNDS: JsonDict = {
    "action_limit": 192,
    "completion_limit": 2,
    "generated_token_limit": 4096,
    "session_limit_s": 3000,
}
INVOCATION_COUNTS: JsonDict = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
PROVENANCE_VALIDATOR_SHA256 = (
    "sha256:b9c4888df96ee80331c3fd6d0ac07b2f9ba8cf601f52a5f2a75ab8775c400a0c"
)

FIELD_PRINCIPLES = {
    "schema": "Version the artifact while keeping ordinary top-level values.",
    "experiment_id": "Bind the receipt to the authority repair task.",
    "milestone": "Bind the receipt to milestone 2026.09.643.",
    "status": "Publish only a terminal complete, blocked, or disqualified state.",
    "run_date": "Use the directed execution date and retain actual UTC bounds.",
    "started_at_utc": "Record when current CPU work started.",
    "ended_at_utc": "Record when terminal construction ended.",
    "preconditions_checked": "Authenticate required inputs and preserve exact failures.",
    "MODEL_SPECS": "List only models executed by this invocation.",
    "model_invoked": "Any attempted current load or generation must set this true.",
    "invocation_counts": "Separate every current load and generation disposition.",
    "inference_substrate": "Describe the CPU-only mechanism checks that ran.",
    "inference_substrate_class": "Use the recognized class for the actual computation.",
    "execution_venue": "This task executes on the host.",
    "duration_s": "Use measured monotonic elapsed time without padding.",
    "phase_spans": "Retain disjoint measured phases and completed units.",
    "random_seed": "Freeze separate development and evaluation seeds before outcomes.",
    "reproducibility_checksum": "Bind code, inputs, settings, and raw evidence.",
    "source_artifact_hashes": "Authenticate producers without treating history as readiness.",
    "rows": "Keep every mechanism and control unit with costs and failures.",
    "sample_size_budget": "Freeze planned and observed control counts and stopping rule.",
    "acceptance_gate_results": "Retain expected, observed, pass, and principle per gate.",
    "gate_check_summary": "A block names the exact upstream field and comparison.",
    "verifier_is_oracle": "Shared execution authority forbids a positive scientific class.",
    "honest_verdict": "State the terminal mechanism finding with the required prefix.",
    "verdict_class": "Use the closed verdict vocabulary.",
    "validation_receipts": "Keep exact commands, exits, elapsed times, and log hashes.",
    "repository_health": "Keep unrelated suite failures separate from affected checks.",
    "field_principles": "Explain why every top-level field exists.",
    "arc_authority_ready_score": "Require the real handoff and every denial control.",
    "authority_handoff_rows": "Show the same grant at each issuer-to-caller hop.",
    "denial_control_rows": "Show invalid grants remain rejected.",
    "live_entrypoint_receipt": "Hash the real launcher, child environment, and caller.",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _function_source_hash(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == function_name
    )
    source = ast.get_source_segment(text, node)
    if source is None:  # pragma: no cover - parsed local function nodes retain source ranges.
        raise RuntimeError(f"source unavailable for {function_name}")
    return _sha256_bytes(source.encode())


def artifact_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(_canonical_bytes(payload))


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a terminal JSON file only after complete bytes are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:
    print(
        json.dumps(
            {
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 6),
                **details,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def gate_check(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [row for row in checks if row.get("passed") is not True]
    first = None
    if failed:
        row = failed[0]
        first = {
            key: row.get(key)
            for key in ("upstream", "check", "artifact_field", "expected", "observed")
        }
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "first_failure": first,
    }


def _finish_lease(lease: lease_api.GpuLease) -> None:
    if lease.document.get("phase") not in lease_api.TERMINAL_PHASES:
        lease.transition("terminal_blocked")
    lease.release()


def _acquire_lease(work_dir: Path, *, ttl_s: float = 120.0) -> lease_api.GpuLease:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=work_dir / "gpu-lease",
        task_id=EXPERIMENT_ID,
        device_uuid="GPU-exp7318-cpu-fixture",
        expected_model=MODEL_IDENTITY["model_path"],
        vram_before_mb=0,
        ttl_s=ttl_s,
    )
    lease.transition("admitted")
    lease.transition("loading")
    return lease


def _issue(
    lease: lease_api.GpuLease,
    work_dir: Path,
    *,
    episode_id: str = "r11l:direct_selfparse",
    game: str = "r11l",
    valid_for_s: float = 60.0,
    suffix: str = "valid",
) -> JsonDict:
    return lease.issue_arc_episode_authority(
        episode_id=episode_id,
        game=game,
        model_identity=MODEL_IDENTITY,
        resource_bounds=RESOURCE_BOUNDS,
        nonce_ledger_path=work_dir / "nonce-ledgers" / suffix,
        valid_for_s=valid_for_s,
        nonce=f"exp7318-{suffix}-nonce",
    )


def _validate(
    authority: Mapping[str, Any] | None,
    *,
    game: str = "r11l",
    model_identity: Mapping[str, Any] = MODEL_IDENTITY,
    owner_pid: int | None = None,
    owner_ticks: int | None = None,
    now_utc: datetime | None = None,
    now_ns: int | None = None,
    consume: bool = False,
) -> JsonDict:
    return lease_api.validate_arc_episode_authority(
        authority,
        expected_game=game,
        expected_model_identity=model_identity,
        expected_owner_pid=owner_pid,
        expected_owner_start_ticks=owner_ticks,
        now_utc=now_utc,
        now_monotonic_ns=now_ns,
        consume=consume,
    )


def authority_bundle_environment(
    base_environment: Mapping[str, str], authorities: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Carry only owner-issued public grants into the live child."""

    environment = dict(base_environment)
    environment[lease_api.ARC_AUTHORITY_BUNDLE_ENV] = json.dumps(
        list(authorities), sort_keys=True, separators=(",", ":")
    )
    environment.pop(lease_api.ARC_AUTHORITY_ENV, None)
    return environment


def _read_authority_bundle(environment: Mapping[str, str]) -> list[JsonDict]:
    raw = environment.get(lease_api.ARC_AUTHORITY_BUNDLE_ENV, "")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        return []
    return [dict(row) for row in value]


def validate_authority_bundle_before_model_load(
    environment: Mapping[str, str],
    schedule: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Validate every scheduled grant before a blocking model load starts."""

    bundle = _read_authority_bundle(environment)
    if not bundle:
        return {"allowed": False, "reason": "missing_authority_bundle", "model_load_count": 0}
    if len(bundle) != len(schedule):
        return {"allowed": False, "reason": "authority_schedule_mismatch", "model_load_count": 0}
    for scheduled in schedule:
        episode_id = str(scheduled.get("episode_id"))
        authority = next(
            (
                row
                for row in bundle
                if (row.get("episode_scope") or {}).get("episode_id") == episode_id
            ),
            None,
        )
        result = lease_api.validate_arc_episode_authority(
            authority,
            expected_game=str(scheduled.get("game")),
            expected_episode_id=episode_id,
            expected_model_identity=model_identity,
            consume=False,
        )
        if result.get("allowed") is not True:
            return {**result, "model_load_count": 0}
    return {"allowed": True, "reason": "allowed", "model_load_count": 0}


def select_episode_authority(
    environment: Mapping[str, str],
    schedule_row: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> tuple[JsonDict, dict[str, str]]:
    """Select one exact episode grant and expose the legacy flat mapping."""

    episode_id = str(schedule_row.get("episode_id"))
    authority = next(
        (
            row
            for row in _read_authority_bundle(environment)
            if (row.get("episode_scope") or {}).get("episode_id") == episode_id
        ),
        None,
    )
    result = lease_api.validate_arc_episode_authority(
        authority,
        expected_game=str(schedule_row.get("game")),
        expected_episode_id=episode_id,
        expected_model_identity=model_identity,
        consume=False,
    )
    if result.get("allowed") is not True or authority is None:
        raise RuntimeError(f"ARC episode authority rejected: {result.get('reason')}")
    selected_environment = dict(environment)
    selected_environment[lease_api.ARC_AUTHORITY_ENV] = json.dumps(
        authority, sort_keys=True, separators=(",", ":")
    )
    return dict(authority), selected_environment


def attach_episode_authority(proposer: Any, authority: Mapping[str, Any]) -> JsonDict:
    """Put the validated public grant on the proposer used by the scored caller."""

    proposer.arc_eval_authority = deepcopy(dict(authority))
    return deepcopy(proposer.arc_eval_authority)


def _authority_handoff_row(hop: str, authority: Mapping[str, Any]) -> JsonDict:
    fields = {
        key: deepcopy(authority.get(key))
        for key in (
            "lease_id",
            "lease_hash",
            "lease_issued_at",
            "lease_expires_at",
            "issuer_record",
            "model_identity",
            "episode_scope",
            "resource_bounds",
        )
    }
    return {
        "hop": hop,
        "authority_hash": authority.get("authority_hash"),
        "authority_fields": fields,
        "required_fields_present": all(value is not None for value in fields.values()),
        "field_dropped": False,
    }


def run_authority_denial_panel(work_dir: Path) -> JsonDict:
    """Exercise all authority denials with a current kernel-backed issuer."""

    work_dir.mkdir(parents=True, exist_ok=True)
    lease = _acquire_lease(work_dir)
    try:
        valid = _issue(lease, work_dir, suffix="valid")
        valid_result = _validate(valid, consume=True)
        replayed = _validate(valid, consume=True)

        expired = _issue(lease, work_dir, valid_for_s=1.0, suffix="expired")
        issued_at = datetime.fromisoformat(str(expired["lease_issued_at"]))
        expired_result = _validate(
            expired,
            now_utc=issued_at + timedelta(seconds=2),
            now_ns=int(expired["issued_monotonic_ns"]) + 2_000_000_000,
        )
        tampered = deepcopy(expired)
        tampered["resource_bounds"]["action_limit"] = 193
        wrong_model = {**MODEL_IDENTITY, "model_hash": "sha256:" + "8" * 64}
        controls = [
            ("missing", _validate(None)),
            ("expired", expired_result),
            ("tampered", _validate(tampered)),
            (
                "wrong_owner",
                _validate(
                    _issue(lease, work_dir, suffix="wrong-owner"),
                    owner_pid=lease.pid + 1,
                ),
            ),
            (
                "wrong_game",
                _validate(_issue(lease, work_dir, suffix="wrong-game"), game="other"),
            ),
            (
                "wrong_model",
                _validate(
                    _issue(lease, work_dir, suffix="wrong-model"),
                    model_identity=wrong_model,
                ),
            ),
            ("replayed", replayed),
            ("valid", valid_result),
        ]
        rows = [
            {
                "case": case,
                **result,
                "denied_before_model_load": result.get("allowed") is not True,
                "model_load_count": 0,
                "generation_count": 0,
            }
            for case, result in controls
        ]
        return {
            "issuer_kind": "current_kernel_backed_gpu_lease_owner",
            "issuer_secret_exported": False,
            "rows": rows,
        }
    finally:
        _finish_lease(lease)


def run_expiry_controls(work_dir: Path) -> JsonDict:
    """Show that preflight cannot keep a grant valid after its expiry."""

    work_dir.mkdir(parents=True, exist_ok=True)
    missing = validate_authority_bundle_before_model_load({}, [], MODEL_IDENTITY)
    lease = _acquire_lease(work_dir)
    try:
        authority = _issue(lease, work_dir, valid_for_s=1.0, suffix="during-call")
        before = _validate(authority)
        issued_at = datetime.fromisoformat(str(authority["lease_issued_at"]))
        after = _validate(
            authority,
            now_utc=issued_at + timedelta(seconds=2),
            now_ns=int(authority["issued_monotonic_ns"]) + 2_000_000_000,
        )
        return {
            "missing_before_launch": missing,
            "valid_before_call": {key: before[key] for key in ("allowed", "reason")},
            "expired_after_call": {
                "allowed": after["allowed"],
                "reason": after["reason"],
                "provenance_accepted": False,
            },
        }
    finally:
        _finish_lease(lease)


def _authority_child(args: argparse.Namespace) -> int:
    schedule = [{"episode_id": args.episode_id, "game": args.expected_game}]
    model_identity = json.loads(args.expected_model_json)
    preflight = validate_authority_bundle_before_model_load(os.environ, schedule, model_identity)
    if preflight.get("allowed") is not True:
        atomic_write(Path(args.authority_output), {"preflight_allowed": False, **preflight})
        return 3
    authority, episode_environment = select_episode_authority(
        os.environ, schedule[0], model_identity
    )

    class Proposer:
        pass

    proposer = Proposer()
    attached = attach_episode_authority(proposer, authority)
    from carnot.agentic.arc_eval_provenance import _explicit_lease

    caller_mapping = dict(_explicit_lease(getattr(proposer, "arc_eval_authority", None)))
    hops = [
        _authority_handoff_row("child_environment", authority),
        _authority_handoff_row(
            "episode_environment",
            json.loads(episode_environment[lease_api.ARC_AUTHORITY_ENV]),
        ),
        _authority_handoff_row("E3AgentPolicy.proposer", attached),
        _authority_handoff_row("arc_eval_provenance_caller", caller_mapping),
    ]
    atomic_write(
        Path(args.authority_output),
        {
            "preflight_allowed": True,
            "preflight_reason": "allowed",
            "authority_hash": authority["authority_hash"],
            "authority_handoff_rows": hops,
            "model_load_count": 0,
            "generation_count": 0,
        },
    )
    return 0


def run_owned_child_probe(work_dir: Path) -> JsonDict:
    """Launch one real child with an owner-issued public authority bundle."""

    work_dir.mkdir(parents=True, exist_ok=True)
    lease = _acquire_lease(work_dir)
    owner = lease.owner_receipt()
    try:
        authority = _issue(lease, work_dir, suffix="child")
        environment = authority_bundle_environment(os.environ, [authority])
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONPATH"] = f"{REPO_ROOT / 'python'}:{REPO_ROOT}"
        output = work_dir / "child-receipt.json"
        command = [
            sys.executable,
            "-u",
            str(REPO_ROOT / WRAPPER_PATH),
            "--role",
            "authority-child",
            "--date",
            RUN_DATE,
            "--episode-id",
            "r11l:direct_selfparse",
            "--expected-game",
            "r11l",
            "--expected-model-json",
            json.dumps(MODEL_IDENTITY, sort_keys=True),
            "--authority-output",
            str(output),
        ]
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
        exit_code = process.wait(timeout=30)
        child = json.loads(output.read_text(encoding="utf-8"))
        issuer_hop = _authority_handoff_row("gpu_lease_issuer", authority)
        return {
            "child_exit_code": exit_code,
            "owned_child_survives": process.poll() is None,
            "model_load_count": child["model_load_count"],
            "generation_count": child["generation_count"],
            "child_receipt": child,
            "authority_handoff_rows": [issuer_hop, *child["authority_handoff_rows"]],
            "former_field_drop_location": (
                "carnot.experiment_7263_v639_arc_live."
                "run_child_with_lease:child_environment_before_subprocess"
            ),
            "gpu_resource_ownership": owner,
            "episode_authority": authority,
            "command": command,
        }
    finally:
        _finish_lease(lease)


_ENGINE_CODE = """import numpy as np
def engine(grid, action, data):
    return grid + 1
def is_level_complete(grid):
    return bool(np.all(grid >= 1))
"""


def _transition_rows() -> list[Any]:
    from carnot.agentic.arc_executable_world_model import Transition

    return [
        Transition(
            np.full((2, 2), index, dtype=np.int16),
            1,
            None,
            np.full((2, 2), index + 1, dtype=np.int16),
            0,
            0,
        )
        for index in range(8)
    ]


def _chat_reply(content: str) -> JsonDict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"completion_tokens": 20, "prompt_tokens": 100},
    }


def _run_e3_case(case: str, case_dir: Path) -> tuple[JsonDict, list[JsonDict]]:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_induction_tool_loop as tool_loop
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    case_dir.mkdir(parents=True, exist_ok=True)
    if case == "tool_result_to_action":
        answers = [
            _chat_reply(json.dumps({"name": "diff_grids", "arguments": {"t": 0}})),
            _chat_reply(
                json.dumps(
                    {
                        "name": "run_engine_on_transitions",
                        "arguments": {"code": _ENGINE_CODE},
                    }
                )
            ),
        ]
    elif case == "no_tool_needed":
        answers = [
            _chat_reply(
                json.dumps(
                    {
                        "name": "run_engine_on_transitions",
                        "arguments": {"code": _ENGINE_CODE},
                    }
                )
            )
        ]
    else:
        answers = [_chat_reply('{"name":"diff_grids","arguments":[]}')]

    proposer = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=1024, tries=1)
    proposer._ensure_server = lambda: True
    payloads: list[JsonDict] = []
    scripted_events: list[JsonDict] = []

    def request(req: Any, timeout: float | None = None) -> io.BytesIO:
        del timeout
        payload = json.loads(req.data)
        payloads.append(payload)
        response = answers.pop(0)
        scripted_events.append(
            {
                "case": case,
                "request_sha256": _sha256_bytes(req.data),
                "response_sha256": _sha256_bytes(_canonical_bytes(response)),
            }
        )
        return io.BytesIO(json.dumps(response).encode())

    old_urlopen = urllib.request.urlopen
    old_e3_dir = e3.E3_DIR
    environment_keys = {
        "CARNOT_ARC_INDUCE_THINK": "0",
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
        "CARNOT_ARC_INDUCE_TOOL_GRAMMAR": "1",
        "CARNOT_ARC_INDUCE_TOOL_TURNS": "2",
        "CARNOT_ARC_GENERATOR_SEED": "7318",
        "CARNOT_ARC_LLM_BACKEND": "llamacpp",
        "CARNOT_ARC_STALL_REFACTOR_LOOP": "0",
        "CARNOT_ARC_CEGIS_ACCEPT_SPLIT": "1",
        "CARNOT_ARC_STRUCTURED_NAV": "0",
        "CARNOT_ARC_LIVE_TTT": "0",
    }
    old_environment = {key: os.environ.get(key) for key in environment_keys}
    old_disabled = os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    try:
        os.environ.update(environment_keys)
        e3.E3_DIR = case_dir / "engines"
        urllib.request.urlopen = request
        transitions = _transition_rows()
        if case == "malformed_tool":
            dispatches: list[str] = []
            original_dispatch = tool_loop.dispatch_tool

            def dispatch(  # pragma: no cover - a correct grammar rejects before dispatch.
                session: Any, name: str, *args: Any, **kwargs: Any
            ) -> Any:
                dispatches.append(name)
                return original_dispatch(session, name, *args, **kwargs)

            tool_loop.dispatch_tool = dispatch
            try:
                ok, _ = tool_loop.induce_with_tool_loop(proposer, "r11l", transitions, 1)
            finally:
                tool_loop.dispatch_tool = original_dispatch
            return (
                {
                    "case": case,
                    "policy_class": "E3AgentPolicy",
                    "http_completion_count": len(payloads),
                    "tool_dispatch_count": len(dispatches),
                    "successful_tool_result_count": 0,
                    "later_request_contains_first_tool_result": False,
                    "plan_installed": False,
                    "subsequent_environment_action": False,
                    "passed": not ok and len(payloads) == 1 and not dispatches,
                },
                scripted_events,
            )

        policy = E3AgentPolicy("r11l", proposer=proposer, value_head=None)
        policy.transitions = transitions
        policy.root_grid = transitions[0].grid
        policy.cell = 1
        policy._episode_transition_start = 0
        policy.program_synthesis_filter_enabled = False
        policy.active_probe_controller_enabled = False
        policy.think_arm_fallback_enabled = False
        policy.max_refinement_rounds = 1
        policy._induce_and_plan()
        plan_installed = bool(policy.plan)
        move = policy._next_plan_move() if plan_installed else (None, None)
        stats = proposer.last_tool_loop_stats
        later_has_result = len(payloads) > 1 and any(
            '"after": 1' in str(message.get("content", ""))
            for message in payloads[1].get("messages", [])
        )
        passed = bool(
            len(payloads) == (2 if case == "tool_result_to_action" else 1)
            and plan_installed
            and move[0] not in {None, "RESET"}
            and (later_has_result if case == "tool_result_to_action" else True)
        )
        return (
            {
                "case": case,
                "policy_class": type(policy).__name__,
                "http_completion_count": len(payloads),
                "tool_dispatch_count": int(stats.get("tool_calls_total") or 0),
                "successful_tool_result_count": int(stats.get("tool_calls_total") or 0),
                "later_request_contains_first_tool_result": later_has_result,
                "plan_installed": plan_installed,
                "subsequent_environment_action": move[0] not in {None, "RESET"},
                "environment_action": {"action": move[0], "data": move[1]},
                "passed": passed,
            },
            scripted_events,
        )
    finally:
        urllib.request.urlopen = old_urlopen
        e3.E3_DIR = old_e3_dir
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_disabled is not None:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disabled


def run_e3_conformance_panel(work_dir: Path) -> JsonDict:
    """Drive the real E3 policy with model-shaped CPU fixtures only."""

    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    events: list[JsonDict] = []
    for case in ("tool_result_to_action", "no_tool_needed", "malformed_tool"):
        row, case_events = _run_e3_case(case, work_dir / case)
        rows.append(row)
        events.extend(case_events)
    sidecar = work_dir / "scripted_model_events.json"
    atomic_write(
        sidecar,
        {
            "schema": "carnot.experiment_7318.scripted_model_events.v1",
            "counts_as_current_model_invocation": False,
            "events": events,
        },
    )
    return {
        "rows": rows,
        "counts_as_current_model_invocation": False,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": sha256_file(sidecar),
    }


def _source_hashes() -> JsonDict:
    paths = tuple(
        dict.fromkeys(
            (
                *AUTHENTICATED_INPUTS,
                Path("python/carnot/experiment_7263_v639_arc_live.py"),
                Path("scripts/arc_leaderboard_eval.py"),
            )
        )
    )
    historical_paths = {
        Path("results/experiment_7304_v642_arc_receipt.json"),
        Path("results/experiment_7305_v642_arc_selfparse.json"),
    }
    return {
        path.as_posix(): {
            "sha256": sha256_file(REPO_ROOT / path),
            "role": (
                "historical_diagnostic_evidence"
                if path in historical_paths
                else "current_source_or_spec"
            ),
            "authorizes_readiness": path not in historical_paths,
        }
        for path in paths
        if (REPO_ROOT / path).is_file()
    }


def _manifest_rejects_current_task() -> bool:
    try:
        manifest = yaml.safe_load((REPO_ROOT / EXCLUSION_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return True
    if not isinstance(manifest, Mapping):
        return True
    identities = {"7318", EXPERIMENT_ID}
    for key in ("retired", "retired_experiments", "retired_extras"):
        rows = manifest.get(key, [])
        if not isinstance(rows, list):
            return True
        for row in rows:
            if isinstance(row, Mapping) and identities & {
                str(row.get("id")),
                str(row.get("experiment_id")),
            }:
                return True
    return False


def _preconditions(*, issuer_available: bool = True) -> list[JsonDict]:
    rows = [
        gate_check(
            "required_input",
            path.as_posix(),
            "exists",
            True,
            (REPO_ROOT / path).is_file(),
            "The task cannot authenticate a missing source or specification.",
        )
        for path in AUTHENTICATED_INPUTS
    ]
    rows.append(
        gate_check(
            "exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "current_task_rejected",
            False,
            _manifest_rejects_current_task(),
            "A quarantined or retired current task cannot report readiness.",
        )
    )
    spec = (REPO_ROOT / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        gate_check(
            "driving_capability",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7318",
            True,
            "REQ-ARC-WMTE-7318" in spec,
            "Implementation must remain anchored to its named requirement.",
        )
    )
    rows.append(
        gate_check(
            "current_authorized_issuer",
            "gpu_lease_phase_journal.GpuLease",
            "issuer_available",
            True,
            issuer_available,
            "Only the current kernel-lock owner can record an episode grant.",
        )
    )
    return rows


def _repository_health(full_suite_receipt: Mapping[str, Any] | None = None) -> JsonDict:
    historical_failures: list[JsonDict] = [
        {
            "source_experiment": "exp7289-arc-boundary",
            "command": ".venv/bin/pytest -o addopts= tests/python -q --no-cov -n 0",
            "exit_code": 2,
            "log_path": (
                "results/raw/experiment_7289_v641_arc_boundary/validation/04_full_python_suite.log"
            ),
            "log_sha256": (
                "sha256:6016a29aa53fd835cbaf1857e1f59fb2f598258edabb3b5d7f39b6ed16763d80"
            ),
            "resolved": False,
        }
    ]
    if full_suite_receipt is not None and full_suite_receipt.get("passed") is not True:
        historical_failures.append(
            {
                "source_experiment": EXPERIMENT_ID,
                "observed_at_run_date": RUN_DATE,
                "command": full_suite_receipt.get("command"),
                "exit_code": full_suite_receipt.get("exit_code"),
                "timed_out": full_suite_receipt.get("timed_out"),
                "log_path": full_suite_receipt.get("log_path"),
                "log_sha256": full_suite_receipt.get("log_sha256"),
                "output_tail": full_suite_receipt.get("output_tail"),
                "resolved": False,
            }
        )
    return {
        "status": "degraded_open",
        "incident_open": True,
        "affects_required_checks": False,
        "historical_failures": historical_failures,
        "current_full_suite_passed": (
            None if full_suite_receipt is None else full_suite_receipt.get("passed") is True
        ),
    }


def _live_entrypoint_receipt() -> JsonDict:
    files = {
        "live_launcher": Path("python/carnot/experiment_7263_v639_arc_live.py"),
        "environment_construction": MODULE_PATH,
        "scored_caller": Path("scripts/arc_leaderboard_eval.py"),
        "provenance_construction": Path("python/carnot/agentic/arc_eval_provenance.py"),
    }
    return {
        "entrypoint": "carnot.experiment_7263_v639_arc_live.run_child_with_lease",
        "child_entrypoint": "carnot.experiment_7263_v639_arc_live.run_live_session",
        "environment_function": (
            "carnot.experiment_7318_v643_arc_authority.authority_bundle_environment"
        ),
        "episode_environment_function": (
            "carnot.experiment_7318_v643_arc_authority.select_episode_authority"
        ),
        "provenance_caller": "scripts.arc_leaderboard_eval.run_game",
        "provenance_builder": (
            "carnot.agentic.arc_eval_provenance.build_arc_eval_provenance_for_policy"
        ),
        "bounded_runtime_environment": deepcopy(RESOURCE_BOUNDS),
        "file_hashes": {
            name: {"path": path.as_posix(), "sha256": sha256_file(REPO_ROOT / path)}
            for name, path in files.items()
        },
        "production_defaults_changed": False,
    }


def _acceptance_gates(
    denial: Mapping[str, Any],
    handoff: Mapping[str, Any],
    expiry: Mapping[str, Any],
    e3_panel: Mapping[str, Any],
    *,
    validation_passed: bool,
    terminal_linters_passed: bool,
) -> list[JsonDict]:
    denial_rows = list(denial.get("rows") or [])
    expected_denials = {
        "missing",
        "expired",
        "tampered",
        "wrong_owner",
        "wrong_game",
        "wrong_model",
        "replayed",
    }
    observed_denials = {
        str(row.get("case"))
        for row in denial_rows
        if row.get("allowed") is False and row.get("denied_before_model_load") is True
    }
    hashes = [row.get("authority_hash") for row in handoff.get("authority_handoff_rows", [])]
    e3_rows = list(e3_panel.get("rows") or [])
    e3_passed = len(e3_rows) == 3 and all(row.get("passed") is True for row in e3_rows)
    validator_hash = _function_source_hash(
        REPO_ROOT / "python/carnot/agentic/arc_eval_provenance.py",
        "validate_arc_eval_provenance",
    )
    values = (
        (
            "owner_issued_denials",
            sorted(expected_denials),
            sorted(observed_denials),
            "All invalid authority cases must fail before model work.",
        ),
        (
            "issuer_to_caller_handoff",
            True,
            bool(len(hashes) == 5 and len(set(hashes)) == 1),
            "Every runtime hop must retain the exact owner-recorded grant.",
        ),
        (
            "expiry_during_scored_call",
            {
                "missing_before_launch": "missing_authority_bundle",
                "expired_after_call": "authority_expired",
            },
            {
                "missing_before_launch": (expiry.get("missing_before_launch") or {}).get("reason"),
                "expired_after_call": (expiry.get("expired_after_call") or {}).get("reason"),
            },
            "Preflight cannot authorize a missing or later-expired grant.",
        ),
        (
            "strict_provenance_validator_unchanged",
            PROVENANCE_VALIDATOR_SHA256,
            validator_hash,
            "The repair may hand authority to the validator but may not weaken it.",
        ),
        (
            "e3_tool_result_to_action",
            True,
            e3_passed,
            "Mechanism readiness requires the actual E3 policy and both controls.",
        ),
        (
            "zero_current_model_invocations",
            True,
            e3_panel.get("counts_as_current_model_invocation") is False,
            "Scripted model-shaped bytes are fixtures, not current inference.",
        ),
        (
            "affected_scoped_validation",
            True,
            validation_passed,
            "Affected tests, coverage, lint, types, spec checks, and E2E smoke must pass.",
        ),
        (
            "terminal_candidate_linters",
            True,
            terminal_linters_passed,
            "Both independent terminal checks must accept the measured candidate.",
        ),
        (
            "owned_child_cleanup",
            False,
            handoff.get("owned_child_survives"),
            "No task-owned child can remain after the bounded probe.",
        ),
    )
    return [
        {
            "check": check,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "principle": principle,
        }
        for check, expected, observed, principle in values
    ]


def _artifact_row(row: Mapping[str, Any], *, arm: str) -> JsonDict:
    case = str(row.get("case"))
    passed = row.get("passed") is True or (
        arm == "authority_denial_control" and row.get("allowed") is False
    )
    return {
        **deepcopy(dict(row)),
        "unit": case,
        "arm": arm,
        "metrics": {
            "mechanism_passed": passed,
            "model_load_count": int(row.get("model_load_count") or 0),
            "generation_count": int(
                row.get("generation_count") or row.get("http_completion_count") or 0
            ),
        },
        "costs": {
            "current_model_loads": 0,
            "current_model_generations": 0,
            "environment_actions": int(bool(row.get("subsequent_environment_action"))),
        },
        "failures": [] if passed else [str(row.get("reason") or "mechanism_control_failed")],
        "abstentions": 0,
        "censored": False,
        "disposition": "complete",
    }


def _artifact_from_evidence(
    *,
    started_at: datetime,
    started_monotonic: float,
    preconditions: list[JsonDict],
    denial: Mapping[str, Any],
    handoff: Mapping[str, Any],
    expiry: Mapping[str, Any],
    e3_panel: Mapping[str, Any],
    validation_receipts: list[JsonDict],
    validation_passed: bool,
    terminal_linters_passed: bool,
    phase_spans: list[JsonDict],
    repository_health: Mapping[str, Any] | None = None,
) -> JsonDict:
    preconditions_ok = all(row.get("passed") is True for row in preconditions)
    gates = _acceptance_gates(
        denial,
        handoff,
        expiry,
        e3_panel,
        validation_passed=validation_passed,
        terminal_linters_passed=terminal_linters_passed,
    )
    all_gates = preconditions_ok and all(row["passed"] for row in gates)
    blocked = not preconditions_ok
    disqualified = preconditions_ok and not all_gates
    ready = int(all_gates)
    status = "blocked" if blocked else "disqualified" if disqualified else "complete"
    verdict_class = "blocked" if blocked else "disqualified" if disqualified else "null"
    if blocked:
        verdict = "blocked_current_authorized_issuer_unavailable"
    elif disqualified:
        verdict = "complete_disqualified_arc_authority_required_check_failed"
    else:
        verdict = "complete_arc_authority_handoff_mechanism_ready_no_efficacy_claim"
    ended_at = datetime.now(UTC)
    denial_rows = [
        _artifact_row(row, arm="authority_denial_control") for row in denial.get("rows", [])
    ]
    e3_rows = [_artifact_row(row, arm="e3_cpu_conformance") for row in e3_panel.get("rows", [])]
    source_hashes = _source_hashes()
    scripted_sidecar = Path(str(e3_panel.get("sidecar_path", "")))
    if scripted_sidecar.is_file():
        source_hashes[str(scripted_sidecar)] = {
            "sha256": sha256_file(scripted_sidecar),
            "role": "scripted_model_event_conformance_fixture",
            "authorizes_readiness": True,
            "counts_as_current_model_invocation": False,
        }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "inference_substrate": SUBSTRATE,
        "inference_substrate_class": SUBSTRATE,
        "execution_venue": "host",
        "duration_s": max(0.0, time.monotonic() - started_monotonic),
        "phase_spans": phase_spans,
        "random_seed": {"development": 731820260915, "independent_evaluation": 1731820260915},
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": [*denial_rows, *e3_rows],
        "sample_size_budget": {
            "planned": 11,
            "attempted": len(denial_rows) + len(e3_rows),
            "complete": len(denial_rows) + len(e3_rows),
            "censored": 0,
            "stopping_rule": (
                "one valid grant, seven denial controls, and three fixed E3 mechanism cases"
            ),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "validation_receipts": validation_receipts,
        "repository_health": deepcopy(
            dict(repository_health) if repository_health is not None else _repository_health()
        ),
        "field_principles": {},
        "arc_authority_ready_score": ready,
        "authority_handoff_rows": [dict(row) for row in handoff.get("authority_handoff_rows", [])],
        "denial_control_rows": denial_rows,
        "live_entrypoint_receipt": {
            **_live_entrypoint_receipt(),
            "former_field_drop_location": handoff.get("former_field_drop_location"),
            "gpu_resource_ownership": handoff.get("gpu_resource_ownership"),
            "episode_authority": handoff.get("episode_authority"),
            "tested_child_command": handoff.get("command"),
            "expiry_controls": deepcopy(dict(expiry)),
        },
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"Retain the measured {key} value for audit.")
        for key in artifact
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_fixture_artifact(work_dir: Path) -> JsonDict:
    """Build a complete CPU fixture artifact for unit and CLI tests."""

    started_at = datetime.now(UTC)
    started = time.monotonic()
    denial = run_authority_denial_panel(work_dir / "denials")
    handoff = run_owned_child_probe(work_dir / "handoff")
    expiry = run_expiry_controls(work_dir / "expiry")
    e3_panel = run_e3_conformance_panel(work_dir / "e3")
    receipt = {
        "name": "unit_fixture_validation",
        "command": "in_process_exp7318_fixture_assertions",
        "scope": "CPU authority and E3 mechanism fixtures",
        "exit_code": 0,
        "duration_s": time.monotonic() - started,
        "log_path": None,
        "log_sha256": None,
        "passed": True,
    }
    return _artifact_from_evidence(
        started_at=started_at,
        started_monotonic=started,
        preconditions=_preconditions(),
        denial=denial,
        handoff=handoff,
        expiry=expiry,
        e3_panel=e3_panel,
        validation_receipts=[receipt],
        validation_passed=True,
        terminal_linters_passed=True,
        phase_spans=[
            {
                "phase": "cpu_fixture_panel",
                "started_monotonic_s": started,
                "ended_monotonic_s": time.monotonic(),
                "units": 11,
                "checkpoint_boundary": "fixture_complete",
                "pending_operations": [],
            }
        ],
    )


def build_blocked_fixture_artifact(work_dir: Path) -> JsonDict:
    """Build the exact terminal block used when no current issuer exists."""

    work_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    return _artifact_from_evidence(
        started_at=started_at,
        started_monotonic=started,
        preconditions=[
            gate_check(
                "current_authorized_issuer",
                "gpu_lease_phase_journal.GpuLease",
                "issuer_available",
                True,
                False,
                "Only the current kernel-lock owner can record an episode grant.",
            )
        ],
        denial={"rows": []},
        handoff={"authority_handoff_rows": [], "owned_child_survives": False},
        expiry={},
        e3_panel={"rows": [], "counts_as_current_model_invocation": False},
        validation_receipts=[],
        validation_passed=False,
        terminal_linters_passed=False,
        phase_spans=[],
    )


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check current declarations, denial rows, gates, and checksum."""

    errors: list[str] = []
    required = set(FIELD_PRINCIPLES)
    if missing := sorted(required - set(value)):
        errors.extend(f"missing required field: {field}" for field in missing)
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema or experiment identity mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("milestone or run date mismatch")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_invoked") is not False
        or value.get("invocation_counts") != INVOCATION_COUNTS
        or value.get("inference_substrate") != SUBSTRATE
        or value.get("inference_substrate_class") != SUBSTRATE
        or value.get("execution_venue") != "host"
    ):
        errors.append("current model declaration mismatch")
    status = value.get("status")
    verdict_class = value.get("verdict_class")
    score = value.get("arc_authority_ready_score")
    if status not in {"complete", "blocked", "disqualified"}:
        errors.append("terminal status mismatch")
    if verdict_class not in {"null", "blocked", "disqualified"}:
        errors.append("verdict class mismatch")
    checks = value.get("preconditions_checked")
    checks = checks if isinstance(checks, list) else []
    summary = gate_summary(checks)
    if value.get("gate_check_summary") != summary:
        errors.append("gate check summary mismatch")
    denial_rows = value.get("denial_control_rows")
    denial_rows = denial_rows if isinstance(denial_rows, list) else []
    denial_cases = {
        str(row.get("case"))
        for row in denial_rows
        if isinstance(row, Mapping) and row.get("allowed") is False
    }
    required_denials = {
        "missing",
        "expired",
        "tampered",
        "wrong_owner",
        "wrong_game",
        "wrong_model",
        "replayed",
    }
    if status == "complete" and denial_cases != required_denials:
        errors.append("denial controls mismatch")
    gates = value.get("acceptance_gate_results")
    gates = gates if isinstance(gates, list) else []
    all_gates = bool(gates) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in gates
    )
    if status == "complete" and (score != 1 or not all_gates or verdict_class != "null"):
        errors.append("complete readiness mismatch")
    if status in {"blocked", "disqualified"} and score != 0:
        errors.append("unsafe readiness on non-ready artifact")
    if status == "blocked" and (
        verdict_class != "blocked"
        or not str(value.get("honest_verdict", "")).startswith("blocked_")
    ):
        errors.append("blocked verdict mismatch")
    if status != "blocked" and not str(value.get("honest_verdict", "")).startswith("complete_"):
        errors.append("complete verdict prefix mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field principles mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility checksum mismatch")
    return list(dict.fromkeys(errors))


def _run_scoped_validation(raw_dir: Path) -> JsonDict:  # pragma: no cover - subprocess E2E.
    from carnot.reporting.experiment_7303_validation_scope import run_scoped_validation

    private = Path(tempfile.mkdtemp(prefix="exp7318-validation-", dir="/tmp"))
    return run_scoped_validation(
        REPO_ROOT,
        [
            TEST_PATH.as_posix(),
            "tests/python/test_arc_eval_provenance_contract_20260905.py",
            "tests/python/test_arc_induction_state_persistence.py",
            "tests/python/test_arc_tool_grammar_transport.py",
            "tests/python/test_gpu_lease_phase_journal.py",
            "tests/python/test_experiment_7263_v639_arc_live.py",
        ],
        [MODULE_PATH.as_posix()],
        static_paths=[
            WRAPPER_PATH.as_posix(),
            "python/carnot/gpu_lease_phase_journal.py",
            "python/carnot/experiment_7263_v639_arc_live.py",
            "python/carnot/agentic/arc_eval_provenance.py",
            "scripts/arc_leaderboard_eval.py",
        ],
        basetemp=private,
        coverage_file=private / ".coverage",
        log_dir=raw_dir / "validation",
        historical_failures=_repository_health()["historical_failures"],
    )


def _run_full_python_suite(raw_dir: Path) -> list[JsonDict]:  # pragma: no cover - required once.
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    return run_commands(
        REPO_ROOT,
        [
            CommandSpec(
                "full_python_suite",
                (str(REPO_ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
                "repository_python_suite_required_by_exp7318",
                timeout_s=3600.0,
            )
        ],
        log_dir=raw_dir / "full-suite",
        heartbeat_s=60.0,
    )


def _prior_full_suite_receipt(output: Path) -> JsonDict | None:  # pragma: no cover - resume path.
    try:
        prior = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if prior.get("experiment_id") != EXPERIMENT_ID:
        return None
    receipt = next(
        (
            row
            for row in prior.get("validation_receipts", [])
            if isinstance(row, Mapping) and row.get("name") == "full_python_suite"
        ),
        None,
    )
    return deepcopy(dict(receipt)) if receipt is not None else None


def _run_smoke(raw_dir: Path) -> list[JsonDict]:  # pragma: no cover - subprocess E2E.
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    python = str(REPO_ROOT / ".venv/bin/python")
    private_output = Path("/tmp/exp7318-e2e009-llm-off.json")
    commands = [
        CommandSpec(
            "e2e_009_llm_off_environment_smoke",
            (
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private_output),
            ),
            "E2E-009 prescribed LLM-off environment smoke",
        )
    ]
    return run_commands(
        REPO_ROOT,
        commands,
        log_dir=raw_dir / "environment-smoke",
        extra_env={"CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=60.0,
    )


def _run_terminal_linters(
    raw_dir: Path, candidate: Path
) -> list[JsonDict]:  # pragma: no cover - subprocess validation.
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    python = str(REPO_ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "terminal_candidate_adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured terminal candidate",
        ),
        CommandSpec(
            "terminal_candidate_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured terminal candidate",
        ),
    ]
    return run_commands(
        REPO_ROOT,
        commands,
        log_dir=raw_dir / "terminal-validation",
        heartbeat_s=60.0,
    )


def run_experiment(output: Path) -> JsonDict:  # pragma: no cover - terminal orchestration.
    started_at = datetime.now(UTC)
    started = time.monotonic()
    raw_dir = REPO_ROOT / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []
    _progress(started, "startup", "entrypoint_and_paths_authenticated", output=str(output))
    preconditions = _preconditions()
    if not all(row["passed"] for row in preconditions):
        artifact = build_blocked_fixture_artifact(raw_dir / "blocked")
        artifact["preconditions_checked"] = preconditions
        artifact["gate_check_summary"] = gate_summary(preconditions)
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_write(output, artifact)
        return artifact

    _progress(started, "authority_panel", "begin")
    phase_start = time.monotonic()
    denial = run_authority_denial_panel(raw_dir / "denials")
    handoff = run_owned_child_probe(raw_dir / "handoff")
    expiry = run_expiry_controls(raw_dir / "expiry")
    spans.append(
        {
            "phase": "authority_panel",
            "started_monotonic_s": phase_start,
            "ended_monotonic_s": time.monotonic(),
            "units": 9,
            "checkpoint_boundary": "authority_controls_complete",
            "pending_operations": [],
        }
    )
    _progress(started, "authority_panel", "end", completed_units=9)
    _progress(started, "e3_cpu_conformance", "begin")
    phase_start = time.monotonic()
    e3_panel = run_e3_conformance_panel(raw_dir / "e3")
    spans.append(
        {
            "phase": "e3_cpu_conformance",
            "started_monotonic_s": phase_start,
            "ended_monotonic_s": time.monotonic(),
            "units": 3,
            "checkpoint_boundary": "e3_controls_complete",
            "pending_operations": [],
        }
    )
    _progress(started, "e3_cpu_conformance", "end", completed_units=3)

    _progress(started, "full_python_suite", "before_subprocess_group")
    phase_start = time.monotonic()
    prior_full_suite = _prior_full_suite_receipt(output)
    if prior_full_suite is None:
        full_suite_receipts = _run_full_python_suite(raw_dir)
    else:
        prior_full_suite["reused_as_repository_health_observation"] = True
        prior_full_suite["current_required_check"] = False
        full_suite_receipts = [prior_full_suite]
        _progress(started, "full_python_suite", "prior_terminal_receipt_reused")
    receipts = [dict(row) for row in full_suite_receipts]
    full_suite_passed = bool(
        len(full_suite_receipts) == 1 and full_suite_receipts[0].get("passed") is True
    )
    spans.append(
        {
            "phase": "full_python_suite",
            "started_monotonic_s": phase_start,
            "ended_monotonic_s": time.monotonic(),
            "units": 1,
            "checkpoint_boundary": "full_python_suite_complete",
            "pending_operations": [],
        }
    )
    _progress(started, "full_python_suite", "after_subprocess_group", passed=full_suite_passed)

    _progress(started, "scoped_validation", "before_subprocess_group")
    phase_start = time.monotonic()
    validation = _run_scoped_validation(raw_dir)
    receipts.extend(dict(row) for row in validation["validation_receipts"])
    smoke_receipts = _run_smoke(raw_dir)
    receipts.extend(smoke_receipts)
    validation_passed = bool(
        validation["required_checks_passed"]
        and smoke_receipts
        and smoke_receipts[0].get("passed") is True
    )
    repository_health = _repository_health(full_suite_receipts[0])
    spans.append(
        {
            "phase": "scoped_validation",
            "started_monotonic_s": phase_start,
            "ended_monotonic_s": time.monotonic(),
            "units": len(receipts),
            "checkpoint_boundary": "affected_checks_complete",
            "pending_operations": [],
        }
    )
    _progress(started, "scoped_validation", "after_subprocess_group", passed=validation_passed)

    candidate = _artifact_from_evidence(
        started_at=started_at,
        started_monotonic=started,
        preconditions=preconditions,
        denial=denial,
        handoff=handoff,
        expiry=expiry,
        e3_panel=e3_panel,
        validation_receipts=receipts,
        validation_passed=validation_passed,
        terminal_linters_passed=False,
        phase_spans=spans,
        repository_health=repository_health,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_write(candidate_path, candidate)
    _progress(started, "terminal_validation", "before_subprocess_group")
    terminal_receipts = _run_terminal_linters(raw_dir, candidate_path)
    receipts.extend(terminal_receipts)
    terminal_passed = bool(
        len(terminal_receipts) == 2 and all(row.get("passed") is True for row in terminal_receipts)
    )
    _progress(started, "terminal_validation", "after_subprocess_group", passed=terminal_passed)
    artifact = _artifact_from_evidence(
        started_at=started_at,
        started_monotonic=started,
        preconditions=preconditions,
        denial=denial,
        handoff=handoff,
        expiry=expiry,
        e3_panel=e3_panel,
        validation_receipts=receipts,
        validation_passed=validation_passed,
        terminal_linters_passed=terminal_passed,
        phase_spans=spans,
        repository_health=repository_health,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("terminal artifact validation failed: " + "; ".join(errors))
    atomic_write(output, artifact)
    _progress(started, "publication", "terminal_artifact_written", path=str(output))
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_PATH))
    parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--role", choices=["experiment", "authority-child"], default="experiment")
    parser.add_argument("--episode-id", default="r11l:direct_selfparse")
    parser.add_argument("--expected-game", default="r11l")
    parser.add_argument("--expected-model-json", default=json.dumps(MODEL_IDENTITY, sort_keys=True))
    parser.add_argument("--authority-output", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    _progress(started, "startup", "arguments_parsed", role=args.role)
    if args.role == "authority-child":
        return _authority_child(args)
    output = Path(args.output)
    artifact = (
        build_fixture_artifact(output.parent / "exp7318-fixture")
        if args.fixture_only
        else run_experiment(output)
    )
    if args.fixture_only:
        atomic_write(output, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin module execution guard.
    raise SystemExit(main())
