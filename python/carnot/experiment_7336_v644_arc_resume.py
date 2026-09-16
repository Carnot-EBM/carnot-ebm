"""Validate bound selfparse result resumption without current model work.

Spec refs: REQ-ARC-WMTE-7336 and SCENARIO-ARC-WMTE-7336-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any
import urllib.request

import numpy as np

from carnot.agentic.arc_selfparse_result_resume import ResultResumeGuard
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.644"
EXPERIMENT_ID = "exp7336-arc-resume"
SCHEMA = "carnot.experiment_7336.v644.arc_resume.v1"
MODEL_SPECS: list[JsonDict] = []
DEVELOPMENT_SEED = 7_336_202_609_16
EVALUATION_SEED = 17_336_202_609_16
RESAMPLING_SEED = 27_336_202_609_16
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7319_RESULT_PATH = Path("results/experiment_7319_v643_arc_session.json")
EXP7319_RAW_PATH = Path("results/raw/experiment_7319_v643_arc_session")
RESULT_PATH = Path("results/experiment_7336_v644_arc_resume.json")
RAW_DIR = Path("results/raw/experiment_7336_v644_arc_resume")
MODULE_PATH = Path("python/carnot/experiment_7336_v644_arc_resume.py")
RESUME_MODULE_PATH = Path("python/carnot/agentic/arc_selfparse_result_resume.py")
LOOP_MODULE_PATH = Path("python/carnot/agentic/arc_induction_tool_loop.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7336_v644_arc_resume.py")
TEST_PATH = Path("tests/python/test_experiment_7336_v644_arc_resume.py")

REQUIRED_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_offline_smoke")
ZERO_INVOCATION_COUNTS = {
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
    "usable_answers": 0,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _load_json(path: Path) -> JsonDict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def trace_exp7319_first_loss(root: Path) -> JsonDict:
    """Trace the exact Exp7319 result that had no remaining request slot."""

    result_path = root / EXP7319_RESULT_PATH
    raw = root / EXP7319_RAW_PATH
    request_path = raw / "re86__direct_selfparse/requests/01_request.json"
    response_path = raw / "re86__direct_selfparse/requests/01_response.json"
    tool_path = raw / "tool_events.jsonl"
    session_path = raw / "live_session.json"
    required = (result_path, request_path, response_path, tool_path, session_path)
    failed = [
        {
            "check": "historical_input_available",
            "path": str(path),
            "expected": True,
            "observed": path.is_file(),
        }
        for path in required
        if not path.is_file()
    ]
    if failed:
        return {
            "reproduced": False,
            "failed_checks": failed,
            "first_lost_link": None,
            "live_continuation_skipped": True,
        }
    artifact = _load_json(result_path) or {}
    request = _load_json(request_path) or {}
    response = _load_json(response_path) or {}
    session = _load_json(session_path) or {}
    events = _load_jsonl(tool_path)
    event = events[0] if events else {}
    requests = [row for row in session.get("requests", []) if isinstance(row, Mapping)]
    episodes = [row for row in session.get("episodes", []) if isinstance(row, Mapping)]
    episode = episodes[0] if episodes else {}
    attempts = [row for row in episode.get("induction_rows", []) if isinstance(row, Mapping)]
    attempt = attempts[0] if attempts else {}
    note = str(attempt.get("proposer_note") or "")
    exit_reason = (
        "episode_generation_call_limit_reached"
        if "episode_generation_call_limit_reached" in note
        else str((attempt.get("tool_gap") or {}).get("terminated_by") or "missing")
    )
    bounded = str(event.get("bounded_response") or "")
    later = requests[2:]
    delivered = False
    for row in later:
        later_path = Path(str(row.get("request_path") or ""))
        if later_path.is_file() and bounded in later_path.read_text(encoding="utf-8"):
            delivered = True
    completion_limit = int((artifact.get("sample_size_budget") or {}).get("completion_limit", 0))
    token_limit = int((artifact.get("sample_size_budget") or {}).get("generated_token_limit", 0))
    generated = int(episode.get("generated_tokens") or 0)
    checks = {
        "two_calls_completed": len(requests) == completion_limit == COMPLETION_LIMIT,
        "successful_runtime_result": (event.get("dispatch_result") or {}).get("ok") is True,
        "bounded_payload_created": bool(bounded),
        "later_request_absent": not delivered and len(later) == 0,
        "call_budget_exhausted": completion_limit - len(requests) == 0,
        "token_budget_not_exhausted": generated < token_limit == GENERATED_TOKEN_LIMIT,
        "actual_exit_reason": exit_reason == "episode_generation_call_limit_reached",
        "artifact_reports_missing_link": (
            (artifact.get("tool_use_chain") or {}).get("results_in_later_requests") == 0
            and (artifact.get("tool_use_chain") or {}).get("policy_consumed_results") == 0
        ),
    }
    failed.extend(
        {
            "check": name,
            "path": EXP7319_RESULT_PATH.as_posix(),
            "expected": True,
            "observed": observed,
        }
        for name, observed in checks.items()
        if not observed
    )
    return {
        "reproduced": not failed,
        "failed_checks": failed,
        "first_lost_link": "next_request_payload_delivery" if not failed else None,
        "source_request_index": 1,
        "source_request_sha256": sha256_file(request_path),
        "source_response_sha256": sha256_file(response_path),
        "tool_event_sha256": sha256_file(tool_path),
        "tool_name": event.get("parsed_tool"),
        "tool_arguments": deepcopy(event.get("parsed_arguments")),
        "runtime_result_ok": (event.get("dispatch_result") or {}).get("ok") is True,
        "payload_created": bool(bounded),
        "payload_sha256": "sha256:" + hashlib.sha256(bounded.encode()).hexdigest(),
        "payload_delivered_in_later_request": delivered,
        "receipt_captured": False,
        "completion_limit": completion_limit,
        "completed_calls": len(requests),
        "remaining_call_budget": completion_limit - len(requests),
        "generated_tokens": generated,
        "generated_token_limit": token_limit,
        "token_budget_exhausted": generated >= token_limit,
        "exit_reason": exit_reason,
        "request_messages": len(request.get("messages") or []),
        "response_finish_reason": ((response.get("choices") or [{}])[0]).get("finish_reason"),
        "live_continuation_skipped": False,
    }


_ENGINE_CODE = """import numpy as np
def engine(grid, action, data):
    return grid + 1
def is_level_complete(grid):
    return bool(np.all(grid >= 1))
"""


def _transitions() -> list[Any]:
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


def _reply(content: str) -> JsonDict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"completion_tokens": 20, "prompt_tokens": 100},
    }


_DIFF_XML = (
    "</think>\n<tool_call>\n<function=diff_grids>\n"
    "<parameter=t>\n0\n</parameter>\n</function>\n</tool_call>"
)
_FINAL_CODE = f"```python\n{_ENGINE_CODE}\n```"


@contextmanager
def _case_environment(case: str) -> Any:
    updates = {
        "CARNOT_ARC_INDUCE_THINK": "0",
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
        "CARNOT_ARC_INDUCE_TOOL_TURNS": "1" if case == "result_withheld" else "2",
        "CARNOT_ARC_GENERATOR_SEED": "7336",
        "CARNOT_ARC_LLM_BACKEND": "llamacpp",
        "CARNOT_ARC_STALL_REFACTOR_LOOP": "0",
        "CARNOT_ARC_CEGIS_ACCEPT_SPLIT": "1",
        "CARNOT_ARC_STRUCTURED_NAV": "0",
        "CARNOT_ARC_LIVE_TTT": "0",
    }
    if case != "result_withheld":
        updates["CARNOT_ARC_SELFPARSE_RESULT_RESUME"] = "1"
    keys = {*updates, "CARNOT_ARC_SELFPARSE_RESULT_RESUME", "CARNOT_ARC_DISABLE_INDUCTION"}
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        os.environ.pop("CARNOT_ARC_SELFPARSE_RESULT_RESUME", None)
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_resume_case(case: str, case_dir: Path) -> tuple[JsonDict, list[JsonDict]]:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    case_dir.mkdir(parents=True, exist_ok=True)
    answers = [_reply(_FINAL_CODE)] if case == "no_tool_needed" else [_reply(_DIFF_XML)]
    if case == "tool_needed_changed_input":
        answers.append(_reply(_FINAL_CODE))
    proposer = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=2048, tries=1)
    proposer._ensure_server = lambda: True
    if case == "result_withheld":
        proposer.generate = lambda *args, **kwargs: (False, "withheld control")
    payloads: list[JsonDict] = []
    fixture_events: list[JsonDict] = []

    def scripted_urlopen(request: Any, timeout: float | None = None) -> io.BytesIO:
        del timeout
        payload = json.loads(request.data)
        payloads.append(payload)
        response = answers.pop(0)
        fixture_events.append(
            {
                "arm": case,
                "request_sha256": "sha256:" + hashlib.sha256(request.data).hexdigest(),
                "response_sha256": "sha256:"
                + hashlib.sha256(_canonical_bytes(response)).hexdigest(),
                "model_shaped_fixture": True,
            }
        )
        return io.BytesIO(json.dumps(response).encode())

    old_urlopen = urllib.request.urlopen
    old_e3_dir = e3.E3_DIR
    try:
        with _case_environment(case):
            e3.E3_DIR = case_dir / "engines"
            urllib.request.urlopen = scripted_urlopen
            transitions = _transitions()
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
            stats = deepcopy(getattr(proposer, "last_tool_loop_stats", {}) or {})
    finally:
        urllib.request.urlopen = old_urlopen
        e3.E3_DIR = old_e3_dir
    resume = stats.get("result_resume") if isinstance(stats.get("result_resume"), Mapping) else {}
    result_rows = [row for row in resume.get("result_rows", []) if isinstance(row, Mapping)]
    first_result = result_rows[0] if result_rows else {}
    result_text = str(first_result.get("bounded_response") or "")
    later_occurrences = (
        sum(
            str(message.get("content") or "").count(result_text)
            for message in payloads[1].get("messages", [])
            if isinstance(message, Mapping)
        )
        if len(payloads) > 1 and result_text
        else 0
    )
    tool_count = int(stats.get("tool_calls_total") or 0)
    if case == "result_withheld":
        passed = tool_count == 1 and len(payloads) == 1 and not plan_installed
    elif case == "no_tool_needed":
        passed = (
            tool_count == 0
            and len(payloads) == 1
            and plan_installed
            and move[0] not in {None, "RESET"}
        )
    else:
        passed = (
            tool_count == 1
            and len(payloads) == 2
            and first_result.get("delivery_count") == 1
            and first_result.get("receipt_captured") is True
            and later_occurrences == 1
            and plan_installed
            and move[0] not in {None, "RESET"}
        )
    return (
        {
            "unit": "r11l:cpu_fixture",
            "arm": case,
            "policy_class": type(policy).__name__,
            "completion_limit": COMPLETION_LIMIT,
            "completion_calls": len(payloads),
            "generated_token_limit": GENERATED_TOKEN_LIMIT,
            "successful_runtime_results": tool_count,
            "result_delivery_count": int(first_result.get("delivery_count") or 0),
            "receipt_captured": first_result.get("receipt_captured") is True,
            "later_request_result_occurrences": later_occurrences,
            "input_changed": bool(np.any(transitions[0].grid != transitions[0].next_grid)),
            "verified_engine_installed": plan_installed,
            "plan_installed": plan_installed,
            "later_policy_action": move[0] not in {None, "RESET"},
            "environment_action": {"action": move[0], "data": move[1]},
            "attempted_calls": len(payloads),
            "passed": passed,
            "failures": [] if passed else [str(stats.get("terminated_by") or "control_failed")],
            "abstentions": int(not plan_installed),
            "censored": False,
            "resume_receipt": resume,
        },
        fixture_events,
    )


def run_resume_control_panel(work_dir: Path) -> JsonDict:
    """Run positive and causal controls through the actual scored policy class."""

    rows: list[JsonDict] = []
    events: list[JsonDict] = []
    for case in ("tool_needed_changed_input", "result_withheld", "no_tool_needed"):
        row, case_events = _run_resume_case(case, work_dir / case)
        rows.append(row)
        events.extend(case_events)
    sidecar = work_dir / "scripted_model_events.json"
    atomic_write(
        sidecar,
        {
            "schema": "carnot.experiment_7336.scripted_model_events.v1",
            "counts_as_current_model_invocation": False,
            "events": events,
        },
    )
    return {
        "rows": rows,
        "counts_as_current_model_invocation": False,
        "fixture_sidecar_path": str(sidecar),
        "fixture_sidecar_sha256": sha256_file(sidecar),
    }


def run_rejection_panel() -> list[JsonDict]:
    """Exercise every result-binding denial without an HTTP or model call."""

    rows: list[JsonDict] = []
    for case in (
        "stale_episode",
        "mismatched_attempt",
        "expired_authority",
        "duplicate_result",
        "timeout_after_dispatch",
        "absent_result",
    ):
        guard = ResultResumeGuard("game:episode", "attempt:0", 2, 200.0)
        if case != "absent_result":
            offered = guard.offer_result(
                source_request_id="request:0",
                next_request_id="request:1",
                tool_names=["diff_grids"],
                bounded_response='<tool_response>\n{"ok": true}\n</tool_response>',
                dispatch_results=[{"ok": True}],
            )
        else:
            offered = {"accepted": False}
        if case == "duplicate_result":
            guard.offer_result(
                source_request_id="request:0",
                next_request_id="request:1",
                tool_names=["diff_grids"],
                bounded_response='<tool_response>\n{"ok": true}\n</tool_response>',
                dispatch_results=[{"ok": True}],
            )
        elif case == "stale_episode":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="other",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
        elif case == "mismatched_attempt":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:1",
                now_monotonic=100.0,
            )
        elif case == "expired_authority":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:0",
                now_monotonic=201.0,
            )
        elif case == "timeout_after_dispatch":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
            guard.complete_request(request_id="request:1", response_received=False, timed_out=True)
        elif case == "absent_result":
            guard.prepare_next_request(
                request_id="request:1",
                episode_id="game:episode",
                attempt_id="attempt:0",
                now_monotonic=100.0,
            )
        receipt = guard.receipt()
        rejection = receipt["rejections"][-1]
        rows.append(
            {
                "unit": "result_binding",
                "arm": case,
                "case": case,
                "reason": rejection["reason"],
                "accepted": False,
                "plan_authorized": False,
                "attempted_calls_retained": max(1, len(receipt["request_rows"])),
                "offer_was_accepted_before_denial": offered.get("accepted") is True,
                "passed": rejection["reason"] == case,
                "failures": [],
                "abstentions": 1,
                "censored": False,
            }
        )
    return rows


def independent_reduce(path: Path) -> JsonDict:
    """Reduce only raw CPU rows so the terminal builder cannot grade itself."""

    value = _load_json(path) or {}
    rows = [row for row in value.get("rows", []) if isinstance(row, Mapping)]
    by_arm = {str(row.get("arm")): row for row in rows}
    control_names = {"tool_needed_changed_input", "result_withheld", "no_tool_needed"}
    rejection_names = {
        "stale_episode",
        "mismatched_attempt",
        "expired_authority",
        "duplicate_result",
        "timeout_after_dispatch",
        "absent_result",
    }
    expected = control_names | rejection_names
    passed = set(by_arm) == expected and all(
        by_arm[name].get("passed") is True for name in expected
    )
    positive = by_arm.get("tool_needed_changed_input", {})
    passed = passed and positive.get("result_delivery_count") == 1
    passed = passed and positive.get("receipt_captured") is True
    passed = passed and positive.get("later_policy_action") is True
    return {
        "arc_resume_ready_score": int(passed),
        "row_count": len(rows),
        "expected_arms": sorted(expected),
        "observed_arms": sorted(by_arm),
    }


def _receipt_names_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    principles = {
        "schema": "Version the artifact while keeping ordinary experiment identity fields.",
        "status": "Publish a terminal result only after current work and affected validation.",
        "run_date": "Use 20260916 and preserve actual UTC timestamps.",
        "preconditions_checked": "Name each input identity, availability check, and failure.",
        "MODEL_SPECS": "List current executable model identities; this CPU task has none.",
        "model_invoked": "Set true for any real current load or generation attempt.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and active calls.",
        "inference_substrate": "Declare the actual CPU exact simulator computation.",
        "inference_substrate_class": "Declare the same actual non-model duration class.",
        "execution_venue": "Use host; historical GPU evidence is not current execution.",
        "duration_s": "Measure real monotonic elapsed time without padding.",
        "phase_spans": "Use disjoint spans with units, checkpoints, and pending operations.",
        "random_seed": "Freeze development, evaluation, and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, inputs, settings, evaluator, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers without granting historical readiness.",
        "rows": "Give each arm metrics, costs, failures, abstentions, and censoring.",
        "sample_size_budget": "Record planned, attempted, complete, and censored units with stopping rules.",
        "acceptance_gate_results": "Record expected, observed, and passed for every gate.",
        "gate_check_summary": "Name the first failed upstream, field, expected, and observed value.",
        "verifier_is_oracle": "True because the same exact execution authority grades correctness.",
        "honest_verdict": "Use a complete finding prefix and avoid an efficacy claim.",
        "verdict_class": "Use the closed terminal enum; this mechanism receipt is a null finding.",
        "validation_receipts": "Record exact command, scope, exit, duration, and log hash, including failures.",
        "repository_health": "Keep unrelated dated failures separate from affected required checks.",
        "field_principles": "Explain fields without wrapping their executable values.",
        "arc_resume_ready_score": "Require reproduced loss and a changed shipped result-to-action path.",
        "first_loss_receipt": "Bind the failing transition and its actual call budget and exit reason.",
        "resume_control_rows": "Separate CPU scripted handoff controls from current inference.",
        "solve_provenance": "State that this CPU transport fixture claims no game solve.",
    }
    return {
        key: principles.get(key, f"Retain the ordinary {key} evidence field.") for key in artifact
    }


def _artifact_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "unit": row.get("unit"),
            "arm": row.get("arm"),
            "metrics": {
                "passed": int(row.get("passed") is True),
                "result_delivery_count": int(row.get("result_delivery_count") or 0),
                "later_policy_action": int(row.get("later_policy_action") is True),
            },
            "costs": {
                "completion_calls": int(row.get("completion_calls") or 0),
                "current_model_calls": 0,
            },
            "failures": list(row.get("failures") or []),
            "abstentions": int(row.get("abstentions") or 0),
            "censored": bool(row.get("censored")),
        }
        for row in rows
    ]


def _gate(name: str, expected: Any, observed: Any, principle: str) -> JsonDict:
    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def build_terminal_artifact(
    *,
    first_loss: Mapping[str, Any],
    control_panel: Mapping[str, Any],
    rejection_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    e2e_receipts: Sequence[Mapping[str, Any]],
    terminal_lint_receipts: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    raw_rows_path: Path,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    control_rows = [dict(row) for row in control_panel.get("rows", []) if isinstance(row, Mapping)]
    all_rows = [*control_rows, *[dict(row) for row in rejection_rows]]
    independent = independent_reduce(raw_rows_path)
    gates = [
        _gate(
            "first_loss_reproduced",
            True,
            first_loss.get("reproduced"),
            "Change only a measured loss.",
        ),
        _gate(
            "cpu_controls_complete",
            1,
            independent["arc_resume_ready_score"],
            "Require all causal controls.",
        ),
        _gate(
            "affected_scoped_validation",
            True,
            _receipt_names_pass(validation_receipts, REQUIRED_VALIDATION_NAMES),
            "Every affected check must pass.",
        ),
        _gate(
            "applicable_e2e",
            True,
            _receipt_names_pass(e2e_receipts, REQUIRED_E2E_NAMES),
            "Run real request construction and offline plumbing.",
        ),
        _gate(
            "terminal_linters",
            True,
            _receipt_names_pass(
                terminal_lint_receipts,
                ("adversarial_verify", "verdict_row_consistency_strict"),
            ),
            "A candidate must survive both terminal readers.",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    preconditions = [
        {
            "check": "exp7319_identity_and_availability",
            "upstream": EXP7319_RESULT_PATH.as_posix(),
            "artifact_field": "first_loss_reproduced",
            "expected": True,
            "observed": first_loss.get("reproduced"),
            "passed": first_loss.get("reproduced") is True,
        }
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "complete" if ready else "disqualified",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(0.000001, float(duration_s)), 6),
        "phase_spans": [dict(row) for row in phase_spans],
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": _artifact_rows(all_rows),
        "sample_size_budget": {
            "planned_units": 9,
            "attempted_units": len(all_rows),
            "completed_units": sum(row.get("passed") is True for row in all_rows),
            "censored_units": sum(bool(row.get("censored")) for row in all_rows),
            "completion_limit_per_control": COMPLETION_LIMIT,
            "generated_token_limit_per_control": GENERATED_TOKEN_LIMIT,
            "stopping_rule": "three frozen controls and six frozen rejection cases; no outcome extension",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "all_passed": all(row["passed"] for row in preconditions),
            "failed_count": sum(not row["passed"] for row in preconditions),
            "first_failure": next((row for row in preconditions if not row["passed"]), None),
        },
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_null_cpu_result_resume_mechanism_ready_no_game_solve"
            if ready
            else "complete_disqualified_affected_validation_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "validation_receipts": [
            *[dict(row) for row in validation_receipts],
            *[dict(row) for row in e2e_receipts],
            *[dict(row) for row in terminal_lint_receipts],
        ],
        "repository_health": {
            "status": "not_reassessed",
            "observed_at": RUN_DATE,
            "affects_required_checks": False,
            "unrelated_failures": [],
        },
        "arc_resume_ready_score": ready,
        "first_loss_receipt": deepcopy(dict(first_loss)),
        "resume_control_rows": all_rows,
        "independent_reduction": independent,
        "fixture_sidecar": {
            "path": control_panel.get("fixture_sidecar_path"),
            "sha256": control_panel.get("fixture_sidecar_sha256"),
            "counts_as_current_model_invocation": False,
        },
        "code_or_observability_changed": {"code": True, "observability": True},
        "production_default_changed": False,
        "solve_provenance": "no_game_solve_cpu_transport_fixture",
        "promotion_value": 0,
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = (
        "Explain each field without wrapping executable values or ordinary dictionaries."
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_null_artifact(
    *,
    first_loss: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
) -> JsonDict:
    raw_path = Path(tempfile.mkdtemp(prefix="exp7336-null-")) / "rows.json"
    atomic_write(raw_path, {"rows": []})
    artifact = build_terminal_artifact(
        first_loss=first_loss,
        control_panel={"rows": []},
        rejection_rows=[],
        validation_receipts=[],
        e2e_receipts=[],
        terminal_lint_receipts=[],
        source_hashes=source_hashes,
        raw_rows_path=raw_path,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        duration_s=duration_s,
        phase_spans=[],
    )
    artifact.update(
        {
            "status": "complete",
            "honest_verdict": "complete_null_first_loss_not_reproduced_live_continuation_skipped",
            "verdict_class": "null",
            "arc_resume_ready_score": 0,
            "resume_control_rows": [],
            "rows": [],
            "acceptance_gate_results": [],
            "validation_receipts": [],
        }
    )
    artifact["field_principles"] = _field_principles(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, current-inference truth, readiness, and checksum."""

    errors: list[str] = []
    required = {
        "schema",
        "status",
        "experiment_id",
        "milestone",
        "run_date",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "honest_verdict",
        "verdict_class",
        "validation_receipts",
        "repository_health",
        "field_principles",
        "arc_resume_ready_score",
        "first_loss_receipt",
        "resume_control_rows",
        "solve_provenance",
    }
    missing = sorted(required - set(value))
    errors.extend(f"missing required field: {name}" for name in missing)
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema or experiment identity mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("milestone or run date mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current model declaration mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current invocation counts mismatch")
    if (
        value.get("inference_substrate") != "cpu_exact_solver_or_simulator"
        or value.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
    ):
        errors.append("inference substrate mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution venue mismatch")
    if value.get("solve_provenance") != "no_game_solve_cpu_transport_fixture":
        errors.append("solve provenance mismatch")
    ready = int(value.get("arc_resume_ready_score") or 0)
    if value.get("verdict_class") in {"blocked", "disqualified", "partial"} and ready:
        errors.append("unsafe readiness on non-ready artifact")
    controls = {str(row.get("arm")): row for row in value.get("resume_control_rows", [])}
    required_arms = {
        "tool_needed_changed_input",
        "result_withheld",
        "no_tool_needed",
        "stale_episode",
        "mismatched_attempt",
        "expired_authority",
        "duplicate_result",
        "timeout_after_dispatch",
        "absent_result",
    }
    reduced = int(
        value.get("first_loss_receipt", {}).get("reproduced") is True
        and set(controls) == required_arms
        and all(controls[name].get("passed") is True for name in required_arms)
        and bool(value.get("acceptance_gate_results"))
        and all(row.get("passed") is True for row in value.get("acceptance_gate_results", []))
    )
    if reduced != ready:
        errors.append("readiness reduction mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field principles mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("checksum mismatch")
    return errors


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _progress(started: float, phase: str, event: str, **detail: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in detail.items())
    print(
        f"[exp7336] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def _source_hashes(root: Path, panel: Mapping[str, Any]) -> JsonDict:
    paths = (
        SPEC_PATH,
        EXP7319_RESULT_PATH,
        MODULE_PATH,
        RESUME_MODULE_PATH,
        LOOP_MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        Path("ops/e2e-test-plan.md"),
        Path("ops/exclusion_manifest.yaml"),
    )
    hashes = {
        path.as_posix(): {
            "sha256": sha256_file(root / path),
            "role": "historical_diagnostic_source"
            if path == EXP7319_RESULT_PATH
            else "current_source_or_input",
            "authorizes_readiness": False,
        }
        for path in paths
        if (root / path).is_file()
    }
    sidecar = Path(str(panel.get("fixture_sidecar_path") or ""))
    if sidecar.is_file():
        hashes[str(sidecar)] = {
            "sha256": sha256_file(sidecar),
            "role": "scripted_model_event_fixture",
            "authorizes_readiness": False,
            "counts_as_current_model_invocation": False,
        }
    return hashes


def _run_e2e(root: Path, private: Path) -> list[JsonDict]:
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    commands = [
        validation_scope.CommandSpec(
            "e2e_009",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ),
            "E2E-009 real request construction",
        ),
        validation_scope.CommandSpec(
            "e2e_010",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ),
            "E2E-010 local grammar transport",
        ),
        validation_scope.CommandSpec(
            "e2e_offline_smoke",
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
                str(private / "offline-smoke.json"),
            ),
            "E2E-009 offline LLM-off environment smoke",
        ),
    ]
    return validation_scope.run_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "e2e-validation",
        extra_env={"CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=60.0,
    )


def _run_scoped_validation(root: Path, private: Path) -> list[JsonDict]:
    result = validation_scope.run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix(), RESUME_MODULE_PATH.as_posix()],
        static_paths=[LOOP_MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix()],
        basetemp=private / "scoped",
        coverage_file=private / ".coverage",
        log_dir=root / RAW_DIR / "validation",
    )
    return list(result["validation_receipts"])


def _terminal_lint_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal candidate",
        ),
    ]


def run_experiment(args: argparse.Namespace) -> JsonDict:
    """Run CPU fixtures, affected checks, independent reduction, and atomic publication."""

    started = time.monotonic()
    started_utc = _utc_now()
    _progress(started, "startup", "begin", run_date=args.date)
    first_loss = trace_exp7319_first_loss(REPO_ROOT)
    _progress(started, "first_loss", "after_trace", reproduced=first_loss.get("reproduced"))
    if not first_loss.get("reproduced"):
        artifact = build_null_artifact(
            first_loss=first_loss,
            source_hashes={},
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
        )
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        return artifact
    phase_start = time.monotonic()
    _progress(started, "cpu_panel", "before_scripted_policy_benchmark")
    panel = run_resume_control_panel(REPO_ROOT / RAW_DIR / "fixtures")
    rejection_rows = run_rejection_panel()
    _progress(started, "cpu_panel", "after_scripted_policy_benchmark", completed_units=9)
    phase_spans = [
        {
            "phase": "cpu_panel",
            "started_offset_s": round(phase_start - started, 6),
            "ended_offset_s": round(time.monotonic() - started, 6),
            "duration_s": round(time.monotonic() - phase_start, 6),
            "completed_units": 9,
            "checkpoint_position": "cpu_panel_complete",
            "pending_operations": [],
        }
    ]
    raw_rows_path = REPO_ROOT / RAW_DIR / "independent_reduction_input.json"
    atomic_write(raw_rows_path, {"rows": [*panel["rows"], *rejection_rows]})
    private = Path(tempfile.mkdtemp(prefix="exp7336-validation-"))
    _progress(started, "validation", "before_scoped_subprocesses")
    validation_receipts = _run_scoped_validation(REPO_ROOT, private)
    e2e_receipts = _run_e2e(REPO_ROOT, private)
    _progress(started, "validation", "after_scoped_subprocesses")
    source_hashes = _source_hashes(REPO_ROOT, panel)
    provisional_lints = [
        {"name": "adversarial_verify", "passed": True, "exit_code": 0},
        {"name": "verdict_row_consistency_strict", "passed": True, "exit_code": 0},
    ]
    candidate = build_terminal_artifact(
        first_loss=first_loss,
        control_panel=panel,
        rejection_rows=rejection_rows,
        validation_receipts=validation_receipts,
        e2e_receipts=e2e_receipts,
        terminal_lint_receipts=provisional_lints,
        source_hashes=source_hashes,
        raw_rows_path=raw_rows_path,
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
    )
    candidate_path = REPO_ROOT / RAW_DIR / "measured_terminal_candidate.json"
    atomic_write(candidate_path, candidate)
    _progress(started, "terminal", "before_candidate_linters")
    lint_receipts = validation_scope.run_commands(
        REPO_ROOT,
        _terminal_lint_specs(REPO_ROOT, candidate_path),
        log_dir=REPO_ROOT / RAW_DIR / "terminal-validation",
        heartbeat_s=60.0,
    )
    candidate = build_terminal_artifact(
        first_loss=first_loss,
        control_panel=panel,
        rejection_rows=rejection_rows,
        validation_receipts=validation_receipts,
        e2e_receipts=e2e_receipts,
        terminal_lint_receipts=lint_receipts,
        source_hashes=source_hashes,
        raw_rows_path=raw_rows_path,
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
    )
    errors = validate_artifact(candidate)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    atomic_write(candidate_path, candidate)
    atomic_write(REPO_ROOT / RESULT_PATH, candidate)
    _progress(
        started, "terminal", "after_atomic_publication", ready=candidate["arc_resume_ready_score"]
    )
    return candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    artifact = run_experiment(parse_args(argv))
    return 0 if artifact.get("status") in {"complete", "disqualified"} else 1
