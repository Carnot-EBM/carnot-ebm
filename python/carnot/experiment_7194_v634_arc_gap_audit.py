"""Audit cached ARC tool calls and banked progress without new inference.

The audit uses the Exp7193 receipts as immutable inputs. It runs the shipped
refinement CLIs against private ledgers, replays the shipped XML parser, and
keeps shadow supervisor credit separate from completed level transitions.

Spec refs: REQ-ARC-WMTE-7194 and SCENARIO-ARC-WMTE-7194-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, TextIO

import yaml

from carnot.agentic.arc_induction_tools import TOOL_NAMES, parse_xml_tool_calls
from carnot.agentic.arc_supervisor_refinement import empty_ledger as empty_supervisor_ledger
from carnot.agentic.arc_tool_gap_refinement import empty_ledger as empty_tool_gap_ledger


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260910"
RANDOM_SEED = 7_194_001
UPSTREAM_SEED = 7_193_001
MODEL_SPECS: list[JsonDict] = []

SCHEMA = "carnot.experiment_7194.arc_gap_audit.v1"
TASK_ID = "exp7194-arc-gap-audit"
MILESTONE = "2026.09.634"
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
UPSTREAM_PATH = Path("results/experiment_7193_v634_arc_direct_tool.json")
SESSION_PATH = Path("results/raw/experiment_7193/session_receipt.json")
RUN_ROW_PATH = Path("results/raw/experiment_7193/run_game_row.json")
COMPLETION_MANIFEST_PATH = Path("results/raw/experiment_7193/completion_manifest.json")
TOOL_GAP_MANIFEST_PATH = Path("results/raw/experiment_7193/tool_gap_receipts.json")
PRIOR_GAP_PATH = Path("results/experiment_6845_tool_gap_causal_support_audit.json")
PRIOR_CREDIT_PATH = Path("results/experiment_6921_arc_dynamic_supervisor_banked_credit.json")
OUTPUT_PATH = Path("results/experiment_7194_v634_arc_gap_audit.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7194_v634_arc_gap_audit")
MODULE_PATH = Path("python/carnot/experiment_7194_v634_arc_gap_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7194_v634_arc_gap_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7194_v634_arc_gap_audit.py")

TOOL_GAP_SCRIPT = Path("scripts/arc_tool_gap_refine.py")
SUPERVISOR_SCRIPT = Path("scripts/arc_supervisor_refine.py")
DURABLE_TOOL_GAP_LEDGER = Path("ops/arc_tool_gap_ledger.json")
DURABLE_SUPERVISOR_LEDGER = Path("ops/arc_supervisor_refinement_ledger.json")

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": OUTPUT_PATH.as_posix(),
    "gated_on": [
        {
            "upstream": "exp7193-arc-direct-tool",
            "artifact_field": "arc_tool_measurement_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    "prior_failures": [
        {
            "experiment_id": "exp6845-tool-gap-causal-support-audit",
            "verdict": "complete_blocked_tool_gap_causal_support_audit",
            "addressed_by": (
                "Use fresh directly engaged selfparse receipts from Exp7193; actual induction "
                "IDs replace absent tool-gap support."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "experiment_id": "exp6921-arc-dynamic-supervisor-banked-credit",
            "verdict": "complete_insufficient_banked_progress_evidence",
            "addressed_by": (
                "Count actual tool-loop calls and banked game progress; exclude the no-op "
                "supervisor arm and its shared helped credit."
            ),
            "retire_if_same_verdict": True,
        },
    ],
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ROADMAP_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    SPEC_PATH,
    TOOL_GAP_SCRIPT,
    SUPERVISOR_SCRIPT,
    Path("python/carnot/agentic/arc_induction_tools.py"),
    Path("python/carnot/agentic/arc_tool_gap_refinement.py"),
    Path("python/carnot/agentic/arc_supervisor_refinement.py"),
    UPSTREAM_PATH,
    SESSION_PATH,
    RUN_ROW_PATH,
    COMPLETION_MANIFEST_PATH,
    TOOL_GAP_MANIFEST_PATH,
    PRIOR_GAP_PATH,
    PRIOR_CREDIT_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260910, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": (
        "Record planned and completed counts, independent units and exclusions."
    ),
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": (
        "Every blocked verdict names the failed check, upstream, field, expected and "
        "observed value."
    ),
    "verifier_is_oracle": (
        "True when verification uses the same correctness authority; separate "
        "implementations alone do not remove circularity."
    ),
    "verdict_class": (
        "Use positive | circular_positive | null | blocked | disqualified | partial. "
        "Only incomplete own work can be partial."
    ),
    "honest_verdict": (
        "Use complete_ or complete: for completed findings, including nulls; blocked_* "
        "for external blocks. Never promote infrastructure readiness as scientific benefit."
    ),
    "arc_gap_audit_complete_score": "A complete audit may establish no missing tool.",
    "gap_rows": "Each candidate needs a real call, failure and source receipt.",
    "banked_credit_rows": ("Persistent progress must be separated from transient level readings."),
    "session_cost_rows": ("Recompute the single measured session; make no paired efficacy claim."),
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
    "schema": "A versioned schema lets later reducers reject incompatible evidence.",
    "experiment_id": "The task identifier binds this result to REQ-ARC-WMTE-7194.",
    "raw_manifest_receipts": (
        "Manifest receipts show which cached session bytes supplied each join."
    ),
    "refinement_tool_runs": (
        "Private-ledger command receipts prove both shipped refinement tools ran."
    ),
    "refinement_mutation_counts": (
        "Zero durable mutations keep recommendations separate from policy changes."
    ),
    "generalization_recommendation": (
        "A bounded recommendation is either counterexample-backed or an honest no-gap count."
    ),
    "historical_context": (
        "Historical calls remain context and never increase the new-session denominator."
    ),
    "causal_claims": (
        "Explicit false claim fields prevent one unpaired session from becoming efficacy evidence."
    ),
}

REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}


def canonical_json(value: Any) -> str:
    """Encode one stable JSON representation for evidence identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    """Label a SHA-256 digest so it cannot be confused with an untyped ID."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one file in bounded blocks so large receipts do not enter memory twice."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable artifact content while excluding its digest and measured duration."""

    body = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(body).encode())


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
    **extra: Any,
) -> JsonDict:
    """Keep every gate reason next to its exact evidence contract."""

    row = {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed) if passed is None else bool(passed),
    }
    row.update(deepcopy(extra))
    return row


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the first failed check and retain every check for diagnosis."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    if failed is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
            "checks": rows,
        }
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
        "checks": rows,
    }


def _is_quarantined(payload: Mapping[str, Any]) -> bool:
    """Use the conductor's artifact quarantine rule and fail closed on direct flags."""

    scripts = REPO_ROOT / "scripts"
    try:
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from conductor_gates import _is_quarantined as conductor_is_quarantined

        return bool(conductor_is_quarantined(dict(payload)))
    except (ImportError, AttributeError, TypeError):
        flag = payload.get("flagged_adversarial")
        if isinstance(flag, Mapping):
            flag = flag.get("value")
        return flag is True


def upstream_measurement_gate(payload: Mapping[str, Any], upstream: str) -> JsonDict:
    """Reject quarantine before reading the Exp7193 completion score."""

    quarantined = _is_quarantined(payload)
    if quarantined:
        observed = {"value": "not_consumed", "quarantined": True, "consumed": False}
    else:
        observed = {
            "value": payload.get("arc_tool_measurement_complete_score"),
            "quarantined": False,
            "consumed": True,
        }
    return gate_check(
        "upstream_measurement_complete",
        upstream,
        "arc_tool_measurement_complete_score",
        {"value": 1, "quarantined": False, "consumed": True},
        observed,
        not quarantined and observed["value"] == 1,
    )


def _resolve_content_path(root: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _induction_id(attempt: Mapping[str, Any], index: int, seed: int) -> str:
    identity = {
        "game": "r11l",
        "seed": seed,
        "attempt_index": index,
        "started_at": attempt.get("started_at"),
        "reason": attempt.get("reason"),
    }
    return sha256_bytes(canonical_json(identity).encode())


def _persistent_gap_events(
    tool_gap_manifest: Mapping[str, Any], induction_id: str
) -> list[JsonDict]:
    """Project only first-party dispatch failures joined to this induction."""

    events: list[JsonDict] = []
    raw_rows = tool_gap_manifest.get("rows", [])
    for raw in raw_rows if isinstance(raw_rows, list) else []:
        if not isinstance(raw, Mapping) or raw.get("attempt_identity") != induction_id:
            continue
        response = raw.get("hops", {}).get("tool_response", {}).get("payload", {})
        response = response if isinstance(response, Mapping) else {}
        result = response.get("response", {})
        result = result if isinstance(result, Mapping) else {}
        events.append(
            {
                "kind": raw.get("gap_kind"),
                "requested_tool": raw.get("requested_tool"),
                "error": result.get("error"),
                "runtime_counterexample": {
                    "decision_point_identity": raw.get("decision_point_identity"),
                    "result": deepcopy(dict(result)),
                },
                "source_receipt_identity": raw.get("receipt_identity"),
            }
        )
    return events


def recompute_gap_rows(
    *,
    root: Path,
    run_row: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
    tool_gap_manifest: Mapping[str, Any],
    upstream_tool_rows: Sequence[Mapping[str, Any]],
    seed: int,
    source_hashes: Mapping[str, Any],
) -> list[JsonDict]:
    """Replay strict parsing and retain one row for every policy induction.

    A completion can prove that a call was emitted and parsed. Only the live
    gap fields or first-party dispatch receipt can prove a missing capability.
    This keeps ordinary model prose outside the gap decision.
    """

    diagnostics = run_row.get("policy_diagnostics", {})
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    attempts = attempts if isinstance(attempts, list) else []
    upstream_by_index = {
        int(row["attempt_index"]): row
        for row in upstream_tool_rows
        if isinstance(row, Mapping) and isinstance(row.get("attempt_index"), int)
    }
    rows: list[JsonDict] = []
    for index, raw_attempt in enumerate(attempts):
        if not isinstance(raw_attempt, Mapping):
            continue
        attempt = dict(raw_attempt)
        induction_id = str(
            upstream_by_index.get(index, {}).get("induction_id")
            or _induction_id(attempt, index, seed)
        )
        matching = [
            dict(row)
            for row in completions
            if row.get("stage") == "environment" and row.get("induction_attempt_index") == index
        ]
        calls: Counter[str] = Counter()
        blocks = 0
        parser_failures = 0
        completion_errors: list[str] = []
        for completion in matching:
            path = _resolve_content_path(root, completion.get("content_path"))
            text = path.read_text(encoding="utf-8")
            expected_hash = completion.get("content_sha256")
            if expected_hash and sha256_bytes(text.encode()) != expected_hash:
                raise ValueError(f"completion_hash_mismatch:{path}")
            parsed, seen, unparsed = parse_xml_tool_calls(text)
            blocks += seen
            parser_failures += unparsed
            calls.update(
                str((call.get("function") or {}).get("name") or "")
                for call in parsed
                if isinstance(call, Mapping)
            )
            if completion.get("error"):
                completion_errors.append(str(completion["error"]))
        gap = attempt.get("tool_gap")
        capture_complete = isinstance(gap, Mapping)
        gap_mapping = gap if isinstance(gap, Mapping) else {}
        captured = [
            deepcopy(dict(event))
            for event in gap_mapping.get("tool_gap_events", []) or []
            if isinstance(event, Mapping)
        ]
        captured.extend(_persistent_gap_events(tool_gap_manifest, induction_id))
        upstream = upstream_by_index.get(index, {})
        rows.append(
            {
                "unit_id": f"r11l:{seed}:induction:{index}",
                "game": run_row.get("game"),
                "seed": seed,
                "attempt_index": index,
                "induction_id": induction_id,
                "reason": attempt.get("reason"),
                "started_at": attempt.get("started_at"),
                "elapsed_s": float(attempt.get("wall_s", 0.0) or 0.0),
                "completion_ids": [row.get("completion_id") for row in matching],
                "completion_count": len(matching),
                "selfparse_blocks_seen": blocks,
                "parsed_tool_calls": sum(calls.values()),
                "tool_calls_by_name": dict(sorted(calls.items())),
                "unknown_parsed_tool_names": sorted(set(calls) - set(TOOL_NAMES)),
                "parser_failures": parser_failures,
                "recorded_tool_calls_total": (
                    int(gap_mapping.get("tool_calls_total", 0) or 0) if capture_complete else None
                ),
                "gap_capture_state": (
                    "capture_complete" if capture_complete else "capture_not_recorded"
                ),
                "gap_events": captured,
                "gap_event_count": len(captured),
                "gap_events_dropped": int(gap_mapping.get("tool_gap_events_dropped", 0) or 0),
                "terminal_result_returned": bool(upstream.get("terminal_result_returned")),
                "engagement_completed": bool(upstream.get("engaged")),
                "policy_attempt_completed": isinstance(attempt.get("wall_s"), (int, float)),
                "source_receipt_hashes": deepcopy(dict(source_hashes)),
                "error": attempt.get("skipped") or ";".join(completion_errors) or None,
                "abstention": not matching,
            }
        )
    return rows


def generalization_recommendation(gap_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return one counterexample-backed candidate or an exact honest empty."""

    eligible: list[JsonDict] = []
    total_events = 0
    for row in gap_rows:
        call_names = set((row.get("tool_calls_by_name") or {}).keys())
        for raw in row.get("gap_events", []) or []:
            if not isinstance(raw, Mapping):
                continue
            total_events += 1
            name = str(raw.get("requested_tool") or raw.get("tool") or "")
            if (
                raw.get("kind") in {"unknown_tool", "bad_arguments"}
                and name in call_names
                and bool(str(raw.get("error") or "").strip())
                and isinstance(raw.get("runtime_counterexample"), Mapping)
            ):
                eligible.append(
                    {
                        "candidate_name": name,
                        "kind": raw.get("kind"),
                        "induction_id": row.get("induction_id"),
                        "failure": raw.get("error"),
                        "runtime_counterexample": deepcopy(raw["runtime_counterexample"]),
                    }
                )
    if eligible:
        first = eligible[0]
        return {
            "kind": "reusable_tool_candidate",
            "candidate_name": first["candidate_name"],
            "evidence": first,
            "eligible_candidate_count": len(eligible),
            "captured_gap_event_count": total_events,
            "automatic_tool_creation": False,
            "recommendation_only": True,
        }
    return {
        "kind": "honest_no_gap",
        "candidate_name": None,
        "new_induction_row_count": len(gap_rows),
        "capture_complete_induction_count": sum(
            row.get("gap_capture_state") == "capture_complete" for row in gap_rows
        ),
        "parsed_tool_call_count": sum(int(row.get("parsed_tool_calls", 0)) for row in gap_rows),
        "parser_failure_count": sum(int(row.get("parser_failures", 0)) for row in gap_rows),
        "captured_gap_event_count": total_events,
        "eligible_candidate_count": 0,
        "automatic_tool_creation": False,
        "recommendation_only": True,
    }


def recompute_banked_credit_rows(run_row: Mapping[str, Any], *, seed: int) -> list[JsonDict]:
    """Build banked transitions only from completed per-level evidence."""

    per_level = run_row.get("per_level", [])
    levels = (
        [row for row in per_level if isinstance(row, Mapping)]
        if isinstance(per_level, list)
        else []
    )
    charged_positions = run_row.get("level_up_charged", [])
    charged_positions = charged_positions if isinstance(charged_positions, list) else []
    attribution = run_row.get("level_reset_attribution", {})
    segments = attribution.get("segments", []) if isinstance(attribution, Mapping) else []
    segments = segments if isinstance(segments, list) else []
    rows: list[JsonDict] = []
    ordinal = 0
    for level_row in levels:
        if level_row.get("completed") is not True:
            continue
        level = int(level_row.get("level", ordinal))
        segment = (
            segments[ordinal]
            if ordinal < len(segments) and isinstance(segments[ordinal], Mapping)
            else {}
        )
        rows.append(
            {
                "row_kind": "banked_level_transition",
                "unit_id": f"r11l:{seed}:level:{level}:to:{level + 1}",
                "game": run_row.get("game"),
                "seed": seed,
                "from_level": level,
                "to_level": level + 1,
                "offline_actions": int(level_row.get("agent_actions", 0) or 0),
                "resets": int(segment.get("resets", 0) or 0),
                "charged_action_index": (
                    charged_positions[ordinal] if ordinal < len(charged_positions) else None
                ),
                "banked_progress": True,
                "promoted_banked_credit": 1,
                "evidence_contract": "per_level.completed+level_up_charged",
                "causal_tool_credit": False,
            }
        )
        ordinal += 1
    supervisor = run_row.get("trajectory_supervisor", {})
    would_have = (
        supervisor.get("would_have_redirects", []) if isinstance(supervisor, Mapping) else []
    )
    for index, raw in enumerate(would_have if isinstance(would_have, list) else []):
        if not isinstance(raw, Mapping):
            continue
        co_credited = raw.get("co_credited_count")
        rows.append(
            {
                "row_kind": "shadow_supervisor_context",
                "unit_id": f"r11l:{seed}:shadow_redirect:{index}",
                "game": run_row.get("game"),
                "seed": seed,
                "arm": raw.get("arm"),
                "action_index": raw.get("action_index"),
                "level": raw.get("level"),
                "supervisor_helped_flag": raw.get("levelup_followed_without_redirect") is True,
                "actions_to_levelup_without_redirect": raw.get(
                    "actions_to_levelup_without_redirect"
                ),
                "co_credited_count": co_credited,
                "shared_supervisor_credit": (
                    isinstance(co_credited, int)
                    and not isinstance(co_credited, bool)
                    and co_credited > 1
                ),
                "banked_progress": False,
                "promoted_banked_credit": 0,
                "evidence_contract": "shadow_counterfactual_noncausal_context",
                "causal_tool_credit": False,
            }
        )
    return rows


def recompute_session_cost_rows(
    run_row: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Charge every cached completion and measured induction without an efficacy ratio."""

    rows: list[JsonDict] = []
    for gap in gap_rows:
        index = gap.get("attempt_index")
        matching = [row for row in completions if row.get("induction_attempt_index") == index]
        rows.append(
            {
                "row_kind": "induction_cost",
                "attempt_index": index,
                "induction_id": gap.get("induction_id"),
                "elapsed_s": gap.get("elapsed_s", 0.0),
                "completion_count": len(matching),
                "completion_tokens": sum(
                    int(row.get("completion_tokens", 0) or 0) for row in matching
                ),
                "parsed_tool_calls": int(gap.get("parsed_tool_calls", 0) or 0),
                "parser_failures": int(gap.get("parser_failures", 0) or 0),
                "paired_efficacy_estimate": None,
            }
        )
    attribution = run_row.get("level_reset_attribution", {})
    resets = attribution.get("run_total_resets", 0) if isinstance(attribution, Mapping) else 0
    rows.append(
        {
            "row_kind": "session_total",
            "game": run_row.get("game"),
            "actions": int(run_row.get("actions", 0) or 0),
            "resets": int(resets or 0),
            "charged_actions": int(run_row.get("charged_actions", 0) or 0),
            "banked_levels": int(run_row.get("levels", 0) or 0),
            "completion_count": len(completions),
            "completion_tokens": sum(
                int(row.get("completion_tokens", 0) or 0) for row in completions
            ),
            "parsed_tool_calls": sum(int(row.get("parsed_tool_calls", 0) or 0) for row in gap_rows),
            "parser_failures": sum(int(row.get("parser_failures", 0) or 0) for row in gap_rows),
            "generator_wall_s": float(run_row.get("generator_wall_s", 0.0) or 0.0),
            "session_wall_s": float(run_row.get("wall_s", 0.0) or 0.0),
            "paired_efficacy_estimate": None,
        }
    )
    return rows


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace JSON only after the complete new document exists beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _pump_stream(stream: TextIO, sink: list[str], label: str) -> None:
    """Copy child output while retaining the same bytes for its receipt."""

    for line in stream:
        sink.append(line)
        print(f"[{label}] {line}", end="", flush=True)


def _run_streaming(command: Sequence[str], root: Path, timeout_s: float) -> JsonDict:
    """Run one bounded unbuffered child with a live external heartbeat."""

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(root / "python")
    print(canonical_json({"event": "external_heartbeat_start", "interval_s": 30}), flush=True)
    process = subprocess.Popen(
        list(command),
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    stdout: list[str] = []
    stderr: list[str] = []
    out_thread = threading.Thread(
        target=_pump_stream, args=(process.stdout, stdout, "child:stdout"), daemon=True
    )
    err_thread = threading.Thread(
        target=_pump_stream, args=(process.stderr, stderr, "child:stderr"), daemon=True
    )
    out_thread.start()
    err_thread.start()
    started = time.monotonic()
    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            returncode = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=5)
    out_thread.join(timeout=2)
    err_thread.join(timeout=2)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": time.monotonic() - started,
        "stdout": "".join(stdout),
        "stderr": "".join(stderr),
    }


def run_refinement_tools(
    *, root: Path, run_row: Mapping[str, Any], seed: int, scratch_dir: Path
) -> list[JsonDict]:
    """Run both shipped analyzers against one private new-session rows file."""

    scratch_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = scratch_dir / "new_session_rows.json"
    receipt_row = deepcopy(dict(run_row))
    receipt_row["seed"] = seed
    receipt_row["arm"] = "direct_selfparse"
    _atomic_write_json(receipt_path, {"rows": [receipt_row]})
    definitions = (
        (
            "tool_gap",
            TOOL_GAP_SCRIPT,
            scratch_dir / "tool_gap_ledger.json",
            empty_tool_gap_ledger(),
        ),
        (
            "supervisor",
            SUPERVISOR_SCRIPT,
            scratch_dir / "supervisor_ledger.json",
            empty_supervisor_ledger(),
        ),
    )
    runs: list[JsonDict] = []
    for name, script, ledger_path, empty in definitions:
        _atomic_write_json(ledger_path, empty)
        command = [
            sys.executable,
            "-u",
            str(root / script),
            str(receipt_path),
            "--ledger",
            str(ledger_path),
            "--json",
        ]
        print(canonical_json({"phase": 3, "event": "subprocess_start", "tool": name}), flush=True)
        result = _run_streaming(command, root, timeout_s=120)
        print(
            canonical_json(
                {
                    "phase": 3,
                    "event": "subprocess_end",
                    "tool": name,
                    "returncode": result["returncode"],
                }
            ),
            flush=True,
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        result.update(
            {
                "tool": name,
                "command": command,
                "input_path": str(receipt_path),
                "input_sha256": sha256_file(receipt_path),
                "ledger_path": str(ledger_path),
                "ledger_sha256": sha256_file(ledger_path),
                "ingest_counts": deepcopy(
                    dict(
                        (ledger.get("specification") or {}).get("ingest_counts", {})
                        if name == "tool_gap"
                        else (ledger.get("recommendation") or {}).get("ingest_counts", {})
                    )
                ),
                "result": deepcopy(
                    dict(
                        ledger.get("specification", {})
                        if name == "tool_gap"
                        else ledger.get("recommendation", {})
                    )
                ),
            }
        )
        runs.append(result)
    return runs


def _comparison_rows(
    gap_rows: Sequence[Mapping[str, Any]], banked_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for gap in gap_rows:
        rows.append(
            {
                "unit_id": gap.get("unit_id"),
                "arm": "direct_selfparse",
                "seed": gap.get("seed"),
                "metric": "parsed_tool_calls",
                "metric_value": int(gap.get("parsed_tool_calls", 0) or 0),
                "error": gap.get("error"),
                "abstention": bool(gap.get("abstention")),
            }
        )
        rows.append(
            {
                "unit_id": gap.get("unit_id"),
                "arm": "direct_selfparse",
                "seed": gap.get("seed"),
                "metric": "captured_gap_events",
                "metric_value": int(gap.get("gap_event_count", 0) or 0),
                "error": None,
                "abstention": gap.get("gap_capture_state") != "capture_complete",
            }
        )
    banked = [row for row in banked_rows if row.get("row_kind") == "banked_level_transition"]
    if banked:
        rows.append(
            {
                "unit_id": "r11l:7193001:session",
                "arm": "unpaired_observed_session",
                "seed": UPSTREAM_SEED,
                "metric": "banked_level_transitions",
                "metric_value": len(banked),
                "error": None,
                "abstention": False,
            }
        )
    return rows


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    gap_rows: Sequence[Mapping[str, Any]],
    banked_credit_rows: Sequence[Mapping[str, Any]],
    session_cost_rows: Sequence[Mapping[str, Any]],
    refinement_tool_runs: Sequence[Mapping[str, Any]],
    historical_context: Mapping[str, Any],
    blocked: bool,
    raw_manifest_receipts: Mapping[str, Any] | None = None,
    refinement_mutation_counts: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Construct a terminal artifact from already checked immutable evidence."""

    gaps = [deepcopy(dict(row)) for row in gap_rows]
    credit_rows = [deepcopy(dict(row)) for row in banked_credit_rows]
    costs = [deepcopy(dict(row)) for row in session_cost_rows]
    history = deepcopy(dict(historical_context))
    recommendation = generalization_recommendation(gaps)
    engagement = any(row.get("engagement_completed") is True for row in gaps)
    if blocked:
        status = "blocked"
        verdict_class = "blocked"
        failed = gate_summary(preconditions_checked).get("failed_check") or "external_prerequisite"
        honest_verdict = f"blocked_{failed}"
        substrate = "precondition_checks_only"
        substrate_class = "blocked_no_run"
        complete_score = 0
    else:
        status = "complete"
        verdict_class = "null" if recommendation["kind"] == "honest_no_gap" else "positive"
        if not engagement and gaps:
            honest_verdict = "complete_null_zero_engagement_upstream_audited"
        elif recommendation["kind"] == "honest_no_gap":
            honest_verdict = "complete_null_no_missing_tool_requested_banked_progress_noncausal"
        else:
            honest_verdict = "complete_reusable_tool_candidate_audited_no_causal_efficacy"
        substrate = "aggregation_from_upstream_artifacts"
        substrate_class = "aggregation"
        complete_score = 1
    comparison_rows = [] if blocked else _comparison_rows(gaps, credit_rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 7194,
        "field_principles": {},
        "status": status,
        "run_date": run_date,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "inference_substrate": substrate,
        "inference_substrate_class": substrate_class,
        "execution_venue": f"host:{os.uname().nodename}",
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "raw_manifest_receipts": deepcopy(dict(raw_manifest_receipts or {})),
        "rows": comparison_rows,
        "sample_size_budget": {
            "planned_sessions": 1,
            "completed_sessions": int(not blocked),
            "independent_units": int(not blocked),
            "new_policy_induction_attempts": len(gaps),
            "new_gap_capture_capable_inductions": sum(
                row.get("gap_capture_state") == "capture_complete" for row in gaps
            ),
            "new_terminal_tool_loop_inductions": sum(
                row.get("engagement_completed") is True for row in gaps
            ),
            "new_actual_parsed_tool_calls": sum(
                int(row.get("parsed_tool_calls", 0) or 0) for row in gaps
            ),
            "historical_tool_loop_inductions": int(
                history.get("historical_real_tool_loop_inductions", 0) or 0
            ),
            "historical_tool_calls": int(history.get("historical_tool_calls", 0) or 0),
            "historical_counts_toward_new_volume": False,
            "exclusions": [gate_summary(preconditions_checked).get("failed_check")]
            if blocked
            else [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions_checked),
        "verifier_is_oracle": True,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "arc_gap_audit_complete_score": complete_score,
        "gap_rows": gaps,
        "banked_credit_rows": credit_rows,
        "session_cost_rows": costs,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "refinement_tool_runs": [deepcopy(dict(row)) for row in refinement_tool_runs],
        "refinement_mutation_counts": deepcopy(
            dict(
                refinement_mutation_counts
                or {
                    "durable_tool_gap_ledger": 0,
                    "durable_supervisor_ledger": 0,
                    "curated_arm_table": 0,
                    "candidate_tool_registry": 0,
                }
            )
        ),
        "generalization_recommendation": recommendation,
        "historical_context": history,
        "causal_claims": {
            "paired_causal_efficacy_estimate_reported": False,
            "supervisor_help_promoted_to_banked_progress": False,
            "new_solve_claimed": False,
            "official_score_reported": False,
            "submitted": False,
            "registry_modified": False,
        },
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute all closed fields instead of trusting the artifact verdict."""

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_must_cover_every_top_level_field")
    elif any(principles.get(key) != FIELD_PRINCIPLES.get(key) for key in artifact):
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("MODEL_SPECS") != []:
        errors.append("MODEL_SPECS_must_be_empty")
    if artifact.get("model_invoked") is not False:
        errors.append("model_invoked_must_be_false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    checks = artifact.get("preconditions_checked", [])
    checks = checks if isinstance(checks, list) else []
    if artifact.get("gate_check_summary") != gate_summary(checks):
        errors.append("gate_check_summary_mismatch")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("status") != "blocked" or artifact.get("arc_gap_audit_complete_score") != 0:
            errors.append("blocked_terminal_inconsistent")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_inconsistent")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
    else:
        if (
            artifact.get("status") != "complete"
            or artifact.get("arc_gap_audit_complete_score") != 1
        ):
            errors.append("complete_terminal_inconsistent")
        if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
            errors.append("aggregation_substrate_invalid")
        if artifact.get("inference_substrate_class") != "aggregation":
            errors.append("aggregation_class_invalid")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("complete_verdict_prefix_missing")
    for row in artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []:
        if not isinstance(row, Mapping) or not {
            "unit_id",
            "arm",
            "seed",
            "metric",
            "error",
            "abstention",
        } <= set(row):
            errors.append("comparison_row_contract_invalid")
            break
    for row in artifact.get("banked_credit_rows", []):
        if (
            isinstance(row, Mapping)
            and row.get("row_kind") == "shadow_supervisor_context"
            and row.get("promoted_banked_credit") != 0
        ):
            errors.append("shadow_supervisor_credit_promoted")
            break
    claims = artifact.get("causal_claims", {})
    if not isinstance(claims, Mapping) or any(value is not False for value in claims.values()):
        errors.append("causal_or_solve_claim_forbidden")
    sample = artifact.get("sample_size_budget", {})
    if (
        not isinstance(sample, Mapping)
        or sample.get("historical_counts_toward_new_volume") is not False
    ):
        errors.append("historical_volume_must_remain_separate")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _load_json(path: Path) -> tuple[JsonDict, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}:{exc}"
    return (dict(value), None) if isinstance(value, Mapping) else ({}, "top_level_not_object")


def _task_contract(root: Path) -> tuple[JsonDict, str | None]:
    try:
        roadmap = yaml.safe_load((root / ROADMAP_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, f"{type(exc).__name__}:{exc}"
    tasks = roadmap.get("tasks") if isinstance(roadmap, Mapping) else roadmap
    if not isinstance(tasks, list):
        return {}, "roadmap_tasks_not_list"
    row = next(
        (item for item in tasks if isinstance(item, Mapping) and item.get("id") == TASK_ID),
        None,
    )
    if row is None:
        return {}, "task_not_found"
    keys = ("id", "milestone", "deliverable", "gated_on", "prior_failures")
    return ({key: deepcopy(row.get(key)) for key in keys}, None)


def _manifest_excludes(manifest: Any, experiment_id: int) -> bool:
    if isinstance(manifest, Mapping):
        if manifest.get("experiment_id") == experiment_id:
            return True
        return any(_manifest_excludes(value, experiment_id) for value in manifest.values())
    if isinstance(manifest, list):
        return any(_manifest_excludes(value, experiment_id) for value in manifest)
    return False


def _progress(phase: int, event: str, **fields: Any) -> None:
    """Emit one flushed progress record before and after each audit phase."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Check source bytes, quarantine, exact gates, and raw-manifest joins."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    payloads: JsonDict = {}
    _progress(0, "check_start", check="required_source_bytes")
    sizes: JsonDict = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        size = path.stat().st_size if path.is_file() else 0
        sizes[relative.as_posix()] = size
        hashes[relative.as_posix()] = sha256_file(path) if size else "missing"
    expected_sizes = {path.as_posix(): "nonempty" for path in REQUIRED_SOURCE_PATHS}
    checks.append(
        gate_check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            expected_sizes,
            sizes,
            all(size > 0 for size in sizes.values()),
        )
    )
    _progress(0, "check_end", check="required_source_bytes", passed=checks[-1]["passed"])

    _progress(0, "check_start", check="driving_capability_spec")
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    spec_ok = "REQ-ARC-WMTE-7194" in spec_text
    checks.append(
        gate_check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7194",
            True,
            spec_ok,
        )
    )
    _progress(0, "check_end", check="driving_capability_spec", passed=spec_ok)

    _progress(0, "check_start", check="exact_v634_task_contract")
    contract, contract_error = _task_contract(root)
    checks.append(
        gate_check(
            "exact_v634_task_contract",
            ROADMAP_PATH.as_posix(),
            "id,milestone,deliverable,gated_on,prior_failures",
            EXPECTED_TASK_CONTRACT,
            contract if contract_error is None else contract_error,
            contract_error is None and contract == EXPECTED_TASK_CONTRACT,
        )
    )
    _progress(0, "check_end", check="exact_v634_task_contract", passed=checks[-1]["passed"])

    _progress(0, "check_start", check="required_tools")
    tools = {
        "python": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
        "arc_tool_gap_refine": (root / TOOL_GAP_SCRIPT).is_file(),
        "arc_supervisor_refine": (root / SUPERVISOR_SCRIPT).is_file(),
    }
    checks.append(
        gate_check("required_tools", "host", "tools", {key: True for key in tools}, tools)
    )
    _progress(0, "check_end", check="required_tools", passed=checks[-1]["passed"])

    _progress(0, "check_start", check="output_directories")
    storage = {
        "result_parent_writable": os.access(root / OUTPUT_PATH.parent, os.W_OK),
        "checkpoint_parent_writable": os.access(root / CHECKPOINT_DIR.parent, os.W_OK),
    }
    checks.append(
        gate_check(
            "output_directories",
            "host_filesystem",
            "result,checkpoint",
            {key: True for key in storage},
            storage,
        )
    )
    _progress(0, "check_end", check="output_directories", passed=checks[-1]["passed"])

    _progress(0, "check_start", check="exclusion_manifest")
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
        exclusion_error = None
    except (OSError, yaml.YAMLError) as exc:
        exclusion, exclusion_error = {}, f"{type(exc).__name__}:{exc}"
    excluded = _manifest_excludes(exclusion, 7193) if exclusion_error is None else None
    checks.append(
        gate_check(
            "upstream_not_in_exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "experiment_id=7193",
            False,
            excluded if exclusion_error is None else exclusion_error,
            exclusion_error is None and excluded is False,
        )
    )
    _progress(0, "check_end", check="exclusion_manifest", passed=checks[-1]["passed"])

    _progress(0, "check_start", check="upstream_measurement_complete")
    upstream, upstream_error = _load_json(root / UPSTREAM_PATH)
    payloads["upstream"] = upstream
    if upstream_error is None:
        upstream_check = upstream_measurement_gate(upstream, UPSTREAM_PATH.as_posix())
    else:
        upstream_check = gate_check(
            "upstream_measurement_complete",
            UPSTREAM_PATH.as_posix(),
            "arc_tool_measurement_complete_score",
            {"value": 1, "quarantined": False, "consumed": True},
            upstream_error,
            False,
        )
    checks.append(upstream_check)
    _progress(
        0, "check_end", check="upstream_measurement_complete", passed=upstream_check["passed"]
    )
    if upstream_check["passed"] is not True:
        return checks, hashes, payloads

    for label, relative in (
        ("session", SESSION_PATH),
        ("run_row", RUN_ROW_PATH),
        ("completion_manifest", COMPLETION_MANIFEST_PATH),
        ("tool_gap_manifest", TOOL_GAP_MANIFEST_PATH),
        ("prior_gap", PRIOR_GAP_PATH),
        ("prior_credit", PRIOR_CREDIT_PATH),
    ):
        _progress(0, "check_start", check=f"load_{label}")
        payload, error = _load_json(root / relative)
        payloads[label] = payload
        quarantined = _is_quarantined(payload) if error is None else None
        checks.append(
            gate_check(
                f"{label}_readable_not_quarantined",
                relative.as_posix(),
                "json_object,flagged_adversarial",
                {"readable": True, "quarantined": False},
                {"readable": error is None, "quarantined": quarantined, "error": error},
                error is None and quarantined is False,
            )
        )
        _progress(0, "check_end", check=f"load_{label}", passed=checks[-1]["passed"])

    if any(row["passed"] is not True for row in checks):
        return checks, hashes, payloads

    session = payloads["session"]
    run_row = payloads["run_row"]
    completion_manifest = payloads["completion_manifest"]
    gap_manifest = payloads["tool_gap_manifest"]
    prior_gap = payloads["prior_gap"]
    prior_credit = payloads["prior_credit"]
    joins = (
        gate_check(
            "session_terminal_receipt",
            SESSION_PATH.as_posix(),
            "terminal_receipt,status",
            {"terminal_receipt": True, "status": "complete"},
            {"terminal_receipt": session.get("terminal_receipt"), "status": session.get("status")},
        ),
        gate_check(
            "run_row_hash_join",
            SESSION_PATH.as_posix(),
            "run_row_sha256",
            session.get("run_row_sha256"),
            hashes.get(RUN_ROW_PATH.as_posix()),
        ),
        gate_check(
            "completion_manifest_join",
            COMPLETION_MANIFEST_PATH.as_posix(),
            "completions",
            upstream.get("completion_receipts"),
            completion_manifest.get("completions"),
        ),
        gate_check(
            "tool_gap_manifest_schema",
            TOOL_GAP_MANIFEST_PATH.as_posix(),
            "schema,rows",
            {"schema": "carnot.arc.first_party_tool_gap_receipt.v1", "rows_list": True},
            {
                "schema": gap_manifest.get("schema"),
                "rows_list": isinstance(gap_manifest.get("rows"), list),
            },
        ),
        gate_check(
            "known_prior_gap_failure_not_promoted",
            PRIOR_GAP_PATH.as_posix(),
            "honest_verdict,evidence_role",
            {
                "honest_verdict": "complete_blocked_tool_gap_causal_support_audit",
                "evidence_role": "historical_not_promoted",
            },
            {
                "honest_verdict": prior_gap.get("honest_verdict"),
                "evidence_role": "historical_not_promoted",
            },
        ),
        gate_check(
            "known_prior_credit_failure_not_promoted",
            PRIOR_CREDIT_PATH.as_posix(),
            "honest_verdict,evidence_role",
            {
                "honest_verdict": "complete_insufficient_banked_progress_evidence",
                "evidence_role": "historical_not_promoted",
            },
            {
                "honest_verdict": prior_credit.get("honest_verdict"),
                "evidence_role": "historical_not_promoted",
            },
        ),
        gate_check(
            "run_row_shape",
            RUN_ROW_PATH.as_posix(),
            "policy_diagnostics,per_level,trajectory_supervisor",
            True,
            all(
                key in run_row
                for key in ("policy_diagnostics", "per_level", "trajectory_supervisor")
            ),
        ),
    )
    for check in joins:
        _progress(0, "check_start", check=check["check"])
        checks.append(check)
        _progress(0, "check_end", check=check["check"], passed=check["passed"])

    completion_rows = completion_manifest.get("completions", [])
    completion_rows = completion_rows if isinstance(completion_rows, list) else []
    _progress(0, "check_start", check="completion_content_hashes")
    mismatches: list[str] = []
    for row in completion_rows:
        if not isinstance(row, Mapping):
            mismatches.append("non_object_completion")
            continue
        path = _resolve_content_path(root, row.get("content_path"))
        try:
            digest = sha256_file(path)
        except OSError:
            digest = "missing"
        hashes[str(row.get("content_path"))] = digest
        if digest != row.get("content_sha256"):
            mismatches.append(str(row.get("completion_id")))
    checks.append(
        gate_check(
            "completion_content_hashes",
            COMPLETION_MANIFEST_PATH.as_posix(),
            "content_sha256",
            [],
            mismatches,
        )
    )
    _progress(0, "check_end", check="completion_content_hashes", passed=not mismatches)
    return checks, hashes, payloads


def _ledger_hash(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.is_file() else "missing"


def run_experiment(root: Path, *, run_date: str, output: Path) -> JsonDict:
    """Execute the complete read-only audit and atomically publish one result."""

    started = time.monotonic()
    _progress(0, "phase_start", name="preconditions")
    checks, hashes, payloads = collect_preconditions(root)
    _progress(0, "phase_end", name="preconditions", passed=gate_summary(checks)["passed"])
    if gate_summary(checks)["passed"] is not True:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            preconditions_checked=checks,
            source_artifact_hashes=hashes,
            gap_rows=[],
            banked_credit_rows=[],
            session_cost_rows=[],
            refinement_tool_runs=[],
            historical_context={"counts_toward_new_volume": False},
            blocked=True,
        )
        _progress(7, "atomic_write_start", output=str(output))
        _atomic_write_json(output, artifact)
        _progress(7, "atomic_write_end", output=str(output))
        return artifact

    _progress(1, "phase_start", name="load_raw_manifests")
    upstream = payloads["upstream"]
    run_row = payloads["run_row"]
    completion_rows = payloads["completion_manifest"]["completions"]
    gap_manifest = payloads["tool_gap_manifest"]
    raw_receipts = {
        "upstream": UPSTREAM_PATH.as_posix(),
        "session": SESSION_PATH.as_posix(),
        "run_row": RUN_ROW_PATH.as_posix(),
        "completion_manifest": COMPLETION_MANIFEST_PATH.as_posix(),
        "tool_gap_manifest": TOOL_GAP_MANIFEST_PATH.as_posix(),
    }
    _progress(1, "phase_end", name="load_raw_manifests", completion_count=len(completion_rows))

    _progress(2, "phase_start", name="declare_no_live_inference")
    _progress(2, "phase_end", model_invoked=False, MODEL_SPECS=[])

    durable_before = {
        "tool_gap": _ledger_hash(root, DURABLE_TOOL_GAP_LEDGER),
        "supervisor": _ledger_hash(root, DURABLE_SUPERVISOR_LEDGER),
    }
    _progress(3, "phase_start", name="isolated_refinement_tools")
    refinement_runs = run_refinement_tools(
        root=root,
        run_row=run_row,
        seed=UPSTREAM_SEED,
        scratch_dir=root / CHECKPOINT_DIR / "scratch",
    )
    _progress(3, "phase_end", name="isolated_refinement_tools", completed=len(refinement_runs))
    for row in refinement_runs:
        checks.append(
            gate_check(
                f"{row['tool']}_refinement_subprocess",
                str(row["ledger_path"]),
                "returncode,timed_out",
                {"returncode": 0, "timed_out": False},
                {"returncode": row["returncode"], "timed_out": row["timed_out"]},
            )
        )
    durable_after = {
        "tool_gap": _ledger_hash(root, DURABLE_TOOL_GAP_LEDGER),
        "supervisor": _ledger_hash(root, DURABLE_SUPERVISOR_LEDGER),
    }
    mutations = {
        "durable_tool_gap_ledger": int(durable_before["tool_gap"] != durable_after["tool_gap"]),
        "durable_supervisor_ledger": int(
            durable_before["supervisor"] != durable_after["supervisor"]
        ),
        "curated_arm_table": 0,
        "candidate_tool_registry": 0,
    }
    checks.append(
        gate_check(
            "durable_refinement_state_unchanged",
            "ops refinement ledgers",
            "before_hashes,after_hashes",
            durable_before,
            durable_after,
        )
    )

    _progress(4, "phase_start", name="recompute_joins_progress_cost")
    gap_rows = recompute_gap_rows(
        root=root,
        run_row=run_row,
        completions=completion_rows,
        tool_gap_manifest=gap_manifest,
        upstream_tool_rows=upstream.get("tool_induction_rows", []),
        seed=UPSTREAM_SEED,
        source_hashes={
            "run_row": hashes[RUN_ROW_PATH.as_posix()],
            "completion_manifest": hashes[COMPLETION_MANIFEST_PATH.as_posix()],
            "tool_gap_manifest": hashes[TOOL_GAP_MANIFEST_PATH.as_posix()],
        },
    )
    banked_rows = recompute_banked_credit_rows(run_row, seed=UPSTREAM_SEED)
    cost_rows = recompute_session_cost_rows(run_row, completion_rows, gap_rows)
    _progress(
        4,
        "phase_end",
        name="recompute_joins_progress_cost",
        induction_rows=len(gap_rows),
        banked_transitions=sum(row["row_kind"] == "banked_level_transition" for row in banked_rows),
    )

    _progress(5, "phase_start", name="bounded_generalization_recommendation")
    recommendation = generalization_recommendation(gap_rows)
    _progress(
        5, "phase_end", name="bounded_generalization_recommendation", result=recommendation["kind"]
    )

    history = deepcopy(dict(upstream.get("historical_receipt_summary", {}) or {}))
    history["counts_toward_new_volume"] = False
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        preconditions_checked=checks,
        source_artifact_hashes=hashes,
        raw_manifest_receipts=raw_receipts,
        gap_rows=gap_rows,
        banked_credit_rows=banked_rows,
        session_cost_rows=cost_rows,
        refinement_tool_runs=refinement_runs,
        refinement_mutation_counts=mutations,
        historical_context=history,
        blocked=False,
    )
    _progress(6, "validation_start", operation="validate_artifact")
    errors = validate_artifact(artifact)
    _progress(6, "validation_end", operation="validate_artifact", errors=errors)
    if errors:
        raise ValueError("artifact_validation_failed:" + ";".join(errors))
    _progress(7, "atomic_write_start", output=str(output))
    _atomic_write_json(output, artifact)
    _progress(7, "atomic_write_end", output=str(output))
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or (args.root / OUTPUT_PATH)
    artifact = run_experiment(args.root, run_date=args.date, output=output)
    print(
        canonical_json(
            {
                "output": str(output),
                "status": artifact["status"],
                "honest_verdict": artifact["honest_verdict"],
                "arc_gap_audit_complete_score": artifact["arc_gap_audit_complete_score"],
            }
        ),
        flush=True,
    )
    return 0 if not validate_artifact(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - the requested wrapper calls main.
    raise SystemExit(main())
