"""Audit V653 compact extraction rows and runtime history.

The producer is evidence, not an oracle. This module reads its persisted raw
rows and event sidecars, repeats parsing and accounting, and keeps an audited
null separate from successful audit completion.

Spec refs: REQ-REPORT-7456 and SCENARIO-REPORT-7456-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7437_v652_span_protocol import (
    constructed_qualifier_pairs,
    reduce_constructed_pairs,
)
from carnot.experiment_7443_v652_span_audit import (
    decode_raw_outcome,
    paired_effects,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7456-v653-extraction-audit"
SCHEMA = "carnot.exp7456.v653.extraction_audit.v1"
RANDOM_SEED = 6_537_456
ARMS = ("span", "verbatim")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7456_v653_extraction_audit.json")
RAW_DIR = Path("results/raw/experiment_7456_v653_extraction_audit")
MODULE_PATH = Path("python/carnot/experiment_7456_v653_extraction_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7456_v653_extraction_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7456_v653_extraction_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXP7451_PATH = Path("results/experiment_7451_v653_span_capture.json")
EXP7437_PATH = Path("results/experiment_7437_v652_span_protocol.json")
EXP7451_RAW = Path("results/raw/experiment_7451_v653_span_capture/owned_runtime")

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7443_v652_span_audit.py"),
    Path("python/carnot/experiment_7437_v652_span_protocol.py"),
    ROADMAP_PATH,
)

NONFACTUAL_INTRODUCTION_HASHES = (
    "sha256:22a50ca6be231159af6dea0d09863f908a749f4c9a3d7dad453a55632c198852",
    "sha256:cd2cae2ccf09751f07d8d069b205f8ff0ce5ad23ab225bc80486a5d21799170f",
    "sha256:28de621d5caa191b95c94fa35cb95f69b488a0053c91a63a7b5e72bd27aa8acf",
)

REQUIRED_MUTATIONS = (
    "deleted_modifier",
    "off_by_one_offset",
    "truncated_json",
    "absent_source_claim",
    "missing_runtime_terminal",
    "lease_identity",
)

TERMINAL_CHECKS = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "started_at_utc",
    "completed_at_utc",
    "preconditions_checked",
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
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "promotion_score",
    "extraction_audit_complete_score",
    "coverage_rows",
    "runtime_audit",
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned top-level schema with exact experiment identity and terminal status.",
    "run_date": "Use 20260920 and retain actual UTC and monotonic run boundaries.",
    "preconditions_checked": "Name exact source paths, identities, and observed values before reduction.",
    "MODEL_SPECS": "Use an empty list because this audit performs no current LLM work.",
    "model_invoked": "Keep current model attempts separate from authenticated producer history.",
    "invocation_counts": "Balance current loads and generations; every current count is zero.",
    "inference_substrate": "Name host JSON and numeric aggregation without implying generation.",
    "inference_substrate_class": "Use aggregation because this run only reduces persisted bytes.",
    "execution_venue": "Use host and keep producer CUDA identity in typed historical evidence.",
    "duration_s": "Measure actual audit work and separate computation and validation durations.",
    "phase_spans": "Bind phase timing, completed units, progress, and checkpoints.",
    "random_seed": "Freeze the paired bootstrap seed; fitting and resampling are otherwise absent.",
    "reproducibility_checksum": "Bind code, protocol, sources, raw reductions, and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes, verdict classes, and original flags.",
    "rows": "Keep one independent row for all eight development and 96 evaluation calls.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Separate validity, audit completion, and scientific benefit checks.",
    "gate_check_summary": "Name the exact first failure while retaining every failed check.",
    "verifier_is_oracle": "The deployed verifier does not supply semantic truth for real paragraphs.",
    "honest_verdict": "Report the completed audit while preserving the producer's scientific null.",
    "verdict_class": "Use the closed terminal enum without calling a valid null partial.",
    "flagged_adversarial": "Preserve critical current findings; none are hidden by source success.",
    "validation_receipts": "Record exact commands, environments, exits, durations, and log hashes.",
    "field_principles": "Explain field intent separately and keep gate values as bare scalars.",
    "promotion_score": "Keep zero because this milestone authorizes no rollout or publication.",
    "extraction_audit_complete_score": "One requires all available raw evidence or an explicit unavailable disposition.",
    "coverage_rows": "Show the denominator and annotation authority for every semantic metric.",
    "runtime_audit": "Bind cleanup and invocation claims to producer PID, start-tick, lease, and event evidence.",
}

V653_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def _load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    path: str,
    field: str,
    operator: str = "==",
    principle: str = "Authenticate the exact field before dependent work.",
) -> JsonDict:
    """Build one ordinary gate row with an exact failure location."""

    passed = observed == expected if operator == "==" else observed in expected
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def _reference(root: Path, relative: Path, receipt_class: str, **metadata: Any) -> JsonDict:
    """Bind one exact file while retaining its original evidence class."""

    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(root / relative),
        "source_receipt_class": receipt_class,
        **deepcopy(metadata),
    }


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, JsonDict], JsonDict]:
    """Authenticate branch-local inputs without using a producer value gate."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "readable_nonempty_bytes",
                observed,
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="bytes",
            )
        )
        if observed is not None:
            sources[relative.as_posix()] = _reference(root, relative, "current_audit_input")

    spec = _load_text(root / SPEC_PATH)
    checks.append(
        _gate(
            "driving_requirement",
            "precondition",
            "REQ-REPORT-7456",
            "REQ-REPORT-7456" if "REQ-REPORT-7456" in spec else None,
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    roadmap = _load_text(root / ROADMAP_PATH)
    declared = "exp7456-extraction-audit" in roadmap and RESULT_PATH.as_posix() in roadmap
    checks.append(
        _gate(
            "roadmap_declaration",
            "precondition",
            True,
            declared,
            upstream=ROADMAP_PATH.as_posix(),
            path=ROADMAP_PATH.as_posix(),
            field="exp7456-extraction-audit",
        )
    )

    producer_path = root / EXP7451_PATH
    available = producer_path.is_file() and producer_path.stat().st_size > 0
    checks.append(
        _gate(
            "exp7451_artifact_bytes",
            "precondition",
            "readable_nonempty_bytes",
            "readable_nonempty_bytes" if available else None,
            upstream="exp7451-v653-span-capture",
            path=EXP7451_PATH.as_posix(),
            field="bytes",
        )
    )
    producer = _load_object(producer_path) if available else {}
    if producer:
        sources[EXP7451_PATH.as_posix()] = _reference(
            root,
            EXP7451_PATH,
            "historical_model_producer_artifact",
            original_verdict_class=producer.get("verdict_class"),
            original_flagged_adversarial=producer.get("flagged_adversarial"),
            original_honest_verdict=producer.get("honest_verdict"),
            original_status=producer.get("status"),
        )
    for field, expected in (
        ("experiment_id", "exp7451-v653-span-capture"),
        ("milestone", MILESTONE),
    ):
        checks.append(
            _gate(
                f"exp7451_{field}",
                "precondition",
                expected,
                producer.get(field),
                upstream="exp7451-v653-span-capture",
                path=EXP7451_PATH.as_posix(),
                field=field,
            )
        )
    attempted = (producer.get("invocation_counts") or {}).get("generation_calls_attempted")
    checks.append(
        _gate(
            "exp7451_current_inference",
            "precondition",
            True,
            producer.get("model_invoked") is True and isinstance(attempted, int) and attempted > 0,
            upstream="exp7451-v653-span-capture",
            path=EXP7451_PATH.as_posix(),
            field="model_invoked/generation_calls_attempted",
            principle="Missing current inference blocks this branch and cannot reuse a prior reply.",
        )
    )
    checks.append(
        _gate(
            "exp7451_original_flag_type",
            "precondition",
            True,
            isinstance(producer.get("flagged_adversarial"), bool),
            upstream="exp7451-v653-span-capture",
            path=EXP7451_PATH.as_posix(),
            field="flagged_adversarial",
        )
    )
    protocol = root / EXP7437_PATH
    protocol_available = protocol.is_file() and protocol.stat().st_size > 0
    checks.append(
        _gate(
            "frozen_protocol_bytes",
            "precondition",
            "readable_nonempty_bytes",
            "readable_nonempty_bytes" if protocol_available else None,
            upstream="exp7437-v652-span-protocol",
            path=EXP7437_PATH.as_posix(),
            field="bytes",
        )
    )
    if protocol_available:
        protocol_value = _load_object(protocol)
        sources[EXP7437_PATH.as_posix()] = _reference(
            root,
            EXP7437_PATH,
            "frozen_protocol_artifact",
            original_verdict_class=protocol_value.get("verdict_class"),
            original_flagged_adversarial=protocol_value.get("flagged_adversarial"),
        )
    return checks, sources, producer


def _load_text(path: Path) -> str:
    """Read text when available so absent branch-local inputs stay explicit."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _byte_named_objects(directory: Path) -> tuple[list[JsonDict], list[JsonDict], list[str]]:
    """Load hash-named JSON sidecars and authenticate their exact file bytes."""

    values: list[JsonDict] = []
    references: list[JsonDict] = []
    errors: list[str] = []
    for path in sorted(directory.glob("sha256-*.json")):
        observed = sha256_file(path)
        expected = "sha256:" + path.stem.removeprefix("sha256-")
        if observed != expected:
            errors.append(f"sidecar_filename_hash_mismatch:{path.name}")
        value = _load_object(path)
        if not value:
            errors.append(f"sidecar_json_invalid:{path.name}")
            continue
        values.append(value)
        references.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": observed,
                "scope": "historical_model_receipts",
            }
        )
    return values, references, errors


def _row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a reduced row without its self-referential checksum field."""

    return canonical_hash({key: value for key, value in row.items() if key != "row_checksum"})


def _decode_row(row: Mapping[str, Any]) -> JsonDict:
    """Repeat strict parsing and add explicit semantic-authority boundaries."""

    reduced = decode_raw_outcome(row)
    errors = list(reduced.get("errors") or [])
    raw_reply = str(row.get("raw_reply") or "")
    if reduced.get("disposition") == "malformed" and raw_reply.lstrip().startswith("{"):
        errors.append("truncated_json_rejected")
    source_text = str(reduced.get("source_text") or "")
    source_hash = "sha256:" + hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if row.get("paragraph_sha256") != source_hash:
        errors.append("source_text_hash_mismatch")
    is_nonfactual_empty = (
        reduced.get("capture_phase") == "development"
        and source_hash in NONFACTUAL_INTRODUCTION_HASHES
        and reduced.get("disposition") == "empty"
    )
    reduced.update(
        {
            "errors": list(dict.fromkeys(errors)),
            "source_text_sha256": source_hash,
            "factual_recall_credit": False if is_nonfactual_empty else None,
            "semantic_ground_truth": "unavailable",
            "semantic_authority": "mechanical_literal_checks_only",
            "correct_empty_transport": is_nonfactual_empty,
            "parser_repair_attempted": False,
        }
    )
    reduced["row_checksum"] = _row_checksum(reduced)
    return reduced


def _constructed_audit(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[str]]:
    """Score only the frozen exact-string qualifier labels."""

    pairs = {str(row["pair_id"]): row for row in constructed_qualifier_pairs()}
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    errors: list[str] = []
    for row in rows:
        grouped[str(row.get("pair_id"))][str(row.get("arm"))] = row
    output: list[JsonDict] = []
    for pair_id, pair in pairs.items():
        arms = grouped.get(pair_id, {})
        if set(arms) != set(ARMS):
            errors.append(f"constructed_arm_count:{pair_id}")
            continue
        retained: dict[str, bool] = {}
        for arm in ARMS:
            row = arms[arm]
            claims = row.get("claims") if isinstance(row.get("claims"), list) else []
            selected = " ".join(str(value) for value in claims)
            retained[arm] = all(
                str(marker) in selected for marker in pair.get("required_modifiers") or []
            )
            if not retained[arm]:
                errors.append(f"constructed_modifier_missing:{pair_id}:{arm}")
            if row.get("literal_span_reconstruction") is not True:
                errors.append(f"constructed_literal_failure:{pair_id}:{arm}")
        output.append(
            {
                "pair_id": pair_id,
                "family": pair["family"],
                "authority": "constructed_exact_string",
                "span_qualifier_retained": retained["span"],
                "verbatim_qualifier_retained": retained["verbatim"],
                "paired_delta": int(retained["span"]) - int(retained["verbatim"]),
            }
        )
    if len(output) != 12:
        errors.append("constructed_pair_count_mismatch")
    return output, list(dict.fromkeys(errors))


def replay_runtime(producer: Mapping[str, Any]) -> JsonDict:
    """Replay producer calls and cleanup from event and process identities."""

    events = producer.get("current_invocation_events")
    event_rows = events if isinstance(events, list) else []
    run_id = producer.get("current_run_id")
    owner_pid = producer.get("current_owner_pid")
    errors: list[str] = []
    by_call: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    previous_time = -1
    for event in event_rows:
        call_id = str(event.get("call_id") or "missing")
        by_call[call_id].append(event)
        timestamp = event.get("monotonic_ns")
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp < previous_time
        ):
            errors.append(f"event_time_invalid:{call_id}")
        if isinstance(timestamp, int):
            previous_time = timestamp
        if (
            event.get("scope") != "current"
            or event.get("transport") != "owned_runtime"
            or event.get("run_id") != run_id
            or event.get("owner_pid") != owner_pid
        ):
            errors.append(f"event_identity_mismatch:{call_id}")
    prefixes = {"model_load": "model_loads", "generation": "generation_calls"}
    for call_id, transitions in by_call.items():
        operations = {str(row.get("operation")) for row in transitions}
        if len(operations) != 1 or next(iter(operations)) not in prefixes:
            errors.append(f"operation_invalid:{call_id}")
            continue
        operation = next(iter(operations))
        prefix = prefixes[operation]
        states = Counter(str(row.get("state")) for row in transitions)
        attempts = states["attempted"]
        terminals = states["completed"] + states["failed"] + states["cancelled"]
        counts[f"{prefix}_attempted"] += attempts
        for state in ("completed", "failed", "cancelled"):
            counts[f"{prefix}_{state}"] += states[state]
        counts[f"{prefix}_in_flight"] += int(attempts == 1 and terminals == 0)
        if attempts != 1:
            errors.append(f"attempt_count_invalid:{call_id}")
        if terminals == 0:
            errors.append(f"unfinished_call:{call_id}")
        elif terminals != 1:
            errors.append(f"terminal_count_invalid:{call_id}")

    if counts != producer.get("invocation_counts"):
        errors.append("producer_invocation_counts_mismatch")
    runner = (
        producer.get("runner_receipt")
        if isinstance(producer.get("runner_receipt"), Mapping)
        else {}
    )
    parent = (
        runner.get("parent_identity") if isinstance(runner.get("parent_identity"), Mapping) else {}
    )
    release = (
        runner.get("lease_release") if isinstance(runner.get("lease_release"), Mapping) else {}
    )
    cleanup = runner.get("cleanup") if isinstance(runner.get("cleanup"), Mapping) else {}
    pid_identity = (
        isinstance(runner.get("pid"), int)
        and isinstance(runner.get("start_time_ticks"), int)
        and parent.get("pid") == owner_pid
        and isinstance(parent.get("start_time_ticks"), int)
    )
    if not pid_identity:
        errors.append("pid_start_tick_identity_invalid")
    lease_identity = (
        runner.get("lease_id") == release.get("lease_id")
        and release.get("pid") == owner_pid
        and release.get("pid_start_ticks") == parent.get("start_time_ticks")
    )
    if not lease_identity:
        errors.append("lease_identity_mismatch")
    lease_release_verified = bool(
        lease_identity
        and runner.get("lease_released") is True
        and release.get("released") is True
        and release.get("signals_sent") == []
    )
    if not lease_release_verified:
        errors.append("lease_release_invalid")
    cleanup_verified = bool(
        cleanup.get("bounded") is True
        and cleanup.get("leak_free") is True
        and cleanup.get("unrelated_process_kill_count_delta") == 0
        and cleanup.get("action") in {"terminated", "already_exited"}
    )
    if not cleanup_verified:
        errors.append("cleanup_invalid")
    return {
        "errors": list(dict.fromkeys(errors)),
        "event_count": len(event_rows),
        "event_sha256": canonical_hash(event_rows),
        "historical_producer_invocation_counts": {
            "scope": "historical",
            "counts": counts,
        },
        "loads_balanced": counts["model_loads_in_flight"] == 0,
        "generations_balanced": counts["generation_calls_in_flight"] == 0,
        "producer_run_id": run_id,
        "producer_owner_pid": owner_pid,
        "server_pid": runner.get("pid"),
        "server_start_time_ticks": runner.get("start_time_ticks"),
        "lease_id": runner.get("lease_id"),
        "lease_release_verified": lease_release_verified,
        "cleanup_verified": cleanup_verified,
        "foreign_signal_count": cleanup.get("unrelated_process_kill_count_delta"),
    }


def audit_producer(repo_root: Path, producer: Mapping[str, Any]) -> JsonDict:
    """Reduce every raw producer row before consulting its aggregate scores."""

    root = repo_root.resolve()
    raw = root / EXP7451_RAW
    errors: list[str] = []
    sidecars: list[JsonDict] = []
    native_rows = [_load_object(path) for path in sorted((raw / "native").glob("call_*.json"))]
    native_rows = [row for row in native_rows if row]
    evaluation_rows, evaluation_refs, evaluation_errors = _byte_named_objects(raw / "evaluation")
    event_rows, event_refs, event_errors = _byte_named_objects(raw / "events")
    response_rows, response_refs, response_errors = _byte_named_objects(raw / "responses")
    sidecars.extend(evaluation_refs + event_refs + response_refs)
    errors.extend(evaluation_errors + event_errors + response_errors)
    for path in sorted((raw / "native").glob("call_*.json")):
        sidecars.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "scope": "historical_model_receipts",
            }
        )

    producer_development = producer.get("development_rows") or []
    producer_evaluation = producer.get("rows") or []
    if {canonical_hash(row) for row in native_rows} != {
        canonical_hash(row) for row in producer_development if isinstance(row, Mapping)
    }:
        errors.append("development_sidecars_artifact_mismatch")
    if {canonical_hash(row) for row in evaluation_rows} != {
        canonical_hash(row) for row in producer_evaluation if isinstance(row, Mapping)
    }:
        errors.append("evaluation_sidecars_artifact_mismatch")
    if sorted(event_rows, key=lambda row: int(row.get("monotonic_ns", 0))) != list(
        producer.get("current_invocation_events") or []
    ):
        errors.append("event_sidecars_artifact_mismatch")
    response_projection = {
        canonical_hash(
            {
                key: row.get(key)
                for key in (
                    "attempted",
                    "call_id",
                    "completion_tokens",
                    "error",
                    "finish_reason",
                    "latency_s",
                    "prompt_tokens",
                    "raw_reply",
                    "raw_request",
                    "raw_response",
                    "runtime_identity_receipt",
                    "terminal_state",
                )
            }
        )
        for row in native_rows
    }
    if {canonical_hash(row) for row in response_rows} != response_projection:
        errors.append("response_sidecars_artifact_mismatch")

    raw_rows = [*native_rows, *evaluation_rows]
    decoded = [_decode_row(row) for row in raw_rows]
    for row in decoded:
        errors.extend(f"{row.get('call_id')}:{error}" for error in row.get("errors") or [])
    manifest = producer.get("raw_capture_manifest") or []
    manifest_by_call = {
        str(row.get("call_id")): row for row in manifest if isinstance(row, Mapping)
    }
    if len(manifest_by_call) != 96:
        errors.append("raw_capture_manifest_count_mismatch")
    for raw_row in evaluation_rows:
        manifest_row = manifest_by_call.get(str(raw_row.get("call_id")))
        if manifest_row is None:
            errors.append(f"raw_capture_manifest_missing:{raw_row.get('call_id')}")
            continue
        for field in ("raw_request_sha256", "raw_response_sha256", "raw_reply_sha256"):
            if manifest_row.get(field) != raw_row.get(field):
                errors.append(
                    f"raw_capture_manifest_hash_mismatch:{raw_row.get('call_id')}:{field}"
                )

    effects = paired_effects(
        [row for row in decoded if row.get("capture_phase") == "evaluation"],
        seed=RANDOM_SEED,
        draws=10_000,
    )
    constructed_raw = reduce_constructed_pairs(constructed_qualifier_pairs())
    constructed_rows, constructed_errors = _constructed_audit(constructed_raw)
    errors.extend(constructed_errors)
    if canonical_hash(constructed_rows) != canonical_hash(producer.get("semantic_pair_rows") or []):
        errors.append("constructed_pair_summary_mismatch")
    runtime = replay_runtime(producer)
    errors.extend(runtime["errors"])
    counts = Counter(str(row.get("disposition")) for row in decoded)
    completed = counts["complete"] + counts["empty"]
    budget = {
        "planned": 104,
        "attempted": sum(row.get("attempted") is True for row in decoded),
        "completed": completed,
        "failed": counts["failed"]
        + counts["cancelled"]
        + counts["malformed"]
        + counts["truncated"],
        "censored": sum(row.get("censored") is True for row in raw_rows),
        "unstarted": counts["unstarted"],
        "development_planned": 8,
        "evaluation_planned": 96,
        "independent_paired_evaluation_units": 48,
        "stop_rule": "audit all eight development calls and all 96 fixed evaluation cells without retry",
    }
    coverage = [
        {
            "metric": "real_paragraph_factual_recall",
            "denominator": 0,
            "observed": None,
            "authority": "unavailable_no_factual_recall_annotations",
        },
        {
            "metric": "nonfactual_correct_empty_transport",
            "denominator": 6,
            "observed": sum(row.get("correct_empty_transport") is True for row in decoded),
            "authority": "frozen_nonfactual_introduction_identity",
        },
        {
            "metric": "constructed_qualifier_retention",
            "denominator": 12,
            "observed": sum(
                row["span_qualifier_retained"] and row["verbatim_qualifier_retained"]
                for row in constructed_rows
            ),
            "authority": "constructed_exact_string",
        },
        {
            "metric": "paired_evaluation_completion",
            "denominator": 48,
            "observed": effects["paired_completion_effect"]["pairs"],
            "authority": "authenticated_terminal_pairs_only",
        },
    ]
    return {
        "errors": list(dict.fromkeys(errors)),
        "rows": decoded,
        "disposition_counts": dict(sorted(counts.items())),
        "sample_size_budget": budget,
        "paired_completion_effect": effects["paired_completion_effect"],
        "paired_token_cost_effect": effects["paired_token_cost_effect"],
        "audited_cohort": effects["cohort"],
        "constructed_pair_rows": constructed_rows,
        "coverage_rows": coverage,
        "synthetic_to_natural_extrapolation": False,
        "runtime_audit": runtime,
        "historical_evidence_sidecars": sidecars,
    }


def _runtime_fixture() -> JsonDict:
    """Build the smallest owned runtime receipt used by mutation controls."""

    events = [
        {
            "call_id": "generation-0",
            "operation": "generation",
            "state": state,
            "scope": "current",
            "transport": "owned_runtime",
            "run_id": "fixture-run",
            "owner_pid": 11,
            "monotonic_ns": index + 1,
        }
        for index, state in enumerate(("attempted", "completed"))
    ]
    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    counts.update({"generation_calls_attempted": 1, "generation_calls_completed": 1})
    return {
        "current_invocation_events": events,
        "current_run_id": "fixture-run",
        "current_owner_pid": 11,
        "invocation_counts": counts,
        "runner_receipt": {
            "pid": 12,
            "start_time_ticks": 120,
            "parent_identity": {"pid": 11, "start_time_ticks": 110},
            "lease_id": "lease:fixture",
            "lease_released": True,
            "lease_release": {
                "lease_id": "lease:fixture",
                "pid": 11,
                "pid_start_ticks": 110,
                "released": True,
                "signals_sent": [],
            },
            "cleanup": {
                "action": "terminated",
                "bounded": True,
                "leak_free": True,
                "unrelated_process_kill_count_delta": 0,
            },
        },
    }


def _transport_row(arm: str, paragraph: str, reply: str) -> JsonDict:
    """Build one raw transport row for parser mutation controls."""

    request = {"max_tokens": 256}
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        "timings": {"predicted_n": 3},
    }
    return {
        "arm": arm,
        "attempted": True,
        "call_id": f"fixture-{arm}",
        "capture_phase": "development",
        "condition": "fixture",
        "finish_reason": "stop",
        "max_new_tokens": 256,
        "paragraph": paragraph,
        "raw_reply": reply,
        "raw_request": request,
        "raw_request_sha256": canonical_hash(request),
        "raw_response": response,
        "raw_response_sha256": canonical_hash(response),
        "raw_reply_sha256": canonical_hash(reply),
        "terminal_state": "response",
        "unit_id": f"fixture-{arm}",
    }


def run_mutation_controls() -> list[JsonDict]:
    """Plant every required corruption and require a named rejection."""

    controls = reduce_constructed_pairs(constructed_qualifier_pairs())
    changed = deepcopy(controls)
    changed[0]["claims"] = ["The trial met its endpoint."]
    _rows, modifier_errors = _constructed_audit(changed)

    paragraph = "The dose increased by 5 mg."
    offset_reply = json.dumps({"claims": [[0, len(paragraph) + 1]]})
    offset_errors = _decode_row(_transport_row("span", paragraph, offset_reply))["errors"]
    truncated_errors = _decode_row(_transport_row("span", paragraph, '{"claims":[[0,5]]'))["errors"]
    absent_errors = _decode_row(
        _transport_row("verbatim", paragraph, '{"claims":["The dose fell."]}')
    )["errors"]

    missing = _runtime_fixture()
    missing["current_invocation_events"].pop()
    missing_errors = replay_runtime(missing)["errors"]
    lease = _runtime_fixture()
    lease["runner_receipt"]["lease_release"]["lease_id"] = "lease:changed"
    lease_errors = replay_runtime(lease)["errors"]
    values = (
        ("deleted_modifier", modifier_errors, "constructed_modifier_missing"),
        ("off_by_one_offset", offset_errors, "span_bounds"),
        ("truncated_json", truncated_errors, "truncated_json_rejected"),
        ("absent_source_claim", absent_errors, "verbatim_not_literal"),
        ("missing_runtime_terminal", missing_errors, "unfinished_call"),
        ("lease_identity", lease_errors, "lease_identity_mismatch"),
    )
    return [
        {
            "attack": attack,
            "expected_error": expected,
            "observed_errors": errors,
            "passed": any(expected in error for error in errors),
        }
        for attack, errors, expected in values
    ]


def _validation_passed(receipts: Sequence[Mapping[str, Any]], *, candidate: bool) -> bool:
    """Require the exact affected set and terminal readers at final publication."""

    required = set(validation_scope.REQUIRED_CHECK_NAMES)
    if not candidate:
        required.update(TERMINAL_CHECKS)
    by_name = Counter(
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
    )
    return all(by_name[name] == 1 for name in required)


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first exact failure and retain all failed check names."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {
            "all_passed": True,
            "failed_count": 0,
            "failed_checks": [],
            "check": None,
            "upstream": None,
            "path": None,
            "field": None,
            "operator": None,
            "expected": None,
            "observed": None,
            "passed": True,
        }
    first = failed[0]
    return {
        "all_passed": False,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        **{
            key: deepcopy(first.get(key))
            for key in (
                "check",
                "upstream",
                "path",
                "field",
                "operator",
                "expected",
                "observed",
                "passed",
            )
        },
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable code, protocol, raw reduction, and exact validation scope."""

    receipts = [
        {
            key: row.get(key)
            for key in ("name", "command_argv", "command_environment", "exit_code", "log_sha256")
        }
        for row in value.get("validation_receipts") or []
        if isinstance(row, Mapping)
    ]
    bound = {
        key: value.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "source_artifact_hashes",
            "rows",
            "sample_size_budget",
            "coverage_rows",
            "runtime_audit",
            "constructed_pair_rows",
            "acceptance_gate_results",
            "audited_scientific_disposition",
            "validation_manifest",
            "extraction_audit_complete_score",
            "verdict_class",
        )
    }
    bound["validation_receipts"] = receipts
    return canonical_hash(bound)


def _build_artifact(
    evidence: Mapping[str, Any],
    *,
    checks: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    producer: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    started_at_utc: str,
    completed_at_utc: str,
    candidate: bool,
    fixture: bool = False,
) -> JsonDict:
    """Build one ordinary-field terminal audit from independent evidence."""

    audit_errors = list(evidence.get("errors") or [])
    runtime = deepcopy(dict(evidence.get("runtime_audit") or {}))
    rows = deepcopy(list(evidence.get("rows") or []))
    sample = deepcopy(dict(evidence.get("sample_size_budget") or {}))
    validation_ok = _validation_passed(receipts, candidate=candidate)
    audit_complete = int(
        not audit_errors
        and len(rows) == 104
        and sample.get("planned") == 104
        and not runtime.get("errors")
        and validation_ok
    )
    gates = [deepcopy(dict(row)) for row in checks]
    gates.extend(
        [
            _gate(
                "raw_evidence_reduced",
                "validity",
                [],
                audit_errors,
                upstream="exp7451-v653-span-capture",
                path=EXP7451_RAW.as_posix(),
                field="audit_errors",
            ),
            _gate(
                "all_planned_cells_accounted",
                "completion",
                104,
                len(rows),
                upstream="exp7451-v653-span-capture",
                path=EXP7451_PATH.as_posix(),
                field="development_plus_evaluation_rows",
            ),
            _gate(
                "runtime_replay_clean",
                "validity",
                [],
                runtime.get("errors") or [],
                upstream="exp7451-v653-span-capture",
                path=EXP7451_RAW.as_posix(),
                field="current_invocation_events/runner_receipt",
            ),
            _gate(
                "required_validation",
                "completion",
                True,
                validation_ok,
                upstream="validation_receipts",
                path="validation_receipts",
                field="required_commands",
            ),
            _gate(
                "evaluation_effect_established",
                "benefit",
                True,
                (evidence.get("paired_completion_effect") or {}).get("estimate") is not None,
                upstream="exp7451-v653-span-capture",
                path=EXP7451_PATH.as_posix(),
                field="paired_evaluation_effect",
                principle="A closed evaluation panel is a valid null, not an extraction effect.",
            ),
        ]
    )
    source_null = producer.get("verdict_class") == "null"
    verdict_class = "null" if audit_complete and source_null else "disqualified"
    honest = (
        "complete_null_extraction_audit_preserved_development_gate_closed"
        if verdict_class == "null"
        else "complete_disqualified_extraction_audit_evidence_invalid"
    )
    phase_spans = current_receipt.get("phase_spans") or []
    computation_duration = sum(
        float(row.get("duration_s") or 0.0)
        for row in phase_spans
        if isinstance(row, Mapping) and row.get("phase") == "source_audit"
    )
    cold_start_duration = sum(
        float(row.get("duration_s") or 0.0)
        for row in phase_spans
        if isinstance(row, Mapping) and row.get("phase") == "preconditions"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_extraction_audit",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        **deepcopy(dict(current_receipt)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host_cpu_json_and_numeric_aggregation",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "model_duration_s": 0.0,
        "computation_duration_s": computation_duration,
        "cold_start_duration_s": cold_start_duration,
        "validation_duration_s": sum(float(row.get("duration_s") or 0.0) for row in receipts),
        "random_seed": {
            "paired_bootstrap": RANDOM_SEED,
            "fit": None,
            "projection": None,
            "stream": None,
            "resampling": "paired_bootstrap_only",
        },
        "source_artifact_hashes": deepcopy(dict(sources)),
        "historical_evidence_sidecars": deepcopy(
            list(evidence.get("historical_evidence_sidecars") or [])
        ),
        "rows": rows,
        "independent_endpoint_rows": deepcopy(rows),
        "raw_disposition_counts": deepcopy(dict(evidence.get("disposition_counts") or {})),
        "sample_size_budget": sample,
        "paired_completion_effect": deepcopy(evidence.get("paired_completion_effect")),
        "paired_token_cost_effect": deepcopy(evidence.get("paired_token_cost_effect")),
        "audited_cohort": evidence.get("audited_cohort"),
        "constructed_pair_rows": deepcopy(list(evidence.get("constructed_pair_rows") or [])),
        "coverage_rows": deepcopy(list(evidence.get("coverage_rows") or [])),
        "synthetic_to_natural_extrapolation": False,
        "runtime_audit": runtime,
        "audit_mutation_rows": run_mutation_controls(),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": verdict_class == "disqualified",
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "validation_manifest": {
            "test_paths": list(V653_MANIFEST.test_paths),
            "changed_modules": list(V653_MANIFEST.changed_modules),
            "static_paths": list(V653_MANIFEST.static_paths),
            "affected_checks": list(validation_scope.REQUIRED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECKS),
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "extraction_audit_complete_score": audit_complete,
        "audited_scientific_disposition": {
            "source_experiment_id": producer.get("experiment_id"),
            "verdict_class": producer.get("verdict_class"),
            "honest_verdict": producer.get("honest_verdict"),
            "flagged_adversarial": producer.get("flagged_adversarial"),
            "span_value_score": producer.get("span_value_score"),
            "preserved_without_laundering": True,
        },
        "candidate_artifact": candidate,
        "fixture_artifact": fixture,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _passing_receipts(candidate: bool = True) -> list[JsonDict]:
    """Create deterministic fixture receipts for artifact unit tests."""

    names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if not candidate:
        names.extend(TERMINAL_CHECKS)
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "command_argv": [name],
            "command_environment": {},
            "log_sha256": canonical_hash(name),
        }
        for name in names
    ]


def _current_receipt(
    duration_ns: int = 1_000_000_000,
    *,
    spans: Sequence[Mapping[str, Any]] = (),
    sidecars: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build zero-current-inference provenance for this host aggregation."""

    return build_current_work_receipt(
        run_id=f"exp7456-audit-{duration_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="host_cpu_json_and_numeric_aggregation",
        inference_substrate_details={
            "cpu": platform.processor() or "host_cpu",
            "cuda": None,
            "external_device": None,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=duration_ns,
        sidecar_references=sidecars,
        phase_spans=spans,
        small_ebm_training={"performed": False, "receipt_class": "small_ebm_training"},
    )


def build_artifact_for_test() -> JsonDict:
    """Build a complete fixture from the real immutable producer sidecars."""

    producer = _load_object(REPO_ROOT / EXP7451_PATH)
    evidence = audit_producer(REPO_ROOT, producer)
    return _build_artifact(
        evidence,
        checks=[],
        sources={},
        producer=producer,
        receipts=_passing_receipts(),
        current_receipt=_current_receipt(),
        started_at_utc="2026-09-20T00:00:00+00:00",
        completed_at_utc="2026-09-20T00:00:01+00:00",
        candidate=True,
        fixture=True,
    )


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Publish missing current inference without reusing historical replies."""

    producer_gate = next(
        (
            row
            for row in checks
            if row.get("check") in {"exp7451_artifact_bytes", "exp7451_current_inference"}
            and row.get("passed") is not True
        ),
        {
            "check": "exp7451_current_inference",
            "upstream": "exp7451-v653-span-capture",
            "path": EXP7451_PATH.as_posix(),
            "field": "model_invoked/generation_calls_attempted",
            "operator": "==",
            "expected": True,
            "observed": None,
            "passed": False,
        },
    )
    receipt = _current_receipt(0)
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_missing_current_inference",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        **receipt,
        "source_artifact_hashes": deepcopy(dict(sources)),
        "rows": [],
        "sample_size_budget": {
            "planned": 104,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 104,
        },
        "acceptance_gate_results": [deepcopy(dict(row)) for row in checks],
        "gate_check_summary": deepcopy(dict(producer_gate)),
        "verifier_is_oracle": False,
        "honest_verdict": "blocked_missing_exp7451_current_inference",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "extraction_audit_complete_score": 1,
        "coverage_rows": [],
        "runtime_audit": {"availability": "unavailable", "errors": []},
        "random_seed": {"paired_bootstrap": RANDOM_SEED},
        "reproducibility_checksum": "",
    }


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    """Cold-check identity, rows, semantic limits, runtime, and validation."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("artifact_schedule_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_declaration_invalid")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_invalid")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if value.get("promotion_score") != 0:
        errors.append("promotion_score_invalid")
    rows = value.get("rows") or []
    independent = value.get("independent_endpoint_rows") or []
    if canonical_hash(rows) != canonical_hash(independent) or any(
        not isinstance(row, Mapping) or row.get("row_checksum") != _row_checksum(row)
        for row in rows
    ):
        errors.append("row_reduction_mismatch")
    counts = Counter(str(row.get("disposition")) for row in rows if isinstance(row, Mapping))
    if value.get("raw_disposition_counts") != dict(sorted(counts.items())):
        errors.append("disposition_counts_mismatch")
    sample = value.get("sample_size_budget") or {}
    if sample.get("planned") != len(rows) or sample.get("unstarted") != counts["unstarted"]:
        errors.append("sample_size_budget_mismatch")
    coverage = {
        row.get("metric"): row
        for row in value.get("coverage_rows") or []
        if isinstance(row, Mapping)
    }
    if (
        (coverage.get("real_paragraph_factual_recall") or {}).get("denominator") != 0
        or (coverage.get("constructed_qualifier_retention") or {}).get("authority")
        != "constructed_exact_string"
        or value.get("synthetic_to_natural_extrapolation") is not False
    ):
        errors.append("semantic_scope_invalid")
    runtime = value.get("runtime_audit") or {}
    if runtime.get("errors") or runtime.get("cleanup_verified") is not True:
        errors.append("runtime_audit_invalid")
    fixture = value.get("fixture_artifact") is True
    if not fixture:
        errors.extend(validate_current_work_receipt(value, root=root))
    candidate = value.get("candidate_artifact") is True
    validation_ok = _validation_passed(value.get("validation_receipts") or [], candidate=candidate)
    expected_complete = int(
        len(rows) == 104
        and not any(row.get("errors") for row in rows if isinstance(row, Mapping))
        and not runtime.get("errors")
        and validation_ok
    )
    if value.get("extraction_audit_complete_score") != expected_complete:
        errors.append("audit_complete_score_mismatch")
    source = value.get("audited_scientific_disposition") or {}
    if (
        source.get("preserved_without_laundering") is not True
        or source.get("verdict_class") != "null"
    ):
        errors.append("source_disposition_laundered")
    if value.get("verdict_class") != "null" or not str(value.get("honest_verdict")).startswith(
        "complete_null_"
    ):
        errors.append("terminal_disposition_invalid")
    attacks = value.get("audit_mutation_rows") or []
    if {row.get("attack") for row in attacks if isinstance(row, Mapping)} != set(
        REQUIRED_MUTATIONS
    ) or any(row.get("passed") is not True for row in attacks if isinstance(row, Mapping)):
        errors.append("mutation_controls_invalid")
    if verify_source_bytes:
        for label, reference in (value.get("source_artifact_hashes") or {}).items():
            if not isinstance(reference, Mapping):
                errors.append(f"source_reference_invalid:{label}")
                continue
            path = Path(str(reference.get("path") or ""))
            resolved = path if path.is_absolute() else root / path
            observed = sha256_file(resolved) if resolved.is_file() else None
            if observed != reference.get("sha256"):
                errors.append(f"source_hash_mismatch:{label}")
        for reference in value.get("historical_evidence_sidecars") or []:
            if not isinstance(reference, Mapping):
                errors.append("historical_sidecar_reference_invalid")
                continue
            resolved = root / str(reference.get("path") or "")
            observed = sha256_file(resolved) if resolved.is_file() else None
            if observed != reference.get("sha256"):
                errors.append(f"historical_sidecar_hash_mismatch:{reference.get('path')}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(
    path: Path, *, root: Path = REPO_ROOT, verify_source_bytes: bool = True
) -> list[str]:
    """Reload one candidate and validate it without producer execution."""

    value = _load_object(path)
    return (
        validate_artifact(value, root=root, verify_source_bytes=verify_source_bytes)
        if value
        else ["candidate_artifact_unreadable"]
    )


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> list[str]:
    """Reload raw producer sidecars and compare every scientific projection."""

    value = _load_object(path)
    errors = validate_artifact(value, root=root)
    producer = _load_object(root / EXP7451_PATH)
    if producer:
        evidence = audit_producer(root, producer)
        for field, observed in (
            ("rows", evidence["rows"]),
            ("raw_disposition_counts", evidence["disposition_counts"]),
            ("sample_size_budget", evidence["sample_size_budget"]),
            ("paired_completion_effect", evidence["paired_completion_effect"]),
            ("paired_token_cost_effect", evidence["paired_token_cost_effect"]),
            ("coverage_rows", evidence["coverage_rows"]),
            ("runtime_audit", evidence["runtime_audit"]),
        ):
            if canonical_hash(value.get(field)) != canonical_hash(observed):
                errors.append(f"independent_reduction_mismatch:{field}")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7358/Exp7303 affected command set."""

    commands = build_command_plan(root, V653_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, V653_MANIFEST, commands)
    if plan_errors:
        raise ValueError("validation_plan_invalid:" + ",".join(plan_errors))
    return commands


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:
    """Build capability replay, independent reduction, and unchanged readers."""

    values = (
        (
            "declared_entrypoint_cold_replay",
            (".venv/bin/python", "-u", WRAPPER_PATH.as_posix(), "--cold-replay", str(candidate)),
            "completion",
        ),
        (
            "independent_cold_reducer",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--independent-reduce",
                str(candidate),
            ),
            "completion",
        ),
        (
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "completion",
        ),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate", timeout_s=1200.0),
            category,
            True,
        )
        for name, argv, category in values
    ]


def _utc_now() -> str:  # pragma: no cover - real clock boundary.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7456] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": _utc_now(),
    }


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised through the capability entrypoint.
    """Authenticate, audit, validate, replay, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    _progress(started, "startup", "start")

    phase_started = time.monotonic()
    _progress(started, "preconditions", "before_authentication")
    checks, sources, producer = collect_preconditions(root)
    blocked = any(
        row.get("check") in {"exp7451_artifact_bytes", "exp7451_current_inference"}
        and row.get("passed") is not True
        for row in checks
    )
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    _progress(started, "preconditions", "after_authentication", blocked=blocked)
    if blocked:
        result = build_blocked_artifact(checks, sources)
        _progress(started, "terminal_write", "before_atomic", path=output)
        atomic_json(output, result)
        _progress(started, "terminal_write", "after_atomic", status=result["status"])
        return result

    phase_started = time.monotonic()
    _progress(started, "source_audit", "before_benchmark")
    evidence = audit_producer(root, producer)
    for relative, receipt_class in (
        (MODULE_PATH, "current_audit_code"),
        (WRAPPER_PATH, "current_audit_entrypoint"),
        (TEST_PATH, "current_audit_tests"),
        (SPEC_PATH, "current_audit_protocol"),
    ):
        sources[relative.as_posix()] = _reference(root, relative, receipt_class)
    spans.append(_span("source_audit", phase_started, started, len(evidence["rows"])))
    _progress(
        started,
        "source_audit",
        "after_benchmark",
        completed_units=len(evidence["rows"]),
        errors=len(evidence["errors"]),
    )

    private = Path(tempfile.mkdtemp(prefix="exp7456-validation-", dir="/tmp"))
    commands = build_validation_commands(root, private)
    phase_started = time.monotonic()
    _progress(started, "affected_validation", "before_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
    )
    affected_reduction = reduce_affected_receipts(root, V653_MANIFEST, affected)
    spans.append(_span("affected_validation", phase_started, started, len(affected)))
    _progress(
        started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )

    elapsed_ns = time.monotonic_ns() - started_ns
    receipt_sidecars = [
        {
            "path": EXP7451_PATH.as_posix(),
            "sha256": sha256_file(root / EXP7451_PATH),
            "scope": "historical_model_receipts",
        }
    ]
    receipt = _current_receipt(elapsed_ns, spans=spans, sidecars=receipt_sidecars)
    candidate = _build_artifact(
        evidence,
        checks=checks,
        sources=sources,
        producer=producer,
        receipts=affected,
        current_receipt=receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        candidate=True,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    _progress(started, "candidate_write", "before_atomic", path=candidate_path)
    atomic_json(candidate_path, candidate)
    _progress(started, "candidate_write", "after_atomic", bytes=candidate_path.stat().st_size)

    phase_started = time.monotonic()
    terminal_plan = _terminal_commands(candidate_path)
    _progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row.get("passed") is True for row in terminal),
    )

    final_receipt = _current_receipt(
        time.monotonic_ns() - started_ns, spans=spans, sidecars=receipt_sidecars
    )
    final = _build_artifact(
        evidence,
        checks=checks,
        sources=sources,
        producer=producer,
        receipts=[*affected, *terminal],
        current_receipt=final_receipt,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        candidate=False,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _progress(started, "terminal_write", "before_atomic", path=output)
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    _progress(started, "terminal_write", "after_atomic", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin audit entrypoint arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--skip-source-bytes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit or one fresh-process candidate reader."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, verify_source_bytes=not args.skip_source_bytes)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = (
            cold_replay(args.independent_reduce, verify_source_bytes=False)
            if args.skip_source_bytes
            else independent_replay(args.independent_reduce)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
