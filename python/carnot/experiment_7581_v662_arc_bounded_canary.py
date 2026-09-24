"""Run the V662 bounded ARC transport canary.

The canary asks only whether two owned local requests can cross the same
transport that later ARC panels will use. It does not judge reply quality and
does not claim a solve or a better world model.

Spec refs: REQ-ARC-WMTE-7581 and SCENARIO-ARC-WMTE-7581-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading
import time
from typing import Any

from carnot.inference.sota_models import cached_current_model
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7581-v662-arc-bounded-canary"
SCHEMA = "carnot.exp7581.v662.arc_bounded_canary.v1"
PROTOCOL_SCHEMA = "carnot.exp7580.live_panel_protocol.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]

RESULT_REL = Path("results/experiment_7581_v662_arc_bounded_canary.json")
RAW_REL = Path("results/raw/experiment_7581_v662_arc_bounded_canary")
EXP7574_REL = Path("results/experiment_7574_v662_measurement_requalification.json")
EXP7580_REL = Path("results/experiment_7580_v662_arc_verifier_support.json")
PROTOCOL_REL = Path(
    "results/raw/experiment_7580_v662_arc_verifier_support/live_panel_protocol.json"
)
HISTORY_REL = Path("results/experiment_7471_v654_arc_seam_observation.json")
SPEC_REL = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_REL = Path("python/carnot/experiment_7581_v662_arc_bounded_canary.py")
WRAPPER_REL = Path("scripts/experiments/experiment_7581_v662_arc_bounded_canary.py")
TEST_REL = Path("tests/python/test_experiment_7581_v662_arc_bounded_canary.py")

EXP7574_SHA256 = "sha256:0502b10f8efdc0b4bf751a531b50562662b0eacd5998ceeae82fe2393facf6e1"
PROTOCOL_SHA256 = "sha256:7fc600eb5dd11f969e841adb1cc62d5e0174b9005d5462dba95db42ddfebb897"
HISTORY_SHA256 = "sha256:a2691ec80fec30cd5128c692ccda6852b638fc7ef5615305479d502e193ce647"

CANARY_MAX_TOKENS = 64
FUTURE_MAX_TOKENS = 4096
REQUEST_TIMEOUT_S = 240.0
CAPACITY_WAIT_S = 300.0
MEASUREMENT_CEILING_S = 3000.0
VALIDATION_RESERVE_S = 900.0
TOTAL_WITH_RESERVE_CEILING_S = 4200.0
MIN_GENERATION_DURATION_S = 10.0
GPU_IDLE_MAX_USED_MB = 500
GPU_REQUIRED_FREE_MB = 20_000
NEUTRAL_PROMPTS = (
    "Reply with one short sentence confirming that this local transport request arrived.",
    "Name one ordinary color in a short sentence. Do not solve a puzzle or write code.",
)

REQUIRED_REPOSITORY_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/arc_solve_registry.yaml"),
    SPEC_REL,
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_decision_telemetry.py"),
    MODULE_REL,
    WRAPPER_REL,
    TEST_REL,
)

ZERO_INVOCATION_COUNTS = {
    **{
        f"{kind}_{state}": 0
        for kind in ("model_loads", "forward_calls", "generation_calls")
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    },
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

E2E_NAMES = (
    "e2e_009",
    "e2e_010",
    "e2e_011",
    "e2e_012",
    "e2e_013",
    "foreign_cwd_llm_off_e3_smoke",
)
TERMINAL_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:
    """Return one timezone-aware wall boundary for a durable receipt."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one truthful phase boundary or pending-operation heartbeat."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7581] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so one changed operand changes the receipt identity."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON only after file sync and an atomic rename."""

    current_work_receipt.atomic_json(path, value)


def load_json(path: Path) -> Json:
    """Read one JSON object and reject a scalar or list top level."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def gate_row(
    check: str,
    *,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    op: str = "eq",
    passed: bool | None = None,
    category: str = "validity",
) -> Json:
    """Keep every operand needed to explain why dependent work did or did not run."""

    result = observed == expected if passed is None else bool(passed)
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": result,
        "category": category,
        "principle": "A failed gate cannot be hidden or replaced by substitute evidence.",
    }


def failed_checks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return failed check names without losing their detailed rows."""

    return [str(row.get("check")) for row in rows if row.get("passed") is not True]


def _file_gate(root: Path, relative: Path) -> Json:
    path = (root / relative).resolve()
    available = path.is_file() and path.stat().st_size > 0
    return gate_row(
        f"required_input:{relative.as_posix()}",
        upstream=relative.as_posix(),
        path=str(path),
        field="path.bytes",
        expected="readable_nonempty_bytes",
        observed="readable_nonempty_bytes" if available else None,
    )


def _read_optional_object(path: Path) -> Json:
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _model_declaration(model: Mapping[str, Any] | None) -> Json:
    """Preserve the mandate even when cache resolution itself is the failed gate."""

    if model:
        return deepcopy(dict(model))
    return {
        "name": "Qwen3.8-27B",
        "hf_id": MODEL_ID,
        "model_path": None,
        "selection_role": "current_headline",
        "resolution_status": "cache_miss",
    }


def collect_preconditions(
    root: Path,
    *,
    model_resolver: Callable[[], Mapping[str, Any] | None] = cached_current_model,
    expected_hashes: Mapping[str, str] | None = None,
) -> tuple[list[Json], Json]:
    """Authenticate exact upstream bytes and local resources before measurement."""

    base = root.resolve()
    frozen_hashes = dict(
        expected_hashes
        if expected_hashes is not None
        else {
            EXP7574_REL.as_posix(): EXP7574_SHA256,
            PROTOCOL_REL.as_posix(): PROTOCOL_SHA256,
            HISTORY_REL.as_posix(): HISTORY_SHA256,
        }
    )
    checks = [_file_gate(base, relative) for relative in REQUIRED_REPOSITORY_INPUTS]
    for relative in (EXP7574_REL, EXP7580_REL, PROTOCOL_REL):
        checks.append(_file_gate(base, relative))
    if HISTORY_REL.as_posix() in frozen_hashes:
        checks.append(_file_gate(base, HISTORY_REL))

    runner_path = (base / EXP7574_REL).resolve()
    support_path = (base / EXP7580_REL).resolve()
    protocol_path = (base / PROTOCOL_REL).resolve()
    history_path = (base / HISTORY_REL).resolve()
    runner = _read_optional_object(runner_path)
    support = _read_optional_object(support_path)
    protocol = _read_optional_object(protocol_path)
    history = _read_optional_object(history_path)

    for relative, name in (
        (EXP7574_REL, "exp7574"),
        (PROTOCOL_REL, "exp7580_protocol"),
        (HISTORY_REL, "historical_transport"),
    ):
        expected = frozen_hashes.get(relative.as_posix())
        if expected is None:
            continue
        path = (base / relative).resolve()
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            gate_row(
                f"{name}_hash",
                upstream=relative.as_posix(),
                path=str(path),
                field="sha256",
                expected=expected,
                observed=observed,
            )
        )

    comparisons = (
        (
            "exp7574_identity",
            EXP7574_REL,
            runner,
            "experiment_id",
            "exp7574-v662-measurement-requalification",
        ),
        ("exp7574_runner_ready", EXP7574_REL, runner, "arc_runner_ready_score", 1),
        ("exp7574_not_flagged", EXP7574_REL, runner, "flagged_adversarial", False),
        ("exp7580_identity", EXP7580_REL, support, "experiment_id", 7580),
        ("exp7580_support_ready", EXP7580_REL, support, "verifier_support_ready_score", 1),
        ("exp7580_not_flagged", EXP7580_REL, support, "flagged_adversarial", False),
        ("exp7580_protocol_schema", PROTOCOL_REL, protocol, "schema", PROTOCOL_SCHEMA),
    )
    for check, relative, value, field, expected in comparisons:
        path = str((base / relative).resolve())
        checks.append(
            gate_row(
                check,
                upstream=relative.as_posix(),
                path=path,
                field=field,
                expected=expected,
                observed=value.get(field),
            )
        )

    spec_path = (base / SPEC_REL).resolve()
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    checks.append(
        gate_row(
            "requirement_declared",
            upstream=SPEC_REL.as_posix(),
            path=str(spec_path),
            field="REQ-*",
            expected="REQ-ARC-WMTE-7581",
            observed="REQ-ARC-WMTE-7581" if "REQ-ARC-WMTE-7581" in spec_text else None,
        )
    )
    resolved_model = model_resolver()
    model_path = Path(str((resolved_model or {}).get("model_path") or "/nonexistent"))
    model_ok = bool(
        resolved_model
        and resolved_model.get("hf_id") == MODEL_ID
        and model_path.is_absolute()
        and model_path.is_file()
    )
    checks.append(
        gate_row(
            "cached_current_model",
            upstream="carnot.inference.sota_models.cached_current_model",
            path=str(model_path),
            field="hf_id,path",
            expected={"hf_id": MODEL_ID, "absolute_file": True},
            observed={
                "hf_id": (resolved_model or {}).get("hf_id"),
                "absolute_file": model_path.is_absolute() and model_path.is_file(),
            },
            passed=model_ok,
        )
    )
    context = {
        "root": str(base),
        "runner": runner,
        "support": support,
        "protocol": protocol,
        "history": history,
        "model": _model_declaration(resolved_model),
        "expected_hashes": frozen_hashes,
    }
    return checks, context


def validate_future_requests(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int = 24
) -> None:
    """Reject roster loss and the obsolete 256-token capture ceiling."""

    required = {
        "panel",
        "game",
        "seed",
        "arm",
        "max_tokens",
        "capture_max_tokens",
        "request_timeout_s",
        "requests_per_episode",
        "request_sent",
    }
    if len(rows) != expected_count:
        raise ValueError(f"future_request_contract:count:{len(rows)}:{expected_count}")
    identities: set[tuple[str, str, int, str]] = set()
    for index, row in enumerate(rows):
        if not required <= set(row):
            raise ValueError(f"future_request_contract:missing_fields:{index}")
        identity = (
            str(row["panel"]),
            str(row["game"]),
            int(row["seed"]),
            str(row["arm"]),
        )
        identities.add(identity)
        if (
            row.get("max_tokens") != FUTURE_MAX_TOKENS
            or row.get("capture_max_tokens") != FUTURE_MAX_TOKENS
            or float(row.get("request_timeout_s", -1)) != REQUEST_TIMEOUT_S
            or row.get("requests_per_episode") != 1
            or row.get("request_sent") is not False
        ):
            raise ValueError(f"future_request_contract:limits:{index}")
    if len(identities) != expected_count:
        raise ValueError("future_request_contract:duplicate_identity")


def construct_future_requests(protocol: Mapping[str, Any]) -> list[Json]:
    """Construct every frozen panel request envelope without sending it."""

    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("future_request_contract:protocol_schema")
    source_rows = protocol.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError("future_request_contract:protocol_rows")
    rows: list[Json] = []
    for index, source in enumerate(source_rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"future_request_contract:protocol_row:{index}")
        rows.append(
            {
                "request_id": (
                    f"future:{source.get('panel')}:{source.get('game')}:"
                    f"{source.get('seed')}:{source.get('arm')}"
                ),
                "panel": str(source.get("panel")),
                "game": str(source.get("game")),
                "seed": int(source.get("seed", -1)),
                "arm": str(source.get("arm")),
                "max_tokens": FUTURE_MAX_TOKENS,
                "capture_max_tokens": FUTURE_MAX_TOKENS,
                "request_timeout_s": REQUEST_TIMEOUT_S,
                "requests_per_episode": 1,
                "request_sent": False,
                "endpoint": "/v1/chat/completions",
                "chat_template": "embedded_model_template",
                "prompt_source": "real_E3AgentPolicy_runtime_state_not_read_by_canary",
                "capture_class": "DurableRequestCapture",
                "censored": False,
                "provenance": "exp7580_frozen_protocol_constructed_unsent",
            }
        )
    validate_future_requests(rows)
    panels = Counter(str(row["panel"]) for row in rows)
    if panels != Counter({"A": 12, "B": 12}):
        raise ValueError(f"future_request_contract:panel_counts:{dict(panels)}")
    return rows


def _linear_p95(values: Sequence[float]) -> float:
    """Return the same deterministic linear percentile used by prior ARC runners."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("historical_request_timings_empty")
    if len(ordered) == 1:
        return ordered[0]
    rank = 0.95 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])


def historical_request_profile(artifact: Mapping[str, Any], *, source_hash: str) -> Json:
    """Authenticate and reduce historical Qwen transport rows to one p95."""

    specs = artifact.get("MODEL_SPECS")
    model_ids = {
        str(row.get("hf_id")) if isinstance(row, Mapping) else str(row) for row in specs or []
    }
    rows = artifact.get("rows")
    durations: list[float] = []
    token_ceilings: list[int] = []
    if isinstance(rows, list):
        for episode in rows:
            if not isinstance(episode, Mapping):
                continue
            for request in episode.get("server_request_rows") or []:
                if not isinstance(request, Mapping) or request.get("response_observed") is not True:
                    continue
                duration = request.get("elapsed_s")
                ceiling = request.get("requested_max_tokens")
                if isinstance(duration, (int, float)) and float(duration) >= 0:
                    durations.append(float(duration))
                    token_ceilings.append(int(ceiling or 0))
    declared_completed = int(
        (artifact.get("invocation_counts") or {}).get("generation_calls_completed") or 0
    )
    authenticated = bool(
        source_hash.startswith("sha256:")
        and MODEL_ID in model_ids
        and artifact.get("flagged_adversarial") is False
        and artifact.get("verdict_class") in {"null", "positive", "circular_positive"}
        and declared_completed == len(durations)
        and durations
        and len(set(token_ceilings)) == 1
        and token_ceilings[0] > 0
    )
    if not authenticated:
        raise ValueError("historical_request_profile_not_authenticated")
    return {
        "source_path": HISTORY_REL.as_posix(),
        "source_sha256": source_hash,
        "model_id": MODEL_ID,
        "sample_count": len(durations),
        "historical_max_tokens": token_ceilings[0],
        "durations_s": durations,
        "p95_request_s": _linear_p95(durations),
        "authenticated": True,
    }


def forecast_panels(
    future_requests: Sequence[Mapping[str, Any]],
    historical_profile: Mapping[str, Any],
    *,
    measured_load_s: float,
    measured_capture_s: float,
) -> dict[str, Json]:
    """Forecast each frozen panel independently before panel outcomes exist."""

    validate_future_requests(future_requests)
    old_ceiling = int(historical_profile["historical_max_tokens"])
    if old_ceiling <= 0 or historical_profile.get("authenticated") is not True:
        raise ValueError("panel_forecast_history_invalid")
    scaled_p95 = float(historical_profile["p95_request_s"]) * (FUTURE_MAX_TOKENS / old_ceiling)
    per_request = min(REQUEST_TIMEOUT_S, scaled_p95)
    forecasts: dict[str, Json] = {}
    for panel in ("A", "B"):
        panel_rows = [row for row in future_requests if row.get("panel") == panel]
        request_count = len(panel_rows)
        measurement = (
            float(measured_load_s) + float(measured_capture_s) + request_count * per_request
        )
        total = measurement + VALIDATION_RESERVE_S
        feasible = measurement <= MEASUREMENT_CEILING_S and total <= TOTAL_WITH_RESERVE_CEILING_S
        forecasts[panel] = {
            "panel": panel,
            "request_count": request_count,
            "episode_count": request_count,
            "historical_p95_request_s": float(historical_profile["p95_request_s"]),
            "historical_max_tokens": old_ceiling,
            "scaled_p95_request_s": scaled_p95,
            "forecast_request_s": per_request,
            "max_tokens": FUTURE_MAX_TOKENS,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "measured_load_s": float(measured_load_s),
            "measured_capture_s": float(measured_capture_s),
            "forecast_measurement_s": measurement,
            "validation_reserve_s": VALIDATION_RESERVE_S,
            "forecast_with_validation_s": total,
            "measurement_ceiling_s": MEASUREMENT_CEILING_S,
            "total_ceiling_s": TOTAL_WITH_RESERVE_CEILING_S,
            "panel_feasible_score": int(feasible and request_count == 12),
            "roster_shrunk_after_outcomes": False,
            "source_sha256": historical_profile["source_sha256"],
        }
    return forecasts


def can_start_future_request(remaining_measurement_s: float) -> bool:
    """Start a later full request only when its whole timeout still fits."""

    return float(remaining_measurement_s) >= REQUEST_TIMEOUT_S


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def request_receipt_from_bytes(
    *,
    request_id: str,
    request_bytes: bytes,
    response_bytes: bytes,
    elapsed_s: float,
    process_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    model_sha256: str,
) -> Json:
    """Parse one captured byte pair without trusting proposer summary fields."""

    request = json.loads(request_bytes)
    response = json.loads(response_bytes)
    messages = request.get("messages") if isinstance(request, Mapping) else None
    first = messages[0] if isinstance(messages, list) and messages else {}
    prompt = str(first.get("content") or "") if isinstance(first, Mapping) else ""
    choices = response.get("choices") if isinstance(response, Mapping) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, Mapping) else {}
    message = choice.get("message")
    message = message if isinstance(message, Mapping) else {}
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    reply = content if content else reasoning
    usage = response.get("usage") if isinstance(response, Mapping) else {}
    usage = usage if isinstance(usage, Mapping) else {}
    owner_pid = process_identity.get("pid")
    owner_tick = process_identity.get("pid_start_ticks")
    server_pid = runtime_identity.get("server_pid")
    server_tick = runtime_identity.get("server_pid_start_ticks")
    owned = all(
        isinstance(value, int) and value > 0
        for value in (owner_pid, owner_tick, server_pid, server_tick)
    )
    return {
        "request_id": request_id,
        "prompt": prompt,
        "reply": reply,
        "request_bytes_sha256": _sha256_bytes(request_bytes),
        "response_bytes_sha256": _sha256_bytes(response_bytes),
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "reply_sha256": _sha256_bytes(reply.encode()),
        "requested_max_tokens": int(request.get("max_tokens") or 0),
        "actual_prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "actual_completion_tokens": int(usage.get("completion_tokens") or 0),
        "actual_total_tokens": int(usage.get("total_tokens") or 0),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "elapsed_s": float(elapsed_s),
        "request_started": True,
        "response_observed": True,
        "owned_runtime": owned,
        "process_identity": deepcopy(dict(process_identity)),
        "runtime_identity": deepcopy(dict(runtime_identity)),
        "model_sha256": model_sha256,
        "censored": False,
        "provenance": "owned_bounded_transport",
    }


def reduce_transport_receipts(
    receipts: Sequence[Mapping[str, Any]], *, generation_duration_s: float
) -> Json:
    """Require exactly two owned, captured 64-token calls and no semantic quality."""

    attempted = sum(row.get("request_started") is True for row in receipts)
    completed = sum(
        row.get("request_started") is True and row.get("response_observed") is True
        for row in receipts
    )
    prompt_tokens = sum(int(row.get("actual_prompt_tokens") or 0) for row in receipts)
    completion_tokens = sum(int(row.get("actual_completion_tokens") or 0) for row in receipts)
    token_caps_ok = all(row.get("requested_max_tokens") == CANARY_MAX_TOKENS for row in receipts)
    bytes_ok = all(
        str(row.get(field) or "").startswith("sha256:")
        for row in receipts
        for field in ("prompt_sha256", "reply_sha256")
    )
    custody_ok = all(row.get("owned_runtime") is True for row in receipts)
    finish_ok = all(bool(row.get("finish_reason")) for row in receipts)
    duration_ok = float(generation_duration_s) >= MIN_GENERATION_DURATION_S
    ready = bool(
        len(receipts) == 2
        and attempted == 2
        and completed == 2
        and token_caps_ok
        and bytes_ok
        and custody_ok
        and finish_ok
        and duration_ok
    )
    return {
        "arc_transport_ready_score": int(ready),
        "generation_calls_attempted": attempted,
        "generation_calls_completed": completed,
        "generation_calls_failed": attempted - completed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "token_caps_passed": token_caps_ok,
        "capture_bytes_passed": bytes_ok,
        "owned_runtime_passed": custody_ok,
        "finish_reason_passed": finish_ok,
        "generation_duration_s": float(generation_duration_s),
        "duration_floor_s": MIN_GENERATION_DURATION_S,
        "duration_floor_passed": duration_ok,
        "semantic_quality_evaluated": False,
    }


def actual_substrate_class(transport: Mapping[str, Any]) -> str:
    """Never call a load-only or zero-call run live bounded generation."""

    return (
        "model_bounded_generation"
        if int(transport.get("generation_calls_attempted") or 0) > 0
        else "blocked_no_run"
    )


def build_comparative_rows(
    *,
    future_requests: Sequence[Mapping[str, Any]],
    raw_request_receipts: Sequence[Mapping[str, Any]],
    forecasts: Mapping[str, Mapping[str, Any]],
) -> list[Json]:
    """Emit one raw-operands row for every current and frozen future unit."""

    rows: list[Json] = []
    for index, receipt in enumerate(raw_request_receipts):
        success = int(
            receipt.get("request_started") is True
            and receipt.get("response_observed") is True
            and receipt.get("owned_runtime") is True
        )
        rows.append(
            {
                "row_kind": "current_canary",
                "comparison_unit": str(receipt.get("request_id") or f"canary-{index}"),
                "arm": "owned_transport",
                "raw_numerator": success,
                "raw_denominator": 1,
                "metric": "authenticated_transport_success",
                "metric_direction": "higher_is_better",
                "seed": 7_581_001 + index,
                "censored": bool(receipt.get("censored")),
                "provenance": str(receipt.get("provenance") or "owned_bounded_transport"),
            }
        )
    for request in future_requests:
        panel = str(request["panel"])
        forecast = forecasts[panel]
        rows.append(
            {
                "row_kind": "future_panel",
                "comparison_unit": str(request["request_id"]),
                "arm": str(request["arm"]),
                "panel": panel,
                "game": str(request["game"]),
                "raw_numerator": float(forecast["forecast_request_s"]),
                "raw_denominator": 1,
                "metric": "forecast_request_seconds",
                "metric_direction": "lower_is_better",
                "seed": int(request["seed"]),
                "censored": bool(request.get("censored")),
                "provenance": str(request["provenance"]),
            }
        )
    return rows


def independent_reduce_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Recompute canary success and per-panel forecast totals from raw operands."""

    keys: set[tuple[str, str]] = set()
    canary_numerator = 0.0
    canary_denominator = 0
    panels = {
        "A": {"row_count": 0, "forecast_request_seconds": 0.0},
        "B": {"row_count": 0, "forecast_request_seconds": 0.0},
    }
    for row in rows:
        key = (str(row.get("comparison_unit")), str(row.get("arm")))
        if key in keys:
            raise ValueError(f"duplicate_comparison_row:{key}")
        keys.add(key)
        denominator = int(row.get("raw_denominator") or 0)
        if denominator <= 0:
            raise ValueError(f"invalid_row_denominator:{key}")
        if row.get("row_kind") == "current_canary":
            canary_numerator += float(row.get("raw_numerator") or 0)
            canary_denominator += denominator
        elif row.get("row_kind") == "future_panel":
            panel = str(row.get("panel"))
            if panel not in panels:
                raise ValueError(f"invalid_panel:{panel}")
            panels[panel]["row_count"] += 1
            panels[panel]["forecast_request_seconds"] += float(row.get("raw_numerator") or 0)
        else:
            raise ValueError(f"invalid_row_kind:{row.get('row_kind')}")
    return {
        "canary": {
            "success_numerator": int(canary_numerator),
            "denominator": canary_denominator,
            "rate": canary_numerator / canary_denominator if canary_denominator else None,
        },
        "panels": panels,
        "row_count": len(rows),
    }


def exercise_learning_lifecycle(state_path: Path) -> Json:
    """Exercise predict, delayed release, update, persistence, and cold reload."""

    from carnot.experiment_7561_v661_recalibration_prototype import (
        RecalibrationEventMachine,
        fixture_count_config,
    )

    machine = RecalibrationEventMachine.create(fixture_count_config())
    before = machine.state_hash()
    prediction = machine.predict("exp7581-control-0", 0.25, 0)
    release = machine.release(0, [("exp7581-control-0", 1)])
    machine.acknowledge(0)
    after_update = machine.state_hash()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    machine.save(state_path)
    persisted_hash = sha256_file(state_path)
    loaded = RecalibrationEventMachine.load(state_path)
    after_reload = loaded.state_hash()
    passed = bool(
        prediction.get("label_available_at_prediction") is False
        and release.get("update_count") == 1
        and before != after_update
        and after_update == after_reload
        and loaded.next_release_index == 1
    )
    return {
        "operations": ["predict", "release", "update", "persist", "reload"],
        "prediction_label_available": prediction.get("label_available_at_prediction"),
        "released_update_count": release.get("update_count"),
        "state_sha256_before_update": before,
        "state_sha256_before_reload": after_update,
        "persisted_file_sha256": persisted_hash,
        "state_sha256_after_reload": after_reload,
        "next_release_index_after_reload": loaded.next_release_index,
        "model_weights_changed": False,
        "benefit_claim": False,
        "passed": passed,
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    principles = {
        "honest_verdict": "Use a complete_ terminal prefix; completion does not establish benefit.",
        "verdict_class": "Use exactly one of positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Persist the exact terminal verification outcome; flagged evidence opens no readiness.",
        "gate_check_summary": "For blocked work, name the failed check and every comparison operand.",
        "acceptance_gate_results": "Separate validity, readiness, and benefit so a valid null remains usable.",
        "rows": "Keep one row per comparison unit and arm with raw operands, direction, seed, censoring, and provenance.",
        "inference_substrate_class": "Record actual substrate separately from planned work; zero calls mean blocked no-run.",
        "planned_inference_substrate_class": "Declare the intended bounded generation class before admission.",
        "MODEL_SPECS": "Name the mandated current Qwen model and its cache resolution without substitution.",
        "invocation_counts": "Count current loads, forwards, generations, and tokens separately from historical evidence.",
        "duration_s": "Measure current monotonic work without inherited time or artificial sleeps.",
        "source_artifact_hashes": "Bind conclusions to exact bytes and distinguish a missing producer from a failed gate.",
        "validation_receipts": "Bind every check to command, worktree, exit code, and log hash.",
        "field_principles": "Carry these one-line interpretation rules beside emitted fields.",
        "verifier_is_oracle": "Oracle or label-accessing controls cannot support an oracle-distinct positive claim.",
        "arc_transport_ready_score": "One requires two authenticated owned bounded requests and qualified runner custody.",
        "panel_a_feasible_score": "One requires the complete frozen Panel A forecast plus reserve to fit.",
        "panel_b_feasible_score": "One requires Panel B's complete forecast independently of Panel A outcomes.",
        "raw_request_receipts": "Bind current model, bytes, timing, tokens, finish reason, and owned runtime.",
        "solve_provenance": "The canary has no solve claim; live_agent_self_discovery begins only in later episodes.",
    }
    return {
        key: principles.get(key, f"Bind {key} so an independent reader can detect drift.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash the full artifact except its self-referential checksum."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Retain all failures and expose the first exact one for operators."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [str(row.get("check")) for row in failures],
        "first_failure": failures[0] if failures else None,
    }


def _unstarted_rows(protocol: Mapping[str, Any]) -> list[Json]:
    """Retain planned comparison units when an external gate blocks all calls."""

    try:
        future = construct_future_requests(protocol)
    except (KeyError, TypeError, ValueError):
        future = []
    rows = [
        {
            "row_kind": "current_canary",
            "comparison_unit": f"canary-{index}",
            "arm": "owned_transport",
            "raw_numerator": 0,
            "raw_denominator": 1,
            "metric": "authenticated_transport_success",
            "metric_direction": "higher_is_better",
            "seed": 7_581_001 + index,
            "censored": True,
            "provenance": "blocked_unstarted",
        }
        for index in range(2)
    ]
    rows.extend(
        {
            "row_kind": "future_panel",
            "comparison_unit": str(request["request_id"]),
            "arm": str(request["arm"]),
            "panel": str(request["panel"]),
            "game": str(request["game"]),
            "raw_numerator": 0.0,
            "raw_denominator": 1,
            "metric": "forecast_request_seconds",
            "metric_direction": "lower_is_better",
            "seed": int(request["seed"]),
            "censored": True,
            "provenance": "blocked_unstarted",
        }
        for request in future
    )
    return rows


def build_blocked_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    duration_s: float,
    reason: str | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
) -> Json:
    """Publish a complete no-run record without invented model evidence."""

    failure = next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)
    if failure is None:
        raise ValueError("blocked_artifact_requires_failed_check")
    raw_reason = reason or str(failure.get("check") or "precondition")
    clean_reason = "".join(character if character.isalnum() else "_" for character in raw_reason)
    rows = _unstarted_rows(context.get("protocol") or {})
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": f"complete_blocked_{clean_reason}",
        "honest_verdict": f"complete_blocked_{clean_reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "positive_claim": False,
        "world_model_quality_claim": False,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "planned_inference_substrate_class": "model_bounded_generation",
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "blocked_no_run",
        "MODEL_SPECS": [deepcopy(dict(context.get("model") or _model_declaration(None)))],
        "model_specs": [deepcopy(dict(context.get("model") or _model_declaration(None)))],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "duration_s": max(0.0, float(duration_s)),
        "raw_request_receipts": [],
        "future_request_constructions": [],
        "panel_forecasts": {},
        "rows": rows,
        "independent_reduction": independent_reduce_rows(rows),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes or {})),
        "acceptance_gate_results": [deepcopy(failure)],
        "gate_check_summary": gate_summary([failure]),
        "verifier_is_oracle": False,
        "arc_transport_ready_score": 0,
        "panel_a_feasible_score": 0,
        "panel_b_feasible_score": 0,
        "solve_provenance": "canary_no_solve_claim",
        "learning_lifecycle": {"passed": False, "operations": []},
        "retired_literal_prior_verdict": {
            "literal": "Exp7570 full generation result",
            "replacement": "Exp7570 failed preflight before any model call",
            "scope": "transport classification only",
            "scientific_hypothesis_retired": False,
        },
    }
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _acceptance_gate(check: str, category: str, expected: Any, observed: Any, passed: bool) -> Json:
    return gate_row(
        check,
        upstream="current_exp7581_reduction",
        path=RESULT_REL.as_posix(),
        field=check,
        expected=expected,
        observed=observed,
        passed=passed,
        category=category,
    )


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    required = (*validation_scope.REQUIRED_CHECK_NAMES, *E2E_NAMES, *TERMINAL_NAMES)
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is not True
        for name in required
    )


def build_complete_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    raw_request_receipts: Sequence[Mapping[str, Any]],
    future_requests: Sequence[Mapping[str, Any]],
    historical_profile: Mapping[str, Any],
    panel_forecasts: Mapping[str, Mapping[str, Any]],
    learning_lifecycle: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    duration_s: float,
    generation_duration_s: float,
    require_validation: bool,
) -> Json:
    """Classify transport, budget, validation, and benefit on separate gates."""

    receipts = [deepcopy(dict(row)) for row in raw_request_receipts]
    transport = reduce_transport_receipts(receipts, generation_duration_s=generation_duration_s)
    rows = build_comparative_rows(
        future_requests=future_requests,
        raw_request_receipts=receipts,
        forecasts=panel_forecasts,
    )
    reduction = independent_reduce_rows(rows)
    preconditions_ok = not failed_checks(checks)
    panel_a = int((panel_forecasts.get("A") or {}).get("panel_feasible_score") or 0)
    panel_b = int((panel_forecasts.get("B") or {}).get("panel_feasible_score") or 0)
    validation_ok = _validation_passed(validation_receipts) if require_validation else True
    lifecycle_ok = learning_lifecycle.get("passed") is True
    gates = [
        _acceptance_gate("preconditions", "validity", True, preconditions_ok, preconditions_ok),
        _acceptance_gate(
            "owned_bounded_transport",
            "readiness",
            1,
            transport["arc_transport_ready_score"],
            transport["arc_transport_ready_score"] == 1,
        ),
        _acceptance_gate("panel_a_budget", "readiness", 1, panel_a, panel_a == 1),
        _acceptance_gate("panel_b_budget", "readiness", 1, panel_b, panel_b == 1),
        _acceptance_gate("learning_lifecycle", "validity", True, lifecycle_ok, lifecycle_ok),
        _acceptance_gate("required_validation", "validity", True, validation_ok, validation_ok),
        _acceptance_gate(
            "world_model_or_solve_benefit",
            "benefit",
            "measured_live_panel_outcome",
            "transport_canary_only",
            False,
        ),
    ]
    readiness = bool(
        transport["arc_transport_ready_score"] == 1
        and panel_a == 1
        and panel_b == 1
        and lifecycle_ok
    )
    complete = bool(preconditions_ok and readiness and validation_ok)
    verdict = (
        "complete_null_transport_ready_panels_budget_feasible_no_efficacy_claim"
        if complete
        else "complete_disqualified_transport_or_validation_contract_failure"
    )
    verdict_class = "null" if complete else "disqualified"
    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    counts.update(
        {
            "model_loads_attempted": int(bool(runtime_receipt.get("model_load_started"))),
            "model_loads_completed": int(bool(runtime_receipt.get("model_load_completed"))),
            "model_loads_failed": int(
                bool(runtime_receipt.get("model_load_started"))
                and not bool(runtime_receipt.get("model_load_completed"))
            ),
            "generation_calls_attempted": transport["generation_calls_attempted"],
            "generation_calls_completed": transport["generation_calls_completed"],
            "generation_calls_failed": transport["generation_calls_failed"],
            "prompt_tokens": transport["prompt_tokens"],
            "completion_tokens": transport["completion_tokens"],
            "total_tokens": transport["total_tokens"],
        }
    )
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": verdict,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "positive_claim": False,
        "world_model_quality_claim": False,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "planned_inference_substrate_class": "model_bounded_generation",
        "inference_substrate_class": actual_substrate_class(transport),
        "inference_substrate": "owned_local_qwen_chat_transport",
        "MODEL_SPECS": [deepcopy(dict(context["model"]))],
        "model_specs": [deepcopy(dict(context["model"]))],
        "model_invoked": transport["generation_calls_attempted"] > 0,
        "invocation_counts": counts,
        "duration_s": float(duration_s),
        "generation_duration_s": float(generation_duration_s),
        "runtime_receipt": deepcopy(dict(runtime_receipt)),
        "raw_request_receipts": receipts,
        "future_request_constructions": [deepcopy(dict(row)) for row in future_requests],
        "historical_request_profile": deepcopy(dict(historical_profile)),
        "panel_forecasts": {key: deepcopy(dict(value)) for key, value in panel_forecasts.items()},
        "rows": rows,
        "independent_reduction": reduction,
        "learning_lifecycle": deepcopy(dict(learning_lifecycle)),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": False,
        "arc_transport_ready_score": transport["arc_transport_ready_score"],
        "panel_a_feasible_score": panel_a,
        "panel_b_feasible_score": panel_b,
        "solve_provenance": "canary_no_solve_claim",
        "effective_flags": {
            "no_think": True,
            "use_chat_template": True,
            "mtp": False,
            "kv_quantization": "q8_0",
            "n_ctx": 49_152,
            "requested_offload_layers": 999,
            "canary_max_new_tokens": CANARY_MAX_TOKENS,
            "future_max_new_tokens": FUTURE_MAX_TOKENS,
            "capture_max_new_tokens": FUTURE_MAX_TOKENS,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "requests_per_episode": 1,
        },
        "retired_literal_prior_verdict": {
            "literal": "Exp7570 full generation result",
            "replacement": "Exp7570 failed preflight before any model call",
            "scope": "transport classification only",
            "scientific_hypothesis_retired": False,
        },
    }
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any], *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, counters, rows, forecasts, lifecycle, and receipts."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list):
        errors.append("rows_invalid")
    else:
        try:
            reduced = independent_reduce_rows(rows)
        except (KeyError, TypeError, ValueError):
            errors.append("row_reduction_invalid")
        else:
            if reduced != value.get("independent_reduction"):
                errors.append("row_reduction_mismatch")
    receipts = value.get("raw_request_receipts")
    if not isinstance(receipts, list):
        errors.append("raw_request_receipts_invalid")
    elif value.get("verdict_class") != "blocked":
        transport = reduce_transport_receipts(
            receipts, generation_duration_s=float(value.get("generation_duration_s") or 0)
        )
        if transport["arc_transport_ready_score"] != value.get("arc_transport_ready_score"):
            errors.append("transport_score_mismatch")
        expected_substrate = actual_substrate_class(transport)
        if value.get("inference_substrate_class") != expected_substrate:
            errors.append("substrate_class_mismatch")
    specs = value.get("MODEL_SPECS")
    if not isinstance(specs, list) or not specs:
        errors.append("model_specs_missing")
    elif not isinstance(specs[0], Mapping) or specs[0].get("hf_id") != MODEL_ID:
        errors.append("model_specs_identity_mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in value):
        errors.append("field_principles_incomplete")
    if (
        value.get("positive_claim") is not False
        or value.get("solve_provenance") != "canary_no_solve_claim"
    ):
        errors.append("claim_scope_invalid")
    if require_validation and value.get("verdict_class") != "blocked":
        if not _validation_passed(value.get("validation_receipts") or []):
            errors.append("required_validation_failed")
    if value.get("verdict_class") == "blocked":
        summary = value.get("gate_check_summary")
        first = summary.get("first_failure") if isinstance(summary, Mapping) else None
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not isinstance(first, Mapping) or not required <= set(first):
            errors.append("blocked_gate_summary_invalid")
        if int((value.get("invocation_counts") or {}).get("generation_calls_attempted") or 0):
            errors.append("blocked_generation_count_nonzero")
    return list(dict.fromkeys(errors))


def _test_request_receipt(index: int) -> Json:
    prompt = NEUTRAL_PROMPTS[index]
    return {
        "request_id": f"canary-{index}",
        "prompt": prompt,
        "reply": "Acknowledged.",
        "request_bytes_sha256": canonical_hash({"prompt": prompt, "max_tokens": 64}),
        "response_bytes_sha256": canonical_hash({"reply": "Acknowledged."}),
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "reply_sha256": _sha256_bytes(b"Acknowledged."),
        "requested_max_tokens": CANARY_MAX_TOKENS,
        "actual_prompt_tokens": 8,
        "actual_completion_tokens": 3,
        "actual_total_tokens": 11,
        "finish_reason": "stop",
        "elapsed_s": 5.25,
        "request_started": True,
        "response_observed": True,
        "owned_runtime": True,
        "process_identity": {"pid": 101, "pid_start_ticks": 201},
        "runtime_identity": {"server_pid": 102, "server_pid_start_ticks": 202},
        "model_sha256": "sha256:" + "a" * 64,
        "censored": False,
        "provenance": "owned_bounded_transport",
    }


def _test_history() -> Json:
    durations = [7.0 + index / 10 for index in range(16)]
    return {
        "verdict_class": "null",
        "flagged_adversarial": False,
        "MODEL_SPECS": [{"hf_id": MODEL_ID}],
        "invocation_counts": {"generation_calls_completed": 16},
        "rows": [
            {
                "server_request_rows": [
                    {
                        "elapsed_s": duration,
                        "requested_max_tokens": 256,
                        "response_observed": True,
                    }
                ]
            }
            for duration in durations
        ],
    }


def build_test_artifact(tmp_path: Path) -> Json:
    """Build a compact valid candidate without subprocesses, CUDA, or network."""

    protocol_rows = [
        {
            "panel": panel,
            "game": game,
            "seed": seed,
            "arm": arm,
        }
        for panel, games in {"A": ("su15", "sp80", "ft09"), "B": ("sb26", "g50t", "dc22")}.items()
        for game in games
        for seed in (7582001, 7582002)
        for arm in ("current_verifier", "integrity_guard")
    ]
    protocol = {"schema": PROTOCOL_SCHEMA, "rows": protocol_rows}
    future = construct_future_requests(protocol)
    profile = historical_request_profile(_test_history(), source_hash="sha256:" + "1" * 64)
    forecasts = forecast_panels(future, profile, measured_load_s=30.0, measured_capture_s=2.0)
    check = gate_row(
        "test_preconditions",
        upstream="test",
        path=str(tmp_path),
        field="ready",
        expected=True,
        observed=True,
    )
    context = {
        "model": {
            "name": "Qwen3.8-27B",
            "hf_id": MODEL_ID,
            "model_path": str(tmp_path / "model.gguf"),
            "selection_role": "current_headline",
            "sha256": "sha256:" + "a" * 64,
        }
    }
    lifecycle = exercise_learning_lifecycle(tmp_path / "learner.json")
    return build_complete_artifact(
        checks=[check],
        context=context,
        raw_request_receipts=[_test_request_receipt(0), _test_request_receipt(1)],
        future_requests=future,
        historical_profile=profile,
        panel_forecasts=forecasts,
        learning_lifecycle=lifecycle,
        runtime_receipt={"model_load_started": True, "model_load_completed": True},
        validation_receipts=[],
        source_artifact_hashes={"test": "sha256:" + "b" * 64},
        duration_s=20.0,
        generation_duration_s=10.5,
        require_validation=False,
    )


def cold_replay(path: Path, *, require_validation: bool = True) -> Json:
    """Reload exact candidate bytes and independently reduce rows and learner state."""

    artifact = load_json(path)
    errors = validate_artifact(artifact, require_validation=require_validation)
    return {
        "passed": not errors,
        "errors": errors,
        "row_reduction": independent_reduce_rows(artifact.get("rows") or []),
        "learning_state_hash": (artifact.get("learning_lifecycle") or {}).get(
            "state_sha256_after_reload"
        ),
    }


def independent_reduce_artifact(path: Path) -> Json:
    """Expose a fresh-process reducer that trusts only persisted per-unit rows."""

    artifact = load_json(path)
    return independent_reduce_rows(artifact.get("rows") or [])


def affected_validation_manifest() -> Json:
    """Freeze the only files allowed into current scoped validation."""

    return {
        "schema": "carnot.exp7581.affected_validation_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "test_paths": [TEST_REL.as_posix()],
        "changed_modules": [MODULE_REL.as_posix()],
        "static_paths": [WRAPPER_REL.as_posix()],
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build serial scoped checks with one command-local coverage database."""

    commands = validation_scope.build_scoped_commands(
        root,
        [TEST_REL.as_posix()],
        [MODULE_REL.as_posix()],
        static_paths=[WRAPPER_REL.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=private_root / "coverage" / ".coverage",
    )
    coverage_file = str(private_root / "coverage" / ".coverage")
    transformed: list[validation_scope.CommandSpec] = []
    for command in commands:
        argv = tuple(item for item in command.argv if not item.startswith("--data-file="))
        if command.name in {"changed_module_coverage", "changed_module_coverage_report"}:
            argv = ("/usr/bin/env", f"COVERAGE_FILE={coverage_file}", *argv)
        transformed.append(
            validation_scope.CommandSpec(
                command.name,
                argv,
                command.scope,
                command.timeout_s,
            )
        )
    return transformed


def build_e2e_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Declare applicable ARC CPU E2E checks and the foreign-CWD real smoke."""

    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    targets = {
        "e2e_009": ("tests/python/test_arc_induction_state_persistence.py",),
        "e2e_010": ("tests/python/test_arc_tool_grammar_transport.py",),
        "e2e_011": ("tests/python/test_arc_decision_telemetry.py",),
        "e2e_012": (
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
            "tests/python/test_arc_decision_telemetry.py",
        ),
        "e2e_013": (
            "tests/python/test_arc_decision_telemetry.py",
            "tests/python/test_experiment_7491_e6_timed_live_profile.py",
            "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
            "tests/python/test_semif_arc_readout_eval.py",
        ),
    }
    commands = [
        validation_scope.CommandSpec(
            name,
            (
                pytest,
                *common,
                f"--basetemp={private_root / name}",
                *paths,
                "-q",
            ),
            f"{name.replace('_', '-').upper()} ARC CPU contract",
            900.0,
        )
        for name, paths in targets.items()
    ]
    commands.append(
        validation_scope.CommandSpec(
            "foreign_cwd_llm_off_e3_smoke",
            (
                "/usr/bin/env",
                "-C",
                str(private_root / "foreign-cwd"),
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                str(root / "scripts/arc_loop_solve.py"),
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private_root / "foreign-cwd" / "r11l-smoke.json"),
            ),
            "private LLM-off real E3 episode from a foreign cwd",
            300.0,
        )
    )
    return commands


def build_terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Declare entrypoint replay, reduction, and both strict exact-byte readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_REL)
    common = ("--root", str(root), "--date", RUN_DATE)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (python, "-u", wrapper, *common, "--validate", str(candidate)),
            "declared read-only entrypoint",
            300.0,
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, *common, "--cold-replay", str(candidate)),
            "fresh-process cold replay",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, *common, "--independent-reduce", str(candidate)),
            "independent per-unit row reduction",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", str(root / "scripts/adversarial_verify.py"), str(candidate)),
            "exact terminal candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                str(root / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ),
            "exact terminal candidate",
            300.0,
        ),
    ]


def prepare_command_parent(command: validation_scope.CommandSpec) -> None:
    """Create pytest, coverage, output, and foreign-CWD parents before each child."""

    for index, argument in enumerate(command.argv):
        if argument.startswith("--basetemp=") or argument.startswith("COVERAGE_FILE="):
            Path(argument.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        elif argument == "-C" and index + 1 < len(command.argv):
            Path(command.argv[index + 1]).mkdir(parents=True, exist_ok=True)
        elif argument == "--output" and index + 1 < len(command.argv):
            Path(command.argv[index + 1]).parent.mkdir(parents=True, exist_ok=True)


def run_prepared_commands(
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    *,
    log_dir: Path,
) -> list[Json]:  # pragma: no cover - bounded subprocess integration.
    """Run children one at a time after creating that command's private parents."""

    receipts: list[Json] = []
    for command in commands:
        prepare_command_parent(command)
        receipts.extend(
            validation_scope.run_commands(
                root,
                [command],
                log_dir=log_dir / command.name,
                heartbeat_s=60.0,
            )
        )
    return receipts


def _parse_csv_line(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def gpu_inventory() -> list[Json]:  # pragma: no cover - host nvidia-smi boundary.
    """Read GPU capacity and process owners without changing either."""

    gpu_command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    process_command = (
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        gpu_result = subprocess.run(
            gpu_command, capture_output=True, text=True, timeout=10.0, check=False
        )
        process_result = subprocess.run(
            process_command, capture_output=True, text=True, timeout=10.0, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if gpu_result.returncode != 0:
        return []
    processes: dict[str, list[Json]] = {}
    if process_result.returncode == 0:
        for line in process_result.stdout.splitlines():
            parts = _parse_csv_line(line)
            if len(parts) != 4:
                continue
            try:
                row = {
                    "pid": int(parts[1]),
                    "name": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            except ValueError:
                continue
            processes.setdefault(parts[0], []).append(row)
    inventory: list[Json] = []
    for line in gpu_result.stdout.splitlines():
        parts = _parse_csv_line(line)
        if len(parts) != 7:
            continue
        try:
            row = {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "memory_free_mb": int(parts[5]),
                "utilization_pct": int(parts[6]),
                "processes": processes.get(parts[1], []),
            }
        except ValueError:
            continue
        inventory.append(row)
    return inventory


def _admissible_gpu(inventory: Sequence[Mapping[str, Any]]) -> Json | None:
    candidates = [
        deepcopy(dict(row))
        for row in inventory
        if int(row.get("memory_used_mb") or 0) <= GPU_IDLE_MAX_USED_MB
        and int(row.get("memory_free_mb") or 0) >= GPU_REQUIRED_FREE_MB
        and not row.get("processes")
    ]
    return min(candidates, key=lambda row: int(row["index"])) if candidates else None


def wait_for_cuda_capacity(
    *,
    inventory_fn: Callable[[], Sequence[Mapping[str, Any]]] = gpu_inventory,
    max_wait_s: float = CAPACITY_WAIT_S,
    poll_s: float = 10.0,
    progress_fn: Callable[[Json], None] | None = None,
) -> tuple[Json | None, Json]:
    """Wait observably for capacity and never signal an existing process."""

    started = time.monotonic()
    attempts = 0
    last_inventory: list[Json] = []
    while True:
        attempts += 1
        last_inventory = [deepcopy(dict(row)) for row in inventory_fn()]
        selected = _admissible_gpu(last_inventory)
        elapsed = time.monotonic() - started
        report = {
            "event": "cuda_capacity_observation",
            "attempt": attempts,
            "elapsed_s": elapsed,
            "selected_uuid": selected.get("uuid") if selected else None,
            "foreign_process_count": sum(len(row.get("processes") or []) for row in last_inventory),
        }
        if progress_fn is not None:
            progress_fn(report)
        if selected is not None:
            return selected, {
                "wait_limit_s": float(max_wait_s),
                "wait_elapsed_s": elapsed,
                "attempts": attempts,
                "selected": selected,
                "foreign_processes": [],
                "signals_sent": [],
            }
        if elapsed >= float(max_wait_s):
            foreign = [
                {"gpu_uuid": row.get("uuid"), **deepcopy(dict(process))}
                for row in last_inventory
                for process in row.get("processes") or []
            ]
            return None, {
                "wait_limit_s": float(max_wait_s),
                "wait_elapsed_s": elapsed,
                "attempts": attempts,
                "last_inventory": last_inventory,
                "foreign_processes": foreign,
                "signals_sent": [],
            }
        time.sleep(max(0.0, min(float(poll_s), float(max_wait_s) - elapsed)))


def process_start_tick(pid: int | None) -> int | None:
    """Bind a Linux PID to its kernel start tick so PID reuse cannot pass."""

    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _call_with_heartbeats(
    operation: Callable[[], Any], *, started: float, phase: str
) -> Any:  # pragma: no cover - silence guard for host I/O.
    """Run one local operation while truthfully reporting that it remains pending."""

    result: list[Any] = []
    error: list[BaseException] = []
    done = threading.Event()

    def target() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # noqa: BLE001 - re-raised in the owner thread.
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=target, name=f"exp7581-{phase}", daemon=True)
    thread.start()
    while not done.wait(55.0):
        progress(started, phase, "pending_operation")
    thread.join(timeout=1.0)
    if error:
        raise error[0]
    return result[0]


def source_hashes(root: Path, extra: Sequence[Path] = ()) -> dict[str, str]:
    """Bind current sources and exact upstream files without reading outcomes twice."""

    paths = (
        EXP7574_REL,
        EXP7580_REL,
        PROTOCOL_REL,
        HISTORY_REL,
        SPEC_REL,
        MODULE_REL,
        WRAPPER_REL,
        TEST_REL,
        *extra,
    )
    return {
        str(path): sha256_file(path if path.is_absolute() else root / path)
        for path in paths
        if (path if path.is_absolute() else root / path).is_file()
    }


def _owned_vram_mb(pid: int | None) -> int | None:  # pragma: no cover - host boundary.
    if not isinstance(pid, int):
        return None
    command = (
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10.0, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        parts = _parse_csv_line(line)
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _observed_offload_layers(log_path: Path | None) -> Json:  # pragma: no cover - live log.
    if log_path is None or not log_path.is_file():
        return {"loaded_layers": None, "total_layers": None, "evidence": None}
    with log_path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2_000_000))
        tail = stream.read().decode("utf-8", "replace")
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers", tail, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"offloading\s+(\d+)\s+repeating layers", tail, flags=re.IGNORECASE)
        if matches:
            loaded = int(matches[-1])
            return {"loaded_layers": loaded, "total_layers": None, "evidence": "stderr"}
        return {"loaded_layers": None, "total_layers": None, "evidence": "stderr_no_match"}
    loaded, total = matches[-1]
    return {"loaded_layers": int(loaded), "total_layers": int(total), "evidence": "stderr"}


def _capture_pairs(event_path: Path) -> list[Json]:  # pragma: no cover - live byte capture.
    from carnot.experiment_7431_v651_arc_live_sentinel import _read_jsonl

    pairs: list[Json] = []
    for event in _read_jsonl(event_path):
        if event.get("event") != "server_response":
            continue
        request_path = Path(str(event.get("request_path")))
        response_path = Path(str(event.get("response_path")))
        if request_path.is_file() and response_path.is_file():
            pairs.append(
                {
                    "request_id": str(event.get("episode_id")),
                    "request_bytes": request_path.read_bytes(),
                    "response_bytes": response_path.read_bytes(),
                    "elapsed_s": float(event.get("elapsed_s") or 0),
                    "request_path": str(request_path),
                    "response_path": str(response_path),
                }
            )
    return pairs


def run_owned_transport(
    *,
    root: Path,
    model_path: Path,
    model_sha256: str,
    selected_gpu: Mapping[str, Any],
    lease: Any,
    started: float,
) -> tuple[list[Json], Json, float, float]:  # pragma: no cover - owned CUDA integration.
    """Load one owned server, make exactly two calls, capture bytes, and unload it."""

    from carnot import experiment_7431_v651_arc_live_sentinel as live_support
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_inference_boundary import BOUNDARY_LEDGER_ENV

    raw_dir = root / RAW_REL / "transport"
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / "runtime_events.jsonl"
    boundary_path = raw_dir / "invocation_events.jsonl"
    event_path.unlink(missing_ok=True)
    boundary_path.unlink(missing_ok=True)
    capture = live_support.DurableRequestCapture(
        raw_dir, event_path, max_new_tokens=CANARY_MAX_TOKENS
    )
    port = live_support._free_port()
    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.8-27B",
        model_path=str(model_path),
        port=port,
        mtp=False,
        kv_quant="q8_0",
        use_chat_template=True,
        n_gpu_layers=999,
        n_ctx=49_152,
        max_tokens=CANARY_MAX_TOKENS,
        timeout=int(REQUEST_TIMEOUT_S),
        tries=1,
    )
    proposer.model_repository = MODEL_ID
    proposer.requested_model_path = str(model_path)
    previous_env = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "CARNOT_ARC_GENERATOR_CUDA_GPU",
            "CARNOT_ARC_INDUCE_THINK",
            "CARNOT_ARC_INDUCE_THINKING_BUDGET",
            BOUNDARY_LEDGER_ENV,
            "CARNOT_ARC_SERVER_LOG_DIR",
        )
    }
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_gpu["index"]),
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(selected_gpu["index"]),
            "CARNOT_ARC_INDUCE_THINK": "0",
            BOUNDARY_LEDGER_ENV: str(boundary_path),
            "CARNOT_ARC_SERVER_LOG_DIR": str(raw_dir),
        }
    )
    os.environ.pop("CARNOT_ARC_INDUCE_THINKING_BUDGET", None)
    owner = lease.owner_receipt()
    runtime: Json = {
        "model_load_started": False,
        "model_load_completed": False,
        "gpu_uuid": selected_gpu["uuid"],
        "gpu_index": selected_gpu["index"],
        "lease_owner": owner,
        "lease_release": None,
        "signals_sent_to_foreign_processes": [],
    }
    load_started = time.monotonic()
    generation_duration = 0.0
    capture_cost = 0.0
    try:
        capture.install()
        lease.transition("admitted")
        lease.transition("loading")
        runtime["model_load_started"] = True
        progress(started, "model_load", "before", gpu_uuid=selected_gpu["uuid"])
        healthy = proposer._ensure_server()
        server_pid = getattr(getattr(proposer, "_proc", None), "pid", None)
        runtime.update(
            {
                "server_pid": server_pid,
                "server_pid_start_ticks": process_start_tick(server_pid),
                "server_command": list(getattr(proposer, "last_launch_argv", ()) or ()),
                "owned_vram_mb": _owned_vram_mb(server_pid),
                "model_load_s": time.monotonic() - load_started,
            }
        )
        runtime["model_load_completed"] = bool(healthy)
        runtime["offload_layers"] = _observed_offload_layers(
            Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        )
        progress(
            started,
            "model_load",
            "after",
            healthy=healthy,
            server_pid=server_pid,
            owned_vram_mb=runtime["owned_vram_mb"],
        )
        if not healthy or runtime["server_pid_start_ticks"] is None:
            raise RuntimeError("owned_model_server_not_authenticated")
        lease.transition("resident", vram_mb=int(runtime["owned_vram_mb"] or 0))
        lease.transition("inferencing")
        props = proposer.server_props()
        runtime["server_props_sha256"] = canonical_hash(props)
        runtime["observed_model_path"] = proposer.observed_model_path()
        runtime["chat_template"] = {
            "endpoint": "/v1/chat/completions",
            "use_chat_template": True,
            "server_props_chat_template_sha256": canonical_hash(
                props.get("chat_template") if isinstance(props, Mapping) else None
            ),
        }
        generation_started = time.monotonic()
        for index, prompt in enumerate(NEUTRAL_PROMPTS):
            capture.begin_episode(f"canary-{index}")
            progress(started, "generation", "before", unit=f"{index + 1}/2")
            proposer._chat_complete_request(
                prompt,
                max_tokens=CANARY_MAX_TOKENS,
                temperature=0.2,
                stop=None,
                attempt=0,
            )
            progress(started, "generation", "after", unit=f"{index + 1}/2")
        generation_duration = time.monotonic() - generation_started
        process_identity = {
            "pid": os.getpid(),
            "pid_start_ticks": process_start_tick(os.getpid()),
            "worktree": str(root),
        }
        pairs = _capture_pairs(event_path)
        receipts = [
            request_receipt_from_bytes(
                request_id=str(pair["request_id"]),
                request_bytes=pair["request_bytes"],
                response_bytes=pair["response_bytes"],
                elapsed_s=float(pair["elapsed_s"]),
                process_identity=process_identity,
                runtime_identity=runtime,
                model_sha256=model_sha256,
            )
            for pair in pairs
        ]
        capture_cost = max(
            0.0,
            generation_duration - sum(float(row["elapsed_s"]) for row in receipts),
        )
        runtime["tokenizer"] = {
            "kind": "embedded_gguf",
            "observed_prompt_tokens": [row["actual_prompt_tokens"] for row in receipts],
            "passed": all(int(row["actual_prompt_tokens"]) > 0 for row in receipts),
        }
        return receipts, runtime, generation_duration, capture_cost
    finally:
        capture.restore()
        progress(started, "model_unload", "before")
        proposer.stop()
        progress(started, "model_unload", "after")
        phase = str(lease.document.get("phase"))
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete" if generation_duration > 0 else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _all_commands_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require a real zero exit and no timeout for every supplied command."""

    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is not True
        for row in receipts
    )


def _append_failed_command_gate(
    checks: list[Json], receipts: Sequence[Mapping[str, Any]], *, check: str
) -> None:
    """Turn exact failed command receipts into one auditable validity operand."""

    failed = [
        {
            "name": row.get("name"),
            "exit_code": row.get("exit_code"),
            "timed_out": row.get("timed_out"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
        if row.get("passed") is not True
        or row.get("exit_code") != 0
        or row.get("timed_out") is True
    ]
    checks.append(
        gate_row(
            check,
            upstream="current_exp7581_scoped_validation",
            path=RAW_REL.as_posix(),
            field="command_receipts",
            expected="all_exit_zero",
            observed=failed,
            passed=not failed,
            category="validity",
        )
    )


def _blocked_forecasts(reason: str) -> dict[str, Json]:
    """Name why no measured load/capture forecast can be authenticated."""

    return {
        panel: {
            "panel": panel,
            "request_count": 12,
            "episode_count": 12,
            "max_tokens": FUTURE_MAX_TOKENS,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "forecast_status": f"blocked_{reason}",
            "panel_feasible_score": 0,
        }
        for panel in ("A", "B")
    }


def _enrich_blocked_artifact(
    artifact: Json,
    *,
    future_requests: Sequence[Mapping[str, Any]],
    historical_profile: Mapping[str, Any] | None,
    panel_forecasts: Mapping[str, Mapping[str, Any]],
    learning_lifecycle: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any] | None,
) -> Json:
    """Attach completed independent work without inventing dependent evidence."""

    artifact["future_request_constructions"] = [deepcopy(dict(row)) for row in future_requests]
    artifact["historical_request_profile"] = deepcopy(dict(historical_profile or {}))
    artifact["panel_forecasts"] = {
        panel: deepcopy(dict(row)) for panel, row in panel_forecasts.items()
    }
    artifact["learning_lifecycle"] = deepcopy(
        dict(learning_lifecycle or {"passed": False, "operations": []})
    )
    artifact["runtime_receipt"] = deepcopy(dict(runtime_receipt or {}))
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _publish_terminal_candidate(
    *,
    root: Path,
    artifact: Json,
    candidate: Path,
    final_path: Path,
    prior_receipts: Sequence[Mapping[str, Any]],
    started: float,
) -> Json:  # pragma: no cover - bounded subprocess integration.
    """Run exact-byte terminal readers, then atomically publish their outcome."""

    candidate.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(candidate, artifact)
    progress(started, "terminal_validation", "before", candidate=str(candidate))
    terminal = run_prepared_commands(
        root,
        build_terminal_commands(root, candidate),
        log_dir=root / RAW_REL / "validation" / "terminal",
    )
    progress(started, "terminal_validation", "after", passed=_all_commands_passed(terminal))
    artifact["validation_receipts"] = [
        *[deepcopy(dict(row)) for row in prior_receipts],
        *[deepcopy(dict(row)) for row in terminal],
    ]
    adversarial = next((row for row in terminal if row.get("name") == "adversarial_verify"), {})
    artifact["flagged_adversarial"] = adversarial.get("passed") is not True
    terminal_passed = _all_commands_passed(terminal)
    if not terminal_passed and artifact.get("verdict_class") != "blocked":
        suffix = "adversarial_failure" if artifact["flagged_adversarial"] else "reader_failure"
        artifact["status"] = f"complete_disqualified_terminal_{suffix}"
        artifact["honest_verdict"] = artifact["status"]
        artifact["verdict_class"] = "disqualified"
        artifact["arc_transport_ready_score"] = 0
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, require_validation=artifact.get("verdict_class") == "null")
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{','.join(errors)}")
    atomic_json(final_path, artifact)
    progress(started, "publish", "after", path=str(final_path))
    return artifact


def _make_blocked(
    *,
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    reason: str,
    receipts: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    future_requests: Sequence[Mapping[str, Any]],
    historical_profile: Mapping[str, Any] | None,
    learning_lifecycle: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any] | None,
    started: float,
) -> Json:  # pragma: no cover - experiment orchestration.
    """Build, independently read, and publish one complete blocked result."""

    artifact = build_blocked_artifact(
        checks=checks,
        context=context,
        duration_s=time.monotonic() - started,
        reason=reason,
        validation_receipts=receipts,
        source_artifact_hashes=hashes,
    )
    _enrich_blocked_artifact(
        artifact,
        future_requests=future_requests,
        historical_profile=historical_profile,
        panel_forecasts=_blocked_forecasts("current_load_capture_unavailable"),
        learning_lifecycle=learning_lifecycle,
        runtime_receipt=runtime_receipt,
    )
    candidate = root / RAW_REL / "terminal_candidate.json"
    return _publish_terminal_candidate(
        root=root,
        artifact=artifact,
        candidate=candidate,
        final_path=root / RESULT_REL,
        prior_receipts=receipts,
        started=started,
    )


def run_experiment(repo_root: Path, run_date: str) -> Json:  # pragma: no cover
    """Execute the bounded canary without borrowing or stopping foreign work."""

    started = time.monotonic()
    root = repo_root.resolve()
    if root != REPO_ROOT.resolve():
        raise ValueError(f"root_mismatch:{root}")
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    progress(started, "startup", "begin", root=str(root), run_date=run_date)
    raw_root = root / RAW_REL
    private_root = (Path(tempfile.gettempdir()) / f"carnot-exp7581-v662-{os.getpid()}").resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    manifest = affected_validation_manifest()
    atomic_json(raw_root / "affected_validation_manifest.json", manifest)

    progress(started, "preconditions", "before")
    checks, context = collect_preconditions(root)
    progress(
        started,
        "preconditions",
        "after",
        passed=not failed_checks(checks),
        failed=failed_checks(checks),
    )
    hashes = source_hashes(root)
    future_requests: list[Json] = []
    historical_profile: Json | None = None
    if not failed_checks(checks):
        future_requests = construct_future_requests(context["protocol"])
        historical_profile = historical_request_profile(
            context["history"], source_hash=hashes[HISTORY_REL.as_posix()]
        )
    if failed_checks(checks):
        return _make_blocked(
            root=root,
            checks=checks,
            context=context,
            reason=failed_checks(checks)[0],
            receipts=(),
            hashes=hashes,
            future_requests=future_requests,
            historical_profile=historical_profile,
            learning_lifecycle=None,
            runtime_receipt=None,
            started=started,
        )

    progress(started, "scoped_validation", "before")
    affected = run_prepared_commands(
        root,
        build_validation_commands(root, private_root),
        log_dir=raw_root / "validation" / "affected",
    )
    progress(started, "scoped_validation", "after", passed=_all_commands_passed(affected))
    _append_failed_command_gate(checks, affected, check="affected_validation")
    if not _all_commands_passed(affected):
        artifact = build_blocked_artifact(
            checks=checks,
            context=context,
            duration_s=time.monotonic() - started,
            reason="affected_validation",
            validation_receipts=affected,
            source_artifact_hashes=hashes,
        )
        artifact["status"] = "complete_disqualified_affected_validation_failure"
        artifact["honest_verdict"] = artifact["status"]
        artifact["verdict_class"] = "disqualified"
        _enrich_blocked_artifact(
            artifact,
            future_requests=future_requests,
            historical_profile=historical_profile,
            panel_forecasts=_blocked_forecasts("validation_failure"),
            learning_lifecycle=None,
            runtime_receipt=None,
        )
        return _publish_terminal_candidate(
            root=root,
            artifact=artifact,
            candidate=raw_root / "terminal_candidate.json",
            final_path=root / RESULT_REL,
            prior_receipts=affected,
            started=started,
        )

    progress(started, "arc_e2e", "before")
    e2e = run_prepared_commands(
        root,
        build_e2e_commands(root, private_root),
        log_dir=raw_root / "validation" / "e2e",
    )
    progress(started, "arc_e2e", "after", passed=_all_commands_passed(e2e))
    all_preterminal = [*affected, *e2e]
    _append_failed_command_gate(checks, e2e, check="arc_e2e")
    if not _all_commands_passed(e2e):
        return _make_blocked(
            root=root,
            checks=checks,
            context=context,
            reason="arc_e2e",
            receipts=all_preterminal,
            hashes=hashes,
            future_requests=future_requests,
            historical_profile=historical_profile,
            learning_lifecycle=None,
            runtime_receipt=None,
            started=started,
        )

    progress(started, "learning_lifecycle", "before")
    learning = exercise_learning_lifecycle(raw_root / "learner_state.json")
    progress(started, "learning_lifecycle", "after", passed=learning["passed"])
    model_path = Path(str(context["model"]["model_path"])).resolve()
    progress(started, "model_hash", "before", path=str(model_path))
    model_sha256 = _call_with_heartbeats(
        lambda: sha256_file(model_path), started=started, phase="model_hash"
    )
    progress(started, "model_hash", "after", sha256=model_sha256)
    context["model"]["sha256"] = model_sha256
    context["model"]["bytes"] = model_path.stat().st_size
    hashes[str(model_path)] = model_sha256

    progress(started, "cuda_capacity", "before", wait_limit_s=CAPACITY_WAIT_S)
    selected, capacity = wait_for_cuda_capacity(
        progress_fn=lambda row: progress(started, "cuda_capacity", "observation", **row)
    )
    progress(
        started,
        "cuda_capacity",
        "after",
        selected_uuid=selected.get("uuid") if selected else None,
    )
    if selected is None:
        checks.append(
            gate_row(
                "exclusive_cuda_capacity",
                upstream="nvidia-smi_current_host",
                path="/proc/driver/nvidia",
                field="exclusive_gpu_with_required_headroom",
                expected={
                    "memory_used_mb_lte": GPU_IDLE_MAX_USED_MB,
                    "memory_free_mb_gte": GPU_REQUIRED_FREE_MB,
                    "foreign_processes": 0,
                },
                observed=capacity,
                passed=False,
                category="readiness",
            )
        )
        return _make_blocked(
            root=root,
            checks=checks,
            context=context,
            reason="exclusive_cuda_capacity",
            receipts=all_preterminal,
            hashes=hashes,
            future_requests=future_requests,
            historical_profile=historical_profile,
            learning_lifecycle=learning,
            runtime_receipt={"cuda_capacity": capacity},
            started=started,
        )

    from carnot.gpu_lease_phase_journal import GpuLease

    progress(started, "gpu_lease", "before", gpu_uuid=selected["uuid"])
    try:
        lease = GpuLease.acquire(
            runtime_dir=raw_root / "gpu_leases",
            task_id=EXPERIMENT_ID,
            device_uuid=str(selected["uuid"]),
            expected_model=str(model_path),
            vram_before_mb=int(selected["memory_used_mb"]),
            ttl_s=4800.0,
        )
    except Exception as exc:  # noqa: BLE001 - exact lease failure becomes evidence.
        lease_error = f"{type(exc).__name__}:{exc}"
        progress(started, "gpu_lease", "after", acquired=False, error=lease_error)
        checks.append(
            gate_row(
                "exclusive_gpu_lease",
                upstream="carnot.gpu_lease_phase_journal.GpuLease",
                path=str(raw_root / "gpu_leases"),
                field="exclusive_owner",
                expected={"gpu_uuid": selected["uuid"], "task_id": EXPERIMENT_ID},
                observed={"error": lease_error},
                passed=False,
                category="readiness",
            )
        )
        return _make_blocked(
            root=root,
            checks=checks,
            context=context,
            reason="exclusive_gpu_lease",
            receipts=all_preterminal,
            hashes=hashes,
            future_requests=future_requests,
            historical_profile=historical_profile,
            learning_lifecycle=learning,
            runtime_receipt={"cuda_capacity": capacity, "lease_error": lease_error},
            started=started,
        )
    progress(started, "gpu_lease", "after", acquired=True, lease_id=lease.lease_id)

    try:
        request_receipts, runtime, generation_duration, capture_cost = run_owned_transport(
            root=root,
            model_path=model_path,
            model_sha256=model_sha256,
            selected_gpu=selected,
            lease=lease,
            started=started,
        )
    except Exception as exc:  # noqa: BLE001 - persist exact owned transport failure.
        transport_error = f"{type(exc).__name__}:{exc}"
        progress(started, "owned_transport", "after", passed=False, error=transport_error)
        checks.append(
            gate_row(
                "owned_transport_completed",
                upstream="LocalGGUFProposer._chat_complete_request",
                path=str(raw_root / "transport"),
                field="two_owned_64_token_requests",
                expected=2,
                observed={"completed": 0, "error": transport_error},
                passed=False,
                category="readiness",
            )
        )
        return _make_blocked(
            root=root,
            checks=checks,
            context=context,
            reason="owned_transport_completed",
            receipts=all_preterminal,
            hashes=hashes,
            future_requests=future_requests,
            historical_profile=historical_profile,
            learning_lifecycle=learning,
            runtime_receipt={
                "cuda_capacity": capacity,
                "transport_error": transport_error,
                "signals_sent_to_foreign_processes": [],
            },
            started=started,
        )
    progress(
        started,
        "owned_transport",
        "after",
        completed=len(request_receipts),
        generation_duration_s=generation_duration,
    )
    runtime["cuda_capacity"] = capacity
    forecasts = forecast_panels(
        future_requests,
        historical_profile,
        measured_load_s=float(runtime.get("model_load_elapsed_s") or 0.0),
        measured_capture_s=capture_cost,
    )
    candidate = build_complete_artifact(
        checks=checks,
        context=context,
        raw_request_receipts=request_receipts,
        future_requests=future_requests,
        historical_profile=historical_profile,
        panel_forecasts=forecasts,
        learning_lifecycle=learning,
        runtime_receipt=runtime,
        validation_receipts=all_preterminal,
        source_artifact_hashes=hashes,
        duration_s=time.monotonic() - started,
        generation_duration_s=generation_duration,
        require_validation=False,
    )
    return _publish_terminal_candidate(
        root=root,
        artifact=candidate,
        candidate=raw_root / "terminal_candidate.json",
        final_path=root / RESULT_REL,
        prior_receipts=all_preterminal,
        started=started,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit repository, date, and read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--date", choices=(RUN_DATE,), required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate", type=Path)
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the experiment or one fresh-process exact-byte reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.validate is not None:
        errors = validate_artifact(load_json(args.validate), require_validation=False)
        print(json.dumps({"passed": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.cold_replay is not None:
        replay = cold_replay(args.cold_replay, require_validation=False)
        print(json.dumps(replay, sort_keys=True), flush=True)
        return int(not replay["passed"])
    if args.independent_reduce is not None:
        print(
            json.dumps(independent_reduce_artifact(args.independent_reduce), sort_keys=True),
            flush=True,
        )
        return 0
    result = run_experiment(root, args.date)
    print(
        json.dumps(
            {
                "artifact": str(root / RESULT_REL),
                "honest_verdict": result["honest_verdict"],
                "verdict_class": result["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
