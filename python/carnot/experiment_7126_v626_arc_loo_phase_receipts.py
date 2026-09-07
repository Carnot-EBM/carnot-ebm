"""Reconstruct Exp7123 phase evidence without running a model or ARC game.

The prior artifact proves that two early bookkeeping actions happened, but it
does not contain clocks for those actions or receipts for later phases. This
module keeps that uncertainty visible and defines the stricter receipt a future
paired cell must write while it runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
RUN_DATE = "20260907"
RANDOM_SEED = 7_126_202_609_07
RESULT_PATH = Path("results/experiment_7126_v626_arc_loo_phase_receipts.json")
EXP7123_PATH = Path("results/experiment_7123_v625_arc_loo_shard_a.json")
EXP7113_PATH = Path("results/experiment_7113_v624_arc_generation_liveness.json")
EXP7099_PATH = Path("results/experiment_7099_v623_adapter_withheld_preflight.json")
CONDUCTOR_PATH = Path("ops/conductor-log.md")
MODULE_PATH = Path("python/carnot/experiment_7126_v626_arc_loo_phase_receipts.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_7126_v626_arc_loo_phase_receipts.py")
TEST_PATH = Path("tests/python/test_experiment_7126_v626_arc_loo_phase_receipts.py")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts: offline ARC phase forensics"

PHASES = (
    "artifact_initialization",
    "registry_freeze",
    "setup",
    "model_resolution",
    "lease",
    "server_start",
    "withheld_arm",
    "control_arm",
    "validation",
    "finalization",
)
PHASE_RECEIPT_FIELDS = (
    "phase",
    "monotonic_start_ns",
    "monotonic_end_ns",
    "wall_clock_start",
    "wall_clock_end",
    "deadline",
    "subprocess_pid",
    "exit_state",
    "timeout_state",
    "evidence_hash",
)
ATTACK_IDS = (
    "missing_timestamps",
    "contradictory_clocks",
    "absent_start_receipt",
    "absent_end_receipt",
    "negative_interval",
    "duplicate_events",
    "fabricated_phase_completion",
)
CAP_FIELDS = (
    "setup_cap_s",
    "withheld_arm_cap_s",
    "control_arm_cap_s",
    "finalization_cap_s",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "source_timeline_rows",
    "phase_timing_rows",
    "clock_domain_rows",
    "missing_receipt_rows",
    "contradiction_rows",
    "first_absent_start_receipt",
    "observed_stall_interval",
    "phase_receipt_schema",
    "phase_budget_rows",
    "setup_cap_s",
    "withheld_arm_cap_s",
    "control_arm_cap_s",
    "finalization_cap_s",
    "value_measurement_run",
    "solve_claim_made",
    "arc_phase_receipt_contract_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Each required field explains the evidence boundary it protects.",
    "preconditions_checked": "Readability and destination checks distinguish missing inputs from forensic findings.",
    "run_date": "The fixed date binds this diagnosis to the requested execution window.",
    "inference_substrate": "The aggregation label prevents offline forensics from being read as model execution.",
    "inference_substrate_class": "The closed class distinguishes a completed aggregation from a no-run block.",
    "execution_venue": "The host venue identifies where filesystem and process evidence was inspected.",
    "duration_s": "Measured diagnostic wall time describes this aggregation, not Exp7123 runtime.",
    "source_artifact_hashes": "Byte hashes bind every readable source used by the reconstruction.",
    "rows": "One canonical ledger permits independent consistency checks across every projection.",
    "source_timeline_rows": "Source-local events preserve paths, clocks, precision, and absent timestamps.",
    "phase_timing_rows": "One row per required phase prevents missing work from disappearing in prose.",
    "clock_domain_rows": "Clock declarations prevent wall, monotonic, and untimed evidence from being merged.",
    "missing_receipt_rows": "Explicit gaps prevent absent receipts from becoming estimated timestamps.",
    "contradiction_rows": "Conflicting claims remain visible instead of being silently reconciled.",
    "first_absent_start_receipt": "The first absent start marker localizes where auditable progress stopped.",
    "observed_stall_interval": "A source-derived interval separates silence from a claimed phase duration.",
    "phase_receipt_schema": "The reusable schema tells a future run what it must persist at each boundary.",
    "phase_budget_rows": "Fixed future caps reserve time for both arms and terminal validation.",
    "setup_cap_s": "The five-minute setup cap prevents preparation from consuming either arm budget.",
    "withheld_arm_cap_s": "The twenty-five-minute withheld cap bounds the primary subprocess.",
    "control_arm_cap_s": "The twenty-five-minute control cap keeps the pair comparable.",
    "finalization_cap_s": "The five-minute finalization cap protects validation and artifact publication.",
    "value_measurement_run": "False states that this diagnostic did not measure ARC value.",
    "solve_claim_made": "False prevents forensic evidence from becoming level or solve credit.",
    "arc_phase_receipt_contract_ready_score": "One requires phase coverage, bounded budgets, attack rejection, and recomputable rows.",
    "random_seed": "A fixed identifier makes synthetic attack fixtures reproducible.",
    "reproducibility_checksum": "A canonical checksum detects changes to the complete artifact.",
    "gate_check_summary": "Exact expected and observed values explain a blocked reconstruction.",
    "verifier_is_oracle": "False keeps receipt validation separate from ARC correctness.",
    "verdict_class": "A closed class lets downstream readers route the diagnostic consistently.",
    "honest_verdict": "The terminal prefix states whether the forensic contract is ready or blocked.",
}

NAMED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/status.md"),
    Path("ops/known-issues.md"),
    CONDUCTOR_PATH,
    Path("ops/arc_solve_registry.yaml"),
    EXP7099_PATH,
    EXP7113_PATH,
    EXP7123_PATH,
    Path("python/carnot/experiment_7099_v623_adapter_withheld_preflight.py"),
    Path("python/carnot/experiment_7113_v624_arc_generation_liveness.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_live_runner_capability_lease.py"),
    Path("scripts/adversarial_verify.py"),
    Path("openspec/capabilities/arc-agi/spec.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    MODULE_PATH,
    SCRIPT_PATH,
    TEST_PATH,
)
ORPHAN_TEST_BYTECODE = Path(
    "tests/python/__pycache__/test_experiment_7123_v625_arc_loo_shard_a.cpython-312-pytest-9.0.3.pyc"
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable bytes so hashes do not depend on dictionary insertion order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Hash one JSON value after canonical serialization."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str | None:
    """Hash exact source bytes, returning null when the source is unavailable."""

    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: JsonDict) -> str:
    """Hash an artifact after blanking its self-referential checksum field."""

    payload = deepcopy(artifact)
    payload["reproducibility_checksum"] = ""
    return sha256_json(payload)


def _parse_wall(value: Any) -> datetime | None:
    """Parse one timezone-aware wall clock; naive clocks cannot be compared safely."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def make_phase_receipt(**values: Any) -> JsonDict:
    """Build the deliberately thin receipt without inventing optional defaults."""

    return {field: values.get(field) for field in PHASE_RECEIPT_FIELDS}


def _append_once(errors: list[str], reason: str) -> None:
    if reason not in errors:
        errors.append(reason)


def validate_phase_receipts(rows: list[JsonDict]) -> list[str]:
    """Reject phase receipts whose boundaries cannot be independently audited."""

    errors: list[str] = []
    counts = Counter(row.get("phase") for row in rows)
    for phase, count in counts.items():
        if count > 1:
            _append_once(errors, "duplicate_phase_event")
        if phase not in PHASES:
            _append_once(errors, "unknown_phase")
    for phase in PHASES:
        if counts[phase] == 0:
            errors.append(f"missing_phase:{phase}")

    for row in rows:
        phase = str(row.get("phase"))
        for field in PHASE_RECEIPT_FIELDS:
            if field not in row:
                errors.append(f"missing_required_field:{phase}:{field}")

        mono_start = row.get("monotonic_start_ns")
        mono_end = row.get("monotonic_end_ns")
        wall_start = _parse_wall(row.get("wall_clock_start"))
        wall_end = _parse_wall(row.get("wall_clock_end"))
        if row.get("wall_clock_start") is None or row.get("wall_clock_end") is None:
            _append_once(errors, "missing_timestamp")
        if mono_start is None or wall_start is None:
            _append_once(errors, "absent_start_receipt")
        if mono_end is None or wall_end is None:
            _append_once(errors, "absent_end_receipt")

        valid_mono = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (mono_start, mono_end)
        )
        if valid_mono and mono_end < mono_start:
            _append_once(errors, "negative_interval")
        if wall_start is not None and wall_end is not None and wall_end < wall_start:
            _append_once(errors, "negative_interval")
        if valid_mono and wall_start is not None and wall_end is not None:
            mono_s = (mono_end - mono_start) / 1_000_000_000
            wall_s = (wall_end - wall_start).total_seconds()
            if abs(mono_s - wall_s) > 0.001:
                _append_once(errors, "contradictory_clocks")

        deadline = row.get("deadline")
        if not isinstance(deadline, int) or isinstance(deadline, bool):
            _append_once(errors, "invalid_deadline")
        elif isinstance(mono_end, int) and not isinstance(mono_end, bool) and deadline < mono_end:
            _append_once(errors, "deadline_before_phase_end")
        pid = row.get("subprocess_pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            _append_once(errors, "invalid_subprocess_pid")
        if row.get("exit_state") not in {"completed", "failed", "terminated", "unknown"}:
            _append_once(errors, "invalid_exit_state")
        if row.get("timeout_state") not in {"not_timed_out", "timed_out", "unknown"}:
            _append_once(errors, "invalid_timeout_state")
        evidence_hash = row.get("evidence_hash")
        hash_valid = bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(evidence_hash or "")))
        if not hash_valid:
            _append_once(errors, "invalid_evidence_hash")
        if evidence_hash == "sha256:" + "0" * 64:
            _append_once(errors, "placeholder_evidence_hash")
            hash_valid = False
        if row.get("exit_state") == "completed" and (
            mono_end is None or wall_end is None or not hash_valid
        ):
            _append_once(errors, "completion_without_complete_evidence")
    return errors


def phase_receipt_schema() -> JsonDict:
    """Describe the future row contract in data so another runner can reuse it."""

    return {
        "schema_version": "carnot.arc_phase_receipt.v1",
        "required_phases": list(PHASES),
        "required_fields": list(PHASE_RECEIPT_FIELDS),
        "deadline_clock_domain": "monotonic_ns",
        "wall_clock_format": "RFC3339 with timezone",
        "clock_duration_tolerance_s": 0.001,
        "exit_state_values": ["completed", "failed", "terminated", "unknown"],
        "timeout_state_values": ["not_timed_out", "timed_out", "unknown"],
        "completion_rule": "completed requires both ends and a non-placeholder evidence hash",
    }


def _contract_fixture_receipts() -> list[JsonDict]:
    wall = datetime(2026, 9, 7, 20, 0, tzinfo=UTC)
    rows = []
    for index, phase in enumerate(PHASES):
        start = 1_000_000_000 + index * 2_000_000_000
        wall_start = wall + timedelta(seconds=index * 2)
        rows.append(
            make_phase_receipt(
                phase=phase,
                monotonic_start_ns=start,
                monotonic_end_ns=start + 1_000_000_000,
                wall_clock_start=wall_start.isoformat().replace("+00:00", "Z"),
                wall_clock_end=(wall_start + timedelta(seconds=1))
                .isoformat()
                .replace("+00:00", "Z"),
                deadline=start + 1_500_000_000,
                subprocess_pid=8000 + index,
                exit_state="completed",
                timeout_state="not_timed_out",
                evidence_hash=f"sha256:{index + 1:064x}",
            )
        )
    return rows


def synthetic_attack_matrix(base_rows: list[JsonDict] | None = None) -> list[JsonDict]:
    """Apply each required mutation and retain the validator's rejection reason."""

    original = deepcopy(base_rows if base_rows is not None else _contract_fixture_receipts())
    attacks: dict[str, list[JsonDict]] = {}

    rows = deepcopy(original)
    rows[0]["wall_clock_start"] = "not-a-timestamp"
    rows[0]["wall_clock_end"] = None
    attacks["missing_timestamps"] = rows

    rows = deepcopy(original)
    start = _parse_wall(rows[0]["wall_clock_start"])
    assert start is not None
    rows[0]["wall_clock_end"] = (start + timedelta(seconds=3)).isoformat().replace("+00:00", "Z")
    attacks["contradictory_clocks"] = rows

    rows = deepcopy(original)
    rows[0]["monotonic_start_ns"] = None
    rows[0]["wall_clock_start"] = None
    attacks["absent_start_receipt"] = rows

    rows = deepcopy(original)
    rows[0]["monotonic_end_ns"] = None
    rows[0]["wall_clock_end"] = None
    attacks["absent_end_receipt"] = rows

    rows = deepcopy(original)
    rows[0]["monotonic_end_ns"] = rows[0]["monotonic_start_ns"] - 1
    attacks["negative_interval"] = rows

    rows = deepcopy(original)
    rows.append(deepcopy(rows[0]))
    attacks["duplicate_events"] = rows

    rows = deepcopy(original)
    rows[0]["monotonic_end_ns"] = None
    rows[0]["wall_clock_end"] = None
    rows[0]["evidence_hash"] = "sha256:" + "0" * 64
    attacks["fabricated_phase_completion"] = rows

    return [
        {
            "attack_id": attack_id,
            "fail_closed": bool(errors := validate_phase_receipts(attacks[attack_id])),
            "errors": errors,
        }
        for attack_id in ATTACK_IDS
    ]


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _source_row(
    *,
    event: str,
    source_path: str,
    source_sha256: str | None,
    clock_domain: str,
    timestamp_precision_s: float | int | None,
    timestamp: str | None = None,
    timestamp_lower: str | None = None,
    timestamp_upper: str | None = None,
    **values: Any,
) -> JsonDict:
    return {
        "row_type": "source_timeline",
        "event": event,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "clock_domain": clock_domain,
        "timestamp_precision_s": timestamp_precision_s,
        "timestamp": timestamp,
        "timestamp_lower": timestamp_lower,
        "timestamp_upper": timestamp_upper,
        **values,
    }


def parse_exp7123_artifact(
    path: Path, *, source_path: str | None = None
) -> tuple[JsonDict, list[JsonDict]]:
    """Read the prior artifact and expose its legacy markers without adding clocks."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    display = source_path or str(path)
    checks = {row.get("check"): row for row in payload.get("preconditions_checked", [])}
    selection_rows = payload.get("game_selection_rows") or []
    initialized = checks.get("artifact_initialized", {}).get("observed_value") is True
    frozen = bool(selection_rows and selection_rows[0].get("selection_frozen"))
    common = {
        "source_path": display,
        "source_sha256": digest,
        "clock_domain": "artifact_declared_state_without_clock",
        "timestamp_precision_s": None,
    }
    rows = [
        _source_row(
            event="artifact_legacy_start_marker",
            phase="artifact_initialization",
            marker_present=initialized,
            **common,
        ),
        _source_row(
            event="registry_legacy_start_marker",
            phase="registry_freeze",
            marker_present=frozen,
            selected_game=selection_rows[0].get("game") if selection_rows else None,
            **common,
        ),
        _source_row(
            event="artifact_snapshot",
            duration_s=payload.get("duration_s"),
            inference_substrate_class=payload.get("inference_substrate_class"),
            verdict_class=payload.get("verdict_class"),
            honest_verdict=payload.get("honest_verdict"),
            reproducibility_checksum=payload.get("reproducibility_checksum"),
            gate_checks=payload.get("gate_check_summary", {}).get("checks"),
            **common,
        ),
    ]
    return payload, rows


def _minute_bounds(value: str) -> tuple[str, str]:
    start = datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    end = start + timedelta(minutes=1, microseconds=-1)
    return (
        start.isoformat().replace("+00:00", "Z"),
        end.isoformat().replace("+00:00", "Z"),
    )


def parse_conductor_rows(path: Path, *, source_path: str | None = None) -> list[JsonDict]:
    """Parse only Exp7123 rows and preserve the log's one-minute precision."""

    text = path.read_text(encoding="utf-8")
    digest = sha256_file(path)
    display = source_path or str(path)
    rows = []
    pattern = re.compile(
        r"^\| (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC "
        r"\| Adapter-withheld ARC leave-one-game-out shard A \| (?P<status>[^|]+) \| (?P<detail>.*) \|$"
    )
    for line_number, line in enumerate(text.splitlines(), 1):
        match = pattern.match(line)
        if match is None:
            continue
        detail = match.group("detail").strip()
        status = match.group("status").strip()
        if "artifact_verdict_not_terminal" in detail:
            event = "artifact_postflight_failure"
        elif "timeout after" in detail:
            event = "conductor_timeout"
        elif status == "SKIP":
            event = "task_exit"
        else:  # pragma: no cover - future conductor rows remain preserved as unknown.
            event = "conductor_event"
        lower, upper = _minute_bounds(match.group("time"))
        elapsed = re.search(r"after (\d+)s", detail)
        silence = re.search(r"\((\d+)s silence\)", detail)
        row = _source_row(
            event=event,
            source_path=display,
            source_sha256=digest,
            clock_domain="conductor_utc_wall_clock",
            timestamp_precision_s=60,
            timestamp_lower=lower,
            timestamp_upper=upper,
            status=status,
            detail=detail,
            source_line=line_number,
            elapsed_s=int(elapsed.group(1)) if elapsed else None,
            silence_s=int(silence.group(1)) if silence else None,
        )
        if event == "conductor_timeout" and elapsed:
            row["derived_task_start_lower"] = _shift_iso(lower, -int(elapsed.group(1)))
            row["derived_task_start_upper"] = _shift_iso(upper, -int(elapsed.group(1)))
        rows.append(row)
    return rows


def _parse_stat_timestamp(value: str) -> tuple[str, float, str] | None:
    """Normalize GNU stat time while retaining all reported fractional digits."""

    match = re.fullmatch(
        r"(?P<base>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d+))? (?P<offset>[+-]\d{4})",
        value.strip(),
    )
    if match is None:
        return None
    fraction = match.group("fraction") or ""
    parsed = datetime.strptime(
        f"{match.group('base')} {match.group('offset')}", "%Y-%m-%d %H:%M:%S %z"
    ).astimezone(UTC)
    normalized = parsed.strftime("%Y-%m-%dT%H:%M:%S")
    if fraction:
        normalized += f".{fraction}"
    normalized += "Z"
    precision = 10 ** (-len(fraction)) if fraction else 1.0
    return normalized, precision, value.strip()


def filesystem_timestamp_rows(
    path: Path,
    *,
    source_path: str | None = None,
    event_prefix: str = "artifact_filesystem",
) -> list[JsonDict]:
    """Read birth and modification clocks without treating either as a phase clock."""

    completed = subprocess.run(
        ["stat", "--printf=%w\n%y", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    display = source_path or str(path)
    digest = sha256_file(path)
    rows = []
    for event, raw in zip(
        (f"{event_prefix}_birth", f"{event_prefix}_mtime"), completed.stdout.splitlines()
    ):
        parsed = _parse_stat_timestamp(raw)
        if parsed is None:
            rows.append(
                _source_row(
                    event=f"{event}_unavailable",
                    source_path=display,
                    source_sha256=digest,
                    clock_domain="filesystem_wall_clock",
                    timestamp_precision_s=None,
                    source_timestamp=raw,
                )
            )
            continue
        timestamp, precision, source_timestamp = parsed
        rows.append(
            _source_row(
                event=event,
                source_path=display,
                source_sha256=digest,
                clock_domain="filesystem_wall_clock",
                timestamp_precision_s=precision,
                timestamp=timestamp,
                timestamp_lower=timestamp,
                timestamp_upper=timestamp,
                source_timestamp=source_timestamp,
            )
        )
    return rows


def _receipt_inventory_rows(
    payload: JsonDict,
    path: Path,
    *,
    display: str,
    root: Path,
) -> list[JsonDict]:
    digest = sha256_file(path)
    rows = []
    for field in ("process_rows", "raw_trace_receipts"):
        values = payload.get(field) or []
        rows.append(
            _source_row(
                event=f"{field}_inventory",
                source_path=display,
                source_sha256=digest,
                clock_domain="artifact_declared_state_without_clock",
                timestamp_precision_s=None,
                receipt_field=field,
                receipt_count=len(values),
            )
        )
    selection_rows = payload.get("game_selection_rows") or []
    raw_paths = selection_rows[0].get("raw_output_paths", {}) if selection_rows else {}
    for arm, raw_path in sorted(raw_paths.items()):
        candidate = Path(str(raw_path))
        rows.append(
            _source_row(
                event="raw_trace_path_present" if candidate.exists() else "raw_trace_path_absent",
                source_path=_display_path(candidate, root),
                source_sha256=sha256_file(candidate),
                clock_domain="no_clock_receipt",
                timestamp_precision_s=None,
                phase=f"{arm}_arm" if arm == "adapter_withheld" else "control_arm",
                arm=arm,
            )
        )
    return rows


def _upstream_inventory(path: Path, *, display: str) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = {
        field: len(payload.get(field) or [])
        for field in ("rows", "process_rows", "raw_trace_receipts", "request_timing_rows")
    }
    return _source_row(
        event="upstream_receipt_inventory",
        source_path=display,
        source_sha256=sha256_file(path),
        clock_domain="artifact_declared_state_without_clock",
        timestamp_precision_s=None,
        receipt_counts=counts,
        transferable_to_exp7123=False,
    )


def _precondition(check: str, expected: Any, observed: Any, *, source_path: str) -> JsonDict:
    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "source_path": source_path,
    }


def collect_source_evidence(
    root: Path,
    *,
    output_path: Path | None = None,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    """Read each named source independently and retain exact missing-file gaps."""

    root = root.resolve()
    output = output_path or root / RESULT_PATH
    source_hashes: JsonDict = {}
    source_rows: list[JsonDict] = []
    preconditions = []
    for relative in NAMED_SOURCE_PATHS:
        path = root / relative
        readable = path.is_file() and os.access(path, os.R_OK)
        preconditions.append(
            _precondition("source_readable", True, readable, source_path=relative.as_posix())
        )
        source_hashes[relative.as_posix()] = sha256_file(path)
        if not readable:
            source_rows.append(
                _source_row(
                    event="source_missing",
                    source_path=relative.as_posix(),
                    source_sha256=None,
                    clock_domain="no_clock_receipt",
                    timestamp_precision_s=None,
                )
            )

    output_parent_writable = output.parent.is_dir() and os.access(output.parent, os.W_OK)
    if output.exists():
        output_parent_writable = output_parent_writable and os.access(output, os.W_OK)
    preconditions.append(
        _precondition(
            "artifact_path_writable",
            True,
            output_parent_writable,
            source_path=_display_path(output, root),
        )
    )

    exp_path = root / EXP7123_PATH
    if exp_path.is_file():
        payload, artifact_rows = parse_exp7123_artifact(
            exp_path, source_path=EXP7123_PATH.as_posix()
        )
        source_rows.extend(artifact_rows)
        source_rows.extend(
            filesystem_timestamp_rows(
                exp_path,
                source_path=EXP7123_PATH.as_posix(),
            )
        )
        source_rows.extend(
            _receipt_inventory_rows(
                payload,
                exp_path,
                display=EXP7123_PATH.as_posix(),
                root=root,
            )
        )
    conductor = root / CONDUCTOR_PATH
    if conductor.is_file():
        source_rows.extend(parse_conductor_rows(conductor, source_path=CONDUCTOR_PATH.as_posix()))
    for relative in (EXP7099_PATH, EXP7113_PATH):
        path = root / relative
        if path.is_file():
            source_rows.append(_upstream_inventory(path, display=relative.as_posix()))

    bytecode = root / ORPHAN_TEST_BYTECODE
    if bytecode.is_file():
        source_hashes[ORPHAN_TEST_BYTECODE.as_posix()] = sha256_file(bytecode)
        source_rows.extend(
            filesystem_timestamp_rows(
                bytecode,
                source_path=ORPHAN_TEST_BYTECODE.as_posix(),
                event_prefix="setup_test_bytecode_filesystem",
            )
        )
    else:
        source_rows.append(
            _source_row(
                event="setup_test_bytecode_absent",
                source_path=ORPHAN_TEST_BYTECODE.as_posix(),
                source_sha256=None,
                clock_domain="no_clock_receipt",
                timestamp_precision_s=None,
            )
        )
    return source_hashes, source_rows, preconditions


def _event(rows: list[JsonDict], name: str) -> JsonDict | None:
    return next((row for row in rows if row.get("event") == name), None)


def _shift_iso(value: str, seconds: int) -> str:
    parsed = _parse_wall(value)
    assert parsed is not None
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _phase_rows(source_rows: list[JsonDict]) -> list[JsonDict]:
    init = _event(source_rows, "artifact_legacy_start_marker")
    freeze = _event(source_rows, "registry_legacy_start_marker")
    birth = _event(source_rows, "artifact_filesystem_birth")
    setup_activity = _event(source_rows, "setup_test_bytecode_filesystem_birth")
    validation = _event(source_rows, "artifact_postflight_failure")
    snapshot = _event(source_rows, "artifact_snapshot")
    rows = []
    for index, phase in enumerate(PHASES):
        start_marker = None
        end_evidence = None
        interval_kind = "unobserved_missing_receipts"
        observed_status = "no_phase_evidence"
        if phase == "artifact_initialization" and init and init.get("marker_present"):
            start_marker = init
            end_evidence = birth
            interval_kind = "right_bounded_by_filesystem_birth"
            observed_status = "legacy_marker_present_without_start_clock"
        elif phase == "registry_freeze" and freeze and freeze.get("marker_present"):
            start_marker = freeze
            end_evidence = birth
            interval_kind = "right_bounded_by_artifact_birth"
            observed_status = "legacy_marker_present_without_start_clock"
        elif phase == "setup" and setup_activity:
            interval_kind = "observed_activity_without_phase_boundary"
            observed_status = "test_bytecode_written_without_phase_receipt"
        elif phase == "validation" and validation:
            interval_kind = "external_validation_event_without_phase_boundary"
            observed_status = "conductor_rejected_nonterminal_artifact"
        elif phase == "finalization" and snapshot:
            interval_kind = "artifact_exists_without_valid_finalization_receipt"
            observed_status = "placeholder_checksum_and_partial_verdict"
        rows.append(
            {
                "row_type": "phase_timing",
                "phase_index": index,
                "phase": phase,
                "start_receipt_present": start_marker is not None,
                "end_receipt_present": end_evidence is not None,
                "start_timestamp": start_marker.get("timestamp") if start_marker else None,
                "end_timestamp": end_evidence.get("timestamp") if end_evidence else None,
                "interval_start_lower": None,
                "interval_start_upper": None,
                "interval_end_lower": end_evidence.get("timestamp_lower") if end_evidence else None,
                "interval_end_upper": end_evidence.get("timestamp_upper") if end_evidence else None,
                "interval_kind": interval_kind,
                "observed_status": observed_status,
                "future_schema_complete": False,
                "completion_claim_supported": False,
                "evidence_events": [
                    event["event"]
                    for event in (
                        start_marker,
                        end_evidence,
                        setup_activity if phase == "setup" else None,
                        validation if phase == "validation" else None,
                        snapshot if phase == "finalization" else None,
                    )
                    if event is not None
                ],
            }
        )
    return rows


def _first_absent_start(phase_rows: list[JsonDict]) -> JsonDict:
    row = next((item for item in phase_rows if not item["start_receipt_present"]), None)
    if row is None:  # pragma: no cover - historical rows cannot satisfy the future schema.
        return {"phase": None, "receipt": None, "reason": None}
    reason = (
        "no_setup_start_marker_in_any_source"
        if row["phase"] == "setup"
        else f"no_{row['phase']}_start_marker_in_any_source"
    )
    return {"phase": row["phase"], "receipt": "legacy_start_marker", "reason": reason}


def _missing_rows(source_rows: list[JsonDict], phase_rows: list[JsonDict]) -> list[JsonDict]:
    rows = []
    for phase in phase_rows:
        if phase["start_receipt_present"] and phase["start_timestamp"] is None:
            rows.append(
                {
                    "row_type": "missing_receipt",
                    "phase": phase["phase"],
                    "gap": "phase_start_timestamp_absent",
                }
            )
        if not phase["start_receipt_present"]:
            rows.append(
                {
                    "row_type": "missing_receipt",
                    "phase": phase["phase"],
                    "gap": "phase_start_receipt_absent",
                }
            )
        if not phase["end_receipt_present"]:
            rows.append(
                {
                    "row_type": "missing_receipt",
                    "phase": phase["phase"],
                    "gap": "phase_end_receipt_absent",
                }
            )
    if _event(source_rows, "artifact_snapshot"):
        snapshot_fields = {
            "phase_timing_rows_absent": True,
            "request_rows_absent": True,
            "token_rows_absent": True,
            "action_rows_absent": True,
            "process_rows_absent": (_event(source_rows, "process_rows_inventory") or {}).get(
                "receipt_count"
            )
            == 0,
            "raw_trace_rows_absent": (
                _event(source_rows, "raw_trace_receipts_inventory") or {}
            ).get("receipt_count")
            == 0,
            "stop_reason_receipt_absent": True,
        }
        rows.extend(
            {"row_type": "missing_receipt", "phase": None, "gap": gap}
            for gap, absent in snapshot_fields.items()
            if absent
        )
    rows.extend(
        {
            "row_type": "missing_receipt",
            "phase": row.get("phase"),
            "gap": "raw_trace_path_absent",
            "source_path": row["source_path"],
        }
        for row in source_rows
        if row.get("event") == "raw_trace_path_absent"
    )
    return rows


def _contradiction_rows(source_rows: list[JsonDict]) -> list[JsonDict]:
    snapshot = _event(source_rows, "artifact_snapshot") or {}
    timeout = _event(source_rows, "conductor_timeout")
    rows = []
    if snapshot.get("duration_s") == 0.0 and timeout:
        rows.append(
            {
                "row_type": "contradiction",
                "kind": "zero_duration_vs_observed_lifecycle",
                "left": 0.0,
                "right": "later conductor timeout exists",
                "resolution": "Exp7123 duration is not used as elapsed evidence",
            }
        )
    if snapshot.get("inference_substrate_class") == "blocked_no_run" and not str(
        snapshot.get("honest_verdict") or ""
    ).startswith("blocked_"):
        rows.append(
            {
                "row_type": "contradiction",
                "kind": "substrate_class_vs_verdict_prefix",
                "left": snapshot.get("inference_substrate_class"),
                "right": snapshot.get("honest_verdict"),
                "resolution": "preserve the historical corrigendum; do not infer execution",
            }
        )
    if snapshot.get("reproducibility_checksum") == "sha256:" + "0" * 64:
        rows.append(
            {
                "row_type": "contradiction",
                "kind": "completion_prefix_vs_placeholder_checksum",
                "left": snapshot.get("honest_verdict"),
                "right": snapshot.get("reproducibility_checksum"),
                "resolution": "finalization remains unsupported",
            }
        )
    return rows


def _clock_rows(source_rows: list[JsonDict]) -> list[JsonDict]:
    seen = {}
    for row in source_rows:
        domain = row.get("clock_domain")
        if domain not in seen:
            seen[domain] = {
                "row_type": "clock_domain",
                "clock_domain": domain,
                "timestamp_precision_s": row.get("timestamp_precision_s"),
                "cross_domain_exact_ordering_allowed": False,
                "source_paths": [],
            }
        if row.get("source_path") not in seen[domain]["source_paths"]:
            seen[domain]["source_paths"].append(row.get("source_path"))
    seen["monotonic_ns_unavailable_for_exp7123"] = {
        "row_type": "clock_domain",
        "clock_domain": "monotonic_ns_unavailable_for_exp7123",
        "timestamp_precision_s": None,
        "cross_domain_exact_ordering_allowed": False,
        "source_paths": [EXP7123_PATH.as_posix()],
    }
    return list(seen.values())


def _stall_interval(source_rows: list[JsonDict]) -> JsonDict:
    timeout = _event(source_rows, "conductor_timeout")
    if timeout is None or timeout.get("silence_s") is None:
        return {
            "observed": False,
            "duration_s": None,
            "start_lower": None,
            "start_upper": None,
            "end_lower": None,
            "end_upper": None,
            "clock_domain": None,
            "source_path": None,
        }
    silence = int(timeout["silence_s"])
    return {
        "observed": True,
        "duration_s": silence,
        "start_lower": _shift_iso(timeout["timestamp_lower"], -silence),
        "start_upper": _shift_iso(timeout["timestamp_upper"], -silence),
        "end_lower": timeout["timestamp_lower"],
        "end_upper": timeout["timestamp_upper"],
        "clock_domain": timeout["clock_domain"],
        "source_path": timeout["source_path"],
    }


def phase_budget_rows() -> list[JsonDict]:
    """Return the four future caps; none describes elapsed Exp7123 time."""

    return [
        {
            "row_type": "phase_budget",
            "budget_group": "setup",
            "phases": list(PHASES[:6]),
            "cap_s": 300,
            "contract_for_future_run": True,
            "evidence_about_exp7123": False,
        },
        {
            "row_type": "phase_budget",
            "budget_group": "withheld_arm",
            "phases": ["withheld_arm"],
            "cap_s": 1500,
            "contract_for_future_run": True,
            "evidence_about_exp7123": False,
        },
        {
            "row_type": "phase_budget",
            "budget_group": "control_arm",
            "phases": ["control_arm"],
            "cap_s": 1500,
            "contract_for_future_run": True,
            "evidence_about_exp7123": False,
        },
        {
            "row_type": "phase_budget",
            "budget_group": "finalization",
            "phases": list(PHASES[8:]),
            "cap_s": 300,
            "contract_for_future_run": True,
            "evidence_about_exp7123": False,
        },
    ]


def _projection(source_rows: list[JsonDict]) -> JsonDict:
    phase_rows = _phase_rows(source_rows)
    return {
        "phase_timing_rows": phase_rows,
        "clock_domain_rows": _clock_rows(source_rows),
        "missing_receipt_rows": _missing_rows(source_rows, phase_rows),
        "contradiction_rows": _contradiction_rows(source_rows),
        "first_absent_start_receipt": _first_absent_start(phase_rows),
        "observed_stall_interval": _stall_interval(source_rows),
    }


def _combined_rows(
    source_rows: list[JsonDict], projection: JsonDict, budgets: list[JsonDict]
) -> list[JsonDict]:
    return [
        *source_rows,
        *projection["phase_timing_rows"],
        *projection["clock_domain_rows"],
        *projection["missing_receipt_rows"],
        *projection["contradiction_rows"],
        *budgets,
    ]


def _gate_summary(defensible: bool, ready: bool) -> JsonDict:
    checks = [
        {
            "check": "defensible_timeline",
            "expected_value": True,
            "observed_value": defensible,
            "passed": defensible,
        },
        {
            "check": "phase_contract_ready",
            "expected_value": True,
            "observed_value": ready,
            "passed": ready,
        },
    ]
    failed = next((row for row in checks if not row["passed"]), None)
    return {
        "passed": failed is None,
        "failed_check": failed["check"] if failed else None,
        "expected_value": failed["expected_value"] if failed else None,
        "observed_value": failed["observed_value"] if failed else None,
        "checks": checks,
    }


def build_artifact(
    *,
    root: Path,
    run_date: str,
    duration_s: float,
    output_path: Path | None = None,
) -> JsonDict:
    """Build a terminal diagnostic from immutable historical evidence only."""

    output = output_path or root / RESULT_PATH
    source_hashes, source_rows, preconditions = collect_source_evidence(root, output_path=output)
    projection = _projection(source_rows)
    budgets = phase_budget_rows()
    schema = phase_receipt_schema()
    attacks = synthetic_attack_matrix()
    schema["synthetic_attack_rows"] = attacks
    schema_complete = schema["required_phases"] == list(PHASES) and schema[
        "required_fields"
    ] == list(PHASE_RECEIPT_FIELDS)
    budget_phases = [phase for row in budgets for phase in row["phases"]]
    budgets_complete = (
        sorted(budget_phases) == sorted(PHASES) and sum(row["cap_s"] for row in budgets) <= 3600
    )
    defensible = all(
        _event(source_rows, event) is not None
        for event in ("artifact_snapshot", "artifact_filesystem_birth", "conductor_timeout")
    )
    ready = (
        defensible
        and schema_complete
        and budgets_complete
        and all(row["fail_closed"] for row in attacks)
    )
    verdict_class = "positive" if ready else "blocked"
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "aggregation" if defensible else "blocked_no_run",
        "execution_venue": "host",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": source_hashes,
        "rows": _combined_rows(source_rows, projection, budgets),
        "source_timeline_rows": source_rows,
        **projection,
        "phase_receipt_schema": schema,
        "phase_budget_rows": budgets,
        "setup_cap_s": 300,
        "withheld_arm_cap_s": 1500,
        "control_arm_cap_s": 1500,
        "finalization_cap_s": 300,
        "value_measurement_run": False,
        "solve_claim_made": False,
        "arc_phase_receipt_contract_ready_score": int(ready),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(defensible, ready),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": (
            "complete_positive_arc_phase_receipt_contract_ready_no_value_run"
            if ready
            else "blocked_no_defensible_exp7123_phase_timeline"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> list[str]:
    """Recompute every projection and refuse a forged positive diagnostic."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("artifact_fields_mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS) or any(
        not str(value).strip() for value in artifact.get("field_principles", {}).values()
    ):
        errors.append("field_principles_mismatch")
    source_rows = artifact.get("source_timeline_rows") or []
    projection = _projection(source_rows)
    for field, expected in projection.items():
        if artifact.get(field) != expected:
            errors.append(f"{field}_mismatch")
    budgets = phase_budget_rows()
    if artifact.get("phase_budget_rows") != budgets:
        errors.append("phase_budget_rows_mismatch")
    schema = phase_receipt_schema()
    schema["synthetic_attack_rows"] = synthetic_attack_matrix()
    if artifact.get("phase_receipt_schema") != schema:
        errors.append("phase_receipt_schema_mismatch")
    defensible = all(
        _event(source_rows, event) is not None
        for event in ("artifact_snapshot", "artifact_filesystem_birth", "conductor_timeout")
    )
    budget_phases = [phase for row in budgets for phase in row["phases"]]
    ready = (
        defensible
        and sorted(budget_phases) == sorted(PHASES)
        and sum(row["cap_s"] for row in budgets) <= 3600
        and all(row["fail_closed"] for row in schema["synthetic_attack_rows"])
    )
    expected_class = "positive" if ready else "blocked"
    expected_substrate_class = "aggregation" if defensible else "blocked_no_run"
    expected_verdict = (
        "complete_positive_arc_phase_receipt_contract_ready_no_value_run"
        if ready
        else "blocked_no_defensible_exp7123_phase_timeline"
    )
    checks = {
        "rows": artifact.get("rows") == _combined_rows(source_rows, projection, budgets),
        "score": artifact.get("arc_phase_receipt_contract_ready_score") == int(ready),
        "gate": artifact.get("gate_check_summary") == _gate_summary(defensible, ready),
        "substrate": artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "substrate_class": artifact.get("inference_substrate_class") == expected_substrate_class,
        "venue": artifact.get("execution_venue") == "host",
        "no_value": artifact.get("value_measurement_run") is False,
        "no_solve": artifact.get("solve_claim_made") is False,
        "non_oracle": artifact.get("verifier_is_oracle") is False,
        "verdict_class": artifact.get("verdict_class") == expected_class,
        "honest_verdict": artifact.get("honest_verdict") == expected_verdict,
        "caps": [artifact.get(field) for field in CAP_FIELDS] == [300, 1500, 1500, 300],
        "checksum": artifact.get("reproducibility_checksum") == artifact_checksum(artifact),
    }
    errors.extend(f"{name}_mismatch" for name, passed in checks.items() if not passed)
    for path, digest in artifact.get("source_artifact_hashes", {}).items():
        if digest is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)):
            errors.append(f"source_hash_invalid:{path}")
    return errors


def write_artifact(path: Path, artifact: JsonDict) -> None:
    """Publish once through the repository's atomic JSON writer."""

    # An explicit CLI path belongs to the caller. Ignoring the test-suite output
    # redirect here lets temporary absolute paths stay temporary.
    atomic_write_json(path, artifact, env={})


def run(*, root: Path, run_date: str, output_path: Path) -> JsonDict:
    """Measure this diagnostic's own runtime, validate it, then write once."""

    started = time.perf_counter()
    artifact = build_artifact(
        root=root,
        run_date=run_date,
        duration_s=0.0,
        output_path=output_path,
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"Exp7126 artifact validation failed: {errors}")
    write_artifact(output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    run(root=root, run_date=args.date, output_path=output)
    return 0


if __name__ == "__main__":  # pragma: no cover - the thin script calls main in production.
    raise SystemExit(main())
