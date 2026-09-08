#!/usr/bin/env python3
"""Audit changed GateMate state before a bounded continuity attempt.

Spec refs: REQ-HARDWARE-7146, SCENARIO-HARDWARE-7146-1,
SCENARIO-HARDWARE-7146-2, SCENARIO-HARDWARE-7146-3,
SCENARIO-HARDWARE-7146-4.

The artifact exists before this module reads a receipt or inspects a tool. A
new, complete operator receipt can authorize one detect. A clean expected
identity can then authorize only the fixed n=16 smoke from the historical
source. The ledger stops after the first failed action.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
CommandRunner = Callable[[tuple[str, ...], float], "CommandResult"]
CheckpointWriter = Callable[[JsonDict, Path], Path]

SCHEMA = "carnot.experiment_7146.v627.gatemate_changed_state.v1"
EXPERIMENT = "experiment_7146_v627_gatemate_changed_state"
RUN_DATE = "20260908"
RANDOM_SEED = 7146
RESULT_PATH = Path("results/experiment_7146_v627_gatemate_changed_state.json")
SPEC_REFS = (
    "REQ-HARDWARE-7146",
    "SCENARIO-HARDWARE-7146-1",
    "SCENARIO-HARDWARE-7146-2",
    "SCENARIO-HARDWARE-7146-3",
    "SCENARIO-HARDWARE-7146-4",
)

EXPECTED_BOARD = "Cologne Chip GateMate A1-EVB-2M"
EXPECTED_IDCODE = "0x20000001"
EXPECTED_USB_JTAG_IDENTITY = "1209:c0ca DirtyJTAG"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
DETECT_TIMEOUT_S = 30.0
SMOKE_TIMEOUT_S = 60.0
NO_COMMAND_SUBSTRATE = "dated_hardware_receipt_audit_no_command_no_llm"
HARDWARE_SUBSTRATE = "hardware_smoke"

EXP3866_REL_PATH = Path("results/experiment_3866_gatemate_ising_tile_flash_v2.json")
EXP6325_REL_PATH = Path("results/experiment_6325_gatemate_dated_receipt_single_detect.json")
EXP6525_REL_PATH = Path("results/experiment_6525_gatemate_changed_state_continuity.json")
EXP6559_REL_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
EXCLUSION_REL_PATH = Path("ops/exclusion_manifest.yaml")
SMOKE_BITSTREAM_REL_PATH = Path(
    "build/gatemate/experiment_3866_gatemate_ising_tile_flash_v2/gatemate_ising_n16.bit"
)
SMOKE_RTL_REL_PATH = Path("rtl/gatemate_ising_n16.v")
RECEIPT_PATHS = (
    Path("ops/known-issues.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/operator-followup.md"),
)
PHYSICAL_CHANGE_FIELDS = frozenset(
    {
        "board",
        "board_presence",
        "power",
        "usb_jtag_cable_state",
        "cable",
        "port",
        "host_path",
        "dirtyjtag",
    }
)
VALID_RECOVERY_ACTIONS = frozenset({"detect", "detect_then_existing_n16_smoke"})

PROMPT_REQUIRED_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "receipt_rows",
    "physical_state_receipt",
    "receipt_cutoff_experiment",
    "receipt_newer_than_exp6559_score",
    "command_rows",
    "hardware_command_count",
    "detect_rows",
    "identity_rows",
    "smoke_rows",
    "first_failure_stop_score",
    "bitstream_redesigned",
    "exclusion_manifest_modified",
    "gatemate_terminal_receipt_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Every required field states the evidence rule that controls it.",
    "preconditions_checked": "Artifact initialization, receipt cutoff, source identity, and output readiness precede hardware access.",
    "run_date": "The declared execution date bounds receipt selection and artifact identity.",
    "inference_substrate": "The substrate distinguishes a no-command audit from direct hardware contact.",
    "inference_substrate_class": "The class distinguishes blocked work from hardware access with no model load.",
    "execution_venue": "The venue is host until an authorized board command starts.",
    "duration_s": "Measured elapsed time records work without padding.",
    "source_artifact_hashes": "Hashes pin the cutoff, bitstream, constraints, manifest, and historical evidence.",
    "receipt_rows": "One row per candidate records every acceptance or rejection reason.",
    "physical_state_receipt": "Only one complete operator receipt can authorize board access.",
    "receipt_cutoff_experiment": "Exp6559 is the strict no-repeat boundary.",
    "receipt_newer_than_exp6559_score": "A bare score states whether the cutoff and physical-change contract passed.",
    "command_rows": "An append-only ledger preserves every attempted hardware action.",
    "hardware_command_count": "The count must equal the command ledger length.",
    "detect_rows": "The first hardware action must be the one allowed detect command.",
    "identity_rows": "Raw detect output must prove one clean expected GateMate identity.",
    "smoke_rows": "The fixed n=16 smoke can follow only a clean detect.",
    "first_failure_stop_score": "No action may follow the first failed or unclean action.",
    "bitstream_redesigned": "False preserves the existing n=16 source and constraints.",
    "exclusion_manifest_modified": "False keeps exclusion changes outside this continuity task.",
    "gatemate_terminal_receipt_score": "A terminal safe block or bounded action sequence completes continuity.",
    "random_seed": "The experiment identifier supplies the stable seed.",
    "reproducibility_checksum": "The final checksum detects receipt, ledger, or source drift.",
    "gate_check_summary": "The first failed gate records its expected and observed values.",
    "verifier_is_oracle": "False prevents the continuity transcript from becoming a model oracle.",
    "verdict_class": "The class separates positive, null, blocked, disqualified, and partial outcomes.",
    "honest_verdict": "The terminal prefix must agree with the verdict class and observed rows.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "spec_refs",
    "result_path",
    *PROMPT_REQUIRED_FIELDS,
)


@dataclass(frozen=True)
class CommandResult:
    """Hold raw output from one bounded board command."""

    return_code: int | None
    stdout: str
    stderr: str
    timeout: bool
    duration_s: float


class ActionLedger:
    """Append actions and reject work after the first failed action."""

    def __init__(self) -> None:
        self._rows: list[JsonDict] = []

    @property
    def rows(self) -> list[JsonDict]:
        """Return a copy so callers cannot rewrite earlier evidence."""

        return deepcopy(self._rows)

    def append(self, row: Mapping[str, Any]) -> None:
        """Add one row only when no earlier action failed."""

        if any(item.get("success") is False for item in self._rows):
            raise RuntimeError("cannot append an action after the first failed action")
        copied = dict(row)
        copied.setdefault("action_index", len(self._rows) + 1)
        self._rows.append(copied)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def normalized_date(value: object) -> str | None:
    text = "" if value is None else str(value)
    compact = re.search(r"(?<!\d)(20\d{6})(?!\d)", text)
    if compact:
        return compact.group(1)
    dashed = re.search(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)", text)
    return "".join(dashed.groups()) if dashed else None


def path_receipt(root: Path, relative_path: Path) -> JsonDict:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path.as_posix(), "present": False, "bytes": 0, "sha256": None}
    data = path.read_bytes()
    return {
        "path": relative_path.as_posix(),
        "present": True,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def read_json_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(encoded)


def _progress(phase: str, **details: object) -> None:
    payload = {"phase": phase, **details}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _base_artifact(run_date: str) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [
            {
                "check": "artifact_schema_initialized",
                "expected_value": True,
                "observed_value": True,
                "passed": True,
            }
        ],
        "run_date": run_date,
        "inference_substrate": NO_COMMAND_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "receipt_rows": [],
        "physical_state_receipt": {
            "exists": False,
            "reason": "receipt audit has not run",
            "latest_candidate_date": None,
        },
        "receipt_cutoff_experiment": {
            "experiment": "Exp6559",
            "path": EXP6559_REL_PATH.as_posix(),
            "run_date": None,
            "sha256": None,
        },
        "receipt_newer_than_exp6559_score": 0.0,
        "command_rows": [],
        "hardware_command_count": 0,
        "detect_rows": [],
        "identity_rows": [],
        "smoke_rows": [],
        "first_failure_stop_score": 1.0,
        "bitstream_redesigned": False,
        "exclusion_manifest_modified": False,
        "gatemate_terminal_receipt_score": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "artifact_initialization_pending_receipt_audit",
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_artifact_initialized_pending_receipt_audit",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def checkpoint_artifact(artifact: JsonDict, output_path: Path) -> Path:
    """Atomically replace the artifact at one complete decision boundary."""

    return atomic_write_json(output_path, artifact, allow_override=False)


def _refresh_derived(artifact: JsonDict, started: float) -> None:
    rows = artifact["command_rows"]
    artifact["hardware_command_count"] = len(rows)
    artifact["detect_rows"] = [deepcopy(row) for row in rows if row.get("action") == "detect"]
    artifact["smoke_rows"] = [deepcopy(row) for row in rows if row.get("action") == "n16_smoke"]
    first_failure = next(
        (index for index, row in enumerate(rows) if row.get("success") is False), None
    )
    artifact["first_failure_stop_score"] = float(
        first_failure is None or first_failure == len(rows) - 1
    )
    artifact["duration_s"] = round(max(0.0, time.monotonic() - started), 6)
    artifact["reproducibility_checksum"] = _checksum(artifact)


def _write_checkpoint(
    artifact: JsonDict,
    output_path: Path,
    writer: CheckpointWriter,
    started: float,
    phase: str,
) -> None:
    _refresh_derived(artifact, started)
    writer(artifact, output_path)
    _progress(phase, command_count=artifact["hardware_command_count"])


def source_artifact_hashes(root: Path) -> dict[str, JsonDict]:
    paths = (
        EXP3866_REL_PATH,
        EXP6325_REL_PATH,
        EXP6525_REL_PATH,
        EXP6559_REL_PATH,
        EXCLUSION_REL_PATH,
        SMOKE_BITSTREAM_REL_PATH,
        SMOKE_RTL_REL_PATH,
    )
    return {path.as_posix(): path_receipt(root, path) for path in paths}


def receipt_cutoff(root: Path) -> JsonDict:
    receipt = path_receipt(root, EXP6559_REL_PATH)
    source = read_json_object(root / EXP6559_REL_PATH)
    run_date = normalized_date(source.get("run_date"))
    return {
        "experiment": "Exp6559",
        "path": EXP6559_REL_PATH.as_posix(),
        "run_date": run_date,
        "sha256": receipt["sha256"],
        "present": receipt["present"],
        "honest_verdict": source.get("honest_verdict"),
    }


def _changed_fields(candidate: Mapping[str, Any]) -> list[str]:
    fields = {str(item).lower() for item in candidate.get("changed_physical_fields", [])}
    for item in candidate.get("changes", []):
        if isinstance(item, Mapping):
            fields.add(str(item.get("field") or "").lower())
    return sorted(field for field in fields if field)


def receipt_row(
    candidate: Mapping[str, Any],
    *,
    source_path: str,
    row_index: int,
    cutoff_date: str,
    run_date: str,
    structured: bool,
) -> JsonDict:
    raw = dict(candidate)
    source = str(raw.get("source") or "")
    blocked_author_markers = ("agent-authored", "agent plan", "planner")
    explicit_operator = raw.get("operator_authored") is True
    sourced_operator = source.lower().startswith("operator directive")
    operator_authored = (explicit_operator or sourced_operator) and not any(
        marker in source.lower() for marker in blocked_author_markers
    )
    date = normalized_date(raw.get("receipt_date")) or normalized_date(source)
    board = str(raw.get("board") or "")
    changed_fields = _changed_fields(raw)
    reject_reason: str | None = None
    if not structured:
        reject_reason = "not_structured_receipt"
    elif not operator_authored:
        reject_reason = "not_operator_authored"
    elif date is None:
        reject_reason = "undated_receipt"
    elif date <= cutoff_date:
        reject_reason = "stale_or_not_newer_than_exp6559"
    elif date > run_date:
        reject_reason = "future_dated_receipt"
    elif board != EXPECTED_BOARD:
        reject_reason = "wrong_board"
    elif raw.get("board_present") is not True:
        reject_reason = "board_not_present"
    elif not str(raw.get("power_state") or raw.get("power") or "").strip():
        reject_reason = "missing_power_state"
    elif not str(
        raw.get("usb_jtag_cable_state") or raw.get("usb_dirtyjtag") or raw.get("cable") or ""
    ).strip():
        reject_reason = "missing_usb_jtag_cable_state"
    elif not str(raw.get("host_path") or "").strip():
        reject_reason = "missing_host_path"
    elif (
        str(raw.get("intended_recovery_action") or raw.get("action") or "")
        not in VALID_RECOVERY_ACTIONS
    ):
        reject_reason = "invalid_recovery_action"
    elif not PHYSICAL_CHANGE_FIELDS.intersection(changed_fields):
        reject_reason = "no_changed_physical_state"
    return {
        "row_id": f"receipt-{row_index:03d}",
        "source_path": source_path,
        "structured_receipt": structured,
        "receipt_date": date,
        "operator_authored": operator_authored,
        "board": board or None,
        "board_present": raw.get("board_present") is True,
        "power_state": raw.get("power_state") or raw.get("power"),
        "usb_jtag_cable_state": raw.get("usb_jtag_cable_state")
        or raw.get("usb_dirtyjtag")
        or raw.get("cable"),
        "host_path": raw.get("host_path"),
        "intended_recovery_action": raw.get("intended_recovery_action") or raw.get("action"),
        "changed_physical_fields": changed_fields,
        "valid": reject_reason is None,
        "reject_reason": reject_reason,
        "raw_receipt": raw,
    }


def _markdown_sections(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^#{1,6}\s+", text)]
    if not starts:
        return [text] if text.strip() else []
    bounds = [*starts, len(text)]
    return [text[bounds[index] : bounds[index + 1]].strip() for index in range(len(starts))]


def _json_receipts(section: str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for match in re.finditer(r"```json\s*(\{.*?\})\s*```", section, flags=re.DOTALL):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def audit_receipts(
    root: Path,
    *,
    cutoff_date: str,
    run_date: str,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    dry_run: bool = True,
) -> tuple[list[JsonDict], JsonDict]:
    """Parse receipt evidence without exposing any command execution path."""

    if not dry_run:
        raise ValueError("receipt audit must run in dry_run mode")
    source_rows: list[tuple[str, Mapping[str, Any], bool]] = []
    if candidates is not None:
        source_rows.extend(("caller_supplied", candidate, True) for candidate in candidates)
    else:
        for relative_path in RECEIPT_PATHS:
            path = root / relative_path
            if not path.is_file():
                continue
            for section in _markdown_sections(path.read_text(encoding="utf-8")):
                if not any(
                    token in section.lower() for token in ("gatemate", "dirtyjtag", "1209:c0ca")
                ):
                    continue
                parsed = _json_receipts(section)
                if parsed:
                    source_rows.extend((relative_path.as_posix(), item, True) for item in parsed)
                else:
                    source_rows.append((relative_path.as_posix(), {"source": section}, False))
    rows = [
        receipt_row(
            candidate,
            source_path=source_path,
            row_index=index,
            cutoff_date=cutoff_date,
            run_date=run_date,
            structured=structured,
        )
        for index, (source_path, candidate, structured) in enumerate(source_rows, start=1)
    ]
    valid = [row for row in rows if row["valid"]]
    dates = [row["receipt_date"] for row in rows if row["receipt_date"]]
    if not valid:
        return rows, {
            "exists": False,
            "reason": "no complete operator-authored physical-state receipt newer than Exp6559",
            "latest_candidate_date": max(dates) if dates else None,
        }
    selected = max(valid, key=lambda row: (str(row["receipt_date"]), str(row["row_id"])))
    return rows, {"exists": True, **deepcopy(selected)}


def binary_identity(name: str) -> JsonDict:
    path_text = shutil.which(name)
    if path_text is None:
        return {"present": False, "path": None, "sha256": None}
    path = Path(path_text)
    return {
        "present": path.is_file(),
        "path": str(path),
        "sha256": sha256_bytes(path.read_bytes()) if path.is_file() else None,
    }


def smoke_command(root: Path) -> tuple[str, ...]:
    return (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        str((root / SMOKE_BITSTREAM_REL_PATH).resolve()),
    )


def smoke_source_receipt(root: Path) -> JsonDict:
    historical = read_json_object(root / EXP3866_REL_PATH)
    bitstream = path_receipt(root, SMOKE_BITSTREAM_REL_PATH)
    expected = str(historical.get("bitstream_sha256") or "")
    if expected and not expected.startswith("sha256:"):
        expected = "sha256:" + expected
    valid = bool(bitstream["present"] and expected and bitstream["sha256"] == expected)
    return {
        "valid": valid,
        "bitstream_path": SMOKE_BITSTREAM_REL_PATH.as_posix(),
        "bitstream_sha256": bitstream["sha256"],
        "expected_sha256_from_exp3866": expected or None,
        "constraints_policy": "reuse packed Exp3866 bitstream; no synthesis, place, route, or pack",
    }


def parse_identity(stdout: str, stderr: str) -> JsonDict:
    text = f"{stdout}\n{stderr}"
    idcodes = [value.lower() for value in re.findall(r"\bidcode\s+(0x[0-9a-fA-F]+)", text)]
    lowered = text.lower()
    return {
        "expected_idcode": EXPECTED_IDCODE,
        "observed_idcodes": idcodes,
        "device_count": len(idcodes),
        "gatemate_series_seen": "gatemate series" in lowered,
        "gm1ax_seen": "gm1ax" in lowered,
        "colognechip_seen": "colognechip" in lowered,
        "clean_expected_identity": (
            idcodes == [EXPECTED_IDCODE]
            and "gatemate series" in lowered
            and "gm1ax" in lowered
            and "colognechip" in lowered
        ),
    }


def run_hardware_command(
    argv: tuple[str, ...], timeout_s: float
) -> CommandResult:  # pragma: no cover
    """Run one production hardware command with raw output and a timeout."""

    started = time.monotonic()
    try:
        result = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return CommandResult(
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timeout=False,
            duration_s=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            return_code=None,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            timeout=True,
            duration_s=time.monotonic() - started,
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _action_row(
    action: str,
    argv: tuple[str, ...],
    timeout_s: float,
    command_runner: CommandRunner,
    now: Callable[[], str],
    physical_receipt: Mapping[str, Any],
    *,
    bitstream_sha256: str | None = None,
) -> tuple[JsonDict, JsonDict | None]:
    started_at = now()
    result = command_runner(argv, timeout_s)
    ended_at = now()
    identity = parse_identity(result.stdout, result.stderr) if action == "detect" else None
    if result.timeout:
        failure = f"{action}_timeout"
    elif result.return_code != 0:
        failure = f"{action}_return_code"
    elif action == "detect" and not identity["clean_expected_identity"]:
        failure = "unexpected_gatemate_identity"
    else:
        failure = None
    row: JsonDict = {
        "action": action,
        "argv": list(argv),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
        "timeout": result.timeout,
        "timeout_s": timeout_s,
        "duration_s": round(max(0.0, result.duration_s), 6),
        "usb_jtag_identity": {
            "expected": EXPECTED_USB_JTAG_IDENTITY,
            "receipt_cable_state": physical_receipt.get("usb_jtag_cable_state"),
            "receipt_host_path": physical_receipt.get("host_path"),
        },
        "success": failure is None,
        "failure_reason": failure,
    }
    if bitstream_sha256 is not None:
        row["bitstream_sha256"] = bitstream_sha256
    return row, identity


def _terminal_gate(
    failed_check: str | None,
    expected: object,
    observed: object,
    *,
    cutoff_date: str | None = None,
) -> JsonDict:
    gate: JsonDict = {
        "failed_check": failed_check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": failed_check is None,
    }
    if failed_check == "receipt_newer_than_exp6559":
        gate["receipt_cutoff_experiment"] = "Exp6559"
        gate["receipt_cutoff_date"] = cutoff_date
    return gate


def _finish(
    artifact: JsonDict,
    *,
    verdict_class: str,
    honest_verdict: str,
    gate: JsonDict,
) -> None:
    artifact["verdict_class"] = verdict_class
    artifact["honest_verdict"] = honest_verdict
    artifact["gate_check_summary"] = gate
    artifact["gatemate_terminal_receipt_score"] = 1.0


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
    receipt_candidates: Sequence[Mapping[str, Any]] | None = None,
    command_runner: CommandRunner = run_hardware_command,
    checkpoint_writer: CheckpointWriter = checkpoint_artifact,
    tool_identity: Mapping[str, Any] | None = None,
    utc_now: Callable[[], str] = utc_now,
) -> JsonDict:
    """Build and checkpoint one terminal GateMate continuity artifact."""

    started = time.monotonic()
    root = Path(root).resolve()
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = root / output_path
    artifact = _base_artifact(run_date)
    _write_checkpoint(artifact, output_path, checkpoint_writer, started, "artifact_initialized")

    artifact["source_artifact_hashes"] = source_artifact_hashes(root)
    cutoff = receipt_cutoff(root)
    artifact["receipt_cutoff_experiment"] = cutoff
    cutoff_date = cutoff["run_date"]
    cutoff_ok = bool(cutoff["present"] and cutoff_date)
    artifact["preconditions_checked"].append(
        {
            "check": "exp6559_cutoff_available",
            "expected_value": True,
            "observed_value": cutoff_ok,
            "passed": cutoff_ok,
        }
    )
    _write_checkpoint(artifact, output_path, checkpoint_writer, started, "cutoff_checked")
    if not cutoff_ok:
        _finish(
            artifact,
            verdict_class="blocked",
            honest_verdict="blocked_exp6559_receipt_cutoff_unavailable",
            gate=_terminal_gate("exp6559_cutoff_available", True, cutoff_ok),
        )
        _write_checkpoint(artifact, output_path, checkpoint_writer, started, "terminal_block")
        return artifact

    receipt_rows, selected = audit_receipts(
        root,
        cutoff_date=str(cutoff_date),
        run_date=run_date,
        candidates=receipt_candidates,
    )
    artifact["receipt_rows"] = receipt_rows
    artifact["physical_state_receipt"] = selected
    receipt_ok = bool(selected["exists"])
    artifact["receipt_newer_than_exp6559_score"] = float(receipt_ok)
    artifact["preconditions_checked"].append(
        {
            "check": "receipt_newer_than_exp6559",
            "expected_value": 1.0,
            "observed_value": float(receipt_ok),
            "passed": receipt_ok,
        }
    )
    _write_checkpoint(artifact, output_path, checkpoint_writer, started, "receipt_audit_complete")
    if not receipt_ok:
        _finish(
            artifact,
            verdict_class="blocked",
            honest_verdict="blocked_no_new_operator_physical_state_receipt_after_exp6559",
            gate=_terminal_gate(
                "receipt_newer_than_exp6559", 1.0, 0.0, cutoff_date=str(cutoff_date)
            ),
        )
        _write_checkpoint(
            artifact, output_path, checkpoint_writer, started, "terminal_zero_command"
        )
        return artifact

    resolved_tool = (
        dict(tool_identity) if tool_identity is not None else binary_identity("openFPGALoader")
    )
    tool_ok = resolved_tool.get("present") is True
    artifact["preconditions_checked"].append(
        {
            "check": "openfpgaloader_available",
            "expected_value": True,
            "observed_value": tool_ok,
            "passed": tool_ok,
            "tool_identity": resolved_tool,
        }
    )
    smoke_source = smoke_source_receipt(root)
    artifact["preconditions_checked"].append(
        {
            "check": "existing_n16_smoke_source_unchanged",
            "expected_value": True,
            "observed_value": smoke_source["valid"],
            "passed": smoke_source["valid"],
            "source_receipt": smoke_source,
        }
    )
    _write_checkpoint(
        artifact, output_path, checkpoint_writer, started, "tool_and_smoke_source_checked"
    )
    if not tool_ok:
        _finish(
            artifact,
            verdict_class="blocked",
            honest_verdict="blocked_openfpgaloader_unavailable",
            gate=_terminal_gate("openfpgaloader_available", True, tool_ok),
        )
        _write_checkpoint(artifact, output_path, checkpoint_writer, started, "terminal_tool_block")
        return artifact

    ledger = ActionLedger()
    artifact["execution_venue"] = "gatemate"
    artifact["inference_substrate"] = HARDWARE_SUBSTRATE
    artifact["inference_substrate_class"] = "no_model_load"
    detect, identity = _action_row(
        "detect",
        DETECT_COMMAND,
        DETECT_TIMEOUT_S,
        command_runner,
        utc_now,
        selected,
    )
    ledger.append(detect)
    artifact["command_rows"] = ledger.rows
    artifact["identity_rows"] = [identity]
    _write_checkpoint(artifact, output_path, checkpoint_writer, started, "detect_complete")
    if not detect["success"]:
        _finish(
            artifact,
            verdict_class="partial",
            honest_verdict="partial_detect_action_failed_or_identity_unclean",
            gate=_terminal_gate("detect_clean_expected_identity", True, False),
        )
        _write_checkpoint(
            artifact, output_path, checkpoint_writer, started, "terminal_detect_failure"
        )
        return artifact

    if not smoke_source["valid"]:
        _finish(
            artifact,
            verdict_class="partial",
            honest_verdict="partial_detect_clean_existing_n16_smoke_source_invalid",
            gate=_terminal_gate("existing_n16_smoke_source_unchanged", True, False),
        )
        _write_checkpoint(
            artifact, output_path, checkpoint_writer, started, "terminal_source_block"
        )
        return artifact

    smoke, _ = _action_row(
        "n16_smoke",
        smoke_command(root),
        SMOKE_TIMEOUT_S,
        command_runner,
        utc_now,
        selected,
        bitstream_sha256=str(smoke_source["bitstream_sha256"]),
    )
    ledger.append(smoke)
    artifact["command_rows"] = ledger.rows
    _write_checkpoint(artifact, output_path, checkpoint_writer, started, "n16_smoke_complete")
    if not smoke["success"]:
        _finish(
            artifact,
            verdict_class="partial",
            honest_verdict="partial_existing_n16_smoke_failed",
            gate=_terminal_gate("existing_n16_smoke_succeeded", True, False),
        )
    else:
        _finish(
            artifact,
            verdict_class="positive",
            honest_verdict="positive_gatemate_continuity_detect_and_existing_n16_smoke_complete",
            gate=_terminal_gate(None, 1.0, 1.0),
        )
    _write_checkpoint(artifact, output_path, checkpoint_writer, started, "terminal_continuity")
    return artifact


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    if isinstance(value, Path):
        artifact = read_json_object(value)
    else:
        artifact = dict(value)
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing required fields: " + ", ".join(missing)]
    errors: list[str] = []
    if artifact["schema"] != SCHEMA:
        errors.append("schema mismatch")
    if artifact["spec_refs"] != list(SPEC_REFS):
        errors.append("spec_refs mismatch")
    if artifact["run_date"] != RUN_DATE:
        errors.append("run_date mismatch")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed mismatch")
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or any(
        not str(principles.get(field) or "").strip() for field in PROMPT_REQUIRED_FIELDS
    ):
        errors.append("field_principles missing required entries")
    rows = artifact["command_rows"]
    if not isinstance(rows, list):
        errors.append("command_rows must be a list")
        rows = []
    if artifact["hardware_command_count"] != len(rows):
        errors.append("hardware_command_count must equal command_rows length")
    if len(rows) > 2:
        errors.append("command ledger exceeds detect plus one smoke")
    if rows and rows[0].get("argv") != list(DETECT_COMMAND):
        errors.append("first action is not the exact detect command")
    if rows and rows[0].get("action") != "detect":
        errors.append("first action is not detect")
    if len(rows) == 2:
        second_argv = rows[1].get("argv")
        fixed_prefix = ["openFPGALoader", "-c", "dirtyJtag", "-b", "olimex_gatemateevb"]
        if (
            rows[1].get("action") != "n16_smoke"
            or not isinstance(second_argv, list)
            or second_argv[:5] != fixed_prefix
            or not str(second_argv[-1]).endswith(SMOKE_BITSTREAM_REL_PATH.as_posix())
        ):
            errors.append("second action is not the fixed n16 smoke command")
        identities = artifact["identity_rows"]
        if not identities or identities[0].get("clean_expected_identity") is not True:
            errors.append("smoke lacks a clean expected detect identity")
    first_failure = next(
        (index for index, row in enumerate(rows) if row.get("success") is False), None
    )
    expected_stop = float(first_failure is None or first_failure == len(rows) - 1)
    if artifact["first_failure_stop_score"] != expected_stop or expected_stop != 1.0:
        errors.append("first_failure_stop_score mismatch")
    expected_detect = [row for row in rows if row.get("action") == "detect"]
    expected_smoke = [row for row in rows if row.get("action") == "n16_smoke"]
    if artifact["detect_rows"] != expected_detect:
        errors.append("detect_rows mismatch")
    if artifact["smoke_rows"] != expected_smoke:
        errors.append("smoke_rows mismatch")
    count = len(rows)
    if count == 0:
        if artifact["inference_substrate"] != NO_COMMAND_SUBSTRATE:
            errors.append("zero-command inference_substrate mismatch")
        if artifact["inference_substrate_class"] != "blocked_no_run":
            errors.append("zero-command inference_substrate_class mismatch")
        if artifact["execution_venue"] != "host":
            errors.append("zero-command execution_venue mismatch")
    else:
        if artifact["receipt_newer_than_exp6559_score"] != 1.0:
            errors.append("hardware action lacks a valid new receipt")
        if artifact["inference_substrate"] != HARDWARE_SUBSTRATE:
            errors.append("hardware inference_substrate mismatch")
        if artifact["inference_substrate_class"] != "no_model_load":
            errors.append("hardware inference_substrate_class mismatch")
        if artifact["execution_venue"] != "gatemate":
            errors.append("hardware execution_venue mismatch")
    if artifact["bitstream_redesigned"] is not False:
        errors.append("bitstream_redesigned must be false")
    if artifact["exclusion_manifest_modified"] is not False:
        errors.append("exclusion_manifest_modified must be false")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact["gatemate_terminal_receipt_score"] != 1.0:
        errors.append("gatemate_terminal_receipt_score mismatch")
    verdict_class = artifact["verdict_class"]
    prefixes = {
        "positive": "positive_",
        "circular_positive": "circular_positive_",
        "null": "null_",
        "blocked": "blocked_",
        "disqualified": "disqualified_",
        "partial": "partial_",
    }
    if verdict_class not in prefixes or not str(artifact["honest_verdict"]).startswith(
        prefixes.get(verdict_class, "")
    ):
        errors.append("honest_verdict is inconsistent with verdict_class")
    if str(artifact["honest_verdict"]).startswith("blocked_"):
        gate = artifact["gate_check_summary"]
        if not isinstance(gate, Mapping) or not {
            "failed_check",
            "expected_value",
            "observed_value",
        }.issubset(gate):
            errors.append("blocked gate_check_summary is incomplete")
    cutoff = artifact["receipt_cutoff_experiment"]
    if not isinstance(cutoff, Mapping) or cutoff.get("experiment") != "Exp6559":
        errors.append("receipt_cutoff_experiment mismatch")
    if artifact["reproducibility_checksum"] != _checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=find_repo_root())
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(args.validate)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    if args.date != RUN_DATE:
        print(f"execution date must be {RUN_DATE}", file=sys.stderr)
        return 2
    output = args.output if args.output.is_absolute() else args.root / args.output
    artifact = build_artifact(args.root, args.date, output_path=output)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
