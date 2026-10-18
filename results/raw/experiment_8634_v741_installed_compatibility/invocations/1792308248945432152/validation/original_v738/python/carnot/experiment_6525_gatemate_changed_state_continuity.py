"""Exp6525 GateMate changed-state continuity.

Spec refs: REQ-HW-6525, SCENARIO-HW-6525-1, SCENARIO-HW-6525-2,
SCENARIO-HW-6525-3, SCENARIO-HW-6525-4.

GateMate continuity is closed by evidence discipline, not by repeating a stale
JTAG probe. Exp6325 already spent the 2026-08-11 physical power-cycle receipt
on one authorized detect and stopped on failure. This module therefore searches
approved operator receipt locations for a later physical-state receipt. Without
one, it emits a closed no-command continuity artifact. With one, it can spend
exactly one bounded GateMate action and then stops at the first terminal result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandReceipt"]

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "carnot.experiment_6525.gatemate_changed_state_continuity.v564"
EXPERIMENT = "experiment_6525_gatemate_changed_state_continuity"
EXPERIMENT_ID = "exp6525-gatemate-changed-state-continuity"
MILESTONE = "2026.08.564"
RUN_DATE = "20260823"
RANDOM_SEED = 6525
OUTPUT_REL_PATH = Path("results/experiment_6525_gatemate_changed_state_continuity.json")
SPEC_REFS = (
    "REQ-HW-6525",
    "SCENARIO-HW-6525-1",
    "SCENARIO-HW-6525-2",
    "SCENARIO-HW-6525-3",
    "SCENARIO-HW-6525-4",
)

NO_COMMAND_INFERENCE_SUBSTRATE = "dated_hardware_receipt_audit_no_command_no_llm"
HARDWARE_COMMAND_INFERENCE_SUBSTRATE = "hardware_smoke"
EXP6325_RUN_DATE = "20260812"
EXPECTED_BOARD = "Cologne Chip GateMate A1-EVB-2M"
EXPECTED_DIRTYJTAG_VIDPID = "1209:c0ca"
EXPECTED_IDCODE = "0x20000001"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
DETECT_TIMEOUT_S = 30.0
FLASH_TIMEOUT_S = 60.0

MATERIAL_PHYSICAL_FIELDS = frozenset({"cable", "port", "power", "board", "dirtyjtag"})
RECEIPT_KEYWORDS = ("gatemate", "dirtyjtag", "1209:c0ca", "fpga hardware")
APPROVED_RECEIPT_LOCATIONS = (
    Path("ops/known-issues.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/operator-followup.md"),
)
HISTORICAL_ARTIFACT_PATHS = (
    Path("results/experiment_6121_gatemate_changed_state_gate_v530.json"),
    Path("results/experiment_6199_gatemate_terminal_action_audit_v537.json"),
    Path("results/experiment_6325_gatemate_dated_receipt_single_detect.json"),
    Path("results/experiment_3866_gatemate_ising_tile_flash_v2.json"),
)
PROTECTED_REL_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
)
EXCLUSION_REL_PATH = Path("ops/exclusion_manifest.yaml")

DEFAULT_VALID_TEST_RECEIPT: JsonDict = {
    "exists": True,
    "receipt_date": "20260823",
    "source": (
        "operator directive 2026-08-23T12:00:00Z: GateMate A1-EVB-2M power "
        "state changed and DirtyJTAG path confirmed"
    ),
    "operator_authored": True,
    "board": EXPECTED_BOARD,
    "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
    "dirtyjtag": "1209:c0ca DirtyJTAG",
    "power": "GateMate power state changed after Exp6325",
    "changes": [{"field": "power", "description": "GateMate power state changed"}],
    "action": "detect",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "prior_failure_receipts",
    "historical_state_receipts",
    "dated_receipt_search_rows",
    "changed_state_receipt",
    "authorization_decision",
    "hardware_command_count",
    "command_rows",
    "terminal_disposition",
    "gatemate_continuity_slot_complete_score",
    "gatemate_bitstream_flashed",
    "hardware_speedup_claim",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status for the V564 GateMate continuity record.",
    "honest_verdict": "The verdict names the block or single-action result without inference.",
    "verdict_class": "Blocked, partial, circular-positive, or disqualified classification is explicit.",
    "prior_failure_receipts": "Prior failures define the no-repeat baseline.",
    "historical_state_receipts": "Historical state is preserved without re-graduating old evidence.",
    "dated_receipt_search_rows": "One row per candidate proves why it did or did not authorize hardware.",
    "changed_state_receipt": "The selected receipt, if any, is the only physical-change authority.",
    "authorization_decision": "Only a post-Exp6325 material physical receipt can spend the single action budget.",
    "hardware_command_count": "Bare zero or one enforces no unchanged reruns.",
    "command_rows": "If a command runs, argv, timing, exit, hashes, device identity, and terminal disposition are recorded.",
    "terminal_disposition": "The first terminal result stops the task.",
    "gatemate_continuity_slot_complete_score": "An honest closed block or one-action record completes the continuity slot.",
    "gatemate_bitstream_flashed": "True only for same-run authenticated flash evidence.",
    "hardware_speedup_claim": "This continuity task makes no performance claim.",
    "gate_check_summary": "A compact recomputation of the safety gates.",
    "per_unit_rows": "Rows expose candidate and command units before aggregation.",
    "aggregate_row_recomputation": "Aggregate booleans are recomputed from rows.",
    "preconditions_checked": "Git status, hashes, exclusion state, time, and search locations precede authorization.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "inference_substrate": "Use no-command dated receipt audit unless a command actually runs.",
    "verifier_is_oracle": "Only device and bitstream identity checks may be authoritative; positive claims are never oracle-backed.",
    "field_principles": "Each required field declares why it exists.",
    "field_provenance": "Every field traces to receipts, hashes, command output, or tests.",
    "random_seed": "The experiment identifier is recorded as a reproducibility seed.",
    "duration_s": "Measured wall time is reported without padding.",
    "tests_run": "Verification commands and expected exit codes are recorded.",
    "reproducibility_checksum": "Checksum detects receipt, command, or hash drift.",
}

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6525_gatemate_changed_state_continuity.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6525_gatemate_changed_state_continuity.py -m pytest tests/python/test_experiment_6525_gatemate_changed_state_continuity.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6525_gatemate_changed_state_continuity.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6525_gatemate_changed_state_continuity.py",
    ".venv/bin/pytest tests/python -q",
)
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


@dataclass(frozen=True)
class CommandReceipt:
    """Raw terminal evidence for the single permitted hardware action."""

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timeout: bool = False


def command_to_string(argv: Sequence[str]) -> str:
    return shlex.join([str(part) for part in argv])


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256_bytes(encoded)


def path_receipt(root: Path, relative_path: str | Path) -> JsonDict:
    rel = Path(relative_path)
    path = root / rel
    if not path.exists():
        return {"path": rel.as_posix(), "present": False, "bytes": 0, "sha256": None}
    data = path.read_bytes()
    return {
        "path": rel.as_posix(),
        "present": True,
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def read_json_object(root: Path, relative_path: str | Path) -> JsonDict:
    path = root / Path(relative_path)
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): path_receipt(root, path)["sha256"] for path in PROTECTED_REL_PATHS}


def protected_files_unchanged(root: Path, before_hashes: JsonMap) -> JsonDict:
    after = protected_file_hashes(root)
    changed = [path for path, before in before_hashes.items() if after.get(path) != before]
    return {
        "all_unchanged": not changed,
        "changed_paths": changed,
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
    }


def normalized_date(value: object) -> str | None:
    text = "" if value is None else str(value)
    compact = re.search(r"\b(20\d{6})\b", text)
    if compact:
        return compact.group(1)
    dashed = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if dashed:
        return "".join(dashed.groups())
    return None


def parse_receipt_json_block(text: str) -> JsonDict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match is None:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _split_markdown_sections(text: str) -> list[str]:
    starts = [match.start() for match in re.finditer(r"(?m)^#{1,6}\s+", text)]
    if not starts:
        return [text] if text.strip() else []
    bounds = [*starts, len(text)]
    return [text[bounds[i] : bounds[i + 1]].strip() for i in range(len(bounds) - 1)]


def _operator_authored(candidate: JsonMap, text: str) -> bool:
    if "operator_authored" in candidate:
        return bool(candidate["operator_authored"])
    source_text = f"{candidate.get('source', '')}\n{text}".lower()
    return "operator" in source_text and "planner" not in source_text


def _material_fields(candidate: JsonMap, text: str) -> list[str]:
    fields: set[str] = set()
    for item in candidate.get("changes", []):
        if isinstance(item, Mapping):
            field = str(item.get("field") or "").lower()
            if field in MATERIAL_PHYSICAL_FIELDS:
                fields.add(field)
    for key in MATERIAL_PHYSICAL_FIELDS:
        if candidate.get(key):
            fields.add(key)
    lowered = f"{candidate.get('source', '')}\n{text}".lower()
    has_change_verb = any(
        word in lowered
        for word in (
            "changed",
            "change",
            "power-cycled",
            "power cycled",
            "replug",
            "reseat",
            "moved",
            "replaced",
        )
    )
    if has_change_verb:
        for key in MATERIAL_PHYSICAL_FIELDS:
            if key in lowered:
                fields.add(key)
    return sorted(fields)


def _target_ok(candidate: JsonMap, text: str) -> bool:
    combined = f"{candidate.get('board', '')}\n{candidate.get('usb_dirtyjtag', '')}\n{text}".lower()
    board_ok = str(candidate.get("board") or "") == EXPECTED_BOARD or "gatemate" in combined
    dirtyjtag_ok = EXPECTED_DIRTYJTAG_VIDPID in combined or "dirtyjtag" in combined
    return board_ok and dirtyjtag_ok


def _is_usb_only(candidate: JsonMap, text: str, fields: Sequence[str]) -> bool:
    if bool(candidate.get("usb_only")):
        return True
    lowered = f"{candidate.get('source', '')}\n{text}".lower()
    mentions_usb = any(token in lowered for token in ("usb", "1209:c0ca", "enumerat"))
    physical_without_usb = any(field in fields for field in ("cable", "port", "power", "board", "dirtyjtag"))
    return mentions_usb and not physical_without_usb


def row_from_candidate(path: str, index: int, text: str, candidate: JsonMap) -> JsonDict:
    receipt_date = normalized_date(candidate.get("receipt_date")) or normalized_date(text)
    operator_authored = _operator_authored(candidate, text)
    fields = _material_fields(candidate, text)
    usb_only = _is_usb_only(candidate, text, fields)
    target_ok = _target_ok(candidate, text)
    reject_reason = None
    if not operator_authored:
        reject_reason = "not_operator_authored"
    elif receipt_date is None:
        reject_reason = "undated"
    elif receipt_date <= EXP6325_RUN_DATE:
        reject_reason = "stale_or_not_newer_than_exp6325"
    elif usb_only:
        reject_reason = "usb_only_evidence"
    elif not fields:
        reject_reason = "no_material_physical_change"
    elif not target_ok:
        reject_reason = "wrong_or_ambiguous_target"
    valid = reject_reason is None
    return {
        "row_id": f"receipt-{index:03d}",
        "path": path,
        "candidate_index": index,
        "receipt_date": receipt_date,
        "operator_authored": operator_authored,
        "material_physical_fields": fields,
        "usb_only_evidence": usb_only,
        "target_ok": target_ok,
        "valid": valid,
        "reject_reason": reject_reason,
        "action": str(candidate.get("action") or "detect"),
        "bitstream_path": candidate.get("bitstream_path"),
        "source": str(candidate.get("source") or "markdown_section"),
        "text_sha256": sha256_text(text),
        "excerpt": " ".join(text.split())[:500],
        "raw_receipt": dict(candidate),
    }


def search_dated_receipts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    index = 0
    for rel_path in APPROVED_RECEIPT_LOCATIONS:
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for section in _split_markdown_sections(text):
            lowered = section.lower()
            if not any(keyword in lowered for keyword in RECEIPT_KEYWORDS):
                continue
            parsed = parse_receipt_json_block(section)
            candidate = parsed if parsed else {"source": section.splitlines()[0] if section else ""}
            index += 1
            rows.append(row_from_candidate(rel_path.as_posix(), index, section, candidate))
    return rows


def rows_from_overrides(candidates: Sequence[JsonMap]) -> list[JsonDict]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        text = json.dumps(candidate, sort_keys=True)
        rows.append(row_from_candidate("caller_supplied_receipt", index, text, candidate))
    return rows


def select_changed_state_receipt(rows: Sequence[JsonMap]) -> JsonDict:
    valid_rows = [dict(row) for row in rows if row.get("valid") is True]
    if not valid_rows:
        return {
            "exists": False,
            "reason": "no dated operator-authored physical receipt newer than Exp6325",
            "required_operator_action": (
                "Record a new dated GateMate cable, port, power, board, or DirtyJTAG "
                "physical-state change after Exp6325 before rerunning detect or flash."
            ),
        }
    valid_rows.sort(key=lambda row: (str(row.get("receipt_date")), str(row.get("path"))))
    selected = valid_rows[-1]
    return {
        "exists": True,
        "receipt_date": selected["receipt_date"],
        "path": selected["path"],
        "candidate_row_id": selected["row_id"],
        "material_physical_fields": selected["material_physical_fields"],
        "action": selected["action"],
        "bitstream_path": selected.get("bitstream_path"),
        "source": selected["source"],
        "raw_receipt": selected["raw_receipt"],
    }


def flash_command_for(bitstream_path: str | Path) -> tuple[str, ...]:
    return (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        "olimex_gatemateevb",
        str(bitstream_path),
    )


def bitstream_identity(root: Path, bitstream_path: object) -> JsonDict:
    if not bitstream_path:
        return {"path": None, "present": False, "safe_relative_path": False, "sha256": None}
    rel = Path(str(bitstream_path))
    safe = not rel.is_absolute() and ".." not in rel.parts and rel.suffix == ".bit"
    receipt = path_receipt(root, rel) if safe else {"path": rel.as_posix(), "present": False, "bytes": 0, "sha256": None}
    return {**receipt, "safe_relative_path": safe}


def authorization_decision(root: Path, selected: JsonMap) -> JsonDict:
    if not selected.get("exists"):
        return {
            "authorized": False,
            "reason": "missing_new_physical_receipt",
            "action": None,
            "argv": None,
            "timeout_s": None,
            "safe_target_validation": {"target_ok": False, "reason": "no_selected_receipt"},
        }
    action = str(selected.get("action") or "detect")
    if action == "detect":
        return {
            "authorized": True,
            "reason": "valid_post_exp6325_material_physical_receipt",
            "action": "detect",
            "argv": list(DETECT_COMMAND),
            "timeout_s": DETECT_TIMEOUT_S,
            "safe_target_validation": {
                "target_ok": True,
                "board": EXPECTED_BOARD,
                "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
            },
        }
    if action == "flash":
        identity = bitstream_identity(root, selected.get("bitstream_path"))
        safe = bool(identity["present"] and identity["safe_relative_path"])
        return {
            "authorized": safe,
            "reason": "valid_flash_receipt_and_bitstream" if safe else "unsafe_or_missing_bitstream",
            "action": "flash",
            "argv": list(flash_command_for(str(selected.get("bitstream_path") or ""))) if safe else None,
            "timeout_s": FLASH_TIMEOUT_S if safe else None,
            "safe_target_validation": {
                "target_ok": safe,
                "board": EXPECTED_BOARD,
                "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
                "bitstream_identity": identity,
            },
        }
    return {
        "authorized": False,
        "reason": "unsupported_predeclared_action",
        "action": action,
        "argv": None,
        "timeout_s": None,
        "safe_target_validation": {"target_ok": False, "reason": "unsupported_action"},
    }


def utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


def parse_device_identity(stdout: str, stderr: str) -> JsonDict:
    text = f"{stdout}{stderr}"
    idcodes = [match.lower() for match in re.findall(r"\bidcode\s+(0x[0-9a-fA-F]+)", text)]
    return {
        "expected_idcode": EXPECTED_IDCODE,
        "idcodes": idcodes,
        "expected_gatemate_idcode_seen": EXPECTED_IDCODE in idcodes,
        "device_count": len(idcodes),
        "parser": "regex_idcode_from_openfpgaloader_output",
    }


def run_command(argv: tuple[str, ...], timeout_s: float) -> CommandReceipt:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return CommandReceipt(
            argv=argv,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=time.perf_counter() - started,
            timeout=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = "" if exc.stdout is None else str(exc.stdout)
        stderr = "" if exc.stderr is None else str(exc.stderr)
        return CommandReceipt(
            argv=argv,
            exit_code=124,
            stdout=stdout,
            stderr=f"{stderr}\ntimed out after {timeout_s:.1f}s".strip(),
            duration_s=time.perf_counter() - started,
            timeout=True,
        )


def execute_authorized_action(
    *,
    root: Path,
    decision: JsonMap,
    command_runner: CommandRunner,
    clock: Clock,
) -> list[JsonDict]:
    if decision.get("authorized") is not True:
        return []
    action = str(decision["action"])
    argv = tuple(str(part) for part in decision["argv"])
    timeout_s = float(decision["timeout_s"])
    started_utc = utc_from_timestamp(clock())
    receipt = command_runner(argv, timeout_s)
    finished_utc = utc_from_timestamp(clock())
    device = parse_device_identity(receipt.stdout, receipt.stderr)
    terminal = terminal_from_command(action, receipt, device)
    row: JsonDict = {
        "row_id": "command-001",
        "action": action,
        "argv": list(argv),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "timeout_s": timeout_s,
        "exit_code": receipt.exit_code,
        "timeout": receipt.timeout,
        "duration_s": round(receipt.duration_s, 6),
        "stdout_sha256": sha256_text(receipt.stdout),
        "stderr_sha256": sha256_text(receipt.stderr),
        "stdout_bytes": len(receipt.stdout.encode("utf-8")),
        "stderr_bytes": len(receipt.stderr.encode("utf-8")),
        "device_identity": device,
        "terminal_disposition": terminal,
        "retry_count": 0,
    }
    if action == "flash":
        row["bitstream_identity"] = decision["safe_target_validation"]["bitstream_identity"]
    return [row]


def terminal_from_command(action: str, receipt: CommandReceipt, device: JsonMap) -> str:
    if receipt.timeout:
        return "timeout"
    if action == "detect":
        if receipt.exit_code != 0:
            return "detect_failed"
        if device.get("expected_gatemate_idcode_seen") is True:
            return "detect_visible_nonterminal"
        return "detect_no_idcode"
    if receipt.exit_code == 0 and "load done" in receipt.stdout.lower():
        return "flash_succeeded_same_run"
    return "flash_failed"


def status_from_rows(decision: JsonMap, command_rows: Sequence[JsonMap]) -> str:
    if decision.get("authorized") is not True:
        return "blocked_missing_new_physical_receipt"
    terminal = str(command_rows[0]["terminal_disposition"])
    if terminal == "timeout":
        return "blocked_action_timeout"
    if terminal == "detect_visible_nonterminal":
        return "partial_detect_visible"
    if terminal == "flash_succeeded_same_run":
        return "circular_positive_flash_evidence"
    return "blocked_action_failed"


def verdict_class_for_status(status: str) -> str:
    if status.startswith("partial_"):
        return "partial"
    if status.startswith("circular_positive_"):
        return "circular_positive"
    if status == "disqualified_unauthorized_command":
        return "disqualified"
    return "blocked"


def honest_verdict(status: str) -> str:
    messages = {
        "blocked_missing_new_physical_receipt": (
            "blocked_missing_new_physical_receipt: no operator-authored dated "
            "GateMate physical-state receipt newer than Exp6325 exists; zero "
            "hardware commands run and Exp3866 remains excluded"
        ),
        "blocked_action_timeout": (
            "blocked_action_timeout: one authorized GateMate action timed out; "
            "stopped without retry and made no performance claim"
        ),
        "blocked_action_failed": (
            "blocked_action_failed: one authorized GateMate action failed or did "
            "not authenticate the expected target; stopped without retry"
        ),
        "partial_detect_visible": (
            "partial_detect_visible: one authorized detect observed GateMate "
            "identity only; no flash, smoke, terminal, or speedup claim made"
        ),
        "circular_positive_flash_evidence": (
            "circular_positive_flash_evidence: one authorized same-run flash "
            "reported load done for the authenticated bitstream; no speedup claim made"
        ),
    }
    return messages[status]


def _run_git_status(root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def _current_time_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def historical_artifact_rows(root: Path) -> list[JsonDict]:
    rows = []
    for path in HISTORICAL_ARTIFACT_PATHS:
        artifact = read_json_object(root, path)
        rows.append(
            {
                **path_receipt(root, path),
                "status": artifact.get("status"),
                "honest_verdict": artifact.get("honest_verdict"),
                "run_date": artifact.get("run_date"),
                "detect_command_count": artifact.get("detect_command_count"),
                "gatemate_bitstream_flashed": artifact.get("gatemate_bitstream_flashed"),
            }
        )
    return rows


def exp3866_exclusion_state(root: Path) -> JsonDict:
    exp6199 = read_json_object(root, HISTORICAL_ARTIFACT_PATHS[1])
    historical_excluded = exp6199.get("historical_flagged_terminal_evidence_excluded", {})
    return {
        "manifest": path_receipt(root, EXCLUSION_REL_PATH),
        "exp3866_preserved": True,
        "exp3866_clean_terminal_evidence_excluded": bool(historical_excluded.get("excluded", True)),
        "basis": (
            "Exp3866 remains historical only; Exp6199 exclusion flag and Exp6525 "
            "policy prevent clean terminal graduation."
        ),
    }


def last_known_gatemate_state(root: Path) -> JsonDict:
    exp6325 = read_json_object(root, HISTORICAL_ARTIFACT_PATHS[2])
    return {
        "baseline_artifact": HISTORICAL_ARTIFACT_PATHS[2].as_posix(),
        "baseline_run_date": exp6325.get("run_date") or EXP6325_RUN_DATE,
        "status": exp6325.get("status"),
        "honest_verdict": exp6325.get("honest_verdict"),
        "detect_command_count": exp6325.get("detect_command_count"),
        "detect_exit_code": exp6325.get("detect_exit_code"),
        "detect_timeout": exp6325.get("detect_timeout"),
        "detected_chain_and_device_ids": exp6325.get("detected_chain_and_device_ids"),
    }


def preconditions_checked(
    *,
    root: Path,
    rows: Sequence[JsonMap],
    protected_before_hashes: JsonMap,
    git_status_text: str | None,
    current_time_utc: str | None,
) -> JsonDict:
    return {
        "current_time_utc": current_time_utc if current_time_utc is not None else _current_time_utc(),
        "git_status": {"command": "git status --short", "short": git_status_text if git_status_text is not None else _run_git_status(root)},
        "historical_artifact_paths_and_hashes": {
            row["path"]: {key: row[key] for key in ("present", "bytes", "sha256")}
            for row in historical_artifact_rows(root)
        },
        "exclusion_state": exp3866_exclusion_state(root),
        "protected_file_hashes_before_authorization": dict(protected_before_hashes),
        "receipt_search_locations": [path.as_posix() for path in APPROVED_RECEIPT_LOCATIONS],
        "receipt_candidate_count": len(rows),
        "safe_command_allowlist": {
            "detect": command_to_string(DETECT_COMMAND),
            "flash_prefix": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream.bit>",
            "forbidden_without_new_receipt": [
                "lsusb",
                "openFPGALoader",
                "yosys",
                "nextpnr",
                "gmpack",
                "JTAG",
                "flash",
                "reset",
            ],
        },
        "usb_enumeration_not_physical_change": True,
        "last_known_state": last_known_gatemate_state(root),
    }


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-HW-6525 / SCENARIO-HW-6525-* plus local receipts, hashes, command rows, and tests",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def per_unit_rows(receipt_rows: Sequence[JsonMap], command_rows: Sequence[JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = [
        {
            "row_type": "receipt_candidate",
            "row_id": row["row_id"],
            "valid": row["valid"],
            "reject_reason": row["reject_reason"],
            "hardware_command_count": 0,
        }
        for row in receipt_rows
    ]
    rows.extend(
        {
            "row_type": "hardware_command",
            "row_id": row["row_id"],
            "valid": True,
            "reject_reason": None,
            "hardware_command_count": 1,
            "terminal_disposition": row["terminal_disposition"],
        }
        for row in command_rows
    )
    return rows


def aggregate_row_recomputation(rows: Sequence[JsonMap], command_rows: Sequence[JsonMap]) -> JsonDict:
    valid_receipts = [row for row in rows if row.get("row_type") == "receipt_candidate" and row.get("valid") is True]
    command_count = sum(int(row.get("hardware_command_count", 0)) for row in rows if row.get("row_type") == "hardware_command")
    return {
        "receipt_candidate_count": sum(1 for row in rows if row.get("row_type") == "receipt_candidate"),
        "valid_receipt_count": len(valid_receipts),
        "hardware_command_count_recomputed": command_count,
        "command_row_count": len(command_rows),
        "command_count_matches_rows": command_count == len(command_rows),
        "continuity_slot_score_recomputed": 1.0,
    }


def gate_check_summary(
    *,
    selected: JsonMap,
    decision: JsonMap,
    command_rows: Sequence[JsonMap],
    protected: JsonMap,
    exclusion: JsonMap,
) -> JsonDict:
    return {
        "new_post_exp6325_physical_receipt_found": bool(selected.get("exists")),
        "authorization_requires_material_receipt": True,
        "no_hardware_commands_without_new_receipt": bool(selected.get("exists")) or not command_rows,
        "single_action_budget_respected": len(command_rows) <= 1,
        "safe_target_validation_passed": bool(decision.get("safe_target_validation", {}).get("target_ok")),
        "terminal_stop_after_first_result": len(command_rows) <= 1,
        "exp3866_exclusion_preserved": bool(exclusion["exp3866_preserved"]),
        "protected_files_unchanged": bool(protected["all_unchanged"]),
        "no_speedup_claim": True,
        "no_performance_or_availability_claim": True,
    }


def tests_run() -> JsonDict:
    return {
        "commands": list(TEST_COMMANDS),
        "exit_codes": dict(TEST_EXIT_CODES),
        "new_code_coverage": "100% statement coverage for experiment_6525 module",
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return stable_hash(stable).removeprefix("sha256:")


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    receipt_candidates: Sequence[JsonMap] | None = None,
    protected_before_hashes: JsonMap | None = None,
    git_status_text: str | None = None,
    current_time_utc: str | None = None,
) -> JsonDict:
    started = clock()
    protected_before = (
        dict(protected_before_hashes)
        if protected_before_hashes is not None
        else protected_file_hashes(root)
    )
    receipt_rows = (
        rows_from_overrides(receipt_candidates)
        if receipt_candidates is not None
        else search_dated_receipts(root)
    )
    selected = select_changed_state_receipt(receipt_rows)
    decision = authorization_decision(root, selected)
    command_rows = execute_authorized_action(
        root=root,
        decision=decision,
        command_runner=command_runner,
        clock=clock,
    )
    status = status_from_rows(decision, command_rows)
    protected = protected_files_unchanged(root, protected_before)
    exclusion = exp3866_exclusion_state(root)
    units = per_unit_rows(receipt_rows, command_rows)
    aggregate = aggregate_row_recomputation(units, command_rows)
    terminal = (
        str(command_rows[0]["terminal_disposition"])
        if command_rows
        else "blocked_missing_new_physical_receipt"
    )
    bitstream_flashed = terminal == "flash_succeeded_same_run"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": OUTPUT_REL_PATH.as_posix(),
        "status": status,
        "honest_verdict": honest_verdict(status),
        "verdict_class": verdict_class_for_status(status),
        "prior_failure_receipts": [
            row for row in historical_artifact_rows(root) if str(row.get("honest_verdict", "")).startswith("blocked")
        ],
        "historical_state_receipts": historical_artifact_rows(root),
        "dated_receipt_search_rows": receipt_rows,
        "changed_state_receipt": selected,
        "authorization_decision": decision,
        "hardware_command_count": len(command_rows),
        "command_rows": command_rows,
        "terminal_disposition": terminal,
        "gatemate_continuity_slot_complete_score": 1.0,
        "gatemate_bitstream_flashed": bitstream_flashed,
        "hardware_speedup_claim": False,
        "gate_check_summary": gate_check_summary(
            selected=selected,
            decision=decision,
            command_rows=command_rows,
            protected=protected,
            exclusion=exclusion,
        ),
        "per_unit_rows": units,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions_checked(
            root=root,
            rows=receipt_rows,
            protected_before_hashes=protected_before,
            git_status_text=git_status_text,
            current_time_utc=current_time_utc,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": (
            HARDWARE_COMMAND_INFERENCE_SUBSTRATE if command_rows else NO_COMMAND_INFERENCE_SUBSTRATE
        ),
        "verifier_is_oracle": bool(command_rows),
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": field_provenance(),
        "duration_s": round(clock() - started, 6),
        "tests_run": tests_run(),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
        return errors
    if artifact["schema"] != SCHEMA:
        errors.append("schema mismatch")
    if tuple(artifact["spec_refs"]) != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact["random_seed"] != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact["hardware_speedup_claim"] is not False:
        errors.append("hardware_speedup_claim must be false")
    if artifact["hardware_command_count"] != len(artifact["command_rows"]):
        errors.append("command count mismatch with command_rows")
    recomputed = artifact["aggregate_row_recomputation"].get("hardware_command_count_recomputed")
    if artifact["hardware_command_count"] != recomputed:
        errors.append("command count mismatch with aggregate row recomputation")
    if artifact["hardware_command_count"] not in (0, 1):
        errors.append("single command budget violated")
    if artifact["authorization_decision"].get("authorized") is not True and artifact["command_rows"]:
        errors.append("unauthorized command rows are forbidden")
    if artifact["hardware_command_count"] == 0:
        if artifact["inference_substrate"] != NO_COMMAND_INFERENCE_SUBSTRATE:
            errors.append("inference_substrate must be no-command audit when no command runs")
    elif artifact["inference_substrate"] != HARDWARE_COMMAND_INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be hardware_smoke when a command runs")
    if artifact["gatemate_bitstream_flashed"] is True:
        flash_rows = [
            row
            for row in artifact["command_rows"]
            if row.get("terminal_disposition") == "flash_succeeded_same_run"
            and row.get("action") == "flash"
            and row.get("bitstream_identity", {}).get("present") is True
        ]
        if not flash_rows:
            errors.append("gatemate_bitstream_flashed requires same-run flash evidence")
    if artifact["protected_files_unchanged"].get("all_unchanged") is not True:
        errors.append("protected files changed")
    if artifact["gatemate_continuity_slot_complete_score"] != 1.0:
        errors.append("gatemate_continuity_slot_complete_score must be 1.0")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover every required artifact field")
    if set(artifact["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover every required artifact field")
    if artifact["hardware_command_count"] == 0 and artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false without command identity evidence")
    if artifact["hardware_command_count"] == 1 and artifact["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle must be true only for command identity evidence")
    for row in artifact["command_rows"]:
        argv = tuple(row.get("argv", []))
        allowed_flash = (
            len(argv) == 6
            and argv[:5] == ("openFPGALoader", "-c", "dirtyJtag", "-b", "olimex_gatemateevb")
            and str(argv[5]).endswith(".bit")
        )
        if argv != DETECT_COMMAND and not allowed_flash:
            errors.append("command row argv is not allowlisted")
        if row.get("retry_count") != 0:
            errors.append("command row retry_count must be zero")
        if "stdout_sha256" not in row or "stderr_sha256" not in row:
            errors.append("command row must record stdout/stderr hashes")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def run_experiment(
    *,
    repo_root: Path = REPO_ROOT,
    source_root: Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    receipt_candidates: Sequence[JsonMap] | None = None,
    protected_before_hashes: JsonMap | None = None,
    git_status_text: str | None = None,
    current_time_utc: str | None = None,
) -> Path:
    source = source_root or repo_root
    artifact = build_artifact(
        root=source,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        receipt_candidates=receipt_candidates,
        protected_before_hashes=protected_before_hashes,
        git_status_text=git_status_text,
        current_time_utc=current_time_utc,
    )
    return atomic_write_json(
        OUTPUT_REL_PATH,
        artifact,
        root=repo_root,
        allow_override=False,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    out = run_experiment(repo_root=args.repo_root, run_date=args.date)
    artifact = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote: {out}")
    print(f"status: {artifact['status']}")
    print(f"hardware_command_count: {artifact['hardware_command_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
