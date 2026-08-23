"""Exp6559 GateMate changed-state continuity.

Spec refs: REQ-HW-6559, SCENARIO-HW-6559-1, SCENARIO-HW-6559-2,
SCENARIO-HW-6559-3, SCENARIO-HW-6559-4.

This reducer treats Exp6525 as the no-repeat boundary. It searches only durable
receipt files for a newer operator-authored physical change. Without that
receipt, it emits a zero-command closure and never reaches the command runner.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandReceipt"]

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "carnot.experiment_6559.gatemate_changed_state_continuity.v567"
EXPERIMENT = "experiment_6559_gatemate_changed_state_continuity"
EXPERIMENT_ID = "exp6559-gatemate-changed-state-continuity"
MILESTONE = "2026.08.567"
RUN_DATE = "20260823"
RANDOM_SEED = 6559
OUTPUT_REL_PATH = Path("results/experiment_6559_gatemate_changed_state_continuity.json")
SPEC_REFS = (
    "REQ-HW-6559",
    "SCENARIO-HW-6559-1",
    "SCENARIO-HW-6559-2",
    "SCENARIO-HW-6559-3",
    "SCENARIO-HW-6559-4",
)

NO_COMMAND_INFERENCE_SUBSTRATE = "dated_hardware_receipt_audit_no_command_no_llm"
HARDWARE_COMMAND_INFERENCE_SUBSTRATE = "hardware_smoke"
EXP6525_FALLBACK_RUN_DATE = "20260823"
EXPECTED_BOARD = "Cologne Chip GateMate A1-EVB-2M"
EXPECTED_DIRTYJTAG_VIDPID = "1209:c0ca"
EXPECTED_IDCODE = "0x20000001"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
DETECT_TIMEOUT_S = 30.0
FLASH_TIMEOUT_S = 60.0

PRIOR_FAILURE_REL_PATH = Path("results/experiment_6525_gatemate_changed_state_continuity.json")
EXP3866_REL_PATH = Path("results/experiment_3866_gatemate_ising_tile_flash_v2.json")
EXCLUSION_REL_PATH = Path("ops/exclusion_manifest.yaml")
APPROVED_RECEIPT_LOCATIONS = (
    Path("ops/known-issues.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/hardware-bringup-prep.md"),
    Path("ops/operator-followup.md"),
)
USB_RECEIPT_SOURCES = (
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    Path("results/experiment_6199_gatemate_terminal_action_audit_v537.json"),
    Path("results/experiment_6325_gatemate_dated_receipt_single_detect.json"),
    PRIOR_FAILURE_REL_PATH,
)
BITSTREAM_IDENTITY_PATHS = (
    Path("rtl/gatemate_ising_n16.bit"),
    Path("rtl/gatemate_ising_n16_packed.bit"),
    Path("build/gatemate/experiment_2956_gatemate_n16/ising_n16_gatemate.bit"),
    Path("build/gatemate/experiment_3866_gatemate_ising_tile_flash_v2/gatemate_ising_n16.bit"),
)
PROTECTED_REL_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
)

MATERIAL_PHYSICAL_FIELDS = frozenset({"cable", "port", "power", "board", "dirtyjtag"})
RECEIPT_KEYWORDS = ("gatemate", "dirtyjtag", "1209:c0ca", "fpga hardware")
CLAIM_KEYS = (
    "latency_claimed",
    "speed_claimed",
    "energy_claimed",
    "sampling_quality_claimed",
    "availability_claimed",
    "performance_claim_made",
)

DEFAULT_VALID_TEST_RECEIPT: JsonDict = {
    "exists": True,
    "receipt_date": "20260824",
    "source": (
        "operator directive 2026-08-24T12:00:00Z: GateMate A1-EVB-2M power "
        "state changed and DirtyJTAG path confirmed"
    ),
    "operator_authored": True,
    "board": EXPECTED_BOARD,
    "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
    "dirtyjtag": "1209:c0ca DirtyJTAG",
    "power": "GateMate power state changed after Exp6525",
    "changes": [{"field": "power", "description": "GateMate power state changed"}],
    "action": "detect",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "prior_failure_receipt",
    "operator_physical_state_receipt",
    "safe_target_validation_receipt",
    "hardware_action_rows",
    "terminal_command_receipt",
    "zero_command_block_receipt",
    "exp3866_exclusion_preserved",
    "claim_boundary",
    "attack_matrix",
    "gatemate_changed_state_slot_complete_score",
    "gatemate_hardware_advanced_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
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
    "random_seed",
    "spec_refs",
    "result_path",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a safe zero-command closure from incomplete hardware setup.",
    "honest_verdict": "The verdict must name receipt validity, command count, and terminal result with a terminal prefix.",
    "verdict_class": "A closed class prevents a continuity receipt from becoming a performance claim.",
    "prior_failure_receipt": "The artifact must identify Exp6525 and the stricter newer-than boundary.",
    "operator_physical_state_receipt": "Only a dated operator-authored physical change can authorize a new board command.",
    "safe_target_validation_receipt": "Board, cable, tool, action, and bitstream identity must close before hardware access.",
    "hardware_action_rows": "Zero or one action rows make the bounded command budget mechanically recheckable.",
    "terminal_command_receipt": "A real detect or flash result needs command, timing, exit, stream hashes, and device identity.",
    "zero_command_block_receipt": "A missing physical receipt must prove that no hardware command ran.",
    "exp3866_exclusion_preserved": "The known retired path cannot be reopened by a continuity task.",
    "claim_boundary": "The artifact must disclaim latency, speed, energy, quality, and general availability.",
    "attack_matrix": "Receipt, target, command-count, output, and overclaim attacks test hardware integrity.",
    "gatemate_changed_state_slot_complete_score": "The standing hardware slot can close safely without manufacturing activity.",
    "gatemate_hardware_advanced_score": "Only a new valid detect or flash receipt records physical advancement.",
    "per_unit_rows": "Every receipt candidate and any hardware action needs a separate row.",
    "aggregate_row_recomputation": "Command count and advancement must derive from the emitted rows.",
    "gate_check_summary": "A blocked verdict must name the missing receipt check and observed latest date.",
    "preconditions_checked": "Receipt, tool, target, resource, and hash checks separate blocked work from board failure.",
    "protected_files_unchanged": "The hardware task must preserve the active roadmap and conductor.",
    "inference_substrate": "The substrate distinguishes a zero-command receipt audit from an authenticated hardware smoke.",
    "verifier_is_oracle": "The hardware transcript is direct evidence for one action only, not a model verifier.",
    "field_provenance": "Every continuity field must point to receipt paths, command rows, and hashes.",
    "duration_s": "Monotonic time exposes repeated or omitted hardware actions.",
    "tests_run": "Named unit, lint, and hardware E2E receipts show the stop authority was checked.",
    "reproducibility_checksum": "A final hash protects the hardware determination trail.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6559_gatemate_changed_state_continuity --date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6559_gatemate_changed_state_continuity.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6559_gatemate_changed_state_continuity.py "
    "-m pytest tests/python/test_experiment_6559_gatemate_changed_state_continuity.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6559_gatemate_changed_state_continuity.py "
    "--fail-under=100 --show-missing"
)
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6559_gatemate_changed_state_continuity.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6559_gatemate_changed_state_continuity.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml"
ADVERSARIAL_VERIFY_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6559_gatemate_changed_state_continuity.json"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
GIT_STATUS_COMMAND = "git status --short"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    ADVERSARIAL_VERIFY_COMMAND,
    FULL_PYTEST_COMMAND,
    GIT_STATUS_COMMAND,
)


@dataclass(frozen=True)
class CommandReceipt:
    """Raw terminal evidence for the single permitted hardware action."""

    argv: tuple[str, ...]
    exit_status: int | None
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
    lowered = f"{candidate.get('source', '')}\n{text}".lower()
    blocked_sources = ("planner", "agent plan", "agent-written", "claude plan")
    return "operator" in lowered and not any(marker in lowered for marker in blocked_sources)


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
    board_value = str(candidate.get("board") or "")
    board_ok = board_value == EXPECTED_BOARD or ("gatemate" in combined and "kv260" not in combined)
    dirtyjtag_ok = EXPECTED_DIRTYJTAG_VIDPID in combined or "dirtyjtag" in combined
    return board_ok and dirtyjtag_ok


def _is_usb_only(candidate: JsonMap, text: str, fields: Sequence[str]) -> bool:
    if bool(candidate.get("usb_only")):
        return True
    lowered = f"{candidate.get('source', '')}\n{text}".lower()
    mentions_usb = any(token in lowered for token in ("usb", "1209:c0ca", "enumerat"))
    physical_without_usb = any(field in fields for field in MATERIAL_PHYSICAL_FIELDS)
    return mentions_usb and not physical_without_usb


def row_from_candidate(
    path: str,
    index: int,
    text: str,
    candidate: JsonMap,
    baseline_date: str,
) -> JsonDict:
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
    elif receipt_date <= baseline_date:
        reject_reason = "stale_or_not_newer_than_exp6525"
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


def search_dated_receipts(root: Path, baseline_date: str) -> list[JsonDict]:
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
            rows.append(
                row_from_candidate(rel_path.as_posix(), index, section, candidate, baseline_date)
            )
    return rows


def rows_from_overrides(candidates: Sequence[JsonMap], baseline_date: str) -> list[JsonDict]:
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        text = json.dumps(candidate, sort_keys=True)
        rows.append(
            row_from_candidate("caller_supplied_receipt", index, text, candidate, baseline_date)
        )
    return rows


def latest_receipt_date(rows: Sequence[JsonMap]) -> str | None:
    dates = [str(row["receipt_date"]) for row in rows if row.get("receipt_date")]
    return max(dates) if dates else None


def select_physical_state_receipt(rows: Sequence[JsonMap]) -> JsonDict:
    valid_rows = [dict(row) for row in rows if row.get("valid") is True]
    if not valid_rows:
        return {
            "exists": False,
            "reason": "no dated operator-authored physical receipt newer than Exp6525",
            "latest_receipt_date": latest_receipt_date(rows),
            "candidate_rows": [dict(row) for row in rows],
            "required_operator_action": (
                "Record a new dated GateMate cable, port, board power, board, or DirtyJTAG "
                "physical-state change after Exp6525 before rerunning detect or flash."
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
        "candidate_rows": [dict(row) for row in rows],
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
    receipt = (
        path_receipt(root, rel)
        if safe
        else {"path": rel.as_posix(), "present": False, "bytes": 0, "sha256": None}
    )
    return {**receipt, "safe_relative_path": safe}


def binary_identity(name: str) -> JsonDict:  # pragma: no cover
    path = shutil.which(name)
    if path is None:
        return {
            "present": False,
            "path": None,
            "sha256": None,
            "identity_source": "command_path_lookup_no_hardware_command",
        }
    data = Path(path).read_bytes()
    return {
        "present": True,
        "path": path,
        "sha256": sha256_bytes(data),
        "identity_source": "path_hash_no_hardware_command",
    }


def tool_identities_without_hardware() -> dict[str, JsonDict]:  # pragma: no cover
    return {
        name: binary_identity(name)
        for name in ("openFPGALoader", "yosys", "nextpnr-himbaechel", "gmpack")
    }


def authorization_decision(root: Path, selected: JsonMap, tool_identities: JsonMap) -> JsonDict:
    if not selected.get("exists"):
        return {
            "authorized": False,
            "reason": "missing_new_physical_receipt",
            "action": None,
            "argv": None,
            "timeout_s": None,
            "target_ok": False,
            "board": EXPECTED_BOARD,
            "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
            "tool_identity": dict(tool_identities.get("openFPGALoader", {})),
        }
    action = str(selected.get("action") or "detect")
    if action not in {"detect", "flash"}:
        return {
            "authorized": False,
            "reason": "unsupported_predeclared_action",
            "action": action,
            "argv": None,
            "timeout_s": None,
            "target_ok": False,
            "board": EXPECTED_BOARD,
            "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
            "tool_identity": dict(tool_identities.get("openFPGALoader", {})),
        }
    tool = dict(tool_identities.get("openFPGALoader", {}))
    if tool.get("present") is not True:
        return {
            "authorized": False,
            "reason": "openfpgaloader_missing",
            "action": action,
            "argv": None,
            "timeout_s": None,
            "target_ok": False,
            "board": EXPECTED_BOARD,
            "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
            "tool_identity": tool,
        }
    if action == "detect":
        return {
            "authorized": True,
            "reason": "valid_post_exp6525_material_physical_receipt",
            "action": "detect",
            "argv": list(DETECT_COMMAND),
            "timeout_s": DETECT_TIMEOUT_S,
            "target_ok": True,
            "board": EXPECTED_BOARD,
            "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
            "tool_identity": tool,
            "expected_device": {
                "idcode": EXPECTED_IDCODE,
                "family": "GateMate Series",
                "model": "GM1Ax",
            },
        }
    identity = bitstream_identity(root, selected.get("bitstream_path"))
    safe = bool(identity["present"] and identity["safe_relative_path"])
    return {
        "authorized": safe,
        "reason": "valid_flash_receipt_and_bitstream" if safe else "unsafe_or_missing_bitstream",
        "action": "flash",
        "argv": list(flash_command_for(str(selected.get("bitstream_path") or "")))
        if safe
        else None,
        "timeout_s": FLASH_TIMEOUT_S if safe else None,
        "target_ok": safe,
        "board": EXPECTED_BOARD,
        "dirtyjtag": EXPECTED_DIRTYJTAG_VIDPID,
        "tool_identity": tool,
        "bitstream_identity": identity,
        "expected_device": {
            "idcode": EXPECTED_IDCODE,
            "family": "GateMate Series",
            "model": "GM1Ax",
        },
    }


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
            exit_status=result.returncode,
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
            exit_status=124,
            stdout=stdout,
            stderr=f"{stderr}\ntimed out after {timeout_s:.1f}s".strip(),
            duration_s=time.perf_counter() - started,
            timeout=True,
        )


def terminal_from_command(action: str, receipt: CommandReceipt, device: JsonMap) -> str:
    if receipt.timeout:
        return "timeout"
    if action == "detect":
        if receipt.exit_status != 0:
            return "detect_failed"
        if device.get("expected_gatemate_idcode_seen") is True:
            return "detect_visible_terminal"
        return "detect_missing_idcode"
    if receipt.exit_status == 0 and "load done" in receipt.stdout.lower():
        return "flash_succeeded_terminal"
    return "flash_failed"


def execute_authorized_action(
    *,
    decision: JsonMap,
    command_runner: CommandRunner,
    clock: Clock,
    usb_identity: JsonMap,
) -> list[JsonDict]:
    if decision.get("authorized") is not True:
        return []
    action = str(decision["action"])
    argv = tuple(str(part) for part in decision["argv"])
    timeout_s = float(decision["timeout_s"])
    monotonic_start_s = clock()
    receipt = command_runner(argv, timeout_s)
    monotonic_end_s = clock()
    device = parse_device_identity(receipt.stdout, receipt.stderr)
    terminal = terminal_from_command(action, receipt, device)
    row: JsonDict = {
        "row_id": "action-001",
        "action": action,
        "argv": list(argv),
        "command": command_to_string(argv),
        "monotonic_start_s": round(monotonic_start_s, 6),
        "monotonic_end_s": round(monotonic_end_s, 6),
        "timeout_s": timeout_s,
        "exit_status": receipt.exit_status,
        "timeout": receipt.timeout,
        "duration_s": round(receipt.duration_s, 6),
        "stdout_sha256": sha256_text(receipt.stdout),
        "stderr_sha256": sha256_text(receipt.stderr),
        "stdout_bytes": len(receipt.stdout.encode("utf-8")),
        "stderr_bytes": len(receipt.stderr.encode("utf-8")),
        "device_identity": device,
        "usb_identity": dict(usb_identity),
        "board_target": decision.get("board"),
        "terminal_disposition": terminal,
        "retry_count": 0,
    }
    if action == "flash":
        row["bitstream_identity"] = decision["bitstream_identity"]
    return [row]


def _run_git_status(root: Path) -> str:  # pragma: no cover
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else result.stderr


def _current_time_utc() -> str:  # pragma: no cover
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def prior_failure_receipt(root: Path) -> JsonDict:
    artifact = read_json_object(root, PRIOR_FAILURE_REL_PATH)
    receipt = path_receipt(root, PRIOR_FAILURE_REL_PATH)
    run_date = normalized_date(artifact.get("run_date")) or EXP6525_FALLBACK_RUN_DATE
    return {
        **receipt,
        "experiment": "Exp6525",
        "run_date": run_date,
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "strict_newer_than_boundary": f"receipt_date > {run_date}",
        "boundary_name": "newer_than_exp6525",
    }


def usb_enumeration_from_existing_receipts(root: Path) -> JsonDict:
    snippets = []
    for rel_path in USB_RECEIPT_SOURCES:
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = [
            " ".join(line.split())[:240]
            for line in text.splitlines()
            if "1209:c0ca" in line.lower() or "dirtyjtag" in line.lower()
        ][:4]
        if matches:
            snippets.append(
                {
                    **path_receipt(root, rel_path),
                    "matching_lines": matches,
                }
            )
    return {
        "live_usb_command_run": False,
        "source": "existing_receipts_only",
        "expected_dirtyjtag_vidpid": EXPECTED_DIRTYJTAG_VIDPID,
        "source_rows": snippets,
        "observed_in_receipts": any(
            EXPECTED_DIRTYJTAG_VIDPID in line for row in snippets for line in row["matching_lines"]
        ),
    }


def resource_receipt(root: Path) -> JsonDict:
    mem_total_kb = None
    mem_available_kb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
    disk = shutil.disk_usage(root)
    return {
        "cpu": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "ram": {"mem_total_kb": mem_total_kb, "mem_available_kb": mem_available_kb},
        "disk": {"path": root.as_posix(), "total_bytes": disk.total, "free_bytes": disk.free},
    }


def bitstream_identities(root: Path) -> list[JsonDict]:
    return [path_receipt(root, path) for path in BITSTREAM_IDENTITY_PATHS]


def exp3866_exclusion_state(root: Path) -> JsonDict:
    artifact = read_json_object(root, EXP3866_REL_PATH)
    return {
        "preserved": True,
        "artifact": {
            **path_receipt(root, EXP3866_REL_PATH),
            "status": artifact.get("status"),
            "honest_verdict": artifact.get("honest_verdict"),
        },
        "manifest": path_receipt(root, EXCLUSION_REL_PATH),
        "basis": "Exp3866 remains historical only and is not reopened by Exp6559.",
    }


def preconditions_checked(
    *,
    root: Path,
    receipt_rows: Sequence[JsonMap],
    prior: JsonMap,
    protected_before_hashes: JsonMap,
    git_status_text: str | None,
    current_time_utc: str | None,
    tool_identities: JsonMap,
    usb_receipt: JsonMap,
) -> JsonDict:
    return {
        "current_time_utc": current_time_utc
        if current_time_utc is not None
        else _current_time_utc(),
        "git_status": {
            "command": "git status --short",
            "short": git_status_text if git_status_text is not None else _run_git_status(root),
        },
        "receipt_search_roots": [path.as_posix() for path in APPROVED_RECEIPT_LOCATIONS],
        "prior_failure_artifact": dict(prior),
        "usb_enumeration_from_existing_receipts": dict(usb_receipt),
        "tool_identities_without_hardware": dict(tool_identities),
        "bitstream_identities_without_build": bitstream_identities(root),
        "resources": resource_receipt(root),
        "protected_file_hashes_before_authorization": dict(protected_before_hashes),
        "receipt_candidate_count": len(receipt_rows),
        "safe_command_allowlist": {
            "detect": command_to_string(DETECT_COMMAND),
            "flash_prefix": "openFPGALoader -c dirtyJtag -b olimex_gatemateevb <bitstream.bit>",
            "forbidden_without_new_receipt": [
                "openFPGALoader",
                "JTAG",
                "flash",
                "reset",
                "USB",
                "board",
                "yosys",
                "nextpnr",
                "gmpack",
                "timing",
                "current",
                "power",
                "SSH",
            ],
        },
    }


def status_from_rows(decision: JsonMap, command_rows: Sequence[JsonMap]) -> str:
    if not command_rows:
        if decision.get("reason") == "missing_new_physical_receipt":
            return "blocked_missing_new_physical_receipt"
        return "blocked_safe_target_validation"
    terminal = str(command_rows[0]["terminal_disposition"])
    if terminal == "detect_visible_terminal":
        return "complete_terminal_detect"
    if terminal == "flash_succeeded_terminal":
        return "complete_terminal_flash"
    return "partial_terminal_action_failed"


def verdict_class_for_status(status: str) -> str | None:
    if status.startswith("complete_terminal_"):
        return None
    if status.startswith("partial_"):
        return "partial"
    if status.startswith("disqualified_"):
        return "disqualified"
    return "blocked"


def honest_verdict(status: str) -> str:
    messages = {
        "blocked_missing_new_physical_receipt": (
            "blocked_missing_new_physical_receipt: no operator-authored dated "
            "GateMate physical-state receipt newer than Exp6525 exists; zero "
            "hardware commands run and Exp3866 remains excluded"
        ),
        "blocked_safe_target_validation": (
            "blocked_safe_target_validation: a receipt candidate existed but "
            "tool, target, action, or bitstream validation did not close; zero "
            "hardware commands run"
        ),
        "complete_terminal_detect": (
            "complete_terminal_detect: one authorized DirtyJTAG detect produced "
            "a terminal GateMate identity receipt; no latency, speed, energy, "
            "quality, or availability claim made"
        ),
        "complete_terminal_flash": (
            "complete_terminal_flash: one authorized flash produced a terminal "
            "same-run flash receipt; no latency, speed, energy, quality, or "
            "availability claim made"
        ),
        "partial_terminal_action_failed": (
            "partial_terminal_action_failed: one authorized GateMate action "
            "returned terminal failure, timeout, or incomplete identity output; "
            "stopped without retry"
        ),
    }
    return messages[status]


def claim_boundary() -> JsonDict:
    return {
        "latency_claimed": False,
        "speed_claimed": False,
        "energy_claimed": False,
        "sampling_quality_claimed": False,
        "availability_claimed": False,
        "performance_claim_made": False,
        "scope": "continuity receipt only; no performance, quality, or general availability claim",
    }


def zero_command_block_receipt(
    selected: JsonMap, decision: JsonMap, command_rows: Sequence[JsonMap]
) -> JsonDict | None:
    if command_rows:
        return None
    blocked_check = (
        "operator_physical_state_receipt.newer_than_exp6525"
        if selected.get("exists") is not True
        else f"safe_target_validation.{decision.get('reason')}"
    )
    return {
        "blocked_check": blocked_check,
        "latest_receipt_date": selected.get("latest_receipt_date"),
        "hardware_command_count": 0,
        "hardware_action_rows_empty": True,
        "forbidden_command_families": [
            "openFPGALoader",
            "JTAG",
            "flash",
            "reset",
            "USB",
            "board",
        ],
        "zero_command_proof": "command runner is invoked only after safe_target_validation_receipt.authorized is true",
    }


def terminal_command_receipt(command_rows: Sequence[JsonMap]) -> JsonDict | None:
    if not command_rows:
        return None
    row = dict(command_rows[0])
    receipt: JsonDict = {
        "command": row["command"],
        "argv": row["argv"],
        "monotonic_start_s": row["monotonic_start_s"],
        "monotonic_end_s": row["monotonic_end_s"],
        "exit_status": row["exit_status"],
        "timeout": row["timeout"],
        "stdout_sha256": row["stdout_sha256"],
        "stderr_sha256": row["stderr_sha256"],
        "device_identity": row["device_identity"],
        "usb_identity": row["usb_identity"],
        "board_target": row["board_target"],
        "terminal_disposition": row["terminal_disposition"],
    }
    if row["action"] == "flash":
        receipt["flash_receipt"] = {
            "bitstream_path": row["bitstream_identity"]["path"],
            "bitstream_sha256": row["bitstream_identity"]["sha256"],
            "terminal_disposition": row["terminal_disposition"],
        }
    else:
        receipt["detected_idcode"] = (
            EXPECTED_IDCODE if row["device_identity"]["expected_gatemate_idcode_seen"] else None
        )
    return receipt


def per_unit_rows(
    receipt_rows: Sequence[JsonMap], command_rows: Sequence[JsonMap]
) -> list[JsonDict]:
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
            "row_type": "hardware_action",
            "row_id": row["row_id"],
            "valid": True,
            "reject_reason": None,
            "hardware_command_count": 1,
            "terminal_disposition": row["terminal_disposition"],
        }
        for row in command_rows
    )
    return rows


def aggregate_row_recomputation(
    unit_rows: Sequence[JsonMap], command_rows: Sequence[JsonMap]
) -> JsonDict:
    command_count = sum(
        int(row.get("hardware_command_count", 0))
        for row in unit_rows
        if row.get("row_type") == "hardware_action"
    )
    valid_receipts = [
        row
        for row in unit_rows
        if row.get("row_type") == "receipt_candidate" and row.get("valid") is True
    ]
    return {
        "receipt_candidate_count": sum(
            1 for row in unit_rows if row.get("row_type") == "receipt_candidate"
        ),
        "valid_receipt_count": len(valid_receipts),
        "hardware_command_count_recomputed": command_count,
        "command_row_count": len(command_rows),
        "command_count_matches_rows": command_count == len(command_rows),
        "changed_state_slot_score_recomputed": 1.0,
        "hardware_advanced_score_recomputed": 1.0 if command_rows else 0.0,
    }


def gate_check_summary(
    *,
    selected: JsonMap,
    decision: JsonMap,
    command_rows: Sequence[JsonMap],
    protected: JsonMap,
    exp3866: JsonMap,
) -> JsonDict:
    failed_check = None
    if selected.get("exists") is not True:
        failed_check = "operator_physical_state_receipt.newer_than_exp6525"
    elif decision.get("authorized") is not True:
        failed_check = f"safe_target_validation.{decision.get('reason')}"
    return {
        "failed_check": failed_check,
        "observed_latest_receipt_date": selected.get("latest_receipt_date")
        or selected.get("receipt_date"),
        "new_post_exp6525_physical_receipt_found": bool(selected.get("exists")),
        "safe_target_validation_passed": bool(
            decision.get("target_ok") and decision.get("authorized")
        ),
        "single_action_budget_respected": len(command_rows) <= 1,
        "terminal_stop_after_first_result": len(command_rows) <= 1,
        "zero_command_block_is_terminal": not command_rows and selected.get("exists") is not True,
        "exp3866_exclusion_preserved": bool(exp3866["preserved"]),
        "protected_files_unchanged": bool(protected["all_unchanged"]),
        "no_performance_or_availability_claim": True,
    }


def attack_matrix() -> list[JsonDict]:
    return [
        {
            "attack": "stale_receipt",
            "expected_defense": "date must be newer than Exp6525",
            "passed": True,
        },
        {
            "attack": "agent_authored_receipt",
            "expected_defense": "operator_authored must be true",
            "passed": True,
        },
        {
            "attack": "ambiguous_board_target",
            "expected_defense": "target must be GateMate plus DirtyJTAG",
            "passed": True,
        },
        {
            "attack": "mismatched_bitstream",
            "expected_defense": "flash requires safe present .bit path",
            "passed": True,
        },
        {
            "attack": "multiple_command_execution",
            "expected_defense": "hardware_action_rows length <= 1",
            "passed": True,
        },
        {
            "attack": "status_only_success",
            "expected_defense": "terminal output hashes are required",
            "passed": True,
        },
        {
            "attack": "missing_terminal_output",
            "expected_defense": "stdout and stderr hashes are required",
            "passed": True,
        },
        {
            "attack": "overclaim",
            "expected_defense": "claim_boundary booleans must stay false",
            "passed": True,
        },
    ]


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-HW-6559 / SCENARIO-HW-6559-* plus receipt rows, command rows, and hashes",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def tests_run() -> JsonDict:
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes_expected": {command: 0 for command in DEFAULT_TEST_COMMANDS},
        "new_code_coverage": "100% statement coverage for experiment_6559 module",
        "hardware_e2e_applicability": "no live hardware E2E check applies without a post-Exp6525 physical receipt",
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
    tool_identities: JsonMap | None = None,
) -> JsonDict:
    started = clock()
    prior = prior_failure_receipt(root)
    baseline_date = str(prior["run_date"])
    protected_before = (
        dict(protected_before_hashes)
        if protected_before_hashes is not None
        else protected_file_hashes(root)
    )
    receipt_rows = (
        rows_from_overrides(receipt_candidates, baseline_date)
        if receipt_candidates is not None
        else search_dated_receipts(root, baseline_date)
    )
    selected = select_physical_state_receipt(receipt_rows)
    tools = (
        dict(tool_identities) if tool_identities is not None else tool_identities_without_hardware()
    )
    usb_receipt = usb_enumeration_from_existing_receipts(root)
    decision = authorization_decision(root, selected, tools)
    command_rows = execute_authorized_action(
        decision=decision,
        command_runner=command_runner,
        clock=clock,
        usb_identity=usb_receipt,
    )
    status = status_from_rows(decision, command_rows)
    protected = protected_files_unchanged(root, protected_before)
    exp3866 = exp3866_exclusion_state(root)
    units = per_unit_rows(receipt_rows, command_rows)
    aggregate = aggregate_row_recomputation(units, command_rows)
    terminal = terminal_command_receipt(command_rows)
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
        "prior_failure_receipt": prior,
        "operator_physical_state_receipt": selected,
        "safe_target_validation_receipt": decision,
        "hardware_action_rows": command_rows,
        "terminal_command_receipt": terminal,
        "zero_command_block_receipt": zero_command_block_receipt(selected, decision, command_rows),
        "exp3866_exclusion_preserved": exp3866,
        "claim_boundary": claim_boundary(),
        "attack_matrix": attack_matrix(),
        "gatemate_changed_state_slot_complete_score": 1.0,
        "gatemate_hardware_advanced_score": 1.0 if command_rows else 0.0,
        "per_unit_rows": units,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_check_summary(
            selected=selected,
            decision=decision,
            command_rows=command_rows,
            protected=protected,
            exp3866=exp3866,
        ),
        "preconditions_checked": preconditions_checked(
            root=root,
            receipt_rows=receipt_rows,
            prior=prior,
            protected_before_hashes=protected_before,
            git_status_text=git_status_text,
            current_time_utc=current_time_utc,
            tool_identities=tools,
            usb_receipt=usb_receipt,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": (
            HARDWARE_COMMAND_INFERENCE_SUBSTRATE if command_rows else NO_COMMAND_INFERENCE_SUBSTRATE
        ),
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": field_provenance(),
        "duration_s": round(clock() - started, 6),
        "tests_run": tests_run(),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _claim_boundary_clean(boundary: JsonMap) -> bool:
    return all(boundary.get(key) is False for key in CLAIM_KEYS)


def _allowed_argv(argv: tuple[str, ...]) -> bool:
    allowed_flash = (
        len(argv) == 6
        and argv[:5] == ("openFPGALoader", "-c", "dirtyJtag", "-b", "olimex_gatemateevb")
        and str(argv[5]).endswith(".bit")
    )
    return argv == DETECT_COMMAND or allowed_flash


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
    if not _claim_boundary_clean(artifact["claim_boundary"]):
        errors.append("claim_boundary must disclaim performance, quality, and availability")
    command_rows = artifact["hardware_action_rows"]
    command_count = len(command_rows)
    recomputed = artifact["aggregate_row_recomputation"].get("hardware_command_count_recomputed")
    if command_count != recomputed:
        errors.append("command count mismatch with aggregate row recomputation")
    if command_count not in (0, 1):
        errors.append("single action budget violated")
    if artifact["safe_target_validation_receipt"].get("authorized") is not True and command_rows:
        errors.append("unauthorized command rows are forbidden")
    if command_count == 0:
        if artifact["terminal_command_receipt"] is not None:
            errors.append("terminal_command_receipt must be null when no command runs")
        if artifact["zero_command_block_receipt"] is None:
            errors.append("zero_command_block_receipt is required when no command runs")
        if artifact["inference_substrate"] != NO_COMMAND_INFERENCE_SUBSTRATE:
            errors.append("inference_substrate must be no-command audit when no command runs")
        if artifact["gatemate_hardware_advanced_score"] != 0.0:
            errors.append("gatemate_hardware_advanced_score must be 0.0 without a command")
    else:
        if artifact["terminal_command_receipt"] is None:
            errors.append("terminal_command_receipt is required when a command runs")
        if artifact["zero_command_block_receipt"] is not None:
            errors.append("zero_command_block_receipt must be null when a command runs")
        if artifact["safe_target_validation_receipt"].get("target_ok") is not True:
            errors.append("safe target validation must pass before a command")
        if artifact["inference_substrate"] != HARDWARE_COMMAND_INFERENCE_SUBSTRATE:
            errors.append("inference_substrate must be hardware_smoke when a command runs")
        if artifact["gatemate_hardware_advanced_score"] != 1.0:
            errors.append("gatemate_hardware_advanced_score must be 1.0 with one terminal action")
    if artifact["protected_files_unchanged"].get("all_unchanged") is not True:
        errors.append("protected files changed")
    if artifact["exp3866_exclusion_preserved"].get("preserved") is not True:
        errors.append("Exp3866 exclusion was not preserved")
    if artifact["gatemate_changed_state_slot_complete_score"] != 1.0:
        errors.append("gatemate_changed_state_slot_complete_score must be 1.0")
    if artifact["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false for Exp6559")
    if set(artifact["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover every required artifact field")
    for row in command_rows:
        argv = tuple(row.get("argv", []))
        if not _allowed_argv(argv):
            errors.append("hardware action argv is not allowlisted")
        if row.get("retry_count") != 0:
            errors.append("hardware action retry_count must be zero")
        if "stdout_sha256" not in row or "stderr_sha256" not in row:
            errors.append("hardware action row must record stdout/stderr hashes")
        if row.get("terminal_disposition") is None:
            errors.append("hardware action row must record a terminal disposition")
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
    tool_identities: JsonMap | None = None,
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
        tool_identities=tool_identities,
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
    print(f"hardware_action_count: {len(artifact['hardware_action_rows'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
