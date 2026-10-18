#!/usr/bin/env python3
"""Exp6121 GateMate changed-state gate.

Spec refs: REQ-HW-6121, SCENARIO-HW-6121.

Why this exists
---------------
The GateMate board has already produced useful evidence: historical bitstreams
were flashed, the DirtyJTAG USB transport enumerated, and the latest diagnostic
narrowed the current blocker to physical cable/port/power state. Re-running the
same `openFPGALoader -c dirtyJtag --detect` command without a physical change
does not add evidence. This module therefore makes the physical state itself the
authorization gate: unchanged state emits an operator action packet and runs no
JTAG command; a newer dated physical receipt permits exactly one read-only
IDCODE detect, and a prebuilt read-only smoke is gated by the expected IDCODE.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
CommandRunner = Callable[[tuple[str, ...], float], "CommandProbe"]

SCHEMA = "carnot.experiment_6121_gatemate_changed_state_gate.v530"
EXPERIMENT = "experiment_6121_gatemate_changed_state_gate"
EXPERIMENT_ID = "exp6121-gatemate-changed-state-gate-v530"
MILESTONE = "2026.08.530"
RUN_DATE = "20260804"
RANDOM_SEED = 6121
SPEC_REFS = ["REQ-HW-6121", "SCENARIO-HW-6121"]
OUTPUT_REL_PATH = Path("results") / "experiment_6121_gatemate_changed_state_gate_v530.json"
INFERENCE_SUBSTRATE = "hardware_state_gate_with_optional_non_destructive_detect"

EXPECTED_IDCODE = "0x20000001"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
PREBUILT_READ_ONLY_SMOKE_COMMAND = ("gatemate-prebuilt-smoke", "--read-only")

PRIOR_BOARD_ARTIFACTS = (
    "results/experiment_3866_gatemate_ising_tile_flash_v2.json",
    "results/experiment_5201_hardware_continuity_gatemate_diagnostic_v476.json",
    "results/experiment_5217_hardware_continuity_v477.json",
    "results/experiment_5861_attached_board_state_receipts.json",
)
OPERATOR_RECEIPT_PATHS = (
    "research-hardware-wishlist.md",
    "ops/known-issues.md",
    "ops/exclusion_manifest.yaml",
    "docs/jtag-wiring-gatemate-dirtyjtag.md",
)
PROTECTED_REL_PATHS = (
    "scripts/research_conductor.py",
    "ops/status.md",
    "ops/changelog.md",
    "_bmad/traceability.md",
)

LAST_ATTEMPT_DATE = "20260704"
LAST_ATTEMPT_PHYSICAL_STATE: JsonDict = {
    "board": "Cologne Chip GateMate A1-EVB-2M",
    "board_provenance": "results/experiment_5217_hardware_continuity_v477.json",
    "cable": "cached GateMate USB-C/onboard DirtyJTAG path; no reseat receipt after v477",
    "port": "cached USB path 3-2.3, bus 003 device 006, shared hub path 2.3",
    "power": "cached physical power unresolved; raw all-ones TDO suggests open or unpowered target",
    "usb_dirtyjtag": "1209:c0ca Jean THOMAS DirtyJTAG serial 1861832311111616",
    "dirtyjtag": "bcdDevice=0111 product=DirtyJTAG, openFPGALoader v1.1.1",
    "expected_idcode": EXPECTED_IDCODE,
    "observed_idcode": None,
    "raw_idcode": "0xffffffff",
    "bitstream_artifact": "results/experiment_3866_gatemate_ising_tile_flash_v2.json",
    "last_attempt_date": LAST_ATTEMPT_DATE,
}

DEFAULT_DATED_OPERATOR_PHYSICAL_RECEIPT: JsonDict = {
    "exists": False,
    "checked_on": RUN_DATE,
    "receipt_date": None,
    "source": "repo search: no newer dated physical cable/port/power/board/DirtyJTAG receipt found",
    "changes": [],
    "cable": LAST_ATTEMPT_PHYSICAL_STATE["cable"],
    "port": LAST_ATTEMPT_PHYSICAL_STATE["port"],
    "power": LAST_ATTEMPT_PHYSICAL_STATE["power"],
    "board": LAST_ATTEMPT_PHYSICAL_STATE["board"],
    "usb_dirtyjtag": LAST_ATTEMPT_PHYSICAL_STATE["usb_dirtyjtag"],
    "dirtyjtag": LAST_ATTEMPT_PHYSICAL_STATE["dirtyjtag"],
    "operator_authorized_detect": False,
}

EXACT_OPERATOR_ACTION_PACKET: JsonDict = {
    "packet_id": "gatemate-physical-change-required-v530",
    "board": "Cologne Chip GateMate A1-EVB-2M",
    "blocked_command": "openFPGALoader -c dirtyJtag --detect",
    "required_physical_delta": "change cable, port, power, board, or DirtyJTAG state",
    "operator_steps": [
        "Reseat or replace the GateMate USB-C/onboard DirtyJTAG connection.",
        "Move the GateMate/DirtyJTAG USB connection to a different host root port or powered hub port.",
        "Confirm and record the GateMate board power LED after the cable/port change.",
        "Record a new dated physical receipt naming cable, port, power, board, USB, and DirtyJTAG descriptors.",
    ],
    "next_software_action_after_receipt": "rerun Exp6121 once to permit exactly one non-destructive IDCODE detect",
    "do_not_do": [
        "do not flash",
        "do not synthesize",
        "do not place or route",
        "do not pack",
        "do not mutate firmware",
        "do not repeat detect with unchanged physical state",
    ],
}

ZERO_MUTATION_COUNTS: JsonDict = {
    "flash": 0,
    "synthesis": 0,
    "place": 0,
    "route": 0,
    "pack": 0,
    "firmware_mutation": 0,
}
ZERO_CLAIM_COUNTS: JsonDict = {
    "speedup": 0,
    "power_efficiency": 0,
    "current_draw": 0,
    "terminal_hardware": 0,
    "tsu": 0,
    "kona": 0,
}

PREBUILT_SMOKE_DESCRIPTOR = (
    "exp6121_prebuilt_read_only_host_io_smoke_v1: may run only after GM1Ax "
    "IDCODE 0x20000001; no flash, synthesis, place, route, pack, or firmware mutation"
)
PREBUILT_SMOKE_EXPECTED_HASH = hashlib.sha256(PREBUILT_SMOKE_DESCRIPTOR.encode()).hexdigest()

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "prior_and_current_physical_state_hashes",
    "dated_operator_physical_receipt",
    "physical_state_changed",
    "cable_port_power_board_usb_and_dirtyjtag_receipts",
    "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code",
    "expected_and_observed_idcode",
    "prebuilt_bitstream_and_smoke_hashes",
    "flash_synthesis_place_route_pack_and_firmware_mutation_counts",
    "operator_action_packet",
    "hardware_execution_authenticated",
    "speed_power_and_terminal_claim_counts",
    "retirement_triggered",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
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
    "status": "Terminal state for the changed-state gate.",
    "preconditions_checked": "Hash all evidence before any optional board command.",
    "prior_and_current_physical_state_hashes": (
        "Physical change, not another software loop, authorizes one attempt."
    ),
    "dated_operator_physical_receipt": (
        "A dated receipt is the only operator authorization for a new physical state."
    ),
    "physical_state_changed": "Bare bool gates every JTAG command.",
    "cable_port_power_board_usb_and_dirtyjtag_receipts": (
        "Every physical and transport assumption is explicit."
    ),
    "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code": (
        "Unchanged state yields zero commands; changed state permits one auditable non-destructive command."
    ),
    "expected_and_observed_idcode": "Smoke execution requires authenticated device identity.",
    "prebuilt_bitstream_and_smoke_hashes": (
        "Existing artifacts must remain immutable before any read-only smoke."
    ),
    "flash_synthesis_place_route_pack_and_firmware_mutation_counts": (
        "All mutation counts remain zero without explicit operator authorization."
    ),
    "operator_action_packet": (
        "An unchanged-state block ends with one actionable physical next step."
    ),
    "hardware_execution_authenticated": (
        "No execution claim survives without raw hardware evidence."
    ),
    "speed_power_and_terminal_claim_counts": (
        "No speed, power, current-draw, or terminal-hardware claim is permitted."
    ),
    "retirement_triggered": (
        "Repeating the same physical block retires this changed-state task shape."
    ),
    "protected_files_unchanged": (
        "Conductor and operator-reconciled files remain byte-identical."
    ),
    "duration_s": (
        "Use measured `hardware_state_gate_with_optional_non_destructive_detect` wall time."
    ),
    "inference_substrate": (
        "Use `hardware_state_gate_with_optional_non_destructive_detect`."
    ),
    "verifier_is_oracle": (
        "Raw IDCODE/host-I/O evidence is authoritative; simulation is not board execution."
    ),
    "missing_verifier_gaps": "Record missing raw IDCODE or host-I/O evidence instead of inferring.",
    "field_provenance": "Every field traces to receipts, hashes, command output, or tests.",
    "test_commands": "Verification commands are recorded.",
    "test_exit_codes": "Exit codes prevent failed checks becoming success.",
    "reproducibility_checksum": "Checksum detects physical-state, artifact, or receipt drift.",
    "honest_verdict": (
        "Use `complete_changed_state:`, `blocked_physical_action:`, `retired:`, or `blocked:`."
    ),
}

DEFAULT_TEST_COMMANDS = [
    ".venv/bin/pytest tests/python/test_experiment_6121_gatemate_changed_state_gate_v530.py -q",
    ".venv/bin/coverage run --source=python/carnot/experiment_6121_gatemate_changed_state_gate_v530.py -m pytest tests/python/test_experiment_6121_gatemate_changed_state_gate_v530.py -q",
    ".venv/bin/coverage report --fail-under=100 -m python/carnot/experiment_6121_gatemate_changed_state_gate_v530.py",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
]


class CommandProbe:
    """Captured stdout/stderr/exit evidence for the one permitted board command.

    Hardware state lives outside the repository. Storing exact command output is
    the difference between an auditable IDCODE receipt and an inferred board
    claim.
    """

    def __init__(
        self,
        command: Sequence[str],
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_s: float,
    ) -> None:
        self.command = tuple(command)
        self.exit_code = int(exit_code)
        self.stdout = stdout
        self.stderr = stderr
        self.duration_s = float(duration_s)

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> JsonDict:
        return {
            "command": command_to_string(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round(self.duration_s, 6),
        }


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:
    """Run a bounded subprocess and return a command receipt."""

    started = time.perf_counter()
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return CommandProbe(
        command,
        result.returncode,
        result.stdout,
        result.stderr,
        time.perf_counter() - started,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return f"sha256:{sha256_bytes(encoded)}"


def path_receipt(root: Path, relative_path: str | Path) -> JsonDict:
    path = root / Path(relative_path)
    if not path.exists():
        return {
            "path": Path(relative_path).as_posix(),
            "present": False,
            "bytes": 0,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "path": Path(relative_path).as_posix(),
        "present": True,
        "bytes": len(data),
        "sha256": f"sha256:{sha256_bytes(data)}",
    }


def read_json_if_present(root: Path, relative_path: str) -> JsonDict:
    path = root / relative_path
    if not path.exists():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {path: path_receipt(root, path)["sha256"] for path in PROTECTED_REL_PATHS}


def protected_files_unchanged(
    root: Path,
    before_hashes: Mapping[str, str | None],
) -> JsonDict:
    after = protected_file_hashes(root)
    changed = [path for path, before in before_hashes.items() if after.get(path) != before]
    return {
        "all_unchanged": not changed,
        "changed_paths": changed,
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
    }


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"git_failed:{result.returncode}:{result.stderr}"
    return result.stdout


def dirty_worktree_receipt(root: Path) -> JsonDict:
    status_lines = _git_output(
        root,
        ["status", "--porcelain=v1", "--untracked-files=normal"],
    ).splitlines()
    output_path = OUTPUT_REL_PATH.as_posix()
    filtered = [line for line in status_lines if output_path not in line]
    diff = _git_output(root, ["diff", "--no-ext-diff", "--binary"])
    staged = _git_output(root, ["diff", "--cached", "--binary"])
    return {
        "status_porcelain_sha256": f"sha256:{sha256_bytes('\\n'.join(filtered).encode())}",
        "tracked_diff_sha256": f"sha256:{sha256_bytes(diff.encode())}",
        "staged_diff_sha256": f"sha256:{sha256_bytes(staged.encode())}",
        "status_line_count_excluding_output": len(filtered),
        "output_path_excluded_from_hash": output_path,
    }


def prior_board_artifact_receipts(root: Path) -> JsonDict:
    hashes = {path: path_receipt(root, path) for path in PRIOR_BOARD_ARTIFACTS}
    return {
        "all_hashed": all(item["present"] for item in hashes.values()),
        "hashes": hashes,
    }


def _find_line_with(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return ""


def operator_receipts(root: Path, dated_receipt: JsonMap) -> JsonDict:
    receipts = {path: path_receipt(root, path) for path in OPERATOR_RECEIPT_PATHS}
    wishlist = root / "research-hardware-wishlist.md"
    wishlist_text = wishlist.read_text(encoding="utf-8") if wishlist.exists() else ""
    return {
        "path_hashes": receipts,
        "hardware_wishlist_gatemate_block_line": _find_line_with(
            wishlist_text,
            "GateMate preserves the v477 physical/JTAG block",
        ),
        "dated_operator_physical_receipt": dict(dated_receipt),
    }


def _apply_physical_receipt(receipt: JsonMap) -> JsonDict:
    state = dict(LAST_ATTEMPT_PHYSICAL_STATE)
    if not _receipt_authorizes_changed_state(receipt):
        return state
    for field in ("cable", "port", "power", "board", "usb_dirtyjtag", "dirtyjtag"):
        if field in receipt and receipt.get(field) is not None:
            state[field] = receipt[field]
    state["operator_receipt_date"] = receipt.get("receipt_date")
    state["operator_receipt_source"] = receipt.get("source")
    state["operator_receipt_changes"] = list(receipt.get("changes", []))
    return state


def _receipt_authorizes_changed_state(receipt: JsonMap) -> bool:
    changes = receipt.get("changes")
    date = str(receipt.get("receipt_date") or "")
    physical_fields = {"cable", "port", "power", "board", "usb_dirtyjtag", "dirtyjtag"}
    changed_fields = {str(item.get("field")) for item in changes if isinstance(item, Mapping)}
    return bool(changes) and date > LAST_ATTEMPT_DATE and bool(changed_fields & physical_fields)


def physical_state_hashes(dated_receipt: JsonMap) -> tuple[JsonDict, JsonDict, bool]:
    prior = dict(LAST_ATTEMPT_PHYSICAL_STATE)
    current = _apply_physical_receipt(dated_receipt)
    authorized = _receipt_authorizes_changed_state(dated_receipt)
    changed = stable_hash(prior) != stable_hash(current) and authorized
    return prior, current, changed


def idcode_from_text(text: str) -> str | None:
    match = re.search(r"idcode\s+(0x[0-9a-fA-F]+)", text)
    return match.group(1).lower() if match else None


def _detect_receipt_without_command(reason: str) -> JsonDict:
    return {
        "allowed": False,
        "attempt_count": 0,
        "command": None,
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "duration_s": 0.0,
        "reason": reason,
    }


def run_optional_detect(
    *,
    physical_state_changed: bool,
    command_runner: CommandRunner,
) -> tuple[JsonDict, str | None]:
    if not physical_state_changed:
        return _detect_receipt_without_command("unchanged_physical_state"), None
    probe = command_runner(DETECT_COMMAND, 30.0)
    receipt = {
        "allowed": True,
        "attempt_count": 1,
        "command": command_to_string(probe.command),
        "stdout": probe.stdout,
        "stderr": probe.stderr,
        "exit_code": probe.exit_code,
        "duration_s": round(probe.duration_s, 6),
        "reason": "dated_physical_state_changed",
    }
    return receipt, idcode_from_text(probe.combined_output)


def bitstream_receipts(root: Path) -> JsonDict:
    exp3866 = read_json_if_present(root, PRIOR_BOARD_ARTIFACTS[0])
    bitstream_path = str(exp3866.get("bitstream_path") or "")
    actual_sha = None
    if bitstream_path:
        path = Path(bitstream_path)
        if path.exists():
            actual_sha = sha256_bytes(path.read_bytes())
    prior_sha = exp3866.get("bitstream_sha256")
    return {
        "bitstream": {
            "path": bitstream_path,
            "prior_receipt_sha256": prior_sha,
            "actual_sha256": actual_sha,
            "matches_prior_receipt": bool(prior_sha and actual_sha == prior_sha),
            "source_artifact": PRIOR_BOARD_ARTIFACTS[0],
        },
        "smoke": {
            "descriptor": PREBUILT_SMOKE_DESCRIPTOR,
            "expected_hash": PREBUILT_SMOKE_EXPECTED_HASH,
            "prior_receipt": "prebuilt descriptor only; no flash or pack command is allowed here",
        },
    }


def _smoke_hash_from_stdout(stdout: str) -> tuple[str | None, bool, bool]:
    parsed: JsonDict = {}
    try:
        value = json.loads(stdout.strip())
        parsed = value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        parsed = {}
    smoke_hash = parsed.get("prebuilt_smoke_sha256")
    return (
        str(smoke_hash) if smoke_hash else None,
        parsed.get("read_only") is True,
        parsed.get("host_io_observed") is True,
    )


def run_optional_smoke(
    *,
    observed_idcode: str | None,
    command_runner: CommandRunner,
    prebuilt_smoke_command: tuple[str, ...] | None,
) -> JsonDict:
    if observed_idcode != EXPECTED_IDCODE:
        return {
            "attempted": False,
            "reason": "expected_idcode_not_observed",
            "command": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "hash_matches_prior_receipt": False,
            "read_only": False,
        }
    if prebuilt_smoke_command is None:
        return {
            "attempted": False,
            "reason": "no_prebuilt_smoke_command_configured",
            "command": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "hash_matches_prior_receipt": False,
            "read_only": False,
        }
    probe = command_runner(tuple(prebuilt_smoke_command), 30.0)
    smoke_hash, read_only, host_io = _smoke_hash_from_stdout(probe.stdout)
    return {
        "attempted": True,
        "reason": "expected_idcode_observed",
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "stdout": probe.stdout,
        "stderr": probe.stderr,
        "hash_matches_prior_receipt": smoke_hash == PREBUILT_SMOKE_EXPECTED_HASH,
        "read_only": read_only,
        "host_io_observed": host_io,
        "duration_s": round(probe.duration_s, 6),
    }


def preconditions_checked(root: Path, dated_receipt: JsonMap) -> JsonDict:
    prior, current, changed = physical_state_hashes(dated_receipt)
    return {
        "prior_board_artifacts": prior_board_artifact_receipts(root),
        "cable_port_power_descriptions": {
            "prior_hash": stable_hash(
                {k: prior[k] for k in ("cable", "port", "power", "board")}
            ),
            "current_hash": stable_hash(
                {k: current[k] for k in ("cable", "port", "power", "board")}
            ),
            "physical_state_changed": changed,
        },
        "usb_dirtyjtag_descriptors": {
            "prior": {
                "usb_dirtyjtag": prior["usb_dirtyjtag"],
                "dirtyjtag": prior["dirtyjtag"],
            },
            "current": {
                "usb_dirtyjtag": current["usb_dirtyjtag"],
                "dirtyjtag": current["dirtyjtag"],
            },
            "hash": stable_hash(
                {
                    "prior": [prior["usb_dirtyjtag"], prior["dirtyjtag"]],
                    "current": [current["usb_dirtyjtag"], current["dirtyjtag"]],
                }
            ),
        },
        "tool_versions": {
            "source": "results/experiment_5217_hardware_continuity_v477.json",
            "openFPGALoader": "openFPGALoader v1.1.1",
            "DirtyJTAG": "bcdDevice=0111",
            "hash": stable_hash({"openFPGALoader": "v1.1.1", "DirtyJTAG": "0111"}),
        },
        "operator_receipts": operator_receipts(root, dated_receipt),
        "bitstream_hashes": bitstream_receipts(root),
        "output_paths": {
            "result_path": OUTPUT_REL_PATH.as_posix(),
            "result_path_hash": stable_hash({"result_path": OUTPUT_REL_PATH.as_posix()}),
        },
        "protected_files": {
            "hashes": protected_file_hashes(root),
        },
        "dirty_worktree": dirty_worktree_receipt(root),
    }


def cable_port_power_board_usb_and_dirtyjtag_receipts(prior: JsonMap, current: JsonMap) -> JsonDict:
    return {
        "prior": {
            key: prior[key]
            for key in ("cable", "port", "power", "board", "usb_dirtyjtag", "dirtyjtag")
        },
        "current": {
            key: current[key]
            for key in ("cable", "port", "power", "board", "usb_dirtyjtag", "dirtyjtag")
        },
    }


def hardware_execution_authenticated(smoke_attempt: JsonMap, observed_idcode: str | None) -> JsonDict:
    authenticated = (
        observed_idcode == EXPECTED_IDCODE
        and smoke_attempt.get("attempted") is True
        and smoke_attempt.get("exit_code") == 0
        and smoke_attempt.get("hash_matches_prior_receipt") is True
        and smoke_attempt.get("read_only") is True
        and smoke_attempt.get("host_io_observed") is True
    )
    return {
        "authenticated": authenticated,
        "idcode_authenticated": observed_idcode == EXPECTED_IDCODE,
        "read_only_smoke_authenticated": authenticated,
        "simulation_used_as_board_execution": False,
    }


def missing_verifier_gaps(
    *,
    physical_state_changed: bool,
    observed_idcode: str | None,
    smoke_attempt: JsonMap,
) -> list[str]:
    gaps: list[str] = []
    if not physical_state_changed:
        gaps.append("no_new_raw_idcode_due_to_unchanged_physical_state_gate")
    if observed_idcode != EXPECTED_IDCODE:
        gaps.append("expected_gatemate_idcode_not_observed")
    if smoke_attempt.get("attempted") is not True:
        gaps.append(str(smoke_attempt.get("reason") or "prebuilt_smoke_not_attempted"))
    return gaps


def artifact_status(
    *,
    physical_state_changed: bool,
    observed_idcode: str | None,
    auth: JsonMap,
) -> str:
    if not physical_state_changed:
        return "blocked_physical_action"
    if observed_idcode == EXPECTED_IDCODE and auth.get("authenticated") is True:
        return "complete_changed_state"
    return "blocked"


def honest_verdict(
    *,
    status: str,
    observed_idcode: str | None,
    smoke_attempt: JsonMap,
) -> str:
    if status == "blocked_physical_action":
        return (
            "blocked_physical_action: unchanged GateMate cable/port/power/board/DirtyJTAG "
            "state; no JTAG command run; operator must change physical state and record a "
            "dated receipt"
        )
    if status == "complete_changed_state":
        return (
            "complete_changed_state: expected GateMate IDCODE observed and matching "
            "read-only prebuilt smoke completed; no mutation or performance claim"
        )
    if observed_idcode == EXPECTED_IDCODE:
        return f"blocked: expected GateMate IDCODE observed but {smoke_attempt.get('reason')}"
    return "blocked: changed physical receipt allowed one detect but expected GateMate IDCODE was not observed"


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-HW-6121 / SCENARIO-HW-6121 and local receipts",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "duration_s",
            "test_exit_codes",
            "reproducibility_checksum",
        }
    }
    return stable_hash(stable).removeprefix("sha256:")


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    dated_operator_physical_receipt: JsonMap | None = None,
    prebuilt_smoke_command: tuple[str, ...] | None = None,
    protected_before_hashes: Mapping[str, str | None] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build the changed-state receipt, optionally running one bounded detect."""

    started = clock()
    root = Path(root)
    dated_receipt = dict(dated_operator_physical_receipt or DEFAULT_DATED_OPERATOR_PHYSICAL_RECEIPT)
    prior_state, current_state, physical_changed = physical_state_hashes(dated_receipt)
    detect_receipt, observed_idcode = run_optional_detect(
        physical_state_changed=physical_changed,
        command_runner=command_runner,
    )
    bitstreams = bitstream_receipts(root)
    smoke_attempt = run_optional_smoke(
        observed_idcode=observed_idcode,
        command_runner=command_runner,
        prebuilt_smoke_command=prebuilt_smoke_command,
    )
    bitstreams["smoke_attempt"] = smoke_attempt
    auth = hardware_execution_authenticated(smoke_attempt, observed_idcode)
    protected_before = dict(protected_before_hashes or protected_file_hashes(root))
    status = artifact_status(
        physical_state_changed=physical_changed,
        observed_idcode=observed_idcode,
        auth=auth,
    )
    commands = list(test_commands) if test_commands is not None else list(DEFAULT_TEST_COMMANDS)
    exit_codes = (
        dict(test_exit_codes)
        if test_exit_codes is not None
        else {command: None for command in commands}
    )
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
        "preconditions_checked": preconditions_checked(root, dated_receipt),
        "prior_and_current_physical_state_hashes": {
            "prior": stable_hash(prior_state),
            "current": stable_hash(current_state),
            "prior_state": prior_state,
            "current_state": current_state,
        },
        "dated_operator_physical_receipt": dated_receipt,
        "physical_state_changed": physical_changed,
        "cable_port_power_board_usb_and_dirtyjtag_receipts": (
            cable_port_power_board_usb_and_dirtyjtag_receipts(prior_state, current_state)
        ),
        "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code": (
            detect_receipt
        ),
        "expected_and_observed_idcode": {
            "expected_idcode": EXPECTED_IDCODE,
            "observed_idcode": observed_idcode,
            "matches": observed_idcode == EXPECTED_IDCODE,
        },
        "prebuilt_bitstream_and_smoke_hashes": bitstreams,
        "flash_synthesis_place_route_pack_and_firmware_mutation_counts": dict(
            ZERO_MUTATION_COUNTS
        ),
        "operator_action_packet": dict(EXACT_OPERATOR_ACTION_PACKET)
        if not physical_changed
        else {},
        "hardware_execution_authenticated": auth,
        "speed_power_and_terminal_claim_counts": dict(ZERO_CLAIM_COUNTS),
        "retirement_triggered": not physical_changed,
        "protected_files_unchanged": protected_files_unchanged(root, protected_before),
        "duration_s": round(float(clock() - started), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "missing_verifier_gaps": missing_verifier_gaps(
            physical_state_changed=physical_changed,
            observed_idcode=observed_idcode,
            smoke_attempt=smoke_attempt,
        ),
        "field_provenance": field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(
        status=status,
        observed_idcode=observed_idcode,
        smoke_attempt=smoke_attempt,
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        return [f"missing required fields: {missing}"]
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    detect = artifact.get("detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code")
    if not isinstance(detect, Mapping):
        errors.append("detect receipt must be a mapping")
        return errors
    attempt_count = detect.get("attempt_count")
    if not isinstance(attempt_count, int) or attempt_count > 1:
        errors.append("at most one detect attempt is permitted")
    if attempt_count == 1 and detect.get("command") != command_to_string(DETECT_COMMAND):
        errors.append("detect command allowlist violation")
    if artifact.get("physical_state_changed") is False:
        if attempt_count != 0:
            errors.append("unchanged physical state must have zero detect attempts")
        if detect.get("command") is not None:
            errors.append("unchanged physical state must not record a command")
        if artifact.get("operator_action_packet") != EXACT_OPERATOR_ACTION_PACKET:
            errors.append("unchanged physical state must emit exact operator action packet")
    if artifact.get("flash_synthesis_place_route_pack_and_firmware_mutation_counts") != ZERO_MUTATION_COUNTS:
        errors.append("mutation count must remain zero for flash/synthesis/place/route/pack/firmware")
    if artifact.get("speed_power_and_terminal_claim_counts") != ZERO_CLAIM_COUNTS:
        errors.append("speed/power/current/terminal/tsu/kona claim counts must be zero")
    smoke = artifact.get("prebuilt_bitstream_and_smoke_hashes", {}).get("smoke_attempt", {})
    observed = artifact.get("expected_and_observed_idcode", {}).get("observed_idcode")
    if isinstance(smoke, Mapping) and smoke.get("attempted") is True and observed != EXPECTED_IDCODE:
        errors.append("prebuilt smoke attempted without expected IDCODE")
    auth = artifact.get("hardware_execution_authenticated")
    if isinstance(auth, Mapping) and auth.get("authenticated") is True:
        if observed != EXPECTED_IDCODE or not isinstance(smoke, Mapping) or smoke.get("attempted") is not True:
            errors.append("hardware execution authenticated without raw IDCODE and smoke evidence")
    protected = artifact.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        errors.append("protected files must be unchanged")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(
        ("complete_changed_state:", "blocked_physical_action:", "retired:", "blocked:")
    ):
        errors.append("honest_verdict prefix mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(repo_root: Path, artifact: JsonMap) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: Path = REPO_ROOT,
    source_root: Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    dated_operator_physical_receipt: JsonMap | None = None,
    prebuilt_smoke_command: tuple[str, ...] | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> Path:
    source = Path(source_root) if source_root is not None else Path(repo_root)
    protected_before = protected_file_hashes(source)
    artifact = build_artifact(
        root=source,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        dated_operator_physical_receipt=dated_operator_physical_receipt,
        prebuilt_smoke_command=prebuilt_smoke_command,
        protected_before_hashes=protected_before,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    return write_artifact(Path(repo_root), artifact)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(repo_root=Path(args.repo_root), run_date=args.date)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    detect = artifact["detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"]
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"physical_state_changed: {artifact['physical_state_changed']}")
    print(f"detect_attempt_count: {detect['attempt_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - live CLI entrypoint.
    raise SystemExit(main())
