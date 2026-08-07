"""Exp6199 GateMate terminal-action audit.

Spec refs: REQ-HW-6199, SCENARIO-HW-6199-1, SCENARIO-HW-6199-2,
SCENARIO-HW-6199-3, SCENARIO-HW-6199-4, SCENARIO-HW-6199-5.

This audit keeps GateMate visible without turning a known physical block into a
software loop. Exp6121 is the cached baseline. A newer operator receipt must
describe a real cable, port, power, board, USB, or DirtyJTAG change before this
module may run one read-only IDCODE detect.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
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

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "carnot.experiment_6199.gatemate_terminal_action_audit.v537"
EXPERIMENT = "experiment_6199_gatemate_terminal_action_audit"
EXPERIMENT_ID = "exp6199-gatemate-terminal-action-audit-v537"
MILESTONE = "2026.08.537"
RUN_DATE = "20260807"
BASELINE_RECEIPT_DATE = "20260804"
RANDOM_SEED = 6199
SPEC_REFS = (
    "REQ-HW-6199",
    "SCENARIO-HW-6199-1",
    "SCENARIO-HW-6199-2",
    "SCENARIO-HW-6199-3",
    "SCENARIO-HW-6199-4",
    "SCENARIO-HW-6199-5",
)
OUTPUT_REL_PATH = Path("results/experiment_6199_gatemate_terminal_action_audit_v537.json")
INFERENCE_SUBSTRATE = "cached_gatemate_terminal_action_audit_with_optional_single_detect"

EXPECTED_IDCODE = "0x20000001"
DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
MATERIAL_PHYSICAL_FIELDS = frozenset(
    {"cable", "port", "power", "board", "usb_dirtyjtag", "dirtyjtag"}
)

EXP6121_REL_PATH = Path("results/experiment_6121_gatemate_changed_state_gate_v530.json")
EXP3866_REL_PATH = Path("results/experiment_3866_gatemate_ising_tile_flash_v2.json")
HASHED_INPUT_PATHS = (
    EXP6121_REL_PATH,
    EXP3866_REL_PATH,
    Path("ops/hardware-bringup-prep.md"),
    Path("research-hardware-wishlist.md"),
    Path("ops/known-issues.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("docs/jtag-wiring-gatemate-dirtyjtag.md"),
)
PROTECTED_REL_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
)

CANONICAL_PRIOR_PHYSICAL_STATE: JsonDict = {
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
    "bitstream_artifact": EXP3866_REL_PATH.as_posix(),
    "baseline_artifact": EXP6121_REL_PATH.as_posix(),
    "baseline_date": BASELINE_RECEIPT_DATE,
}

DEFAULT_OPERATOR_RECEIPT: JsonDict = {
    "exists": False,
    "checked_on": RUN_DATE,
    "receipt_date": None,
    "source": "repo search: no newer dated cable/port/power/board/USB/DirtyJTAG receipt supplied",
    "changes": [],
    "cable": CANONICAL_PRIOR_PHYSICAL_STATE["cable"],
    "port": CANONICAL_PRIOR_PHYSICAL_STATE["port"],
    "power": CANONICAL_PRIOR_PHYSICAL_STATE["power"],
    "board": CANONICAL_PRIOR_PHYSICAL_STATE["board"],
    "usb_dirtyjtag": CANONICAL_PRIOR_PHYSICAL_STATE["usb_dirtyjtag"],
    "dirtyjtag": CANONICAL_PRIOR_PHYSICAL_STATE["dirtyjtag"],
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
    "next_software_action_after_receipt": (
        "rerun Exp6121 once to permit exactly one non-destructive IDCODE detect"
    ),
    "do_not_do": [
        "do not flash",
        "do not synthesize",
        "do not place or route",
        "do not pack",
        "do not mutate firmware",
        "do not repeat detect with unchanged physical state",
    ],
}

ZERO_MUTATION_COMMAND_COUNTS: JsonDict = {
    "synthesis": 0,
    "place": 0,
    "route": 0,
    "pack": 0,
    "flash": 0,
    "firmware": 0,
    "ssh": 0,
    "timing": 0,
    "current": 0,
    "power": 0,
}
ZERO_CLAIM_COUNTS: JsonDict = {
    "speed": 0,
    "power": 0,
    "energy": 0,
    "terminal": 0,
    "terminal_hardware": 0,
    "tsu": 0,
    "kona": 0,
}
PASSIVE_COOLING_NOTE = (
    "GateMate A1-EVB-2M is passively cooled; no active fan; this audit runs no "
    "sustained workload and makes no sustained-load performance claim."
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "prior_receipt_paths_and_hashes",
    "current_dated_operator_receipt",
    "prior_and_current_physical_state_hashes",
    "physical_state_changed",
    "hardware_command_authorized",
    "detect_attempt_count_command_stdout_stderr_exit_code",
    "expected_and_observed_idcode",
    "mutation_command_counts",
    "historical_flagged_terminal_evidence_excluded",
    "operator_action_packet",
    "hardware_execution_authenticated",
    "speed_power_energy_terminal_tsu_kona_claim_counts",
    "passive_cooling_note",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
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
    "status": "Terminal status for the cached GateMate action audit.",
    "prior_receipt_paths_and_hashes": (
        "Hashes precede authorization and prevent stale terminal evidence from graduating."
    ),
    "current_dated_operator_receipt": (
        "Only a newer dated physical receipt can move the GateMate state."
    ),
    "prior_and_current_physical_state_hashes": ("Exp6121 is the canonical no-repeat baseline."),
    "physical_state_changed": "A bare bool gates every hardware command.",
    "hardware_command_authorized": "Visibility checks require a material physical delta.",
    "detect_attempt_count_command_stdout_stderr_exit_code": (
        "Zero on cached state; one exact non-destructive detect on changed state."
    ),
    "expected_and_observed_idcode": "IDCODE is visibility evidence, not performance evidence.",
    "mutation_command_counts": (
        "No synthesis, place, route, pack, flash, firmware, SSH, timing, current, or power command is allowed."
    ),
    "historical_flagged_terminal_evidence_excluded": (
        "Adversarial-flagged Exp3866 evidence stays quarantined."
    ),
    "operator_action_packet": "A blocked audit ends with one concrete bench action.",
    "hardware_execution_authenticated": "Detect visibility is not workload execution.",
    "speed_power_energy_terminal_tsu_kona_claim_counts": (
        "No speed, power, energy, terminal, TSU, or Kona claim is permitted."
    ),
    "passive_cooling_note": (
        "GateMate is passively cooled and no sustained-load inference is made."
    ),
    "protected_files_unchanged": ("Conductor and reconciler-owned docs remain byte-identical."),
    "inference_substrate": (
        "Use cached receipt audit plus optional non-destructive detect, not LLM inference."
    ),
    "verifier_is_oracle": ("Raw hashes and IDCODE text are authoritative for this audit only."),
    "missing_verifier_gaps": "Missing IDCODE or workload evidence is recorded instead of inferred.",
    "field_provenance": "Every field traces to receipts, hashes, command output, or tests.",
    "field_principles": "Each required field carries its reason for existence.",
    "test_commands": "Verification commands are recorded.",
    "test_exit_codes": "Exit codes prevent failed checks becoming success.",
    "duration_s": "Measured wall time is reported without padding.",
    "reproducibility_checksum": ("Checksum detects physical-state, receipt, or artifact drift."),
    "honest_verdict": (
        "Use `blocked_no_change:`, `blocked_missing_receipt:`, `blocked_stale_receipt:`, `blocked_idcode:`, or `complete_visible:`."
    ),
}

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6199_gatemate_terminal_action_audit_v537.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6199_gatemate_terminal_action_audit_v537.py -m pytest tests/python/test_experiment_6199_gatemate_terminal_action_audit_v537.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6199_gatemate_terminal_action_audit_v537.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6199_gatemate_terminal_action_audit_v537.py",
)
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


class CommandReceipt:
    """Raw stdout, stderr, and exit evidence for the only allowed detect command."""

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
            "attempt_count": 1,
            "command": command_to_string(self.command),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 6),
            "reason": "newer_material_operator_physical_receipt",
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandReceipt]


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(
    command: tuple[str, ...], timeout_s: float = 30.0
) -> CommandReceipt:  # pragma: no cover
    started = time.perf_counter()
    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return CommandReceipt(
        command, result.returncode, result.stdout, result.stderr, time.perf_counter() - started
    )


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


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
    changed = [path for path, old_hash in before_hashes.items() if after.get(path) != old_hash]
    return {
        "all_unchanged": not changed,
        "changed_paths": changed,
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
    }


def canonical_prior_physical_state(root: Path) -> JsonDict:
    exp6121 = read_json_object(root, EXP6121_REL_PATH)
    state = exp6121.get("prior_and_current_physical_state_hashes", {}).get("current_state")
    if isinstance(state, Mapping):
        current = dict(state)
        current["baseline_artifact"] = EXP6121_REL_PATH.as_posix()
        current["baseline_date"] = str(exp6121.get("run_date") or BASELINE_RECEIPT_DATE)
        return current
    return dict(CANONICAL_PRIOR_PHYSICAL_STATE)


def baseline_receipt_date(root: Path) -> str:
    exp6121 = read_json_object(root, EXP6121_REL_PATH)
    return str(exp6121.get("run_date") or BASELINE_RECEIPT_DATE)


def normalize_operator_receipt(receipt: JsonMap | None, run_date: str) -> JsonDict:
    normalized = dict(DEFAULT_OPERATOR_RECEIPT)
    if receipt is not None:
        normalized.update(dict(receipt))
    normalized["checked_on"] = run_date
    changes = normalized.get("changes")
    normalized["changes"] = list(changes) if isinstance(changes, list) else []
    return normalized


def receipt_authorization_reason(receipt: JsonMap, baseline_date: str) -> str:
    if not receipt.get("exists") or not receipt.get("receipt_date"):
        return "missing_receipt"
    if str(receipt.get("receipt_date")) <= baseline_date:
        return "stale_receipt"
    changed_fields = {
        str(item.get("field")) for item in receipt.get("changes", []) if isinstance(item, Mapping)
    }
    if not changed_fields & MATERIAL_PHYSICAL_FIELDS:
        return "no_material_physical_change"
    return "newer_material_physical_change"


def apply_operator_receipt(prior_state: JsonMap, receipt: JsonMap) -> JsonDict:
    current = dict(prior_state)
    if (
        receipt_authorization_reason(
            receipt, str(prior_state.get("baseline_date") or BASELINE_RECEIPT_DATE)
        )
        == "newer_material_physical_change"
    ):
        for field in MATERIAL_PHYSICAL_FIELDS:
            if receipt.get(field) is not None:
                current[field] = receipt[field]
        if all(current.get(field) == prior_state.get(field) for field in MATERIAL_PHYSICAL_FIELDS):
            return dict(prior_state)
        current["operator_receipt_date"] = receipt.get("receipt_date")
        current["operator_receipt_source"] = receipt.get("source")
        current["operator_receipt_changes"] = list(receipt.get("changes", []))
    return current


def physical_state_comparison(root: Path, receipt: JsonMap) -> tuple[JsonDict, JsonDict, bool, str]:
    prior = canonical_prior_physical_state(root)
    reason = receipt_authorization_reason(receipt, baseline_receipt_date(root))
    current = apply_operator_receipt(prior, receipt)
    changed = reason == "newer_material_physical_change" and stable_hash(prior) != stable_hash(
        current
    )
    if reason == "newer_material_physical_change" and not changed:
        reason = "no_material_physical_change"
    return prior, current, changed, reason


def idcode_from_text(text: str) -> str | None:
    match = re.search(r"\bidcode\s+(0x[0-9a-fA-F]+)", text)
    return match.group(1).lower() if match else None


def no_detect_receipt(reason: str) -> JsonDict:
    return {
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
    hardware_command_authorized: bool,
    authorization_reason: str,
    command_runner: CommandRunner,
) -> tuple[JsonDict, str | None]:
    if not hardware_command_authorized:
        return no_detect_receipt(authorization_reason), None
    probe = command_runner(DETECT_COMMAND, 30.0)
    return probe.as_dict(), idcode_from_text(probe.combined_output)


def cached_tool_identity_receipt(root: Path) -> JsonDict:
    exp6121 = read_json_object(root, EXP6121_REL_PATH)
    tool_versions = exp6121.get("preconditions_checked", {}).get("tool_versions", {})
    return {
        "source": EXP6121_REL_PATH.as_posix(),
        "tool_versions": tool_versions if isinstance(tool_versions, Mapping) else {},
        "sha256": stable_hash(tool_versions),
        "command_execution": "cached_only_no_tool_identity_command_run",
    }


def prior_receipt_paths_and_hashes(root: Path, receipt: JsonMap) -> JsonDict:
    path_hashes = {path.as_posix(): path_receipt(root, path) for path in HASHED_INPUT_PATHS}
    protected_hashes = protected_file_hashes(root)
    tool_identity = cached_tool_identity_receipt(root)
    current_receipt_hash = stable_hash(receipt)
    return {
        "path_hashes": path_hashes,
        "protected_file_hashes": protected_hashes,
        "tool_identity_receipt": tool_identity,
        "current_dated_operator_receipt_sha256": current_receipt_hash,
        "authorization_inputs_sha256": stable_hash(
            {
                "path_hashes": path_hashes,
                "protected_file_hashes": protected_hashes,
                "tool_identity_receipt": tool_identity,
                "current_dated_operator_receipt_sha256": current_receipt_hash,
            }
        ),
        "hashing_completed_before_hardware_command_authorization": True,
    }


def historical_flagged_terminal_evidence_excluded(root: Path) -> JsonDict:
    exp3866 = read_json_object(root, EXP3866_REL_PATH)
    flagged = bool(exp3866.get("flagged_adversarial") or exp3866.get("corrigendum_pending"))
    return {
        "artifact": EXP3866_REL_PATH.as_posix(),
        "excluded": flagged,
        "flagged_adversarial": bool(exp3866.get("flagged_adversarial")),
        "corrigendum_pending": exp3866.get("corrigendum_pending", []),
        "clean_terminal_evidence_used": False,
        "reason": (
            "Exp3866 is historical context only because it is adversarial-flagged."
            if flagged
            else "Exp3866 is historical context only for this audit."
        ),
    }


def hardware_execution_authenticated(observed_idcode: str | None) -> JsonDict:
    idcode_match = observed_idcode == EXPECTED_IDCODE
    return {
        "authenticated": False,
        "idcode_visibility_authenticated": idcode_match,
        "board_execution_authenticated": False,
        "detect_is_not_workload_execution": True,
        "simulation_used_as_board_execution": False,
    }


def operator_action_packet(status: str) -> JsonDict:
    if status in {"blocked_no_change", "blocked_missing_receipt", "blocked_stale_receipt"}:
        return dict(EXACT_OPERATOR_ACTION_PACKET)
    return {
        "packet_id": "gatemate-idcode-visibility-followup-v537",
        "board": "Cologne Chip GateMate A1-EVB-2M",
        "blocked_command": "any flash, synthesis, place, route, pack, firmware, SSH, timing, current, or power command",
        "required_physical_delta": "resolve GateMate IDCODE visibility before any terminal claim",
        "operator_steps": [
            "Check GateMate power and onboard DirtyJTAG USB connection.",
            "Record a new dated receipt before any further detect attempt.",
        ],
        "do_not_do": list(EXACT_OPERATOR_ACTION_PACKET["do_not_do"]),
    }


def artifact_status(
    *,
    authorization_reason: str,
    physical_state_changed: bool,
    observed_idcode: str | None,
) -> str:
    if authorization_reason == "missing_receipt":
        return "blocked_missing_receipt"
    if authorization_reason == "stale_receipt":
        return "blocked_stale_receipt"
    if not physical_state_changed:
        return "blocked_no_change"
    if observed_idcode == EXPECTED_IDCODE:
        return "complete_visible"
    return "blocked_idcode"


def missing_verifier_gaps(
    *,
    status: str,
    observed_idcode: str | None,
) -> list[str]:
    gaps = ["no_host_io_workload_execution_receipt"]
    if status in {"blocked_no_change", "blocked_missing_receipt", "blocked_stale_receipt"}:
        gaps.append("no_new_raw_idcode_due_to_physical_state_gate")
    if observed_idcode != EXPECTED_IDCODE:
        gaps.append("expected_gatemate_idcode_not_observed")
    return gaps


def honest_verdict(status: str) -> str:
    verdicts = {
        "blocked_no_change": (
            "blocked_no_change: GateMate physical state matches the Exp6121 baseline; "
            "zero hardware commands run and operator physical-action packet emitted"
        ),
        "blocked_missing_receipt": (
            "blocked_missing_receipt: no newer dated GateMate physical receipt exists; "
            "zero hardware commands run and Exp3866 remains excluded"
        ),
        "blocked_stale_receipt": (
            "blocked_stale_receipt: supplied GateMate physical receipt is not newer "
            "than Exp6121; zero hardware commands run"
        ),
        "blocked_idcode": (
            "blocked_idcode: one authorized non-destructive detect ran but expected "
            "GateMate GM1Ax IDCODE was not observed"
        ),
        "complete_visible": (
            "complete_visible: one authorized non-destructive detect observed the "
            "GateMate GM1Ax IDCODE; no execution or performance claim made"
        ),
    }
    return verdicts[status]


def field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-HW-6199 / SCENARIO-HW-6199-* and cached local receipts",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: JsonMap) -> str:
    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "test_exit_codes", "reproducibility_checksum"}
    }
    return stable_hash(stable).removeprefix("sha256:")


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    current_dated_operator_receipt: JsonMap | None = None,
    protected_before_hashes: JsonMap | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: JsonMap | None = None,
) -> JsonDict:
    started = clock()
    source_root = Path(root)
    receipt = normalize_operator_receipt(current_dated_operator_receipt, run_date)
    prior_state, current_state, changed, reason = physical_state_comparison(source_root, receipt)
    hardware_authorized = changed and reason == "newer_material_physical_change"
    detect_receipt, observed_idcode = run_optional_detect(
        hardware_command_authorized=hardware_authorized,
        authorization_reason=reason,
        command_runner=command_runner,
    )
    status = artifact_status(
        authorization_reason=reason,
        physical_state_changed=changed,
        observed_idcode=observed_idcode,
    )
    protected_before = dict(protected_before_hashes or protected_file_hashes(source_root))
    commands = list(test_commands) if test_commands is not None else list(TEST_COMMANDS)
    exit_codes = dict(test_exit_codes) if test_exit_codes is not None else dict(TEST_EXIT_CODES)
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
        "prior_receipt_paths_and_hashes": prior_receipt_paths_and_hashes(source_root, receipt),
        "current_dated_operator_receipt": receipt,
        "prior_and_current_physical_state_hashes": {
            "prior": stable_hash(prior_state),
            "current": stable_hash(current_state),
            "prior_state": prior_state,
            "current_state": current_state,
            "authorization_reason": reason,
        },
        "physical_state_changed": changed,
        "hardware_command_authorized": hardware_authorized,
        "detect_attempt_count_command_stdout_stderr_exit_code": detect_receipt,
        "expected_and_observed_idcode": {
            "expected_idcode": EXPECTED_IDCODE,
            "observed_idcode": observed_idcode,
            "matches": observed_idcode == EXPECTED_IDCODE,
        },
        "mutation_command_counts": dict(ZERO_MUTATION_COMMAND_COUNTS),
        "historical_flagged_terminal_evidence_excluded": (
            historical_flagged_terminal_evidence_excluded(source_root)
        ),
        "operator_action_packet": operator_action_packet(status),
        "hardware_execution_authenticated": hardware_execution_authenticated(observed_idcode),
        "speed_power_energy_terminal_tsu_kona_claim_counts": dict(ZERO_CLAIM_COUNTS),
        "passive_cooling_note": PASSIVE_COOLING_NOTE,
        "protected_files_unchanged": protected_files_unchanged(source_root, protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "missing_verifier_gaps": missing_verifier_gaps(
            status=status, observed_idcode=observed_idcode
        ),
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "duration_s": round(float(clock() - started), 6),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(status)
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
    if tuple(artifact.get("spec_refs", ())) != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")

    detect = artifact.get("detect_attempt_count_command_stdout_stderr_exit_code")
    if not isinstance(detect, Mapping):
        return [*errors, "detect receipt must be a mapping"]
    attempt_count = detect.get("attempt_count")
    command = detect.get("command")
    changed = artifact.get("physical_state_changed")
    authorized = artifact.get("hardware_command_authorized")
    if not isinstance(attempt_count, int) or attempt_count > 1:
        errors.append("at most one detect attempt is permitted")
    if attempt_count == 1:
        if command != command_to_string(DETECT_COMMAND):
            errors.append("detect command allowlist violation")
        if changed is not True or authorized is not True:
            errors.append("detect command recorded in unauthorized state")
    if authorized is False and (attempt_count != 0 or command is not None):
        errors.append("unauthorized state must have zero detect attempts")
    if changed is False:
        if attempt_count != 0 or command is not None:
            errors.append("unchanged physical state must have zero command receipts")
        if detect.get("stdout") != "" or detect.get("stderr") != "":
            errors.append("unchanged physical state must not carry new stdout/stderr")
        if artifact.get("operator_action_packet") != EXACT_OPERATOR_ACTION_PACKET:
            errors.append("unchanged physical state must emit exact operator action packet")
    if artifact.get("mutation_command_counts") != ZERO_MUTATION_COMMAND_COUNTS:
        errors.append("mutation command counts must remain zero")
    if artifact.get("speed_power_energy_terminal_tsu_kona_claim_counts") != ZERO_CLAIM_COUNTS:
        errors.append("speed/power/energy/terminal/tsu/kona claim counts must remain zero")
    historical = artifact.get("historical_flagged_terminal_evidence_excluded")
    if not isinstance(historical, Mapping) or historical.get("excluded") is not True:
        errors.append("historical flagged terminal evidence must be excluded")
    auth = artifact.get("hardware_execution_authenticated")
    if not isinstance(auth, Mapping) or auth.get("authenticated") is not False:
        errors.append("hardware execution must remain unauthenticated")
    protected = artifact.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        errors.append("protected files must be unchanged")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(
        (
            "blocked_no_change:",
            "blocked_missing_receipt:",
            "blocked_stale_receipt:",
            "blocked_idcode:",
            "complete_visible:",
        )
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
    return atomic_write_json(
        OUTPUT_REL_PATH,
        artifact,
        root=repo_root,
        allow_override=False,
        sort_keys=True,
    )


def run_experiment(
    *,
    repo_root: Path = REPO_ROOT,
    source_root: Path | None = None,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = RUN_DATE,
    current_dated_operator_receipt: JsonMap | None = None,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: JsonMap | None = None,
) -> Path:
    source = Path(source_root) if source_root is not None else Path(repo_root)
    protected_before = protected_file_hashes(source)
    artifact = build_artifact(
        root=source,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        current_dated_operator_receipt=current_dated_operator_receipt,
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
    detect = artifact["detect_attempt_count_command_stdout_stderr_exit_code"]
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"physical_state_changed: {artifact['physical_state_changed']}")
    print(f"detect_attempt_count: {detect['attempt_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
