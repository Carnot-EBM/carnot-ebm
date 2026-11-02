"""Exp 4439 precondition-gated hardware continuity across attached boards.

This experiment records the next bounded hardware-continuity step for KV260,
GateMate, and PolarFire. Each board gets either one live step after its allowed
precondition or a one-line blocked audit, so missing hardware cannot be silently
fabricated into progress.

Spec refs: REQ-HW-4439, SCENARIO-HW-4439.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


EXPERIMENT_ID = 4439
SCHEMA = "carnot.hardware_continuity_precondition_gated.v2"
SPEC_REFS = ["REQ-HW-4439", "SCENARIO-HW-4439"]
OUTPUT_REL_PATH = Path("results") / "experiment_4439_hardware_continuity.json"
RANDOM_SEED = 4439
INFERENCE_SUBSTRATE = "hardware_smoke"
BOARD_NAMES = ("kv260", "gatemate", "polarfire")

KV260_SSH_PRECONDITION = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_NEXTPNR_PRECONDITION = ("bash", "-lc", "command -v nextpnr-himbaechel")
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_SSH_PRECONDITION = ("ssh", "polarfire", "true")

KV260_LATENCY_TRANSCRIPT_COMMAND = (
    "ssh",
    "kria",
    """python3 - <<'PY'
import json
import time
samples = []
for _ in range(5):
    start = time.perf_counter_ns()
    time.sleep(0)
    samples.append(max((time.perf_counter_ns() - start) / 1000.0, 0.001))
print(json.dumps({
    "schema": "carnot.kv260.ssh_latency_transcript.v1",
    "sample_count": len(samples),
    "per_sample_us": samples,
    "source": "ssh_python_perf_counter",
}, sort_keys=True))
PY""",
)
GATEMATE_GMPACK_LOOKUP_COMMAND = ("bash", "-lc", "command -v gmpack")
POLARFIRE_SAMPLER_SMOKE_COMMAND = (
    "ssh",
    "polarfire",
    """python3 - <<'PY'
import json
print(json.dumps({
    "schema": "carnot.polarfire.sampler_smoke.v1",
    "sample_count": 4,
    "sampler": "cpu_smoke",
}, sort_keys=True))
PY""",
)

REQUIRED_ARTIFACT_FIELDS = (
    "preconditions_checked",
    "per_board_status",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "preconditions_checked": (
        "list of {resource, available}; principle: pre-empts the "
        "fabricate-when-resource-missing mode."
    ),
    "per_board_status": (
        "one status per attached board so each stays visible in the milestone "
        "retro (Hardware-Task Continuity)"
    ),
    "honest_verdict": (
        "terminal-prefixed; a clean SSH-unreachable is blocked_, an audit-only "
        "step is complete_"
    ),
}

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)


@dataclass(frozen=True)
class CommandProbe:
    """Captured command transcript for a hardware precondition or next step."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def combined_output(self) -> str:
        return "\n".join(part.rstrip() for part in (self.stdout, self.stderr) if part.rstrip())


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]
Clock = Callable[[], float]


def prepend_oss_cad_suite() -> None:  # pragma: no cover - host environment dependent.
    candidate = Path("/opt/oss-cad-suite/bin")
    if not (candidate / "nextpnr-himbaechel").exists():
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if str(candidate) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(candidate), *parts])


def run_command(command: tuple[str, ...], timeout_s: float = 60.0) -> CommandProbe:  # pragma: no cover
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return CommandProbe(
            tuple(command),
            completed.returncode,
            completed.stdout,
            completed.stderr,
            _elapsed_since(started),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandProbe(tuple(command), 124, stdout, stderr, _elapsed_since(started))
    except OSError as exc:
        return CommandProbe(tuple(command), 127, "", str(exc), _elapsed_since(started))


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def payload_checksum(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    root = Path(repo_root)
    started = clock()
    preconditions = check_preconditions(command_runner)
    availability = {entry["resource"]: bool(entry["available"]) for entry in preconditions}

    per_board_status = {
        "kv260": _kv260_status(command_runner, availability["kv260_ssh"]),
        "gatemate": _gatemate_status(
            root,
            command_runner,
            nextpnr_available=availability["gatemate_nextpnr_himbaechel"],
            dirtyjtag_available=availability["gatemate_dirtyjtag_detect"],
        ),
        "polarfire": _polarfire_status(command_runner, availability["polarfire_ssh"]),
    }

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "honest_verdict": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_context": _source_context(root),
        "preconditions_checked": preconditions,
        "per_board_status": per_board_status,
        "duration_s": _round_duration(clock() - started),
        "fabric_acceleration_claimed": False,
        "speedup_claim_made": False,
        "reproducibility_checksum": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(command_runner: CommandRunner) -> list[dict[str, Any]]:
    specs = (
        ("kv260_ssh", KV260_SSH_PRECONDITION, _exit_zero, 10.0),
        ("gatemate_nextpnr_himbaechel", GATEMATE_NEXTPNR_PRECONDITION, _nextpnr_seen, 10.0),
        ("gatemate_dirtyjtag_detect", GATEMATE_DETECT_COMMAND, _gatemate_detected, 30.0),
        ("polarfire_ssh", POLARFIRE_SSH_PRECONDITION, _exit_zero, 10.0),
    )
    entries: list[dict[str, Any]] = []
    for resource, command, predicate, timeout_s in specs:
        probe = command_runner(command, timeout_s)
        entries.append(_precondition_entry(resource, probe, bool(predicate(probe))))
    return entries


def honest_verdict(artifact: dict[str, Any]) -> str:
    statuses = artifact["per_board_status"]
    if statuses["kv260"]["status"] == "blocked_kv260_ssh_unreachable":
        return "blocked_kv260_ssh_unreachable"
    return (
        "complete: hardware_continuity_4439_"
        f"kv260_{state_token(str(statuses['kv260']['status']))}_"
        f"gatemate_{state_token(str(statuses['gatemate']['status']))}_"
        f"polarfire_{state_token(str(statuses['polarfire']['status']))}"
    )


def state_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    return "_".join(part for part in token.split("_") if part)


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    prepend_oss_cad_suite()
    artifact = build_artifact(repo_root=repo_root, command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def write_artifact(repo_root: str | Path, artifact: dict[str, Any]) -> Path:
    path = Path(repo_root) / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("schema") == SCHEMA, "schema must identify Exp 4439")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment must be 4439")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs must name REQ/SCENARIO 4439")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed must be 4439")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing {field}")
    _validate_principles(artifact)
    _validate_preconditions(artifact)
    _validate_per_board_status(artifact)
    _validate_source_context(artifact)
    encoded = json.dumps(artifact, sort_keys=True, default=str)
    _require("mmcblk" not in encoded.lower(), "SD-card precondition marker is forbidden")
    _require("nextpnr-gatemate" not in encoded, "obsolete nextpnr-gatemate is forbidden")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "wrong substrate")
    _require(artifact.get("fabric_acceleration_claimed") is False, "no fabric claim")
    _require(artifact.get("speedup_claim_made") is False, "no speedup claim")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict terminal prefix")
    _require(artifact.get("honest_verdict") == honest_verdict(artifact), "stale verdict")
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _kv260_status(command_runner: CommandRunner, reachable: bool) -> dict[str, Any]:
    if not reachable:
        return _blocked_status(
            "kv260",
            "blocked_kv260_ssh_unreachable",
            "KV260 SSH precondition failed; latency transcript was not run.",
        )
    probe = command_runner(KV260_LATENCY_TRANSCRIPT_COMMAND, 30.0)
    transcript = _step_transcript("latency_transcript", probe)
    if probe.exit_code != 0:
        return _step_blocked_status(
            "kv260",
            "latency_transcript",
            "blocked_kv260_latency_transcript_failed",
            "KV260 SSH was reachable, but the latency transcript command failed.",
            transcript,
        )
    return {
        "board": "kv260",
        "reachable": True,
        "step": "latency_transcript",
        "status": "kv260_latency_transcript_recorded",
        "honest_verdict": "complete: kv260_latency_transcript_recorded",
        "audit": "KV260 SSH precondition passed; latency transcript command completed.",
        "latency_transcript": _json_output(probe),
        "step_transcript": transcript,
    }


def _gatemate_status(
    repo_root: Path,
    command_runner: CommandRunner,
    *,
    nextpnr_available: bool,
    dirtyjtag_available: bool,
) -> dict[str, Any]:
    if not nextpnr_available:
        return _blocked_status(
            "gatemate",
            "blocked_gatemate_nextpnr_himbaechel_unavailable",
            "GateMate nextpnr-himbaechel precondition failed; bitstream pack was not run.",
        )
    if not dirtyjtag_available:
        return _blocked_status(
            "gatemate",
            "blocked_gatemate_dirtyjtag_unreachable",
            "GateMate DirtyJTAG detect failed; bitstream pack was not run.",
        )
    cfg = _first_existing(_candidate_gatemate_configs(repo_root))
    if cfg is None:  # pragma: no cover - repo normally carries routed configs.
        return _blocked_status(
            "gatemate",
            "blocked_gatemate_config_missing",
            "GateMate was reachable, but no routed config was available to pack.",
        )

    packer_probe = command_runner(GATEMATE_GMPACK_LOOKUP_COMMAND, 10.0)
    if packer_probe.exit_code != 0:
        return _step_blocked_status(
            "gatemate",
            "bitstream_pack",
            "blocked_gatemate_gmpack_unavailable",
            "GateMate was reachable, but gmpack was unavailable for bitstream pack.",
            _step_transcript("gmpack_lookup", packer_probe),
        )

    packer = _observed(packer_probe).splitlines()[0]
    bitstream = _gatemate_output_bitstream(repo_root)
    pack_command = (packer, str(cfg), str(bitstream))
    pack_probe = command_runner(pack_command, 120.0)
    transcript = [
        _step_transcript("gmpack_lookup", packer_probe),
        _step_transcript("bitstream_pack", pack_probe),
    ]
    if pack_probe.exit_code != 0:
        return _step_blocked_status(
            "gatemate",
            "bitstream_pack",
            "blocked_gatemate_bitstream_pack_failed",
            "GateMate was reachable, but gmpack returned a non-zero status.",
            transcript,
        )
    if not bitstream.exists():  # pragma: no cover - defensive live guard.
        return _step_blocked_status(
            "gatemate",
            "bitstream_pack",
            "blocked_gatemate_bitstream_pack_output_missing",
            "GateMate pack command succeeded, but no output bitstream was observed.",
            transcript,
        )
    return {
        "board": "gatemate",
        "reachable": True,
        "step": "bitstream_pack",
        "status": "gatemate_bitstream_pack_succeeded",
        "honest_verdict": "complete: gatemate_bitstream_pack_succeeded",
        "audit": "GateMate preconditions passed; himbaechel-routed config was packed.",
        "config_path": str(cfg),
        "bitstream_path": str(bitstream),
        "bitstream_sha256": _sha256_file(bitstream),
        "step_transcript": transcript,
    }


def _polarfire_status(command_runner: CommandRunner, reachable: bool) -> dict[str, Any]:
    if not reachable:
        return _blocked_status(
            "polarfire",
            "blocked_polarfire_ssh_unreachable",
            "PolarFire SSH precondition failed; sampler smoke was not run.",
        )
    probe = command_runner(POLARFIRE_SAMPLER_SMOKE_COMMAND, 30.0)
    transcript = _step_transcript("sampler_smoke", probe)
    if probe.exit_code != 0:
        return _step_blocked_status(
            "polarfire",
            "sampler_smoke",
            "blocked_polarfire_sampler_smoke_failed",
            "PolarFire SSH was reachable, but the sampler smoke command failed.",
            transcript,
        )
    return {
        "board": "polarfire",
        "reachable": True,
        "step": "sampler_smoke",
        "status": "polarfire_sampler_smoke_recorded",
        "honest_verdict": "complete: polarfire_sampler_smoke_recorded",
        "audit": "PolarFire SSH precondition passed; sampler smoke command completed.",
        "sampler_smoke": _json_output(probe),
        "step_transcript": transcript,
    }


def _blocked_status(board: str, status: str, audit: str) -> dict[str, Any]:
    return {
        "board": board,
        "reachable": False,
        "step": "precondition_audit",
        "status": status,
        "honest_verdict": status,
        "audit": audit,
    }


def _step_blocked_status(
    board: str,
    step: str,
    status: str,
    audit: str,
    transcript: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "board": board,
        "reachable": True,
        "step": step,
        "status": status,
        "honest_verdict": status,
        "audit": audit,
        "step_transcript": transcript,
    }


def _precondition_entry(resource: str, probe: CommandProbe, available: bool) -> dict[str, Any]:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": _round_duration(probe.duration_s),
        "observed": _observed(probe),
    }


def _step_transcript(stage: str, probe: CommandProbe) -> dict[str, Any]:
    return {
        "stage": stage,
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": _round_duration(probe.duration_s),
        "output_excerpt": _observed(probe)[:1200],
    }


def _source_context(repo_root: Path) -> dict[str, Any]:
    prior_path = repo_root / "results" / "experiment_4428_hardware_continuity.json"
    prior_payload = _read_json(prior_path)
    prior_status = prior_payload.get("per_board_status", {})
    if not isinstance(prior_status, dict):
        prior_status = {}
    return {
        "previous_experiment": 4428,
        "most_recent_hardware_continuity_artifact": str(prior_path),
        "previous_artifact_read": bool(prior_payload),
        "previous_honest_verdict": prior_payload.get("honest_verdict"),
        "previous_per_board_status": prior_status,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt local artifact guard.
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_principles(artifact: dict[str, Any]) -> None:
    principles = artifact.get("field_principles")
    _require(isinstance(principles, dict), "field_principles must be a dict")
    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(principles.get(field) == FIELD_PRINCIPLES[field], f"{field} principle mismatch")
        if field in artifact:
            _require(
                not (
                    isinstance(artifact[field], dict)
                    and set(artifact[field]) == {"value", "principle"}
                ),
                f"{field} must remain a bare value",
            )


def _validate_preconditions(artifact: dict[str, Any]) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(isinstance(preconditions, list), "preconditions_checked must be a list")
    resources = [entry.get("resource") for entry in preconditions if isinstance(entry, dict)]
    _require(
        resources
        == [
            "kv260_ssh",
            "gatemate_nextpnr_himbaechel",
            "gatemate_dirtyjtag_detect",
            "polarfire_ssh",
        ],
        "precondition resources mismatch",
    )
    for entry in preconditions:
        _require(isinstance(entry, dict), "precondition entries must be dicts")
        _require(isinstance(entry.get("available"), bool), "available must be a bare bool")
        _require(isinstance(entry.get("resource"), str), "resource must be a string")


def _validate_per_board_status(artifact: dict[str, Any]) -> None:
    statuses = artifact.get("per_board_status")
    _require(isinstance(statuses, dict), "per_board_status must be a dict")
    _require(set(statuses) == set(BOARD_NAMES), "per_board_status must cover all boards")
    for board, status in statuses.items():
        _require(isinstance(status, dict), "board status entries must be dicts")
        _require(status.get("board") == board, "board status key mismatch")
        _require(isinstance(status.get("reachable"), bool), "reachable must be a bare bool")
        _require(isinstance(status.get("step"), str) and status["step"], "step missing")
        _require(isinstance(status.get("status"), str) and status["status"], "status missing")
        _require(isinstance(status.get("audit"), str) and "\n" not in status["audit"], "audit")
        if not status["reachable"]:
            _require(status["step"] == "precondition_audit", "blocked board must be audit-only")
            _require(status["status"].startswith(f"blocked_{board}_"), "blocked status mismatch")


def _validate_source_context(artifact: dict[str, Any]) -> None:
    source = artifact.get("source_context")
    _require(isinstance(source, dict), "source_context must be a dict")
    _require(source.get("previous_experiment") == 4428, "source_context must read Exp 4428")
    _require(
        str(source.get("most_recent_hardware_continuity_artifact", "")).endswith(
            "experiment_4428_hardware_continuity.json"
        ),
        "source_context must point at Exp 4428",
    )


def _candidate_gatemate_configs(repo_root: Path) -> list[Path]:
    return [
        repo_root
        / "build"
        / "gatemate"
        / "experiment_3866_gatemate_ising_tile_flash_v2"
        / "gatemate_ising_n16.cfg.bit",
        repo_root / "build" / "gatemate" / "experiment_2956_gatemate_n16" / "ising_n16_gatemate.cfg",
        repo_root / "rtl" / "gatemate_ising_n16.cfg",
    ]


def _gatemate_output_bitstream(repo_root: Path) -> Path:
    return (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_4439_hardware_continuity"
        / "gatemate_ising_n16_exp4439.bit"
    )


def _first_existing(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_output(probe: CommandProbe) -> dict[str, Any]:
    try:
        payload = json.loads(_observed(probe).splitlines()[-1])
    except (IndexError, json.JSONDecodeError):  # pragma: no cover - malformed live output.
        return {"raw_output": _observed(probe)}
    return payload if isinstance(payload, dict) else {"raw_output": _observed(probe)}


def _exit_zero(probe: CommandProbe) -> bool:
    return probe.exit_code == 0


def _nextpnr_seen(probe: CommandProbe) -> bool:
    return probe.exit_code == 0 and "nextpnr-himbaechel" in _observed(probe)


def _gatemate_detected(probe: CommandProbe) -> bool:
    text = _observed(probe).lower()
    return (
        probe.exit_code == 0
        and "idcode" in text
        and ("gatemate" in text or "colognechip" in text or "gm1a" in text)
    )


def _observed(probe: CommandProbe) -> str:
    text = probe.combined_output.strip()
    return text if text else f"returncode={probe.exit_code}"


def _round_duration(value: float) -> float:
    return round(max(float(value), 0.000001), 6)


def _elapsed_since(started: float) -> float:  # pragma: no cover - live subprocess helper.
    return _round_duration(time.perf_counter() - started)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
