#!/usr/bin/env python3
"""Exp 5052 KV260 p-bit timing-ratio parity packet.

Spec refs: REQ-HW-5052, SCENARIO-HW-5052.

This experiment records a local CPU reference and an SSH-attached KV260 board
run for the same bounded n=64 p-bit-style parity workload. The packet is local
evidence only: it does not claim a general FPGA speedup, a GPU benchmark, or an
external paper result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))


JsonDict = dict[str, Any]
Clock = Callable[[], float]

EXPERIMENT_ID = 5052
SCHEMA = "carnot.kv260_pbit_timing_ratio.v1"
SPEC_REFS = ["REQ-HW-5052", "SCENARIO-HW-5052"]
OUTPUT_REL_PATH = Path("results") / "experiment_5052_kv260_pbit_timing_ratio.json"
RANDOM_SEED = 5052
INFERENCE_SUBSTRATE = "hardware_smoke"
WORKLOAD_NAME = "bounded_sparse_pbit_parity_n64"
N_VARIABLES = 64
ITERATIONS = 128
LOCAL_CLAIM_SCOPE = (
    "local_ssh_attached_kv260_python_parity_workload_only_"
    "no_general_fpga_speedup_claim_no_gpu_benchmark_claim_no_external_2026_paper_claim"
)

KV260_SSH_COMMAND = ("ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", "kria", "true")
KV260_LISTAPPS_COMMAND = ("ssh", "kria", "xmutil listapps")
KV260_LISTAPPS_SUDO_COMMAND = ("ssh", "kria", "sudo xmutil listapps")
KV260_UIO_COMMAND = ("ssh", "kria", "ls /dev/uio*")

VALID_OVERLAYS = (
    "carnot_ising_v4_n64",
    "carnot_ising_v2_n64",
    "carnot_ising_v4",
    "carnot_ising",
)

PBIT_WORKLOAD_CODE = f"""
import hashlib
import json
import time

n = {N_VARIABLES}
iterations = {ITERATIONS}
started = time.perf_counter()
spins = [1 if index % 2 == 0 else -1 for index in range(n)]
fields = [((index * 17) % 7) - 3 for index in range(n)]
flips = 0
for step in range(iterations):
    for index in range(n):
        left = spins[(index - 1) % n]
        right = spins[(index + 1) % n]
        long_edge = spins[(index + 7) % n]
        local_field = fields[index] + 2 * left - right + long_edge
        noise = ((1103515245 * (step * n + index + {RANDOM_SEED}) + 12345) >> 16) & 7
        new_spin = 1 if local_field + noise - 3 >= 0 else -1
        if new_spin != spins[index]:
            flips += 1
        spins[index] = new_spin
energy = 0
for index in range(n):
    energy += -spins[index] * spins[(index + 1) % n]
    energy += spins[index] * spins[(index + 7) % n]
    energy += -fields[index] * spins[index]
checksum = hashlib.sha256(json.dumps(spins, separators=(",", ":")).encode()).hexdigest()
print(json.dumps({{
    "workload_name": "{WORKLOAD_NAME}",
    "n_variables": n,
    "iterations": iterations,
    "flips": flips,
    "energy": energy,
    "final_state_checksum": checksum,
    "duration_s": max(time.perf_counter() - started, 0.000001),
}}, sort_keys=True))
""".strip()
KV260_PBIT_WORKLOAD_COMMAND = ("ssh", "kria", f"python3 -c {shlex.quote(PBIT_WORKLOAD_CODE)}")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "kv260_ssh_reachable",
    "overlay_loaded",
    "workload_name",
    "n_variables",
    "timing_ratio_packet_built",
    "cpu_reference_ok",
    "kv260_result_ok",
    "local_claim_scope",
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "inference_substrate",
    "preconditions_checked",
    "overlay_state",
    "loaded_overlay",
    "uio_devices",
    "xmutil_requires_sudo",
    "cpu_reference",
    "kv260_workload",
    "timing_ratio_packet",
    "gpu_reference",
    "duration_s",
    "command_probes",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix for blocked SSH, overlay-not-loaded, parity failure, or packet-built outcomes.",
    "kv260_ssh_reachable": "true only when the SSH BatchMode precondition exits zero; never host SD-card.",
    "overlay_loaded": "true only when xmutil output shows a loaded Carnot Ising overlay.",
    "workload_name": "the bounded p-bit-style parity workload used on CPU and KV260.",
    "n_variables": "the variable count for the bounded parity workload.",
    "timing_ratio_packet_built": "true only when CPU and KV260 outputs pass parity checks.",
    "cpu_reference_ok": "true only when the local deterministic CPU reference is structurally valid.",
    "kv260_result_ok": "true only when the SSH board workload exits zero and matches CPU parity.",
    "local_claim_scope": "limits the claim to local SSH evidence without FPGA, GPU, or paper-result speedup claims.",
}


@dataclass(frozen=True)
class CommandProbe:
    """One command transcript with wall-clock timing."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}{self.stderr}"

    def as_dict(self) -> JsonDict:
        return {
            "command": command_to_string(self.command),
            "exit_code": int(self.exit_code),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "combined_output": self.combined_output,
            "duration_s": float(self.duration_s),
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandProbe]


def command_to_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_command(
    command: tuple[str, ...], timeout_s: float = 60.0
) -> CommandProbe:  # pragma: no cover - live SSH wrapper
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
        return CommandProbe(
            tuple(command),
            int(completed.returncode),
            completed.stdout,
            completed.stderr,
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandProbe(
            tuple(command),
            124,
            exc.stdout or "",
            exc.stderr or f"command timed out after {timeout_s}s",
            time.perf_counter() - started,
        )


def run_pbit_reference(
    *,
    clock: Clock = time.perf_counter,
    n_variables: int = N_VARIABLES,
    iterations: int = ITERATIONS,
) -> JsonDict:
    """Run the deterministic n=64 p-bit-style parity reference on local CPU."""

    started = clock()
    spins = [1 if index % 2 == 0 else -1 for index in range(n_variables)]
    fields = [((index * 17) % 7) - 3 for index in range(n_variables)]
    flips = 0
    for step in range(iterations):
        for index in range(n_variables):
            local_field = (
                fields[index]
                + 2 * spins[(index - 1) % n_variables]
                - spins[(index + 1) % n_variables]
                + spins[(index + 7) % n_variables]
            )
            noise = ((1103515245 * (step * n_variables + index + RANDOM_SEED) + 12345) >> 16) & 7
            new_spin = 1 if local_field + noise - 3 >= 0 else -1
            if new_spin != spins[index]:
                flips += 1
            spins[index] = new_spin
    energy = 0
    for index in range(n_variables):
        energy += -spins[index] * spins[(index + 1) % n_variables]
        energy += spins[index] * spins[(index + 7) % n_variables]
        energy += -fields[index] * spins[index]
    checksum = hashlib.sha256(json.dumps(spins, separators=(",", ":")).encode()).hexdigest()
    return {
        "workload_name": WORKLOAD_NAME,
        "n_variables": int(n_variables),
        "iterations": int(iterations),
        "flips": int(flips),
        "energy": int(energy),
        "final_state_checksum": checksum,
        "duration_s": _duration_positive(clock() - started),
    }


def parse_workload_stdout(stdout: str) -> JsonDict | None:
    """Extract the final JSON object printed by the SSH p-bit workload."""

    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def parse_uio_devices(text: str) -> list[str]:
    devices: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"/dev/uio\d+", text):
        device = match.group(0)
        if device not in seen:
            devices.append(device)
            seen.add(device)
    return devices


def loaded_overlay_from_xmutil(text: str) -> str | None:
    for line in text.splitlines():
        lowered = line.lower()
        for overlay in VALID_OVERLAYS:
            if overlay not in line:
                continue
            if "running" in lowered or "loaded" in lowered:
                return overlay
            if "->" in line and not line.rstrip().endswith("-1"):
                return overlay
    return None


def build_artifact(
    *,
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    """Build the Exp 5052 artifact from local CPU work and SSH board probes."""

    started = clock()
    cpu_reference = run_pbit_reference(clock=clock)
    ssh_probe = command_runner(KV260_SSH_COMMAND, 10.0)
    artifact = _base_artifact(
        cpu_reference=cpu_reference,
        ssh_probe=ssh_probe,
        duration_s=clock() - started,
    )

    if ssh_probe.exit_code != 0:
        return _finalize(artifact)

    command_probes = _command_probes(artifact)
    listapps_probe = command_runner(KV260_LISTAPPS_COMMAND, 30.0)
    command_probes["kv260_xmutil_listapps"] = listapps_probe.as_dict()
    listapps_text = listapps_probe.combined_output if listapps_probe.exit_code == 0 else ""
    requires_sudo = _xmutil_requires_root(listapps_probe)
    if requires_sudo:
        sudo_probe = command_runner(KV260_LISTAPPS_SUDO_COMMAND, 30.0)
        command_probes["kv260_xmutil_listapps_sudo"] = sudo_probe.as_dict()
        if sudo_probe.exit_code == 0:
            listapps_text = sudo_probe.combined_output

    uio_probe = command_runner(KV260_UIO_COMMAND, 10.0)
    command_probes["kv260_uio_devices"] = uio_probe.as_dict()
    loaded_overlay = loaded_overlay_from_xmutil(listapps_text)
    uio_devices = parse_uio_devices(uio_probe.combined_output if uio_probe.exit_code == 0 else "")
    artifact.update(
        {
            "honest_verdict": "success_kv260_reachable_overlay_not_loaded",
            "kv260_ssh_reachable": True,
            "overlay_loaded": loaded_overlay is not None,
            "overlay_state": _overlay_state(
                listapps_probe=listapps_probe,
                listapps_text=listapps_text,
                loaded_overlay=loaded_overlay,
                uio_probe=uio_probe,
                uio_devices=uio_devices,
                requires_sudo=requires_sudo,
            ),
            "loaded_overlay": loaded_overlay,
            "uio_devices": uio_devices,
            "xmutil_requires_sudo": requires_sudo,
            "duration_s": _duration_floor(clock() - started),
        }
    )

    if loaded_overlay is None:
        return _finalize(artifact)

    workload_probe = command_runner(KV260_PBIT_WORKLOAD_COMMAND, 30.0)
    command_probes["kv260_pbit_workload"] = workload_probe.as_dict()
    kv260_workload = (
        parse_workload_stdout(workload_probe.stdout) if workload_probe.exit_code == 0 else None
    )
    kv260_result_ok = _workload_matches(cpu_reference, kv260_workload)
    timing_ratio_packet = (
        _timing_ratio_packet(cpu_reference, kv260_workload, workload_probe)
        if artifact["cpu_reference_ok"] and kv260_result_ok
        else None
    )
    artifact.update(
        {
            "honest_verdict": "success_kv260_pbit_timing_ratio_packet_built"
            if timing_ratio_packet is not None
            else "success_kv260_reachable_overlay_loaded_parity_failed",
            "kv260_result_ok": kv260_result_ok,
            "kv260_workload": kv260_workload,
            "timing_ratio_packet_built": timing_ratio_packet is not None,
            "timing_ratio_packet": timing_ratio_packet,
            "duration_s": _duration_floor(clock() - started),
        }
    )
    return _finalize(artifact)


def write_artifact(repo_root: str | Path, artifact: Mapping[str, Any]) -> Path:
    validate_artifact(dict(artifact))
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
) -> Path:
    artifact = build_artifact(command_runner=command_runner, clock=clock)
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: JsonDict) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"artifact missing required fields: {sorted(missing)}")
    _require(artifact.get("schema") == SCHEMA, "schema mismatch")
    _require(artifact.get("experiment") == EXPERIMENT_ID, "experiment mismatch")
    _require(artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "bad substrate")
    _require(artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "verifier_is_oracle must be false")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    _require(float(artifact.get("duration_s", 0.0)) >= 0.0001, "duration_s below floor")
    _require(artifact.get("workload_name") == WORKLOAD_NAME, "workload_name mismatch")
    _require(artifact.get("n_variables") == N_VARIABLES, "n_variables mismatch")
    _require(artifact.get("local_claim_scope") == LOCAL_CLAIM_SCOPE, "local_claim_scope mismatch")
    _require(_cpu_reference_ok(artifact.get("cpu_reference")), "cpu_reference invalid")
    _require(artifact.get("cpu_reference_ok") is True, "cpu_reference_ok must be true")
    _validate_bare_fields(artifact)
    _validate_no_host_storage(artifact)
    _validate_precondition(artifact)
    _validate_overlay_and_workload(artifact)
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "bad checksum")


def _base_artifact(
    *, cpu_reference: JsonDict, ssh_probe: CommandProbe, duration_s: float
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": "blocked_kv260_ssh_unreachable",
        "kv260_ssh_reachable": ssh_probe.exit_code == 0,
        "overlay_loaded": False,
        "workload_name": WORKLOAD_NAME,
        "n_variables": N_VARIABLES,
        "timing_ratio_packet_built": False,
        "cpu_reference_ok": _cpu_reference_ok(cpu_reference),
        "kv260_result_ok": False,
        "local_claim_scope": LOCAL_CLAIM_SCOPE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": [_precondition_entry(ssh_probe)],
        "overlay_state": _empty_overlay_state(),
        "loaded_overlay": None,
        "uio_devices": [],
        "xmutil_requires_sudo": False,
        "cpu_reference": dict(cpu_reference),
        "kv260_workload": None,
        "timing_ratio_packet": None,
        "gpu_reference": {
            "status": "not_run_no_gpu_claim",
            "reason": "Exp 5052 records CPU and KV260 SSH timing only.",
        },
        "duration_s": _duration_floor(duration_s),
        "command_probes": {
            "kv260_ssh": ssh_probe.as_dict(),
            "kv260_xmutil_listapps": None,
            "kv260_xmutil_listapps_sudo": None,
            "kv260_uio_devices": None,
            "kv260_pbit_workload": None,
        },
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }


def _empty_overlay_state() -> JsonDict:
    return {
        "xmutil_listapps_output": None,
        "xmutil_listapps_exit_code": None,
        "xmutil_requires_sudo": False,
        "loaded_overlay": None,
        "overlay_loaded": False,
        "uio_devices": [],
        "uio_output": None,
        "uio_exit_code": None,
    }


def _overlay_state(
    *,
    listapps_probe: CommandProbe,
    listapps_text: str,
    loaded_overlay: str | None,
    uio_probe: CommandProbe,
    uio_devices: list[str],
    requires_sudo: bool,
) -> JsonDict:
    return {
        "xmutil_listapps_output": listapps_text,
        "xmutil_listapps_exit_code": listapps_probe.exit_code,
        "xmutil_requires_sudo": requires_sudo,
        "loaded_overlay": loaded_overlay,
        "overlay_loaded": loaded_overlay is not None,
        "uio_devices": list(uio_devices),
        "uio_output": uio_probe.combined_output,
        "uio_exit_code": uio_probe.exit_code,
    }


def _precondition_entry(ssh_probe: CommandProbe) -> JsonDict:
    return {
        "resource": "kv260_ssh",
        "available": ssh_probe.exit_code == 0,
        "command": command_to_string(KV260_SSH_COMMAND),
        "exit_code": ssh_probe.exit_code,
        "duration_s": float(ssh_probe.duration_s),
        "observed": _observed_first_line(ssh_probe),
        "discipline": "ssh_only_no_host_sd_card",
    }


def _timing_ratio_packet(
    cpu_reference: JsonDict, kv260_workload: JsonDict, probe: CommandProbe
) -> JsonDict:
    cpu_s = float(cpu_reference["duration_s"])
    command_s = _duration_positive(probe.duration_s)
    board_s = _duration_positive(kv260_workload["duration_s"])
    return {
        "workload_name": WORKLOAD_NAME,
        "n_variables": N_VARIABLES,
        "iterations": int(cpu_reference["iterations"]),
        "flips": int(cpu_reference["flips"]),
        "cpu_wall_clock_s": cpu_s,
        "kv260_command_wall_clock_s": command_s,
        "kv260_board_reported_workload_s": board_s,
        "cpu_to_kv260_command_wall_ratio": round(cpu_s / command_s, 12),
        "cpu_to_kv260_board_workload_ratio": round(cpu_s / board_s, 12),
        "parity_match": True,
        "ratio_claim_scope": LOCAL_CLAIM_SCOPE,
    }


def _cpu_reference_ok(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("workload_name") == WORKLOAD_NAME
        and payload.get("n_variables") == N_VARIABLES
        and isinstance(payload.get("iterations"), int)
        and isinstance(payload.get("flips"), int)
        and isinstance(payload.get("energy"), int)
        and isinstance(payload.get("final_state_checksum"), str)
        and len(str(payload.get("final_state_checksum"))) == 64
        and float(payload.get("duration_s", 0.0)) > 0.0
    )


def _workload_matches(cpu_reference: JsonDict, kv260_workload: JsonDict | None) -> bool:
    if not isinstance(kv260_workload, Mapping):
        return False
    return (
        all(
            kv260_workload.get(field) == cpu_reference.get(field)
            for field in (
                "workload_name",
                "n_variables",
                "iterations",
                "flips",
                "energy",
                "final_state_checksum",
            )
        )
        and float(kv260_workload.get("duration_s", 0.0)) > 0.0
    )


def _validate_bare_fields(artifact: JsonDict) -> None:
    for field in FIELD_PRINCIPLES:
        _require(field in artifact, f"missing required field: {field}")
        _require(
            not (isinstance(artifact[field], Mapping) and "principle" in artifact[field]),
            f"{field} must remain a bare value",
        )


def _validate_no_host_storage(artifact: JsonDict) -> None:
    encoded = json.dumps(artifact, sort_keys=True, default=str).lower()
    _require(
        "mmcblk" not in encoded and "/dev/disk" not in encoded, "forbidden host storage marker"
    )


def _validate_precondition(artifact: JsonDict) -> None:
    preconditions = artifact.get("preconditions_checked")
    _require(
        isinstance(preconditions, list) and len(preconditions) == 1, "bad preconditions_checked"
    )
    entry = preconditions[0]
    _require(isinstance(entry, Mapping), "bad precondition entry")
    _require(entry.get("resource") == "kv260_ssh", "bad KV260 precondition resource")
    _require(entry.get("command") == command_to_string(KV260_SSH_COMMAND), "bad KV260 SSH command")
    _require(entry.get("discipline") == "ssh_only_no_host_sd_card", "bad KV260 discipline")
    _require(entry.get("available") is artifact.get("kv260_ssh_reachable"), "precondition mismatch")


def _validate_overlay_and_workload(artifact: JsonDict) -> None:
    probes = _command_probes(artifact)
    if artifact.get("kv260_ssh_reachable") is False:
        _require(
            artifact.get("honest_verdict") == "blocked_kv260_ssh_unreachable", "bad blocked verdict"
        )
        _require(artifact.get("overlay_loaded") is False, "blocked SSH cannot load overlay")
        _require(artifact.get("kv260_workload") is None, "blocked SSH cannot have KV260 workload")
        _require(probes.get("kv260_xmutil_listapps") is None, "blocked SSH cannot run xmutil")
        _require(probes.get("kv260_pbit_workload") is None, "blocked SSH cannot run workload")
        return

    _require(probes.get("kv260_xmutil_listapps") is not None, "reachable SSH requires xmutil")
    _require(probes.get("kv260_uio_devices") is not None, "reachable SSH requires UIO probe")
    _require(isinstance(artifact.get("overlay_state"), Mapping), "overlay_state must be a dict")
    _require(
        artifact["overlay_state"].get("loaded_overlay") == artifact.get("loaded_overlay"),
        "overlay mismatch",
    )
    _require(
        artifact["overlay_state"].get("uio_devices") == artifact.get("uio_devices"), "UIO mismatch"
    )
    if artifact.get("overlay_loaded") is False:
        _require(
            artifact.get("timing_ratio_packet_built") is False,
            "no overlay cannot build timing ratio",
        )
        _require(artifact.get("kv260_result_ok") is False, "no overlay cannot have KV260 result")
        _require(artifact.get("kv260_workload") is None, "no overlay cannot run workload")
        return

    _require(artifact.get("loaded_overlay") in VALID_OVERLAYS, "invalid overlay")
    _require(
        probes.get("kv260_pbit_workload") is not None, "loaded overlay requires workload probe"
    )
    _require(
        artifact.get("kv260_result_ok")
        is _workload_matches(artifact["cpu_reference"], artifact.get("kv260_workload")),
        "parity mismatch",
    )
    if artifact.get("timing_ratio_packet_built"):
        packet = artifact.get("timing_ratio_packet")
        _require(artifact.get("kv260_result_ok") is True, "timing packet requires KV260 result")
        _require(
            isinstance(packet, Mapping) and packet.get("parity_match") is True, "bad timing packet"
        )
    else:
        _require(
            artifact.get("timing_ratio_packet") is None, "failed parity cannot keep timing packet"
        )


def _command_probes(artifact: JsonDict) -> JsonDict:
    probes = artifact.get("command_probes")
    _require(isinstance(probes, Mapping), "command_probes must be a dict")
    return probes


def _xmutil_requires_root(probe: CommandProbe) -> bool:
    lowered = probe.combined_output.lower()
    return "root privileges" in lowered or "using 'sudo'" in lowered


def _observed_first_line(probe: CommandProbe) -> str:
    observed = probe.combined_output.strip() or f"returncode={probe.exit_code}"
    return observed.splitlines()[0][:300]


def _duration_floor(duration_s: float) -> float:
    return round(max(float(duration_s), 0.0001), 4)


def _duration_positive(duration_s: float) -> float:
    return round(max(float(duration_s), 0.0001), 12)


def _finalize(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:  # pragma: no cover - live hardware entrypoint
    out_path = run_experiment(repo_root=REPO_ROOT)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {"honest_verdict": artifact["honest_verdict"], "result": str(out_path)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint
    raise SystemExit(main())
