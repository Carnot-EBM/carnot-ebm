"""PolarFire SoC Ising dispatch smoke v4 for Exp 3867.

This experiment proves a narrow hardware boundary: a deterministic Carnot-style
Ising energy workload is copied to the PolarFire SoC Discovery Kit, executed by
the board's Linux CPU through SSH, and accepted only when the board result hash
matches the local CPU reference hash. It does not claim FPGA fabric execution.

Spec refs: REQ-HW-3867, SCENARIO-HW-3867.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any


EXPERIMENT_ID = 3867
SCHEMA = "carnot.polarfire_soc_ising_dispatch.v4"
SPEC_REFS = ["REQ-HW-3867", "SCENARIO-HW-3867"]
OUTPUT_REL_PATH = Path("results") / "experiment_3867_polarfire_soc_smoke_v4.json"
DEFAULT_HOST = "polarfire"
RANDOM_SEED = 3867
MIN_TERMINAL_DURATION_S = 5.0
DEFAULT_REMOTE_RUNTIME_S = 5.05
INFERENCE_SUBSTRATE = "hardware_smoke"
THERMAL_NOTE = (
    "passively cooled; no active fan; sustained-load results may differ from "
    "production with active cooling"
)

POLARFIRE_SSH_PRECONDITION = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    DEFAULT_HOST,
    "true",
)
POLARFIRE_PYTHON_PRECONDITION = ("ssh", DEFAULT_HOST, "python3 --version")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "polarfire_workload_validated",
    "result_hash_match",
    "board_result_sha256",
    "cpu_reference_sha256",
    "run_duration_s",
    "soc_temp_max_c",
    "thermal_note",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix records whether the live board run succeeded, completed "
        "with non-terminal evidence, or stopped at a precondition."
    ),
    "polarfire_workload_validated": (
        "Bare bool -- terminal-state gate; true only if the board's result hash "
        "matches the CPU reference and the real dispatch duration reaches the floor."
    ),
    "result_hash_match": (
        "Bare bool -- the structural anti-fabrication check; the board must "
        "reproduce the exact reference result."
    ),
    "board_result_sha256": (
        "The board-side result hash compared against the CPU reference -- auditable "
        "provenance that the workload ran on the board."
    ),
    "cpu_reference_sha256": (
        "The local reference result hash compared against the board result -- "
        "auditable provenance for the deterministic CPU baseline."
    ),
    "run_duration_s": (
        "Real SSH round-trip wall-clock; >=5s floor structurally prevents the "
        "exp1680 run_duration_s=0 fabrication."
    ),
    "soc_temp_max_c": (
        "Per-board hardware-artifact thermal field; null only when the board does "
        "not expose readable thermal sysfs values."
    ),
    "thermal_note": (
        "Per-board hardware-artifact field describing the passive-cooling envelope "
        "of the dispatch."
    ),
    "preconditions_checked": (
        "Pre-Launch + Inference-Substrate hardware_smoke gate; SSH reachability "
        "and board Python checks must pass before dispatch."
    ),
    "random_seed": (
        "Pre-Launch reproducibility field controlling the fixed Ising workload."
    ),
    "reproducibility_checksum": (
        "Content-addressed checksum over the artifact, excluding this checksum, "
        "so later drift is detectable."
    ),
    "inference_substrate": (
        "Inference-Substrate declaration for a hardware_smoke SSH-attached board run."
    ),
}


@dataclass(frozen=True)
class CommandResult:
    """Captured command transcript used by tests and by the live SSH run."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str], float | None], CommandResult]
Clock = Callable[[], float]


class DispatchError(RuntimeError):
    """Raised when a post-precondition transfer or remote command fails."""

    def __init__(
        self,
        verdict: str,
        detail: str,
        transcript: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail)
        self.verdict = verdict
        self.detail = detail
        self.transcript = transcript if isinstance(transcript, list) else list(transcript or [])


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def command_to_string(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


def _observed(result: CommandResult) -> str:
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        return stdout
    if stderr:
        return stderr
    return f"returncode={result.returncode}"


def _check_row(resource: str, command: Sequence[str], result: CommandResult) -> dict[str, Any]:
    return {
        "resource": resource,
        "command": command_to_string(command),
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "observed": _observed(result),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _ssh_precondition(host: str) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        host,
        "true",
    )


def _python_precondition(host: str) -> tuple[str, ...]:
    return ("ssh", host, "python3 --version")


def check_preconditions(
    runner: Runner,
    host: str = DEFAULT_HOST,
) -> tuple[list[dict[str, Any]], dict[str, str], str | None]:
    """Check SSH and board Python before any SCP or workload execution."""
    metadata: dict[str, str] = {}
    checks: list[dict[str, Any]] = []

    ssh_cmd = _ssh_precondition(host)
    ssh_result = runner(ssh_cmd, 10.0)
    checks.append(_check_row("polarfire_ssh", ssh_cmd, ssh_result))
    if ssh_result.returncode != 0:
        return checks, metadata, "blocked_polarfire_ssh_timeout"

    python_cmd = _python_precondition(host)
    python_result = runner(python_cmd, 10.0)
    metadata["polarfire_python"] = (python_result.stdout or python_result.stderr).strip()
    checks.append(_check_row("polarfire_python", python_cmd, python_result))
    if python_result.returncode != 0:
        return checks, metadata, "blocked_polarfire_no_python"

    return checks, metadata, None


def build_ising_workload(random_seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Build a compact deterministic Ising energy workload for host and board.

    The values are generated with simple integer arithmetic instead of runtime
    randomness. The seed still controls every coupling, bias, and spin, but the
    resulting JSON is stable across Python versions and architectures.
    """
    n_spins = 8
    coupling = [[0 for _ in range(n_spins)] for _ in range(n_spins)]
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            value = ((random_seed + i * 17 - j * 11 + i * j * 7) % 7) - 3
            if (i + j + random_seed) % 5 == 0:
                value = 0
            coupling[i][j] = value
            coupling[j][i] = value

    bias = [((random_seed // (idx + 1)) + idx * 3) % 5 - 2 for idx in range(n_spins)]
    spin_configs: list[list[int]] = []
    for row in range(6):
        spin_configs.append(
            [
                1
                if ((random_seed >> (idx % 12)) + row * 3 + idx * 5) % 2 == 0
                else -1
                for idx in range(n_spins)
            ]
        )

    return {
        "schema": "carnot.ising_energy_workload.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": random_seed,
        "n_spins": n_spins,
        "coupling": coupling,
        "bias": bias,
        "spin_configs": spin_configs,
    }


def evaluate_ising_workload(workload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the standard Ising Hamiltonian over every fixed spin config."""
    coupling = workload["coupling"]
    bias = workload["bias"]
    spin_configs = workload["spin_configs"]
    n_spins = int(workload["n_spins"])

    energies: list[int] = []
    for spins in spin_configs:
        energy = 0
        for i in range(n_spins):
            energy -= int(bias[i]) * int(spins[i])
            for j in range(i + 1, n_spins):
                energy -= int(coupling[i][j]) * int(spins[i]) * int(spins[j])
        energies.append(energy)

    return {
        "schema": "carnot.ising_energy_result.v1",
        "spec_refs": SPEC_REFS,
        "workload_sha256": sha256_json(workload),
        "random_seed": int(workload["random_seed"]),
        "n_spins": n_spins,
        "n_configs": len(spin_configs),
        "energies": energies,
        "energy_sum": sum(energies),
        "energy_min": min(energies),
        "energy_max": max(energies),
    }


def build_remote_runner_source() -> str:
    """Return the self-contained Python source copied to the PolarFire board."""
    return r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time

SPEC_REFS = ["REQ-HW-3867", "SCENARIO-HW-3867"]


def canonical_json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload):
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def evaluate_ising_workload(workload):
    coupling = workload["coupling"]
    bias = workload["bias"]
    spin_configs = workload["spin_configs"]
    n_spins = int(workload["n_spins"])
    energies = []
    for spins in spin_configs:
        energy = 0
        for i in range(n_spins):
            energy -= int(bias[i]) * int(spins[i])
            for j in range(i + 1, n_spins):
                energy -= int(coupling[i][j]) * int(spins[i]) * int(spins[j])
        energies.append(energy)
    return {
        "schema": "carnot.ising_energy_result.v1",
        "spec_refs": SPEC_REFS,
        "workload_sha256": sha256_json(workload),
        "random_seed": int(workload["random_seed"]),
        "n_spins": n_spins,
        "n_configs": len(spin_configs),
        "energies": energies,
        "energy_sum": sum(energies),
        "energy_min": min(energies),
        "energy_max": max(energies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload_path")
    parser.add_argument("--min-runtime-s", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.workload_path, "r", encoding="utf-8") as handle:
        workload = json.load(handle)

    started = time.perf_counter()
    result = evaluate_ising_workload(workload)
    cycles = 1
    while time.perf_counter() - started < args.min_runtime_s:
        result = evaluate_ising_workload(workload)
        cycles += 1

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    print(f"cycles={cycles}", file=sys.stderr)


if __name__ == "__main__":
    main()
'''


def extract_local_workload_path(remote_command: str) -> str:
    """Extract the test-only local workload marker embedded as a shell comment."""
    match = re.search(r"local_workload_path=([^\s]+)", remote_command)
    if not match:
        raise ValueError("local_workload_path marker missing")
    return match.group(1)


def parse_soc_temp_max_c(text: str) -> float | None:
    """Parse sysfs thermal values, accepting millidegrees C or degrees C."""
    values: list[float] = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", text):
        value = float(token)
        if abs(value) > 1000:
            value /= 1000.0
        if -50.0 <= value <= 150.0:
            values.append(value)
    if not values:
        return None
    return round(max(values), 1)


def _write_local_payloads(local_dir: Path, workload: dict[str, Any]) -> tuple[Path, Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    workload_path = local_dir / "workload.json"
    runner_path = local_dir / "runner.py"
    workload_path.write_text(json.dumps(workload, sort_keys=True, indent=2), encoding="utf-8")
    runner_path.write_text(build_remote_runner_source(), encoding="utf-8")
    return workload_path, runner_path


def _require_success(
    result: CommandResult,
    stage: str,
    transcript: Sequence[dict[str, Any]] | None = None,
) -> None:
    if result.returncode != 0:
        detail = f"{stage}: rc={result.returncode}; stderr={result.stderr[:300]}"
        raise DispatchError("blocked_polarfire_dispatch_failed", detail, transcript)


def dispatch_to_board(
    *,
    runner: Runner,
    host: str,
    workload_path: Path,
    runner_path: Path,
    remote_dir: str,
    min_remote_runtime_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Copy the workload to the board, run it, and return the parsed stdout JSON."""
    transcript: list[dict[str, Any]] = []
    mkdir_cmd = ("ssh", host, f"mkdir -p {shlex.quote(remote_dir)}")
    cleanup_cmd = ("ssh", host, f"rm -rf {shlex.quote(remote_dir)}")
    try:
        mkdir_result = runner(mkdir_cmd, 20.0)
        transcript.append(_command_transcript("remote_mkdir", mkdir_result))
        _require_success(mkdir_result, "remote_mkdir", transcript)

        scp_cmd = (
            "scp",
            "-q",
            str(runner_path),
            str(workload_path),
            f"{host}:{remote_dir}/",
        )
        scp_result = runner(scp_cmd, 60.0)
        transcript.append(_command_transcript("scp_push", scp_result))
        _require_success(scp_result, "scp_push", transcript)

        remote_shell = (
            f"cd {shlex.quote(remote_dir)} && "
            "python3 runner.py workload.json "
            f"--min-runtime-s {min_remote_runtime_s:.3f} "
            f"# local_workload_path={workload_path}"
        )
        run_cmd = ("ssh", host, remote_shell)
        run_result = runner(run_cmd, max(90.0, min_remote_runtime_s + 30.0))
        transcript.append(_command_transcript("remote_ising_eval", run_result))
        _require_success(run_result, "remote_ising_eval", transcript)
        try:
            board_result = json.loads(run_result.stdout)
        except json.JSONDecodeError as exc:
            raise DispatchError(
                "blocked_polarfire_dispatch_failed",
                f"remote_ising_eval_json: {exc}",
                transcript,
            ) from exc
        return board_result, transcript
    finally:
        cleanup_result = runner(cleanup_cmd, 20.0)
        transcript.append(_command_transcript("remote_cleanup", cleanup_result))


def _command_transcript(stage: str, result: CommandResult) -> dict[str, Any]:
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return {
        "stage": stage,
        "command": command_to_string(result.args),
        "returncode": result.returncode,
        "output_excerpt": output[-1200:],
    }


def probe_soc_temp_max_c(runner: Runner, host: str) -> tuple[float | None, dict[str, Any]]:
    """Read PolarFire thermal sysfs values when Linux exposes them."""
    shell = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "for path in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp')):\n"
        "    try:\n"
        "        print(path.read_text().strip())\n"
        "    except OSError:\n"
        "        pass\n"
        "PY"
    )
    cmd = ("ssh", host, shell)
    result = runner(cmd, 20.0)
    transcript = _command_transcript("thermal_probe", result)
    if result.returncode != 0:
        return None, transcript
    return parse_soc_temp_max_c(result.stdout), transcript


def _field_provenance() -> dict[str, dict[str, str]]:
    return {field: {"principle": FIELD_PRINCIPLES[field]} for field in REQUIRED_ARTIFACT_FIELDS}


def _empty_artifact(
    *,
    verdict: str,
    checks: Sequence[dict[str, Any]],
    workload: dict[str, Any],
    cpu_reference_sha256: str,
    run_duration_s: float,
    soc_temp_max_c: float | None = None,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": SPEC_REFS,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "polarfire_workload_validated": False,
        "result_hash_match": False,
        "board_result_sha256": "",
        "cpu_reference_sha256": cpu_reference_sha256,
        "run_duration_s": round(float(run_duration_s), 6),
        "soc_temp_max_c": soc_temp_max_c,
        "thermal_note": THERMAL_NOTE,
        "preconditions_checked": list(checks),
        "random_seed": int(workload["random_seed"]),
        "reproducibility_checksum": "",
        "field_provenance": _field_provenance(),
        "workload_sha256": sha256_json(workload),
        "no_fpga_fabric_claim": True,
    }


def _terminal_verdict(
    *,
    result_hash_match: bool,
    run_duration_s: float,
    board_result_sha256: str,
    cpu_reference_sha256: str,
    soc_temp_max_c: float | None,
) -> str:
    if not result_hash_match:
        return (
            "complete: polarfire_dispatch_ran_hash_MISMATCH_"
            f"cpu{cpu_reference_sha256[:8]}_board{board_result_sha256[:8]}_"
            "workload_not_validated"
        )
    if run_duration_s < MIN_TERMINAL_DURATION_S:
        return "complete: polarfire_dispatch_ran_duration_lt5_workload_not_validated"
    temp_label = "unknown" if soc_temp_max_c is None else f"{soc_temp_max_c:.1f}"
    return (
        "success: polarfire_carnot_dispatch_hash_verified_terminal_"
        f"duration{run_duration_s:.2f}s_temp{temp_label}"
    )


def _finalize_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    artifact["reproducibility_checksum"] = sha256_json(payload)
    return artifact


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_experiment(
    *,
    repo_root: str | Path = ".",
    runner: Runner | None = None,
    clock: Clock = time.monotonic,
    host: str = DEFAULT_HOST,
    remote_dir: str | None = None,
    min_remote_runtime_s: float = DEFAULT_REMOTE_RUNTIME_S,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the gated PolarFire Ising dispatch and write the JSON artifact."""
    active_runner = runner or _run_local_command
    started = clock()
    workload = build_ising_workload(RANDOM_SEED)
    cpu_reference = evaluate_ising_workload(workload)
    cpu_reference_sha256 = sha256_json(cpu_reference)

    checks, metadata, blocker = check_preconditions(active_runner, host)
    del metadata
    if blocker is not None:
        artifact = _empty_artifact(
            verdict=blocker,
            checks=checks,
            workload=workload,
            cpu_reference_sha256=cpu_reference_sha256,
            run_duration_s=clock() - started,
        )
        artifact = _finalize_checksum(artifact)
        destination = output_path or Path(repo_root) / OUTPUT_REL_PATH
        _write_json(destination, artifact)
        return artifact

    active_remote_dir = remote_dir or f"/tmp/carnot_exp3867_{os.getpid()}_{int(time.time())}"
    command_transcript: list[dict[str, Any]] = []
    board_result: dict[str, Any] | None = None
    failure_detail = ""
    dispatch_blocker = ""
    soc_temp_max_c: float | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="carnot_exp3867_") as tmpdir:
            workload_path, runner_path = _write_local_payloads(Path(tmpdir), workload)
            board_result, command_transcript = dispatch_to_board(
                runner=active_runner,
                host=host,
                workload_path=workload_path,
                runner_path=runner_path,
                remote_dir=active_remote_dir,
                min_remote_runtime_s=min_remote_runtime_s,
            )
        soc_temp_max_c, thermal_transcript = probe_soc_temp_max_c(active_runner, host)
        command_transcript.append(thermal_transcript)
    except DispatchError as exc:
        dispatch_blocker = exc.verdict
        failure_detail = exc.detail
        command_transcript = exc.transcript

    run_duration_s = round(float(clock() - started), 6)
    if dispatch_blocker:
        artifact = _empty_artifact(
            verdict=dispatch_blocker,
            checks=checks,
            workload=workload,
            cpu_reference_sha256=cpu_reference_sha256,
            run_duration_s=run_duration_s,
            soc_temp_max_c=soc_temp_max_c,
        )
        artifact["failure_detail"] = failure_detail
        artifact["command_transcript"] = command_transcript
        artifact = _finalize_checksum(artifact)
        destination = output_path or Path(repo_root) / OUTPUT_REL_PATH
        _write_json(destination, artifact)
        return artifact

    assert board_result is not None
    board_result_sha256 = sha256_json(board_result)
    result_hash_match = board_result_sha256 == cpu_reference_sha256
    workload_validated = result_hash_match and run_duration_s >= MIN_TERMINAL_DURATION_S
    verdict = _terminal_verdict(
        result_hash_match=result_hash_match,
        run_duration_s=run_duration_s,
        board_result_sha256=board_result_sha256,
        cpu_reference_sha256=cpu_reference_sha256,
        soc_temp_max_c=soc_temp_max_c,
    )

    artifact = _empty_artifact(
        verdict=verdict,
        checks=checks,
        workload=workload,
        cpu_reference_sha256=cpu_reference_sha256,
        run_duration_s=run_duration_s,
        soc_temp_max_c=soc_temp_max_c,
    )
    artifact.update(
        {
            "polarfire_workload_validated": workload_validated,
            "result_hash_match": result_hash_match,
            "board_result_sha256": board_result_sha256,
            "cpu_reference_result": cpu_reference,
            "board_result": board_result,
            "command_transcript": command_transcript,
        }
    )
    artifact = _finalize_checksum(artifact)
    destination = output_path or Path(repo_root) / OUTPUT_REL_PATH
    _write_json(destination, artifact)
    return artifact


def _run_local_command(args: Sequence[str], timeout: float | None = None) -> CommandResult:
    try:  # pragma: no cover - exercised by the live hardware run.
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return CommandResult(tuple(args), 124, stdout, stderr)
    except OSError as exc:  # pragma: no cover
        return CommandResult(tuple(args), 127, "", str(exc))


def main() -> None:  # pragma: no cover
    artifact = run_experiment(repo_root=Path(__file__).resolve().parents[2])
    print(json.dumps(artifact, sort_keys=True, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
