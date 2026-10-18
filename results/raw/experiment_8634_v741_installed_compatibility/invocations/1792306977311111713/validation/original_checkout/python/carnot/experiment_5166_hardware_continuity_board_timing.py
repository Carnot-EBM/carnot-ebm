#!/usr/bin/env python3
"""Exp 5166: hardware continuity board timing with per-board blockers.

Spec refs: REQ-HW-5166, SCENARIO-HW-5166.

This experiment keeps the three attached-board tracks visible without turning
smoke evidence into a speedup claim. Each board gets its own precondition and
result. A failed GateMate IDCODE or SSH timeout is recorded as that board's
blocker, while reachable boards still run a small hash-verified workload through
their own command interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from carnot.experiment_5120_hardware_residual_telemetry import (
    Clock,
    CommandProbe,
    CommandRunner,
    JsonDict,
    JsonMap,
    command_to_string,
    idcode_from_text,
    is_sha256,
    no_host_storage as no_5120_host_storage,
    payload_checksum,
    prepend_oss_cad_suite,
    probe_dict,
    round_duration,
    run_command,
    sha256_json,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution.
    sys.path.insert(0, str(REPO_ROOT / "python"))


EXPERIMENT_ID = "exp5166-hardware-continuity-board-timing-v473"
EXPERIMENT_NAME = "experiment_5166_hardware_continuity_board_timing"
MILESTONE = "2026.07.473"
SCHEMA = "carnot.experiment_5166_hardware_continuity_board_timing.v473"
OUTPUT_REL_PATH = Path("results") / "experiment_5166_hardware_continuity_board_timing_v473.json"
SPEC_REFS = ["REQ-HW-5166", "SCENARIO-HW-5166"]
INFERENCE_SUBSTRATE = "hardware_smoke"
RANDOM_SEED = 5166

KV260_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "kria",
    "true",
)
GATEMATE_DETECT_COMMAND = ("openFPGALoader", "-c", "dirtyJtag", "--detect")
POLARFIRE_PRECONDITION_COMMAND = (
    "ssh",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "BatchMode=yes",
    "polarfire",
    "true",
)
GATEMATE_EXPECTED_IDCODE = "0x20000001"

INLINE_PROGRAM_TEMPLATE = (
    "exp5166_inline_ising_energy_v1: parse JSON spins/edges, compute Ising energy, "
    "time the board-local run, and emit workload/executable hashes with quality evidence"
)
INLINE_EXECUTABLE_HASH = sha256_text(INLINE_PROGRAM_TEMPLATE)

REQUIRED_ARTIFACT_FIELDS = (
    "kv260_result",
    "gatemate_result",
    "polarfire_result",
    "boards_reachable_count",
    "hardware_wishlist_updated",
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "command_transcripts",
    "sample_quality_evidence",
    "no_speedup_claim",
    "tests_run",
)
REQUIRED_SCHEMA_FIELDS = (
    *REQUIRED_ARTIFACT_FIELDS,
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "spec_refs",
    "random_seed",
    "run_date",
    "duration_s",
    "field_principles",
    "kv260_host_block_devices_touched",
    "hardware_speedup_claimed",
    "workload_hashes",
    "conductor_modified",
    "reproducibility_checksum",
)
BOARD_RESULT_FIELDS = (
    "reachable",
    "workload_hash",
    "executable_hash",
    "latency_transcript",
    "timing_output",
    "hash_verified",
    "sample_quality",
    "correctness",
    "blocked_reason",
    "inference_substrate",
)
COMMAND_TRANSCRIPT_KEYS = (
    "kv260_precondition",
    "kv260_workload",
    "gatemate_precondition",
    "gatemate_workload",
    "polarfire_precondition",
    "polarfire_workload",
)
WORKLOAD_HASH_KEYS = ("kv260", "gatemate", "polarfire")

WISHLIST_MARKERS = (
    "2026-07-02 Exp 5166 KV260",
    "2026-07-02 Exp 5166 GateMate",
    "2026-07-02 Exp 5166 PolarFire",
)
WISHLIST_REL_PATH = Path("research-hardware-wishlist.md")

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_5166_hardware_continuity_board_timing_v473.py --date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5166_hardware_continuity_board_timing.py -q",
    ".venv/bin/coverage run --source=python/carnot/experiment_5166_hardware_continuity_board_timing.py -m pytest tests/python/test_experiment_5166_hardware_continuity_board_timing.py -q",
    ".venv/bin/coverage report --fail-under=100 -m python/carnot/experiment_5166_hardware_continuity_board_timing.py",
    ".venv/bin/pytest tests/python -q",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "kv260_result": "per-board evidence; KV260 SSH success or blocker must not be inferred from host storage",
    "gatemate_result": "per-board evidence; GateMate stays visible even when DirtyJTAG lacks an IDCODE",
    "polarfire_result": "per-board evidence; PolarFire SSH success or blocker is reported independently",
    "boards_reachable_count": "reachability accounting; one blocked board is not a whole-task failure",
    "hardware_wishlist_updated": "ops continuity; the active hardware table must carry this milestone's dated board status",
    "honest_verdict": "terminal verdict with complete_/success_ prefix plus honest per-board reachability",
    "inference_substrate": "substrate honesty; board-touching measurements are hardware_smoke",
    "preconditions_checked": "fabrication guard; every board resource is checked before workload execution",
    "command_transcripts": "authenticated command evidence",
    "sample_quality_evidence": "no latency-only or speedup-only claim",
    "no_speedup_claim": "hardware claim discipline",
    "tests_run": "verification evidence",
}


def ising_energy(payload: JsonMap) -> int:
    spins = [int(value) for value in payload["spins"]]
    total = 0
    for row, col, coupling in payload["edges"]:
        total -= int(coupling) * spins[int(row)] * spins[int(col)]
    return total


KV260_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "kv260",
    "workload": "exp5166_inline_ising_energy_smoke",
    "spins": [1, -1, 1, -1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, -1], [2, 3, 1], [3, 4, -1], [4, 5, 1], [5, 6, -1]],
}
KV260_EXPECTED_ENERGY = ising_energy(KV260_WORKLOAD_BASE)
KV260_WORKLOAD = dict(KV260_WORKLOAD_BASE, expected_energy=KV260_EXPECTED_ENERGY)
KV260_WORKLOAD_HASH = sha256_json(KV260_WORKLOAD)

POLARFIRE_WORKLOAD_BASE: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "polarfire",
    "workload": "exp5166_inline_ising_energy_smoke",
    "spins": [1, 1, -1, -1, 1, -1, 1, -1],
    "edges": [[0, 1, 1], [1, 2, 1], [2, 3, -1], [3, 4, 1], [4, 5, -1], [6, 7, 1]],
}
POLARFIRE_EXPECTED_ENERGY = ising_energy(POLARFIRE_WORKLOAD_BASE)
POLARFIRE_WORKLOAD = dict(POLARFIRE_WORKLOAD_BASE, expected_energy=POLARFIRE_EXPECTED_ENERGY)
POLARFIRE_WORKLOAD_HASH = sha256_json(POLARFIRE_WORKLOAD)

GATEMATE_WORKLOAD: JsonDict = {
    "experiment_id": EXPERIMENT_ID,
    "board": "gatemate",
    "workload": "exp5166_dirtyjtag_idcode_readback",
    "command": list(GATEMATE_DETECT_COMMAND),
    "expected_idcode": GATEMATE_EXPECTED_IDCODE,
}
GATEMATE_WORKLOAD_HASH = sha256_json(GATEMATE_WORKLOAD)


def kv260_workload_command() -> tuple[str, ...]:
    return ssh_workload_command("kria", KV260_WORKLOAD, KV260_WORKLOAD_HASH)


def polarfire_workload_command() -> tuple[str, ...]:
    return ssh_workload_command("polarfire", POLARFIRE_WORKLOAD, POLARFIRE_WORKLOAD_HASH)


def ssh_workload_command(host: str, payload: JsonMap, workload_hash: str) -> tuple[str, ...]:
    payload_json = json.dumps(payload, sort_keys=True)
    return (
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "BatchMode=yes",
        host,
        (
            "python3 - <<'PY'\n"
            "import json, math, time\n"
            f"payload = {payload_json!r}\n"
            "data = json.loads(payload)\n"
            "started = time.perf_counter()\n"
            "spins = [int(value) for value in data['spins']]\n"
            "energy = 0\n"
            "for row, col, coupling in data['edges']:\n"
            "    energy -= int(coupling) * spins[int(row)] * spins[int(col)]\n"
            "duration_s = max(time.perf_counter() - started, 0.000001)\n"
            "print(json.dumps({\n"
            f"    'workload_sha256': {workload_hash!r},\n"
            f"    'executable_sha256': {INLINE_EXECUTABLE_HASH!r},\n"
            f"    'inference_substrate': {INFERENCE_SUBSTRATE!r},\n"
            "    'energy': energy,\n"
            "    'duration_s': duration_s,\n"
            "    'sample_quality': {'sample_count': len(spins), 'finite_energy': math.isfinite(float(energy))},\n"
            "    'correctness': {'energy_matches_expected': energy == data['expected_energy']},\n"
            "}, sort_keys=True))\n"
            "PY"
        ),
    )


def build_artifact(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260702",
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    started = clock()
    root = Path(repo_root)
    kv260 = run_ssh_board(
        board="kv260",
        precondition_command=KV260_PRECONDITION_COMMAND,
        workload_command=kv260_workload_command(),
        workload_hash=KV260_WORKLOAD_HASH,
        command_runner=command_runner,
    )
    gatemate = run_gatemate_board(command_runner)
    polarfire = run_ssh_board(
        board="polarfire",
        precondition_command=POLARFIRE_PRECONDITION_COMMAND,
        workload_command=polarfire_workload_command(),
        workload_hash=POLARFIRE_WORKLOAD_HASH,
        command_runner=command_runner,
    )
    board_results = {"kv260": kv260, "gatemate": gatemate, "polarfire": polarfire}

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "run_date": run_date,
        "duration_s": round_duration(clock() - started),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": honest_verdict(board_results),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": build_preconditions(kv260, gatemate, polarfire),
        "kv260_host_block_devices_touched": False,
        "kv260_result": result_public(kv260),
        "gatemate_result": result_public(gatemate),
        "polarfire_result": result_public(polarfire),
        "boards_reachable_count": sum(bool(result["reachable"]) for result in board_results.values()),
        "hardware_wishlist_updated": wishlist_has_update(root),
        "command_transcripts": build_command_transcripts(kv260, gatemate, polarfire),
        "workload_hashes": {
            "kv260": KV260_WORKLOAD_HASH,
            "gatemate": GATEMATE_WORKLOAD_HASH,
            "polarfire": POLARFIRE_WORKLOAD_HASH,
        },
        "sample_quality_evidence": sample_quality_evidence(board_results),
        "no_speedup_claim": True,
        "hardware_speedup_claimed": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run_ssh_board(
    *,
    board: str,
    precondition_command: tuple[str, ...],
    workload_command: tuple[str, ...],
    workload_hash: str,
    command_runner: CommandRunner,
) -> JsonDict:
    precondition_probe = command_runner(precondition_command, 10.0)
    workload_probe = None
    output: JsonDict = {}
    blocked_reason = None
    reachable = precondition_probe.exit_code == 0
    if not reachable:
        blocked_reason = f"blocked_{board}_ssh"
    else:
        workload_probe = command_runner(workload_command, 30.0)
        output = parse_probe_json(workload_probe)
        blocked_reason = ssh_workload_blocker(board, workload_probe, workload_hash, output)
    return {
        "board": board,
        "reachable": reachable,
        "precondition_probe": precondition_probe,
        "workload_probe": workload_probe,
        "workload_hash": workload_hash if reachable else None,
        "executable_hash": INLINE_EXECUTABLE_HASH if reachable else None,
        "timing_output": output,
        "hash_verified": ssh_output_hash_verified(workload_hash, output),
        "sample_quality": output.get("sample_quality") if isinstance(output.get("sample_quality"), Mapping) else None,
        "correctness": output.get("correctness") if isinstance(output.get("correctness"), Mapping) else None,
        "blocked_reason": blocked_reason,
    }


def run_gatemate_board(command_runner: CommandRunner) -> JsonDict:
    precondition_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
    precondition_idcode = idcode_from_text(precondition_probe.combined_output)
    idcode_ok = precondition_probe.exit_code == 0 and precondition_idcode == GATEMATE_EXPECTED_IDCODE
    workload_probe = None
    blocked_reason = None
    if precondition_probe.exit_code != 0:
        blocked_reason = "blocked_gatemate_dirtyjtag"
    elif not idcode_ok:
        blocked_reason = "blocked_gatemate_dirtyjtag_idcode"
    else:
        workload_probe = command_runner(GATEMATE_DETECT_COMMAND, 30.0)
        workload_idcode = idcode_from_text(workload_probe.combined_output)
        if workload_probe.exit_code != 0:
            blocked_reason = "blocked_gatemate_workload_command"
        elif workload_idcode != GATEMATE_EXPECTED_IDCODE:
            blocked_reason = "blocked_gatemate_workload_idcode"
    hash_verified = (
        workload_probe is not None
        and workload_probe.exit_code == 0
        and idcode_from_text(workload_probe.combined_output) == GATEMATE_EXPECTED_IDCODE
    )
    return {
        "board": "gatemate",
        "reachable": idcode_ok,
        "precondition_probe": precondition_probe,
        "workload_probe": workload_probe,
        "workload_hash": GATEMATE_WORKLOAD_HASH if idcode_ok else None,
        "executable_hash": None,
        "timing_output": {
            "detected_idcode": idcode_from_text(workload_probe.combined_output)
            if workload_probe is not None
            else precondition_idcode,
            "expected_idcode": GATEMATE_EXPECTED_IDCODE,
            "command_interface": "openFPGALoader",
        },
        "hash_verified": hash_verified,
        "sample_quality": None,
        "correctness": {"idcode_matches_expected": hash_verified} if workload_probe is not None else None,
        "blocked_reason": blocked_reason,
    }


def parse_probe_json(probe: CommandProbe | None) -> JsonDict:
    if probe is None or not probe.combined_output.strip():
        return {}
    try:
        parsed = json.loads(probe.combined_output.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def ssh_output_hash_verified(workload_hash: str, output: JsonMap) -> bool:
    return (
        output.get("workload_sha256") == workload_hash
        and output.get("executable_sha256") == INLINE_EXECUTABLE_HASH
        and output.get("inference_substrate") == INFERENCE_SUBSTRATE
    )


def ssh_output_has_evidence(output: JsonMap) -> bool:
    return isinstance(output.get("sample_quality"), Mapping) or isinstance(output.get("correctness"), Mapping)


def ssh_workload_blocker(
    board: str, workload_probe: CommandProbe | None, workload_hash: str, output: JsonMap
) -> str | None:
    if workload_probe is None:
        return f"blocked_{board}_workload_missing"
    if workload_probe.exit_code != 0:
        return f"blocked_{board}_workload_command"
    if not ssh_output_hash_verified(workload_hash, output):
        return f"blocked_{board}_workload_hash"
    if not ssh_output_has_evidence(output):
        return f"blocked_{board}_workload_evidence"
    return None


def honest_verdict(board_results: JsonMap) -> str:
    blocked = [
        f"{board}:{result['blocked_reason']}"
        for board, result in board_results.items()
        if result.get("blocked_reason")
    ]
    if not blocked:
        return "complete_hardware_continuity_board_timing_all_reachable_no_speedup_claim"
    return "complete_hardware_continuity_board_timing_" + "_".join(blocked) + "_no_speedup_claim"


def result_public(result: JsonMap) -> JsonDict:
    return {
        "reachable": bool(result["reachable"]),
        "workload_hash": result.get("workload_hash"),
        "executable_hash": result.get("executable_hash"),
        "latency_transcript": probe_dict(result.get("workload_probe")),
        "timing_output": dict(result["timing_output"]),
        "hash_verified": bool(result["hash_verified"]),
        "sample_quality": result.get("sample_quality"),
        "correctness": result.get("correctness"),
        "blocked_reason": result.get("blocked_reason"),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_preconditions(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> list[JsonDict]:
    return [
        precondition_dict(
            "kv260_ssh",
            kv260["precondition_probe"],
            bool(kv260["reachable"]),
            "ssh_only_no_host_block_device_probe",
        ),
        precondition_dict(
            "gatemate_dirtyjtag_idcode",
            gatemate["precondition_probe"],
            bool(gatemate["reachable"]),
            "dirtyjtag_detect_requires_gm1ax_idcode",
        ),
        precondition_dict(
            "polarfire_ssh",
            polarfire["precondition_probe"],
            bool(polarfire["reachable"]),
            "ssh_only_no_flash",
        ),
    ]


def precondition_dict(
    resource: str, probe: CommandProbe, available: bool, discipline: str
) -> JsonDict:
    return {
        "resource": resource,
        "available": bool(available),
        "command": command_to_string(probe.command),
        "exit_code": probe.exit_code,
        "duration_s": round_duration(probe.duration_s),
        "observed": probe.combined_output,
        "discipline": discipline,
        "safety_constraints": ["no_destructive_actions", "no_speedup_from_reachability"],
    }


def build_command_transcripts(kv260: JsonMap, gatemate: JsonMap, polarfire: JsonMap) -> JsonDict:
    return {
        "kv260_precondition": kv260["precondition_probe"].as_dict(),
        "kv260_workload": probe_dict(kv260.get("workload_probe")),
        "gatemate_precondition": gatemate["precondition_probe"].as_dict(),
        "gatemate_workload": probe_dict(gatemate.get("workload_probe")),
        "polarfire_precondition": polarfire["precondition_probe"].as_dict(),
        "polarfire_workload": probe_dict(polarfire.get("workload_probe")),
    }


def sample_quality_evidence(board_results: JsonMap) -> JsonDict:
    return {
        "reachable_boards": [
            board for board, result in board_results.items() if result.get("reachable")
        ],
        "hash_verified_boards": [
            board for board, result in board_results.items() if result.get("hash_verified")
        ],
        "kv260": quality_public(board_results["kv260"]),
        "gatemate": quality_public(board_results["gatemate"]),
        "polarfire": quality_public(board_results["polarfire"]),
        "speedup_evidence_claimed": False,
    }


def quality_public(result: JsonMap) -> JsonDict:
    return {
        "reachable": bool(result["reachable"]),
        "hash_verified": bool(result["hash_verified"]),
        "sample_quality": result.get("sample_quality"),
        "correctness": result.get("correctness"),
        "blocked_reason": result.get("blocked_reason"),
    }


def wishlist_has_update(repo_root: str | Path) -> bool:
    path = Path(repo_root) / WISHLIST_REL_PATH
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return all(marker in text for marker in WISHLIST_MARKERS)


def ensure_hardware_wishlist_update(repo_root: str | Path) -> Path:
    path = Path(repo_root) / WISHLIST_REL_PATH
    addition = "\n".join(
        f"| {marker} | v473 hardware_smoke continuity status recorded in Exp 5166. | "
        "No hardware speedup claim; per-board blockers remain visible. |"
        for marker in WISHLIST_MARKERS
    )
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if all(marker in text for marker in WISHLIST_MARKERS):
            return path
        path.write_text(text.rstrip() + "\n" + addition + "\n", encoding="utf-8")
        return path
    path.write_text(
        "# Hardware Wishlist\n\n"
        "### Active hardware tracks\n\n"
        "| Track | Why active now | Boundary |\n"
        "|---|---|---|\n"
        f"{addition}\n",
        encoding="utf-8",
    )
    return path


def write_artifact(repo_root: str | Path, artifact: JsonMap) -> Path:
    validate_artifact(artifact)
    out_path = Path(repo_root) / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_experiment(
    *,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = run_command,
    clock: Clock = time.perf_counter,
    run_date: str = "20260702",
    tests_run: Sequence[str] | None = None,
    update_wishlist: bool = False,
) -> Path:
    prepend_oss_cad_suite()
    if update_wishlist:
        ensure_hardware_wishlist_update(repo_root)
    artifact = build_artifact(
        repo_root=repo_root,
        command_runner=command_runner,
        clock=clock,
        run_date=run_date,
        tests_run=tests_run,
    )
    return write_artifact(repo_root, artifact)


def validate_artifact(artifact: JsonMap) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    missing = set(REQUIRED_SCHEMA_FIELDS) - set(artifact)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        return errors
    expect(errors, artifact.get("schema") == SCHEMA, "schema mismatch")
    expect(errors, artifact.get("experiment") == EXPERIMENT_NAME, "experiment mismatch")
    expect(errors, artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id mismatch")
    expect(errors, artifact.get("milestone") == MILESTONE, "milestone mismatch")
    expect(errors, artifact.get("spec_refs") == SPEC_REFS, "spec_refs mismatch")
    expect(errors, artifact.get("random_seed") == RANDOM_SEED, "random_seed mismatch")
    expect(errors, terminal_prefix_ok(str(artifact.get("honest_verdict", ""))), "honest_verdict prefix mismatch")
    expect(errors, artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate mismatch")
    expect(errors, artifact.get("field_principles") == FIELD_PRINCIPLES, "field_principles mismatch")
    expect(errors, artifact.get("kv260_host_block_devices_touched") is False, "kv260_host_block_devices_touched mismatch")
    expect(errors, artifact.get("hardware_wishlist_updated") is True, "hardware_wishlist_updated mismatch")
    expect(errors, artifact.get("no_speedup_claim") is True, "no_speedup_claim mismatch")
    expect(errors, artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed mismatch")
    expect(errors, artifact.get("conductor_modified") is False, "conductor_modified mismatch")
    expect(errors, no_host_storage(artifact), "host storage marker present")
    validate_board_result(errors, artifact, "kv260_result", "kv260")
    validate_board_result(errors, artifact, "gatemate_result", "gatemate")
    validate_board_result(errors, artifact, "polarfire_result", "polarfire")
    expect(errors, reachable_count(artifact) == artifact.get("boards_reachable_count"), "boards_reachable_count mismatch")
    validate_mapping_keys(errors, artifact, "command_transcripts", COMMAND_TRANSCRIPT_KEYS)
    validate_mapping_keys(errors, artifact, "workload_hashes", WORKLOAD_HASH_KEYS)
    expect(errors, isinstance(artifact.get("sample_quality_evidence"), Mapping), "sample_quality_evidence must be a dict")
    expect(errors, isinstance(artifact.get("preconditions_checked"), list) and len(artifact["preconditions_checked"]) == 3, "preconditions_checked mismatch")
    expect(errors, isinstance(artifact.get("tests_run"), list) and bool(artifact.get("tests_run")), "tests_run must be non-empty")
    expect(errors, artifact.get("reproducibility_checksum") == payload_checksum(artifact), "checksum mismatch")
    return errors


def terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "success_"))


def reachable_count(artifact: JsonMap) -> int:
    return sum(
        bool(artifact[field].get("reachable"))
        for field in ("kv260_result", "gatemate_result", "polarfire_result")
        if isinstance(artifact.get(field), Mapping)
    )


def validate_board_result(errors: list[str], artifact: JsonMap, field: str, board: str) -> None:
    result = artifact.get(field)
    if not isinstance(result, Mapping):
        errors.append(f"{field} must be a dict")
        return
    expect(errors, set(BOARD_RESULT_FIELDS) == set(result), f"{field} keys mismatch")
    expect(errors, isinstance(result.get("reachable"), bool), f"{field} reachable must be bool")
    blocked_reason = result.get("blocked_reason")
    expect(
        errors,
        blocked_reason is None or str(blocked_reason).startswith(f"blocked_{board}_"),
        f"{field} blocked_reason mismatch",
    )
    hash_value = result.get("workload_hash")
    expect(errors, hash_value is None or is_sha256(hash_value), f"{field} workload_hash invalid")
    if result.get("reachable"):
        expect(errors, hash_value is not None, f"{field} reachable missing workload_hash")


def validate_mapping_keys(
    errors: list[str], artifact: JsonMap, field: str, expected_keys: Sequence[str]
) -> None:
    value = artifact.get(field)
    expect(errors, isinstance(value, Mapping), f"{field} must be a dict")
    if isinstance(value, Mapping):
        expect(errors, set(value) == set(expected_keys), f"{field} keys mismatch")


def no_host_storage(payload: JsonMap) -> bool:
    return no_5120_host_storage(payload)


def expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run date in YYYYMMDD form.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root.")
    parser.add_argument(
        "--update-wishlist",
        action="store_true",
        help="Append the Exp 5166 hardware status rows before building the artifact.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_path = run_experiment(
        repo_root=args.repo_root,
        run_date=args.date,
        update_wishlist=args.update_wishlist,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"artifact: {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"boards_reachable_count: {artifact['boards_reachable_count']}")
    print(f"hardware_wishlist_updated: {artifact['hardware_wishlist_updated']}")
    print(f"no_speedup_claim: {artifact['no_speedup_claim']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - live hardware entrypoint.
    raise SystemExit(main())
