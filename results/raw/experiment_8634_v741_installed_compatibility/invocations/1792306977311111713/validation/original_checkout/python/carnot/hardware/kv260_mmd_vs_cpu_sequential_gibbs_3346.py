"""Exp 3346 KV260 MMD-vs-CPU sequential-Gibbs continuity rerun (live SSH path).

Spec refs: REQ-HW-101, SCENARIO-HW-101.

WHY THIS EXPERIMENT EXISTS
--------------------------
Exp 2938 already answered the scientific question once: the KV260 fabric's
synchronous fixed-sweep Glauber sampler produces an energy distribution that is
statistically *distinguishable* from a detailed-balance CPU sequential-Gibbs
chain on the same dense n=64 Ising problems, so the board output cannot be
called "exact Boltzmann sampling".  This experiment is the deferred *continuity*
rerun: it re-confirms that finding using the board's CURRENT access path
(``ssh kria`` + ``xmutil`` overlay + UIO register map) on 2026-05-29, and it
captures board-local evidence (uname, overlay status, UIO device list) so a
third party can see the run actually touched the board.

WHY IT NEVER LOOKS AT A HOST SD CARD
------------------------------------
Five consecutive milestones (.254-.259) escalated the operator for a phantom
"insert SD card into the host" action because an earlier task checked
``/dev/mmcblk*`` on the development machine.  That checks the HOST's card slot,
which is irrelevant once the KV260 has booted Ubuntu Xilinx from its own onboard
storage.  The only meaningful precondition is SSH reachability of the board.
This module's sole hardware precondition is therefore
``ssh -o ConnectTimeout=5 -o BatchMode=yes kria true`` and there is no
host-block-device check anywhere (CLAUDE.md "KV260 SSH-Not-SD-Card Discipline").

DESIGN: TESTABLE CORE + THIN LIVE LAYER
---------------------------------------
The statistics (RBF MMD², KS test, CPU sequential Gibbs, Exp 2898 problem
reproduction) are imported from the Exp 2938 module so we do not fork the
science.  The pure, unit-tested layer here is :func:`build_artifact`,
:func:`summarize_energies`, :func:`validate_artifact`, and :func:`run_experiment`
(driven through an injected board collector).  The only ``# pragma: no cover``
code is :func:`collect_kv260`, which requires the physical board.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from carnot.hardware import kv260_mmd_vs_cpu_sequential_gibbs as exp2938
from carnot.hardware.kv260_mmd_vs_cpu_sequential_gibbs import (
    CommandResult,
    DenseIsingProblem,
    EnergyRunResult,
    ProblemReproductionError,
    compare_energy_distributions,
    recover_exp2898_problems,
    run_cpu_sequential_gibbs,
    sha256_canonical,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP2898_REL_PATH = exp2938.EXP2898_REL_PATH
OUTPUT_REL_PATH = Path(
    "results/experiment_3346_kv260_mmd_vs_cpu_sequential_gibbs_v1.json"
)
TRANSCRIPT_REL_PATH = Path("results/experiment_3346_kv260_mmd_transcript.log")
OUTPUT_DIR = REPO_ROOT / "output" / "experiment_3346_kv260_mmd"
LOCAL_PROBLEM_PATH = OUTPUT_DIR / "problem_payload.json"
LOCAL_HARNESS_PATH = OUTPUT_DIR / "board_harness.py"
REMOTE_PROBLEM_PATH = "/tmp/experiment_3346_kv260_problem.json"
REMOTE_HARNESS_PATH = "/tmp/experiment_3346_kv260_board_harness.py"

EXPERIMENT_ID = 3346
EXPERIMENT_NAME = "exp3346-kv260-mmd-vs-cpu-sequential-gibbs-v1"
RUN_DATE = "20260529"
INFERENCE_SUBSTRATE = "hardware_smoke"
KV260_HOST = exp2938.KV260_HOST

RANDOM_SEEDS = list(exp2938.RANDOM_SEEDS)
PRIMARY_SEED = RANDOM_SEEDS[0]
N_SPINS = exp2938.N_SPINS
SIGNIFICANCE_ALPHA = exp2938.SIGNIFICANCE_ALPHA

# Continuity smoke: the "smallest available sampler path".  The Exp 2938
# headline used 10,000 energies/seed; the distinguishability signal is enormous
# (KS statistic ~0.998), so a reduced trace re-confirms it quickly while keeping
# the board run short.  The count actually used is always recorded in the
# artifact so the reduction is auditable rather than hidden.
CONTINUITY_N_SAMPLES = 512
CONTINUITY_BURN_IN = 300
N_PERMUTATIONS = 200
MAX_PERMUTATION_SAMPLES = exp2938.MAX_PERMUTATION_SAMPLES

CPU_UPDATE_SCHEDULE = exp2938.CPU_UPDATE_SCHEDULE
KV260_UPDATE_SCHEDULE = exp2938.KV260_UPDATE_SCHEDULE

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "files_updated",
    "ssh_reachable",
    "board_uname",
    "xmutil_status",
    "uio_status",
    "cpu_baseline_summary",
    "kv260_summary",
    "mmd_vs_cpu",
    "sample_count_cpu",
    "sample_count_kv260",
    "command_transcript",
    "blocked_reasons",
}


@dataclass(frozen=True)
class BoardEvidence:
    """Everything the continuity rerun learned by touching the board.

    A blocked run still returns a populated instance (with ``blocked_reasons``
    non-empty) so the artifact can preserve the transcript and the precise
    failure, exactly as the Pre-Launch Preconditions discipline requires.
    """

    ssh_reachable: bool
    board_uname: str = ""
    xmutil_status: str = ""
    uio_status: str = ""
    bitstream_sha256: str = ""
    energies_by_seed: dict[int, list[float]] = field(default_factory=dict)
    command_transcript: list[dict[str, Any]] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    transcript_path: str = ""
    board_summary: dict[str, Any] = field(default_factory=dict)


def summarize_energies(energies: Sequence[float]) -> dict[str, Any]:
    """Reduce an energy trace to an auditable, JSON-safe summary.

    We keep mean/std/min/max plus a SHA256 of the full trace.  The summary is
    small enough to live in the artifact, and the checksum lets a reproducer
    confirm they recovered bit-identical energies without us storing thousands
    of floats inline.
    """

    values = np.asarray(list(energies), dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sha256": sha256_canonical([]),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "sha256": sha256_canonical([round(float(v), 12) for v in values.tolist()]),
    }


def _record(transcript: list[dict[str, Any]], label: str, result: CommandResult) -> str:
    """Append one command to the transcript and return its combined output."""

    combined = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    transcript.append(
        {
            "label": label,
            "returncode": int(result.returncode),
            "duration_s": round(float(result.duration_s), 6),
            "stdout_tail": (result.stdout or "").strip()[-600:],
            "stderr_tail": (result.stderr or "").strip()[-600:],
        }
    )
    return combined


def _flush_transcript(root_path: Path, transcript: list[dict[str, Any]]) -> None:  # pragma: no cover - live board only
    """Persist the in-memory transcript to the Exp 3346 log file.

    Writing to the Exp 3346-specific path (never the Exp 2938 path) is what
    keeps the historical Exp 2938 transcript untouched.
    """

    path = root_path / TRANSCRIPT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["experiment_3346 continuity rerun transcript"]
    for entry in transcript:
        lines.append(json.dumps(entry, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_kv260(
    problems: Sequence[DenseIsingProblem],
    *,
    n_samples: int = CONTINUITY_N_SAMPLES,
    root_path: Path = REPO_ROOT,
) -> BoardEvidence:  # pragma: no cover - requires the live KV260 board
    """Drive the live KV260 over SSH and return board evidence + energies.

    Reuses only the *stateless* Exp 2938 SSH/UIO helpers (``_ssh``, ``_scp``,
    ``_run``, payload/harness builders) and the on-board harness source, but
    owns its own transcript file and ``/tmp`` payload paths so it never writes
    to — and therefore never clobbers — any Exp 2938 artifact.
    """

    transcript: list[dict[str, Any]] = []

    # --- Sole hardware precondition: board reachable over SSH. ---
    # ssh -o ConnectTimeout=5 -o BatchMode=yes kria true  (no host SD-card check)
    ssh_probe = exp2938._run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", KV260_HOST, "true"],
        timeout=10,
    )
    _record(transcript, "precondition_ssh", ssh_probe)
    if ssh_probe.returncode != 0:
        _flush_transcript(root_path, transcript)
        return BoardEvidence(
            ssh_reachable=False,
            command_transcript=transcript,
            blocked_reasons=[
                f"blocked_kv260_ssh_unreachable: rc={ssh_probe.returncode} {ssh_probe.stderr.strip()[:200]}"
            ],
            transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        )

    uname = exp2938._ssh("uname -a", timeout=15)
    board_uname = _record(transcript, "board_uname", uname).strip()
    overlays = exp2938._ssh("sudo xmutil listapps 2>&1", timeout=20)
    xmutil_status = _record(transcript, "xmutil_listapps", overlays).strip()
    uio = exp2938._ssh("ls -1 /dev/uio* 2>&1", timeout=15)
    uio_status = _record(transcript, "uio_devices", uio).strip()

    overlay_ok = overlays.returncode == 0 and exp2938._detect_overlay(
        overlays.stdout + overlays.stderr
    )
    if not overlay_ok:
        _flush_transcript(root_path, transcript)
        return BoardEvidence(
            ssh_reachable=True,
            board_uname=board_uname,
            xmutil_status=xmutil_status,
            uio_status=uio_status,
            command_transcript=transcript,
            blocked_reasons=["blocked_kv260_overlay_missing"],
            transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        )

    load = exp2938._ssh(exp2938.OVERLAY_LOAD_COMMAND, timeout=60)
    _record(transcript, "overlay_load", load)

    uio0 = exp2938._ssh("ls /dev/uio0 2>/dev/null && echo ok", timeout=15)
    _record(transcript, "precondition_uio0", uio0)
    if not (uio0.returncode == 0 and "ok" in uio0.stdout.split()):
        _flush_transcript(root_path, transcript)
        return BoardEvidence(
            ssh_reachable=True,
            board_uname=board_uname,
            xmutil_status=xmutil_status,
            uio_status=uio_status,
            command_transcript=transcript,
            blocked_reasons=["blocked_kv260_uio_devices_absent"],
            transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        )

    bit = exp2938._ssh(
        "sha256sum /lib/firmware/xilinx/carnot_ising_v4/*.bit 2>/dev/null | head -n 1",
        timeout=30,
    )
    _record(transcript, "bitstream_sha256", bit)
    bitstream_sha, bitstream_path = exp2938._parse_sha256sum(bit.stdout)
    if bitstream_sha is None:
        _flush_transcript(root_path, transcript)
        return BoardEvidence(
            ssh_reachable=True,
            board_uname=board_uname,
            xmutil_status=xmutil_status,
            uio_status=uio_status,
            command_transcript=transcript,
            blocked_reasons=["blocked_active_bitstream_sha256_missing"],
            transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROBLEM_PATH.write_text(
        json.dumps(
            exp2938._problem_payload_for_board(list(problems), n_samples),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    LOCAL_HARNESS_PATH.write_text(exp2938.BOARD_HARNESS_SOURCE, encoding="utf-8")

    problem_scp = exp2938._scp(LOCAL_PROBLEM_PATH, REMOTE_PROBLEM_PATH, timeout=60)
    _record(transcript, "scp_problem_payload", problem_scp)
    harness_scp = exp2938._scp(LOCAL_HARNESS_PATH, REMOTE_HARNESS_PATH, timeout=60)
    _record(transcript, "scp_board_harness", harness_scp)
    if problem_scp.returncode != 0 or harness_scp.returncode != 0:
        _flush_transcript(root_path, transcript)
        return BoardEvidence(
            ssh_reachable=True,
            board_uname=board_uname,
            xmutil_status=xmutil_status,
            uio_status=uio_status,
            bitstream_sha256=bitstream_sha,
            command_transcript=transcript,
            blocked_reasons=["blocked_kv260_payload_scp_failed"],
            transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        )

    harness = exp2938._ssh(
        f"sudo python3 {REMOTE_HARNESS_PATH} {REMOTE_PROBLEM_PATH}", timeout=1800
    )
    _record(transcript, "run_board_harness", harness)
    if harness.returncode != 0:
        _flush_transcript(root_path, transcript)
        return BoardEvidence(
            ssh_reachable=True,
            board_uname=board_uname,
            xmutil_status=xmutil_status,
            uio_status=uio_status,
            bitstream_sha256=bitstream_sha,
            command_transcript=transcript,
            blocked_reasons=["blocked_kv260_board_harness_failed"],
            transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        )

    board_payload = exp2938._extract_board_json(harness.stdout)
    energies_by_seed = {
        int(seed): [float(value) for value in values]
        for seed, values in board_payload.get("energies_by_seed", {}).items()
    }
    _flush_transcript(root_path, transcript)
    return BoardEvidence(
        ssh_reachable=True,
        board_uname=board_uname,
        xmutil_status=xmutil_status,
        uio_status=uio_status,
        bitstream_sha256=bitstream_sha,
        energies_by_seed=energies_by_seed,
        command_transcript=transcript,
        blocked_reasons=[],
        transcript_path=TRANSCRIPT_REL_PATH.as_posix(),
        board_summary={
            "bitstream_path": bitstream_path,
            "selected_uio": board_payload.get("selected_uio"),
            "selected_uio_addr_hex": board_payload.get("selected_uio_addr_hex"),
            "board_harness_duration_s": board_payload.get("duration_s"),
        },
    )


def _recommendation(distinguishable: bool) -> str:
    if distinguishable:
        return (
            "retract: continuity rerun re-confirms the KV260 synchronous Glauber "
            "energy distribution is distinguishable from CPU sequential Gibbs at "
            "p<0.01; paper-v6 must frame board output as fixed-schedule heuristic "
            "samples, not exact Boltzmann sampling."
        )
    return (
        "retain: continuity rerun finds all per-seed MMD and KS p-values >=0.01; "
        "the narrow approximately-Boltzmann claim survives this board path."
    )


def build_artifact(
    *,
    evidence: BoardEvidence,
    problems: Sequence[DenseIsingProblem],
    cpu_runs: Mapping[int, EnergyRunResult],
    comparisons: Mapping[int, Mapping[str, float]],
    sample_count_cpu: int,
    sample_count_kv260: int,
    duration_s: float,
    files_updated: Sequence[str],
    blocked_verdict: str = "",
) -> dict[str, Any]:
    """Assemble the Exp 3346 artifact (success or blocked) as a plain dict."""

    base: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT_NAME,
        "run_date": RUN_DATE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": PRIMARY_SEED,
        "random_seeds_used": list(RANDOM_SEEDS),
        "duration_s": round(float(duration_s), 6),
        "files_updated": list(files_updated),
        "ssh_reachable": bool(evidence.ssh_reachable),
        "board_uname": evidence.board_uname,
        "xmutil_status": evidence.xmutil_status,
        "uio_status": evidence.uio_status,
        "bitstream_sha256_cited": evidence.bitstream_sha256,
        "command_transcript": list(evidence.command_transcript),
        "hardware_transcript_path": evidence.transcript_path,
        "board_summary": dict(evidence.board_summary),
        "cpu_update_schedule": CPU_UPDATE_SCHEDULE,
        "kv260_update_schedule": KV260_UPDATE_SCHEDULE,
        "source_artifacts": [
            EXP2898_REL_PATH.as_posix(),
            "results/experiment_2938_kv260_mmd_vs_cpu_sequential_gibbs_v1.json",
        ],
    }

    if blocked_verdict:
        base.update(
            {
                "honest_verdict": blocked_verdict,
                "cpu_baseline_summary": {},
                "kv260_summary": {},
                "mmd_vs_cpu": {},
                "sample_count_cpu": int(sample_count_cpu),
                "sample_count_kv260": int(sample_count_kv260),
                "blocked_reasons": list(evidence.blocked_reasons) or [blocked_verdict],
                "reproducibility_checksum": "",
                "paper_v6_recommendation": (
                    f"{blocked_verdict}: KV260 continuity comparison did not run."
                ),
            }
        )
        return base

    per_seed = {}
    distinguishable = False
    for seed in RANDOM_SEEDS:
        stats = comparisons[seed]
        per_seed[str(seed)] = {
            "mmd_squared": float(stats["mmd_squared"]),
            "mmd_pvalue": float(stats["mmd_pvalue"]),
            "ks_statistic": float(stats["ks_statistic"]),
            "ks_pvalue": float(stats["ks_pvalue"]),
            "bandwidth": float(stats["bandwidth"]),
        }
        if (
            float(stats["mmd_pvalue"]) < SIGNIFICANCE_ALPHA
            or float(stats["ks_pvalue"]) < SIGNIFICANCE_ALPHA
        ):
            distinguishable = True

    cpu_summary = {
        str(seed): summarize_energies(cpu_runs[seed].energies) for seed in RANDOM_SEEDS
    }
    cpu_summary["update_schedule"] = CPU_UPDATE_SCHEDULE
    kv260_summary = {
        str(seed): summarize_energies(evidence.energies_by_seed[seed])
        for seed in RANDOM_SEEDS
    }
    kv260_summary["update_schedule"] = KV260_UPDATE_SCHEDULE
    kv260_summary["bitstream_sha256"] = evidence.bitstream_sha256

    mmd_vs_cpu = {
        "per_seed": per_seed,
        "distributions_distinguishable": bool(distinguishable),
        "significance_alpha": SIGNIFICANCE_ALPHA,
        "paper_v6_recommendation": _recommendation(distinguishable),
    }

    reproducibility_payload = {
        "problem_checksums": {
            str(p.seed): {
                "j_matrix_sha256": p.j_matrix_sha256,
                "h_vector_sha256": p.h_vector_sha256,
            }
            for p in problems
        },
        "cpu_summary": {seed: cpu_summary[str(seed)] for seed in map(str, RANDOM_SEEDS)},
        "kv260_summary": {
            seed: kv260_summary[str(seed)] for seed in map(str, RANDOM_SEEDS)
        },
        "bitstream_sha256": evidence.bitstream_sha256,
        "mmd_vs_cpu": per_seed,
        "sample_count_cpu": int(sample_count_cpu),
        "sample_count_kv260": int(sample_count_kv260),
    }

    base.update(
        {
            "honest_verdict": "complete: kv260_mmd_vs_cpu_sequential_gibbs_continuity_recorded",
            "cpu_baseline_summary": cpu_summary,
            "kv260_summary": kv260_summary,
            "mmd_vs_cpu": mmd_vs_cpu,
            "sample_count_cpu": int(sample_count_cpu),
            "sample_count_kv260": int(sample_count_kv260),
            "blocked_reasons": [],
            "reproducibility_checksum": sha256_canonical(reproducibility_payload),
            "paper_v6_recommendation": _recommendation(distinguishable),
            "distributions_distinguishable": bool(distinguishable),
            "continuity_n_permutations": int(N_PERMUTATIONS),
            "cpu_burn_in_sweeps": int(CONTINUITY_BURN_IN),
            "methodology_note": (
                "Continuity rerun of Exp 2938 over the live ssh kria path on "
                f"{RUN_DATE}. CPU baseline: dense n=64 sequential single-spin Gibbs, "
                "random spin order, beta=1.0. KV260: existing Exp 2898 UIO upload + "
                "synchronous fixed-sweep Glauber schedule, bitstream unmodified. "
                f"Reduced continuity sample count ({sample_count_cpu} energies/seed, "
                f"{CONTINUITY_BURN_IN} CPU burn-in sweeps) is the smallest sampler "
                "path that re-confirms the Exp 2938 finding; the headline run used "
                "10,000/seed. MMD2 uses an RBF kernel with median pairwise distance "
                f"bandwidth; MMD p-value uses {N_PERMUTATIONS} balanced permutations; "
                "KS uses scipy.stats.ks_2samp."
            ),
        }
    )
    return base


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail loudly if the artifact does not honor the REQ-HW-101 schema."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be hardware_smoke")
    verdict = str(artifact["honest_verdict"])
    if verdict.startswith("blocked_"):
        if not artifact["blocked_reasons"]:
            raise ValueError("blocked artifact must record blocked_reasons")
        if artifact["mmd_vs_cpu"]:
            raise ValueError("blocked artifact must leave mmd_vs_cpu empty")
        return
    if not (verdict.startswith("complete:") or verdict.startswith("success:")):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if artifact["random_seed"] not in RANDOM_SEEDS:
        raise ValueError("random_seed must be one of the Exp 2898 seeds")
    per_seed = artifact["mmd_vs_cpu"].get("per_seed", {})
    if len(per_seed) != 3:
        raise ValueError("successful mmd_vs_cpu must record three seeds")
    if int(artifact["sample_count_cpu"]) != int(artifact["sample_count_kv260"]):
        raise ValueError("sample_count_cpu must equal sample_count_kv260")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 string")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - float(started_s)), 6)


def run_experiment(
    *,
    root_path: Path = REPO_ROOT,
    board_collector: Callable[[Sequence[DenseIsingProblem]], BoardEvidence] | None = None,
    cpu_energy_runner: Callable[..., EnergyRunResult] | None = None,
    n_samples: int = CONTINUITY_N_SAMPLES,
    burn_in_sweeps: int = CONTINUITY_BURN_IN,
    n_permutations: int = N_PERMUTATIONS,
    max_permutation_samples: int = MAX_PERMUTATION_SAMPLES,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Orchestrate the continuity rerun and write the v1 artifact.

    The board interaction is injected via ``board_collector`` so the whole
    control flow (provenance gate, blocked branches, success path) is unit
    testable without the physical board.
    """

    started = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    files_updated = [OUTPUT_REL_PATH.as_posix(), TRANSCRIPT_REL_PATH.as_posix()]

    exp2898_path = root_path / EXP2898_REL_PATH
    if not exp2898_path.exists():
        artifact = build_artifact(
            evidence=BoardEvidence(
                ssh_reachable=False,
                blocked_reasons=["blocked_exp2898_artifact_missing"],
            ),
            problems=[],
            cpu_runs={},
            comparisons={},
            sample_count_cpu=n_samples,
            sample_count_kv260=0,
            duration_s=_duration(started, now_s),
            files_updated=files_updated,
            blocked_verdict="blocked_exp2898_artifact_missing",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    try:
        exp2898 = json.loads(exp2898_path.read_text(encoding="utf-8"))
        problems = recover_exp2898_problems(exp2898)
    except (json.JSONDecodeError, OSError, ProblemReproductionError) as exc:
        artifact = build_artifact(
            evidence=BoardEvidence(
                ssh_reachable=False,
                blocked_reasons=[f"blocked_exp2898_problem_reproduction_failed: {exc}"],
            ),
            problems=[],
            cpu_runs={},
            comparisons={},
            sample_count_cpu=n_samples,
            sample_count_kv260=0,
            duration_s=_duration(started, now_s),
            files_updated=files_updated,
            blocked_verdict="blocked_exp2898_problem_reproduction_failed",
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    collector = board_collector or (
        lambda active: collect_kv260(active, n_samples=n_samples, root_path=root_path)
    )
    evidence = collector(problems)

    if not evidence.ssh_reachable or evidence.blocked_reasons:
        verdict = (
            evidence.blocked_reasons[0].split(":", 1)[0]
            if evidence.blocked_reasons
            else "blocked_kv260_ssh_unreachable"
        )
        if not verdict.startswith("blocked_"):
            verdict = "blocked_kv260_board_path_failed"
        artifact = build_artifact(
            evidence=evidence,
            problems=problems,
            cpu_runs={},
            comparisons={},
            sample_count_cpu=n_samples,
            sample_count_kv260=0,
            duration_s=_duration(started, now_s),
            files_updated=files_updated,
            blocked_verdict=verdict,
        )
        validate_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    cpu_runner = cpu_energy_runner or run_cpu_sequential_gibbs
    cpu_runs = {
        problem.seed: cpu_runner(
            problem, n_samples=n_samples, burn_in_sweeps=burn_in_sweeps
        )
        for problem in problems
    }

    for seed in RANDOM_SEEDS:
        if len(evidence.energies_by_seed.get(seed, [])) != int(n_samples):
            artifact = build_artifact(
                evidence=BoardEvidence(
                    ssh_reachable=True,
                    board_uname=evidence.board_uname,
                    xmutil_status=evidence.xmutil_status,
                    uio_status=evidence.uio_status,
                    bitstream_sha256=evidence.bitstream_sha256,
                    command_transcript=evidence.command_transcript,
                    blocked_reasons=["blocked_kv260_energy_trace_incomplete"],
                    transcript_path=evidence.transcript_path,
                    board_summary=evidence.board_summary,
                ),
                problems=problems,
                cpu_runs={},
                comparisons={},
                sample_count_cpu=n_samples,
                sample_count_kv260=len(evidence.energies_by_seed.get(seed, [])),
                duration_s=_duration(started, now_s),
                files_updated=files_updated,
                blocked_verdict="blocked_kv260_energy_trace_incomplete",
            )
            validate_artifact(artifact)
            _write_json(output_path, artifact)
            return artifact

    comparisons = {
        seed: compare_energy_distributions(
            cpu_runs[seed].energies,
            evidence.energies_by_seed[seed],
            seed=seed,
            n_permutations=n_permutations,
            max_permutation_samples=max_permutation_samples,
        )
        for seed in RANDOM_SEEDS
    }

    artifact = build_artifact(
        evidence=evidence,
        problems=problems,
        cpu_runs=cpu_runs,
        comparisons=comparisons,
        sample_count_cpu=n_samples,
        sample_count_kv260=n_samples,
        duration_s=_duration(started, now_s),
        files_updated=files_updated,
    )
    validate_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--print-result-path", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(root_path=args.root)
    if args.print_result_path:
        print(args.root / OUTPUT_REL_PATH)
    else:
        print(
            json.dumps(
                {
                    "honest_verdict": artifact["honest_verdict"],
                    "result": str(args.root / OUTPUT_REL_PATH),
                }
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
