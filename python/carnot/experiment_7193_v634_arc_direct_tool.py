"""Run one bounded direct-selfparse ARC session through the shipped E3 policy.

The driver checks all external inputs before model work. The session child uses
the real offline arcade and ``arc_leaderboard_eval.run_game``. Policy input is
limited to public frames and its own transitions.

Spec refs: REQ-ARC-WMTE-7193 and SCENARIO-ARC-WMTE-7193-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

TASK_ID = "exp7193-arc-direct-tool"
MILESTONE = "2026.09.634"
RUN_DATE = "20260910"
GAME = "r11l"
RANDOM_SEED = 7_193_001
ACTION_BUDGET = 4000
SESSION_TIMEOUT_S = 3600
INDUCTION_TIMEOUT_S = 2400
N_CTX = 49152
COMPLETION_BUDGET = 4096
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
EXPECTED_PRIOR_VERDICT = "blocked_required_source_bytes"
HISTORICAL_TARGET = 10

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
PRIOR_ARTIFACT_PATH = Path("results/experiment_7186_v633_arc_withheld_transfer.json")
HISTORICAL_RECEIPT_PATH = Path("results/arc_leaderboard_eval_runs/r11l-1594772.json")
EVAL_PATH = Path("scripts/arc_leaderboard_eval.py")
POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
MODULE_PATH = Path("python/carnot/experiment_7193_v634_arc_direct_tool.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7193_v634_arc_direct_tool.py")
TEST_PATH = Path("tests/python/test_experiment_7193_v634_arc_direct_tool.py")
RESULT_PATH = Path("results/experiment_7193_v634_arc_direct_tool.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7193_v634_arc_direct_tool/running.json")
RAW_DIR = Path("results/raw/experiment_7193")

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    Path("ops/arc_solve_registry.yaml"),
    ROADMAP_PATH,
    EVAL_PATH,
    POLICY_PATH,
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_tool_gap_receipt.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/inference/llama_server_supervisor.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    SPEC_PATH,
    PRIOR_ARTIFACT_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260910, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when verification uses the same correctness authority; separate implementations alone do not remove circularity.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings, including nulls; blocked_* for external blocks. Never promote infrastructure readiness as scientific benefit.",
    "arc_tool_measurement_complete_score": "One means the single scheduled session has a terminal receipt, not that tool engagement succeeded.",
    "MODEL_SPECS": "Include unsloth/Qwen3.8-27B-GGUF and resolved local bytes for every actual load.",
    "model_specs": "Mirror actual loaded identity; do not claim uninvoked comparators.",
    "phase_spans": "Monotonic spans attribute the cost of each call.",
    "gpu_receipts": "Task-linked samples establish CUDA execution rather than idle allocations.",
    "runner_receipt": "Record one model, replica count, lease ownership and runner choice.",
    "per_game_results": "The single session retains seed, caps, observed tool engagement, banked progress, errors and costs.",
    "tool_induction_rows": "Actual tool engagement, not supervisor firings, supplies the denominator.",
    "solve_provenance": "Use live_agent_self_discovery for this policy-owned runtime discovery.",
    "adapter_isolation_receipt": "The live policy must have no per-game solution access.",
    "arc_volume_sufficient_score": "One requires ten new measured tool-loop inductions; keep historical and new counts separate.",
    "arc_tool_engagement_score": "One requires a real returned tool-loop induction, not merely a model load or cancelled session.",
    "inference_mode": "live_gpu requires measured CUDA execution in the task window.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_DURATION_FLOORS = {
    "model_load_no_generation": 2.0,
    "model_bounded_generation": 10.0,
    "model_full_generation": 60.0,
}


def canonical_json(value: Any) -> str:
    """Serialize evidence in one stable form for hashes and atomic files."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    """Return a labeled digest so hashes cannot be confused with file names."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash the bytes opened from one required regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Replace a file only after all new bytes are durable in its directory."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def atomic_write(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically so a killed run cannot publish half a receipt."""

    atomic_write_bytes(Path(path), (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the terminal artifact except for the digest field itself."""

    body = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_bytes(canonical_json(body).encode())


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
    **extra: Any,
) -> JsonDict:
    """Keep each gate reason beside its expected and observed evidence."""

    row = {
        "check": str(check),
        "upstream": str(upstream),
        "field": str(field),
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed) if passed is None else bool(passed),
    }
    row.update(deepcopy(extra))
    return row


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed external check without hiding the complete list."""

    rows = [deepcopy(dict(row)) for row in checks]
    failure = next((row for row in rows if row.get("passed") is not True), None)
    if failure is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
            "checks": rows,
        }
    return {
        "passed": False,
        "failed_check": failure.get("check"),
        "upstream": failure.get("upstream"),
        "field": failure.get("field"),
        "expected_value": failure.get("expected_value"),
        "observed_value": failure.get("observed_value"),
        "checks": rows,
    }


def _quarantined(payload: Mapping[str, Any]) -> bool:
    """Read the repository's quarantine authority and fail closed to its base flag."""

    try:
        if str(SCRIPTS_ROOT) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_ROOT))
        from conductor_gates import _is_quarantined

        return bool(_is_quarantined(dict(payload)))
    except Exception:
        flag = payload.get("flagged_adversarial")
        if isinstance(flag, Mapping) and "value" in flag:
            flag = flag.get("value")
        return flag is True


def upstream_field_gate(
    payload: Mapping[str, Any], *, upstream: str, field: str, expected: Any
) -> JsonDict:
    """Reject quarantine before reading a matching structured field as authority."""

    quarantine = _quarantined(payload)
    value = payload.get(field)
    observed = {"value": value, "quarantined": quarantine, "consumed": False}
    return gate_check(
        "upstream_field_not_quarantined",
        upstream,
        field,
        {"value": expected, "quarantined": False, "consumed": False},
        observed,
        not quarantine and value == expected,
        evidence_role="addressed_prior_failure_only",
    )


def adapter_isolation_receipt() -> JsonDict:
    """Declare the enforced process boundary used by the live session child."""

    denied = [
        "per_game_adapter",
        "game_source",
        "registry_contents",
        "solved_trajectories",
        "banked_solutions",
    ]
    return {
        "passed": True,
        "verified_before_model_work": True,
        "policy_class": "E3AgentPolicy",
        "policy_constructor_solutions_argument": "not_present",
        "policy_inputs": ["public_frames", "available_actions", "own_transitions"],
        "policy_denied": denied,
        "environment_executable_source_allowed": True,
        "registry_read_role": "evaluator_historical_count_before_launch_only",
        "entrypoint": "arc_leaderboard_eval.run_game -> E3AgentPolicy",
    }


def session_environment(
    base: Mapping[str, str],
    *,
    model_path: str,
    gpu_index: int,
    port: int,
    raw_dir: Path,
) -> dict[str, str]:
    """Build the child environment before any ARC module with import-time state loads."""

    env = dict(base)
    env.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPO_ROOT / "python"),
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_INDUCE_N_CTX": str(N_CTX),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(COMPLETION_BUDGET),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(INDUCTION_TIMEOUT_S),
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu_index),
            "CARNOT_ARC_GGUF_PATH": str(model_path),
            "CARNOT_ARC_LLM_BACKEND": "llamacpp",
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_RANDOM_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_PROPOSER_PORT": str(port),
            "CARNOT_ARC_TOOL_GAP_RECEIPTS": "1",
            "CARNOT_ARC_TOOL_GAP_RECEIPT_PATH": str(raw_dir / "tool_gap_receipts.json"),
            "CARNOT_ARC_E3_DIR": str(raw_dir / "e3"),
            "CARNOT_ARC_SERVER_LOG_DIR": str(raw_dir / "server_logs"),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
        }
    )
    return env


def trace_shipped_path(eval_path: Path, policy_path: Path) -> JsonDict:
    """Trace the existing environment loop and the attempt-level gap attachment."""

    try:
        eval_source = eval_path.read_text(encoding="utf-8")
        policy_source = policy_path.read_text(encoding="utf-8")
    except OSError:
        eval_source = ""
        policy_source = ""
    run_game = "def run_game(" in eval_source
    e3_policy = "class E3AgentPolicy" in policy_source or "tool_gap_events" in policy_source
    attachment = "tool_gap_events" in policy_source and 'attempt["tool_gap"]' in policy_source
    induction = "_induce_and_plan" in policy_source or attachment
    return {
        "run_game": run_game,
        "e3_policy": e3_policy,
        "induction_call": induction,
        "tool_gap_attachment": attachment,
        "passed": run_game and e3_policy and induction and attachment,
    }


def parse_gpu_inventory(gpu_text: str, app_text: str) -> list[JsonDict]:
    """Join GPU capacity and compute processes by UUID from ``nvidia-smi`` output."""

    devices: list[JsonDict] = []
    by_uuid: dict[str, JsonDict] = {}
    for line in gpu_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            row = {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "free_memory_mb": int(parts[3]),
                "total_memory_mb": int(parts[4]),
                "compute_apps": [],
            }
        except ValueError:
            continue
        devices.append(row)
        by_uuid[str(row["uuid"])] = row
    for line in app_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or parts[1] not in by_uuid:
            continue
        try:
            memory = int(parts[2].removesuffix(" MiB").strip())
            by_uuid[parts[1]]["compute_apps"].append(
                {"pid": int(parts[0]), "used_memory_mb": memory}
            )
        except ValueError:
            continue
    return devices


def choose_gpu(
    inventory: Sequence[Mapping[str, Any]], minimum_free_mb: int = 21000
) -> JsonDict | None:
    """Choose the freest 24 GiB NVIDIA card with no existing compute process."""

    eligible = [
        dict(row)
        for row in inventory
        if int(row.get("free_memory_mb", 0)) >= minimum_free_mb
        and int(row.get("total_memory_mb", 0)) >= 24000
        and not row.get("compute_apps")
    ]
    return max(eligible, key=lambda row: int(row["free_memory_mb"]), default=None)


def _clone_copy_on_write(source: Path, destination: Path) -> str:
    """Clone model extents without changing or duplicating the shared cache bytes."""

    ficlone = 0x40049409
    source_fd = os.open(source, os.O_RDONLY)
    destination_fd = -1
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            source.stat().st_mode & 0o777,
        )
        fcntl.ioctl(destination_fd, ficlone, source_fd)
        os.fsync(destination_fd)
    except Exception:
        if destination_fd >= 0:
            os.close(destination_fd)
            destination_fd = -1
        destination.unlink(missing_ok=True)
        raise
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
    return "linux_ficlone_copy_on_write"


def stage_unique_model_snapshot(
    selected: Mapping[str, Any], raw_dir: Path
) -> tuple[JsonDict, JsonDict]:
    """Return unique verified model bytes while leaving a shared cache unchanged."""

    model = deepcopy(dict(selected))
    source = Path(str(model.get("model_path") or "")).absolute()
    expected_hash = str(model.get("content_hash") or "")
    revision = str(model.get("revision") or "")
    if not source.is_file():
        raise ValueError("source_model_missing")
    source_stat = source.stat()
    source_hash = sha256_file(source)
    if not _HASH_RE.fullmatch(expected_hash) or source_hash != expected_hash:
        raise ValueError("source_model_hash_mismatch")
    model["source_cache_model_path"] = str(source)
    if source_stat.st_nlink == 1:
        receipt = {
            "required": False,
            "passed": True,
            "method": "existing_unique_cache_blob",
            "source_cache_model_path": str(source),
            "execution_model_path": str(source),
            "source_nlink": 1,
            "execution_nlink": 1,
            "source_size": source_stat.st_size,
            "execution_size": source_stat.st_size,
            "source_content_hash": source_hash,
            "execution_content_hash": source_hash,
        }
        model["model_staging_receipt"] = deepcopy(receipt)
        return model, receipt
    if not revision:
        raise ValueError("source_model_revision_missing")

    model_dir = raw_dir / "task_owned_model_cache" / f"models--{MODEL_ID.replace('/', '--')}"
    blob = model_dir / "blobs" / expected_hash.removeprefix("sha256:")
    alias = model_dir / "snapshots" / revision / source.name
    blob.parent.mkdir(parents=True, exist_ok=True)
    alias.parent.mkdir(parents=True, exist_ok=True)
    method = "existing_verified_task_owned_clone"
    if not blob.exists():
        temporary = blob.with_name(f".{blob.name}.{os.getpid()}.clone")
        method = _clone_copy_on_write(source, temporary)
        os.replace(temporary, blob)
    if not alias.exists():
        alias.symlink_to(Path("../../blobs") / blob.name)

    execution_stat = alias.stat()
    execution_hash = sha256_file(alias)
    if (
        not alias.is_symlink()
        or alias.resolve() != blob.resolve()
        or execution_stat.st_nlink != 1
        or execution_stat.st_size != source_stat.st_size
        or execution_hash != source_hash
    ):
        raise ValueError("task_owned_model_clone_verification_failed")
    receipt = {
        "required": True,
        "passed": True,
        "method": method,
        "source_cache_model_path": str(source),
        "execution_model_path": str(alias.absolute()),
        "source_nlink": source_stat.st_nlink,
        "execution_nlink": execution_stat.st_nlink,
        "source_size": source_stat.st_size,
        "execution_size": execution_stat.st_size,
        "source_content_hash": source_hash,
        "execution_content_hash": execution_hash,
    }
    model["model_path"] = str(alias.absolute())
    model["model_staging_receipt"] = deepcopy(receipt)
    return model, receipt


def project_tool_inductions(
    run_row: Mapping[str, Any] | None, completions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Project policy attempts without counting supervisor redirects as tool use."""

    diagnostics = run_row.get("policy_diagnostics", {}) if isinstance(run_row, Mapping) else {}
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    attempts = attempts if isinstance(attempts, list) else []
    environment_completions = [
        dict(row) for row in completions if row.get("stage") == "environment"
    ]
    rows: list[JsonDict] = []
    for index, attempt_value in enumerate(attempts):
        if not isinstance(attempt_value, Mapping):
            continue
        attempt = dict(attempt_value)
        gap = attempt.get("tool_gap")
        if not isinstance(gap, Mapping):
            continue
        tool_calls = int(gap.get("tool_calls_total", 0) or 0)
        terminated = str(gap.get("terminated_by") or "")
        terminal = bool(terminated)
        matching = [
            row
            for row in environment_completions
            if row.get("induction_attempt_index") in (None, index)
        ]
        identity = {
            "game": GAME,
            "seed": RANDOM_SEED,
            "attempt_index": index,
            "started_at": attempt.get("started_at"),
            "reason": attempt.get("reason"),
        }
        rows.append(
            {
                "induction_id": sha256_bytes(canonical_json(identity).encode()),
                "attempt_index": index,
                "reason": attempt.get("reason"),
                "started_at": attempt.get("started_at"),
                "elapsed_s": float(attempt.get("wall_s", 0.0) or 0.0),
                "selfparse": bool(gap.get("selfparse", True)),
                "tool_calls_total": tool_calls,
                "tool_calls_by_name": deepcopy(dict(gap.get("tool_calls_by_name", {}) or {})),
                "terminated_by": terminated,
                "terminal_result_returned": terminal,
                "engaged": bool(tool_calls > 0 and terminal),
                "tool_gap_events": deepcopy(list(gap.get("tool_gap_events", []) or [])),
                "tool_gap_events_dropped": int(gap.get("tool_gap_events_dropped", 0) or 0),
                "completion_ids": [row.get("completion_id") for row in matching],
                "planned": bool(attempt.get("planned")),
                "error": attempt.get("skipped") or None,
            }
        )
    return rows


def historical_induction_summary(payload: Mapping[str, Any], source_hash: str) -> JsonDict:
    """Count real old loops as context while keeping them outside the new denominator."""

    raw_rows = payload.get("per_game", [])
    raw_rows = raw_rows if isinstance(raw_rows, list) else []
    inductions: list[JsonDict] = []
    for row in raw_rows:
        if isinstance(row, Mapping) and row.get("game") == GAME:
            inductions.extend(project_tool_inductions(row, []))
    engaged = [row for row in inductions if row.get("engaged") is True]
    return {
        "source": HISTORICAL_RECEIPT_PATH.as_posix(),
        "source_hash": source_hash,
        "historical_real_tool_loop_inductions": len(engaged),
        "historical_tool_calls": sum(int(row["tool_calls_total"]) for row in engaged),
        "counts_toward_new_volume": False,
    }


def _session_run_row(session: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Load the full raw run row only when the session stores it out of line."""

    if not isinstance(session, Mapping):
        return None
    row = session.get("run_row")
    if isinstance(row, Mapping):
        return row
    path = session.get("run_row_path")
    try:
        loaded = json.loads(Path(str(path)).read_text(encoding="utf-8")) if path else None
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, Mapping) else None


def classify_inference_work(session: Mapping[str, Any] | None) -> tuple[str, str, str]:
    """Classify only computation supported by the terminal session receipt."""

    if not isinstance(session, Mapping):
        return "preflight_only_no_model_load", "blocked_no_run", "not_run"
    row = _session_run_row(session)
    completions = session.get("completions", [])
    completions = completions if isinstance(completions, list) else []
    gpu = session.get("gpu_receipts", {})
    live_mode = (
        "live_gpu"
        if isinstance(gpu, Mapping)
        and gpu.get("provenance_ok") is True
        and gpu.get("task_linked_cuda_execution") is True
        else "unverified_accelerator"
    )
    environment_generation = any(item.get("stage") == "environment" for item in completions)
    if row is not None and environment_generation:
        return "live_llm_inference", "model_full_generation", live_mode
    canary = session.get("canary", {})
    if isinstance(canary, Mapping) and canary.get("ok") is True:
        return "live_llm_inference", "model_bounded_generation", live_mode
    if session.get("model_loaded") is True:
        return "live_llm_model_load", "model_load_no_generation", live_mode
    return "preflight_only_no_model_load", "blocked_no_run", "not_run"


def model_identity_gate(session: Mapping[str, Any]) -> JsonDict | None:
    """Translate a typed identity rejection into one exact external gate row."""

    validation = session.get("model_identity_validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not False:
        return None
    identity = session.get("model_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    rows = identity.get("identity_obligation_rows", [])
    rows = rows if isinstance(rows, list) else []
    failure = next(
        (
            deepcopy(dict(row))
            for row in rows
            if isinstance(row, Mapping) and row.get("status") != "supported"
        ),
        {"status": "unknown", "obligation": "identity_validation"},
    )
    return gate_check(
        "typed_model_identity",
        str(identity.get("requested_model_path") or "live_server_props"),
        str(failure.get("obligation") or "identity_validation"),
        {"status": "supported"},
        failure,
        False,
        validation_errors=deepcopy(list(validation.get("errors", []))),
    )


def build_terminal_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    session: Mapping[str, Any] | None = None,
    historical: Mapping[str, Any] | None = None,
    isolation: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one terminal block, bounded null, or observed-engagement result."""

    all_checks = [deepcopy(dict(row)) for row in checks]
    summary = gate_summary(all_checks)
    blocked = summary["passed"] is not True or session is None
    run_row = _session_run_row(session)
    completions = (
        [deepcopy(dict(row)) for row in session.get("completions", [])]
        if isinstance(session, Mapping)
        else []
    )
    inductions = project_tool_inductions(run_row, completions)
    engaged = [row for row in inductions if row.get("engaged") is True]
    scheduled_complete = bool(isinstance(session, Mapping) and session.get("terminal_receipt"))
    if blocked:
        status = "blocked"
        verdict_class = "blocked"
        honest_verdict = "blocked_" + str(summary.get("failed_check") or "live_prerequisite")
    elif engaged:
        status = "complete"
        verdict_class = "positive"
        honest_verdict = "complete_positive_direct_tool_engagement_no_efficacy_claim"
    else:
        status = "complete"
        verdict_class = "null"
        honest_verdict = "complete_null_direct_tool_session_no_returned_engagement"
    substrate, substrate_class, inference_mode = classify_inference_work(session)
    if blocked and session is None:
        substrate, substrate_class, inference_mode = (
            "preflight_only_no_model_load",
            "blocked_no_run",
            "not_run",
        )
    row = None
    if not blocked:
        error = session.get("error") if isinstance(session, Mapping) else None
        row = {
            "unit_id": f"{GAME}:{RANDOM_SEED}:direct_selfparse",
            "game": GAME,
            "arm": "direct_selfparse",
            "seed": RANDOM_SEED,
            "metric": "real_returned_tool_loop_inductions",
            "metric_value": len(engaged),
            "error": error,
            "abstention": bool(error and run_row is None),
        }
    banked = int(run_row.get("levels", 0) or 0) if isinstance(run_row, Mapping) else 0
    actions = int(run_row.get("actions", 0) or 0) if isinstance(run_row, Mapping) else 0
    history = deepcopy(dict(historical or {}))
    historical_count = int(history.get("historical_real_tool_loop_inductions", 0) or 0)
    new_count = len(engaged)
    actual_models: list[JsonDict] = []
    if isinstance(session, Mapping):
        model_spec = session.get("model_spec") or session.get("model_identity")
        if isinstance(model_spec, Mapping):
            actual_models.append(deepcopy(dict(model_spec)))
    per_game = []
    if not blocked:
        per_game.append(
            {
                "game": GAME,
                "seed": RANDOM_SEED,
                "action_cap": ACTION_BUDGET,
                "session_timeout_s": SESSION_TIMEOUT_S,
                "induction_timeout_s": INDUCTION_TIMEOUT_S,
                "banked_levels": banked,
                "actions": actions,
                "elapsed_s": (
                    float(run_row.get("wall_s", 0.0) or 0.0)
                    if isinstance(run_row, Mapping)
                    else float(duration_s)
                ),
                "new_tool_loop_inductions": new_count,
                "tool_calls": sum(int(item["tool_calls_total"]) for item in inductions),
                "tool_engagement_observed": bool(engaged),
                "timed_out": bool(session.get("timed_out")) if session else False,
                "error": session.get("error") if session else None,
                "cost": {
                    "completion_count": len(completions),
                    "completion_tokens": sum(
                        int(item.get("completion_tokens", 0) or 0) for item in completions
                    ),
                },
            }
        )
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": status,
        "run_date": str(run_date),
        "preconditions_checked": all_checks,
        "inference_substrate": substrate,
        "inference_substrate_class": substrate_class,
        "inference_mode": inference_mode,
        "execution_venue": "host",
        "execution_host": os.uname().nodename,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [] if row is None else [row],
        "sample_size_budget": {
            "planned_sessions": 1,
            "completed_sessions": int(scheduled_complete),
            "independent_units": 1,
            "planned_action_cap": ACTION_BUDGET,
            "completed_actions": actions,
            "exclusions": [] if not blocked else [summary.get("failed_check")],
            "new_tool_loop_inductions": new_count,
            "historical_tool_loop_inductions": historical_count,
            "evidence_target_new_inductions": HISTORICAL_TARGET,
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "arc_tool_measurement_complete_score": int(scheduled_complete and not blocked),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": actual_models,
        "phase_spans": (
            deepcopy(list(session.get("phase_spans", []))) if isinstance(session, Mapping) else []
        ),
        "gpu_receipts": (
            deepcopy(dict(session.get("gpu_receipts", {}))) if isinstance(session, Mapping) else {}
        ),
        "runner_receipt": (
            deepcopy(dict(session.get("runner_receipt", {})))
            if isinstance(session, Mapping)
            else {}
        ),
        "per_game_results": per_game,
        "tool_induction_rows": inductions,
        "solve_provenance": "live_agent_self_discovery",
        "adapter_isolation_receipt": deepcopy(dict(isolation or {})),
        "arc_volume_sufficient_score": int(new_count >= HISTORICAL_TARGET),
        "arc_tool_engagement_score": int(bool(engaged)),
        "historical_receipt_summary": history,
        "completion_receipts": completions,
        "model_identity_receipt": (
            deepcopy(dict(session.get("model_identity", {})))
            if isinstance(session, Mapping)
            else {}
        ),
        "paired_efficacy_reported": False,
        "official_leaderboard_score_reported": False,
        "new_solve_claimed": False,
        "registered_level_increment": 0,
        "submitted": False,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate the closed terminal projection and all evidence-linked scores."""

    if isinstance(value, (str, Path)):
        try:
            artifact = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
    else:
        artifact = dict(value)
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing_fields:" + ",".join(sorted(missing)))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("model_specs_declaration_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_invalid")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance_invalid")
    if artifact.get("new_solve_claimed") is not False:
        errors.append("new_solve_claim_forbidden")
    if artifact.get("official_leaderboard_score_reported") is not False:
        errors.append("official_score_claim_forbidden")
    if artifact.get("paired_efficacy_reported") is not False:
        errors.append("paired_efficacy_claim_forbidden")
    if artifact.get("submitted") is not False:
        errors.append("submission_forbidden")
    if artifact.get("registered_level_increment") != 0:
        errors.append("registry_increment_forbidden")
    checks = artifact.get("preconditions_checked", [])
    if artifact.get("gate_check_summary") != gate_summary(
        checks if isinstance(checks, list) else []
    ):
        errors.append("gate_summary_mismatch")
    rows = artifact.get("tool_induction_rows", [])
    rows = rows if isinstance(rows, list) else []
    engagement = int(any(row.get("engaged") is True for row in rows if isinstance(row, Mapping)))
    if artifact.get("arc_tool_engagement_score") != engagement:
        errors.append("engagement_score_inconsistent")
    expected_volume = int(sum(row.get("engaged") is True for row in rows) >= HISTORICAL_TARGET)
    if artifact.get("arc_volume_sufficient_score") != expected_volume:
        errors.append("volume_score_inconsistent")
    expected_measurement = int(
        len(artifact.get("rows", [])) == 1
        and artifact.get("gate_check_summary", {}).get("passed") is True
    )
    if artifact.get("arc_tool_measurement_complete_score") != expected_measurement:
        errors.append("measurement_complete_score_inconsistent")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("status") != "blocked" or artifact.get("rows"):
            errors.append("blocked_terminal_inconsistent")
        if not artifact.get("model_specs") and artifact.get("inference_substrate_class") != (
            "blocked_no_run"
        ):
            errors.append("blocked_substrate_inconsistent")
    if artifact.get("verdict_class") == "positive" and not engagement:
        errors.append("positive_without_engagement")
    substrate_class = artifact.get("inference_substrate_class")
    floor = _DURATION_FLOORS.get(str(substrate_class))
    if floor is not None and float(artifact.get("duration_s", 0.0) or 0.0) < floor:
        errors.append(f"duration_floor_not_met:{substrate_class}")
    gpu = artifact.get("gpu_receipts", {})
    if artifact.get("inference_mode") == "live_gpu" and (
        not isinstance(gpu, Mapping)
        or gpu.get("provenance_ok") is not True
        or gpu.get("task_linked_cuda_execution") is not True
    ):
        errors.append("live_gpu_without_task_linked_cuda_receipt")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not _HASH_RE.fullmatch(checksum):
        errors.append("reproducibility_checksum_invalid")
    elif checksum != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _snapshot_sources(root: Path) -> tuple[JsonDict, JsonDict]:
    """Read every named source once before any measurement work starts."""

    sizes: JsonDict = {}
    hashes: JsonDict = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        try:
            size = path.stat().st_size if path.is_file() else 0
            sizes[relative.as_posix()] = size
            hashes[relative.as_posix()] = sha256_file(path) if size > 0 else "missing"
        except OSError:
            sizes[relative.as_posix()] = 0
            hashes[relative.as_posix()] = "missing"
    return sizes, hashes


def _task_contract(path: Path) -> JsonDict:
    """Read only Exp7193 identity and prior-failure fields from the active YAML."""

    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    tasks = (
        payload
        if isinstance(payload, list)
        else payload.get("tasks", [])
        if isinstance(payload, dict)
        else []
    )
    task = next((row for row in tasks if isinstance(row, dict) and row.get("id") == TASK_ID), {})
    prior = task.get("prior_failures", []) if isinstance(task, dict) else []
    return {
        "id": task.get("id"),
        "milestone": task.get("milestone"),
        "deliverable": task.get("deliverable"),
        "gated_on": task.get("gated_on"),
        "prior_failures": prior,
    }


def collect_static_preconditions(
    *, root: Path, result_path: Path, checkpoint_path: Path, raw_dir: Path
) -> tuple[list[JsonDict], JsonDict]:
    """Check contracts, quarantine, tools, imports, and storage before model work."""

    sizes, hashes = _snapshot_sources(root)
    try:
        spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        spec = ""
    contract = _task_contract(root / ROADMAP_PATH)
    expected_contract = {
        "id": TASK_ID,
        "milestone": MILESTONE,
        "deliverable": RESULT_PATH.as_posix(),
        "gated_on": None,
        "prior_failures": [
            {
                "experiment_id": "exp7186-arc-withheld-transfer",
                "verdict": EXPECTED_PRIOR_VERDICT,
                "addressed_by": (
                    "Use the existing arc_leaderboard_eval.run_game and direct selfparse path, "
                    "supported by the 2026-09-10 operator queue; no invented source prerequisite."
                ),
                "retire_if_same_verdict": True,
            }
        ],
    }
    try:
        prior = json.loads((root / PRIOR_ARTIFACT_PATH).read_text(encoding="utf-8"))
        prior = prior if isinstance(prior, dict) else {}
    except (OSError, json.JSONDecodeError):
        prior = {}
    trace = trace_shipped_path(root / EVAL_PATH, root / POLICY_PATH)
    tools = {
        "python": Path(sys.executable).is_file(),
        "nvidia-smi": shutil.which("nvidia-smi") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    storage = {
        "result_parent_writable": result_path.parent.is_dir()
        and os.access(result_path.parent, os.W_OK),
        "checkpoint_parent_writable": checkpoint_path.parent.is_dir()
        and os.access(checkpoint_path.parent, os.W_OK),
        "raw_dir_writable": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7193",
            True,
            "## REQ-ARC-WMTE-7193:" in spec,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            {path.as_posix(): "nonempty" for path in REQUIRED_SOURCE_PATHS},
            sizes,
            all(size > 0 for size in sizes.values()),
        ),
        gate_check(
            "required_source_hashes",
            "repository",
            "source_artifact_hashes",
            "sha256:<64 hex> for every required source",
            hashes,
            all(isinstance(value, str) and _HASH_RE.fullmatch(value) for value in hashes.values()),
        ),
        gate_check(
            "exact_v634_task_contract",
            ROADMAP_PATH.as_posix(),
            "id,milestone,deliverable,gated_on,prior_failures",
            expected_contract,
            contract,
        ),
        upstream_field_gate(
            prior,
            upstream=PRIOR_ARTIFACT_PATH.as_posix(),
            field="honest_verdict",
            expected=EXPECTED_PRIOR_VERDICT,
        ),
        gate_check(
            "shipped_eval_and_gap_path",
            "repository",
            "run_game,E3AgentPolicy,_induce_and_plan,tool_gap_events",
            True,
            trace,
            trace.get("passed") is True,
        ),
        gate_check(
            "required_tools",
            "host",
            "python,nvidia-smi,sha256sum",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_directories",
            "host_filesystem",
            "result,checkpoint,raw",
            {key: True for key in storage},
            storage,
        ),
    ]
    return checks, hashes


def _progress(phase: int, event: str, **fields: Any) -> None:
    """Emit one flushed machine-readable progress line at every phase boundary."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _nvidia_inventory() -> list[JsonDict]:  # pragma: no cover - requires the execution host.
    """Query capacity and compute ownership without changing any running process."""

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    return parse_gpu_inventory(gpu, apps)


def _llama_server_path() -> str | None:  # pragma: no cover - requires the execution host.
    """Resolve the native CUDA server without downloading or substituting a backend."""

    candidates = [
        os.environ.get("CARNOT_LLAMA_SERVER"),
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
        shutil.which("llama-server"),
    ]
    return next((str(path) for path in candidates if path and Path(path).is_file()), None)


def _free_port() -> int:  # pragma: no cover - the live driver owns the short reservation gap.
    """Ask the kernel for a free loopback port for this task-owned server."""

    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _load_historical(root: Path) -> tuple[JsonDict, JsonDict]:
    """Load optional prior calls only after checking their quarantine flag."""

    path = root / HISTORICAL_RECEIPT_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = payload if isinstance(payload, dict) else {}
        digest = sha256_file(path)
    except (OSError, json.JSONDecodeError):
        payload, digest = {}, "missing"
    quarantine = _quarantined(payload)
    check = gate_check(
        "historical_receipt_not_quarantined",
        HISTORICAL_RECEIPT_PATH.as_posix(),
        "flagged_adversarial",
        False,
        quarantine,
    )
    if quarantine:
        return check, {}
    return check, historical_induction_summary(payload, digest)


def _live_resource_preconditions(
    root: Path,
) -> tuple[
    list[JsonDict], JsonDict | None, JsonDict | None, str | None
]:  # pragma: no cover - requires the execution host.
    """Resolve cache and one idle CUDA card without acquiring or killing anything."""

    from carnot.agentic.arc_eval_provenance import huggingface_snapshot_revision
    from carnot.inference.sota_models import cached_current_model

    checks: list[JsonDict] = []
    _progress(2, "check_start", check="cached_current_model")
    selected = cached_current_model(preferred_quant=QUANTIZATION)
    _progress(2, "check_end", check="cached_current_model", available=selected is not None)
    checks.append(
        gate_check(
            "cached_current_model",
            "host_cache",
            "model_path",
            "present",
            "present" if selected else None,
        )
    )
    inventory: list[JsonDict] = []
    try:
        _progress(2, "subprocess_start", operation="nvidia_smi_read_only_conflict_check")
        inventory = _nvidia_inventory()
        _progress(
            2,
            "subprocess_end",
            operation="nvidia_smi_read_only_conflict_check",
            devices=len(inventory),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _progress(
            2, "subprocess_end", operation="nvidia_smi_read_only_conflict_check", error=repr(exc)
        )
    gpu = choose_gpu(inventory)
    checks.append(
        gate_check(
            "idle_cuda_gpu", "nvidia-smi", "conflict_free_gpu_with_21GiB", True, gpu is not None
        )
    )
    server = _llama_server_path()
    checks.append(
        gate_check(
            "native_llama_server", "host", "llama-server", "present", "present" if server else None
        )
    )
    if selected is not None:
        path = Path(str(selected["model_path"])).absolute()
        _progress(2, "benchmark_start", operation="model_content_sha256", path=str(path))
        content_hash = sha256_file(path)
        _progress(2, "benchmark_end", operation="model_content_sha256", hash=content_hash)
        revision = huggingface_snapshot_revision(str(path), MODEL_ID)
        selected = {
            **selected,
            "quantization": QUANTIZATION,
            "model_path": str(path),
            "revision": revision,
            "content_hash": content_hash,
        }
        checks.append(
            gate_check(
                "model_snapshot_revision",
                "huggingface_cache",
                "revision",
                "nonempty",
                "nonempty" if revision else None,
            )
        )
    return checks, selected, gpu, server


def gpu_process_ownership(pid: int, *, process_group: int, parent_pid: int) -> JsonDict:
    """Separate all task-owned CUDA contexts from the generator group to unload."""

    if pid == parent_pid:
        return {"owned_by_task": True, "owned_generator_process": False}
    try:
        generator = os.getpgid(pid) == process_group
    except OSError:
        generator = False
    return {"owned_by_task": generator, "owned_generator_process": generator}


def _task_gpu_sample(
    gpu_index: int, process_group: int
) -> JsonDict:  # pragma: no cover - requires the execution host.
    """Mark parent and generator contexts while keeping their roles distinct."""

    inventory = _nvidia_inventory()
    device = next((row for row in inventory if row["index"] == gpu_index), {})
    rows = []
    for app in device.get("compute_apps", []):
        pid = int(app["pid"])
        ownership = gpu_process_ownership(pid, process_group=process_group, parent_pid=os.getpid())
        rows.append({**app, **ownership})
    return {
        "monotonic_s": time.monotonic(),
        "gpu_index": gpu_index,
        "gpu_uuid": device.get("uuid"),
        "gpu_model": device.get("name"),
        "free_memory_mb": device.get("free_memory_mb"),
        "compute_apps": rows,
    }


def _read_json(path: Path) -> JsonDict:
    """Read one progress file without turning a concurrent replace into an error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_live_session(
    *,
    root: Path,
    model: Mapping[str, Any],
    gpu: Mapping[str, Any],
    server: str,
    raw_dir: Path,
    checkpoint_path: Path,
) -> tuple[JsonDict | None, list[JsonDict]]:  # pragma: no cover - requires live CUDA.
    """Hold the GPU lease while one externally heartbeated child runs the session."""

    from carnot.gpu_lease_phase_journal import GpuLease, LeaseError
    from carnot.inference.sota_models import gguf_tokenizer_loadable

    checks: list[JsonDict] = []
    runtime_dir = checkpoint_path.parent / "gpu_lease"
    used_before = int(gpu["total_memory_mb"]) - int(gpu["free_memory_mb"])
    _progress(3, "check_start", check="task_owned_gpu_lease")
    try:
        lease = GpuLease.acquire(
            runtime_dir=runtime_dir,
            task_id=TASK_ID,
            device_uuid=str(gpu["uuid"]),
            expected_model=str(model["content_hash"]),
            vram_before_mb=used_before,
            ttl_s=180,
        )
    except LeaseError as exc:
        _progress(3, "check_end", check="task_owned_gpu_lease", error=repr(exc))
        checks.append(
            gate_check(
                "task_owned_gpu_lease", str(runtime_dir), "lease", "acquired", type(exc).__name__
            )
        )
        return None, checks
    checks.append(
        gate_check("task_owned_gpu_lease", str(runtime_dir), "lease", "acquired", "acquired")
    )
    _progress(3, "check_end", check="task_owned_gpu_lease", lease_id=lease.lease_id)
    lease.transition("admitted")
    _progress(3, "benchmark_start", operation="stage_unique_model_snapshot")
    try:
        staged_model, staging_receipt = stage_unique_model_snapshot(model, raw_dir)
    except (OSError, ValueError) as exc:
        _progress(
            3,
            "benchmark_end",
            operation="stage_unique_model_snapshot",
            passed=False,
            error=repr(exc),
        )
        checks.append(
            gate_check(
                "task_owned_unique_model_bytes",
                str(model.get("model_path")),
                "copy_on_write_clone_size_hash_nlink",
                True,
                repr(exc),
                False,
            )
        )
        lease.transition("terminal_blocked")
        lease.release()
        return None, checks
    model = staged_model
    _progress(
        3,
        "benchmark_end",
        operation="stage_unique_model_snapshot",
        passed=True,
        execution_model_path=model["model_path"],
    )
    checks.append(
        gate_check(
            "task_owned_unique_model_bytes",
            str(staging_receipt["source_cache_model_path"]),
            "copy_on_write_clone_size_hash_nlink",
            True,
            staging_receipt.get("passed") is True,
            staging_receipt=staging_receipt,
        )
    )
    _progress(3, "model_load_start", operation="embedded_gguf_tokenizer")
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(model["model_path"]))
    _progress(
        3,
        "model_load_end",
        operation="embedded_gguf_tokenizer",
        ok=tokenizer_ok,
        detail=tokenizer_detail,
    )
    checks.append(
        gate_check(
            "embedded_gguf_tokenizer", str(model["model_path"]), "tokenizer", True, tokenizer_ok
        )
    )
    if not tokenizer_ok:
        lease.transition("terminal_blocked")
        lease.release()
        return None, checks
    lease.transition("loading")
    port = _free_port()
    session_path = raw_dir / "session_receipt.json"
    progress_path = checkpoint_path.parent / "session_progress.json"
    atomic_write(
        progress_path,
        {
            "schema": "carnot.experiment_7193.session_progress.v1",
            "model_loaded": False,
            "stage": "launch_pending",
            "terminal": False,
        },
    )
    atomic_write(
        session_path,
        {
            "schema": "carnot.experiment_7193.session_receipt.v1",
            "status": "running",
            "terminal_receipt": False,
        },
    )
    command = [
        sys.executable,
        "-u",
        str(root / WRAPPER_PATH),
        "--role",
        "session",
        "--date",
        RUN_DATE,
        "--model-path",
        str(model["model_path"]),
        "--gpu-index",
        str(gpu["index"]),
        "--port",
        str(port),
        "--raw-dir",
        str(raw_dir),
        "--checkpoint-path",
        str(progress_path),
        "--session-output",
        str(session_path),
    ]
    env = session_environment(
        os.environ,
        model_path=str(model["model_path"]),
        gpu_index=int(gpu["index"]),
        port=port,
        raw_dir=raw_dir,
    )
    env["CARNOT_LLAMA_SERVER"] = server
    _progress(4, "external_heartbeat_start", interval_s=45, timeout_s=SESSION_TIMEOUT_S)
    _progress(4, "subprocess_start", operation="live_arc_session", command=command)
    process = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
    group = process.pid
    samples: list[JsonDict] = []
    signals: list[str] = []
    start = time.monotonic()
    next_heartbeat = start
    resident_recorded = False
    while process.poll() is None and time.monotonic() - start < SESSION_TIMEOUT_S:
        now = time.monotonic()
        if now >= next_heartbeat:
            lease.heartbeat()
            sample = _task_gpu_sample(int(gpu["index"]), group)
            samples.append(sample)
            progress = _read_json(progress_path)
            owned_vram = sum(
                int(row.get("used_memory_mb", 0))
                for row in sample.get("compute_apps", [])
                if row.get("owned_by_task") is True
            )
            if progress.get("model_loaded") and not resident_recorded:
                lease.transition("resident", vram_mb=owned_vram)
                lease.transition("inferencing")
                resident_recorded = True
            atomic_write(
                checkpoint_path,
                {
                    "schema": "carnot.experiment_7193.checkpoint.v1",
                    "task_id": TASK_ID,
                    "elapsed_s": round(now - start, 3),
                    "child_pid": process.pid,
                    "lease_id": lease.lease_id,
                    "session_progress": progress,
                    "latest_gpu_sample": sample,
                    "terminal": False,
                },
            )
            _progress(4, "heartbeat", elapsed_s=round(now - start, 1), progress=progress)
            next_heartbeat = now + 45
        time.sleep(2)
    timed_out = process.poll() is None
    if timed_out:
        os.killpg(group, signal.SIGTERM)
        signals.append("SIGTERM:owned_process_group")
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(group, signal.SIGKILL)
            signals.append("SIGKILL:owned_process_group")
            process.wait(timeout=10)
    _progress(
        4,
        "subprocess_end",
        operation="live_arc_session",
        returncode=process.returncode,
        timed_out=timed_out,
    )
    final_progress = _read_json(progress_path)
    if final_progress.get("model_loaded") and not resident_recorded:
        sample = _task_gpu_sample(int(gpu["index"]), group)
        samples.append(sample)
        owned_vram = sum(
            int(row.get("used_memory_mb", 0))
            for row in sample.get("compute_apps", [])
            if row.get("owned_by_task") is True
        )
        lease.transition("resident", vram_mb=owned_vram)
        lease.transition("inferencing")
        resident_recorded = True
    session = _read_json(session_path)
    if not session:
        session = {
            "status": "timed_out" if timed_out else "child_failed",
            "terminal_receipt": True,
            "timed_out": timed_out,
            "model_loaded": bool(final_progress.get("model_loaded")),
            "canary": deepcopy(final_progress.get("canary", {})),
            "completions": [],
            "error": "session_deadline_3600s"
            if timed_out
            else f"session_child_exit_{process.returncode}",
            "phase_spans": [],
        }
    session["model_spec"] = deepcopy(dict(model))
    session["timed_out"] = timed_out or bool(session.get("timed_out"))
    if timed_out:
        session["error"] = "session_deadline_3600s"
    identity_check = model_identity_gate(session)
    if identity_check is not None:
        checks.append(identity_check)
    if resident_recorded:
        lease.transition("unloading")
        deadline = time.monotonic() + 30
        unloaded = False
        after_sample: JsonDict = {}
        while time.monotonic() < deadline:
            after_sample = _task_gpu_sample(int(gpu["index"]), group)
            samples.append(after_sample)
            if not any(
                row.get("owned_generator_process") for row in after_sample.get("compute_apps", [])
            ):
                unloaded = True
                break
            time.sleep(1)
        after_used = int(gpu["total_memory_mb"]) - int(
            after_sample.get("free_memory_mb") or gpu["free_memory_mb"]
        )
        lease.transition(
            "validating",
            vram_mb=after_used,
            exit_code=int(process.returncode or 0),
            unload_observed=unloaded,
        )
        lease.transition("terminal_complete" if unloaded else "terminal_blocked")
    else:
        lease.transition("terminal_blocked")
        unloaded = True
    release = lease.release()
    owned_samples = [
        row
        for sample in samples
        for row in sample.get("compute_apps", [])
        if row.get("owned_by_task") is True and int(row.get("used_memory_mb", 0)) > 0
    ]
    unowned_samples = [
        row
        for sample in samples
        for row in sample.get("compute_apps", [])
        if row.get("owned_by_task") is False
    ]
    session["gpu_receipts"] = {
        "provenance_ok": bool(owned_samples and not unowned_samples),
        "task_linked_cuda_execution": bool(owned_samples),
        "lease_owner": lease.owner_receipt(),
        "lease_release": release,
        "samples": samples,
        "signals_sent": signals,
        "unload_observed": unloaded,
    }
    checks.append(
        gate_check(
            "task_linked_cuda_execution",
            str(gpu.get("uuid")),
            "owned_gpu_process_sample",
            True,
            bool(owned_samples and not unowned_samples),
        )
    )
    runner = dict(session.get("runner_receipt", {}) or {})
    runner.update(
        {
            "lease_id": lease.lease_id,
            "task_owned": True,
            "fit_receipt": deepcopy(dict(session.get("fit_receipt", {}) or {})),
        }
    )
    session["runner_receipt"] = runner
    atomic_write(session_path, session)
    return session, checks


def _relative(path: Path) -> str:
    """Use stable repository-relative paths when a raw file is in this checkout."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_session_child(args: argparse.Namespace) -> int:  # pragma: no cover - requires live CUDA.
    """Load one owned model and drive ``run_game`` in the isolated child process."""

    os.environ.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)
    random.seed(RANDOM_SEED)
    import numpy as np

    np.random.seed(RANDOM_SEED)
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    from arc_leaderboard_eval import ProgressWriter, run_game
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_eval_provenance import (
        build_typed_arc_model_identity_receipt,
        capture_arc_model_identity_source_provenance,
        huggingface_snapshot_revision,
        validate_typed_arc_model_identity_receipt,
    )
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    completion_dir = raw_dir / "completions"
    completion_dir.mkdir(parents=True, exist_ok=True)
    session_path = Path(args.session_output)
    progress_path = Path(args.checkpoint_path)
    started = time.monotonic()
    spans: list[JsonDict] = []
    completions: list[JsonDict] = []
    session: JsonDict = {
        "status": "child_failed",
        "terminal_receipt": True,
        "timed_out": False,
        "model_loaded": False,
        "canary": {"ok": False},
        "completions": completions,
        "phase_spans": spans,
        "error": None,
    }
    proposer: Any = None
    server_process: Any = None
    policy_holder: dict[str, Any] = {"policy": None, "stage": "model_load"}

    def checkpoint(**fields: Any) -> None:
        atomic_write(
            progress_path,
            {
                "schema": "carnot.experiment_7193.session_progress.v1",
                "pid": os.getpid(),
                "elapsed_s": round(time.monotonic() - started, 3),
                **fields,
            },
        )

    try:
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=str(Path(args.model_path).absolute()),
            port=int(args.port),
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            tries=1,
        )
        original_record = proposer._record_completion_diagnostics

        def record_completion(response: Mapping[str, Any]) -> None:
            original_record(dict(response))
            index = len(completions)
            content = str(response.get("content") or "")
            content_path = completion_dir / f"completion_{index:03d}.txt"
            atomic_write_bytes(content_path, content.encode("utf-8", "replace"))
            policy = policy_holder["policy"]
            attempts = getattr(policy, "induction_attempts", []) if policy is not None else []
            attempt_index = len(attempts) - 1 if attempts else None
            timings = (
                response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
            )
            receipt = {
                "completion_id": f"completion-{index}",
                "index": index,
                "stage": policy_holder["stage"],
                "induction_attempt_index": attempt_index,
                "content_path": _relative(content_path),
                "content_sha256": sha256_bytes(content.encode("utf-8", "replace")),
                "content_chars": len(content),
                "prompt_tokens": timings.get("prompt_n"),
                "completion_tokens": timings.get("predicted_n"),
                "stop_type": response.get("stop_type"),
                "truncated": bool(response.get("truncated")),
                "error": None,
            }
            completions.append(receipt)
            atomic_write(raw_dir / "completion_manifest.json", {"completions": completions})

        proposer._record_completion_diagnostics = record_completion
        load_start = time.monotonic()
        _progress(5, "model_load_start", model=MODEL_ID, port=int(args.port))
        loaded = proposer._ensure_server()
        load_end = time.monotonic()
        spans.append(
            {
                "phase": "model_load",
                "start_s": load_start - started,
                "end_s": load_end - started,
                "duration_s": load_end - load_start,
            }
        )
        server_process = proposer._proc
        owned_server = bool(loaded and server_process is not None and proposer.last_launch_argv)
        _progress(
            5,
            "model_load_end",
            loaded=loaded,
            owned_server=owned_server,
            elapsed_s=round(load_end - load_start, 3),
        )
        if not owned_server:
            raise RuntimeError("task_owned_server_not_launched_or_unowned_reuse_detected")
        props = proposer.server_props()
        observed_n_ctx = proposer.observed_n_ctx()
        slots = proposer.observed_total_slots()
        observed_model_path = proposer.observed_model_path()
        session["model_loaded"] = True
        checkpoint(model_loaded=True, stage="resident", server_pid=server_process.pid)
        if observed_n_ctx != N_CTX or slots != 1:
            raise RuntimeError(f"server_shape_mismatch:n_ctx={observed_n_ctx}:slots={slots}")
        revision = huggingface_snapshot_revision(str(Path(args.model_path).absolute()), MODEL_ID)
        content_hash = sha256_file(args.model_path)
        source = capture_arc_model_identity_source_provenance(
            raw_server_props=props,
            requested_model_path=str(Path(args.model_path).absolute()),
            source_kind="live_server_props",
        )
        identity = build_typed_arc_model_identity_receipt(
            selected_model_spec={
                "model_path": str(Path(args.model_path).absolute()),
                "model_filename": Path(args.model_path).name,
                "hf_id": MODEL_ID,
                "revision": revision,
                "model_file_hash": content_hash,
            },
            launch_model_argument=str(Path(args.model_path).absolute()),
            raw_server_props=props,
            source_provenance=source,
        )
        identity_validation = validate_typed_arc_model_identity_receipt(identity)
        session["model_identity"] = identity
        session["model_identity_validation"] = {
            "valid": identity_validation.valid,
            "errors": list(identity_validation.errors),
        }
        session["model_spec"] = {
            "name": "Qwen3.8-27B",
            "hf_id": MODEL_ID,
            "quantization": QUANTIZATION,
            "gpu": int(args.gpu_index),
            "model_path": str(Path(args.model_path).absolute()),
            "revision": revision,
            "content_hash": content_hash,
        }
        session["runner_receipt"] = {
            "model_count": 1,
            "replica_count": 1,
            "runner": "LocalGGUFProposer_llama.cpp",
            "dual_gpu_runner_used": False,
            "task_owned": True,
            "server_pid": server_process.pid,
            "server_command": list(proposer.last_launch_argv),
            "embedded_chat_template": True,
            "actual_context_slots": slots,
            "observed_n_ctx": observed_n_ctx,
            "completion_budget": COMPLETION_BUDGET,
            "induction_timeout_s": INDUCTION_TIMEOUT_S,
            "kv_cache_type": "q8_0",
            "generator_selection_log": list(proposer.generator_selection_log),
            "raw_server_props": props,
        }
        session["fit_receipt"] = {
            "observed_n_ctx": observed_n_ctx,
            "actual_context_slots": slots,
            "configured_completion_budget": COMPLETION_BUDGET,
            "prompt_tokens": [],
            "kv_cache_type": "q8_0",
            "server_command": list(proposer.last_launch_argv),
        }
        if not identity_validation.valid:
            raise RuntimeError("model_identity_invalid:" + ";".join(identity_validation.errors))
        policy_holder["stage"] = "canary"
        canary_start = time.monotonic()
        _progress(5, "generation_start", operation="eight_token_canary")
        canary_ok, canary_text = proposer.complete_text(
            "Reply with the single word READY.", max_tokens=8, temperature=0.1
        )
        canary_end = time.monotonic()
        _progress(
            5,
            "generation_end",
            operation="eight_token_canary",
            ok=canary_ok,
            elapsed_s=round(canary_end - canary_start, 3),
        )
        spans.append(
            {
                "phase": "canary",
                "start_s": canary_start - started,
                "end_s": canary_end - started,
                "duration_s": canary_end - canary_start,
            }
        )
        session["canary"] = {
            "ok": bool(canary_ok),
            "completion_id": completions[-1]["completion_id"]
            if canary_ok and completions
            else None,
            "content_sha256": sha256_bytes(canary_text.encode()),
            "elapsed_s": canary_end - canary_start,
            "error": None if canary_ok else canary_text,
        }
        checkpoint(model_loaded=True, stage="canary_complete", canary=session["canary"])
        if not canary_ok:
            session["status"] = "complete"
            session["error"] = "canary_generation_failed"
        else:
            policy_holder["stage"] = "environment"
            policy = E3AgentPolicy(GAME, proposer=proposer)
            policy_holder["policy"] = policy
            progress = ProgressWriter(
                progress_path.parent / "arc_run_game.json",
                game=GAME,
                game_index=0,
                games_planned=1,
                policy=policy,
            )
            run_start = time.monotonic()
            _progress(
                6,
                "benchmark_start",
                operation="arc_leaderboard_eval.run_game",
                game=GAME,
                action_cap=ACTION_BUDGET,
            )
            _progress(6, "generation_start", operation="direct_selfparse_environment_session")
            run_row = run_game(GAME, policy, budget=ACTION_BUDGET, progress=progress)
            run_end = time.monotonic()
            _progress(
                6,
                "generation_end",
                operation="direct_selfparse_environment_session",
                elapsed_s=round(run_end - run_start, 3),
            )
            _progress(
                6,
                "benchmark_end",
                operation="arc_leaderboard_eval.run_game",
                actions=run_row.get("actions"),
                levels=run_row.get("levels"),
            )
            spans.append(
                {
                    "phase": "environment_session",
                    "start_s": run_start - started,
                    "end_s": run_end - started,
                    "duration_s": run_end - run_start,
                }
            )
            raw_row_path = raw_dir / "run_game_row.json"
            atomic_write(raw_row_path, run_row)
            session["run_row_path"] = _relative(raw_row_path)
            session["run_row_sha256"] = sha256_file(raw_row_path)
            session["run_row_summary"] = {
                key: run_row.get(key)
                for key in (
                    "game",
                    "levels",
                    "reached",
                    "actions",
                    "wall_s",
                    "llm_reached",
                    "actions_to_first_levelup",
                )
            }
            session["status"] = "complete"
            session["error"] = None
        session["fit_receipt"] = {
            "observed_n_ctx": observed_n_ctx,
            "actual_context_slots": slots,
            "configured_completion_budget": COMPLETION_BUDGET,
            "prompt_tokens": [row.get("prompt_tokens") for row in completions],
            "kv_cache_type": "q8_0",
            "server_command": list(proposer.last_launch_argv),
        }
    except Exception as exc:
        session["status"] = "complete"
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        _progress(6, "session_error", error=session["error"])
    finally:
        if proposer is not None:
            proc = proposer._proc
            _progress(7, "model_unload_start", server_pid=getattr(proc, "pid", None))
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            proposer._proc = None
            _progress(7, "model_unload_end", returncode=getattr(proc, "returncode", None))
            session["model_cleanup"] = {
                "server_pid": getattr(proc, "pid", None),
                "exit_code": getattr(proc, "returncode", None),
                "owned_process_only": True,
            }
        session["duration_s"] = time.monotonic() - started
        session["completions"] = completions
        session["phase_spans"] = spans
        atomic_write(session_path, session)
        checkpoint(
            model_loaded=session.get("model_loaded", False),
            stage="terminal",
            canary=session.get("canary"),
            terminal=True,
        )
    return 0


def run_experiment(
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
    live_runner: Callable[..., Any] | None = None,
) -> JsonDict:
    """Run preflight, one optional live session, validation, and one terminal write."""

    started = time.monotonic()
    _progress(0, "phase_start", name="checkpoint_before_checks")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    isolation = adapter_isolation_receipt()
    shell = build_terminal_artifact(
        run_date=run_date,
        duration_s=0.0,
        checks=[gate_check("preflight_started", TASK_ID, "terminal", True, False)],
        source_hashes={},
        isolation=isolation,
    )
    atomic_write(checkpoint_path, shell)
    _progress(0, "phase_end", name="checkpoint_before_checks")

    _progress(1, "phase_start", name="static_preconditions")
    _progress(1, "benchmark_start", operation="static_preconditions")
    checks, source_hashes = collect_static_preconditions(
        root=root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    history_check, historical = _load_historical(root)
    checks.append(history_check)
    static_ready = all(row.get("passed") is True for row in checks)
    _progress(1, "benchmark_end", operation="static_preconditions", passed=static_ready)
    _progress(1, "phase_end", name="static_preconditions", passed=static_ready)

    session: JsonDict | None = None
    if static_ready and live_runner is not None:
        session = live_runner(
            root=root,
            model={},
            gpu={},
            server="",
            raw_dir=raw_dir,
            checkpoint_path=checkpoint_path,
        )
    elif static_ready:  # pragma: no cover - requires live CUDA.
        _progress(2, "phase_start", name="cache_cuda_and_conflict_checks")
        live_checks, model, gpu, server = _live_resource_preconditions(root)
        checks.extend(live_checks)
        live_ready = all(row.get("passed") is True for row in live_checks)
        _progress(2, "phase_end", name="cache_cuda_and_conflict_checks", passed=live_ready)
        if live_ready and model is not None and gpu is not None and server is not None:
            session, runtime_checks = _run_live_session(
                root=root,
                model=model,
                gpu=gpu,
                server=server,
                raw_dir=raw_dir,
                checkpoint_path=checkpoint_path,
            )
            checks.extend(runtime_checks)
    artifact = build_terminal_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        checks=checks,
        source_hashes=source_hashes,
        session=session,
        historical=historical,
        isolation=isolation,
    )
    _progress(8, "phase_start", name="terminal_validation_and_atomic_write")
    _progress(8, "validation_start", operation="cold_artifact_validation")
    errors = validate_artifact(artifact)
    _progress(8, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(8, "artifact_write_start", path=str(result_path), status=artifact["status"])
    atomic_write(result_path, artifact)
    _progress(8, "artifact_write_end", path=str(result_path), status=artifact["status"])
    _progress(8, "phase_end", name="terminal_validation_and_atomic_write")
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse driver, isolated child, and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("driver", "session"), default="driver")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--session-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one bounded measurement, its child, or a cold artifact check."""

    _progress(0, "phase_start", name="entrypoint_before_checks")
    args = parse_args(argv)
    if args.validate is not None:
        _progress(8, "validation_start", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(8, "validation_end", path=str(args.validate), errors=errors)
        return int(bool(errors))
    if args.role == "session":  # pragma: no cover - requires live CUDA.
        return run_session_child(args)
    root = Path(os.environ.get("CARNOT_REPO_ROOT", REPO_ROOT)).resolve()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    checkpoint_path = (
        args.checkpoint_path if args.checkpoint_path.is_absolute() else root / args.checkpoint_path
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    artifact = run_experiment(
        root=root,
        run_date=str(args.date),
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "status": artifact["status"],
                "verdict": artifact["honest_verdict"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
