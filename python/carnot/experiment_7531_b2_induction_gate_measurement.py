"""Run B2 induction-timing telemetry on the frozen E6 panel.

The live child reuses Experiment 7491's qualified E3 path and composes its
exclusive timer with REQ-ARC-WMTE-7530 telemetry. No gate changes behavior.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import importlib.util
import os
from pathlib import Path
import time
from types import ModuleType
from typing import Any

from carnot import experiment_7471_v654_arc_seam_observation as exp7471
from carnot import experiment_7491_e6_timed_live_profile as e6
from carnot.agentic.arc_decision_telemetry import (
    TELEMETRY_ENV_FLAG,
    TELEMETRY_PATH_ENV,
    load_telemetry,
)
from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger


def _load_evaluator() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/experiments/semif_arc_readout_eval.py"
    spec = importlib.util.spec_from_file_location("semif_arc_readout_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load B2 evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator()


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
EXPERIMENT_ID = 7531
EXPERIMENT_NAME = "exp7531-b2-induction-gate-measurement"
TASK_ID = "experiment_7531_b2_induction_gate_measurement"
SCHEMA = "carnot.arc.b2_induction_gate_measurement.v1"
TOTAL_LIVE_LIMIT_S = 3 * 60 * 60.0

PANEL_GAMES = e6.PANEL_GAMES
EPISODE_SEEDS = (
    *e6.EPISODE_SEEDS,
    7_531_001,
    7_531_002,
    7_531_003,
    7_531_004,
    7_531_005,
    7_531_006,
    7_531_007,
    7_531_008,
    7_531_009,
)

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
FROZEN_PANEL_PATH = e6.FROZEN_PANEL_PATH
STAGE1_PATH = Path("results/experiment_7530_b2_induction_gate_telemetry.json")
RESULT_PATH = Path("results/experiment_7531_b2_induction_gate_measurement.json")
RAW_DIR = Path("results/raw/experiment_7531_b2_induction_gate_measurement")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
TELEMETRY_PATH = RAW_DIR / "induction_gate_telemetry.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7531_b2_induction_gate_measurement.json")
MODULE_PATH = Path("python/carnot/experiment_7531_b2_induction_gate_measurement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7531_b2_induction_gate_measurement.py")
TEST_PATH = Path("tests/python/test_experiment_7531_b2_induction_gate_measurement.py")
E6_ARTIFACTS = (
    Path("results/experiment_7490_e6_live_loop_cost_profile.json"),
    Path("results/experiment_7491_e6_timed_live_profile.json"),
    Path("results/experiment_7492_e6_timed_cost_profile.json"),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, step: str, event: str, **details: Any) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7531] step={step} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def build_schedule(frozen: Mapping[str, Any]) -> list[JsonDict]:
    """Repeat only the frozen E6 games under fixed B2 seeds."""

    games = tuple(str(row.get("game")) for row in frozen.get("selected_games", []))
    if games != PANEL_GAMES:
        raise ValueError(f"frozen E6 panel mismatch: {games}")
    rows: list[JsonDict] = []
    for seed in EPISODE_SEEDS:
        for game in PANEL_GAMES:
            rows.append(
                {
                    "episode_id": f"{game}:seed-{seed}",
                    "game": game,
                    "seed": seed,
                    "execution_order": len(rows),
                    "action_limit": e6.ACTION_LIMIT,
                    "episode_limit_s": e6.EPISODE_LIMIT_S,
                    "request_limit": e6.REQUEST_LIMIT,
                    "max_new_tokens_per_call": e6.MAX_NEW_TOKENS,
                    "adapter_disabled": True,
                    "game_source_read": False,
                    "stored_engines_disabled": True,
                    "banked_trajectories_disabled": True,
                }
            )
    return rows


def _cite(path: Path, role: str) -> JsonDict:
    return {
        "path": path.as_posix(),
        "sha256": e6.sha256_file(REPO_ROOT / path),
        "bytes": (REPO_ROOT / path).stat().st_size,
        "role": role,
    }


def static_preconditions() -> tuple[list[JsonDict], list[JsonDict], Path | None]:
    """Check non-CUDA inputs before the corrected nvidia-smi admission."""

    paths = (SPEC_PATH, FROZEN_PANEL_PATH, STAGE1_PATH, MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    paths = (*paths, *E6_ARTIFACTS)
    checks = [
        e6._check(
            "cuda_visible_devices_already_gpu_1",
            "1",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            passed=os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
            path="process_environment",
        )
    ]
    cited: list[JsonDict] = []
    for path in paths:
        present = (REPO_ROOT / path).is_file() and (REPO_ROOT / path).stat().st_size > 0
        checks.append(
            e6._check(
                f"source:{path}",
                "readable_nonempty",
                "readable_nonempty" if present else None,
                passed=present,
                path=path.as_posix(),
            )
        )
        if present:
            cited.append(_cite(path, "e6_evidence" if path in E6_ARTIFACTS else "protocol"))
    frozen = e6.load_json(REPO_ROOT / FROZEN_PANEL_PATH)
    try:
        schedule = build_schedule(frozen)
    except ValueError:
        schedule = []
    checks.append(
        e6._check(
            "frozen_e6_panel_identity",
            list(PANEL_GAMES),
            sorted({row.get("game") for row in schedule}),
            passed=len(schedule) == len(PANEL_GAMES) * len(EPISODE_SEEDS),
            path=FROZEN_PANEL_PATH.as_posix(),
        )
    )
    environment_dir = e6.resolve_environment_dir(REPO_ROOT)
    available = (
        {path.name for path in environment_dir.iterdir() if path.is_dir()}
        if environment_dir is not None
        else set()
    )
    missing = sorted(set(PANEL_GAMES) - available)
    checks.append(
        e6._check(
            "public_environment_panel_available",
            [],
            missing,
            passed=not missing,
            path="environment_files_names_only",
        )
    )
    return checks, cited, environment_dir


def configure_e6_driver() -> None:
    """Point the qualified E6 process boundary at B2-owned paths."""

    values = {
        "RUN_DATE": RUN_DATE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "TASK_ID": TASK_ID,
        "SCHEMA": SCHEMA,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "SCHEDULE_PATH": SCHEDULE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "RUNTIME_EVENT_PATH": RUNTIME_EVENT_PATH,
        "ACTION_PATH": ACTION_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "EPISODE_SEEDS": EPISODE_SEEDS,
        "PANEL_GAMES": PANEL_GAMES,
        "TOTAL_LIVE_LIMIT_S": TOTAL_LIVE_LIMIT_S,
    }
    for name, value in values.items():
        setattr(e6, name, value)
    e6._configure_live_driver()


def _episode_summaries(session: Mapping[str, Any]) -> list[JsonDict]:
    keys = (
        "episode_id",
        "game",
        "seed",
        "execution_order",
        "disposition",
        "action_count",
        "start_level",
        "peak_level",
        "terminal_level",
        "elapsed_s",
        "solve_provenance",
        "recorder_error_count",
        "error",
    )
    return [
        {key: row.get(key) for key in keys}
        for row in session.get("episodes", [])
        if isinstance(row, Mapping)
    ]


def blocked_artifact(
    *,
    failed_check: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    cited: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build one terminal blocker without claiming a measurement."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_date": RUN_DATE,
        "status": "blocked_precondition",
        "honest_verdict": f"blocked_{failed_check.get('check', 'precondition')}",
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "model_invoked": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "random_seed": {"episodes": list(EPISODE_SEEDS), "ordering": 7_531},
        "duration_s": max(0.000001, float(duration_s)),
        "preconditions_checked": deepcopy(list(checks)),
        "failed_precondition": deepcopy(dict(failed_check)),
        "protocol_schedule": deepcopy(list(schedule)),
        "episode_rows": [
            {
                "episode_id": row.get("episode_id"),
                "game": row.get("game"),
                "seed": row.get("seed"),
                "disposition": "unstarted",
            }
            for row in schedule
        ],
        "gate_opportunity_count": 0,
        "induction_attempt_count": 0,
        "sample_floor": {
            "minimum_gate_opportunities": evaluator.MIN_GATE_OPPORTUNITIES,
            "minimum_induction_attempts": evaluator.MIN_INDUCTION_ATTEMPTS,
            "met": False,
        },
        "publication_mode": "blocked",
        "positive_control": {"analysis_only": True, "headroom_exists": None},
        "positive_control_headroom_exists": None,
        "per_attempt_rows": [],
        "numeric_gate_quality_claim": False,
        "gate_ready_to_ship": False,
        "cited_artifacts": deepcopy(list(cited)),
        "solve_provenance": "live_agent_self_discovery",
        "read_game_source": False,
        "per_game_adapters": False,
        "remote_submission": False,
        "submission_kernel_changed": False,
        "flagged_adversarial": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = evaluator.canonical_hash(artifact)
    return artifact


def terminal_metadata(
    *,
    started_at: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    cited: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    session: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    boundary_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    invocation = exp7471._invocation_reduction(boundary_events, child_terminal=True)
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "status": "complete_b2_measurement",
        "inference_substrate": invocation["inference_substrate"],
        "inference_substrate_class": invocation["inference_substrate_class"],
        "execution_venue": "host",
        "execution_venue_details": deepcopy(dict(session.get("runtime_receipt") or {})),
        "model_invoked": invocation["model_invoked"],
        "invocation_counts": invocation["invocation_counts"],
        "MODEL_SPECS": [deepcopy(dict(model_spec))],
        "model_specs": [deepcopy(dict(model_spec))],
        "random_seed": {"episodes": list(EPISODE_SEEDS), "ordering": 7_531},
        "duration_s": max(0.000001, float(duration_s)),
        "preconditions_checked": deepcopy(list(checks)),
        "protocol_schedule": deepcopy(list(schedule)),
        "episode_rows": _episode_summaries(session),
        "cited_artifacts": deepcopy(list(cited)),
        "solve_provenance": "live_agent_self_discovery",
        "generalization_scope": "public_adapter_withheld_development_proxy",
        "read_game_source": False,
        "per_game_adapters": False,
        "hidden_game_efficacy_claim": False,
        "remote_submission": False,
        "submission_kernel_changed": False,
        "flagged_adversarial": False,
    }


def run_experiment() -> JsonDict:  # pragma: no cover - host GPU orchestration.
    """Run corrected preflight, one owned child, and the pure B2 reducer."""

    started = time.monotonic()
    started_at = utc_now()
    progress(started, "1", "static_preconditions_before")
    static_checks, cited, environment_dir = static_preconditions()
    frozen = e6.load_json(REPO_ROOT / FROZEN_PANEL_PATH)
    try:
        schedule = build_schedule(frozen)
    except ValueError:
        schedule = []
    progress(
        started,
        "1",
        "static_preconditions_after",
        passed=all(row.get("passed") is True for row in static_checks),
    )
    if not all(row.get("passed") is True for row in static_checks):
        failed = next(row for row in static_checks if row.get("passed") is not True)
        artifact = blocked_artifact(
            failed_check=failed,
            checks=static_checks,
            cited=cited,
            schedule=schedule,
            duration_s=time.monotonic() - started,
        )
        e6.write_json(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    progress(started, "2", "nvidia_smi_admission_before")
    runtime_checks, runtime_cited, resources = e6.collect_runtime_preconditions(
        REPO_ROOT,
        started,
    )
    cited.extend(runtime_cited)
    checks = [*static_checks, *runtime_checks]
    progress(
        started,
        "2",
        "nvidia_smi_admission_after",
        passed=all(row.get("passed") is True for row in runtime_checks),
    )
    if not all(row.get("passed") is True for row in runtime_checks):
        failed = next(row for row in runtime_checks if row.get("passed") is not True)
        artifact = blocked_artifact(
            failed_check=failed,
            checks=checks,
            cited=cited,
            schedule=schedule,
            duration_s=time.monotonic() - started,
        )
        e6.write_json(REPO_ROOT / RESULT_PATH, artifact)
        return artifact
    if resources.get("gpu") is None or environment_dir is None:
        raise RuntimeError("validated B2 resources disappeared")

    configure_e6_driver()
    for relative in (
        BOUNDARY_PATH,
        RUNTIME_EVENT_PATH,
        ACTION_PATH,
        SESSION_PATH,
        CHECKPOINT_PATH,
        TELEMETRY_PATH,
        RAW_DIR / "episode_rows.json",
    ):
        (REPO_ROOT / relative).unlink(missing_ok=True)
    e6.write_json(
        REPO_ROOT / SCHEDULE_PATH,
        {"rows": schedule, "frozen_panel": FROZEN_PANEL_PATH.as_posix()},
    )
    os.environ["CARNOT_ARC_PUBLIC_ENV_DIR"] = str(environment_dir)
    os.environ[TELEMETRY_ENV_FLAG] = "1"
    os.environ[TELEMETRY_PATH_ENV] = str(REPO_ROOT / TELEMETRY_PATH)
    os.environ["CARNOT_B2_MIN_GATE_OPPORTUNITIES"] = str(evaluator.MIN_GATE_OPPORTUNITIES)
    os.environ["CARNOT_B2_MIN_INDUCTION_ATTEMPTS"] = str(evaluator.MIN_INDUCTION_ATTEMPTS)
    remaining_s = max(0.0, TOTAL_LIVE_LIMIT_S - (time.monotonic() - started))
    exp7471.AGGREGATE_LIVE_LIMIT_S = remaining_s
    os.environ["CARNOT_E6_DEADLINE_MONOTONIC_NS"] = str(
        time.monotonic_ns() + int(remaining_s * 1_000_000_000)
    )
    progress(
        started,
        "3",
        "live_run_before",
        gpu_index=1,
        maximum_units=len(schedule),
    )
    session = exp7471.run_child_with_lease(
        resources=resources,
        schedule_path=REPO_ROOT / SCHEDULE_PATH,
        started=started,
    )
    progress(
        started,
        "3",
        "live_run_after",
        observed_units=len(session.get("episodes") or []),
    )

    boundary_events = InvocationBoundaryLedger(REPO_ROOT / BOUNDARY_PATH).read_events()
    runtime = dict(session.get("runtime_receipt") or {})
    offload_check = e6._check(
        "owned_server_cuda_offload_near_18gb",
        {"minimum_mb": e6.OFFLOAD_MIN_MB, "maximum_mb": e6.OFFLOAD_MAX_MB},
        runtime.get("owned_server_vram_mb_after_load"),
        passed=runtime.get("offload_real") is True,
        path="nvidia-smi_compute_process_used_gpu_memory",
    )
    checks.append(offload_check)
    for relative, role in (
        (SCHEDULE_PATH, "b2_protocol"),
        (SESSION_PATH, "live_session"),
        (BOUNDARY_PATH, "current_invocation_ledger"),
        (RUNTIME_EVENT_PATH, "request_events"),
        (ACTION_PATH, "action_events"),
        (TELEMETRY_PATH, "b2_induction_telemetry"),
    ):
        path = REPO_ROOT / relative
        if path.is_file():
            cited.append(_cite(relative, role))
    if not offload_check["passed"]:
        artifact = blocked_artifact(
            failed_check=offload_check,
            checks=checks,
            cited=cited,
            schedule=schedule,
            duration_s=time.monotonic() - started,
        )
        e6.write_json(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    progress(started, "4", "analysis_before")
    metadata = terminal_metadata(
        started_at=started_at,
        duration_s=time.monotonic() - started,
        checks=checks,
        cited=cited,
        schedule=schedule,
        session=session,
        model_spec=dict(resources["model_spec"]),
        boundary_events=boundary_events,
    )
    telemetry_rows = load_telemetry(REPO_ROOT / TELEMETRY_PATH)
    artifact = evaluator.build_measurement(telemetry_rows, metadata=metadata)
    e6.write_json(REPO_ROOT / RESULT_PATH, artifact)
    progress(
        started,
        "4",
        "analysis_after",
        gate_opportunities=artifact["gate_opportunity_count"],
        induction_attempts=artifact["induction_attempt_count"],
        floor_met=artifact["sample_floor"]["met"],
        headroom=artifact["positive_control_headroom_exists"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Reuse the E6 child contract after installing B2-owned constants."""

    configure_e6_driver()
    return e6.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    args = parse_args(argv)
    if args.role == "live-session":
        return e6.run_live_session(args)
    artifact = run_experiment()
    return 0 if str(artifact.get("honest_verdict", "")).startswith(("complete_", "blocked_")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
