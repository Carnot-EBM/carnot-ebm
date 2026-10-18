"""Run codeonly-suppressed B2 induction telemetry on the frozen E6 panel.

The live child reuses Experiment 7491's qualified E3 path and composes its
exclusive timer with REQ-ARC-WMTE-7530 telemetry. REQ-ARC-WMTE-10009 borrows
the validated non-default codeonly mechanism for bounded B2 induction calls.
This is not parity with the live path's default think-mode induction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import importlib.util
import os
from pathlib import Path
import statistics
import threading
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
from carnot.agentic.arc_executable_world_model import _L2_CODEONLY_DIRECTIVE


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
EXPERIMENT_ID = 10009
EXPERIMENT_NAME = "exp10009-b2-induction-gate-measurement-v3"
TASK_ID = "experiment_10009_b2_induction_gate_measurement_v3"
SCHEMA = "carnot.arc.b2_induction_gate_measurement.v3"
TOTAL_LIVE_LIMIT_S = 3 * 60 * 60.0
INDUCTION_MAX_TOKENS = 4096
INDUCTION_CODEONLY = True
METHODOLOGY_SCOPE = (
    "This measures codeonly-suppressed induction under a bounded 4,096-token budget. "
    "It does not measure or restore parity with the live path's default think-mode induction."
)
SUPERSEDES_PATH = Path("results/experiment_10008_b2_induction_gate_measurement_v2.json")
SUPERSEDED_REASON = (
    "Experiment 10008 exhausted all 4,096 completion tokens in hidden reasoning: "
    "durable responses had finish_reason=length and empty content. Experiment 10009 "
    "borrows the validated non-default codeonly mechanism so bounded attempts emit code."
)

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
RESULT_PATH = Path("results/experiment_10009_b2_induction_gate_measurement_v3.json")
RAW_DIR = Path("results/raw/experiment_10009_b2_induction_gate_measurement_v3")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
TELEMETRY_PATH = RAW_DIR / "induction_gate_telemetry.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_10009_b2_induction_gate_measurement_v3.json")
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
        f"[exp10009] step={step} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
                    "max_new_tokens_per_call": INDUCTION_MAX_TOKENS,
                    "induction_codeonly": INDUCTION_CODEONLY,
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


def completion_token_distribution(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize completion lengths and expose a possible remaining hard cap."""

    values = sorted(
        int(row["completion_tokens"])
        for row in attempts
        if isinstance(row.get("completion_tokens"), int)
        and not isinstance(row.get("completion_tokens"), bool)
        and int(row["completion_tokens"]) >= 0
    )
    histogram = Counter(values)
    uniform = bool(values) and len(histogram) == 1
    return {
        "count": len(values),
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "unique_count": len(histogram),
        "uniform": uniform,
        "possible_remaining_hard_cap": uniform and len(values) > 1,
    }


def durable_completion_token_evidence(
    attempts: Sequence[Mapping[str, Any]], raw_dir: Path
) -> JsonDict:
    """Use transport response receipts instead of possibly stale policy fields."""

    response_tokens: list[int] = []
    response_counts: Counter[str] = Counter()
    for response_path in sorted(raw_dir.glob("*/requests/*_response.json")):
        if not _is_codeonly_response(response_path):
            continue
        response = e6.load_json(response_path)
        usage = response.get("usage")
        completion = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        if not isinstance(completion, int) or isinstance(completion, bool):
            completion = response.get("tokens_predicted")
        if not isinstance(completion, int) or isinstance(completion, bool):
            continue
        response_tokens.append(completion)
        response_counts[response_path.parents[1].name.replace("__", ":")] += 1

    attempts_by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in attempts:
        attempts_by_episode.setdefault(str(row.get("episode_id")), []).append(row)
    rows_without_distinct_response: list[str] = []
    for episode_id, episode_attempts in attempts_by_episode.items():
        completed = response_counts[episode_id]
        rows_without_distinct_response.extend(
            str(row.get("attempt_id")) for row in episode_attempts[completed:]
        )

    reported = completion_token_distribution(attempts)
    durable = completion_token_distribution(
        [{"completion_tokens": value} for value in response_tokens]
    )
    return {
        "authoritative_source": "durable request response usage receipts",
        "distribution": durable,
        "completed_response_count": len(response_tokens),
        "attempt_row_count": len(attempts),
        "attempt_rows_with_reported_completion_tokens": reported["count"],
        "rows_without_distinct_completed_response": rows_without_distinct_response,
        "per_attempt_reported_distribution": reported,
        "warning": (
            "Rows without a distinct completed response can retain the prior proposer "
            "usage after request-budget exhaustion; they are excluded from the published "
            "completion-token distribution."
            if rows_without_distinct_response
            else None
        ),
    }


def _is_codeonly_response(response_path: Path) -> bool:
    request_path = response_path.with_name(
        response_path.name.replace("_response.json", "_request.json")
    )
    if not request_path.is_file():
        return False
    request = e6.load_json(request_path)
    prompt = request.get("prompt")
    return (
        isinstance(prompt, str)
        and prompt.startswith(_L2_CODEONLY_DIRECTIVE)
        and prompt.endswith("\n```python\n")
        and request.get("stop") == ["```"]
    )


def raw_response_evidence(raw_dir: Path) -> JsonDict:
    """Read durable response files and report content and termination directly."""

    rows: list[JsonDict] = []
    for response_path in sorted(raw_dir.glob("*/requests/*_response.json")):
        response = e6.load_json(response_path)
        try:
            displayed_path = response_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            displayed_path = response_path.as_posix()
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if isinstance(message, Mapping):
            content = message.get("content")
            reasoning = message.get("reasoning_content")
        else:
            content = response.get("content")
            reasoning = response.get("reasoning_content")
        raw_finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        raw_stop_type = response.get("stop_type")
        if isinstance(raw_finish_reason, str):
            normalized_finish_reason = raw_finish_reason
        elif raw_stop_type == "limit":
            normalized_finish_reason = "length"
        elif raw_stop_type in {"eos", "stop", "word"}:
            normalized_finish_reason = "stop"
        else:
            normalized_finish_reason = None
        rows.append(
            {
                "path": displayed_path,
                "response_schema": (
                    "openai_chat_completions" if isinstance(choice, Mapping) else "llama_completion"
                ),
                "content_nonempty": isinstance(content, str) and bool(content.strip()),
                "content_char_count": len(content) if isinstance(content, str) else 0,
                "reasoning_content_char_count": len(reasoning) if isinstance(reasoning, str) else 0,
                "raw_finish_reason": raw_finish_reason,
                "raw_stop_type": raw_stop_type,
                "normalized_finish_reason": normalized_finish_reason,
                "codeonly_induction_response": _is_codeonly_response(response_path),
            }
        )
    codeonly_rows = [row for row in rows if row["codeonly_induction_response"] is True]
    excluded_rows = [row for row in rows if row["codeonly_induction_response"] is not True]
    finish_reasons = Counter(str(row["normalized_finish_reason"]) for row in codeonly_rows)
    excluded_finish_reasons = Counter(str(row["normalized_finish_reason"]) for row in excluded_rows)
    return {
        "authoritative_source": (
            "direct reads of durable response files joined to codeonly request payloads"
        ),
        "scope": "requests carrying _L2_CODEONLY_DIRECTIVE, pre-opened fence, and stop sequence",
        "response_count": len(codeonly_rows),
        "content_nonempty_count": sum(row["content_nonempty"] is True for row in codeonly_rows),
        "all_content_nonempty": bool(codeonly_rows)
        and all(row["content_nonempty"] is True for row in codeonly_rows),
        "normalized_finish_reason_histogram": dict(sorted(finish_reasons.items())),
        "all_finish_reason_stop": bool(codeonly_rows)
        and all(row["normalized_finish_reason"] == "stop" for row in codeonly_rows),
        "directly_inspected_response_count": min(2, len(codeonly_rows)),
        "directly_inspected_responses": codeonly_rows[:2],
        "response_rows": codeonly_rows,
        "excluded_non_codeonly_responses": {
            "response_count": len(excluded_rows),
            "content_nonempty_count": sum(row["content_nonempty"] is True for row in excluded_rows),
            "normalized_finish_reason_histogram": dict(sorted(excluded_finish_reasons.items())),
            "reason": (
                "Refactor/chat calls have codeonly_eligible=False and are outside the B2 "
                "induction-codeonly intervention."
            ),
        },
    }


def planned_attempt_summary(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    attempt_count = len(attempts)
    planned_count = sum(row.get("planned") is True for row in attempts)
    return {
        "attempt_count": attempt_count,
        "planned_true_count": planned_count,
        "planned_true_fraction": planned_count / attempt_count if attempt_count else None,
    }


def induction_model_spec(model_spec: Mapping[str, Any]) -> JsonDict:
    """Return the B2 receipt with explicit budget and codeonly scope."""

    corrected = deepcopy(dict(model_spec))
    decoding = corrected.get("decoding")
    if isinstance(decoding, Mapping):
        corrected["decoding"] = {**decoding, "max_new_tokens": INDUCTION_MAX_TOKENS}
    runtime = corrected.get("runtime_settings")
    if isinstance(runtime, Mapping):
        corrected["runtime_settings"] = {
            **runtime,
            "max_new_tokens_per_call": INDUCTION_MAX_TOKENS,
            "induction_codeonly": INDUCTION_CODEONLY,
            "induction_codeonly_transport": "raw_completion",
            "live_default_think_mode": True,
        }
    return corrected


def positive_control_diagnostic(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose whether the analysis-only oracle inputs carry useful variation."""

    attempt_count = len(attempts)
    planned = sum(row.get("planned") is True for row in attempts)
    verifier_observed = sum(
        row.get("verifier_result") not in (None, "not_observed") for row in attempts
    )
    progress = sum(row.get("progress_within_window") is True for row in attempts)
    saturated = attempt_count > 0 and progress == attempt_count
    return {
        "attempt_count": attempt_count,
        "planned_true_count": planned,
        "verifier_observed_count": verifier_observed,
        "progress_within_window_true_count": progress,
        "progress_signal_saturated": saturated,
        "interpretation": (
            "The progress proxy is saturated and cannot establish that B2 has no "
            "positive-control headroom."
            if saturated
            else "The oracle inputs contain variation; interpret headroom descriptively."
        ),
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
        "supersedes": SUPERSEDES_PATH.as_posix(),
        "superseded_reason": SUPERSEDED_REASON,
        "methodology_scope": METHODOLOGY_SCOPE,
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
        "completion_tokens_distribution": completion_token_distribution([]),
        "raw_response_evidence": raw_response_evidence(REPO_ROOT / RAW_DIR),
        "planned_attempt_summary": planned_attempt_summary([]),
        "induction_token_budget": {
            "max_tokens": INDUCTION_MAX_TOKENS,
            "override_method": ("keyword-only proposer and durable-capture constructor parameters"),
            "scope": "B2 E3 world-model induction and reinduction proposer only",
            "experiment_7471_source_constant_changed": False,
        },
        "induction_codeonly": {
            "enabled": INDUCTION_CODEONLY,
            "selection_seam": "keyword-only E6 live-child parameter, default off",
            "eligible_scope": "generate calls with codeonly_eligible=True only",
            "directive_source": (
                "carnot.agentic.arc_executable_world_model._L2_CODEONLY_DIRECTIVE"
            ),
            "preopened_fence": "```python",
            "stop_sequence": ["```"],
            "transport": "raw_completion",
            "live_default_parity_claim": False,
        },
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
    corrected_model_spec = induction_model_spec(model_spec)
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "supersedes": SUPERSEDES_PATH.as_posix(),
        "superseded_reason": SUPERSEDED_REASON,
        "methodology_scope": METHODOLOGY_SCOPE,
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
        "MODEL_SPECS": [deepcopy(corrected_model_spec)],
        "model_specs": [deepcopy(corrected_model_spec)],
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
        "induction_token_budget": {
            "max_tokens": INDUCTION_MAX_TOKENS,
            "override_method": ("keyword-only proposer and durable-capture constructor parameters"),
            "scope": "B2 E3 world-model induction and reinduction proposer only",
            "experiment_7471_source_constant_changed": False,
        },
        "induction_codeonly": {
            "enabled": INDUCTION_CODEONLY,
            "selection_seam": "keyword-only E6 live-child parameter, default off",
            "eligible_scope": "generate calls with codeonly_eligible=True only",
            "directive_source": (
                "carnot.agentic.arc_executable_world_model._L2_CODEONLY_DIRECTIVE"
            ),
            "preopened_fence": "```python",
            "stop_sequence": ["```"],
            "transport": "raw_completion",
            "live_default_parity_claim": False,
        },
    }


def apply_terminal_diagnostics(artifact: JsonDict, raw_dir: Path) -> JsonDict:
    """Join durable codeonly receipts and label diagnostics onto a reduced artifact."""

    completion_evidence = durable_completion_token_evidence(artifact["per_attempt_rows"], raw_dir)
    artifact["completion_token_attribution"] = completion_evidence
    artifact["completion_tokens_distribution"] = completion_evidence["distribution"]
    artifact["positive_control_diagnostic"] = positive_control_diagnostic(
        artifact["per_attempt_rows"]
    )
    response_evidence = raw_response_evidence(raw_dir)
    artifact["raw_response_evidence"] = response_evidence
    artifact["planned_attempt_summary"] = planned_attempt_summary(artifact["per_attempt_rows"])
    artifact["codeonly_limit_termination_count"] = response_evidence[
        "normalized_finish_reason_histogram"
    ].get("length", 0)
    artifact["possible_second_completion_cap"] = artifact["completion_tokens_distribution"][
        "possible_remaining_hard_cap"
    ]
    artifact["completion_cap_interpretation"] = (
        "Every observed completion has the same length. Codeonly did not establish a "
        "non-binding completion distribution; inspect raw termination evidence."
        if artifact["possible_second_completion_cap"]
        else (
            "Observed codeonly completion lengths vary, so no uniform cap remains. "
            f"However, {artifact['codeonly_limit_termination_count']} codeonly response(s) "
            "still ended at the 4,096-token length limit."
        )
    )
    artifact["reproducibility_checksum"] = evaluator.canonical_hash(artifact)
    return artifact


def run_experiment() -> JsonDict:  # pragma: no cover - host GPU orchestration.
    """Run codeonly preflight, one owned child, and the pure B2 reducer."""

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
    artifact = apply_terminal_diagnostics(artifact, REPO_ROOT / RAW_DIR)
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


def run_live_session(args: argparse.Namespace) -> int:
    """Run only B2 induction generation with codeonly and the 4,096-token budget."""

    return e6.run_live_session(
        args,
        induction_max_tokens=INDUCTION_MAX_TOKENS,
        induction_codeonly=INDUCTION_CODEONLY,
    )


def _run_with_heartbeat(args: argparse.Namespace) -> int:
    """Keep every live-process silence gap below the measured 60-second limit."""

    heartbeat_started = time.monotonic()
    stop = threading.Event()

    def emit() -> None:
        while not stop.wait(55.0):
            progress(heartbeat_started, "heartbeat", "alive")

    thread = threading.Thread(target=emit, name="exp10009-progress", daemon=True)
    thread.start()
    try:
        if args.role == "live-session":
            return run_live_session(args)
        artifact = run_experiment()
        return (
            0
            if str(artifact.get("honest_verdict", "")).startswith(("complete_", "blocked_"))
            else 1
        )
    finally:
        stop.set()
        thread.join(timeout=1.0)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    args = parse_args(argv)
    return _run_with_heartbeat(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
