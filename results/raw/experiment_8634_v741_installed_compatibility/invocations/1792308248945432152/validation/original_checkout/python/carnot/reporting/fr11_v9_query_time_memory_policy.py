"""Exp 1484 FR-11 v9 query-time memory policy replay.

This audit reuses Exp 1471 verified memory rows as an opt-in verifier signal
and compares memory-disabled against memory-enabled outcomes on the same
bounded replay set. It does not generate new LLM rows and does not make memory
global default behavior.

Spec: REQ-LEARN-1484, SCENARIO-LEARN-1484, SCENARIO-LEARN-1485.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline import query_time_memory_policy as memory_policy


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = "experiment_1484_fr11_v9_query_time_memory_policy.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_EXP1471_PATH = DEFAULT_RESULTS_DIR / "experiment_1471_fr11_v8_verified_memory_growth_pivot.json"
DEFAULT_EXP1472_PATH = DEFAULT_RESULTS_DIR / "experiment_1472_online_verifier_asymmetric_mistake_budget.json"

EXPERIMENT = "1484_fr11_v9_query_time_memory_policy"
SCHEMA = "fr11_v9_query_time_memory_policy_v1"
RUN_DATE = "20260507"
MIN_NONFORGETTING_RATE = 0.99
MAX_REPLAY_PAIRS = 32
MEMORY_POLICY_PATH = (
    "carnot.pipeline.query_time_memory_policy.evaluate_query_time_memory_policy"
)

MODEL_SPECS: list[dict[str, str]] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "optional_fresh_row_generator_if_new_llm_cases_are_needed",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "optional_fresh_row_generator_if_new_llm_cases_are_needed",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "optional_fresh_row_generator_if_new_llm_cases_are_needed",
    },
]

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "policy_integration_ready",
    "continuous_self_learning_task",
    "replay_cases_evaluated",
    "baseline_task_success_rate",
    "memory_task_success_rate",
    "task_success_delta",
    "soundness_mistakes",
    "completeness_mistakes",
    "nonforgetting_rate",
    "memory_policy_path",
    "tests_run",
    "promotion_allowed",
    "honest_verdict",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _to_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")  # pragma: no cover
    return payload


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1484-1/7: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1484", "SCENARIO-LEARN-1484", "SCENARIO-LEARN-1485"],
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "model_specs": MODEL_SPECS,
            "policy_integration_ready": False,
            "continuous_self_learning_task": "FR-11 query-time memory policy replay",
            "replay_cases_evaluated": 0,
            "baseline_task_success_rate": 0.0,
            "memory_task_success_rate": 0.0,
            "task_success_delta": 0.0,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "nonforgetting_rate": None,
            "memory_policy_path": MEMORY_POLICY_PATH,
            "tests_run": [],
            "promotion_allowed": False,
            "honest_verdict": "in_progress",
        },
    )


def _memory_updates(exp1471_artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    updates = exp1471_artifact.get("memory_updates")
    return updates if isinstance(updates, Mapping) else {}


def _bounded_replay_cases(
    exp1471_artifact: Mapping[str, Any],
    *,
    max_replay_pairs: int,
) -> tuple[tuple[memory_policy.QueryReplayCase, ...], memory_policy.VerifiedMemoryIndex]:
    updates = _memory_updates(exp1471_artifact)
    promoted = _as_str_list(updates.get("promoted"))[:max_replay_pairs]
    demoted = _as_str_list(updates.get("demoted"))[: len(promoted)]
    cases = [
        memory_policy.QueryReplayCase(
            case_id=case_id,
            expects_memory_signal=True,
            source="exp1471_promoted_verified_memory",
        )
        for case_id in promoted
    ]
    cases.extend(
        memory_policy.QueryReplayCase(
            case_id=case_id,
            expects_memory_signal=False,
            source="exp1471_demoted_negative_control",
        )
        for case_id in demoted
    )
    return tuple(cases), memory_policy.VerifiedMemoryIndex.from_ids(promoted)


def _source_soundness_mistakes(
    exp1471_artifact: Mapping[str, Any],
    exp1472_artifact: Mapping[str, Any],
) -> int:
    return max(
        _to_int(exp1471_artifact.get("soundness_mistakes")),
        _to_int(exp1472_artifact.get("soundness_mistakes")),
    )


def _source_completeness_mistakes(
    exp1471_artifact: Mapping[str, Any],
    exp1472_artifact: Mapping[str, Any],
) -> int:
    return max(
        _to_int(exp1471_artifact.get("completeness_mistakes")),
        _to_int(exp1472_artifact.get("completeness_mistakes")),
    )


def _integration_ready(
    *,
    exp1471_artifact: Mapping[str, Any],
    exp1472_artifact: Mapping[str, Any],
    replay_cases_evaluated: int,
    soundness_mistakes: int,
    nonforgetting_rate: float,
) -> bool:
    return (
        exp1471_artifact.get("status") == "complete"
        and exp1472_artifact.get("status") == "complete"
        and exp1471_artifact.get("headline_result_allowed") is True
        and exp1471_artifact.get("pivot_preserved") is True
        and exp1472_artifact.get("self_learning_claim_preserved") is True
        and replay_cases_evaluated > 0
        and soundness_mistakes == 0
        and nonforgetting_rate >= MIN_NONFORGETTING_RATE
    )


def _honest_verdict(
    *,
    soundness_mistakes: int,
    policy_integration_ready: bool,
    task_success_delta: float,
) -> str:
    if soundness_mistakes > 0:
        return "query_time_memory_policy_blocked_by_soundness_risk"
    if not policy_integration_ready:
        return "query_time_memory_policy_blocked_by_source_or_replay_gate"
    if task_success_delta > 0:
        return "query_time_memory_policy_improves_bounded_replay_without_false_accepts"
    return "query_time_memory_policy_no_positive_task_benefit"


def build_artifact(
    *,
    exp1471_artifact: Mapping[str, Any],
    exp1472_artifact: Mapping[str, Any],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
    max_replay_pairs: int = MAX_REPLAY_PAIRS,
) -> dict[str, Any]:
    """REQ-LEARN-1484: build the terminal query-time policy artifact."""

    replay_cases, memory_index = _bounded_replay_cases(
        exp1471_artifact,
        max_replay_pairs=max_replay_pairs,
    )
    baseline_eval = memory_policy.evaluate_query_time_memory_policy(
        replay_cases,
        memory_index,
    )
    memory_eval = memory_policy.evaluate_query_time_memory_policy(
        replay_cases,
        memory_index,
        memory_enabled=True,
    )
    baseline_rate = baseline_eval.task_success_rate
    memory_rate = memory_eval.task_success_rate
    delta = memory_rate - baseline_rate
    nonforgetting_rate = _to_float(
        exp1471_artifact.get(
            "nonforgetting_rate",
            exp1472_artifact.get("source_nonforgetting_rate"),
        )
    )
    source_soundness = _source_soundness_mistakes(exp1471_artifact, exp1472_artifact)
    soundness_mistakes = max(source_soundness, memory_eval.soundness_mistakes)
    completeness_mistakes = max(
        _source_completeness_mistakes(exp1471_artifact, exp1472_artifact),
        memory_eval.completeness_mistakes,
    )
    policy_ready = _integration_ready(
        exp1471_artifact=exp1471_artifact,
        exp1472_artifact=exp1472_artifact,
        replay_cases_evaluated=len(replay_cases),
        soundness_mistakes=soundness_mistakes,
        nonforgetting_rate=nonforgetting_rate,
    )
    promotion_allowed = (
        policy_ready and soundness_mistakes == 0 and delta >= 0.0
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1484", "SCENARIO-LEARN-1484", "SCENARIO-LEARN-1485"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "model_specs": exp1471_artifact.get("model_specs") or MODEL_SPECS,
        "policy_integration_ready": policy_ready,
        "continuous_self_learning_task": {
            "family": "FR-11",
            "task": "query-time verified-memory policy replay",
            "source_experiments": [1471, 1472],
            "opt_in_memory_signal": True,
            "new_headline_llm_generation_used": False,
        },
        "replay_cases_evaluated": len(replay_cases),
        "baseline_task_success_rate": round(baseline_rate, 6),
        "memory_task_success_rate": round(memory_rate, 6),
        "task_success_delta": round(delta, 6),
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "nonforgetting_rate": round(nonforgetting_rate, 6),
        "memory_policy_path": MEMORY_POLICY_PATH,
        "tests_run": list(commands_run or []),
        "promotion_allowed": promotion_allowed,
        "honest_verdict": _honest_verdict(
            soundness_mistakes=soundness_mistakes,
            policy_integration_ready=policy_ready,
            task_success_delta=delta,
        ),
        "memory_policy_replay": {
            "baseline_memory_disabled": baseline_eval.to_dict(),
            "memory_enabled": memory_eval.to_dict(),
            "source_soundness_mistakes": source_soundness,
            "source_completeness_mistakes": _source_completeness_mistakes(
                exp1471_artifact,
                exp1472_artifact,
            ),
            "bounded_replay_pairs": min(
                max_replay_pairs,
                len(_as_str_list(_memory_updates(exp1471_artifact).get("promoted"))),
            ),
        },
        "source_artifacts": [
            "results/experiment_1471_fr11_v8_verified_memory_growth_pivot.json",
            "results/experiment_1472_online_verifier_asymmetric_mistake_budget.json",
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1484-4/6/7: enforce required fields and promotion gate."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "in_progress":
        return
    if artifact["status"] != "complete":
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover

    baseline_rate = float(artifact["baseline_task_success_rate"])
    memory_rate = float(artifact["memory_task_success_rate"])
    delta = float(artifact["task_success_delta"])
    expected_delta = round(memory_rate - baseline_rate, 6)
    soundness = int(artifact["soundness_mistakes"])
    nonforgetting = float(artifact["nonforgetting_rate"])
    ready = bool(artifact["policy_integration_ready"])
    expected_promotion = ready and soundness == 0 and delta >= 0.0

    if int(artifact["replay_cases_evaluated"]) <= 0:
        raise AssertionError("replay_cases_evaluated must be positive")
    if not 0.0 <= baseline_rate <= 1.0 or not 0.0 <= memory_rate <= 1.0:
        raise AssertionError("task success rates must be probabilities")  # pragma: no cover
    if delta != expected_delta:
        raise AssertionError("task_success_delta must equal memory minus baseline rate")
    if ready and nonforgetting < MIN_NONFORGETTING_RATE:
        raise AssertionError("policy_integration_ready requires nonforgetting preservation")
    if bool(artifact["promotion_allowed"]) != expected_promotion:
        raise AssertionError("promotion_allowed must match soundness/delta/readiness gates")


def run(
    *,
    exp1471_path: Path | str = DEFAULT_EXP1471_PATH,
    exp1472_path: Path | str = DEFAULT_EXP1472_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1484 end-to-end and write the final artifact."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1471_artifact=load_json(exp1471_path),
        exp1472_artifact=load_json(exp1472_path),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
