"""Exp 1497 FR-11 v10 trace2skill daily-eval cadence and rot check.

The v10 step does not generate new LLM traces. It turns the already-measured
Exp 1484/1485 bounded replay into a durable daily manifest: one trace-derived
skill row per replay case, with explicit resolver checks and conservative rot
criteria. This keeps the continuous self-learning claim narrow: a daily
evaluation cadence over the measured suite, not broad autonomous learning.

Spec: REQ-LEARN-1497, SCENARIO-LEARN-1497, SCENARIO-LEARN-1498.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

OUTPUT_FILE = "experiment_1497_fr11_trace2skill_daily_eval_v10.json"
MANIFEST_FILE = "fr11_trace2skill_daily_eval_manifest_1497.jsonl"
OPS_NOTE_FILE = "fr11_trace2skill_daily_eval_1497.md"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_MANIFEST_PATH = DEFAULT_RESULTS_DIR / MANIFEST_FILE
DEFAULT_OPS_NOTE_PATH = REPO_ROOT / "ops" / OPS_NOTE_FILE
DEFAULT_EXP1484_PATH = DEFAULT_RESULTS_DIR / "experiment_1484_fr11_v9_query_time_memory_policy.json"
DEFAULT_EXP1485_PATH = DEFAULT_RESULTS_DIR / "experiment_1485_fr11_completeness_reduction_audit.json"

EXPERIMENT = "1497_fr11_trace2skill_daily_eval_v10"
SCHEMA = "fr11_trace2skill_daily_eval_v10"
RUN_DATE = "20260507"
CONTINUOUS_SELF_LEARNING_TASK = (
    "FR-11 v10 bounded trace2skill daily evaluation cadence and rot check"
)
DETERMINISTIC_EVAL_NOTE = (
    "deterministic replay evaluation was sufficient; no LLM judge or generator was used"
)
READY_VERDICT = "complete: fr11_v10_trace2skill_daily_eval_manifest_ready_zero_soundness"
NOT_READY_VERDICT = "complete: fr11_v10_trace2skill_daily_eval_manifest_not_ready"
SOURCE_POLICY_BLOCKED_VERDICT = "complete: fr11_v10_trace2skill_source_policy_not_allowed"

MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "continuous_self_learning_task",
    "model_specs",
    "daily_eval_manifest_ready",
    "trace2skill_cases_evaluated",
    "skills_evaluated",
    "rotted_skill_count",
    "promoted_skill_count",
    "retired_skill_count",
    "baseline_task_success_rate",
    "memory_task_success_rate",
    "task_success_delta",
    "soundness_mistakes",
    "completeness_mistakes",
    "daily_eval_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)
DECISION_REQUIRED_FIELDS: tuple[str, ...] = (
    "case_id",
    "task_success",
    "soundness_mistake",
    "completeness_mistake",
    "verifier_signal",
)
TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
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


def _display_path(path: Path | str, *, project_root: str | Path = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.relative_to(Path(project_root)).as_posix()
    except ValueError:
        return target.name


def _deterministic_gpu_probe() -> dict[str, Any]:
    return {
        "gpu_required": False,
        "cuda_available": None,
        "gpu_count": None,
        "probe_scope": "not_required_for_deterministic_replay",
        "reason": DETERMINISTIC_EVAL_NOTE,
    }


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"artifact must be a JSON object: {path}")
    return payload


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1497-1/7: write the durable bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1497", "SCENARIO-LEARN-1497", "SCENARIO-LEARN-1498"],
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "continuous_self_learning_task": CONTINUOUS_SELF_LEARNING_TASK,
            "model_specs": list(MODEL_SPECS),
            "daily_eval_manifest_ready": False,
            "trace2skill_cases_evaluated": 0,
            "skills_evaluated": 0,
            "rotted_skill_count": 0,
            "promoted_skill_count": 0,
            "retired_skill_count": 0,
            "baseline_task_success_rate": 0.0,
            "memory_task_success_rate": 0.0,
            "task_success_delta": 0.0,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "daily_eval_manifest_path": _display_path(
                DEFAULT_MANIFEST_PATH,
                project_root=project_root,
            ),
            "models_used": [],
            "gpu_probe": _deterministic_gpu_probe(),
            "blockers": [],
            "honest_verdict": "in_progress",
        },
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise AssertionError(f"{name} is required")


def _sequence(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AssertionError(f"{name} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise AssertionError(f"{name} entries must be objects")
    return list(value)


def _replay_eval(exp1484_artifact: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    replay = _mapping(exp1484_artifact.get("memory_policy_replay"), "memory_policy_replay")
    return _mapping(replay.get(name), f"{name} replay ledger")


def _safe_skill_id(case_id: str) -> str:
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "-", case_id).strip("-")
    return f"fr11_v10_trace2skill/{safe_case or 'unknown-case'}"


def _missing_fields(decision: Mapping[str, Any]) -> list[str]:
    return [field for field in DECISION_REQUIRED_FIELDS if field not in decision]


def _outcome(decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_success": bool(decision.get("task_success")),
        "soundness_mistake": bool(decision.get("soundness_mistake")),
        "completeness_mistake": bool(decision.get("completeness_mistake")),
        "verifier_signal": str(decision.get("verifier_signal") or ""),
    }


def _resolver_checks(
    *,
    missing_source_artifact: bool,
    paired_replay_case: bool,
    verifier_signal_present: bool,
    source_policy_allowed: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "source_artifact_present",
            "expected": True,
            "observed": not missing_source_artifact,
        },
        {
            "name": "paired_replay_case",
            "expected": True,
            "observed": paired_replay_case,
        },
        {
            "name": "verifier_signal_present",
            "expected": True,
            "observed": verifier_signal_present,
        },
        {
            "name": "zero_soundness_policy_allowed",
            "expected": True,
            "observed": source_policy_allowed,
        },
    ]


def _source_policy_allowed(exp1485_artifact: Mapping[str, Any]) -> bool:
    return (
        exp1485_artifact.get("status") == "complete"
        and exp1485_artifact.get("policy_change_allowed") is True
        and int(exp1485_artifact.get("candidate_soundness_mistakes") or 0) == 0
    )


def _row_decision(*, rotted: bool, improved: bool, source_policy_allowed: bool) -> str:
    if rotted:
        return "retire"
    if improved and source_policy_allowed:
        return "promote"
    return "retain"


def build_manifest_rows(
    exp1484_artifact: Mapping[str, Any],
    exp1485_artifact: Mapping[str, Any],
    *,
    source_paths: Sequence[Path | str],
    run_date: str = RUN_DATE,
) -> list[dict[str, Any]]:
    """REQ-LEARN-1497-2/3/4/5: derive daily trace2skill rows from v9 replay."""

    baseline_eval = _replay_eval(exp1484_artifact, "baseline_memory_disabled")
    memory_eval = _replay_eval(exp1484_artifact, "memory_enabled")
    baseline_decisions = _sequence(
        baseline_eval.get("decisions"),
        "baseline_memory_disabled.decisions",
    )
    memory_decisions = _sequence(memory_eval.get("decisions"), "memory_enabled.decisions")
    memory_by_case = {str(decision.get("case_id")): decision for decision in memory_decisions}
    missing_source_artifact = bool(source_paths) and not all(Path(path).exists() for path in source_paths)
    source_policy_allowed = _source_policy_allowed(exp1485_artifact)
    rows: list[dict[str, Any]] = []

    for baseline in baseline_decisions:
        case_id = str(baseline.get("case_id") or "unknown-case")
        memory = memory_by_case.get(case_id, {})
        baseline_missing = _missing_fields(baseline)
        memory_missing = _missing_fields(memory)
        paired_replay_case = bool(memory)
        verifier_signal_present = not baseline_missing and not memory_missing
        baseline_outcome = _outcome(baseline)
        memory_outcome = _outcome(memory)
        rot_criteria = {
            "missing_source_artifact": missing_source_artifact,
            "unresolved_verifier_dependency": not verifier_signal_present
            or not source_policy_allowed,
            "reduced_task_success": baseline_outcome["task_success"]
            and not memory_outcome["task_success"],
            "new_soundness_mistake": memory_outcome["soundness_mistake"],
            "schema_drift": bool(baseline_missing or memory_missing or not paired_replay_case),
        }
        rotted = any(rot_criteria.values())
        improved = memory_outcome["task_success"] and not baseline_outcome["task_success"]
        decision = _row_decision(
            rotted=rotted,
            improved=improved,
            source_policy_allowed=source_policy_allowed,
        )
        rows.append(
            {
                "schema": "fr11_trace2skill_daily_eval_row_v1",
                "spec": ["REQ-LEARN-1497", "SCENARIO-LEARN-1497", "SCENARIO-LEARN-1498"],
                "run_date": run_date,
                "skill_id": _safe_skill_id(case_id),
                "case_id": case_id,
                "source_artifacts": [str(path) for path in source_paths],
                "expected_resolver_checks": _resolver_checks(
                    missing_source_artifact=missing_source_artifact,
                    paired_replay_case=paired_replay_case,
                    verifier_signal_present=verifier_signal_present,
                    source_policy_allowed=source_policy_allowed,
                ),
                "baseline_outcome": baseline_outcome,
                "memory_assisted_outcome": memory_outcome,
                "rot_criteria": rot_criteria,
                "rot_reasons": [name for name, active in rot_criteria.items() if active],
                "rotted": rotted,
                "decision": decision,
            }
        )
    return rows


def write_manifest(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """REQ-LEARN-1497-6: persist one JSONL row per evaluated skill/case."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _rate(rows: Sequence[Mapping[str, Any]], outcome_key: str) -> float:
    if not rows:
        return 0.0
    successes = sum(1 for row in rows if row[outcome_key]["task_success"])
    return successes / len(rows)


def _blockers(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path | str],
    source_policy_allowed: bool,
    manifest_exists: bool,
) -> list[str]:
    blockers: list[str] = []
    if source_paths and not all(Path(path).exists() for path in source_paths):
        blockers.append("missing_source_artifact")
    if not rows:
        blockers.append("no_trace2skill_rows")
    if not source_policy_allowed:
        blockers.append("source_policy_not_allowed")
    if not manifest_exists:
        blockers.append("daily_eval_manifest_not_written")
    for criterion in ("schema_drift", "unresolved_verifier_dependency", "new_soundness_mistake"):
        if any(row["rot_criteria"][criterion] for row in rows):
            blockers.append(criterion)
    return blockers


def _honest_verdict(*, manifest_ready: bool, source_policy_allowed: bool) -> str:
    if not source_policy_allowed:
        return SOURCE_POLICY_BLOCKED_VERDICT
    if manifest_ready:
        return READY_VERDICT
    return NOT_READY_VERDICT


def build_artifact(
    *,
    exp1484_artifact: Mapping[str, Any],
    exp1485_artifact: Mapping[str, Any],
    manifest_path: Path | str,
    source_paths: Sequence[Path | str],
    manifest_exists: bool,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1497: build the terminal daily-eval artifact."""

    rows = build_manifest_rows(
        exp1484_artifact,
        exp1485_artifact,
        source_paths=source_paths,
        run_date=run_date,
    )
    baseline_rate = _rate(rows, "baseline_outcome")
    memory_rate = _rate(rows, "memory_assisted_outcome")
    delta = memory_rate - baseline_rate
    promoted = sum(1 for row in rows if row["decision"] == "promote")
    retired = sum(1 for row in rows if row["decision"] == "retire")
    rotted = sum(1 for row in rows if row["rotted"])
    manifest_ready = bool(manifest_exists) and len(rows) > 0
    source_policy_allowed = _source_policy_allowed(exp1485_artifact)
    soundness_mistakes = sum(1 for row in rows if row["memory_assisted_outcome"]["soundness_mistake"])
    completeness_mistakes = sum(
        1 for row in rows if row["memory_assisted_outcome"]["completeness_mistake"]
    )
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1497", "SCENARIO-LEARN-1497", "SCENARIO-LEARN-1498"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "continuous_self_learning_task": CONTINUOUS_SELF_LEARNING_TASK,
        "model_specs": list(MODEL_SPECS),
        "daily_eval_manifest_ready": manifest_ready,
        "trace2skill_cases_evaluated": len(rows),
        "skills_evaluated": len(rows),
        "rotted_skill_count": rotted,
        "promoted_skill_count": promoted,
        "retired_skill_count": retired,
        "baseline_task_success_rate": round(baseline_rate, 6),
        "memory_task_success_rate": round(memory_rate, 6),
        "task_success_delta": round(delta, 6),
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "daily_eval_manifest_path": _display_path(manifest_path, project_root=project_root),
        "models_used": [],
        "gpu_probe": _deterministic_gpu_probe(),
        "blockers": _blockers(
            rows=rows,
            source_paths=source_paths,
            source_policy_allowed=source_policy_allowed,
            manifest_exists=bool(manifest_exists),
        ),
        "honest_verdict": _honest_verdict(
            manifest_ready=manifest_ready,
            source_policy_allowed=source_policy_allowed,
        ),
        "llm_usage_note": DETERMINISTIC_EVAL_NOTE,
        "source_artifacts": [_display_path(path, project_root=project_root) for path in source_paths],
        "source_experiments": [
            "experiment_1484_fr11_v9_query_time_memory_policy",
            "experiment_1485_fr11_completeness_reduction_audit",
        ],
        "tests_run": list(commands_run or []),
    }
    validate_artifact(artifact, manifest_path=manifest_path)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path | str | None = None,
) -> None:
    """REQ-LEARN-1497-6/7: enforce artifact fields and manifest readiness."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] == "in_progress":
        return
    if artifact["status"] != "complete":
        raise AssertionError(f"unsupported status: {artifact['status']}")
    honest_verdict = str(artifact["honest_verdict"])
    if not honest_verdict.startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")

    baseline_rate = float(artifact["baseline_task_success_rate"])
    memory_rate = float(artifact["memory_task_success_rate"])
    delta = float(artifact["task_success_delta"])
    if not 0.0 <= baseline_rate <= 1.0 or not 0.0 <= memory_rate <= 1.0:
        raise AssertionError("task success rates must be probabilities")
    if delta != round(memory_rate - baseline_rate, 6):
        raise AssertionError("task_success_delta must equal memory minus baseline rate")

    skills = int(artifact["skills_evaluated"])
    trace_cases = int(artifact["trace2skill_cases_evaluated"])
    rotted = int(artifact["rotted_skill_count"])
    promoted = int(artifact["promoted_skill_count"])
    retired = int(artifact["retired_skill_count"])
    if trace_cases != skills or promoted + retired > skills or rotted != retired:
        raise AssertionError("skill counts must be explicit")
    if min(
        trace_cases,
        skills,
        rotted,
        promoted,
        retired,
    ) < 0:
        raise AssertionError("skill counts must be non-negative")

    if bool(artifact["daily_eval_manifest_ready"]):
        candidate_path = Path(manifest_path or artifact["daily_eval_manifest_path"])
        if not candidate_path.exists():
            raise AssertionError("daily_eval_manifest_ready requires a manifest file")


def _ops_note(run_date: str) -> str:
    return f"""# FR-11 Trace2Skill Daily Eval 1497

Run date: {run_date}

## Cadence

Run the bounded trace2skill manifest build once per day after the query-time
memory replay artifacts are available. The cadence replays only measured
Exp 1484/1485 rows and writes `results/{MANIFEST_FILE}` plus the terminal
experiment artifact.

## Promotion Rules

Promote a trace-derived skill only when the source artifacts are present, the
verifier signal resolves, the Exp 1485 zero-soundness policy is allowed, no rot
criterion fires, and the memory-assisted outcome improves the paired baseline.

## Retirement Rules

Retire a skill when any rot criterion fires: missing source artifact,
unresolved verifier dependency, reduced task success, new soundness mistake, or
schema drift. Retirement is counted separately from promotion.

## Boundaries

This is a bounded daily evaluation cadence over replayed FR-11 trace-derived
skills. It does not claim broad autonomous learning, fresh LLM-generated skill
discovery, or production-default memory routing beyond the measured suite.
"""


def write_ops_note(path: Path | str, *, run_date: str = RUN_DATE) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_ops_note(run_date), encoding="utf-8")


def run(
    *,
    exp1484_path: Path | str = DEFAULT_EXP1484_PATH,
    exp1485_path: Path | str = DEFAULT_EXP1485_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    ops_note_path: Path | str = DEFAULT_OPS_NOTE_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1497 end-to-end and write the manifest, ops note, and artifact."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1484 = load_json(exp1484_path)
    exp1485 = load_json(exp1485_path)
    source_paths = (Path(exp1484_path), Path(exp1485_path))
    rows = build_manifest_rows(
        exp1484,
        exp1485,
        source_paths=source_paths,
        run_date=run_date,
    )
    write_manifest(manifest_path, rows)
    write_ops_note(ops_note_path, run_date=run_date)
    artifact = build_artifact(
        exp1484_artifact=exp1484,
        exp1485_artifact=exp1485,
        manifest_path=manifest_path,
        source_paths=source_paths,
        manifest_exists=Path(manifest_path).exists(),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
