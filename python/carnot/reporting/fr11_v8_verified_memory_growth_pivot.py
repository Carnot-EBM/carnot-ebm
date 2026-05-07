"""Exp 1471 FR-11 v8 verified memory-growth pivot.

Exp 1459 allowed exactly one follow-up: reuse Exp 1447's verified
SessionMemory growth policy on fresh verified local rows, with Exp 1449
temporal cases only as supporting feed. This module deliberately avoids a new
self-learning architecture; it is a narrow gate-and-persist workflow around the
existing v7 asymmetric fresh/replay threshold policy.

Spec: REQ-LEARN-1471, SCENARIO-LEARN-1471, SCENARIO-LEARN-1472.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseMemory, CaseRecord
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.session_memory import SessionMemory
from carnot.reporting import dvi_discriminative_verifier_training_v1 as dvi
from carnot.reporting import fr11_self_learning_v5 as fr11
from carnot.reporting import fr11_self_learning_v6_dvi_v3_gated as v6
from carnot.reporting import fr11_v7_memory_policy_growth as v7
from carnot.reporting import ltlzinc_temporal_continual_learning_adapter as temporal


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_TEMPORAL_DATASET_PATH = DEFAULT_RESULTS_DIR / temporal.DATASET_FILE
DEFAULT_SESSION_MEMORY_DIR = DEFAULT_RESULTS_DIR / "session_memory_1471"

OUTPUT_FILE = "experiment_1471_fr11_v8_verified_memory_growth_pivot.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / "experiment_1395_fr11_self_learning_v5.json"
DEFAULT_EXP1432_PATH = DEFAULT_RESULTS_DIR / v6.EXP1432_FILE
DEFAULT_EXP1446_PATH = DEFAULT_RESULTS_DIR / v7.EXP1446_FILE
DEFAULT_EXP1447_PATH = DEFAULT_RESULTS_DIR / v7.OUTPUT_FILE
DEFAULT_EXP1459_PATH = DEFAULT_RESULTS_DIR / "experiment_1459_self_learning_nonheadline_lineage_decision.json"

EXPERIMENT = "1471_fr11_v8_verified_memory_growth_pivot"
SCHEMA = "fr11_v8_verified_memory_growth_pivot_v1"
RUN_DATE = "20260507"
MIN_NONFORGETTING_RATE = 0.99
SESSION_MEMORY_MODEL_ID = "fr11_v8_exp1447_policy_fresh_verified_rows"
PROMOTED_PREFIX = "dvi_v8:verified:"
MAX_FOVER_ROWS = 128

MODEL_SPECS: list[dict[str, str]] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "preferred_fresh_row_generator_if_new_llm_cases_are_needed",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "secondary_fresh_row_generator_if_new_llm_cases_are_needed",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "secondary_fresh_row_generator_if_new_llm_cases_are_needed",
    },
]

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "self_learning_artifact_ready",
    "baseline_fresh_verified_sample_count",
    "fresh_verified_sample_count",
    "self_learning_delta_overall",
    "new_promoted_count",
    "memory_entries_added",
    "session_memory_updated",
    "nonforgetting_rate",
    "soundness_mistakes",
    "completeness_mistakes",
    "headline_result_allowed",
    "pivot_preserved",
    "pivot_retired",
    "honest_verdict",
)


@dataclass(frozen=True)
class CandidateLoad:
    """Fresh verified candidates plus skip counts for auditability."""

    cases: list[dvi.DviCase]
    counts: dict[str, int]


@dataclass(frozen=True)
class SessionPersistence:
    """Result of persisting v8 promotions through SessionMemory."""

    entries_added: int
    path: str | None


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1471-1: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1471", "SCENARIO-LEARN-1471", "SCENARIO-LEARN-1472"],
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "model_specs": MODEL_SPECS,
            "live_sota_model_inference_used": False,
            "live_sota_model_inference_model_path": None,
            "self_learning_artifact_ready": False,
            "baseline_fresh_verified_sample_count": None,
            "fresh_verified_sample_count": None,
            "self_learning_delta_overall": None,
            "new_promoted_count": None,
            "memory_entries_added": None,
            "session_memory_updated": None,
            "nonforgetting_rate": None,
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "headline_result_allowed": False,
            "pivot_preserved": False,
            "pivot_retired": False,
            "honest_verdict": "in_progress",
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")  # pragma: no cover
    return payload


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _to_int(value: Any, default: int = 0) -> int:
    return default if value is None else int(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _exp1447_promoted_case_ids(exp1447_artifact: Mapping[str, Any]) -> set[str]:
    promoted = exp1447_artifact.get("memory_updates", {}).get("promoted", [])
    if not isinstance(promoted, Sequence) or isinstance(promoted, (str, bytes)):
        return set()
    return {str(item).split(":")[-1] for item in promoted}


def _pivot_prerequisites_ready(
    exp1447_artifact: Mapping[str, Any],
    exp1459_artifact: Mapping[str, Any],
) -> bool:
    delta = _to_int(exp1447_artifact.get("self_learning_delta_overall"))
    entries = _to_int(exp1447_artifact.get("memory_entries_added"))
    promoted = _to_int(exp1447_artifact.get("new_promoted_count"))
    shape = exp1459_artifact.get("next_allowed_experiment_shape")
    shape_scope = shape.get("scope") if isinstance(shape, Mapping) else None
    return (
        exp1447_artifact.get("status") == "complete"
        and delta > 0
        and entries == delta
        and promoted == delta
        and bool(exp1447_artifact.get("session_memory_updated"))
        and _to_float(exp1447_artifact.get("nonforgetting_rate")) >= MIN_NONFORGETTING_RATE
        and exp1459_artifact.get("self_learning_headline_pivot_selected") is True
        and shape_scope == "exp1447_verified_memory_policy_growth_pivot"
    )


def _temporal_text(case: Mapping[str, Any]) -> str:
    return " ".join(
        (
            str(case.get("ltl_formula") or ""),
            str(case.get("minizinc_constraint") or ""),
            json.dumps(case.get("trace") or [], sort_keys=True),
        )
    ).strip()


def temporal_cases_to_dvi_candidates(
    temporal_cases: Sequence[Mapping[str, Any]],
) -> CandidateLoad:
    """REQ-LEARN-1471-3: verify Exp 1449 rows before DVI-style ingestion."""

    counts: Counter[str] = Counter()
    cases: list[dvi.DviCase] = []
    for row in temporal_cases:
        counts["temporal_supporting_feed_count"] += 1
        temporal.validate_case_schema(row)
        expected_satisfied = bool(row.get("expected_satisfied"))
        verifier_satisfied = temporal.verify_temporal_case(row)
        expected_state = "SAT" if expected_satisfied else "REPAIR_HINT"
        expected_label = 0 if expected_satisfied else 1
        if (
            verifier_satisfied != expected_satisfied
            or row.get("certificate_state") != expected_state
            or int(row.get("dvi_label")) != expected_label
        ):
            counts["temporal_verifier_mismatch"] += 1
            continue

        cases.append(
            dvi.DviCase(
                case_id=str(row["case_id"]),
                text=_temporal_text(row),
                label=expected_label,
                source="exp1449_temporal_supporting_feed",
            )
        )
        counts["verified_temporal_candidates"] += 1
    return CandidateLoad(cases=cases, counts=dict(counts))


def fresh_fover_candidates_after_exp1447(
    *,
    exp1395_artifact: Mapping[str, Any],
    exp1447_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    max_rows: int = MAX_FOVER_ROWS,
) -> CandidateLoad:
    """Load bounded FoVer rows that were not already promoted by Exp 1447."""

    exclude_ids = v6.exp1395_promoted_case_ids(exp1395_artifact) | _exp1447_promoted_case_ids(
        exp1447_artifact
    )
    loaded = v7.fresh_candidates_from_local_fover(fover_rows, exclude_case_ids=exclude_ids)
    bounded = loaded.cases[:max_rows]
    counts = dict(loaded.counts)
    counts["bounded_fover_candidates_used"] = len(bounded)
    return CandidateLoad(cases=bounded, counts=counts)


def _score_with_v8_ids(
    cases: Sequence[dvi.DviCase],
    state: v6.DviV3CheckpointState,
    policy: v7.V7Policy,
) -> list[dict[str, Any]]:
    variants = v7.verify_cases_with_v7_policy(cases, state, policy)
    cases_by_id = {case.case_id: case for case in cases}
    normalized: list[dict[str, Any]] = []
    for variant in variants:
        case = cases_by_id[str(variant["case_id"])]
        source_prefix = (
            "exp1449"
            if case.source == "exp1449_temporal_supporting_feed"
            else "fover"
        )
        updated = dict(variant)
        updated["variant_id"] = f"{PROMOTED_PREFIX}{source_prefix}:{case.case_id}"
        updated["source"] = "exp1471_v8_exp1447_policy_fresh_verified_rows"
        updated["candidate_source"] = case.source
        normalized.append(updated)
    return normalized


def _policy_mistakes(variants: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    soundness = 0
    completeness = 0
    for variant in variants:
        semantic_state = variant.get("semantic_state")
        certificate_state = variant.get("certificate_state")
        if semantic_state == fr11.STATE_SAT and certificate_state == fr11.STATE_REPAIR_HINT:
            soundness += 1
        elif semantic_state == fr11.STATE_REPAIR_HINT and certificate_state == fr11.STATE_SAT:
            completeness += 1
    return soundness, completeness


def persist_promotions_to_session_memory(
    promoted_variants: Sequence[Mapping[str, Any]],
    *,
    cases_by_id: Mapping[str, dvi.DviCase],
    session_memory_dir: Path | str,
) -> SessionPersistence:
    """REQ-LEARN-1471-4: save promoted v8 rows through SessionMemory."""

    if not promoted_variants:
        return SessionPersistence(entries_added=0, path=None)

    case_memory = CaseMemory()
    for variant in promoted_variants:
        case = cases_by_id[str(variant["case_id"])]
        is_incorrect = int(case.label) == 1
        violation_type = (
            "fr11_v8_verified_repair_hint"
            if is_incorrect
            else "fr11_v8_verified_sat"
        )
        case_memory.record(
            CaseRecord.normalize(
                benchmark="fr11_v8_verified_memory_growth",
                benchmark_slice=f"{case.source}:{case.case_id}",
                model_name=SESSION_MEMORY_MODEL_ID,
                case_id=case.case_id,
                violation_types=(violation_type,),
                prompt_text=case.text,
                description_texts=(
                    "Exp 1471 reused Exp 1447's asymmetric memory policy and "
                    "persisted this fresh verified row through SessionMemory.",
                ),
                baseline_success=not is_incorrect,
                repair_success=True,
                confidence=float(variant.get("secl_confidence") or 0.0),
                source_experiment=1471,
                source_artifact=f"results/{OUTPUT_FILE}",
                response_mode=case.source,
                verifier_path="exp1447_v7_asymmetric_policy_reused",
            )
        )

    library = ConstraintTemplateLibrary()
    library.register_builtin_templates()
    tracker = PerModelFPTracker()
    session = SessionMemory(str(session_memory_dir), SESSION_MEMORY_MODEL_ID)
    session.save(case_memory, library, tracker)
    loaded = session.load()
    if loaded is None:
        return SessionPersistence(entries_added=0, path=str(session._state_path()))  # pragma: no cover
    persisted_case_memory, _, _ = loaded
    return SessionPersistence(
        entries_added=len(persisted_case_memory.entries()),
        path=str(session._state_path()),
    )


def _future_block_rule(headline_allowed: bool) -> str | None:
    if headline_allowed:
        return None
    return (
        "Do not rerun the FR-11 v8 verified-memory-growth pivot unless a new "
        "verified fresh-row source and changed root cause are supplied."
    )


def _honest_verdict(*, status: str, headline_allowed: bool, nonforgetting_rate: float | None) -> str:
    if status != "complete":
        return "fr11_v8_blocked_pivot_prerequisite_missing"
    if headline_allowed:
        return "fr11_v8_positive_verified_memory_growth_persisted_without_forgetting"
    if nonforgetting_rate is None or nonforgetting_rate < MIN_NONFORGETTING_RATE:
        return "fr11_v8_pivot_retired_nonforgetting_gate_failed"
    return "fr11_v8_pivot_retired_no_verified_persisted_growth"


def build_artifact(
    *,
    exp1395_artifact: Mapping[str, Any],
    exp1432_artifact: Mapping[str, Any],
    exp1446_artifact: Mapping[str, Any],
    exp1447_artifact: Mapping[str, Any],
    exp1459_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    temporal_cases: Sequence[Mapping[str, Any]],
    session_memory_dir: Path | str = DEFAULT_SESSION_MEMORY_DIR,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
    max_fover_rows: int = MAX_FOVER_ROWS,
) -> dict[str, Any]:
    """REQ-LEARN-1471: build the terminal v8 pivot artifact."""

    policy = v7.load_v7_policy(exp1446_artifact)
    activation = v6.activate_dvi_v3_checkpoint(exp1432_artifact)
    pivot_ready = _pivot_prerequisites_ready(exp1447_artifact, exp1459_artifact)
    baseline_count = _to_int(exp1447_artifact.get("fresh_verified_sample_count"))
    fover_load = fresh_fover_candidates_after_exp1447(
        exp1395_artifact=exp1395_artifact,
        exp1447_artifact=exp1447_artifact,
        fover_rows=fover_rows,
        max_rows=max_fover_rows,
    )
    temporal_load = temporal_cases_to_dvi_candidates(temporal_cases)
    fresh_cases = [*fover_load.cases, *temporal_load.cases]

    variants: list[dict[str, Any]] = []
    if activation.active and activation.state is not None and pivot_ready:
        variants = _score_with_v8_ids(fresh_cases, activation.state, policy)

    memory_updates = v7.apply_memory_updates(variants)
    nonforgetting_rate = v7.measure_v7_nonforgetting_rate(
        exp1432_artifact=exp1432_artifact,
        exp1395_artifact=exp1395_artifact,
        fover_rows=fover_rows,
        activation=activation,
        policy=policy,
    )
    nonforgetting_preserved = (
        nonforgetting_rate is not None
        and float(nonforgetting_rate) >= MIN_NONFORGETTING_RATE
        and _to_float(exp1447_artifact.get("nonforgetting_rate")) >= MIN_NONFORGETTING_RATE
    )
    promoted_variants = [
        variant for variant in variants if variant.get("memory_action") == fr11.POLICY_PROMOTE
    ]
    persistence = (
        persist_promotions_to_session_memory(
            promoted_variants,
            cases_by_id={case.case_id: case for case in fresh_cases},
            session_memory_dir=session_memory_dir,
        )
        if nonforgetting_preserved
        else SessionPersistence(entries_added=0, path=None)
    )

    memory_entries_added = persistence.entries_added
    new_promoted_count = memory_entries_added
    delta = memory_entries_added
    soundness_mistakes, completeness_mistakes = _policy_mistakes(variants)
    status = "complete" if activation.active and pivot_ready else "blocked"
    headline_allowed = (
        status == "complete"
        and delta > 0
        and new_promoted_count >= 1
        and nonforgetting_rate is not None
        and float(nonforgetting_rate) >= MIN_NONFORGETTING_RATE
    )
    pivot_preserved = bool(headline_allowed)
    pivot_retired = not headline_allowed

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1471", "SCENARIO-LEARN-1471", "SCENARIO-LEARN-1472"],
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": status,
        "model_specs": MODEL_SPECS,
        "live_sota_model_inference_used": False,
        "live_sota_model_inference_model_path": None,
        "new_headline_llm_generation_used": False,
        "self_learning_artifact_ready": status == "complete",
        "baseline_fresh_verified_sample_count": baseline_count,
        "fresh_verified_sample_count": baseline_count + delta,
        "self_learning_delta_overall": delta,
        "new_promoted_count": new_promoted_count,
        "memory_entries_added": memory_entries_added,
        "session_memory_updated": memory_entries_added > 0,
        "nonforgetting_rate": round(float(nonforgetting_rate), 6)
        if nonforgetting_rate is not None
        else None,
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "headline_result_allowed": headline_allowed,
        "pivot_preserved": pivot_preserved,
        "pivot_retired": pivot_retired,
        "future_block_rule": _future_block_rule(headline_allowed),
        "honest_verdict": _honest_verdict(
            status=status,
            headline_allowed=headline_allowed,
            nonforgetting_rate=float(nonforgetting_rate) if nonforgetting_rate is not None else None,
        ),
        "source_artifacts": [
            "results/experiment_1395_fr11_self_learning_v5.json",
            f"results/{v6.EXP1432_FILE}",
            f"results/{v7.EXP1446_FILE}",
            f"results/{v7.OUTPUT_FILE}",
            "results/experiment_1459_self_learning_nonheadline_lineage_decision.json",
            f"results/{temporal.DATASET_FILE}",
            "data/fover_corpus.jsonl",
        ],
        "policy_reused": {
            "source_experiment": 1447,
            "policy_name": policy.policy_name,
            "fresh_secl_confidence_threshold": policy.fresh_secl_confidence_threshold,
            "replay_nonforgetting_secl_confidence_threshold": (
                policy.replay_nonforgetting_secl_confidence_threshold
            ),
            "dvi_incorrect_threshold": policy.dvi_incorrect_threshold,
        },
        "fresh_case_set": {
            "bounded": True,
            "fover_rows_scored": len(fover_load.cases),
            "temporal_rows_scored": len(temporal_load.cases),
            "fover_counts": fover_load.counts,
            "temporal_counts": temporal_load.counts,
        },
        "ltlzinc_benchmark_role": (
            "supporting benchmark feed only; not a standalone headline self-learning claim"
        ),
        "memory_updates": memory_updates,
        "session_memory_model_id": SESSION_MEMORY_MODEL_ID,
        "session_memory_path": persistence.path,
        "nonforgetting_threshold": MIN_NONFORGETTING_RATE,
        "pivot_prerequisites_ready": pivot_ready,
        "commands_run": list(commands_run or []),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1471-5/6/7: enforce required fields and headline gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")  # pragma: no cover
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return

    baseline = int(artifact["baseline_fresh_verified_sample_count"])
    fresh_count = int(artifact["fresh_verified_sample_count"])
    delta = int(artifact["self_learning_delta_overall"])
    promoted = int(artifact["new_promoted_count"])
    entries = int(artifact["memory_entries_added"])
    nonforgetting = artifact["nonforgetting_rate"]
    nonforgetting_value = float(nonforgetting) if nonforgetting is not None else 0.0
    expected_headline = delta > 0 and promoted >= 1 and nonforgetting_value >= MIN_NONFORGETTING_RATE

    if fresh_count != baseline + delta:
        raise AssertionError("fresh_verified_sample_count must equal baseline plus delta")
    if promoted != entries:
        raise AssertionError("new_promoted_count must equal memory_entries_added")
    if delta != entries:
        raise AssertionError("self_learning_delta_overall must equal memory_entries_added")
    if bool(artifact["session_memory_updated"]) != (entries > 0):
        raise AssertionError("session_memory_updated must match persisted entries")
    if bool(artifact["headline_result_allowed"]) != expected_headline:
        raise AssertionError("headline gate must match delta/promoted/nonforgetting thresholds")
    if artifact["pivot_preserved"] and artifact["pivot_retired"]:
        raise AssertionError("pivot cannot be both preserved and retired")  # pragma: no cover
    if expected_headline:
        if not artifact["pivot_preserved"] or artifact["pivot_retired"]:
            raise AssertionError("headline pivot must be preserved and not retired")  # pragma: no cover
        if not artifact["self_learning_artifact_ready"]:
            raise AssertionError("headline artifact must be ready")  # pragma: no cover
    elif not artifact["pivot_retired"]:
        raise AssertionError("failed pivot gate must retire the pivot")
    if not expected_headline and not artifact.get("future_block_rule"):
        raise AssertionError("retired pivot requires a future-block rule")  # pragma: no cover


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1432_path: Path | str = DEFAULT_EXP1432_PATH,
    exp1446_path: Path | str = DEFAULT_EXP1446_PATH,
    exp1447_path: Path | str = DEFAULT_EXP1447_PATH,
    exp1459_path: Path | str = DEFAULT_EXP1459_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    temporal_dataset_path: Path | str = DEFAULT_TEMPORAL_DATASET_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    session_memory_dir: Path | str = DEFAULT_SESSION_MEMORY_DIR,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1471 end-to-end and write the final artifact."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1395_artifact=load_json(exp1395_path),
        exp1432_artifact=load_json(exp1432_path),
        exp1446_artifact=load_json(exp1446_path),
        exp1447_artifact=load_json(exp1447_path),
        exp1459_artifact=load_json(exp1459_path),
        fover_rows=v7.dvi.load_jsonl_rows(fover_path),
        temporal_cases=load_jsonl(temporal_dataset_path),
        session_memory_dir=session_memory_dir,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
