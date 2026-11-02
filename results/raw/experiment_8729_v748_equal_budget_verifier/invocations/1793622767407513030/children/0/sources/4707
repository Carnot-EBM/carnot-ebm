"""Exp 1447 FR-11 v7 memory-policy growth measurement.

Exp 1433 had an active DVI v3 verifier but zero fresh promotions because it
reused Exp 1432's replay-calibrated SECL threshold for fresh memory writes.
Exp 1446 identified the bounded v7 change: keep the stricter replay threshold
for nonforgetting, but use the base fresh threshold for new local FoVer
promotions. This module applies that changed policy and counts growth only
after the promoted rows are saved through SessionMemory.

Spec: REQ-LEARN-1447, SCENARIO-LEARN-1447, SCENARIO-LEARN-1448.
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

import numpy as np

from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.case_memory import CaseMemory, CaseRecord
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.session_memory import SessionMemory
from carnot.reporting import dvi_discriminative_verifier_training_v1 as dvi
from carnot.reporting import dvi_v3_1508_fresh_cases as dvi_v3
from carnot.reporting import fr11_self_learning_v5 as fr11
from carnot.reporting import fr11_self_learning_v6_dvi_v3_gated as v6
from carnot.reporting import secl_discriminative_self_calibration as secl
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_SESSION_MEMORY_DIR = DEFAULT_RESULTS_DIR / "session_memory_1447"

EXP1395_FILE = v6.EXP1395_FILE
EXP1432_FILE = v6.EXP1432_FILE
EXP1433_FILE = v6.OUTPUT_FILE
EXP1446_FILE = "experiment_1446_fr11_zero_growth_root_cause_diagnosis.json"
OUTPUT_FILE = "experiment_1447_fr11_v7_memory_policy_growth.json"

DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_EXP1432_PATH = DEFAULT_RESULTS_DIR / EXP1432_FILE
DEFAULT_EXP1446_PATH = DEFAULT_RESULTS_DIR / EXP1446_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1447_fr11_v7_memory_policy_growth"
SCHEMA = "fr11_v7_memory_policy_growth_v1"
RUN_DATE = "20260506"
MIN_NONFORGETTING_RATE = 0.99
SESSION_MEMORY_MODEL_ID = "fr11_v7_dvi_v3_local_fover"
PROMOTED_PREFIX = "dvi_v7:fover:"

MODEL_SPECS: list[dict[str, str]] = [
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary_candidate_generator_or_judge",
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense_candidate_generator_or_judge",
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "fallback_candidate_generator_or_judge",
    },
]

POLICY_CHANGES_APPLIED: tuple[str, ...] = (
    "promotion_thresholds_changed_asymmetric_fresh_vs_replay",
    "candidate_source_changed_local_verified_fover_deduped",
    "memory_update_policy_changed_delta_requires_session_memory_persistence",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "model_specs",
    "policy_changes_applied",
    "live_sota_inference_used",
    "fresh_verified_sample_count",
    "new_promoted_count",
    "self_learning_delta_overall",
    "nonforgetting_rate",
    "session_memory_updated",
    "memory_entries_added",
    "retire_if_zero_growth_repeats",
    "commands_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class V7Policy:
    """Exp 1446 policy normalized into typed threshold values."""

    policy_name: str
    fresh_secl_confidence_threshold: float
    replay_nonforgetting_secl_confidence_threshold: float
    dvi_incorrect_threshold: float
    expected_promotions_under_v7_policy: int


@dataclass(frozen=True)
class CandidateLoad:
    """Fresh candidate rows plus audit counts for the changed v7 source policy."""

    cases: list[dvi.DviCase]
    counts: dict[str, int]


@dataclass(frozen=True)
class SessionPersistence:
    """Result of saving verified v7 promotions through SessionMemory."""

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


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1447-1: persist the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "model_specs": MODEL_SPECS,
            "policy_changes_applied": [],
            "live_sota_inference_used": False,
            "fresh_verified_sample_count": None,
            "new_promoted_count": None,
            "self_learning_delta_overall": None,
            "nonforgetting_rate": None,
            "session_memory_updated": None,
            "memory_entries_added": None,
            "retire_if_zero_growth_repeats": False,
            "commands_run": [],
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load one JSON artifact and reject non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")  # pragma: no cover
    return payload


def load_v7_policy(exp1446_artifact: Mapping[str, Any]) -> V7Policy:
    """REQ-LEARN-1447-2: load Exp 1446's changed v7 policy."""

    recommended = exp1446_artifact.get("recommended_v7_policy")
    if not isinstance(recommended, Mapping):
        raise ValueError("Exp 1446 recommended_v7_policy is required")
    if recommended.get("changes_exp1433_policy") is not True:
        raise ValueError("recommended_v7_policy.changes_exp1433_policy must be true")
    return V7Policy(
        policy_name=str(recommended.get("policy_name") or "fr11_v7_policy"),
        fresh_secl_confidence_threshold=float(recommended["fresh_secl_confidence_threshold"]),
        replay_nonforgetting_secl_confidence_threshold=float(
            recommended["replay_nonforgetting_secl_confidence_threshold"]
        ),
        dvi_incorrect_threshold=float(recommended["dvi_incorrect_threshold"]),
        expected_promotions_under_v7_policy=int(
            recommended.get("expected_promotions_under_v7_policy") or 0
        ),
    )


def fresh_candidates_from_local_fover(
    fover_rows: Sequence[Mapping[str, Any]],
    *,
    exclude_case_ids: set[str],
) -> CandidateLoad:
    """REQ-LEARN-1447-3: load de-duplicated local FoVer candidates."""

    counts: Counter[str] = Counter()
    cases: list[dvi.DviCase] = []
    seen: set[str] = set()
    for row in fover_rows:
        case_id = str(row.get("question_id") or "")
        if not case_id:
            counts["missing_case_id"] += 1
            continue
        if case_id in exclude_case_ids:
            counts["novelty_threshold"] += 1
            continue
        if case_id in seen:
            counts["duplicate_candidate_rows_skipped"] += 1
            continue

        is_correct = secl.row_is_correct(row)
        text = secl.row_text(row)
        if is_correct is None or not text:
            counts["unusable_candidate"] += 1
            continue

        seen.add(case_id)
        counts["candidate_supply"] += 1
        cases.append(
            dvi.DviCase(
                case_id=case_id,
                text=text,
                label=0 if is_correct else 1,
                source=str(row.get("source") or "fover"),
            )
        )
    return CandidateLoad(cases=cases, counts=dict(counts))


def verify_cases_with_v7_policy(
    cases: Sequence[dvi.DviCase],
    state: v6.DviV3CheckpointState,
    policy: V7Policy,
) -> list[dict[str, Any]]:
    """REQ-LEARN-1447-3: score fresh candidates with the v7 fresh threshold."""

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    variants: list[dict[str, Any]] = []
    for case in cases:
        incorrect_probability = dvi.predict_incorrect_probability(
            verifier,
            state.metric,
            state.bias,
            case.text,
        )
        dvi_predicts_incorrect = incorrect_probability >= policy.dvi_incorrect_threshold
        semantic_state = fr11.STATE_REPAIR_HINT if dvi_predicts_incorrect else fr11.STATE_SAT
        certificate_state = fr11.STATE_REPAIR_HINT if int(case.label) == 1 else fr11.STATE_SAT
        predicted_state_probability = (
            incorrect_probability if dvi_predicts_incorrect else 1.0 - incorrect_probability
        )
        secl_confidence = float(state.confidence_head.predict([predicted_state_probability])[0])
        state_matches = semantic_state == certificate_state
        threshold_passed = secl_confidence >= policy.fresh_secl_confidence_threshold
        memory_action = (
            fr11.POLICY_PROMOTE if state_matches and threshold_passed else fr11.POLICY_DEMOTE
        )
        rejection_reason = None
        if not state_matches:
            rejection_reason = "verifier_rejection"
        elif not threshold_passed:
            rejection_reason = "dvi_threshold"

        variants.append(
            {
                "variant_id": f"{PROMOTED_PREFIX}{case.case_id}",
                "source": "exp1447_v7_local_fover_asymmetric_fresh_threshold",
                "case_id": case.case_id,
                "memory_action": memory_action,
                "support": 1,
                "semantic_state": semantic_state,
                "certificate_state": certificate_state,
                "incorrect_probability": round(float(incorrect_probability), 8),
                "secl_confidence": round(float(secl_confidence), 8),
                "rejection_reason": rejection_reason,
            }
        )
    return variants


def apply_memory_updates(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect promoted and demoted v7 memory actions."""

    promoted = [
        str(variant["variant_id"])
        for variant in variants
        if variant.get("memory_action") == fr11.POLICY_PROMOTE
    ]
    demoted = [
        str(variant["variant_id"])
        for variant in variants
        if variant.get("memory_action") == fr11.POLICY_DEMOTE
    ]
    rejection_counts = Counter(
        str(variant.get("rejection_reason"))
        for variant in variants
        if variant.get("rejection_reason")
    )
    return {
        "promoted": promoted,
        "demoted": demoted,
        "promoted_memory_count": len(promoted),
        "demoted_memory_count": len(demoted),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
    }


def persist_promotions_to_session_memory(
    promoted_variants: Sequence[Mapping[str, Any]],
    *,
    cases_by_id: Mapping[str, dvi.DviCase],
    session_memory_dir: Path | str,
) -> SessionPersistence:
    """REQ-LEARN-1447-5: save promoted cases through SessionMemory."""

    if not promoted_variants:
        return SessionPersistence(entries_added=0, path=None)

    case_memory = CaseMemory()
    for variant in promoted_variants:
        case_id = str(variant["case_id"])
        case = cases_by_id[case_id]
        is_incorrect = int(case.label) == 1
        case_memory.record(
            CaseRecord.normalize(
                benchmark="fover",
                benchmark_slice=f"fover:{case_id}",
                model_name=SESSION_MEMORY_MODEL_ID,
                case_id=case_id,
                violation_types=(
                    ("fr11_v7_dvi_verified_incorrect",)
                    if is_incorrect
                    else ("fr11_v7_dvi_verified_correct",)
                ),
                prompt_text=case.text,
                description_texts=(
                    "Exp 1447 DVI v3 and v7 fresh threshold verified this row "
                    "for SessionMemory promotion.",
                ),
                baseline_success=not is_incorrect,
                repair_success=True,
                confidence=float(variant.get("secl_confidence") or 0.0),
                source_experiment=1447,
                source_artifact=f"results/{OUTPUT_FILE}",
                response_mode="local_fover_verified",
                verifier_path="dvi_v3_asymmetric_fresh_threshold",
            )
        )

    library = ConstraintTemplateLibrary()
    library.register_builtin_templates()
    tracker = PerModelFPTracker()
    session = SessionMemory(str(session_memory_dir), SESSION_MEMORY_MODEL_ID)
    session.save(case_memory, library, tracker)
    loaded = session.load()
    if loaded is None:
        return SessionPersistence(entries_added=0, path=str(session._state_path()))
    persisted_case_memory, _, _ = loaded
    return SessionPersistence(
        entries_added=len(persisted_case_memory.entries()),
        path=str(session._state_path()),
    )


def replay_cases_from_exp1395(
    exp1395_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
) -> list[dvi.DviCase]:
    """Build local replay cases for the v7 nonforgetting check."""

    raw_demoted = exp1395_artifact.get("memory_updates", {}).get("demoted", [])
    if not isinstance(raw_demoted, Sequence) or isinstance(raw_demoted, (str, bytes)):
        return []

    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for row in fover_rows:
        case_id = str(row.get("question_id") or "")
        if case_id and case_id not in rows_by_id:
            rows_by_id[case_id] = row

    replay: list[dvi.DviCase] = []
    seen: set[str] = set()
    for item in raw_demoted:
        case_id = str(item).split(":")[-1]
        if case_id in seen:
            continue
        seen.add(case_id)
        row = rows_by_id.get(case_id)
        if row is None:
            continue
        is_correct = secl.row_is_correct(row)
        text = secl.row_text(row)
        if is_correct is None or not text:
            continue
        replay.append(
            dvi.DviCase(
                case_id=case_id,
                text=text,
                label=0 if is_correct else 1,
                source="exp1395_demoted_replay_nonforgetting",
            )
        )
    return replay


def measure_v7_nonforgetting_rate(
    *,
    exp1432_artifact: Mapping[str, Any],
    exp1395_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    activation: v6.DviV3Activation,
    policy: V7Policy,
) -> float | None:
    """REQ-LEARN-1447-4: measure replay preservation with the replay threshold."""

    if not activation.active or activation.state is None:
        return None
    replay_cases = replay_cases_from_exp1395(exp1395_artifact, fover_rows)
    if not replay_cases:
        return float(exp1432_artifact.get("nonforgetting_rate") or 1.0)
    return dvi_v3.measure_nonforgetting_rate(
        replay_cases=replay_cases,
        metric=activation.state.metric,
        bias=activation.state.bias,
        confidence_head=activation.state.confidence_head,
        incorrect_threshold=policy.dvi_incorrect_threshold,
        secl_confidence_threshold=policy.replay_nonforgetting_secl_confidence_threshold,
    )


def build_artifact(
    *,
    exp1395_artifact: Mapping[str, Any],
    exp1432_artifact: Mapping[str, Any],
    exp1446_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    session_memory_dir: Path | str = DEFAULT_SESSION_MEMORY_DIR,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1447: build the terminal v7 memory-growth artifact."""

    policy = load_v7_policy(exp1446_artifact)
    activation = v6.activate_dvi_v3_checkpoint(exp1432_artifact)
    baseline_count = int(
        exp1395_artifact.get("fresh_verified_sample_count", v6.EXP1395_BASELINE_COUNT)
    )
    exclude_ids = v6.exp1395_promoted_case_ids(exp1395_artifact)
    candidate_load = fresh_candidates_from_local_fover(
        fover_rows,
        exclude_case_ids=exclude_ids,
    )

    variants: list[dict[str, Any]] = []
    if activation.active and activation.state is not None:
        variants = verify_cases_with_v7_policy(candidate_load.cases, activation.state, policy)

    memory_updates = apply_memory_updates(variants)
    memory_updates["duplicate_candidate_rows_skipped"] = int(
        candidate_load.counts.get("duplicate_candidate_rows_skipped", 0)
    )

    nonforgetting_rate = measure_v7_nonforgetting_rate(
        exp1432_artifact=exp1432_artifact,
        exp1395_artifact=exp1395_artifact,
        fover_rows=fover_rows,
        activation=activation,
        policy=policy,
    )
    nonforgetting_preserved = bool(
        activation.active
        and nonforgetting_rate is not None
        and float(nonforgetting_rate) >= MIN_NONFORGETTING_RATE
    )

    promoted_variants = [
        variant for variant in variants if variant.get("memory_action") == fr11.POLICY_PROMOTE
    ]
    persistence = (
        persist_promotions_to_session_memory(
            promoted_variants,
            cases_by_id={case.case_id: case for case in candidate_load.cases},
            session_memory_dir=session_memory_dir,
        )
        if nonforgetting_preserved
        else SessionPersistence(entries_added=0, path=None)
    )

    memory_entries_added = int(persistence.entries_added)
    new_promoted_count = memory_entries_added
    delta = memory_entries_added
    fresh_verified_count = baseline_count + delta
    status = "complete" if activation.active else "blocked"
    retire = status == "complete" and delta == 0
    next_root_cause = _next_root_cause(
        status=status,
        candidates=candidate_load.cases,
        memory_updates=memory_updates,
        promoted_variants=promoted_variants,
        memory_entries_added=memory_entries_added,
        nonforgetting_preserved=nonforgetting_preserved,
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or datetime.now(tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": status,
        "spec": ["REQ-LEARN-1447", "SCENARIO-LEARN-1447", "SCENARIO-LEARN-1448"],
        "source_artifacts": [
            f"results/{EXP1395_FILE}",
            f"results/{EXP1432_FILE}",
            f"results/{EXP1433_FILE}",
            f"results/{EXP1446_FILE}",
            "data/fover_corpus.jsonl",
        ],
        "model_specs": MODEL_SPECS,
        "policy_changes_applied": list(POLICY_CHANGES_APPLIED),
        "recommended_v7_policy_loaded": {
            "policy_name": policy.policy_name,
            "fresh_secl_confidence_threshold": policy.fresh_secl_confidence_threshold,
            "replay_nonforgetting_secl_confidence_threshold": (
                policy.replay_nonforgetting_secl_confidence_threshold
            ),
            "dvi_incorrect_threshold": policy.dvi_incorrect_threshold,
            "expected_promotions_under_v7_policy": (policy.expected_promotions_under_v7_policy),
        },
        "candidate_policy": {
            "candidate_source": "local_verified_fover_rows_not_exp1395_promoted_deduped",
            "duplicate_policy": "skip_duplicate_case_ids_before_session_memory_persistence",
            "promotion_threshold_source": "exp1446_recommended_v7_policy",
            "memory_delta_policy": "count_only_promotions_persisted_to_session_memory",
        },
        "live_sota_inference_used": False,
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "dvi_v3_artifact_used": f"results/{EXP1432_FILE}",
        "dvi_v3_checkpoint_active": bool(activation.active),
        "dvi_v3_checkpoint_path": activation.path,
        "dvi_v3_checkpoint_blocker": activation.blocker,
        "candidate_supply_count": len(candidate_load.cases),
        "candidate_generation_counts": candidate_load.counts,
        "memory_updates": memory_updates,
        "baseline_fresh_verified_sample_count": baseline_count,
        "fresh_verified_sample_count": fresh_verified_count,
        "new_promoted_count": new_promoted_count,
        "self_learning_delta_overall": delta,
        "nonforgetting_rate": round(float(nonforgetting_rate), 6)
        if nonforgetting_rate is not None
        else None,
        "nonforgetting_preserved": nonforgetting_preserved,
        "session_memory_updated": memory_entries_added > 0,
        "session_memory_model_id": SESSION_MEMORY_MODEL_ID,
        "session_memory_path": persistence.path,
        "memory_entries_added": memory_entries_added,
        "headline_result_allowed": delta > 0 and nonforgetting_preserved,
        "retire_if_zero_growth_repeats": retire,
        "next_root_cause": next_root_cause,
        "commands_run": list(commands_run or []),
        "honest_verdict": _honest_verdict(
            status=status,
            delta=delta,
            next_root_cause=next_root_cause,
            nonforgetting_preserved=nonforgetting_preserved,
        ),
    }
    validate_artifact(artifact)
    return artifact


def _next_root_cause(
    *,
    status: str,
    candidates: Sequence[dvi.DviCase],
    memory_updates: Mapping[str, Any],
    promoted_variants: Sequence[Mapping[str, Any]],
    memory_entries_added: int,
    nonforgetting_preserved: bool,
) -> str | None:
    if status != "complete":
        return "dvi_v3_inactive_or_unavailable"
    if not candidates:
        return "no_local_verified_candidates_after_novelty_filter"
    if not nonforgetting_preserved:
        return "replay_nonforgetting_not_preserved"
    if promoted_variants and memory_entries_added == 0:
        return "session_memory_persistence_failed"
    if memory_entries_added > 0:
        return None
    rejections = memory_updates.get("rejection_reason_counts", {})
    if isinstance(rejections, Mapping):
        if int(rejections.get("verifier_rejection") or 0) > 0:
            return "dvi_state_mismatch_after_v7_policy_change"
        if int(rejections.get("dvi_threshold") or 0) > 0:
            return "fresh_threshold_still_blocks_candidates"
    return "no_promotable_candidates_after_v7_policy_change"


def _honest_verdict(
    *,
    status: str,
    delta: int,
    next_root_cause: str | None,
    nonforgetting_preserved: bool,
) -> str:
    if status != "complete":
        return "fr11_v7_blocked_dvi_v3_inactive"
    if delta > 0 and nonforgetting_preserved:
        return "fr11_v7_positive_verified_growth_persisted_without_forgetting"
    if next_root_cause == "dvi_state_mismatch_after_v7_policy_change":
        return "fr11_v7_zero_growth_after_changed_policy_retire_dvi_state_mismatch"
    return f"fr11_v7_zero_growth_after_changed_policy_retire_{next_root_cause or 'unknown'}"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1447-5/6: enforce required fields and growth invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return

    memory_entries = int(artifact["memory_entries_added"])
    new_promoted = int(artifact["new_promoted_count"])
    delta = int(artifact["self_learning_delta_overall"])
    if delta > 0 and (memory_entries <= 0 or not artifact["session_memory_updated"]):
        raise AssertionError("positive growth requires persisted SessionMemory entries")
    if new_promoted != memory_entries:
        raise AssertionError("new_promoted_count must equal memory_entries_added")
    if delta != memory_entries:
        raise AssertionError("self_learning_delta_overall must equal memory_entries_added")
    if bool(artifact["session_memory_updated"]) != (memory_entries > 0):
        raise AssertionError("session_memory_updated must match persisted SessionMemory entries")
    if delta > 0:
        nonforgetting = artifact["nonforgetting_rate"]
        if nonforgetting is None or float(nonforgetting) < MIN_NONFORGETTING_RATE:
            raise AssertionError("positive growth requires preserved nonforgetting")
        if artifact["retire_if_zero_growth_repeats"]:
            raise AssertionError("positive growth cannot set retire_if_zero_growth_repeats")
    elif artifact["status"] == "complete" and not artifact["retire_if_zero_growth_repeats"]:
        raise AssertionError("zero growth after changed policy must set retirement gate")


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1432_path: Path | str = DEFAULT_EXP1432_PATH,
    exp1446_path: Path | str = DEFAULT_EXP1446_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    session_memory_dir: Path | str = DEFAULT_SESSION_MEMORY_DIR,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1447 end-to-end and write the final artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1395_artifact=load_json(exp1395_path),
        exp1432_artifact=load_json(exp1432_path),
        exp1446_artifact=load_json(exp1446_path),
        fover_rows=dvi.load_jsonl_rows(fover_path),
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
