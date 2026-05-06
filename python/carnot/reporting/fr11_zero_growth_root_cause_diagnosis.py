"""Exp 1446 FR-11 zero-growth root-cause diagnosis.

This module does not rerun FR-11. It replays the already-finished Exp 1433
candidate path with the deployed Exp 1432 DVI v3 checkpoint, then explains why
the active verifier produced no positive SessionMemory update. The important
distinction is between replay nonforgetting calibration and fresh promotion:
Exp 1432 raised the SECL threshold just enough to preserve replay demotions,
and Exp 1433 reused that stricter replay threshold for fresh promotions.

Spec: REQ-LEARN-1446, SCENARIO-LEARN-1446.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.reporting import dvi_discriminative_verifier_training_v1 as dvi
from carnot.reporting import fr11_self_learning_v5 as fr11
from carnot.reporting import fr11_self_learning_v6_dvi_v3_gated as v6
from carnot.reporting import secl_discriminative_self_calibration as secl
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

EXP1395_FILE = v6.EXP1395_FILE
EXP1432_FILE = v6.EXP1432_FILE
EXP1433_FILE = v6.OUTPUT_FILE
OUTPUT_FILE = "experiment_1446_fr11_zero_growth_root_cause_diagnosis.json"

DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_EXP1432_PATH = DEFAULT_RESULTS_DIR / EXP1432_FILE
DEFAULT_EXP1433_PATH = DEFAULT_RESULTS_DIR / EXP1433_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1446_fr11_zero_growth_root_cause_diagnosis"
SCHEMA = "fr11_zero_growth_root_cause_diagnosis_v1"
RUN_DATE = "20260506"
REJECTION_CATEGORIES = (
    "no_candidates",
    "verifier_rejection",
    "dvi_threshold",
    "novelty_threshold",
    "duplicate_memory",
    "persistence_blocker",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "fr11_zero_growth_root_cause_identified",
    "candidate_supply_count",
    "candidate_rejection_reason_counts",
    "promotion_thresholds",
    "memory_update_policy",
    "recommended_v7_policy",
    "exact_rerun_forbidden",
    "commands_run",
    "honest_verdict",
)


def _zero_rejection_counts() -> dict[str, int]:
    return {category: 0 for category in REJECTION_CATEGORIES}


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
    """REQ-LEARN-1446-1: persist the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "fr11_zero_growth_root_cause_identified": False,
            "candidate_supply_count": 0,
            "candidate_rejection_reason_counts": _zero_rejection_counts(),
            "promotion_thresholds": {},
            "memory_update_policy": {},
            "recommended_v7_policy": {},
            "exact_rerun_forbidden": False,
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


def _candidate_generation_counts(
    fover_rows: Sequence[Mapping[str, Any]],
    *,
    exclude_case_ids: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    candidate_ids: Counter[str] = Counter()
    for row in fover_rows:
        case_id = str(row.get("question_id") or "")
        if not case_id:
            counts["missing_case_id"] += 1
        elif case_id in exclude_case_ids:
            counts["novelty_threshold"] += 1
        elif secl.row_is_correct(row) is None or not secl.row_text(row):
            counts["unusable_candidate"] += 1
        else:
            counts["candidate_supply"] += 1
            candidate_ids[case_id] += 1
    counts["duplicate_candidate_rows"] = sum(
        count - 1 for count in candidate_ids.values() if count > 1
    )
    counts["duplicate_candidate_ids"] = sum(1 for count in candidate_ids.values() if count > 1)
    return dict(counts)


def _score_candidate_rejections(
    cases: Sequence[dvi.DviCase],
    state: v6.DviV3CheckpointState,
) -> tuple[dict[str, int], dict[str, int], int]:
    rejection_counts = _zero_rejection_counts()
    detail: Counter[str] = Counter()
    expected_v7_promotions = 0
    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    for case in cases:
        incorrect_probability = dvi.predict_incorrect_probability(
            verifier,
            state.metric,
            state.bias,
            case.text,
        )
        dvi_predicts_incorrect = incorrect_probability >= float(state.dvi_incorrect_threshold)
        semantic_state = fr11.STATE_REPAIR_HINT if dvi_predicts_incorrect else fr11.STATE_SAT
        certificate_state = fr11.STATE_REPAIR_HINT if int(case.label) == 1 else fr11.STATE_SAT
        predicted_state_probability = (
            incorrect_probability if dvi_predicts_incorrect else 1.0 - incorrect_probability
        )
        secl_confidence = float(state.confidence_head.predict([predicted_state_probability])[0])

        if semantic_state != certificate_state:
            rejection_counts["verifier_rejection"] += 1
            detail["dvi_state_mismatch"] += 1
        elif secl_confidence < float(state.secl_confidence_threshold):
            rejection_counts["dvi_threshold"] += 1
            detail["secl_confidence_below_threshold"] += 1
        else:
            detail["exp1433_promotable"] += 1

        if semantic_state == certificate_state and secl_confidence >= float(
            fr11.SECL_CONFIDENCE_THRESHOLD
        ):
            expected_v7_promotions += 1
    return rejection_counts, dict(detail), expected_v7_promotions


def _promotion_thresholds(
    activation: v6.DviV3Activation,
) -> dict[str, float | None]:
    state = activation.state
    exp1433_secl = float(state.secl_confidence_threshold) if state is not None else None
    v5_secl = float(fr11.SECL_CONFIDENCE_THRESHOLD)
    return {
        "exp1433_dvi_incorrect_threshold": float(state.dvi_incorrect_threshold)
        if state is not None
        else None,
        "exp1433_secl_confidence_threshold": exp1433_secl,
        "v5_base_secl_confidence_threshold": v5_secl,
        "v7_fresh_secl_confidence_threshold": v5_secl,
        "v7_replay_nonforgetting_secl_confidence_threshold": exp1433_secl,
        "fresh_vs_replay_secl_threshold_delta": round(float(exp1433_secl - v5_secl), 9)
        if exp1433_secl is not None
        else None,
    }


def _memory_update_policy(
    exp1433_artifact: Mapping[str, Any],
    *,
    expected_v7_promotions: int,
) -> dict[str, Any]:
    memory_updates = exp1433_artifact.get("memory_updates", {})
    promoted = memory_updates.get("promoted", []) if isinstance(memory_updates, Mapping) else []
    demoted = memory_updates.get("demoted", []) if isinstance(memory_updates, Mapping) else []
    return {
        "exp1433_policy": (
            "session_memory_updated is true only when DVI v3 verification produces "
            "one or more promoted rows; demotions are audited but do not count as "
            "positive FR-11 growth."
        ),
        "exp1433_session_memory_updated": bool(exp1433_artifact.get("session_memory_updated")),
        "exp1433_promoted_memory_count": len(promoted) if isinstance(promoted, Sequence) else 0,
        "exp1433_demoted_memory_count": len(demoted) if isinstance(demoted, Sequence) else 0,
        "v7_policy_change": (
            "keep replay nonforgetting calibration for replay rows, but score fresh "
            "promotion rows with the base SECL threshold so a replay-preservation "
            "epsilon cannot zero all fresh true positives."
        ),
        "expected_v7_promotions_under_changed_threshold": int(expected_v7_promotions),
    }


def _recommended_v7_policy(
    *,
    activation: v6.DviV3Activation,
    expected_v7_promotions: int,
) -> dict[str, Any]:
    state = activation.state
    replay_threshold = (
        float(state.secl_confidence_threshold)
        if state is not None
        else fr11.SECL_CONFIDENCE_THRESHOLD
    )
    return {
        "policy_name": "fr11_v7_asymmetric_fresh_threshold",
        "minimum_change": (
            "Do not reuse Exp 1432's replay-calibrated SECL threshold as the fresh "
            "promotion threshold. Use the v5 base fresh threshold for fresh FoVer "
            "promotion while retaining the calibrated threshold for replay "
            "nonforgetting audits."
        ),
        "fresh_secl_confidence_threshold": float(fr11.SECL_CONFIDENCE_THRESHOLD),
        "replay_nonforgetting_secl_confidence_threshold": float(replay_threshold),
        "dvi_incorrect_threshold": float(state.dvi_incorrect_threshold)
        if state is not None
        else None,
        "expected_promotions_under_v7_policy": int(expected_v7_promotions),
        "changes_exp1433_policy": True,
        "exact_rerun_avoidance": (
            "promotion_thresholds_changed; replay_nonforgetting_threshold_kept_separate"
        ),
    }


def build_artifact(
    *,
    exp1395_artifact: Mapping[str, Any],
    exp1432_artifact: Mapping[str, Any],
    exp1433_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1446: build the terminal zero-growth diagnosis artifact."""

    activation = v6.activate_dvi_v3_checkpoint(exp1432_artifact)
    exclude_ids = v6.exp1395_promoted_case_ids(exp1395_artifact)
    generation_counts = _candidate_generation_counts(fover_rows, exclude_case_ids=exclude_ids)
    candidates = (
        v6.candidate_cases_from_fover(fover_rows, exclude_case_ids=exclude_ids)
        if activation.active and activation.state is not None
        else []
    )
    rejection_counts = _zero_rejection_counts()
    rejection_counts["novelty_threshold"] = int(generation_counts.get("novelty_threshold", 0))
    rejection_detail: dict[str, int] = {}
    expected_v7_promotions = 0
    if not activation.active:
        rejection_counts["persistence_blocker"] = 1
    elif not candidates:
        rejection_counts["no_candidates"] = 1
    else:
        scored_counts, rejection_detail, expected_v7_promotions = _score_candidate_rejections(
            candidates,
            activation.state,
        )
        for category in ("verifier_rejection", "dvi_threshold"):
            rejection_counts[category] = scored_counts[category]

    root_cause_identified = bool(
        activation.active
        and int(exp1433_artifact.get("self_learning_delta_overall") or 0) == 0
        and candidates
        and (rejection_counts["verifier_rejection"] > 0 or rejection_counts["dvi_threshold"] > 0)
    )
    exact_rerun_forbidden = bool(
        activation.active
        and bool(exp1433_artifact.get("dvi_v3_checkpoint_active", activation.active))
        and int(exp1433_artifact.get("self_learning_delta_overall") or 0) == 0
    )
    recommended = _recommended_v7_policy(
        activation=activation,
        expected_v7_promotions=expected_v7_promotions,
    )
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or datetime.now(tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "spec": ["REQ-LEARN-1446", "SCENARIO-LEARN-1446"],
        "source_artifacts": [
            f"results/{EXP1395_FILE}",
            f"results/{EXP1432_FILE}",
            f"results/{EXP1433_FILE}",
            "data/fover_corpus.jsonl",
        ],
        "fr11_zero_growth_root_cause_identified": root_cause_identified,
        "fr11_zero_growth_root_cause": (
            "Exp 1433 had candidate supply and an active DVI v3 checkpoint, but "
            "all candidates were demoted. Most candidates disagreed with the DVI "
            "state; the DVI-agreeing true positives sat at SECL confidence 0.5 "
            "and were rejected by the replay-calibrated 0.500001 threshold."
        )
        if root_cause_identified
        else "root_cause_not_identified",
        "candidate_supply_count": len(candidates),
        "candidate_generation_counts": generation_counts,
        "candidate_rejection_reason_counts": rejection_counts,
        "candidate_rejection_detail": rejection_detail,
        "promotion_thresholds": _promotion_thresholds(activation),
        "memory_update_policy": _memory_update_policy(
            exp1433_artifact,
            expected_v7_promotions=expected_v7_promotions,
        ),
        "recommended_v7_policy": recommended,
        "exact_rerun_forbidden": exact_rerun_forbidden,
        "commands_run": list(commands_run or []),
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": _honest_verdict(
            root_cause_identified=root_cause_identified,
            expected_v7_promotions=expected_v7_promotions,
            activation=activation,
        ),
    }
    validate_artifact(artifact)
    return artifact


def _honest_verdict(
    *,
    root_cause_identified: bool,
    expected_v7_promotions: int,
    activation: v6.DviV3Activation,
) -> str:
    if not activation.active:
        return "fr11_v6_zero_growth_diagnosis_blocked_dvi_v3_inactive"
    if root_cause_identified and expected_v7_promotions > 0:
        return "fr11_v6_zero_growth_root_cause_identified_asymmetric_fresh_threshold_v7_required"
    if root_cause_identified:
        return "fr11_v6_zero_growth_root_cause_identified_policy_change_required"
    return "fr11_v6_zero_growth_root_cause_not_identified"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1446-4/5: enforce required fields and no-exact-rerun gate."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return
    counts = artifact["candidate_rejection_reason_counts"]
    missing_counts = [category for category in REJECTION_CATEGORIES if category not in counts]
    if missing_counts:
        raise AssertionError(f"missing rejection count categories: {missing_counts}")
    if artifact["fr11_zero_growth_root_cause_identified"] and not artifact["exact_rerun_forbidden"]:
        raise AssertionError("exact_rerun_forbidden must be true when root cause is identified")
    recommended = artifact["recommended_v7_policy"]
    if artifact["exact_rerun_forbidden"] and not recommended.get("changes_exp1433_policy"):
        raise AssertionError("exact_rerun_forbidden requires changed v7 policy")
    if not isinstance(artifact["commands_run"], list):
        raise AssertionError("commands_run must be a list")  # pragma: no cover


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1432_path: Path | str = DEFAULT_EXP1432_PATH,
    exp1433_path: Path | str = DEFAULT_EXP1433_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the diagnosis and write the final artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_artifact(
        exp1395_artifact=load_json(exp1395_path),
        exp1432_artifact=load_json(exp1432_path),
        exp1433_artifact=load_json(exp1433_path),
        fover_rows=dvi.load_jsonl_rows(fover_path),
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=commands_run,
    )
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
