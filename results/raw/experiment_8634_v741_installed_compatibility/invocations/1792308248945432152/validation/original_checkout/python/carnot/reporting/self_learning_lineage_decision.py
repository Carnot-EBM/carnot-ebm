"""Exp 1459 self-learning lineage decision.

This module makes a decision about the research line, not a new training run.
The narrow question is whether Exp 1447's persisted verified memory growth is
strong enough to become a constrained headline pivot, despite the broader
self-learning lineage containing many useful but explicitly non-headline
replay and adapter artifacts.

Spec: REQ-LEARN-1459, SCENARIO-LEARN-1459, SCENARIO-LEARN-1460.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = "experiment_1459_self_learning_nonheadline_lineage_decision.json"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_EXP1433_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json"
)
DEFAULT_EXP1447_PATH = DEFAULT_RESULTS_DIR / "experiment_1447_fr11_v7_memory_policy_growth.json"
DEFAULT_EXP1449_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1449_ltlzinc_temporal_continual_learning_adapter.json"
)
DEFAULT_CLASSIFICATION_PATH = REPO_ROOT / "ops" / "experiment_signal_noise_classification.csv"
DEFAULT_DECISION_NOTE_PATH = (
    REPO_ROOT / "docs" / "research-notes" / "self_learning_lineage_decision.md"
)

EXPERIMENT = "1459_self_learning_nonheadline_lineage_decision"
SCHEMA = "self_learning_lineage_decision_v1"
RUN_DATE = "20260507"
NONFORGETTING_THRESHOLD = 0.99
PIVOT_VERDICT = "self_learning_headline_pivot_selected_exp1447_verified_growth_only"
RETIRE_VERDICT = "self_learning_lineage_retired_from_headline_internal_memory_policy_only"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "self_learning_artifacts_reviewed",
    "decision_note_path",
    "self_learning_headline_pivot_selected",
    "self_learning_lineage_retired",
    "exp1447_delta_overall",
    "nonforgetting_rate",
    "ltlzinc_benchmark_role",
    "next_allowed_experiment_shape",
    "honest_verdict",
)

DEFAULT_REVIEWED_ARTIFACTS: tuple[str, ...] = (
    "research-program.md#Continuous Self-Learning",
    "results/experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json",
    "results/experiment_1447_fr11_v7_memory_policy_growth.json",
    "results/experiment_1449_ltlzinc_temporal_continual_learning_adapter.json",
    "ops/experiment_signal_noise_classification.csv",
    "research-complete.yaml",
)

REQUIRED_PIVOT_METRICS: tuple[str, ...] = (
    "baseline_fresh_verified_sample_count",
    "fresh_verified_sample_count",
    "self_learning_delta_overall",
    "new_promoted_count",
    "memory_entries_added",
    "session_memory_updated",
    "nonforgetting_rate",
    "headline_result_allowed",
)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    """REQ-LEARN-1459-1: write the visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec": ["REQ-LEARN-1459", "SCENARIO-LEARN-1459", "SCENARIO-LEARN-1460"],
            "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
            "run_date": run_date,
            "started_at": _timestamp(),
            "status": "in_progress",
            "self_learning_artifacts_reviewed": [],
            "decision_note_path": None,
            "self_learning_headline_pivot_selected": False,
            "self_learning_lineage_retired": False,
            "exp1447_delta_overall": None,
            "nonforgetting_rate": None,
            "ltlzinc_benchmark_role": None,
            "next_allowed_experiment_shape": None,
            "honest_verdict": "in_progress",
        },
    )


def _row_text(row: Mapping[str, str]) -> str:
    return " ".join(str(value or "") for value in row.values()).lower()


def _is_self_learning_classification_row(row: Mapping[str, str]) -> bool:
    text = _row_text(row)
    has_self_learning_scope = any(
        token in text
        for token in ("self learning", "self-learning", "self_learning", "fr11", "continual")
    )
    has_nonheadline_signal = (
        "non_headline" in text
        or "not_headline" in text
        or "headline_result_allowed=false" in text
    )
    return has_self_learning_scope and has_nonheadline_signal


def parse_self_learning_classification_rows(path: Path | str) -> list[dict[str, str]]:
    """Extract the non-headline self-learning lineage rows from the CSV table."""

    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            normalized = {str(key): str(value or "") for key, value in row.items()}
            if _is_self_learning_classification_row(normalized):
                rows.append(normalized)
    return rows


def _classification_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for row in rows:
        summary.append(
            {
                "experiment_id": str(row.get("experiment_id") or ""),
                "path": str(row.get("path") or ""),
                "honest_verdict": str(row.get("honest_verdict") or ""),
                "headline_fields": str(row.get("headline_fields") or ""),
                "classification": str(row.get("classification") or ""),
            }
        )
    return summary


def _exp1447_supports_pivot(exp1447_artifact: Mapping[str, Any]) -> bool:
    delta = _to_int(exp1447_artifact.get("self_learning_delta_overall"))
    memory_entries = _to_int(exp1447_artifact.get("memory_entries_added"))
    promoted = _to_int(exp1447_artifact.get("new_promoted_count"))
    nonforgetting_rate = _to_float(exp1447_artifact.get("nonforgetting_rate"))
    persisted = (
        bool(exp1447_artifact.get("session_memory_updated"))
        and memory_entries == delta
        and promoted == delta
    )
    return (
        exp1447_artifact.get("status") == "complete"
        and delta > 0
        and nonforgetting_rate >= NONFORGETTING_THRESHOLD
        and persisted
    )


def _pivot_shape(exp1447_delta: int) -> dict[str, Any]:
    return {
        "allowed_count": 1,
        "scope": "exp1447_verified_memory_policy_growth_pivot",
        "allowed_future_experiment": (
            "Run one bounded follow-on that reuses the Exp 1447 DVI-v7 asymmetric "
            "fresh/replay threshold policy on fresh verified local rows. Exp 1449 "
            "temporal cases may be included only as extra verified benchmark feed."
        ),
        "required_metrics": list(REQUIRED_PIVOT_METRICS),
        "minimum_new_promotions": 1,
        "reference_exp1447_delta_overall": exp1447_delta,
        "nonforgetting_threshold": NONFORGETTING_THRESHOLD,
        "forbidden_shapes": [
            "replay-only memory refresh as a headline self-learning claim",
            "adapter-only LTLZinc dataset as a headline self-learning claim",
            "broad self-learning improves everything wording without persisted memory delta",
        ],
    }


def _retirement_shape() -> dict[str, Any]:
    return {
        "allowed_count": 0,
        "scope": "retired_from_headline_scope",
        "retained_internal_use": (
            "Self-learning mechanisms may continue as internal memory-policy engineering, "
            "but not as active headline research scope."
        ),
        "reopen_rule": (
            "Operator must provide a new root cause, non-replay evidence, persisted "
            "memory growth, and nonforgetting >= 0.99 before headline scope can reopen."
        ),
    }


def build_decision_artifact(
    *,
    exp1433_artifact: Mapping[str, Any],
    exp1447_artifact: Mapping[str, Any],
    exp1449_artifact: Mapping[str, Any],
    classification_rows: Sequence[Mapping[str, str]],
    decision_note_path: Path | str,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    artifacts_reviewed: Sequence[str] = DEFAULT_REVIEWED_ARTIFACTS,
) -> dict[str, Any]:
    """REQ-LEARN-1459-2/3/4: build the terminal lineage decision artifact."""

    exp1447_delta = _to_int(exp1447_artifact.get("self_learning_delta_overall"))
    nonforgetting_rate = _to_float(exp1447_artifact.get("nonforgetting_rate"))
    pivot_selected = _exp1447_supports_pivot(exp1447_artifact)
    lineage_retired = not pivot_selected
    next_shape = _pivot_shape(exp1447_delta) if pivot_selected else _retirement_shape()
    ltlzinc_role = (
        "supporting benchmark feed only; Exp 1449 supplies verified temporal SAT and "
        "REPAIR_HINT cases for later FR-11/DVI ingestion, not a standalone headline claim"
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1459", "SCENARIO-LEARN-1459", "SCENARIO-LEARN-1460"],
        "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete",
        "self_learning_artifacts_reviewed": list(artifacts_reviewed),
        "decision_note_path": str(decision_note_path),
        "self_learning_headline_pivot_selected": pivot_selected,
        "self_learning_lineage_retired": lineage_retired,
        "exp1447_delta_overall": exp1447_delta,
        "nonforgetting_rate": round(nonforgetting_rate, 6),
        "ltlzinc_benchmark_role": ltlzinc_role,
        "next_allowed_experiment_shape": next_shape,
        "honest_verdict": PIVOT_VERDICT if pivot_selected else RETIRE_VERDICT,
        "source_artifact_summaries": {
            "exp1433": {
                "honest_verdict": str(exp1433_artifact.get("honest_verdict") or ""),
                "headline_result_allowed": bool(exp1433_artifact.get("headline_result_allowed")),
                "self_learning_delta_overall": _to_int(
                    exp1433_artifact.get("self_learning_delta_overall")
                ),
                "nonforgetting_rate": _to_float(exp1433_artifact.get("nonforgetting_rate")),
            },
            "exp1447": {
                "honest_verdict": str(exp1447_artifact.get("honest_verdict") or ""),
                "headline_result_allowed": bool(exp1447_artifact.get("headline_result_allowed")),
                "fresh_verified_sample_count": _to_int(
                    exp1447_artifact.get("fresh_verified_sample_count")
                ),
                "baseline_fresh_verified_sample_count": _to_int(
                    exp1447_artifact.get("baseline_fresh_verified_sample_count")
                ),
                "memory_entries_added": _to_int(exp1447_artifact.get("memory_entries_added")),
                "new_promoted_count": _to_int(exp1447_artifact.get("new_promoted_count")),
                "session_memory_updated": bool(exp1447_artifact.get("session_memory_updated")),
            },
            "exp1449": {
                "honest_verdict": str(exp1449_artifact.get("honest_verdict") or ""),
                "ltlzinc_adapter_ready": bool(exp1449_artifact.get("ltlzinc_adapter_ready")),
                "temporal_cases_generated": _to_int(
                    exp1449_artifact.get("temporal_cases_generated")
                ),
                "accepted_case_count": _to_int(exp1449_artifact.get("accepted_case_count")),
                "rejected_case_count": _to_int(exp1449_artifact.get("rejected_case_count")),
            },
        },
        "nonheadline_lineage_evidence": _classification_summary(classification_rows),
        "training_run_launched": False,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1459-2/3/4: enforce the final artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return
    if artifact["status"] != "complete":
        raise AssertionError("Exp 1459 terminal artifact must be complete")  # pragma: no cover

    pivot = bool(artifact["self_learning_headline_pivot_selected"])
    retired = bool(artifact["self_learning_lineage_retired"])
    if pivot == retired:
        raise AssertionError("exactly one of pivot or retirement must be selected")  # pragma: no cover

    delta = _to_int(artifact["exp1447_delta_overall"])
    nonforgetting_rate = _to_float(artifact["nonforgetting_rate"])
    next_shape = artifact["next_allowed_experiment_shape"]
    if not isinstance(next_shape, Mapping):
        raise AssertionError("next_allowed_experiment_shape must be an object")  # pragma: no cover
    if pivot:
        if delta <= 0:
            raise AssertionError("headline pivot requires positive exp1447 growth")  # pragma: no cover
        if nonforgetting_rate < NONFORGETTING_THRESHOLD:
            raise AssertionError("headline pivot requires preserved nonforgetting")  # pragma: no cover
        if next_shape.get("allowed_count") != 1:
            raise AssertionError("headline pivot must define one allowed shape")  # pragma: no cover
    elif next_shape.get("scope") != "retired_from_headline_scope":
        raise AssertionError("retired lineage must publish retirement scope")  # pragma: no cover


def write_decision_note(artifact: Mapping[str, Any], path: Path | str) -> str:
    """REQ-LEARN-1459-5: write the Markdown evidence note for the decision."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    decision = (
        "Narrow headline pivot selected"
        if artifact["self_learning_headline_pivot_selected"]
        else "Retired from headline scope"
    )
    exp1433 = artifact["source_artifact_summaries"]["exp1433"]
    exp1447 = artifact["source_artifact_summaries"]["exp1447"]
    exp1449 = artifact["source_artifact_summaries"]["exp1449"]
    nonheadline_rows = artifact["nonheadline_lineage_evidence"]
    row_lines = "\n".join(
        f"- Exp {row['experiment_id']}: {row['honest_verdict']} ({row['classification']})"
        for row in nonheadline_rows
    )
    content = f"""# Self-Learning Lineage Decision

Run date: {artifact['run_date']}
Decision: {decision}
Honest verdict: {artifact['honest_verdict']}

## Evidence

- Continuous Self-Learning remains a core architectural goal in
  `research-program.md`, but headline wording needs persisted growth rather
  than replay-only improvement.
- Exp 1433 reported `{exp1433['honest_verdict']}` with
  `self_learning_delta_overall={exp1433['self_learning_delta_overall']}` and
  `headline_result_allowed={exp1433['headline_result_allowed']}`.
- Exp 1447 reported `{exp1447['honest_verdict']}` with
  `exp1447_delta_overall={artifact['exp1447_delta_overall']}`,
  `memory_entries_added={exp1447['memory_entries_added']}`, and
  `nonforgetting_rate={artifact['nonforgetting_rate']}`.
- Exp 1449 reported `{exp1449['honest_verdict']}` with
  `temporal_cases_generated={exp1449['temporal_cases_generated']}`. It is a
  supporting benchmark feed, not a standalone headline claim.

## Non-Headline Lineage Pattern

{row_lines}

## Next Allowed Shape

`next_allowed_experiment_shape` in the Exp 1459 artifact is the governing
boundary. Only the Exp 1447 verified memory-policy mechanism may advance as a
narrow headline pivot, and it must report the required metrics plus
`nonforgetting_rate >= {NONFORGETTING_THRESHOLD}`. Replay-only, adapter-only,
and broad self-learning claims remain outside headline scope.
"""
    destination.write_text(content, encoding="utf-8")
    return content


def run(
    *,
    exp1433_path: Path | str = DEFAULT_EXP1433_PATH,
    exp1447_path: Path | str = DEFAULT_EXP1447_PATH,
    exp1449_path: Path | str = DEFAULT_EXP1449_PATH,
    classification_path: Path | str = DEFAULT_CLASSIFICATION_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    decision_note_path: Path | str = DEFAULT_DECISION_NOTE_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run the Exp 1459 decision pass and write the final JSON and note."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    artifact = build_decision_artifact(
        exp1433_artifact=load_json(exp1433_path),
        exp1447_artifact=load_json(exp1447_path),
        exp1449_artifact=load_json(exp1449_path),
        classification_rows=parse_self_learning_classification_rows(classification_path),
        decision_note_path=decision_note_path,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
    )
    write_decision_note(artifact, decision_note_path)
    return _write_json(out_path, artifact)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
