"""Exp 1434 FoVer PRM label completion v2.

PRM v1 had enough local labels to train, but it missed promoted traces whose
IDs were created by Exp 1395's duplicate-row normalization.  Those IDs are not
new annotations: they are ordinal names such as ``gsm8k_3280_1`` for the second
local FoVer row with raw ``question_id=gsm8k_3280``.  This module replays that
normalization, recovers only rows that can be mapped back to local evidence,
writes a blocker ledger for anything still missing, and retrains the same
CPU-only feature classifier as PRM v2.

Spec: REQ-VERIFY-1434, SCENARIO-VERIFY-1434.
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

from carnot.reporting import fr11_self_learning_v5 as fr11_v5
from carnot.reporting import process_reward_model_v1_fover_1508 as prm_v1


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "research"
DEFAULT_MODELS_DIR = REPO_ROOT / "python" / "carnot" / "models"

EXP1395_FILE = "experiment_1395_fr11_self_learning_v5.json"
EXP1397_FILE = "experiment_1397_fullscale_pipeline_v2_200cases.json"
EXP1423_FILE = "experiment_1423_process_reward_model_v1_fover_1508.json"
OUTPUT_FILE = "experiment_1434_fover_prm_label_completion_v2.json"
LEDGER_FILE = "prm_missing_label_ledger_v2.md"

DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_EXP1397_PATH = DEFAULT_RESULTS_DIR / EXP1397_FILE
DEFAULT_EXP1423_PATH = DEFAULT_RESULTS_DIR / EXP1423_FILE
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_STEP_PRM_PATH = REPO_ROOT / "data" / "step_level_prm_training.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_LEDGER_PATH = DEFAULT_DOCS_DIR / LEDGER_FILE
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODELS_DIR / "prmv2_fover_1508_checkpoint.pt"

EXPERIMENT = "1434_fover_prm_label_completion_v2"
SCHEMA = "fover_prm_label_completion_v2"
RUN_DATE = "20260506"
ORDINAL_REPLAY_LABEL_SOURCE = "exp1395_normalized_fover_ordinal_replay"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "missing_labels_before",
    "missing_labels_filled",
    "missing_labels_remaining",
    "label_blocker_ledger_path",
    "training_traces_used",
    "prmv2_trained",
    "prmv2_auroc",
    "prmv2_precision",
    "prmv2_recall",
    "headline_label_coverage_ready",
    "honest_verdict",
)


@dataclass(frozen=True)
class LabelRecovery:
    """Recovered local labels plus the exact blockers for unrecovered traces."""

    recovered_labels: list[prm_v1.StepLabel]
    blockers: list[dict[str, str]]


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
    """REQ-VERIFY-1434: write the visible bootstrap artifact before recovery."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "missing_labels_before": 0,
            "missing_labels_filled": 0,
            "missing_labels_remaining": 0,
            "label_blocker_ledger_path": str(DEFAULT_LEDGER_PATH),
            "training_traces_used": 0,
            "prmv2_trained": False,
            "prmv2_auroc": None,
            "prmv2_precision": None,
            "prmv2_recall": None,
            "headline_label_coverage_ready": False,
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def missing_trace_ids(
    exp1395_artifact: Mapping[str, Any],
    labels: Sequence[prm_v1.StepLabel],
    *,
    expected_promoted_count: int | None = prm_v1.FRESH_VERIFIED_CASE_COUNT,
) -> list[str]:
    """Return promoted Exp 1395 trace IDs with no local PRM v1 label."""

    labeled_trace_ids = {label.case_id for label in labels}
    return [
        case_id
        for case_id in prm_v1.promoted_case_ids(
            exp1395_artifact,
            expected_count=expected_promoted_count,
        )
        if case_id not in labeled_trace_ids
    ]


def recover_with_ordinal_replay(
    missing_ids: Sequence[str],
    fover_rows: Sequence[Mapping[str, Any]],
) -> LabelRecovery:
    """SCENARIO-VERIFY-1434: replay Exp 1395 duplicate-ID normalization."""

    normalized_cases = {case.case_id: case for case in fr11_v5.normalize_fover_cases(fover_rows)}
    recovered: list[prm_v1.StepLabel] = []
    blockers: list[dict[str, str]] = []
    for case_id in missing_ids:
        case = normalized_cases.get(case_id)
        if case is None:
            blockers.append(
                {
                    "case_id": str(case_id),
                    "blocker": "no_local_ordinal_replay_source_row",
                    "recovery_scope": "local_recovery_scope",
                }
            )
        else:
            recovered.append(
                prm_v1.StepLabel(
                    case_id=case.case_id,
                    text=" ".join(case.response.split()),
                    correct=not case.is_incorrect,
                    label_source=ORDINAL_REPLAY_LABEL_SOURCE,
                    trace_source=case.source,
                    prefix_fraction=1.0,
                )
            )
    return LabelRecovery(recovered_labels=recovered, blockers=blockers)


def write_label_blocker_ledger(
    path: Path | str,
    *,
    missing_ids: Sequence[str],
    recovered_labels: Sequence[prm_v1.StepLabel],
    blockers: Sequence[Mapping[str, str]],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> Path:
    """Write the human-readable ledger of recovered and unrecovered labels."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_counts = Counter(label.trace_source for label in recovered_labels)
    lines = [
        "# PRM Missing Label Ledger V2",
        "",
        f"- Spec: REQ-VERIFY-1434 / SCENARIO-VERIFY-1434",
        f"- Project root: `{project_root}`",
        f"- Run date: `{run_date}`",
        f"- Missing labels before replay: {len(missing_ids)}",
        f"- Missing labels filled: {len(recovered_labels)}",
        f"- Missing labels remaining: {len(blockers)}",
        "",
        "## Recovery Summary",
        "",
    ]
    if recovered_labels:
        lines.extend(
            [
                "| trace_id | source_case_id | label_source | trace_source | label |",
                "|---|---|---|---|---|",
            ]
        )
        for label in recovered_labels:
            lines.append(
                "| "
                f"{label.case_id} | {_source_case_id(label.case_id)} | "
                f"{label.label_source} | {label.trace_source} | "
                f"{'correct' if label.correct else 'incorrect'} |"
            )
    else:
        lines.append("No labels were recovered.")
    lines.extend(["", "## Recovered Counts By Trace Source", ""])
    for source, count in sorted(source_counts.items()):
        lines.append(f"- `{source}`: {count}")
    if not source_counts:
        lines.append("- none")
    lines.extend(["", "## Unrecovered Labels", ""])
    if blockers:
        lines.extend(["| case_id | blocker | recovery_scope |", "|---|---|---|"])
        for blocker in blockers:
            lines.append(
                "| "
                f"{blocker.get('case_id', '')} | "
                f"{blocker.get('blocker', '')} | "
                f"{blocker.get('recovery_scope', '')} |"
            )
    else:
        lines.append("No unrecovered labels remain.")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def headline_ready(missing_remaining: int, blockers: Sequence[Mapping[str, str]]) -> bool:
    """Gate headline coverage on full recovery or out-of-scope residual blockers."""

    if int(missing_remaining) == 0:
        return True
    return len(blockers) == int(missing_remaining) and all(
        blocker.get("recovery_scope") == "outside_local_recovery_scope" for blocker in blockers
    )


def build_artifact(
    *,
    exp1423_artifact: Mapping[str, Any],
    labels: Sequence[prm_v1.StepLabel],
    missing_ids: Sequence[str],
    recovery: LabelRecovery,
    training_result: prm_v1.TrainingResult,
    ledger_path: Path | str,
    started_at: str,
    duration_s: float,
    tests_run: Sequence[str],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the terminal Exp 1434 artifact and enforce schema invariants."""

    trained = bool(training_result.trained)
    missing_remaining = len(recovery.blockers)
    headline = headline_ready(missing_remaining, recovery.blockers)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete" if trained else "blocked",
        "spec": ["REQ-VERIFY-1434", "SCENARIO-VERIFY-1434"],
        "source_artifacts": [
            f"results/{EXP1395_FILE}",
            f"results/{EXP1423_FILE}",
            f"results/{EXP1397_FILE}",
            "data/fover_corpus.jsonl",
            "data/step_level_prm_training.jsonl",
        ],
        "exp1423_reported_missing_labels": int(exp1423_artifact.get("missing_trace_labels", 0)),
        "exp1423_training_traces_used": int(exp1423_artifact.get("training_traces_used", 0)),
        "missing_labels_before": len(missing_ids),
        "missing_labels_filled": len(recovery.recovered_labels),
        "missing_labels_remaining": missing_remaining,
        "label_blocker_ledger_path": str(ledger_path),
        "step_labels_available": len(labels),
        "training_traces_used": len({label.case_id for label in labels}),
        "prmv2_trained": trained,
        "prmv2_auroc": _rounded_or_none(training_result.auroc),
        "prmv2_precision": _rounded_or_none(training_result.precision),
        "prmv2_recall": _rounded_or_none(training_result.recall),
        "checkpoint_path": training_result.checkpoint_path if trained else None,
        "train_step_labels_used": int(training_result.train_labels_used),
        "heldout_step_labels_used": int(training_result.heldout_labels_used),
        "training_loss_history": [
            round(float(loss), 6) for loss in training_result.loss_history
        ],
        "recovery_method": ORDINAL_REPLAY_LABEL_SOURCE,
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "tests_run": list(tests_run),
        "headline_label_coverage_ready": headline,
        "honest_verdict": _honest_verdict(
            trained=trained,
            headline=headline,
            missing_remaining=missing_remaining,
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-VERIFY-1434: enforce required fields and trained-checkpoint invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["status"] == "complete":
        if artifact["prmv2_trained"] is not True:
            raise AssertionError("complete PRM v2 artifact requires prmv2_trained=true")
        for field in ("prmv2_auroc", "prmv2_precision", "prmv2_recall"):
            if artifact[field] is None:
                raise AssertionError(f"complete PRM v2 artifact requires {field}")
        if not Path(str(artifact.get("checkpoint_path"))).exists():
            raise AssertionError("trained PRM v2 artifact requires an existing checkpoint path")
        if not Path(str(artifact["label_blocker_ledger_path"])).exists():
            raise AssertionError("PRM v2 artifact requires an existing label blocker ledger")
    if artifact["status"] == "blocked" and artifact.get("checkpoint_path") is not None:
        raise AssertionError("blocked PRM v2 artifacts must not expose a checkpoint path")


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1423_path: Path | str = DEFAULT_EXP1423_PATH,
    exp1397_path: Path | str = DEFAULT_EXP1397_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    step_prm_path: Path | str = DEFAULT_STEP_PRM_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    expected_promoted_count: int | None = prm_v1.FRESH_VERIFIED_CASE_COUNT,
    n_epochs: int = prm_v1.N_EPOCHS,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run local label recovery, ledgering, and PRM v2 retraining."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1395 = prm_v1.load_json(exp1395_path)
    exp1423 = prm_v1.load_json(exp1423_path)
    exp1397 = prm_v1.load_json(exp1397_path)
    fover_rows = prm_v1.load_jsonl_rows(fover_path)
    labels_v1, _coverage_v1 = prm_v1.collect_promoted_step_labels(
        exp1395,
        fover_rows=fover_rows,
        step_prm_rows=prm_v1.load_jsonl_rows(step_prm_path),
        exp1397_artifact=exp1397,
        expected_promoted_count=expected_promoted_count,
    )
    missing_ids = missing_trace_ids(
        exp1395,
        labels_v1,
        expected_promoted_count=expected_promoted_count,
    )
    recovery = recover_with_ordinal_replay(missing_ids, fover_rows)
    all_labels = [*labels_v1, *recovery.recovered_labels]
    write_label_blocker_ledger(
        ledger_path,
        missing_ids=missing_ids,
        recovered_labels=recovery.recovered_labels,
        blockers=recovery.blockers,
        project_root=project_root,
        run_date=run_date,
    )
    training_result = prm_v1.train_and_evaluate(
        all_labels,
        checkpoint_path=checkpoint_path,
        n_epochs=n_epochs,
    )
    artifact = build_artifact(
        exp1423_artifact=exp1423,
        labels=all_labels,
        missing_ids=missing_ids,
        recovery=recovery,
        training_result=training_result,
        ledger_path=ledger_path,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        tests_run=list(tests_run or []),
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out_path, artifact)


def _source_case_id(case_id: str) -> str:
    stem, separator, ordinal = str(case_id).rpartition("_")
    return stem if separator and ordinal.isdigit() and stem else str(case_id)


def _honest_verdict(*, trained: bool, headline: bool, missing_remaining: int) -> str:
    if not trained:
        return "prmv2_blocked_insufficient_trainable_local_labels"
    if headline:
        return "prmv2_trained_all_promoted_traces_have_local_labels"
    return f"prmv2_trained_with_{missing_remaining}_promoted_traces_still_blocked"


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
