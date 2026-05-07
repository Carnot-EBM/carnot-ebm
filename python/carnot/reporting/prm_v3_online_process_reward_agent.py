"""Exp 1448 PRM v3 online process-reward repair agent reporting.

This runner consumes local Exp 1429, Exp 1430, and Exp 1434 artifacts, loads the
PRM v2 checkpoint, scores repair candidates step by step, and writes a bounded
selection comparison.  It intentionally reports ties and false-acceptance
changes without turning a prototype candidate pool into a headline claim.

Spec: REQ-VERIFY-1448, SCENARIO-VERIFY-1448.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.pipeline.prm_v3_online_process_reward_agent import (
    StepTextScorer,
    bounded_pra_step_score,
    evaluate_online_process_reward_selection,
)
from carnot.reporting import process_reward_model_v1_fover_1508 as prm_v1


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

EXP1429_FILE = "experiment_1429_mcmc_constrained_repair_candidate_search.json"
EXP1430_FILE = "experiment_1430_prm_guided_repair_selector.json"
EXP1434_FILE = "experiment_1434_fover_prm_label_completion_v2.json"
OUTPUT_FILE = "experiment_1448_prm_v3_online_process_reward_agent.json"

DEFAULT_EXP1429_PATH = DEFAULT_RESULTS_DIR / EXP1429_FILE
DEFAULT_EXP1430_PATH = DEFAULT_RESULTS_DIR / EXP1430_FILE
DEFAULT_EXP1434_PATH = DEFAULT_RESULTS_DIR / EXP1434_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1448_prm_v3_online_process_reward_agent"
SCHEMA = "prm_v3_online_process_reward_agent_v1"
RUN_DATE = "20260506"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "pra_selector_ready",
    "prm_v2_labels_used",
    "traces_evaluated",
    "step_scores_generated",
    "selection_improvement_pp",
    "false_acceptance_rate_delta",
    "regression_against_prm_v1",
    "commands_run",
    "honest_verdict",
)

WriteObserver = Callable[[Path, Mapping[str, Any]], None]


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-1448: write the bootstrap artifact before loading sources."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "pra_selector_ready": False,
            "prm_v2_labels_used": False,
            "traces_evaluated": 0,
            "step_scores_generated": 0,
            "selection_improvement_pp": 0.0,
            "false_acceptance_rate_delta": 0.0,
            "regression_against_prm_v1": False,
            "commands_run": [],
            "honest_verdict": "in_progress",
        },
        write_observer=write_observer,
    )


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1429_path: Path | str = DEFAULT_EXP1429_PATH,
    exp1430_path: Path | str = DEFAULT_EXP1430_PATH,
    exp1434_path: Path | str = DEFAULT_EXP1434_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    commands_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run Exp 1448 and write a complete or blocked terminal artifact."""

    write_in_progress_artifact(
        output_path,
        project_root=project_root,
        run_date=run_date,
        write_observer=write_observer,
    )
    exp1429 = load_json(exp1429_path)
    exp1430 = load_json(exp1430_path)
    exp1434 = load_json(exp1434_path)
    candidate_pool = _candidate_pool(exp1429)
    scorer_metadata: dict[str, Any]
    try:
        scorer, scorer_metadata = _scorer_from_prmv2(exp1434)
    except (OSError, KeyError, ValueError):
        return _write_json(
            output_path,
            _blocked_artifact(
                project_root=project_root,
                run_date=run_date,
                commands_run=list(commands_run or []),
                source_statuses={
                    "exp1429_status": exp1429.get("status"),
                    "exp1430_status": exp1430.get("status"),
                    "exp1434_status": exp1434.get("status"),
                },
            ),
            write_observer=write_observer,
        )
    if not candidate_pool or not _prmv1_ready(exp1430):
        return _write_json(
            output_path,
            _blocked_artifact(
                project_root=project_root,
                run_date=run_date,
                commands_run=list(commands_run or []),
                source_statuses={
                    "exp1429_status": exp1429.get("status"),
                    "exp1430_status": exp1430.get("status"),
                    "exp1434_status": exp1434.get("status"),
                },
                honest_verdict="blocked_candidate_pool_or_prm_v1_comparison_unavailable",
            ),
            write_observer=write_observer,
        )

    aggregate = evaluate_online_process_reward_selection(
        candidate_pool,
        scorer,
        exp1430.get("case_selections") or [],
    )
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "status": "complete",
        "spec": ["REQ-VERIFY-1448", "SCENARIO-VERIFY-1448"],
        "source_artifacts": [
            f"results/{EXP1429_FILE}",
            f"results/{EXP1430_FILE}",
            f"results/{EXP1434_FILE}",
        ],
        "pra_selector_ready": True,
        "prm_v2_labels_used": True,
        "prm_v2_artifact_used": True,
        "prm_v2_checkpoint_path": scorer_metadata["prm_v2_checkpoint_path"],
        "prm_v2_reported_auroc": scorer_metadata["prm_v2_reported_auroc"],
        "cases_evaluated": aggregate["cases_evaluated"],
        "traces_evaluated": aggregate["traces_evaluated"],
        "step_scores_generated": aggregate["step_scores_generated"],
        "selector_auroc": aggregate["selector_auroc"],
        "raw_best_of_n_repair_success_rate": aggregate["raw_best_of_n_repair_success_rate"],
        "prm_v1_selected_repair_success_rate": aggregate["prm_v1_selected_repair_success_rate"],
        "prm_v3_selected_repair_success_rate": aggregate["prm_v3_selected_repair_success_rate"],
        "selection_improvement_pp": aggregate["selection_improvement_pp"],
        "prm_v3_vs_prm_v1_selection_delta_pp": aggregate["prm_v3_vs_prm_v1_selection_delta_pp"],
        "raw_best_of_n_false_acceptance_rate": aggregate["raw_best_of_n_false_acceptance_rate"],
        "prm_v1_false_acceptance_rate": aggregate["prm_v1_false_acceptance_rate"],
        "prm_v3_false_acceptance_rate": aggregate["prm_v3_false_acceptance_rate"],
        "false_acceptance_rate_delta": aggregate["false_acceptance_rate_delta"],
        "regression_against_prm_v1": aggregate["regression_against_prm_v1"],
        "exp1429_executor_runtime_mode": exp1429.get("executor_runtime_mode"),
        "exp1430_honest_verdict": exp1430.get("honest_verdict"),
        "exp1434_honest_verdict": exp1434.get("honest_verdict"),
        "case_selections": aggregate["case_selections"],
        "commands_run": list(commands_run or []),
        "honest_verdict": _complete_verdict(
            aggregate,
            exp1429=exp1429,
        ),
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact, write_observer=write_observer)


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL rows, ignoring blanks, malformed lines, and non-objects."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-VERIFY-1448: enforce required fields and terminal readiness."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["status"] == "complete":
        if artifact["pra_selector_ready"] is not True:
            raise AssertionError("complete PRM v3 artifact requires pra_selector_ready=true")
        if artifact["prm_v2_labels_used"] is not True:
            raise AssertionError("complete PRM v3 artifact requires prm_v2_labels_used=true")
    if artifact["status"] == "blocked" and artifact["pra_selector_ready"]:
        raise AssertionError("blocked PRM v3 artifact must not be ready")


def _scorer_from_prmv2(exp1434: Mapping[str, Any]) -> tuple[StepTextScorer, dict[str, Any]]:
    if exp1434.get("status") != "complete" or exp1434.get("prmv2_trained") is not True:
        raise ValueError("PRM v2 artifact is not a complete trained artifact")
    if exp1434.get("headline_label_coverage_ready") is not True:
        raise ValueError("PRM v2 label coverage is not headline-ready")
    checkpoint_path = Path(str(exp1434.get("checkpoint_path") or ""))
    if not checkpoint_path.exists():
        raise ValueError("PRM v2 checkpoint path does not exist")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        weights = np.asarray(checkpoint["weights"], dtype=np.float32)
        bias = float(np.asarray(checkpoint["bias"], dtype=np.float32).reshape(-1)[0])
        feature_dim = int(np.asarray(checkpoint["feature_dim"], dtype=np.int32).reshape(-1)[0])
    if feature_dim != prm_v1.FEATURE_DIM or weights.shape != (prm_v1.FEATURE_DIM,):
        raise ValueError("PRM v2 checkpoint feature_dim does not match PRM features")

    def scorer(text: str) -> float:
        label = prm_v1.StepLabel(case_id="exp1448_step", text=text, correct=True)
        logit = float(np.dot(weights, prm_v1.extract_features(label)) + bias)
        return bounded_pra_step_score(text, _sigmoid(logit))

    return scorer, {
        "prm_v2_checkpoint_path": str(checkpoint_path),
        "prm_v2_reported_auroc": _float_or_none(exp1434.get("prmv2_auroc")),
    }


def _write_json(
    path: Path | str,
    artifact: Mapping[str, Any],
    *,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if write_observer is not None:
        write_observer(destination, payload)
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def _candidate_pool(exp1429: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in exp1429.get("candidate_search_results") or []
        if isinstance(row, Mapping) and row.get("candidate_results")
    ]


def _prmv1_ready(exp1430: Mapping[str, Any]) -> bool:
    return exp1430.get("status") == "complete" and exp1430.get("prm_guided_selection_ready") is True


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    commands_run: Sequence[str],
    source_statuses: Mapping[str, Any],
    honest_verdict: str = "blocked_prm_v2_labels_or_checkpoint_unavailable",
) -> dict[str, Any]:
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "status": "blocked",
        "spec": ["REQ-VERIFY-1448", "SCENARIO-VERIFY-1448"],
        "source_artifacts": [
            f"results/{EXP1429_FILE}",
            f"results/{EXP1430_FILE}",
            f"results/{EXP1434_FILE}",
        ],
        "pra_selector_ready": False,
        "prm_v2_labels_used": False,
        "traces_evaluated": 0,
        "step_scores_generated": 0,
        "selection_improvement_pp": 0.0,
        "false_acceptance_rate_delta": 0.0,
        "regression_against_prm_v1": False,
        "source_statuses": dict(source_statuses),
        "commands_run": list(commands_run),
        "honest_verdict": honest_verdict,
    }
    validate_artifact(artifact)
    return artifact


def _complete_verdict(aggregate: Mapping[str, Any], *, exp1429: Mapping[str, Any]) -> str:
    if aggregate["regression_against_prm_v1"]:
        verdict = "complete_prmv3_regression_against_prm_v1_no_improvement_claim"
    elif float(aggregate["selection_improvement_pp"]) <= 0.0:
        verdict = "complete_prmv3_no_headline_improvement"
    elif float(aggregate["false_acceptance_rate_delta"]) > 0.0:
        verdict = "complete_prmv3_selection_improved_but_false_acceptance_worsened"
    else:
        verdict = "complete_prmv3_selection_improved_non_regressing"

    if float(aggregate["false_acceptance_rate_delta"]) < 0.0:
        verdict += "_false_acceptance_reduced"
    runtime_mode = str(exp1429.get("executor_runtime_mode") or "")
    if runtime_mode != "live_local_sota_gguf":
        verdict += "_prototype_candidate_pool_no_headline_claim"
    return verdict


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return round(parsed, 6) if math.isfinite(parsed) else None


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))
