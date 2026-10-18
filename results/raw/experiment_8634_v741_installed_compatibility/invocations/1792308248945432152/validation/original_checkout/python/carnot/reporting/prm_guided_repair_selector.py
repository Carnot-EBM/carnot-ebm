"""Exp 1430 PRM-guided repair selector reporting.

This runner consumes Exp 1429's bounded repair candidate pool and Exp 1423's
trained PRM v1 checkpoint. It scores candidates before looking at semantic
acceptance labels, selects one candidate per case, and writes the measured
selected-candidate acceptance rate beside the raw best-of-N rate.

Spec: REQ-VERIFY-1430, SCENARIO-VERIFY-1430
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.pipeline.prm_guided_repair_selector import (
    CandidateTextScorer,
    evaluate_prm_guided_selection,
)
from carnot.reporting import process_reward_model_v1_fover_1508 as prmv1


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
EXP1429_FILE = "experiment_1429_mcmc_constrained_repair_candidate_search.json"
PRMV1_FILE = "experiment_1423_process_reward_model_v1_fover_1508.json"
OUTPUT_FILE = "experiment_1430_prm_guided_repair_selector.json"

DEFAULT_EXP1429_PATH = DEFAULT_RESULTS_DIR / EXP1429_FILE
DEFAULT_PRMV1_ARTIFACT_PATH = DEFAULT_RESULTS_DIR / PRMV1_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1430_prm_guided_repair_selector"
SCHEMA = "prm_guided_repair_selector_v1"
RUN_DATE = "20260506"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "prm_guided_selection_ready",
    "cases_evaluated",
    "selector_auroc",
    "raw_best_of_n_repair_success_rate",
    "selected_repair_success_rate",
    "selection_improvement_pp",
    "prmv1_artifact_used",
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
    """REQ-VERIFY-1430: write the bootstrap artifact before loading inputs."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "prm_guided_selection_ready": False,
            "cases_evaluated": 0,
            "selector_auroc": None,
            "raw_best_of_n_repair_success_rate": None,
            "selected_repair_success_rate": None,
            "selection_improvement_pp": None,
            "prmv1_artifact_used": False,
            "honest_verdict": "in_progress",
        },
        write_observer=write_observer,
    )


def run_experiment(
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    exp1429_path: Path | str = DEFAULT_EXP1429_PATH,
    prmv1_artifact_path: Path | str = DEFAULT_PRMV1_ARTIFACT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    tests_run: Sequence[str] | None = None,
    write_observer: WriteObserver | None = None,
) -> dict[str, Any]:
    """Run Exp 1430 and write a complete or blocked selector artifact."""

    write_in_progress_artifact(
        output_path,
        project_root=project_root,
        run_date=run_date,
        write_observer=write_observer,
    )
    exp1429 = load_json(exp1429_path)
    candidate_pool = _candidate_pool(exp1429)
    if not candidate_pool:
        return _write_json(
            output_path,
            _blocked_artifact(
                project_root=project_root,
                run_date=run_date,
                exp1429=exp1429,
                tests_run=tests_run or [],
            ),
            write_observer=write_observer,
        )

    scorer, scorer_metadata = _scorer_from_prmv1_or_proxy(prmv1_artifact_path)
    aggregate = evaluate_prm_guided_selection(candidate_pool, scorer)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "status": "complete",
        "spec": ["REQ-VERIFY-1430", "SCENARIO-VERIFY-1430"],
        "source_artifacts": [f"results/{EXP1429_FILE}", f"results/{PRMV1_FILE}"],
        "prm_guided_selection_ready": True,
        "cases_evaluated": aggregate["cases_evaluated"],
        "selector_auroc": aggregate["selector_auroc"],
        "raw_best_of_n_repair_success_rate": aggregate["raw_best_of_n_repair_success_rate"],
        "selected_repair_success_rate": aggregate["selected_repair_success_rate"],
        "selection_improvement_pp": aggregate["selection_improvement_pp"],
        "prmv1_artifact_used": scorer_metadata["prmv1_artifact_used"],
        "prmv1_artifact_path": scorer_metadata["prmv1_artifact_path"],
        "prmv1_checkpoint_path": scorer_metadata["prmv1_checkpoint_path"],
        "prmv1_reported_auroc": scorer_metadata["prmv1_reported_auroc"],
        "selector_scoring_mode": scorer_metadata["selector_scoring_mode"],
        "deterministic_proxy_used": scorer_metadata["deterministic_proxy_used"],
        "candidate_pool_artifact": str(exp1429_path),
        "candidate_pool_cases_available": len(candidate_pool),
        "exp1429_executor_runtime_mode": exp1429.get("executor_runtime_mode"),
        "case_selections": aggregate["case_selections"],
        "tests_run": list(tests_run or []),
        "honest_verdict": _complete_verdict(
            aggregate["selection_improvement_pp"],
            scorer_metadata=scorer_metadata,
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """SCENARIO-VERIFY-1430: enforce required fields and terminal invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["status"] == "complete" and not artifact["prm_guided_selection_ready"]:
        raise AssertionError("complete selector artifact requires prm_guided_selection_ready=true")
    if artifact["status"] == "blocked" and artifact["prm_guided_selection_ready"]:
        raise AssertionError("blocked selector artifact must not be ready")


def deterministic_proxy_score(text: str) -> float:
    """Non-headline fallback score used only when PRM v1 cannot be loaded."""

    lower = text.lower()
    positives = sum(marker in lower for marker in ("sat", "valid", "correct", "therefore"))
    negatives = sum(marker in lower for marker in ("repair_hint", "invalid", "wrong", "incorrect"))
    return max(0.0, min(1.0, 0.5 + 0.2 * (positives - negatives)))


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


def _blocked_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    exp1429: Mapping[str, Any],
    tests_run: Sequence[str],
) -> dict[str, Any]:
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "status": "blocked",
        "spec": ["REQ-VERIFY-1430", "SCENARIO-VERIFY-1430"],
        "source_artifacts": [f"results/{EXP1429_FILE}", f"results/{PRMV1_FILE}"],
        "prm_guided_selection_ready": False,
        "cases_evaluated": 0,
        "selector_auroc": None,
        "raw_best_of_n_repair_success_rate": _float_or_none(
            exp1429.get("repair_success_rate_best_of_n")
        ),
        "selected_repair_success_rate": None,
        "selection_improvement_pp": None,
        "prmv1_artifact_used": False,
        "selector_scoring_mode": None,
        "deterministic_proxy_used": False,
        "case_selections": [],
        "tests_run": list(tests_run),
        "honest_verdict": "blocked_exp1429_candidate_pool_unavailable",
    }
    validate_artifact(artifact)
    return artifact


def _scorer_from_prmv1_or_proxy(path: Path | str) -> tuple[CandidateTextScorer, dict[str, Any]]:
    try:
        return _load_prmv1_checkpoint_scorer(path)
    except (OSError, ValueError, KeyError):
        return deterministic_proxy_score, {
            "prmv1_artifact_used": False,
            "prmv1_artifact_path": str(path),
            "prmv1_checkpoint_path": None,
            "prmv1_reported_auroc": None,
            "selector_scoring_mode": "deterministic_proxy_non_headline",
            "deterministic_proxy_used": True,
        }


def _load_prmv1_checkpoint_scorer(path: Path | str) -> tuple[CandidateTextScorer, dict[str, Any]]:
    artifact = load_json(path)
    if artifact.get("status") != "complete" or artifact.get("prmv1_trained") is not True:
        raise ValueError("PRM v1 artifact is not a complete trained artifact")
    checkpoint_path = Path(str(artifact.get("checkpoint_path") or ""))
    if not checkpoint_path.exists():
        raise ValueError("PRM v1 checkpoint path does not exist")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        weights = np.asarray(checkpoint["weights"], dtype=np.float32)
        bias = float(np.asarray(checkpoint["bias"], dtype=np.float32).reshape(-1)[0])
        feature_dim = int(np.asarray(checkpoint["feature_dim"], dtype=np.int32).reshape(-1)[0])
    if feature_dim != prmv1.FEATURE_DIM or weights.shape != (prmv1.FEATURE_DIM,):
        raise ValueError("PRM v1 checkpoint feature_dim does not match Exp 1423 features")

    def scorer(text: str) -> float:
        label = prmv1.StepLabel(case_id="exp1430_candidate", text=text, correct=True)
        logit = float(np.dot(weights, prmv1.extract_features(label)) + bias)
        return _sigmoid(logit)

    return scorer, {
        "prmv1_artifact_used": True,
        "prmv1_artifact_path": str(path),
        "prmv1_checkpoint_path": str(checkpoint_path),
        "prmv1_reported_auroc": _float_or_none(artifact.get("prmv1_auroc")),
        "selector_scoring_mode": "prmv1_checkpoint",
        "deterministic_proxy_used": False,
    }


def _complete_verdict(
    improvement_pp: float,
    *,
    scorer_metadata: Mapping[str, Any],
    exp1429: Mapping[str, Any],
) -> str:
    trend = (
        "improved"
        if improvement_pp > 0
        else "regressed"
        if improvement_pp < 0
        else "no_improvement"
    )
    verdict = f"complete_prm_guided_selector_{trend}"
    runtime_mode = str(exp1429.get("executor_runtime_mode") or "")
    if scorer_metadata.get("deterministic_proxy_used"):
        verdict += "_proxy_non_headline"
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
