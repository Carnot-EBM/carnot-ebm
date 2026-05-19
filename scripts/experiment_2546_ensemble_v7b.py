from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.group_conditional_calibration import run_group_conditional_calibration
from carnot.verify.nco_constraint import compute_nco_rejection_rate


SEEDS = (42, 123, 456, 789, 1337)
RANDOM_SEED = SEEDS[0]
ENSEMBLE_V6_BASELINE = 0.9750
ENSEMBLE_V7_REGRESSION = 0.9607
MIN_ADAPTIVE_CONFORMAL_AUROC = 0.970
RESTORE_HEADLINE_AUROC = 0.975
OUTPUT_FILENAME = "experiment_2546_ensemble_v7b.json"
GROUP_ORDER = ("A", "B", "C", "D")
GROUP_LABELS = {
    "A": "Group A (logprob)",
    "B": "Group B (semantic)",
    "C": "Group C (logic)",
    "D": "Group D (proof-path)",
}
GROUP_SCORE_SOURCES = {
    "A": (
        "experiment_2395_fregelogic.json:semantic_energy_score",
        "experiment_2450_laab_meta_scores.json:score",
        "experiment_2395_fregelogic.json:fregelogic_risk_score",
    ),
    "B": (
        "experiment_2435_tier0k_scores.json:score",
        "experiment_2436_tier0l_scores.json:score",
        "experiment_2449_tier0m_scores.json:score",
    ),
    "C": (
        "experiment_2437_logcons_z3_scores.json:score",
        "nco_rejection_rate(token_logprobs)",
        "experiment_2460_tier0n_scores.json:score",
    ),
    "D": ("Tier0rVerifier.score",),
}
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix required.",
    "ensemble_v7b_auroc": (
        "Primary metric - must be >= 0.970 to unblock adaptive conformal "
        "(exp2547). >= 0.975 to restore cite-safe headline."
    ),
    "ensemble_v7b_auroc_std": "Multi-seed std - required for paper-v6 cite-safe reporting.",
    "ensemble_v6_baseline": "Explicitly records 0.9750 baseline for regression detection.",
    "ensemble_v7_regression": (
        "Explicitly records 0.9607 regression from exp2521 so delta is unambiguous."
    ),
    "regression_resolved": "True if ensemble_v7b_auroc >= 0.975. Key success flag.",
    "tier0r_group_assignment": (
        "Must be 'Group D (proof-path)'. Documents that Group C was not modified."
    ),
    "n_seeds": "Must be >= 3 for meaningful std.",
    "preconditions_checked": "Records which resources were verified.",
    "duration_s": "Wall-clock measurement.",
    "random_seed": "Base random seed - set to 42.",
}


class Tier0rImportError(RuntimeError):
    """Raised when the proof-path verifier precondition is not satisfied."""


def robust_load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj, _idx = json.JSONDecoder().raw_decode(text.lstrip())
        if isinstance(obj, dict):
            return obj
        raise


def normalize_label(label: str) -> int:
    if label == "correct":
        return 0
    if label == "incorrect":
        return 1
    return int(label)


def get_scores(path: Path, key: str) -> list[float]:
    data = robust_load_json(path)
    if "scores" in data:
        scores = sorted(data["scores"], key=lambda item: item["idx"])
        return [float(item["score"]) for item in scores]
    if "per_entry_results" in data:
        return [float(item[key]) for item in data["per_entry_results"]]
    return []


def load_manifest_rows(results_dir: Path) -> list[dict[str, Any]]:
    manifest = results_dir / "live_sota_balanced_telemetry_manifest_1480.jsonl"
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").strip().split("\n")]


def _load_tier0r_verifier_class() -> type:
    try:
        from carnot.verify.tier0r_curry_howard import Tier0rVerifier
    except ImportError as exc:
        raise Tier0rImportError(str(exc)) from exc
    return Tier0rVerifier


def build_score_groups(results_dir: Path, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    Tier0rVerifier = _load_tier0r_verifier_class()

    A1 = get_scores(results_dir / "experiment_2395_fregelogic.json", "semantic_energy_score")
    A2 = get_scores(results_dir / "experiment_2450_laab_meta_scores.json", "score")
    A3 = get_scores(results_dir / "experiment_2395_fregelogic.json", "fregelogic_risk_score")

    B1 = get_scores(results_dir / "experiment_2435_tier0k_scores.json", "score")
    B2 = get_scores(results_dir / "experiment_2436_tier0l_scores.json", "score")
    B3 = get_scores(results_dir / "experiment_2449_tier0m_scores.json", "score")

    C1 = get_scores(results_dir / "experiment_2437_logcons_z3_scores.json", "score")
    C2 = [compute_nco_rejection_rate(row.get("token_logprobs", [])) for row in rows]
    C3 = get_scores(results_dir / "experiment_2460_tier0n_scores.json", "score")

    tier0r = Tier0rVerifier()
    D1 = [tier0r.score(row.get("response_text", "")) for row in rows]

    groups = {
        "A": np.column_stack([A1, A2, A3]).astype(np.float64),
        "B": np.column_stack([B1, B2, B3]).astype(np.float64),
        "C": np.column_stack([C1, C2, C3]).astype(np.float64),
        "D": np.column_stack([D1]).astype(np.float64),
    }
    _validate_group_lengths(groups, len(rows))
    return groups


def _validate_group_lengths(groups: dict[str, np.ndarray], n_rows: int) -> None:
    bad = {
        name: matrix.shape[0]
        for name, matrix in groups.items()
        if matrix.ndim != 2 or matrix.shape[0] != n_rows
    }
    if bad:
        raise ValueError(f"score group row-count mismatch: {bad}; expected {n_rows}")


def _required_score_files(results_dir: Path) -> tuple[Path, ...]:
    return (
        results_dir / "live_sota_balanced_telemetry_manifest_1480.jsonl",
        results_dir / "experiment_2395_fregelogic.json",
        results_dir / "experiment_2450_laab_meta_scores.json",
        results_dir / "experiment_2435_tier0k_scores.json",
        results_dir / "experiment_2436_tier0l_scores.json",
        results_dir / "experiment_2449_tier0m_scores.json",
        results_dir / "experiment_2437_logcons_z3_scores.json",
        results_dir / "experiment_2460_tier0n_scores.json",
    )


def _blocked_artifact(reason: str, duration_s: float, preconditions_checked: list[str]) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "ensemble_v7b_auroc": None,
        "ensemble_v7b_auroc_std": None,
        "ensemble_v6_baseline": ENSEMBLE_V6_BASELINE,
        "ensemble_v7_regression": ENSEMBLE_V7_REGRESSION,
        "regression_resolved": False,
        "tier0r_group_assignment": "Group D (proof-path)",
        "n_seeds": len(SEEDS),
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
    }


def _write_artifact(results_dir: Path, deliverable: dict[str, Any]) -> None:
    out_path = results_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(deliverable, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def run_experiment(results_dir: Path = Path("results"), write: bool = True) -> dict[str, Any]:
    start_time = time.time()
    results_dir = Path(results_dir)
    preconditions_checked = ["sklearn_importable"]

    if (results_dir / "experiment_2485_group_conformal_v5.json").exists():
        preconditions_checked.append("exp2485_deliverable_present")
    if (results_dir / "experiment_2498_auroc_adversarial_v2_group_cond.json").exists():
        preconditions_checked.append("exp2498_deliverable_present")
    if (results_dir / "experiment_2521_ensemble_v7.json").exists():
        preconditions_checked.append("exp2521_regression_deliverable_present")

    missing_files = [path.name for path in _required_score_files(results_dir) if not path.exists()]
    if missing_files:
        duration_s = time.time() - start_time
        deliverable = _blocked_artifact(
            f"blocked_missing_score_files: {','.join(missing_files)}",
            duration_s,
            preconditions_checked,
        )
        if write:
            _write_artifact(results_dir, deliverable)
        return deliverable
    preconditions_checked.append("score_files_present")

    try:
        _load_tier0r_verifier_class()
    except Tier0rImportError:
        duration_s = time.time() - start_time
        deliverable = _blocked_artifact(
            "blocked_tier0r_not_importable",
            duration_s,
            preconditions_checked,
        )
        if write:
            _write_artifact(results_dir, deliverable)
        return deliverable
    preconditions_checked.append("tier0r_imported")

    rows = load_manifest_rows(results_dir)
    labels = np.array([normalize_label(row["correctness_label"]) for row in rows], dtype=int)
    score_groups = build_score_groups(results_dir, rows)
    preconditions_checked.append("group_d_configured")

    seed_results, ensemble_v7b_auroc, ensemble_v7b_auroc_std = run_group_conditional_calibration(
        score_groups=score_groups,
        labels=labels,
        seeds=SEEDS,
        group_order=GROUP_ORDER,
    )

    regression_resolved = bool(ensemble_v7b_auroc >= RESTORE_HEADLINE_AUROC)
    adaptive_gate_passed = bool(ensemble_v7b_auroc >= MIN_ADAPTIVE_CONFORMAL_AUROC)
    if regression_resolved:
        honest_verdict = f"complete: {ensemble_v7b_auroc:.4f} regression resolved"
    elif adaptive_gate_passed:
        honest_verdict = f"complete: {ensemble_v7b_auroc:.4f} adaptive conformal unblocked"
    else:
        honest_verdict = f"terminal: {ensemble_v7b_auroc:.4f} below 0.970 acceptance gate"

    duration_s = time.time() - start_time
    deliverable = {
        "honest_verdict": honest_verdict,
        "ensemble_v7b_auroc": ensemble_v7b_auroc,
        "ensemble_v7b_auroc_std": ensemble_v7b_auroc_std,
        "ensemble_v6_baseline": ENSEMBLE_V6_BASELINE,
        "ensemble_v7_regression": ENSEMBLE_V7_REGRESSION,
        "regression_resolved": regression_resolved,
        "tier0r_group_assignment": "Group D (proof-path)",
        "n_seeds": len(SEEDS),
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "random_seed": RANDOM_SEED,
        "n_groups": len(GROUP_ORDER),
        "n_verifiers": int(sum(score_groups[name].shape[1] for name in GROUP_ORDER)),
        "group_order": list(GROUP_ORDER),
        "group_labels": GROUP_LABELS,
        "group_score_sources": GROUP_SCORE_SOURCES,
        "acceptance_gates": {
            "ensemble_v7b_auroc >= 0.970": adaptive_gate_passed,
        },
        "field_principles": FIELD_PRINCIPLES,
        "results_by_seed": seed_results,
    }

    if write:
        _write_artifact(results_dir, deliverable)
    return deliverable


if __name__ == "__main__":
    run_experiment()
