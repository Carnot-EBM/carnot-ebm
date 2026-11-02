"""Exp 1292: DSP feasibility-channel diagnostics for continuous repair.

Spec refs: REQ-KONA-030, SCENARIO-KONA-030.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.phase3.continuous_ebm import (  # noqa: E402
    FeasibilityChannelCase,
    evaluate_feasibility_channels,
)

RUN_DATE = "20260504"
RESULT_PATH = Path("results/experiment_1292_dsp_feasibility_channel_diagnostic.json")
EXP1275_PATH = Path("results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json")
EXP1276_PATH = Path("results/experiment_1276_snarenet_repair_layer_gated.json")
EXP1291_PATH = Path("results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _distortion_increment(row: dict[str, Any], before: str, after: str) -> float:
    before_value = float(row.get(f"{before}_distortion_from_initial", 0.0) or 0.0)
    after_value = float(row.get(f"{after}_distortion_from_initial", 0.0) or 0.0)
    return max(0.0, after_value - before_value)


def _case(
    *,
    case_id: str,
    cohort: str,
    row: dict[str, Any],
    before: str,
    after: str,
    distortion_delta: float,
) -> FeasibilityChannelCase:
    return FeasibilityChannelCase(
        case_id=case_id,
        cohort=cohort,
        before_violation_energy=float(row[f"{before}_violation_energy"]),
        before_violation_count=int(row[f"{before}_violation_count"]),
        after_violation_energy=float(row[f"{after}_violation_energy"]),
        after_violation_count=int(row[f"{after}_violation_count"]),
        distortion_delta=distortion_delta,
    )


def _linear_cases(
    exp1275: dict[str, Any],
    exp1276: dict[str, Any],
) -> list[FeasibilityChannelCase]:
    cases: list[FeasibilityChannelCase] = []

    for row in exp1275.get("per_seed", []):
        seed = int(row["seed"])
        cases.append(
            _case(
                case_id=f"exp1275_seed{seed}_raw_to_fsnet",
                cohort="exp1275_raw_to_fsnet",
                row=row,
                before="raw",
                after="feasibility",
                distortion_delta=float(row["distortion_l2"]),
            )
        )

    for row in exp1276.get("per_seed", []):
        seed = int(row["seed"])
        cases.append(
            _case(
                case_id=f"exp1276_seed{seed}_raw_to_fsnet",
                cohort="exp1276_raw_to_fsnet",
                row=row,
                before="raw",
                after="fsnet",
                distortion_delta=float(row["fsnet_distortion_from_initial"]),
            )
        )
        cases.append(
            _case(
                case_id=f"exp1276_seed{seed}_fsnet_to_adaptive",
                cohort="exp1276_fsnet_to_adaptive",
                row=row,
                before="fsnet",
                after="adaptive",
                distortion_delta=float(row["adaptive_distortion_from_fsnet"]),
            )
        )

    return cases


def _nonlinear_cases(exp1291: dict[str, Any]) -> list[FeasibilityChannelCase]:
    cases: list[FeasibilityChannelCase] = []
    for row in exp1291.get("per_seed", []):
        seed = int(row["seed"])
        cases.extend(
            [
                _case(
                    case_id=f"exp1291_seed{seed}_raw_to_fsnet",
                    cohort="exp1291_raw_to_fsnet_local_linear",
                    row=row,
                    before="raw",
                    after="fsnet",
                    distortion_delta=float(row["fsnet_distortion_from_initial"]),
                ),
                _case(
                    case_id=f"exp1291_seed{seed}_raw_to_snarenet",
                    cohort="exp1291_raw_to_snarenet_local_linear",
                    row=row,
                    before="raw",
                    after="snarenet",
                    distortion_delta=float(row["snarenet_distortion_from_initial"]),
                ),
                _case(
                    case_id=f"exp1291_seed{seed}_raw_to_hardnetpp",
                    cohort="exp1291_raw_to_hardnetpp",
                    row=row,
                    before="raw",
                    after="hardnetpp",
                    distortion_delta=float(row["hardnetpp_distortion_from_initial"]),
                ),
                _case(
                    case_id=f"exp1291_seed{seed}_fsnet_to_snarenet",
                    cohort="exp1291_fsnet_to_snarenet_local_linear",
                    row=row,
                    before="fsnet",
                    after="snarenet",
                    distortion_delta=_distortion_increment(row, "fsnet", "snarenet"),
                ),
                _case(
                    case_id=f"exp1291_seed{seed}_fsnet_to_hardnetpp",
                    cohort="exp1291_fsnet_to_hardnetpp",
                    row=row,
                    before="fsnet",
                    after="hardnetpp",
                    distortion_delta=_distortion_increment(row, "fsnet", "hardnetpp"),
                ),
                _case(
                    case_id=f"exp1291_seed{seed}_snarenet_to_hardnetpp",
                    cohort="exp1291_snarenet_to_hardnetpp",
                    row=row,
                    before="snarenet",
                    after="hardnetpp",
                    distortion_delta=_distortion_increment(row, "snarenet", "hardnetpp"),
                ),
            ]
        )
    return cases


def _blocked_artifact(source_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "carnot.phase3.dsp_feasibility_channel.v1",
        "experiment": "1292_dsp_feasibility_channel_diagnostic",
        "run_date": RUN_DATE,
        "status": "blocked",
        "spec_refs": ["REQ-KONA-030", "SCENARIO-KONA-030"],
        "source_context": source_context,
        "phi_local": 0.0,
        "Phi_global": 0.0,
        "feasibility_channel_auc": 0.5,
        "repair_help_prediction_accuracy": 0.0,
        "false_continue_rate": 0.0,
        "false_stop_rate": 0.0,
        "distortion_when_wrong": 0.0,
        "feasibility_channel_predictive": False,
        "recommended_repair_stop_policy": "blocked until Exp 1275 and Exp 1276 artifacts exist",
        "honest_verdict": "blocked_missing_required_repair_artifacts",
        "per_case": [],
    }


def _honest_verdict(report: dict[str, Any]) -> str:
    auc = float(report["feasibility_channel_auc"])
    accuracy = float(report["repair_help_prediction_accuracy"])
    false_continue_rate = float(report["false_continue_rate"])
    if auc >= 0.70 and accuracy >= 0.70 and false_continue_rate <= 0.35:
        return "feasibility_channel_predictive"
    if auc >= 0.60 and accuracy >= 0.60:
        return "feasibility_channel_predictive_marginal"
    return "feasibility_channel_not_predictive"


def build_artifact() -> dict[str, Any]:
    exp1275 = _load_json(EXP1275_PATH)
    exp1276 = _load_json(EXP1276_PATH)
    exp1291 = _load_json(EXP1291_PATH)
    source_context = {
        "experiment_1275_loaded": bool(exp1275),
        "experiment_1275_honest_verdict": exp1275.get("honest_verdict"),
        "experiment_1276_loaded": bool(exp1276),
        "experiment_1276_honest_verdict": exp1276.get("honest_verdict"),
        "experiment_1291_loaded": bool(exp1291),
        "experiment_1291_honest_verdict": exp1291.get("honest_verdict"),
    }
    if not exp1275 or not exp1276:
        return _blocked_artifact(source_context)

    cases = _linear_cases(exp1275, exp1276)
    if exp1291:
        cases.extend(_nonlinear_cases(exp1291))

    report = evaluate_feasibility_channels(cases)
    honest_verdict = _honest_verdict(report)
    recommended_policy = (
        "Continue only while phi_local and Phi_global are both above the repair "
        "threshold and hard violation energy/count remain nonzero; stop after "
        "hard feasibility is reached, and switch residual nonlinear cases to "
        "HardNet++ rather than adding more local-linear FSNet/SnareNet steps."
    )

    return {
        "schema": "carnot.phase3.dsp_feasibility_channel.v1",
        "experiment": "1292_dsp_feasibility_channel_diagnostic",
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": ["REQ-KONA-030", "SCENARIO-KONA-030"],
        "source_context": source_context,
        "threshold": 0.5,
        "help_energy_tolerance": 1e-4,
        "recommended_repair_stop_policy": recommended_policy,
        "honest_verdict": honest_verdict,
        **report,
    }


def main() -> dict[str, Any]:
    artifact = build_artifact()
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
