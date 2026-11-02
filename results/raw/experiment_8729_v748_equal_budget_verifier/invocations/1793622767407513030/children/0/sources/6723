"""Exp 1305: HardNet++/DSP feasibility stop policy replay.

Spec refs: REQ-KONA-031, SCENARIO-KONA-031.
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

from carnot.phase3.feasibility_stop_policy import evaluate_stop_policy  # noqa: E402

RUN_DATE = "20260505"
RESULT_PATH = Path("results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json")
EXP1291_PATH = Path("results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json")
EXP1292_PATH = Path("results/experiment_1292_dsp_feasibility_channel_diagnostic.json")
SPEC_REFS = ["REQ-KONA-031", "SCENARIO-KONA-031"]
KAN_PWA_ABSTRACTION_NOTE = (
    "arXiv 2602.06737 replaces KAN units with piecewise-affine abstractions, "
    "carries local/global approximation error bounds into MILP verification, "
    "and uses dynamic programming plus knapsack allocation to control the piece "
    "budget. For Carnot, residual nonlinear local-linear repair cases should be "
    "queued for KAN/PWA bounded abstraction only after HardNet++ leaves hard "
    "violations or the action-family guard cannot pick a certifying repair operator."
)
POLICY_RULES = [
    "Stop immediately when hard violation count is zero and hard violation energy is within tolerance.",
    "Continue repair only when hard violations remain and the DSP feasibility-channel score is at least the configured threshold.",
    "Treat repeated nonlinear local-linear FSNet/SnareNet steps as marginal-gain stops; route those residual cases to HardNet++ rather than spending another local-linear step.",
    "After HardNet++ reaches hard feasibility, stop even if latent task energy would prefer the misleading local basin.",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _source_context(
    hardnetpp_payload: dict[str, Any],
    dsp_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_1291_status": hardnetpp_payload.get("status"),
        "experiment_1291_honest_verdict": hardnetpp_payload.get("honest_verdict"),
        "experiment_1292_status": dsp_payload.get("status"),
        "experiment_1292_honest_verdict": dsp_payload.get("honest_verdict"),
        "kan_pwa_reference": "https://arxiv.org/abs/2602.06737",
    }


def _blocked_artifact(
    hardnetpp_payload: dict[str, Any],
    dsp_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "carnot.phase3.hardnetpp_dsp_feasibility_stop_policy.v1",
        "experiment": "1305_hardnetpp_dsp_feasibility_stop_policy",
        "experiment_id": 1305,
        "run_date": RUN_DATE,
        "status": "blocked",
        "spec_refs": SPEC_REFS,
        "source_context": _source_context(hardnetpp_payload, dsp_payload),
        "feasibility_stop_policy_written": False,
        "hardnetpp_delta_over_snarenet": 0.0,
        "feasibility_channel_auc": 0.5,
        "stop_policy_precision": 0.0,
        "residual_nonlinear_cases": [],
        "kan_pwa_abstraction_note": KAN_PWA_ABSTRACTION_NOTE,
        "honest_verdict": "blocked_missing_required_replay_artifacts",
        "policy_rules": POLICY_RULES,
        "policy_threshold": 0.5,
        "help_energy_tolerance": 1e-4,
        "benchmark_replay": {
            "replay_sources": [
                str(EXP1291_PATH),
                str(EXP1292_PATH),
            ],
            "candidate_transitions": 0,
            "baseline_dsp_continue_precision": 0.0,
            "conservative_continue_recommendations": 0,
            "conservative_stop_recommendations": 0,
            "true_continue_recommendations": 0,
            "false_continue_recommendations": 0,
            "policy_stop_accuracy": 0.0,
        },
    }


def _complete_verdict(stop_policy_precision: float, feasibility_channel_auc: float) -> str:
    if stop_policy_precision >= 0.95 and feasibility_channel_auc >= 0.60:
        return (
            "complete: conservative replay policy is useful as an operator gate, "
            "but DSP feasibility is still marginal and this is not a learned "
            "general stop rule"
        )
    return (
        "complete: stop policy artifact written, but replay precision or "
        "feasibility-channel signal is too weak for an operator gate"
    )


def build_artifact() -> dict[str, Any]:
    hardnetpp_payload = _load_json(EXP1291_PATH)
    dsp_payload = _load_json(EXP1292_PATH)
    if not hardnetpp_payload or not dsp_payload or not dsp_payload.get("per_case"):
        return _blocked_artifact(hardnetpp_payload, dsp_payload)

    threshold = float(dsp_payload.get("threshold", 0.5))
    help_energy_tolerance = float(dsp_payload.get("help_energy_tolerance", 1e-4))
    report = evaluate_stop_policy(
        dsp_payload["per_case"],
        threshold=threshold,
        help_energy_tolerance=help_energy_tolerance,
    )
    stop_policy_precision = float(report["stop_policy_precision"])
    feasibility_channel_auc = float(dsp_payload["feasibility_channel_auc"])

    benchmark_replay = {
        "replay_sources": [
            str(EXP1291_PATH),
            str(EXP1292_PATH),
        ],
        "candidate_transitions": report["candidate_transitions"],
        "baseline_dsp_continue_precision": report["baseline_dsp_continue_precision"],
        "conservative_continue_recommendations": report[
            "conservative_continue_recommendations"
        ],
        "conservative_stop_recommendations": report[
            "conservative_stop_recommendations"
        ],
        "true_continue_recommendations": report["true_continue_recommendations"],
        "false_continue_recommendations": report["false_continue_recommendations"],
        "policy_stop_accuracy": report["policy_stop_accuracy"],
    }

    return {
        "schema": "carnot.phase3.hardnetpp_dsp_feasibility_stop_policy.v1",
        "experiment": "1305_hardnetpp_dsp_feasibility_stop_policy",
        "experiment_id": 1305,
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": SPEC_REFS,
        "source_context": _source_context(hardnetpp_payload, dsp_payload),
        "feasibility_stop_policy_written": True,
        "hardnetpp_delta_over_snarenet": float(
            hardnetpp_payload["hardnetpp_delta_over_snarenet"]
        ),
        "feasibility_channel_auc": feasibility_channel_auc,
        "stop_policy_precision": stop_policy_precision,
        "residual_nonlinear_cases": report["residual_nonlinear_cases"],
        "kan_pwa_abstraction_note": KAN_PWA_ABSTRACTION_NOTE,
        "honest_verdict": _complete_verdict(
            stop_policy_precision,
            feasibility_channel_auc,
        ),
        "policy_rules": POLICY_RULES,
        "policy_threshold": threshold,
        "help_energy_tolerance": help_energy_tolerance,
        "benchmark_replay": benchmark_replay,
    }


def main() -> dict[str, Any]:
    artifact = build_artifact()
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
