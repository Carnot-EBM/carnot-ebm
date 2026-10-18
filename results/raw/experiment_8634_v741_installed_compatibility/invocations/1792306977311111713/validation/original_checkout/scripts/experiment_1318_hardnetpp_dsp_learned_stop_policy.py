"""Exp 1318: learned HardNet++/DSP stop policy held-out replay.

Spec refs: REQ-KONA-032, SCENARIO-KONA-032.
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

from carnot.phase3.learned_stop_policy import (  # noqa: E402
    build_stop_policy_examples,
    conservative_replay_continue_predictions,
    evaluate_learned_stop_policy,
    fit_transparent_stop_policy,
    split_stop_policy_examples,
)

RUN_DATE = "20260505"
RESULT_PATH = Path("results/experiment_1318_hardnetpp_dsp_learned_stop_policy.json")
EXP1305_PATH = Path("results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json")
EXP1291_PATH = Path("results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json")
EXP1292_PATH = Path("results/experiment_1292_dsp_feasibility_channel_diagnostic.json")
PROMPT_NAMED_PATHS = [
    Path("results/experiment_1291_hardnetpp_nonlinear_repair.json"),
    Path("results/experiment_1292_dsp_feasibility_channel.json"),
]
SPEC_REFS = ["REQ-KONA-032", "SCENARIO-KONA-032"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _missing_required_paths() -> list[str]:
    return [
        str(path)
        for path in (EXP1305_PATH, EXP1291_PATH, EXP1292_PATH)
        if not path.exists()
    ]


def _source_context(
    replay_payload: dict[str, Any],
    hardnetpp_payload: dict[str, Any],
    dsp_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_1305_status": replay_payload.get("status"),
        "experiment_1305_honest_verdict": replay_payload.get("honest_verdict"),
        "experiment_1291_status": hardnetpp_payload.get("status"),
        "experiment_1291_honest_verdict": hardnetpp_payload.get("honest_verdict"),
        "experiment_1292_status": dsp_payload.get("status"),
        "experiment_1292_honest_verdict": dsp_payload.get("honest_verdict"),
        "replay_sources": [
            str(EXP1305_PATH),
            str(EXP1291_PATH),
            str(EXP1292_PATH),
        ],
        "prompt_named_paths_missing": [
            str(path) for path in PROMPT_NAMED_PATHS if not path.exists()
        ],
    }


def _empty_split() -> dict[str, Any]:
    return {
        "split_rule": "case_id seed modulo holdout_modulus",
        "holdout_modulus": 5,
        "holdout_remainder": 0,
        "train_count": 0,
        "held_out_count": 0,
        "label_source": "repair_helped",
        "held_out_case_ids": [],
    }


def _blocked_artifact(
    replay_payload: dict[str, Any],
    hardnetpp_payload: dict[str, Any],
    dsp_payload: dict[str, Any],
    missing_paths: list[str],
) -> dict[str, Any]:
    return {
        "schema": "carnot.phase3.hardnetpp_dsp_learned_stop_policy.v1",
        "experiment": "1318_hardnetpp_dsp_learned_stop_policy",
        "experiment_id": 1318,
        "run_date": RUN_DATE,
        "status": "blocked",
        "spec_refs": SPEC_REFS,
        "source_context": _source_context(
            replay_payload,
            hardnetpp_payload,
            dsp_payload,
        ),
        "missing_paths": missing_paths,
        "learned_stop_policy_written": False,
        "generalization_split": _empty_split(),
        "stop_policy_precision": 0.0,
        "stop_policy_recall": 0.0,
        "hardnetpp_delta_over_replay_policy": 0.0,
        "dsp_feasibility_auc": 0.5,
        "honest_verdict": "blocked_missing_required_replay_artifacts",
    }


def _honest_verdict(report: dict[str, Any]) -> str:
    if (
        report["stop_policy_precision"] >= 0.95
        and report["stop_policy_recall"] >= 0.95
    ):
        return (
            "complete: learned policy matched the conservative replay policy on "
            "a deterministic held-out seed split, but this is still "
            "replay-distribution generalization and not a broad general stop rule"
        )
    return (
        "complete: learned policy artifact written, but held-out precision or "
        "recall is too weak to improve on the conservative replay policy"
    )


def build_artifact() -> dict[str, Any]:
    replay_payload = _load_json(EXP1305_PATH)
    hardnetpp_payload = _load_json(EXP1291_PATH)
    dsp_payload = _load_json(EXP1292_PATH)
    missing_paths = _missing_required_paths()
    if missing_paths or not dsp_payload.get("per_case"):
        return _blocked_artifact(
            replay_payload,
            hardnetpp_payload,
            dsp_payload,
            missing_paths,
        )

    threshold = float(dsp_payload.get("threshold", 0.5))
    help_energy_tolerance = float(dsp_payload.get("help_energy_tolerance", 1e-4))
    examples = build_stop_policy_examples(dsp_payload["per_case"])
    split = split_stop_policy_examples(examples, holdout_modulus=5)
    policy = fit_transparent_stop_policy(
        split.train,
        help_energy_tolerance=help_energy_tolerance,
    )
    baseline_predictions = conservative_replay_continue_predictions(
        split.held_out,
        threshold=threshold,
        help_energy_tolerance=help_energy_tolerance,
    )
    report = evaluate_learned_stop_policy(
        policy,
        split.held_out,
        baseline_continue_predictions=baseline_predictions,
    )

    return {
        "schema": "carnot.phase3.hardnetpp_dsp_learned_stop_policy.v1",
        "experiment": "1318_hardnetpp_dsp_learned_stop_policy",
        "experiment_id": 1318,
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": SPEC_REFS,
        "source_context": _source_context(
            replay_payload,
            hardnetpp_payload,
            dsp_payload,
        ),
        "learned_stop_policy_written": True,
        "generalization_split": split.to_metadata(),
        "policy": policy.to_metadata(),
        "stop_policy_precision": report["stop_policy_precision"],
        "stop_policy_recall": report["stop_policy_recall"],
        "hardnetpp_delta_over_replay_policy": report[
            "hardnetpp_delta_over_replay_policy"
        ],
        "dsp_feasibility_auc": report["dsp_feasibility_auc"],
        "replay_policy_comparison": report["replay_policy"],
        "held_out_report": report,
        "metric_definitions": {
            "stop_policy_precision": "precision for held-out stop predictions, with stop as the positive class",
            "stop_policy_recall": "recall for held-out stop predictions, with stop as the positive class",
            "dsp_feasibility_auc": "held-out AUROC of DSP channel_score for verifier-backed repair_helped labels",
            "hardnetpp_delta_over_replay_policy": "learned minus conservative replay continue-recall on held-out HardNet++ rows",
        },
        "honest_verdict": _honest_verdict(report),
    }


def main() -> dict[str, Any]:
    artifact = build_artifact()
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
