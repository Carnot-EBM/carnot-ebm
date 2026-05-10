#!/usr/bin/env python3
"""Experiment 1723: FourierCSP-boundary CIKAN verifier.

Spec: REQ-KAN-1723, SCENARIO-KAN-1723.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:  # pragma: no cover - import-time path guard
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.models.cikan_verifier import CIKAN
from carnot.pipeline.fouriercsp_extractor import FourierCSPExtractor, MultilinearPolynomial


DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1723_cikan.json"
REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "status",
    "experiment_id",
    "run_date",
    "spec_traces",
    "constraint",
    "model",
    "training",
    "metrics",
    "fixed_boundaries_preserved",
    "boundary_snapshot_before",
    "boundary_snapshot_after",
    "honest_verdict",
)


def _mock_generate(prompt: str) -> str:
    """Return a deterministic FourierCSP parse for the Exp 1723 toy constraint."""

    return '{"variables": ["X", "Y"], "expression": "X AND Y"}'


def extract_toy_constraint() -> MultilinearPolynomial:
    """Extract the `X AND Y` toy constraint through the FourierCSP interface."""

    previous_force_live = os.environ.get("CARNOT_FORCE_LIVE")
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    try:
        extracted = FourierCSPExtractor(generate_fn=_mock_generate).extract(
            "X and Y must both be true"
        )
    finally:
        if previous_force_live is None:
            os.environ.pop("CARNOT_FORCE_LIVE", None)
        else:
            os.environ["CARNOT_FORCE_LIVE"] = previous_force_live

    if extracted is None:
        raise RuntimeError("FourierCSP mock extraction failed")
    return extracted


def build_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic Boolean AND dataset for CIKAN training."""

    xs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    ys = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return xs, ys


def run_experiment(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    run_date: str = "20260510",
    epochs: int = 40,
    tests_run: list[str] | None = None,
) -> dict[str, Any]:
    """Train CIKAN on the toy FourierCSP dataset and write the result artifact."""

    constraint = extract_toy_constraint()
    xs, ys = build_toy_dataset()
    model = CIKAN.from_fouriercsp(
        feature_names=constraint.variables,
        constraints=[constraint],
        boundary_penalty=4.0,
        learning_rate=0.2,
        seed=1723,
    )
    snapshot_before = model.boundary_snapshot()
    loss_history = model.fit(xs, ys, epochs=epochs)
    snapshot_after = model.boundary_snapshot()

    metrics = model.evaluate(xs, ys)
    satisfying_energy = float(model.energy([1.0, 1.0]))
    violating_energy = float(model.energy([1.0, 0.0]))
    metrics.update(
        {
            "satisfying_energy": satisfying_energy,
            "violating_energy": violating_energy,
            "boundary_energy_on_violation": float(model.boundary_energy([1.0, 0.0])),
        }
    )

    fixed_boundaries_preserved = snapshot_before == snapshot_after
    artifact = {
        "schema": "carnot.cikan.experiment_1723.v1",
        "status": "complete",
        "experiment_id": 1723,
        "run_date": run_date,
        "spec_traces": ["REQ-KAN-1723", "SCENARIO-KAN-1723"],
        "constraint": {
            "variables": list(constraint.variables),
            "expression": constraint.expression,
            "polynomial": constraint.polynomial,
            "source": "FourierCSPExtractor(mock_generate)",
        },
        "model": {
            "class": "CIKAN",
            "feature_names": list(model.feature_names),
            "n_fixed_boundaries": len(model.boundaries),
            "n_knots": model.n_knots,
            "boundary_penalty": model.boundary_penalty,
            "residual_trainable": True,
        },
        "training": {
            "dataset": "toy_boolean_and",
            "n_train": int(len(xs)),
            "epochs": int(epochs),
            "learning_rate": model.learning_rate,
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "loss_decreased": bool(loss_history[-1] < loss_history[0]),
            "tests_run": list(tests_run or []),
        },
        "metrics": metrics,
        "fixed_boundaries_preserved": fixed_boundaries_preserved,
        "boundary_snapshot_before": snapshot_before,
        "boundary_snapshot_after": snapshot_after,
        "honest_verdict": (
            "complete: FourierCSP constraint compiled into fixed CIKAN boundary "
            "and preserved through toy residual training"
        ),
    }
    validate_artifact(artifact)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 1723 artifact fields and gates."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["schema"] == "carnot.cikan.experiment_1723.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1723
    assert artifact["spec_traces"] == ["REQ-KAN-1723", "SCENARIO-KAN-1723"]
    assert artifact["fixed_boundaries_preserved"] is True, "fixed_boundaries_preserved"
    metrics = artifact["metrics"]
    assert metrics["accuracy"] >= 1.0, "accuracy"
    assert metrics["energy_gap"] > 0.0, "energy gap"
    assert metrics["violating_energy"] > metrics["satisfying_energy"], "energy ordering"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--run-date", default="20260510")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args(argv)

    artifact = run_experiment(
        output_path=args.output,
        run_date=args.run_date,
        epochs=args.epochs,
    )
    print(f"wrote={args.output} status={artifact['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    raise SystemExit(main())
