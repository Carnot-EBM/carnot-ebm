"""Tests for Exp 1251 NRGPT frozen-prefix evaluation v2.

Spec: REQ-KONA-024, SCENARIO-KONA-024.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.nrgpt_frozen_prefix_v2 import (
    build_artifact,
    classify_nonmonotonicity,
    run,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_req_kona_024_builds_required_type_b_artifact_from_exp1163() -> None:
    """REQ-KONA-024: the artifact records the required type-b framing fields."""

    artifact = build_artifact(
        {
            "schema": "carnot.phase3_nrgpt_energy_native_prototype.v1",
            "n_iters_monotone": False,
            "nrgpt_auroc_n1": 0.920929,
            "nrgpt_auroc_n3": 0.915784,
        },
        source_artifact_file="experiment_1163_nrgpt_energy_native_prototype.json",
    )

    assert artifact["source_experiment"] == "exp1163_nrgpt_energy_recurrence_prototype"
    assert artifact["nrgpt_auroc"] == 0.921
    assert artifact["nonmonotonicity_classification"] == "b_causal_context_shift"
    assert artifact["nonmonotonicity_characterized"] is True
    assert (
        artifact["honest_verdict"]
        == "nrgpt_nonmonotonicity_characterized_type_b_causal_context_shift"
    )
    assert "position-dependent" in artifact["nonmonotonicity_rationale"]
    assert "paper-v6 Section 4" in artifact["paper_v6_framing"]
    assert artifact["source_artifact_file"] == "experiment_1163_nrgpt_energy_native_prototype.json"


def test_scenario_kona_024_run_prefers_requested_source_name_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-KONA-024: running the workflow writes the required JSON."""

    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "experiment_1163_nrgpt_energy_recurrence_prototype.json",
        {"auroc": 0.921, "n_iters_monotone": False},
    )
    _write_json(
        results_dir / "experiment_1163_nrgpt_energy_native_prototype.json",
        {"auroc": 0.5, "n_iters_monotone": False},
    )
    out_path = results_dir / "experiment_1251_nrgpt_frozen_prefix_evaluation_v2.json"

    artifact = run(results_dir=results_dir, out_path=out_path)

    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["nrgpt_auroc"] == 0.921
    assert artifact["source_artifact_file"] == "experiment_1163_nrgpt_energy_recurrence_prototype.json"


def test_req_kona_024_classifies_explicit_nonconservative_first_token_traces() -> None:
    """REQ-KONA-024: non-monotone first-token-only traces classify as type c."""

    payload = {
        "first_token_energy_traces": [
            [1.0, 1.2, 1.1],
            [0.3, 0.4],
            [2.0, 2.1, 2.2],
        ],
    }

    assert classify_nonmonotonicity(payload) == "c_non_conservative_preconditioner"
    artifact = build_artifact(payload)
    assert artifact["nonmonotonicity_classification"] == "c_non_conservative_preconditioner"
    assert "path-dependent" in artifact["nonmonotonicity_rationale"]


def test_req_kona_024_missing_exp1163_artifact_fails_honestly(tmp_path: Path) -> None:
    """REQ-KONA-024: missing source evidence raises instead of fabricating input."""

    with pytest.raises(FileNotFoundError, match="Exp 1163 NRGPT source artifact"):
        run(results_dir=tmp_path / "results", out_path=tmp_path / "out.json")
