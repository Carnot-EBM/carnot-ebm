"""Tests for reusable aggregate-available-report-gaps capstone helper.

Spec refs: REQ-CAPSTONE-4308, SCENARIO-CAPSTONE-4308.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import capstone_aggregate_available as agg
from carnot.reporting import capstone_v397_4301 as v397


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"
EXP4291 = REPO / "results" / "experiment_4291_arcgen_cross_generator_nondegenerate.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_req_capstone_4308_spec_declares_aggregate_available_helper() -> None:
    """REQ-CAPSTONE-4308: OpenSpec declares the robust helper contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4308" in spec
    assert "SCENARIO-CAPSTONE-4308" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "4294_efficiency" in spec


def test_scenario_capstone_4308_missing_efficiency_does_not_zero_cross_generator() -> None:
    """SCENARIO-CAPSTONE-4308: missing exp4294 does not erase cross-generator."""

    artifacts = {
        "4291_cross_generator": _read_json(EXP4291),
        "4294_efficiency": None,
    }
    axes = [
        agg.AxisSpec(
            name="cross_generator",
            required_keys=("4291_cross_generator",),
            verdict_fn=lambda present: v397.cross_generator_read(
                present.get("4291_cross_generator"), False
            )["cross_generator_moat_closes"]
            is True,
        ),
        agg.AxisSpec(
            name="efficiency",
            required_keys=("4294_efficiency",),
            verdict_fn=lambda present: v397.efficiency_read(
                present.get("4294_efficiency"), False
            )["efficiency_pareto_hardened"]
            is True,
        ),
    ]

    report = agg.aggregate_available_report_gaps(
        artifacts,
        axes,
        artifact_experiment_ids={"4291_cross_generator": 4291, "4294_efficiency": 4294},
    )

    assert report["axes"]["cross_generator"]["verdict"] is True
    assert report["axes"]["cross_generator"]["missing_artifacts"] == []
    assert report["axes"]["efficiency"]["verdict"] is False
    assert report["axes"]["efficiency"]["missing_artifacts"] == [
        {"axis": "efficiency", "artifact_key": "4294_efficiency", "experiment_id": 4294}
    ]
    assert report["missing_upstream_artifacts"] == [
        {"axis": "efficiency", "artifact_key": "4294_efficiency", "experiment_id": 4294}
    ]
    assert report["available_artifact_keys"] == ["4291_cross_generator"]


def test_req_capstone_4308_flagged_artifacts_are_reported_not_imported() -> None:
    """REQ-CAPSTONE-4308: flagged inputs become per-axis gaps."""

    report = agg.aggregate_available_report_gaps(
        {"generation": {"flagged_adversarial": True, "diffusiongemma_guidance_moat": True}},
        [
            agg.AxisSpec(
                name="generation",
                required_keys=("generation",),
                verdict_fn=lambda present: bool(
                    present.get("generation", {}).get("diffusiongemma_guidance_moat")
                ),
            )
        ],
        artifact_experiment_ids={"generation": 4293},
    )

    assert report["axes"]["generation"]["verdict"] is False
    assert report["axes"]["generation"]["flagged_artifacts"] == [
        {
            "axis": "generation",
            "artifact_key": "generation",
            "experiment_id": 4293,
            "reason": "flagged_adversarial",
        }
    ]
    assert report["flagged_artifacts_excluded"][0]["artifact_key"] == "generation"
