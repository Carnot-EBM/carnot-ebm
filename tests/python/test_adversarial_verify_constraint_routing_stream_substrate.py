"""Regression tests for the Exp6790 CPU exact chronological substrate.

Spec refs: REQ-VERIFY-5933 and SCENARIO-VERIFY-5933-AGGREGATION-QUOTED-MARKERS.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import adversarial_verify as av


SUBSTRATE = "CPU exact chronological decision fixture, no LLM"


def test_req_verify_5933_constraint_routing_substrate_has_deterministic_floor() -> None:
    """REQ-VERIFY-5933 classifies the required Exp6790 value as deterministic CPU work."""

    floor = av.duration_floor_for_artifact({"inference_substrate": SUBSTRATE})
    assert floor == {
        "substrate": SUBSTRATE,
        "min_duration_s": av.DETERMINISTIC_VERIFIER_MIN_DURATION_S,
        "reason": "deterministic_verifier",
    }


def test_scenario_verify_5933_exp6790_artifact_has_no_substrate_warning(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5933 keeps a real CPU fixture above its small duration floor."""

    path = tmp_path / "artifact.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "experiment_6790_constraint_routing_substrate_control",
                "honest_verdict": "complete: deterministic CPU fixture control",
                "inference_substrate": SUBSTRATE,
                "duration_s": 0.01,
                "random_seed": 6790000,
                "reproducibility_checksum": "sha256:control",
            }
        ),
        encoding="utf-8",
    )
    kinds = {flag["kind"] for flag in av.verify_artifact(path)["flags"]}
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in kinds
    assert "DURATION_TOO_SHORT" not in kinds
