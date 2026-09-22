"""Tests for REQ-ARC-WMTE-7492 timed E6 re-reduction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/experiment_7492_e6_timed_cost_profile.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("experiment_7492", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(index: int, *, replaceable_s: float = 6.0) -> dict[str, Any]:
    phases = [
        {"decision_point": "candidate_selection", "wall_s": 3.0, "tokens": 0},
        {
            "decision_point": "induction_and_generation",
            "wall_s": replaceable_s,
            "tokens": 90,
        },
        {"decision_point": "world_model_verification", "wall_s": 2.0, "tokens": 10},
        {"decision_point": "supervisor", "wall_s": 1.0, "tokens": 0},
        {"decision_point": "planner", "wall_s": 3.0, "tokens": 0},
        {"decision_point": "environment", "wall_s": 5.0, "tokens": 0},
    ]
    return {
        "episode_id": f"timed-{index}",
        "game": f"g{index % 12:02d}",
        "complete": True,
        "current_model": True,
        "numeric_eligible": True,
        "temperature": "warm",
        "episode_wall_s": 20.0,
        "backend_usage_tokens": 100,
        "phases": phases,
        "reconciliation_passed": True,
        "token_join_passed": True,
        "cohort": "experiment_7491_timed",
    }


def test_7492_imports_the_7490_reducer() -> None:
    """REQ-ARC-WMTE-7492 reuses the registered E6 reducer."""

    module = _load()
    assert module.e6_base.EXPERIMENT_ID == 7490
    assert module.e6_base.run_positive_control()["passed"] is True


def test_numeric_profile_passes_all_publication_gates() -> None:
    """SCENARIO-ARC-WMTE-7492-GATES publishes only complete timed evidence."""

    module = _load()
    artifact = module.build_profile(
        timed_episodes=[_episode(index) for index in range(36)],
        earlier_episodes=[
            {
                "episode_id": f"earlier-{index}",
                "game": f"old{index % 10}",
                "complete": True,
                "current_model": True,
                "numeric_eligible": True,
                "cohort": "experiment_7490_compatible",
            }
            for index in range(26)
        ],
        cited_artifacts=[],
    )

    assert artifact["gate_results"]["numeric_share_published"] is True
    assert artifact["honest_verdict"] == "complete_numeric_profile_published"
    assert artifact["sample_size"]["complete_current_model_episodes"] == 62
    assert artifact["sample_size"]["fully_timed_complete_episodes"] == 36
    assert artifact["decision_point_profile"]["publication_mode"] == "numeric"
    generation = artifact["decision_point_profile"]["decision_points"]["induction_and_generation"]
    assert generation["wall_fraction"] == pytest.approx(0.3)
    assert generation["amdahl_ceiling"] == pytest.approx(1.0 / 0.7)
    assert generation["wall_fraction_interval_95"]["lower"] > 0
    assert artifact["positive_control"]["passed"] is True


def test_failed_reconciliation_suppresses_every_numeric_share() -> None:
    """REQ-ARC-WMTE-7492 fails closed on interval accounting."""

    module = _load()
    rows = [_episode(index) for index in range(36)]
    rows[7]["reconciliation_passed"] = False
    artifact = module.build_profile(timed_episodes=rows, earlier_episodes=[], cited_artifacts=[])

    assert artifact["gate_results"]["numeric_share_published"] is False
    assert artifact["honest_verdict"].startswith("complete_")
    for point in artifact["decision_point_profile"]["decision_points"].values():
        assert point["wall_fraction"] is None
        assert point["amdahl_ceiling"] is None


def test_under_five_percent_replaceable_work_stops_speed_claim() -> None:
    """SCENARIO-ARC-WMTE-7492-GATES applies the E6 kill rule."""

    module = _load()
    rows = []
    for index in range(36):
        row = _episode(index, replaceable_s=0.1)
        row["episode_wall_s"] = 250.0
        next(phase for phase in row["phases"] if phase["decision_point"] == "environment")[
            "wall_s"
        ] = 240.0
        rows.append(row)
    artifact = module.build_profile(timed_episodes=rows, earlier_episodes=[], cited_artifacts=[])

    assert artifact["kill_rule"]["replaceable_work_fraction"] < 0.05
    assert artifact["kill_rule"]["speed_claims_stopped"] is True
    assert artifact["gate_results"]["numeric_share_published"] is False


def test_writer_uses_only_explicit_tmp_path(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7492-TERMINAL isolates every test writer."""

    module = _load()
    output = tmp_path / "private" / "experiment_7492.json"
    payload = {
        "experiment_id": 7492,
        "honest_verdict": "complete_fixture",
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }
    module.write_artifact(output, payload)
    assert json.loads(output.read_text()) == payload
    assert list(tmp_path.rglob("*.json")) == [output]
