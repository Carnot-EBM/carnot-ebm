"""Tests for REQ-ARC-WMTE-7490 and its E6 coverage-only reducer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import ANY

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/experiment_7490_e6_live_loop_cost_profile.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("experiment_7490", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episode(
    *,
    episode_id: str = "episode-1",
    game: str = "g1",
    wall_s: float = 20.0,
    phases: list[dict[str, Any]] | None = None,
    backend_tokens: int = 150,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "game": game,
        "complete": True,
        "current_model": True,
        "temperature": "warm",
        "episode_wall_s": wall_s,
        "backend_usage_tokens": backend_tokens,
        "phases": phases
        or [
            {"decision_point": "candidate_selection", "wall_s": 1.0, "tokens": 0},
            {
                "decision_point": "induction_and_generation",
                "wall_s": 6.0,
                "tokens": 120,
            },
            {
                "decision_point": "world_model_verification",
                "wall_s": 2.0,
                "tokens": 10,
            },
            {"decision_point": "supervisor", "wall_s": 1.0, "tokens": 0},
            {"decision_point": "planner", "wall_s": 3.0, "tokens": 20},
            {"decision_point": "environment", "wall_s": 5.0, "tokens": 0},
        ],
    }


def test_spec_names_all_7490_scenarios() -> None:
    """REQ-ARC-WMTE-7490 has positive, exclusion, gate, kill, and terminal cases."""
    text = (ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md").read_text()
    section = text[text.index("REQ-ARC-WMTE-7490") : text.index("REQ-ARC-WMTE-7465")]
    for scenario in (
        "SCENARIO-ARC-WMTE-7490-POSITIVE-CONTROL",
        "SCENARIO-ARC-WMTE-7490-EXCLUSIONS",
        "SCENARIO-ARC-WMTE-7490-COVERAGE-GATE",
        "SCENARIO-ARC-WMTE-7490-KILL-RULE",
        "SCENARIO-ARC-WMTE-7490-TERMINAL",
    ):
        assert scenario in section


def test_terminal_enums_use_the_closed_adversarial_schema() -> None:
    """REQ-ARC-WMTE-7490 uses valid venue and null-verdict enum values."""
    module = _load_module()

    assert module.EXECUTION_VENUE == "host"
    assert module.COVERAGE_VERDICT_CLASS == "null"


def test_positive_control_recovers_both_injections_exactly() -> None:
    """SCENARIO-ARC-WMTE-7490-POSITIVE-CONTROL recovers delay and tokens."""
    module = _load_module()
    control = module.run_positive_control()

    assert control["passed"] is True
    assert control["injected_delay_s"] == control["recovered_injected_delay_s"]
    assert control["injected_token_count"] == control["recovered_injected_token_count"]
    assert control["wall_reconciled"] is True
    assert control["tokens_reconciled"] is True


def test_nonconcurrent_subphases_cannot_exceed_episode_time() -> None:
    """REQ-ARC-WMTE-7490 rejects real non-concurrent time over-accounting."""
    module = _load_module()
    bad = _episode(
        wall_s=10.0,
        phases=[
            {"decision_point": "candidate_selection", "wall_s": 6.0, "tokens": 0},
            {"decision_point": "environment", "wall_s": 5.0, "tokens": 0},
        ],
        backend_tokens=0,
    )

    with pytest.raises(ValueError, match="subphase time exceeds episode time"):
        module.reduce_episodes([bad], publish_numeric=True)


def test_explicit_concurrent_span_can_overlap_episode_time() -> None:
    """REQ-ARC-WMTE-7490 excludes marked concurrent work from exclusive sums."""
    module = _load_module()
    row = _episode(
        wall_s=10.0,
        phases=[
            {"decision_point": "candidate_selection", "wall_s": 6.0, "tokens": 0},
            {
                "decision_point": "world_model_verification",
                "wall_s": 5.0,
                "tokens": 0,
                "concurrent": True,
            },
        ],
        backend_tokens=0,
    )

    reduced = module.reduce_episodes([row], publish_numeric=True)

    assert reduced["reconciliation"]["wall_rows_reconciled"] == 1
    assert reduced["decision_points"]["candidate_selection"]["wall_fraction"] == 0.6
    assert reduced["decision_points"]["world_model_verification"]["concurrent"] is True


def test_reducer_computes_known_shares_and_amdahl_ceilings() -> None:
    """REQ-ARC-WMTE-7490 computes wall, token, and Amdahl values when allowed."""
    module = _load_module()
    reduced = module.reduce_episodes([_episode()], publish_numeric=True)

    candidate = reduced["decision_points"]["candidate_selection"]
    generation = reduced["decision_points"]["induction_and_generation"]
    assert candidate["wall_fraction"] == pytest.approx(0.05)
    assert candidate["amdahl_ceiling"] == pytest.approx(1.0 / 0.95)
    assert generation["token_fraction"] == pytest.approx(0.8)
    assert reduced["reconciliation"]["token_rows_reconciled"] == 1


@pytest.mark.parametrize(
    ("episode_count", "game_count", "passed"),
    [(29, 10, False), (30, 9, False), (30, 10, True)],
)
def test_numeric_share_gate_requires_30_episodes_and_10_games(
    episode_count: int, game_count: int, passed: bool
) -> None:
    """SCENARIO-ARC-WMTE-7490-COVERAGE-GATE enforces both support floors."""
    module = _load_module()
    rows = [
        {
            "episode_id": f"e{index}",
            "game": f"g{index % game_count}",
            "complete": True,
            "current_model": True,
            "numeric_eligible": True,
        }
        for index in range(episode_count)
    ]

    gate = module.numeric_share_gate(rows)

    assert gate["passed"] is passed
    assert gate["complete_current_model_episodes"] == episode_count
    assert gate["current_model_games"] == game_count


def test_flagged_wrong_model_and_controls_never_enter_gate() -> None:
    """SCENARIO-ARC-WMTE-7490-EXCLUSIONS keeps controls out of numeric support."""
    module = _load_module()
    rows = [
        {
            "episode_id": f"eligible-{index}",
            "game": f"g{index}",
            "complete": True,
            "current_model": True,
            "numeric_eligible": True,
        }
        for index in range(10)
    ]
    rows.extend(
        [
            {
                "episode_id": "flagged",
                "game": "bad-1",
                "complete": True,
                "current_model": True,
                "numeric_eligible": False,
                "exclusion_reason": "flagged_adversarial",
            },
            {
                "episode_id": "old-model",
                "game": "bad-2",
                "complete": True,
                "current_model": False,
                "numeric_eligible": False,
                "exclusion_reason": "non_current_model",
            },
            {
                "episode_id": "control",
                "game": "bad-3",
                "complete": True,
                "current_model": True,
                "numeric_eligible": False,
                "exclusion_reason": "schema_control_only",
            },
        ]
    )

    gate = module.numeric_share_gate(rows)

    assert gate["complete_current_model_episodes"] == 10
    assert gate["current_model_games"] == 10


def test_coverage_only_nulls_every_real_share_and_ceiling() -> None:
    """SCENARIO-ARC-WMTE-7490-COVERAGE-GATE suppresses real numeric shares."""
    module = _load_module()
    reduced = module.reduce_episodes([_episode()], publish_numeric=False)

    for point in reduced["decision_points"].values():
        assert point["wall_s"] is None
        assert point["tokens"] is None
        assert point["wall_fraction"] is None
        assert point["token_fraction"] is None
        assert point["amdahl_ceiling"] is None


def test_writer_uses_only_the_explicit_tmp_path(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7490-TERMINAL writes only its explicit target."""
    module = _load_module()
    output_path = tmp_path / "private" / "experiment_7490.json"
    payload = {
        "experiment_id": 7490,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "honest_verdict": "complete_test_fixture",
    }

    module.write_artifact(output_path, payload)

    assert json.loads(output_path.read_text()) == payload
    assert list(tmp_path.rglob("*.json")) == [output_path]


def test_inventory_keeps_model_flag_fields_and_counts(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7490 inventory rows retain every required classification."""
    module = _load_module()
    artifact = tmp_path / "results/experiment_9000_arc_fixture.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "experiment_id": 9000,
                "flagged_adversarial": False,
                "model_specs": [
                    {
                        "name": "Qwen3.8-27B",
                        "revision": "rev-test",
                        "hf_id": "unsloth/Qwen3.8-27B-GGUF",
                    }
                ],
                "sample_size_budget": {
                    "complete_independent_units": 3,
                    "independent_game_clusters": 2,
                },
                "phase_spans": [],
            }
        )
    )

    rows = module.inventory_artifacts(tmp_path)

    assert rows == [
        {
            "path": "results/experiment_9000_arc_fixture.json",
            "artifact_class": "phase_span_terminal",
            "experiment_id": 9000,
            "model_name": "Qwen3.8-27B",
            "model_version": "rev-test",
            "flagged_adversarial": False,
            "flag_stamp_present": True,
            "cost_fields": ["phase_spans"],
            "episode_count": 3,
            "game_count": 2,
            "sha256": ANY,
        }
    ]


def test_missing_verifier_planner_and_environment_fires_kill_rule() -> None:
    """SCENARIO-ARC-WMTE-7490-KILL-RULE stops inseparable speed claims."""
    module = _load_module()
    coverage = {
        point: point in {"candidate_selection", "induction_and_generation", "supervisor"}
        for point in module.DECISION_POINTS
    }

    result = module.evaluate_kill_rule(coverage, replaceable_wall_shares=None)

    assert result["seam_separation_kill_fired"] is True
    assert result["speed_claims_stopped"] is True
    assert result["missing_separable_seams"] == [
        "world_model_verification",
        "planner",
        "environment",
    ]
