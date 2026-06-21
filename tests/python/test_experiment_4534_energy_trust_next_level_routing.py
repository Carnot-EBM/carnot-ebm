"""Tests for Exp 4534 trust-energy next-level frontier routing.

Spec refs: REQ-ARC-WMTE-4534, SCENARIO-ARC-WMTE-4534.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _levels(lp85: int = 1, m0r0: int = 1, sp80: int = 1, vc33: int = 1) -> dict[str, int]:
    return {"lp85": lp85, "m0r0": m0r0, "sp80": sp80, "vc33": vc33}


def _measurement(
    label: str,
    *,
    levels: dict[str, int] | None = None,
    efficiency: float = 2.0074,
) -> dict[str, object]:
    levels = dict(levels or _levels())
    return {
        "measurement": label,
        "target_games": ["lp85", "sp80"],
        "core_efficiency": efficiency,
        "deepest_level_by_game": levels,
        "per_game": [
            {"game": game, "best_level": level, "diagnostics": {"stopped_reason": "cached"}}
            for game, level in levels.items()
        ],
    }


def _signal(auroc: float = 1.0, passed: bool = True) -> dict[str, object]:
    return {
        "energy_separation_auroc": auroc,
        "positive_control": {
            "passed": passed,
            "game": "known_l2_trust_energy_fixture",
            "sample_count": 4,
        },
        "samples": [
            {"state_id": "progress-a", "energy": 0.05, "deeper_progress": True},
            {"state_id": "progress-b", "energy": 0.15, "deeper_progress": True},
            {"state_id": "stuck-a", "energy": 0.75, "deeper_progress": False},
            {"state_id": "stuck-b", "energy": 0.95, "deeper_progress": False},
        ],
    }


def test_req_arc_wmte_4534_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4534: OpenSpec anchors the trust-energy routing artifact."""

    from carnot import experiment_4534_energy_trust_next_level_routing as exp4534

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4534" in spec
    assert "SCENARIO-ARC-WMTE-4534" in spec
    assert exp4534.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4534.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4534_energy_bias_is_lower_is_better_depth_secondary() -> None:
    """SCENARIO-ARC-WMTE-4534: trust energy biases same-depth frontier only."""

    explorer = StepwiseExplorer()
    explorer.cur = "root"
    explorer.set_goal_bias(
        lambda frame: float(frame.energy),
        label="trust_energy_next_level_distance",
        lower_is_better=True,
    )
    explorer.graph = {
        "shallow_high_energy": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 2, "data": None}],
            "value": 0.0,
            "frame": SimpleNamespace(energy=0.95),
        },
        "deep_low_energy": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 3, "data": None}],
            "value": 0.0,
            "frame": SimpleNamespace(energy=0.05),
        },
    }

    assert explorer._frontier() == "shallow_high_energy"

    explorer.graph["shallow_high_energy"]["path"].append({"action": 4, "data": None})
    assert explorer._frontier() == "deep_low_energy"
    assert explorer.goal_bias_diagnostics()["lower_is_better"] is True


def test_req_arc_wmte_4534_auroc_treats_lower_energy_as_progress() -> None:
    """REQ-ARC-WMTE-4534: AUROC characterizes deeper-progress vs L1-stuck states."""

    from carnot import experiment_4534_energy_trust_next_level_routing as exp4534

    assert exp4534.energy_separation_auroc(_signal()["samples"]) == 1.0
    assert exp4534.energy_separation_auroc(
        [
            {"energy": 0.5, "deeper_progress": True},
            {"energy": 0.5, "deeper_progress": False},
        ]
    ) == 0.5
    assert exp4534.energy_separation_auroc([{"energy": 0.2, "deeper_progress": True}]) is None


def test_req_arc_wmte_4534_honest_null_records_signal_characterization(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4534: no L2 still reports oracle-distinct signal evidence."""

    from carnot import experiment_4534_energy_trust_next_level_routing as exp4534

    artifact = exp4534.build_artifact(
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_import_smoke": True,
            "spec_has_req_4534": True,
            "ok": True,
        },
        no_energy_control=_measurement("no_energy_control"),
        energy_routed=_measurement("energy_routed"),
        signal_characterization=_signal(),
        a1_goal={
            "available": False,
            "source": "results/experiment_4533_per_level_goal_reinduction.json",
            "reason": "a1_honest_null",
        },
        random_seed=4534,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == (
        "complete: energy_routing_no_deeper_level_signal_characterized_honest_null"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["core_efficiency_energy_routed"] == exp4534.CORE_EFFICIENCY_BASELINE
    assert artifact["energy_separation_auroc"] == 1.0
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["core_solves_preserved"] is True
    assert artifact["deepest_level_reached_per_core_game"]["energy_routed"]["lp85"] == 1
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert exp4534.artifact_schema_errors(artifact) == []

    out = exp4534.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact

    changed = dict(artifact)
    changed["verifier_is_oracle"] = True
    assert any("oracle" in error for error in exp4534.artifact_schema_errors(changed))


def test_req_arc_wmte_4534_success_requires_energy_only_l2_and_core_preservation() -> None:
    """REQ-ARC-WMTE-4534: an oracle-distinct success must beat matched no-energy."""

    from carnot import experiment_4534_energy_trust_next_level_routing as exp4534

    artifact = exp4534.build_artifact(
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_import_smoke": True,
            "spec_has_req_4534": True,
            "ok": True,
        },
        no_energy_control=_measurement("no_energy_control", levels=_levels(lp85=1)),
        energy_routed=_measurement("energy_routed", levels=_levels(lp85=2), efficiency=3.25),
        signal_characterization=_signal(),
        a1_goal={"available": True, "predicate": "L2_fixture_predicate"},
        random_seed=4534,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "success: energy_routing_lp85_reached_L2_oracle_distinct"
    assert artifact["core_efficiency_energy_routed"] == 3.25
    assert artifact["core_solves_preserved"] is True
    assert artifact["chosen_submitted_config"]["energy_next_level_routing"] is True
    assert exp4534.artifact_schema_errors(artifact) == []

    dropped = exp4534.build_artifact(
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_import_smoke": True,
            "spec_has_req_4534": True,
            "ok": True,
        },
        no_energy_control=_measurement("no_energy_control", levels=_levels(lp85=1, m0r0=1)),
        energy_routed=_measurement("energy_routed", levels=_levels(lp85=2, m0r0=0), efficiency=3.25),
        signal_characterization=_signal(),
        a1_goal={"available": True, "predicate": "L2_fixture_predicate"},
        random_seed=4534,
        duration_s=0.25,
    )
    assert dropped["honest_verdict"] == (
        "complete: energy_routing_no_deeper_level_signal_characterized_honest_null"
    )


def test_scenario_arc_wmte_4534_run_writes_injected_measurements(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4534: injected matched measurements write stable JSON."""

    from carnot import experiment_4534_energy_trust_next_level_routing as exp4534

    artifact = exp4534.run(
        root=tmp_path,
        preconditions_checked={
            "agents_md_read": True,
            "codex_md_read": True,
            "offline_arcade_import_smoke": True,
            "spec_has_req_4534": True,
            "ok": True,
        },
        a1_loader=lambda _root: {"honest_verdict": "complete: a1_null", "chosen_submitted_config": "unchanged"},
        measurement_runner=lambda _root, _a1: (
            _measurement("no_energy_control"),
            _measurement("energy_routed"),
        ),
        signal_runner=lambda: _signal(),
        now=lambda: 1.0,
    )

    assert artifact["result_path"] == exp4534.RESULT_RELATIVE_PATH
    assert artifact["a1_reinduced_goal"]["available"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads((tmp_path / exp4534.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
