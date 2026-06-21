"""Tests for Exp 4535 ARC sprint level-up banking.

Spec refs: REQ-ARC-WMTE-4535, SCENARIO-ARC-WMTE-4535.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4535_levelup_attempt as exp4535


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "sp80",
        "reached_level": reached_level,
        "moves": 33,
        "states_expanded": 33,
        "verifier_src": "hand_verifier_cold_start",
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "reproduction_gate": {
            "game": "sp80",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4535.SP80_L2_SOLUTION_LABELS),
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-20'
games:
- game: sp80
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: first-solve L1 adapter-free; move-right x3 then commit.
  action_model: keyboard-only [ACTION4,4,4,5]; ACTION4 = one direction, ACTION5 commit.
  solver: results/arc_explore_trajectory_sp80.json (graph_explore_solve_v2, 4 actions).
  reproduce: re-gated reproduced=True L1.
  gotchas: []
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 49
  win_condition: existing row used to make totals realistic.
  action_model: existing.
  solver: existing.
  gotchas: []
reproducible_total_levels: 50
reproducible_total_games: 24
"""


def test_req_arc_wmte_4535_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4535: OpenSpec names the level-up artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4535.SPEC_REFS:
        assert ref in spec
    assert exp4535.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4535.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec
    for required in (
        "`offline_reproduced=true`",
        "`reproduced_levels >= 1`",
        "`registry_updated=true`",
        "dead-end evidence",
    ):
        assert required in spec


def test_scenario_arc_wmte_4535_registry_text_update_banks_sp80_l2() -> None:
    """SCENARIO-ARC-WMTE-4535: a reproduced sp80 L2 loop result updates the registry."""

    loop = _loop_result()
    checksum = exp4535.reproducibility_checksum(
        {
            "target_game": "sp80",
            "reproduction_gate": loop["reproduction_gate"],
            "solution_labels": loop["solution_labels"],
        }
    )

    updated_text, update = exp4535.apply_sp80_registry_bank(
        _registry_text(),
        loop_result=loop,
        checksum=checksum,
        artifact_path=exp4535.RESULT_RELATIVE_PATH,
    )
    registry = yaml.safe_load(updated_text)
    sp80 = next(row for row in registry["games"] if row["game"] == "sp80")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert update["prior_total_declared"] == 50
    assert update["prior_total_row_sum"] == 50
    assert registry["reproducible_total_levels"] == 51
    assert sp80["levels_reproduced"] == 2
    assert "splitter placements" in sp80["win_condition"]
    assert "arc_loop_solve --game sp80 --target-level 3" in sp80["dead_ends"][0]
    assert checksum in sp80["reproduce"]


def test_req_arc_wmte_4535_registry_update_requires_reproduced_advance() -> None:
    """REQ-ARC-WMTE-4535: registry updates are gated by a real reproduced advance."""

    unchanged, update = exp4535.apply_sp80_registry_bank(
        _registry_text(),
        loop_result=_loop_result(reached_level=1),
        checksum="unused",
        artifact_path=exp4535.RESULT_RELATIVE_PATH,
    )
    assert unchanged == _registry_text()
    assert update["updated"] is False
    assert update["banked_levels"] == 0

    unreproduced, bad_update = exp4535.apply_sp80_registry_bank(
        _registry_text(),
        loop_result=_loop_result(reached_level=2, reproduced=False),
        checksum="unused",
        artifact_path=exp4535.RESULT_RELATIVE_PATH,
    )
    assert unreproduced == _registry_text()
    assert bad_update["updated"] is False

    try:
        exp4535.apply_sp80_registry_bank(
            "schema_version: 1\ngames: []\nreproducible_total_levels: 0\n",
            loop_result=_loop_result(),
            checksum="unused",
            artifact_path=exp4535.RESULT_RELATIVE_PATH,
        )
    except ValueError as exc:
        assert "registry missing game row: sp80" in str(exc)
    else:
        raise AssertionError("missing sp80 row should fail")


def test_req_arc_wmte_4535_artifact_success_requires_reproduced_bank() -> None:
    """REQ-ARC-WMTE-4535: success needs an offline-reproduced registry bank."""

    loop = _loop_result()
    registry_update = {
        "updated": True,
        "path": exp4535.REGISTRY_RELATIVE_PATH,
        "target_game": "sp80",
        "prior_game_levels": 1,
        "new_game_levels": 2,
        "banked_levels": 1,
        "prior_total_declared": 50,
        "prior_total_row_sum": 50,
        "new_total_declared": 51,
        "new_total_row_sum": 51,
        "reconciled_total_delta": 1,
    }

    artifact = exp4535.build_artifact(
        loop_result=loop,
        registry_update=registry_update,
        preconditions_checked={
            "AGENTS.md": True,
            "CODEX.md": True,
            "offline_arcade_import_smoke": True,
            "spec_refs_present": True,
        },
        dead_ends=exp4535.DEFAULT_DEAD_ENDS,
    )

    assert artifact["honest_verdict"] == "success: sp80_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["registry_updated"] is True
    assert artifact["field_principles"] == exp4535.FIELD_PRINCIPLES
    assert artifact["schema_errors"] == []
    assert exp4535.artifact_schema_errors(artifact) == []

    changed = dict(artifact)
    changed["reproducibility_checksum"] = "bad"
    assert any("checksum" in error for error in exp4535.artifact_schema_errors(changed))

    false_success = dict(artifact)
    false_success["offline_reproduced"] = False
    assert "success artifact missing reproduced registry bank" in exp4535.artifact_schema_errors(false_success)

    repeated = exp4535.build_artifact(
        loop_result=loop,
        registry_update={**registry_update, "updated": False, "banked_levels": 0},
        preconditions_checked={"offline_arcade_import_smoke": True, "spec_refs_present": True},
        dead_ends=[],
    )
    assert repeated["honest_verdict"] == "complete: sp80_delta_identified_no_bank"
    assert repeated["reproduced_levels"] == 0
    assert repeated["schema_errors"] == []


def test_scenario_arc_wmte_4535_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4535: the runner writes the bank artifact and registry."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4535.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4535.LOOP_RESULT_RELATIVE_PATH).write_text(
        json.dumps(_loop_result(), indent=2),
        encoding="utf-8",
    )
    (tmp_path / exp4535.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4535\nSCENARIO-ARC-WMTE-4535\n",
        encoding="utf-8",
    )

    artifact = exp4535.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )

    out = json.loads((tmp_path / exp4535.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4535.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == out
    assert out["spec_refs"] == exp4535.SPEC_REFS
    assert out["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert out["preconditions_checked"]["spec_refs_present"] is True
    assert out["registry_update"]["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 51


def test_scenario_arc_wmte_4535_sp80_adapter_reproduces_l2_offline() -> None:
    """SCENARIO-ARC-WMTE-4535: the registered sp80 adapter replays L2 offline."""

    from carnot.agentic import arc_game_adapters
    from carnot.agentic import arc_solver_kit as kit

    adapter = arc_game_adapters.get_adapter("sp80")
    assert adapter is not None
    assert list(arc_game_adapters.SP80_L2_SOLUTION_LABELS) == list(exp4535.SP80_L2_SOLUTION_LABELS)

    gate = kit.reproduce(
        "sp80",
        arc_game_adapters.SP80_L2_SOLUTION_LABELS,
        adapter.apply,
        claimed_level=2,
    )

    assert gate["reproduced"] is True
    assert gate["reached_level"] == 2
