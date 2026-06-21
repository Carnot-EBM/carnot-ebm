"""Tests for Exp 4525 standing-loop level-up banking.

Spec refs: REQ-ARC-WMTE-4525, SCENARIO-ARC-WMTE-4525.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4525_levelup_attempt as exp4525


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "cd82",
        "reached_level": reached_level,
        "moves": 20,
        "states_expanded": 20,
        "verifier_src": "hand_verifier_cold_start",
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "reproduction_gate": {
            "game": "cd82",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": [
            json.dumps({"action": 4}, sort_keys=True, separators=(",", ":")),
            json.dumps(
                {"action": 6, "color": 12, "role": "palette", "x": 46.0, "y": 4.0},
                sort_keys=True,
                separators=(",", ":"),
            ),
        ],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-20'
reproducible_total_levels: 48
reproducible_total_games: 24
games:
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: first-solve L1 adapter-free.
  action_model: keyboard-only [ACTION3,2,2,4,5].
  solver: results/arc_explore_trajectory_cd82.json.
  reproduce: re-gated reproduced=True L1.
  gotchas: []
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
  win_condition: existing L2 row.
  action_model: keyboard-only.
  solver: existing.
  gotchas: []
"""


def test_req_arc_wmte_4525_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4525: OpenSpec names the level-up artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4525.SPEC_REFS:
        assert ref in spec
    assert exp4525.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4525.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec
    for required in (
        "offline_reproduced=true",
        "`reproduced_levels >= 1`",
        "`registry_updated=true`",
        "dead-end evidence",
    ):
        assert required in spec


def test_scenario_arc_wmte_4525_registry_text_update_banks_cd82_l2() -> None:
    """SCENARIO-ARC-WMTE-4525: a reproduced L2 loop result updates the registry."""

    loop = _loop_result()
    checksum = exp4525.reproducibility_checksum(
        {
            "target_game": "cd82",
            "reproduction_gate": loop["reproduction_gate"],
            "solution_labels": loop["solution_labels"],
        }
    )

    updated_text, update = exp4525.apply_cd82_registry_bank(
        _registry_text(),
        loop_result=loop,
        checksum=checksum,
        artifact_path=exp4525.RESULT_RELATIVE_PATH,
    )
    registry = yaml.safe_load(updated_text)
    cd82 = next(row for row in registry["games"] if row["game"] == "cd82")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert update["prior_total_declared"] == 48
    assert update["prior_total_row_sum"] == 3
    assert registry["reproducible_total_levels"] == 4
    assert cd82["levels_reproduced"] == 2
    assert "arc_loop_solve --game cd82 --target-level 3" in cd82["dead_ends"][0]
    assert checksum in cd82["reproduce"]


def test_req_arc_wmte_4525_registry_update_handles_no_bank_and_bad_registry() -> None:
    """REQ-ARC-WMTE-4525: registry updates are gated by a real reproduced advance."""

    unchanged, update = exp4525.apply_cd82_registry_bank(
        _registry_text(),
        loop_result=_loop_result(reached_level=1),
        checksum="unused",
        artifact_path=exp4525.RESULT_RELATIVE_PATH,
    )
    assert unchanged == _registry_text()
    assert update["updated"] is False
    assert update["banked_levels"] == 0

    single_row = """schema_version: 1
reproducible_total_levels: 1
games:
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 1
"""
    updated, single_update = exp4525.apply_cd82_registry_bank(
        single_row,
        loop_result=_loop_result(),
        checksum="abc",
        artifact_path=exp4525.RESULT_RELATIVE_PATH,
    )
    assert yaml.safe_load(updated)["games"][0]["levels_reproduced"] == 2
    assert single_update["new_total_declared"] == 2

    bad_registry = "schema_version: 1\ngames:\n- game: m0r0\n  reproducibility: reproduced\n"
    try:
        exp4525.apply_cd82_registry_bank(
            bad_registry,
            loop_result=_loop_result(),
            checksum="unused",
            artifact_path=exp4525.RESULT_RELATIVE_PATH,
        )
    except ValueError as exc:
        assert "registry missing game row: cd82" in str(exc)
    else:
        raise AssertionError("missing cd82 row should fail")


def test_req_arc_wmte_4525_artifact_success_requires_reproduced_bank() -> None:
    """REQ-ARC-WMTE-4525: success needs an offline-reproduced registry bank."""

    loop = _loop_result()
    registry_update = {
        "updated": True,
        "path": exp4525.REGISTRY_RELATIVE_PATH,
        "target_game": "cd82",
        "prior_game_levels": 1,
        "new_game_levels": 2,
        "banked_levels": 1,
        "prior_total_declared": 48,
        "prior_total_row_sum": 49,
        "new_total_declared": 50,
        "new_total_row_sum": 50,
        "reconciled_total_delta": 2,
    }

    artifact = exp4525.build_artifact(
        loop_result=loop,
        registry_update=registry_update,
        preconditions_checked={
            "AGENTS.md": True,
            "CODEX.md": True,
            "offline_arcade_import_smoke": True,
            "spec_refs_present": True,
        },
        dead_ends=exp4525.DEFAULT_DEAD_ENDS,
    )

    assert artifact["honest_verdict"] == "success: cd82_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["registry_updated"] is True
    assert artifact["field_principles"] == exp4525.FIELD_PRINCIPLES
    assert artifact["schema_errors"] == []
    assert exp4525.artifact_schema_errors(artifact) == []

    changed = dict(artifact)
    changed["reproducibility_checksum"] = "bad"
    assert any("checksum" in error for error in exp4525.artifact_schema_errors(changed))

    missing = dict(artifact)
    del missing["preconditions_checked"]
    assert any("missing required field: preconditions_checked" == error for error in exp4525.artifact_schema_errors(missing))

    mutations = [
        ("experiment", "wrong", "experiment mismatch"),
        ("schema", "wrong", "schema mismatch"),
        ("spec_refs", [], "spec_refs mismatch"),
        ("field_principles", {}, "field_principles mismatch"),
        ("random_seed", 1, "random_seed mismatch"),
    ]
    for field, value, expected_error in mutations:
        invalid = dict(artifact)
        invalid[field] = value
        assert expected_error in exp4525.artifact_schema_errors(invalid)

    false_success = dict(artifact)
    false_success["offline_reproduced"] = False
    assert "success artifact missing reproduced registry bank" in exp4525.artifact_schema_errors(false_success)

    repeated = exp4525.build_artifact(
        loop_result=loop,
        registry_update={**registry_update, "updated": False, "banked_levels": 0},
        preconditions_checked={"offline_arcade_import_smoke": True, "spec_refs_present": True},
        dead_ends=[],
    )
    assert repeated["honest_verdict"] == "complete: cd82_levelup_honest_residual"
    assert repeated["reproduced_levels"] == 0
    assert repeated["schema_errors"] == []


def test_scenario_arc_wmte_4525_run_experiment_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4525: the runner writes the bank artifact and registry."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4525.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4525.LOOP_RESULT_RELATIVE_PATH).write_text(
        json.dumps(_loop_result(), indent=2),
        encoding="utf-8",
    )
    (tmp_path / exp4525.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4525\nSCENARIO-ARC-WMTE-4525\n",
        encoding="utf-8",
    )

    artifact = exp4525.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )

    out = json.loads((tmp_path / exp4525.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4525.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == out
    assert out["spec_refs"] == exp4525.SPEC_REFS
    assert out["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert out["preconditions_checked"]["spec_refs_present"] is True
    assert out["registry_update"]["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 4
