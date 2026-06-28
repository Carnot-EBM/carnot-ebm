"""Tests for Exp 4570 rotated cn04 level-up bank.

Spec refs: REQ-ARC-WMTE-4570, SCENARIO-ARC-WMTE-4570.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4570_levelup_attempt as exp4570
from carnot.agentic import arc_game_adapters as adapters


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "cn04",
        "reached_level": reached_level,
        "states_expanded": 43,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "reproduction_gate": {
            "game": "cn04",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4570.CN04_L2_SOLUTION_LABELS),
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: first-solve L1 adapter-free.
  solver: existing.
  reproduce: existing L1 gate.
  gotchas: []
- game: sk48
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: existing.
  solver: existing.
  gotchas: []
reproducible_total_levels: 52
reproducible_total_games: 24
"""


def _write_minimal_tree(tmp_path: Path, loop_result: dict[str, object] | None = None) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4570.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4570.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4570\nSCENARIO-ARC-WMTE-4570\n",
        encoding="utf-8",
    )
    (tmp_path / exp4570.LOOP_RESULT_RELATIVE_PATH).write_text(
        json.dumps(loop_result or _loop_result(), indent=2),
        encoding="utf-8",
    )


def test_req_arc_wmte_4570_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4570: OpenSpec declares the cn04 bank contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4570.SPEC_REFS:
        assert ref in spec
    assert exp4570.RESULT_RELATIVE_PATH in spec
    assert "marker-pair shape-alignment mechanic" in spec
    for field, principle in exp4570.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4570_cn04_adapter_exposes_l1_l2_and_current_l3_delta() -> None:
    """REQ-ARC-WMTE-4570: the adapter still schedules reproduced cn04 L1+L2 labels."""

    adapter = adapters.get_adapter("cn04")
    assert adapter is not None
    assert "cn04" in adapters.adaptered_games()

    env = SimpleNamespace(_game=SimpleNamespace())
    first_l1 = adapter.action_labels(env, SimpleNamespace(levels_completed=0), ())
    next_l1 = adapter.action_labels(env, SimpleNamespace(levels_completed=0), ("done",))
    first_l2 = adapter.action_labels(env, SimpleNamespace(levels_completed=1), ())
    exhausted_l2 = adapter.action_labels(
        env,
        SimpleNamespace(levels_completed=1),
        tuple("x" for _ in exp4570.CN04_L2_TAIL_LABELS),
    )
    first_l3 = adapter.action_labels(env, SimpleNamespace(levels_completed=2), ())

    assert first_l1 == [exp4570.CN04_L1_LABELS[0]]
    assert next_l1 == [exp4570.CN04_L1_LABELS[1]]
    assert first_l2 == [exp4570.CN04_L2_TAIL_LABELS[0]]
    assert exhausted_l2 == []
    assert first_l3 == [adapters.CN04_L3_TAIL_LABELS[0]]
    assert adapter.depth_caps[1] == len(exp4570.CN04_L1_LABELS)
    assert adapter.depth_caps[2] == len(exp4570.CN04_L2_TAIL_LABELS)
    assert adapter.branch_mode == "fresh_env"


def test_scenario_arc_wmte_4570_registry_update_banks_cn04_l2_only() -> None:
    """SCENARIO-ARC-WMTE-4570: registry increments only for a reproduced cn04 advance."""

    loop_result = _loop_result()
    checksum = exp4570.reproducibility_checksum(
        {
            "target_game": "cn04",
            "reproduction_gate": loop_result["reproduction_gate"],
            "solution_labels": loop_result["solution_labels"],
        }
    )

    updated_text, update = exp4570.apply_cn04_registry_bank(
        _registry_text(),
        loop_result=loop_result,
        checksum=checksum,
        artifact_path=exp4570.RESULT_RELATIVE_PATH,
    )
    registry = yaml.safe_load(updated_text)
    cn04 = next(row for row in registry["games"] if row["game"] == "cn04")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert update["new_total_declared"] == 53
    assert registry["reproducible_total_levels"] == 53
    assert cn04["mechanic_class"] == "marker_pair_shape_alignment"
    assert exp4570.RESULT_RELATIVE_PATH in cn04["reproduce"]
    assert checksum in cn04["reproduce"]

    repeated_text, repeated_update = exp4570.apply_cn04_registry_bank(
        _registry_text().replace("levels_reproduced: 1", "levels_reproduced: 2", 1),
        loop_result=loop_result,
        checksum="unused",
        artifact_path=exp4570.RESULT_RELATIVE_PATH,
    )
    assert repeated_update["updated"] is False
    assert repeated_update["reason"] == "reproduced_existing_or_lower_level"
    assert "checksum unused" not in repeated_text

    unreproduced_text, unreproduced = exp4570.apply_cn04_registry_bank(
        _registry_text(),
        loop_result=_loop_result(reproduced=False),
        checksum="unused",
        artifact_path=exp4570.RESULT_RELATIVE_PATH,
    )
    assert unreproduced_text == _registry_text()
    assert unreproduced["reason"] == "not_offline_reproduced"

    single_game_text = "\n".join(_registry_text().splitlines()[:9]) + "\nreproducible_total_levels: 52\n"
    single_updated_text, single_update = exp4570.apply_cn04_registry_bank(
        single_game_text,
        loop_result=loop_result,
        checksum=checksum,
        artifact_path=exp4570.RESULT_RELATIVE_PATH,
    )
    assert single_update["updated"] is True
    assert "reproducible_total_levels: 53" in single_updated_text

    try:
        exp4570.apply_cn04_registry_bank(
            "schema_version: 1\ngames: []\nreproducible_total_levels: 52\n",
            loop_result=loop_result,
            checksum="unused",
            artifact_path=exp4570.RESULT_RELATIVE_PATH,
        )
    except ValueError as exc:
        assert "registry missing game row: cn04" in str(exc)
    else:
        raise AssertionError("missing cn04 row should fail")


def test_req_arc_wmte_4570_artifact_schema_success_and_no_bank() -> None:
    """REQ-ARC-WMTE-4570: success requires an offline-reproduced registry bank."""

    success_update = {
        "updated": True,
        "path": exp4570.REGISTRY_RELATIVE_PATH,
        "target_game": "cn04",
        "prior_game_levels": 1,
        "new_game_levels": 2,
        "banked_levels": 1,
        "prior_total_declared": 52,
        "prior_total_row_sum": 52,
        "new_total_declared": 53,
        "new_total_row_sum": 53,
        "reconciled_total_delta": 1,
        "reason": "banked_offline_reproduced_level",
    }
    success = exp4570.build_artifact(
        loop_result=_loop_result(),
        registry_update=success_update,
        preconditions_checked={"offline_arcade_import_smoke": True, "spec_refs_present": True},
        dead_ends=exp4570.DEFAULT_DEAD_ENDS,
    )

    assert success["honest_verdict"] == "success: cn04_L2_offline_reproduced"
    assert success["target_game"] == "cn04"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["registry_updated"] is True
    assert success["schema_errors"] == []
    assert exp4570.artifact_schema_errors(success) == []

    no_bank = exp4570.build_artifact(
        loop_result=_loop_result(reached_level=1),
        registry_update={**success_update, "updated": False, "banked_levels": 0},
        preconditions_checked={"offline_arcade_import_smoke": True},
        dead_ends=[],
    )
    assert no_bank["honest_verdict"] == "complete: cn04_delta_identified_no_bank"
    assert no_bank["reproduced_levels"] == 0
    assert no_bank["schema_errors"] == []

    changed = dict(success)
    changed["reproducibility_checksum"] = "bad"
    assert "checksum mismatch" in exp4570.artifact_schema_errors(changed)

    false_success = dict(success)
    false_success["offline_reproduced"] = False
    assert "success artifact missing reproduced registry bank" in exp4570.artifact_schema_errors(false_success)

    missing = dict(success)
    missing.pop("honest_verdict")
    assert "missing required field: honest_verdict" in exp4570.artifact_schema_errors(missing)

    mismatched = dict(success)
    mismatched.update(
        {
            "experiment": "wrong",
            "schema": "wrong",
            "spec_refs": [],
            "field_principles": {},
            "target_game": "sk48",
            "random_seed": -1,
        }
    )
    mismatch_errors = exp4570.artifact_schema_errors(mismatched)
    assert "experiment mismatch" in mismatch_errors
    assert "schema mismatch" in mismatch_errors
    assert "spec_refs mismatch" in mismatch_errors
    assert "field_principles mismatch" in mismatch_errors
    assert "target_game mismatch" in mismatch_errors
    assert "random_seed mismatch" in mismatch_errors


def test_scenario_arc_wmte_4570_run_experiment_writes_success_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4570: a reproduced cn04 L2 gate writes the bank artifact."""

    _write_minimal_tree(tmp_path)

    artifact = exp4570.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )
    out = json.loads((tmp_path / exp4570.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4570.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == out
    assert out["honest_verdict"] == "success: cn04_L2_offline_reproduced"
    assert out["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert out["preconditions_checked"]["spec_refs_present"] is True
    assert registry["reproducible_total_levels"] == 53
