"""Tests for Exp 4593 ft09 level-up self-play.

Spec refs: REQ-ARC-WMTE-4593, SCENARIO-ARC-WMTE-4593.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_4593_levelup_selfplay as exp4593


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "ft09",
        "reached_level": reached_level,
        "states_expanded": 11,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": exp4593.CHECKPOINT_RELATIVE_PATH if reproduced else None,
        "reproduction_gate": {
            "game": "ft09",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4593.FT09_L2_SOLUTION_LABELS),
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ft09
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: ft09 L1 local constraint.
  action_model: ACTION6 click-only L1.
  solver: existing.
  reproduce: existing L1 gate.
  gotchas:
  - Use the real offline frame level counter as the verifier.
- game: sk48
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: existing.
  solver: existing.
  gotchas: []
reproducible_total_levels: 54
reproducible_total_games: 24
"""


def _write_minimal_tree(tmp_path: Path, loop_result: dict[str, object] | None = None) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(
        parents=True
    )
    (tmp_path / exp4593.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4593.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4593\nSCENARIO-ARC-WMTE-4593\n",
        encoding="utf-8",
    )
    (tmp_path / exp4593.LOOP_RESULT_RELATIVE_PATH).write_text(
        json.dumps(loop_result or _loop_result(), indent=2),
        encoding="utf-8",
    )
    (tmp_path / exp4593.CHECKPOINT_RELATIVE_PATH).write_text(
        json.dumps({"schema": "carnot_arc_learned_verifier_v1", "n_samples": 11}),
        encoding="utf-8",
    )


def test_req_arc_wmte_4593_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4593: OpenSpec declares the ft09 self-play bank contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4593.SPEC_REFS:
        assert ref in spec
    assert exp4593.RESULT_RELATIVE_PATH in spec
    assert exp4593.CHECKPOINT_RELATIVE_PATH in spec
    assert "(22,16)" in spec
    for field, principle in exp4593.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4593_ft09_adapter_exposes_l1_then_l2_delta() -> None:
    """REQ-ARC-WMTE-4593: the adapter schedules the reproduced ft09 L1+L2 labels."""

    from carnot.agentic import arc_game_adapters as adapters

    adapter = adapters.get_adapter("ft09")
    assert adapter is not None
    assert adapter.featurize is not None

    first_l1 = adapter.action_labels(None, type("Frame", (), {"levels_completed": 0})(), ())
    first_l2 = adapter.action_labels(None, type("Frame", (), {"levels_completed": 1})(), ())
    exhausted_l2 = adapter.action_labels(
        None,
        type("Frame", (), {"levels_completed": 1})(),
        tuple("x" for _ in exp4593.FT09_L2_TAIL_LABELS),
    )

    assert first_l1 == [exp4593.FT09_L1_LABELS[0]]
    assert first_l2 == [exp4593.FT09_L2_TAIL_LABELS[0]]
    assert exhausted_l2 == []
    assert adapter.depth_caps[1] == len(exp4593.FT09_L1_LABELS)
    assert adapter.depth_caps[2] == len(exp4593.FT09_L2_TAIL_LABELS)


def test_scenario_arc_wmte_4593_standing_loop_reports_ft09_l2(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4593: ft09 L2 is reported, gated, and checkpointed."""

    import scripts.arc_loop_solve as arc_loop_solve

    monkeypatch.setattr(arc_loop_solve, "REPO", tmp_path)
    monkeypatch.setattr(arc_loop_solve, "CKPT_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()

    result = arc_loop_solve.solve_adaptered("ft09", 2)

    assert result["reached_level"] == 2
    assert result["offline_reproduced"] is True
    assert result["reproduced_levels"] == 2
    assert result["reproduction_gate"]["claimed_level"] == 2
    assert result["reproduction_gate"]["reached_level"] == 2
    assert result["learned_verifier_checkpoint"] == exp4593.CHECKPOINT_RELATIVE_PATH
    assert result["solution_labels"] == list(exp4593.FT09_L2_SOLUTION_LABELS)


def test_scenario_arc_wmte_4593_registry_update_banks_ft09_l2_only() -> None:
    """SCENARIO-ARC-WMTE-4593: registry increments only for reproduced ft09 progress."""

    loop_result = _loop_result()
    checksum = exp4593.reproducibility_checksum(
        {
            "target_game": "ft09",
            "reproduction_gate": loop_result["reproduction_gate"],
            "solution_labels": loop_result["solution_labels"],
        }
    )

    updated_text, update = exp4593.apply_ft09_registry_bank(
        _registry_text(),
        loop_result=loop_result,
        checkpoint_path=exp4593.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=True,
        checksum=checksum,
        artifact_path=exp4593.RESULT_RELATIVE_PATH,
    )
    registry = yaml.safe_load(updated_text)
    ft09 = next(row for row in registry["games"] if row["game"] == "ft09")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 55
    assert ft09["mechanic_class"] == "local_constraint_color_cycle"
    assert exp4593.RESULT_RELATIVE_PATH in ft09["reproduce"]
    assert exp4593.CHECKPOINT_RELATIVE_PATH in ft09["learned_verifier_checkpoint"]
    assert checksum in ft09["reproduce"]

    repeated_text, repeated_update = exp4593.apply_ft09_registry_bank(
        _registry_text().replace("levels_reproduced: 1", "levels_reproduced: 2", 1),
        loop_result=loop_result,
        checkpoint_path=exp4593.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=True,
        checksum="unused",
        artifact_path=exp4593.RESULT_RELATIVE_PATH,
    )
    assert repeated_update["updated"] is False
    assert repeated_update["reason"] == "reproduced_existing_or_lower_level"
    assert "checksum unused" not in repeated_text


def test_req_arc_wmte_4593_artifact_schema_success_and_no_bank() -> None:
    """REQ-ARC-WMTE-4593: success requires reproduced progress and verifier checkpoint."""

    success_update = {
        "updated": True,
        "path": exp4593.REGISTRY_RELATIVE_PATH,
        "target_game": "ft09",
        "prior_game_levels": 1,
        "new_game_levels": 2,
        "banked_levels": 1,
        "prior_total_declared": 54,
        "prior_total_row_sum": 54,
        "new_total_declared": 55,
        "new_total_row_sum": 55,
        "reconciled_total_delta": 1,
        "reason": "banked_offline_reproduced_level",
    }
    success = exp4593.build_artifact(
        loop_result=_loop_result(),
        registry_update=success_update,
        preconditions_checked={"offline_arcade_import_smoke": True, "spec_refs_present": True},
        dead_ends=exp4593.DEFAULT_DEAD_ENDS,
        verifier_delta={
            "checkpoint_path": exp4593.CHECKPOINT_RELATIVE_PATH,
            "checkpoint_updated": True,
            "positive_trace_count": 11,
            "negative_trace_count": 3,
        },
    )

    assert success["honest_verdict"] == "success: ft09_L2_offline_reproduced"
    assert success["target_game"] == "ft09"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["verifier_checkpoint_updated"] is True
    assert success["registry_updated"] is True
    assert success["schema_errors"] == []
    assert exp4593.artifact_schema_errors(success) == []

    no_bank = exp4593.build_artifact(
        loop_result=_loop_result(),
        registry_update={**success_update, "updated": False, "banked_levels": 0},
        preconditions_checked={"offline_arcade_import_smoke": True},
        dead_ends=[],
        verifier_delta={**success["verifier_delta"], "checkpoint_updated": False},
    )
    assert no_bank["honest_verdict"] == "complete: ft09_delta_identified_no_bank"
    assert no_bank["reproduced_levels"] == 0
    assert no_bank["schema_errors"] == []


def test_req_arc_wmte_4593_defensive_branches_are_honest(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4593: defensive branches report honest no-bank causes."""

    with pytest.raises(ValueError, match="registry missing game row"):
        exp4593.apply_ft09_registry_bank(
            "schema_version: 1\ngames: []\nreproducible_total_levels: 0\n",
            loop_result=_loop_result(),
            checkpoint_path=exp4593.CHECKPOINT_RELATIVE_PATH,
            checkpoint_updated=True,
            checksum="unused",
            artifact_path=exp4593.RESULT_RELATIVE_PATH,
        )

    no_repro_text, no_repro_update = exp4593.apply_ft09_registry_bank(
        _registry_text(),
        loop_result=_loop_result(reproduced=False),
        checkpoint_path=exp4593.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=True,
        checksum="unused",
        artifact_path=exp4593.RESULT_RELATIVE_PATH,
    )
    assert no_repro_text == _registry_text()
    assert no_repro_update["reason"] == "not_offline_reproduced"

    no_checkpoint_text, no_checkpoint_update = exp4593.apply_ft09_registry_bank(
        _registry_text(),
        loop_result=_loop_result(),
        checkpoint_path=exp4593.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=False,
        checksum="unused",
        artifact_path=exp4593.RESULT_RELATIVE_PATH,
    )
    assert no_checkpoint_text == _registry_text()
    assert no_checkpoint_update["reason"] == "verifier_checkpoint_not_updated"

    changed = exp4593.build_artifact(
        loop_result=_loop_result(),
        registry_update={
            "updated": True,
            "banked_levels": 1,
            "reason": "banked_offline_reproduced_level",
        },
        preconditions_checked={"offline_arcade_import_smoke": True},
        dead_ends=[],
        verifier_delta={
            "checkpoint_path": exp4593.CHECKPOINT_RELATIVE_PATH,
            "checkpoint_updated": True,
        },
    )
    changed.pop("honest_verdict")
    changed["experiment"] = "wrong"
    changed["schema"] = "wrong"
    changed["spec_refs"] = []
    changed["field_principles"] = {}
    changed["target_game"] = "wrong"
    changed["random_seed"] = -1
    changed["reproducibility_checksum"] = "bad"
    changed["verifier_checkpoint_updated"] = False

    errors = exp4593.artifact_schema_errors(changed)
    assert "missing required field: honest_verdict" in errors
    assert "experiment mismatch" in errors
    assert "schema mismatch" in errors
    assert "spec_refs mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "target_game mismatch" in errors
    assert "random_seed mismatch" in errors
    assert "checksum mismatch" in errors

    false_success = exp4593.build_artifact(
        loop_result=_loop_result(),
        registry_update={
            "updated": True,
            "banked_levels": 1,
            "reason": "banked_offline_reproduced_level",
        },
        preconditions_checked={"offline_arcade_import_smoke": True},
        dead_ends=[],
        verifier_delta={
            "checkpoint_path": exp4593.CHECKPOINT_RELATIVE_PATH,
            "checkpoint_updated": True,
        },
    )
    false_success["verifier_checkpoint_updated"] = False
    assert "success artifact missing reproduced registry bank or verifier checkpoint" in (
        exp4593.artifact_schema_errors(false_success)
    )

    (tmp_path / "models").mkdir()
    bad_checkpoint = tmp_path / exp4593.CHECKPOINT_RELATIVE_PATH
    bad_checkpoint.write_text("{not json", encoding="utf-8")
    delta = exp4593._checkpoint_delta(
        tmp_path,
        {"learned_verifier_checkpoint": exp4593.CHECKPOINT_RELATIVE_PATH, "solution_labels": ["x"]},
    )
    assert delta["checkpoint_updated"] is True
    assert delta["checkpoint_schema"] is None


def test_scenario_arc_wmte_4593_run_experiment_writes_success_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4593: a reproduced ft09 L2 gate writes the bank artifact."""

    _write_minimal_tree(tmp_path)

    artifact = exp4593.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )
    out = json.loads((tmp_path / exp4593.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4593.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == out
    assert out["honest_verdict"] == "success: ft09_L2_offline_reproduced"
    assert out["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert out["preconditions_checked"]["spec_refs_present"] is True
    assert out["verifier_checkpoint_updated"] is True
    assert registry["reproducible_total_levels"] == 55
