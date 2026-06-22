"""Tests for Exp 4581 ar25 level-up self-play.

Spec refs: REQ-ARC-WMTE-4581, SCENARIO-ARC-WMTE-4581.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4581_levelup_selfplay as exp4581


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(reached_level: int = 2, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": "ar25",
        "reached_level": reached_level,
        "states_expanded": 26,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "learned_verifier_checkpoint": exp4581.CHECKPOINT_RELATIVE_PATH if reproduced else None,
        "reproduction_gate": {
            "game": "ar25",
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": list(exp4581.AR25_L2_SOLUTION_LABELS),
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ar25
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: ar25 L1 reflection.
  action_model: keyboard-only L1.
  solver: existing.
  reproduce: existing L1 gate.
  gotchas:
  - ACTION7 hidden undo stack remains a verifier residual.
- game: sk48
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: existing.
  solver: existing.
  gotchas: []
reproducible_total_levels: 53
reproducible_total_games: 24
"""


def _write_minimal_tree(tmp_path: Path, loop_result: dict[str, object] | None = None) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4581.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4581.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4581\nSCENARIO-ARC-WMTE-4581\n",
        encoding="utf-8",
    )
    (tmp_path / exp4581.LOOP_RESULT_RELATIVE_PATH).write_text(
        json.dumps(loop_result or _loop_result(), indent=2),
        encoding="utf-8",
    )
    (tmp_path / exp4581.CHECKPOINT_RELATIVE_PATH).write_text(
        json.dumps({"schema": "carnot_arc_learned_verifier_v1", "n_samples": 26}),
        encoding="utf-8",
    )


def test_req_arc_wmte_4581_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4581: OpenSpec declares the self-play bank contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4581.SPEC_REFS:
        assert ref in spec
    assert exp4581.RESULT_RELATIVE_PATH in spec
    assert exp4581.CHECKPOINT_RELATIVE_PATH in spec
    assert "mirror moves left twice" in spec
    for field, principle in exp4581.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4581_ar25_adapter_exposes_l1_then_l2_delta() -> None:
    """REQ-ARC-WMTE-4581: the adapter schedules the reproduced ar25 L1+L2 labels."""

    from carnot.agentic import arc_game_adapters as adapters

    adapter = adapters.get_adapter("ar25")
    assert adapter is not None
    assert "ar25" in adapters.adaptered_games()
    assert adapter.featurize is not None

    env = SimpleNamespace(_game=SimpleNamespace())
    first_l1 = adapter.action_labels(env, SimpleNamespace(levels_completed=0), ())
    next_l1 = adapter.action_labels(env, SimpleNamespace(levels_completed=0), ("done",))
    first_l2 = adapter.action_labels(env, SimpleNamespace(levels_completed=1), ())
    exhausted_l2 = adapter.action_labels(
        env,
        SimpleNamespace(levels_completed=1),
        tuple("x" for _ in exp4581.AR25_L2_TAIL_LABELS),
    )
    no_l3 = adapter.action_labels(env, SimpleNamespace(levels_completed=2), ())

    assert first_l1 == [exp4581.AR25_L1_LABELS[0]]
    assert next_l1 == [exp4581.AR25_L1_LABELS[1]]
    assert first_l2 == [exp4581.AR25_L2_TAIL_LABELS[0]]
    assert exhausted_l2 == []
    assert no_l3 == []
    assert adapter.depth_caps[1] == len(exp4581.AR25_L1_LABELS)
    assert adapter.depth_caps[2] == len(exp4581.AR25_L2_TAIL_LABELS)
    assert adapter.branch_mode == "fresh_env"


def test_scenario_arc_wmte_4581_standing_loop_reports_fresh_env_l2(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-ARC-WMTE-4581: fresh-env ar25 L2 is reported, gated, and checkpointed."""

    import scripts.arc_loop_solve as arc_loop_solve

    monkeypatch.setattr(arc_loop_solve, "REPO", tmp_path)
    monkeypatch.setattr(arc_loop_solve, "CKPT_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()

    result = arc_loop_solve.solve_adaptered("ar25", 2)

    assert result["reached_level"] == 2
    assert result["offline_reproduced"] is True
    assert result["reproduced_levels"] == 2
    assert result["reproduction_gate"]["claimed_level"] == 2
    assert result["reproduction_gate"]["reached_level"] == 2
    assert result["learned_verifier_checkpoint"] == exp4581.CHECKPOINT_RELATIVE_PATH
    assert result["solution_labels"] == list(exp4581.AR25_L2_SOLUTION_LABELS)


def test_scenario_arc_wmte_4581_registry_update_banks_ar25_l2_only() -> None:
    """SCENARIO-ARC-WMTE-4581: registry increments only for a reproduced ar25 advance."""

    loop_result = _loop_result()
    checksum = exp4581.reproducibility_checksum(
        {
            "target_game": "ar25",
            "reproduction_gate": loop_result["reproduction_gate"],
            "solution_labels": loop_result["solution_labels"],
        }
    )

    updated_text, update = exp4581.apply_ar25_registry_bank(
        _registry_text(),
        loop_result=loop_result,
        checkpoint_path=exp4581.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=True,
        checksum=checksum,
        artifact_path=exp4581.RESULT_RELATIVE_PATH,
    )
    registry = yaml.safe_load(updated_text)
    ar25 = next(row for row in registry["games"] if row["game"] == "ar25")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 1
    assert update["new_game_levels"] == 2
    assert update["banked_levels"] == 1
    assert update["new_total_declared"] == 54
    assert registry["reproducible_total_levels"] == 54
    assert ar25["mechanic_class"] == "reflection_mirror_object_alignment"
    assert exp4581.RESULT_RELATIVE_PATH in ar25["reproduce"]
    assert exp4581.CHECKPOINT_RELATIVE_PATH in ar25["learned_verifier_checkpoint"]
    assert checksum in ar25["reproduce"]

    repeated_text, repeated_update = exp4581.apply_ar25_registry_bank(
        _registry_text().replace("levels_reproduced: 1", "levels_reproduced: 2", 1),
        loop_result=loop_result,
        checkpoint_path=exp4581.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=True,
        checksum="unused",
        artifact_path=exp4581.RESULT_RELATIVE_PATH,
    )
    assert repeated_update["updated"] is False
    assert repeated_update["reason"] == "reproduced_existing_or_lower_level"
    assert "checksum unused" not in repeated_text

    no_checkpoint_text, no_checkpoint_update = exp4581.apply_ar25_registry_bank(
        _registry_text(),
        loop_result=loop_result,
        checkpoint_path=exp4581.CHECKPOINT_RELATIVE_PATH,
        checkpoint_updated=False,
        checksum="unused",
        artifact_path=exp4581.RESULT_RELATIVE_PATH,
    )
    assert no_checkpoint_text == _registry_text()
    assert no_checkpoint_update["reason"] == "verifier_checkpoint_not_updated"


def test_req_arc_wmte_4581_artifact_schema_success_and_no_bank() -> None:
    """REQ-ARC-WMTE-4581: success requires reproduced progress and verifier checkpoint."""

    success_update = {
        "updated": True,
        "path": exp4581.REGISTRY_RELATIVE_PATH,
        "target_game": "ar25",
        "prior_game_levels": 1,
        "new_game_levels": 2,
        "banked_levels": 1,
        "prior_total_declared": 53,
        "prior_total_row_sum": 53,
        "new_total_declared": 54,
        "new_total_row_sum": 54,
        "reconciled_total_delta": 1,
        "reason": "banked_offline_reproduced_level",
    }
    success = exp4581.build_artifact(
        loop_result=_loop_result(),
        registry_update=success_update,
        preconditions_checked={"offline_arcade_import_smoke": True, "spec_refs_present": True},
        dead_ends=exp4581.DEFAULT_DEAD_ENDS,
        verifier_delta={
            "checkpoint_path": exp4581.CHECKPOINT_RELATIVE_PATH,
            "checkpoint_updated": True,
            "positive_trace_count": 26,
            "negative_trace_count": 3,
        },
    )

    assert success["honest_verdict"] == "success: ar25_L2_offline_reproduced"
    assert success["target_game"] == "ar25"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["verifier_checkpoint_updated"] is True
    assert success["registry_updated"] is True
    assert success["schema_errors"] == []
    assert exp4581.artifact_schema_errors(success) == []

    no_bank = exp4581.build_artifact(
        loop_result=_loop_result(),
        registry_update={**success_update, "updated": False, "banked_levels": 0},
        preconditions_checked={"offline_arcade_import_smoke": True},
        dead_ends=[],
        verifier_delta={**success["verifier_delta"], "checkpoint_updated": False},
    )
    assert no_bank["honest_verdict"] == "complete: ar25_delta_identified_no_bank"
    assert no_bank["reproduced_levels"] == 0
    assert no_bank["schema_errors"] == []

    changed = dict(success)
    changed["reproducibility_checksum"] = "bad"
    assert "checksum mismatch" in exp4581.artifact_schema_errors(changed)

    false_success = dict(success)
    false_success["verifier_checkpoint_updated"] = False
    assert "success artifact missing reproduced registry bank or verifier checkpoint" in (
        exp4581.artifact_schema_errors(false_success)
    )


def test_scenario_arc_wmte_4581_run_experiment_writes_success_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4581: a reproduced ar25 L2 gate writes the bank artifact."""

    _write_minimal_tree(tmp_path)

    artifact = exp4581.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )
    out = json.loads((tmp_path / exp4581.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4581.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == out
    assert out["honest_verdict"] == "success: ar25_L2_offline_reproduced"
    assert out["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert out["preconditions_checked"]["spec_refs_present"] is True
    assert out["verifier_checkpoint_updated"] is True
    assert registry["reproducible_total_levels"] == 54
