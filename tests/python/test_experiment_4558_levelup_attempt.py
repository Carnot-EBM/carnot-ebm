"""Tests for Exp 4558 ARC sprint rotation level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4558, SCENARIO-ARC-WMTE-4558.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4558_levelup_attempt as exp4558


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _loop_result(
    game: str = "m0r0",
    reached_level: int = 2,
    reproduced: bool = True,
    labels: list[str] | None = None,
) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "states_expanded": 42,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "reproduction_gate": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": labels or ["{\"action\": 1}", "{\"action\": 4}"],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _route_only(game: str) -> dict[str, object]:
    return {
        "game": game,
        "status": "needs_per_game_RE",
        "recommended_approach": "object_motion_world_model",
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: existing L1 route.
  action_model: existing.
  solver: existing.
  gotchas: []
- game: sk48
  reproducibility: provisional
  levels_reproduced: 1
  win_condition: routed but not counted in this fixture.
  action_model: existing.
  solver: existing.
  gotchas: []
- game: ar25
  reproducibility: provisional
  levels_reproduced: 1
  win_condition: routed but not counted in this fixture.
  action_model: existing.
  solver: existing.
  gotchas: []
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
  win_condition: existing L2 route.
  action_model: existing.
  solver: existing.
  reproduce: existing L2 gate.
  gotchas: []
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: existing L1 route.
  action_model: existing.
  solver: existing.
  gotchas: []
reproducible_total_levels: 4
reproducible_total_games: 3
"""


def _write_minimal_tree(tmp_path: Path, *, m0r0: dict[str, object] | None = None) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4558.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4558.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-WMTE-4558\nSCENARIO-ARC-WMTE-4558\n",
        encoding="utf-8",
    )
    for game, payload in {
        "cn04": _route_only("cn04"),
        "sk48": _route_only("sk48"),
        "ar25": _route_only("ar25"),
        "m0r0": m0r0 or _loop_result("m0r0", reached_level=2),
        "dc22": _loop_result("dc22", reached_level=1),
    }.items():
        (tmp_path / exp4558.loop_result_path(game)).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def test_req_arc_wmte_4558_spec_declares_required_contract() -> None:
    """REQ-ARC-WMTE-4558: OpenSpec names the attempt ledger contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4558.SPEC_REFS:
        assert ref in spec
    assert exp4558.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4558.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec
    for required in (
        "`success: <game>_L<n>_offline_reproduced`",
        "`complete: <game>_delta_identified_no_bank`",
        "`reproduced_levels=0`",
        "`registry_updated=false`",
        "dead-end evidence",
    ):
        assert required in spec


def test_scenario_arc_wmte_4558_registry_update_banks_only_reproduced_advance() -> None:
    """SCENARIO-ARC-WMTE-4558: registry increments only past the prior level."""

    advancing_loop = _loop_result("m0r0", reached_level=3)
    checksum = exp4558.reproducibility_checksum(
        {
            "target_game": "m0r0",
            "reproduction_gate": advancing_loop["reproduction_gate"],
            "solution_labels": advancing_loop["solution_labels"],
        }
    )

    updated_text, update = exp4558.apply_registry_bank(
        _registry_text(),
        loop_result=advancing_loop,
        checksum=checksum,
        artifact_path=exp4558.RESULT_RELATIVE_PATH,
    )
    registry = yaml.safe_load(updated_text)
    m0r0 = next(row for row in registry["games"] if row["game"] == "m0r0")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 2
    assert update["new_game_levels"] == 3
    assert update["banked_levels"] == 1
    assert update["new_total_declared"] == 5
    assert update["new_total_row_sum"] == 5
    assert registry["reproducible_total_levels"] == 5
    assert m0r0["levels_reproduced"] == 3
    assert exp4558.RESULT_RELATIVE_PATH in m0r0["reproduce"]
    assert checksum in m0r0["reproduce"]

    unchanged_text, unchanged = exp4558.apply_registry_bank(
        _registry_text(),
        loop_result=_loop_result("m0r0", reached_level=2),
        checksum="unused",
        artifact_path=exp4558.RESULT_RELATIVE_PATH,
    )
    assert unchanged_text == _registry_text()
    assert unchanged["updated"] is False
    assert unchanged["banked_levels"] == 0

    unreproduced_text, unreproduced = exp4558.apply_registry_bank(
        _registry_text(),
        loop_result=_loop_result("m0r0", reached_level=3, reproduced=False),
        checksum="unused",
        artifact_path=exp4558.RESULT_RELATIVE_PATH,
    )
    assert unreproduced_text == _registry_text()
    assert unreproduced["reason"] == "not_offline_reproduced"

    try:
        exp4558.apply_registry_bank(
            "schema_version: 1\ngames: []\nreproducible_total_levels: 0\n",
            loop_result=_loop_result("m0r0", reached_level=3),
            checksum="unused",
            artifact_path=exp4558.RESULT_RELATIVE_PATH,
        )
    except ValueError as exc:
        assert "registry missing game row: m0r0" in str(exc)
    else:
        raise AssertionError("missing m0r0 row should fail")


def test_req_arc_wmte_4558_candidate_selection_prefers_bank_then_reproduced_rotation() -> None:
    """REQ-ARC-WMTE-4558: cached loop evidence is ranked against registry rows."""

    registry = yaml.safe_load(_registry_text())
    route = ("cn04", exp4558.loop_result_path("cn04"), _route_only("cn04"))
    stall = ("m0r0", exp4558.loop_result_path("m0r0"), _loop_result("m0r0", reached_level=2))
    existing = ("dc22", exp4558.loop_result_path("dc22"), _loop_result("dc22", reached_level=1))

    game, path, selected = exp4558.choose_loop_result([route, stall, existing], registry)
    assert game == "m0r0"
    assert path == exp4558.loop_result_path("m0r0")
    assert selected["reproduction_gate"]["reached_level"] == 2

    bank = ("dc22", exp4558.loop_result_path("dc22"), _loop_result("dc22", reached_level=2))
    bank_game, _, banked = exp4558.choose_loop_result([route, stall, bank], registry)
    assert bank_game == "dc22"
    assert banked["reproduction_gate"]["reached_level"] == 2

    try:
        exp4558.choose_loop_result([], registry)
    except FileNotFoundError as exc:
        assert "no cached ARC loop results" in str(exc)
    else:
        raise AssertionError("empty candidate list should fail")


def test_req_arc_wmte_4558_artifact_schema_success_and_no_bank() -> None:
    """REQ-ARC-WMTE-4558: artifact success is gated by reproduced registry advance."""

    success_update = {
        "updated": True,
        "path": exp4558.REGISTRY_RELATIVE_PATH,
        "target_game": "m0r0",
        "prior_game_levels": 2,
        "new_game_levels": 3,
        "banked_levels": 1,
        "prior_total_declared": 4,
        "prior_total_row_sum": 4,
        "new_total_declared": 5,
        "new_total_row_sum": 5,
        "reconciled_total_delta": 1,
        "reason": "banked_offline_reproduced_level",
    }
    success = exp4558.build_artifact(
        loop_result=_loop_result("m0r0", reached_level=3),
        registry_update=success_update,
        preconditions_checked={
            "AGENTS.md": True,
            "CODEX.md": True,
            "offline_arcade_import_smoke": True,
            "spec_refs_present": True,
        },
        dead_ends=exp4558.DEFAULT_DEAD_ENDS,
        arc_loop_result_path=exp4558.loop_result_path("m0r0"),
    )

    assert success["honest_verdict"] == "success: m0r0_L3_offline_reproduced"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["registry_updated"] is True
    assert success["schema_errors"] == []
    assert exp4558.artifact_schema_errors(success) == []

    no_bank = exp4558.build_artifact(
        loop_result=_loop_result("m0r0", reached_level=2),
        registry_update={**success_update, "updated": False, "banked_levels": 0},
        preconditions_checked={"offline_arcade_import_smoke": True, "spec_refs_present": True},
        dead_ends=[],
        arc_loop_result_path=exp4558.loop_result_path("m0r0"),
    )
    assert no_bank["honest_verdict"] == "complete: m0r0_delta_identified_no_bank"
    assert no_bank["reproduced_levels"] == 0
    assert no_bank["schema_errors"] == []

    changed = dict(success)
    changed["reproducibility_checksum"] = "bad"
    assert "checksum mismatch" in exp4558.artifact_schema_errors(changed)

    false_success = dict(success)
    false_success["offline_reproduced"] = False
    assert "success artifact missing reproduced registry bank" in exp4558.artifact_schema_errors(false_success)

    missing = dict(success)
    missing.pop("honest_verdict")
    assert "missing required field: honest_verdict" in exp4558.artifact_schema_errors(missing)

    mismatched = dict(success)
    mismatched.update(
        {
            "experiment": "wrong",
            "schema": "wrong",
            "spec_refs": [],
            "field_principles": {},
            "random_seed": -1,
        }
    )
    mismatch_errors = exp4558.artifact_schema_errors(mismatched)
    assert "experiment mismatch" in mismatch_errors
    assert "schema mismatch" in mismatch_errors
    assert "spec_refs mismatch" in mismatch_errors
    assert "field_principles mismatch" in mismatch_errors
    assert "random_seed mismatch" in mismatch_errors


def test_scenario_arc_wmte_4558_run_experiment_writes_honest_no_bank_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4558: repeated-level evidence writes a no-bank artifact."""

    _write_minimal_tree(tmp_path)
    before_registry = (tmp_path / exp4558.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = exp4558.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )

    out = json.loads((tmp_path / exp4558.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    after_registry = (tmp_path / exp4558.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")

    assert artifact == out
    assert out["target_game"] == "m0r0"
    assert out["honest_verdict"] == "complete: m0r0_delta_identified_no_bank"
    assert out["offline_reproduced"] is True
    assert out["reproduced_levels"] == 0
    assert out["registry_updated"] is False
    assert out["registry_update"]["new_total_declared"] == 4
    assert out["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert out["preconditions_checked"]["spec_refs_present"] is True
    assert after_registry == before_registry


def test_scenario_arc_wmte_4558_run_experiment_writes_registry_on_bank(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4558: a reproduced next level updates the registry."""

    _write_minimal_tree(tmp_path, m0r0=_loop_result("m0r0", reached_level=3))

    artifact = exp4558.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )
    registry = yaml.safe_load((tmp_path / exp4558.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    m0r0 = next(row for row in registry["games"] if row["game"] == "m0r0")

    assert artifact["honest_verdict"] == "success: m0r0_L3_offline_reproduced"
    assert artifact["registry_update"]["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 5
    assert m0r0["levels_reproduced"] == 3
