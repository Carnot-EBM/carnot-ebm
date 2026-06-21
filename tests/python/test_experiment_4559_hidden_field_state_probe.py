"""Tests for Exp 4559 hidden-field state-key probe.

Spec refs: REQ-ARC-WMTE-4559, SCENARIO-ARC-WMTE-4559.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from carnot import experiment_4559_hidden_field_state_probe as exp4559
from carnot.agentic import arc_game_adapters


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


class FakeLevel:
    def __init__(self, data: dict[str, object] | None = None):
        self._data = data or {}

    def get_data(self, name: str) -> object | None:
        return self._data.get(name)


def _frame() -> SimpleNamespace:
    return SimpleNamespace(levels_completed=1, grid=np.zeros((4, 4), dtype=np.int16))


def _sprite(
    *,
    x: int = 0,
    y: int = 0,
    name: str = "s",
    tags: tuple[str, ...] = ("Hkx",),
    color: int = 9,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        tags=list(tags),
        x=x,
        y=y,
        width=3,
        height=3,
        pixels=np.array([[0, 0, 0], [0, color, 0], [0, 0, 0]], dtype=np.int16),
    )


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_import_smoke": True,
        "spec_refs_present": True,
    }


def test_req_arc_wmte_4559_spec_declares_hidden_field_contract() -> None:
    """REQ-ARC-WMTE-4559: OpenSpec names the hidden-field state-key artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4559.SPEC_REFS:
        assert ref in spec
    assert exp4559.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4559.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec
    for phrase in (
        "ka59",
        "StepCounter",
        "ar25",
        "undo-stack depth",
        "ft09",
        "color-cycle",
        "state_disambiguation_control_passed=true",
        "false_negative_risk_checked=true",
    ):
        assert phrase in spec


def test_scenario_arc_wmte_4559_adapter_keys_split_grid_aliased_hidden_states() -> None:
    """SCENARIO-ARC-WMTE-4559: adapter keys read live hidden registers."""

    frame = _frame()
    grid_key = exp4559.grid_only_state_key(frame)

    ka59 = arc_game_adapters.get_adapter("ka59")
    ka59_a = SimpleNamespace(
        current_level=FakeLevel({"StepCounter": 100}),
        urgssjskot=SimpleNamespace(current_steps=100, koyyeuyzyr=100),
    )
    ka59_b = SimpleNamespace(
        current_level=FakeLevel({"StepCounter": 100}),
        urgssjskot=SimpleNamespace(current_steps=99, koyyeuyzyr=100),
    )
    assert exp4559.grid_only_state_key(frame) == grid_key
    assert ka59 is not None
    assert ka59.state_key(ka59_a, frame) != ka59.state_key(ka59_b, frame)
    assert exp4559.hidden_fields_from_game("ka59", ka59_b)["step_counter_current_steps"] == 99

    ar25 = arc_game_adapters.get_adapter("ar25")
    ar25_a = SimpleNamespace(
        current_level=FakeLevel({"StepCounter": 64}),
        lelsvjlwneo=SimpleNamespace(current_steps=64, ilqnjlrnkk=64),
        flqblmrxsla=[],
    )
    ar25_b = SimpleNamespace(
        current_level=FakeLevel({"StepCounter": 64}),
        lelsvjlwneo=SimpleNamespace(current_steps=64, ilqnjlrnkk=64),
        flqblmrxsla=[{"undo": 1}],
    )
    assert exp4559.grid_only_state_key(frame) == grid_key
    assert ar25 is not None
    assert ar25.state_key(ar25_a, frame) != ar25.state_key(ar25_b, frame)
    assert exp4559.hidden_fields_from_game("ar25", ar25_b)["undo_stack_depth"] == 1

    ft09 = arc_game_adapters.get_adapter("ft09")
    ft09_a = SimpleNamespace(
        current_level=FakeLevel({"cwU": [9, 8]}),
        gqb=[9, 8],
        fhc=[_sprite(x=1, y=2, color=9)],
        mou=[],
        lpw=SimpleNamespace(dzy=32, oro=32),
        our=0,
    )
    ft09_b = SimpleNamespace(
        current_level=FakeLevel({"cwU": [8, 9]}),
        gqb=[8, 9],
        fhc=[_sprite(x=1, y=2, color=9)],
        mou=[],
        lpw=SimpleNamespace(dzy=32, oro=32),
        our=0,
    )
    assert exp4559.grid_only_state_key(frame) == grid_key
    assert ft09 is not None
    assert ft09.state_key(ft09_a, frame) != ft09.state_key(ft09_b, frame)
    assert exp4559.hidden_fields_from_game("ft09", ft09_b)["color_cycle"] == (8, 9)


def test_scenario_arc_wmte_4559_positive_control_passes_for_grid_only_alias() -> None:
    """SCENARIO-ARC-WMTE-4559: positive control guards honest no-bank nulls."""

    control = exp4559.build_state_disambiguation_control()

    assert control["passed"] is True
    assert set(control["per_game"]) == {"ka59", "ar25", "ft09"}
    for row in control["per_game"].values():
        assert row["grid_only_aliased"] is True
        assert row["extended_state_key_disambiguated"] is True


def test_req_arc_wmte_4559_helper_fallbacks_and_unreadable_control(monkeypatch) -> None:
    """REQ-ARC-WMTE-4559: helper fallbacks expose unreadable-register controls."""

    observation_frame = SimpleNamespace(levels_completed=3, observation=np.ones((2, 2), dtype=np.int16))
    no_grid_frame = SimpleNamespace(levels_completed=4)

    assert exp4559.grid_only_state_key(observation_frame)[0] == "grid"
    assert exp4559.grid_only_state_key(no_grid_frame) == ("grid", None, 4)
    assert exp4559._FakeLevel({"field": 7}).get_data("field") == 7
    assert exp4559._new_l2_levels({"reproduced": False, "reached_level": 2}) == 0
    assert exp4559._new_l2_levels({"reproduced": True, "reached_level": 1}) == 0
    assert exp4559._new_l2_levels({"reproduced": True, "reached_level": 2}) == 1

    original_get_adapter = arc_game_adapters.get_adapter

    def missing_ka59(game: str):
        if game == "ka59":
            return None
        return original_get_adapter(game)

    monkeypatch.setattr(arc_game_adapters, "get_adapter", missing_ka59)
    control = exp4559.build_state_disambiguation_control()

    assert control["passed"] is False
    assert control["per_game"]["ka59"]["extended_state_key_disambiguated"] is False


def test_req_arc_wmte_4559_artifact_schema_requires_control_for_no_bank() -> None:
    """REQ-ARC-WMTE-4559: no-bank artifacts require the disambiguation control."""

    attempts = [
        exp4559.DeepenAttempt(
            game="ka59",
            reached_level=1,
            offline_reproduced=False,
            reproduced_levels=0,
            solution_labels=[],
            reproduction_gate={"game": "ka59", "reached_level": 1, "reproduced": False},
            residual="ka59_l2_not_reproduced_after_step_counter_key",
        )
    ]
    artifact = exp4559.build_artifact(
        preconditions_checked=_preconditions(),
        attempts=attempts,
        state_control=exp4559.build_state_disambiguation_control(),
        registry_update={"updated": True, "path": exp4559.REGISTRY_RELATIVE_PATH},
    )

    assert artifact["honest_verdict"] == "complete: hidden_field_state_gap_sharpened_no_bank_honest_null"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["state_disambiguation_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["registry_updated"] is True
    assert artifact["missing_verifier_gaps"]
    assert artifact["schema_errors"] == []
    assert exp4559.artifact_schema_errors(artifact) == []

    unchecked = dict(artifact)
    unchecked["state_disambiguation_control_passed"] = False
    unchecked["false_negative_risk_checked"] = False
    assert "no-bank null requires positive control" in exp4559.artifact_schema_errors(unchecked)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    assert "checksum mismatch" in exp4559.artifact_schema_errors(bad_checksum)


def test_req_arc_wmte_4559_success_artifact_and_schema_error_branches() -> None:
    """REQ-ARC-WMTE-4559: success and malformed schemas are rejected explicitly."""

    attempt = exp4559.DeepenAttempt(
        game="ar25",
        reached_level=2,
        offline_reproduced=True,
        reproduced_levels=1,
        solution_labels=["3", "2"],
        reproduction_gate={"game": "ar25", "reached_level": 2, "reproduced": True},
        residual=None,
    )
    artifact = exp4559.build_artifact(
        preconditions_checked=_preconditions(),
        attempts=[attempt],
        state_control=exp4559.build_state_disambiguation_control(),
        registry_update={"updated": True, "path": exp4559.REGISTRY_RELATIVE_PATH},
    )

    assert artifact["honest_verdict"] == "success: hidden_field_state_ar25_L2_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["schema_errors"] == []

    mutations = [
        (lambda item: item.pop("honest_verdict"), "missing required field: honest_verdict"),
        (lambda item: item.__setitem__("experiment", "wrong"), "experiment mismatch"),
        (lambda item: item.__setitem__("schema", "wrong"), "schema mismatch"),
        (lambda item: item.__setitem__("spec_refs", []), "spec_refs mismatch"),
        (lambda item: item.__setitem__("field_principles", {}), "field_principles mismatch"),
        (lambda item: item.__setitem__("inference_substrate", "live_llm_inference"), "inference_substrate mismatch"),
        (lambda item: item.__setitem__("hidden_fields_added", {}), "hidden_fields_added mismatch"),
        (lambda item: item.__setitem__("random_seed", -1), "random_seed mismatch"),
        (lambda item: item.__setitem__("registry_updated", False), "registry_updated must be true"),
        (lambda item: item.__setitem__("honest_verdict", "bad"), "honest_verdict must start"),
    ]
    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4559.artifact_schema_errors(changed))

    false_success = dict(artifact)
    false_success["offline_reproduced"] = False
    assert "success artifact requires offline L2 reproduction" in exp4559.artifact_schema_errors(false_success)


def test_scenario_arc_wmte_4559_run_experiment_writes_json_and_registry(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4559: runner persists no-bank evidence and registry findings."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4559.SPEC_RELATIVE_PATH).write_text(
        "\n".join(exp4559.SPEC_REFS),
        encoding="utf-8",
    )
    (tmp_path / exp4559.REGISTRY_RELATIVE_PATH).write_text(
        "schema_version: 1\ngames: []\n",
        encoding="utf-8",
    )

    def fake_attempt(game: str, _target_level: int, _depth_cap: int) -> exp4559.DeepenAttempt:
        return exp4559.DeepenAttempt(
            game=game,
            reached_level=1,
            offline_reproduced=False,
            reproduced_levels=0,
            solution_labels=[],
            reproduction_gate={"game": game, "reached_level": 1, "reproduced": False},
            residual=f"{game}_l2_not_reproduced_after_hidden_key",
        )

    artifact = exp4559.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        attempt_runner=fake_attempt,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )

    written = json.loads((tmp_path / exp4559.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4559.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert written["registry_updated"] is True
    assert written["missing_verifier_gaps"] == [
        "ka59_l2_not_reproduced_after_hidden_key",
        "ar25_l2_not_reproduced_after_hidden_key",
        "ft09_l2_not_reproduced_after_hidden_key",
    ]
    assert registry["latest_hidden_field_state_probe_4559"]["artifact"] == exp4559.RESULT_RELATIVE_PATH

    rewritten_text, update = exp4559.apply_registry_probe(
        (tmp_path / exp4559.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"),
        artifact=artifact,
    )
    assert update["updated"] is False
    assert rewritten_text.count("latest_hidden_field_state_probe_4559:") == 1
