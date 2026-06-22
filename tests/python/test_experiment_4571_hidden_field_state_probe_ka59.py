"""Tests for Exp 4571 ka59 hidden-field state-key probe.

Spec refs: REQ-ARC-WMTE-4571, SCENARIO-ARC-WMTE-4571.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from carnot import experiment_4571_hidden_field_state_probe_ka59 as exp4571
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


def _ka59_game(current_steps: int, limit: int = 127) -> SimpleNamespace:
    return SimpleNamespace(
        current_level=FakeLevel({"StepCounter": limit}),
        urgssjskot=SimpleNamespace(current_steps=current_steps, koyyeuyzyr=limit),
    )


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_import_smoke": True,
        "spec_refs_present": True,
    }


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
  gotchas:
  - existing gotcha
"""


def _write_minimal_tree(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "openspec" / "capabilities" / "arc-world-model-trust-energy").mkdir(parents=True)
    (tmp_path / exp4571.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    (tmp_path / exp4571.SPEC_RELATIVE_PATH).write_text(
        "\n".join(exp4571.SPEC_REFS),
        encoding="utf-8",
    )


def test_req_arc_wmte_4571_spec_declares_ka59_contract() -> None:
    """REQ-ARC-WMTE-4571: OpenSpec declares the ka59-only hidden-field contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in exp4571.SPEC_REFS:
        assert ref in spec
    assert exp4571.RESULT_RELATIVE_PATH in spec
    assert "game.urgssjskot.current_steps" in spec
    assert "state_disambiguation_control_passed=false" in spec
    for field, principle in exp4571.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_wmte_4571_adapter_key_reads_step_counter_register() -> None:
    """SCENARIO-ARC-WMTE-4571: ka59 adapter keys split grid-aliased StepCounter states."""

    frame = _frame()
    left = _ka59_game(current_steps=114)
    right = _ka59_game(current_steps=113)
    adapter = arc_game_adapters.get_adapter("ka59")

    assert adapter is not None
    assert exp4571.grid_only_state_key(frame) == exp4571.grid_only_state_key(frame)
    assert adapter.state_key(left, frame) != adapter.state_key(right, frame)
    assert exp4571.hidden_fields_from_game(right)["step_counter_current_steps"] == 113
    assert exp4571.hidden_fields_from_game(right)["step_counter_limit"] == 127


def test_scenario_arc_wmte_4571_positive_control_passes_for_step_counter_pair() -> None:
    """SCENARIO-ARC-WMTE-4571: positive control is the hard gate for no-bank nulls."""

    control = exp4571.build_state_disambiguation_control(
        pair_builder=lambda: exp4571.StatePair(
            frame=_frame(),
            left_game=_ka59_game(current_steps=114),
            right_game=_ka59_game(current_steps=113),
            left_path=("C:1", "1"),
            right_path=("1", "1", "2"),
        )
    )

    assert control["passed"] is True
    assert control["game"] == "ka59"
    assert control["grid_only_aliased"] is True
    assert control["extended_state_key_disambiguated"] is True
    assert control["left_path"] == ["C:1", "1"]
    assert control["right_hidden_fields"]["step_counter_current_steps"] == 113


def test_req_arc_wmte_4571_artifact_schema_success_no_bank_and_control_failure() -> None:
    """REQ-ARC-WMTE-4571: artifacts distinguish success, gated null, and no-op control failure."""

    control = exp4571.build_state_disambiguation_control(
        pair_builder=lambda: exp4571.StatePair(
            frame=_frame(),
            left_game=_ka59_game(current_steps=114),
            right_game=_ka59_game(current_steps=113),
        )
    )
    no_bank_attempt = exp4571.Ka59DeepenAttempt(
        reached_level=1,
        offline_reproduced=False,
        reproduced_levels=0,
        solution_labels=["4"],
        reproduction_gate={"game": "ka59", "reached_level": 1, "reproduced": False},
        residual="ka59_l2_not_reproduced_after_step_counter_state_key",
        states_expanded=12,
    )
    no_bank = exp4571.build_artifact(
        preconditions_checked=_preconditions(),
        attempt=no_bank_attempt,
        state_control=control,
        registry_update={"updated": True, "path": exp4571.REGISTRY_RELATIVE_PATH},
    )

    assert no_bank["honest_verdict"] == (
        "complete: hidden_field_state_ka59_gap_sharpened_no_bank_honest_null"
    )
    assert no_bank["state_disambiguation_control_passed"] is True
    assert no_bank["false_negative_risk_checked"] is True
    assert no_bank["offline_reproduced"] is False
    assert no_bank["missing_verifier_gaps"] == [
        "ka59_l2_not_reproduced_after_step_counter_state_key"
    ]
    assert no_bank["schema_errors"] == []

    success_attempt = exp4571.Ka59DeepenAttempt(
        reached_level=2,
        offline_reproduced=True,
        reproduced_levels=1,
        solution_labels=["4", "2"],
        reproduction_gate={"game": "ka59", "reached_level": 2, "reproduced": True},
        residual=None,
        states_expanded=3,
    )
    success = exp4571.build_artifact(
        preconditions_checked=_preconditions(),
        attempt=success_attempt,
        state_control=control,
        registry_update={"updated": True, "path": exp4571.REGISTRY_RELATIVE_PATH},
    )
    assert success["honest_verdict"] == "success: hidden_field_state_ka59_L2_offline_reproduced"
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["missing_verifier_gaps"] == []
    assert success["schema_errors"] == []

    failed_control = {"passed": False, "unreadable_register": "step_counter_current_steps"}
    gated = exp4571.build_artifact(
        preconditions_checked=_preconditions(),
        attempt=None,
        state_control=failed_control,
        registry_update={"updated": True, "path": exp4571.REGISTRY_RELATIVE_PATH},
    )
    assert gated["state_disambiguation_control_passed"] is False
    assert gated["false_negative_risk_checked"] is False
    assert gated["attempts"] == []
    assert gated["missing_verifier_gaps"] == [
        "ka59_step_counter_current_steps_unreadable_control_failed"
    ]
    assert gated["schema_errors"] == []

    changed = dict(success)
    changed["reproducibility_checksum"] = "bad"
    assert "checksum mismatch" in exp4571.artifact_schema_errors(changed)

    false_success = dict(success)
    false_success["offline_reproduced"] = False
    assert "success artifact requires ka59 L2 offline reproduction" in exp4571.artifact_schema_errors(
        false_success
    )


def test_req_arc_wmte_4571_defensive_schema_and_control_branches(monkeypatch) -> None:
    """REQ-ARC-WMTE-4571: defensive branches still report explicit schema/gap reasons."""

    assert exp4571._new_l2_levels({"reproduced": False, "reached_level": 2}) == 0
    assert exp4571._new_l2_levels({"reproduced": True, "reached_level": 1}) == 0
    assert exp4571._new_l2_levels({"reproduced": True, "reached_level": 2}) == 1

    monkeypatch.setattr(arc_game_adapters, "get_adapter", lambda _game: None)
    missing_adapter = exp4571.build_state_disambiguation_control(
        pair_builder=lambda: exp4571.StatePair(
            frame=_frame(),
            left_game=_ka59_game(current_steps=114),
            right_game=_ka59_game(current_steps=113),
        )
    )
    assert missing_adapter["passed"] is False
    assert missing_adapter["unreadable_register"] == "ka59_adapter_missing"

    control = {"passed": True}
    no_residual = exp4571.build_artifact(
        preconditions_checked=_preconditions(),
        attempt=exp4571.Ka59DeepenAttempt(
            reached_level=1,
            offline_reproduced=False,
            reproduced_levels=0,
            solution_labels=[],
            reproduction_gate={"game": "ka59", "reached_level": 1, "reproduced": False},
            residual=None,
        ),
        state_control=control,
        registry_update={"updated": True, "path": exp4571.REGISTRY_RELATIVE_PATH},
    )
    assert no_residual["missing_verifier_gaps"] == [
        "ka59_l2_not_reproduced_after_step_counter_state_key"
    ]

    malformed = dict(no_residual)
    malformed.pop("honest_verdict")
    malformed.update(
        {
            "experiment": "wrong",
            "schema": "wrong",
            "spec_refs": [],
            "field_principles": {},
            "inference_substrate": "live_llm",
            "hidden_fields_added": {},
            "random_seed": -1,
            "registry_updated": False,
        }
    )
    errors = exp4571.artifact_schema_errors(malformed)
    for expected in (
        "missing required field: honest_verdict",
        "experiment mismatch",
        "schema mismatch",
        "spec_refs mismatch",
        "field_principles mismatch",
        "inference_substrate mismatch",
        "hidden_fields_added mismatch",
        "random_seed mismatch",
        "registry_updated must be true",
        "honest_verdict must start with terminal prefix",
    ):
        assert any(expected in error for error in errors)

    bad_complete = dict(no_residual)
    bad_complete["offline_reproduced"] = True
    bad_complete["missing_verifier_gaps"] = []
    bad_complete["state_disambiguation_control_passed"] = False
    bad_complete["false_negative_risk_checked"] = True
    complete_errors = exp4571.artifact_schema_errors(bad_complete)
    assert "complete artifact must not claim offline reproduction" in complete_errors
    assert "complete artifact requires a verifier/register gap" in complete_errors
    assert "failed control must not claim false-negative risk checked" in complete_errors

    replacement = exp4571._replace_top_level_registry_section(
        "schema_version: 1\nlatest_hidden_field_state_probe_4571:\n  old: true\nnext:\n  value: true\n",
        "latest_hidden_field_state_probe_4571",
        "latest_hidden_field_state_probe_4571:\n  new: true\n",
    )
    assert "old: true" not in replacement
    assert "next:\n  value: true" in replacement


def test_scenario_arc_wmte_4571_run_experiment_gates_before_deepening(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4571: failed control writes a gap and skips the L2 attempt."""

    _write_minimal_tree(tmp_path)

    def should_not_run() -> exp4571.Ka59DeepenAttempt:
        raise AssertionError("attempt runner must not execute before a passing control")

    artifact = exp4571.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        control_builder=lambda: {"passed": False, "unreadable_register": "step_counter_current_steps"},
        attempt_runner=should_not_run,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )
    written = json.loads((tmp_path / exp4571.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4571.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact == written
    assert written["attempts"] == []
    assert written["state_disambiguation_control_passed"] is False
    assert registry["latest_hidden_field_state_probe_4571"]["artifact"] == exp4571.RESULT_RELATIVE_PATH


def test_scenario_arc_wmte_4571_run_experiment_writes_no_bank_json_and_registry(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4571: passing control permits one ka59 standing-loop attempt."""

    _write_minimal_tree(tmp_path)
    control = exp4571.build_state_disambiguation_control(
        pair_builder=lambda: exp4571.StatePair(
            frame=_frame(),
            left_game=_ka59_game(current_steps=114),
            right_game=_ka59_game(current_steps=113),
        )
    )

    def fake_attempt() -> exp4571.Ka59DeepenAttempt:
        return exp4571.Ka59DeepenAttempt(
            reached_level=1,
            offline_reproduced=False,
            reproduced_levels=0,
            solution_labels=["4"],
            reproduction_gate={"game": "ka59", "reached_level": 1, "reproduced": False},
            residual="ka59_l2_not_reproduced_after_step_counter_state_key",
            states_expanded=12,
        )

    artifact = exp4571.run_experiment(
        root=tmp_path,
        precondition_checker=lambda: True,
        control_builder=lambda: control,
        attempt_runner=fake_attempt,
        instructions_checked={"AGENTS.md": True, "CODEX.md": True},
    )
    written = json.loads((tmp_path / exp4571.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry_text = (tmp_path / exp4571.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    registry = yaml.safe_load(registry_text)

    assert artifact == written
    assert written["state_disambiguation_control_passed"] is True
    assert written["attempts"][0]["states_expanded"] == 12
    assert written["missing_verifier_gaps"] == [
        "ka59_l2_not_reproduced_after_step_counter_state_key"
    ]
    assert registry["latest_hidden_field_state_probe_4571"]["offline_reproduced"] is False

    rewritten_text, update = exp4571.apply_registry_probe(registry_text, artifact=artifact)
    assert update["updated"] is False
    assert rewritten_text.count("latest_hidden_field_state_probe_4571:") == 1
