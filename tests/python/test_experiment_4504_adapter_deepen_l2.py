"""Tests for Exp 4504 adapter-routed ARC L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4504, SCENARIO-ARC-WMTE-4504.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4504_adapter_deepen_l2 as exp4504
from carnot.agentic.arc_game_adapters import adaptered_games, get_adapter


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "test",
    }


def test_req_arc_wmte_4504_spec_declares_adapter_deepen_artifact() -> None:
    """REQ-ARC-WMTE-4504: OpenSpec names the 4504 artifact and gate fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in ("REQ-ARC-WMTE-4504", "SCENARIO-ARC-WMTE-4504"):
        assert ref in spec
    assert exp4504.RESULT_RELATIVE_PATH in spec
    for phrase in (
        "cd82",
        "GameAdapter",
        "adapter_registered",
        "solution_labels",
        "offline_reproduced=true",
        "reproduced_levels >= 1",
        "beyond the prior L1",
    ):
        assert phrase in spec
    for field, principle in exp4504.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4504_cd82_adapter_is_registered_for_verifier_routing() -> None:
    """REQ-ARC-WMTE-4504: the selected tractable game has a registered adapter."""

    adapter = get_adapter(exp4504.TARGET_GAME)

    assert exp4504.TARGET_GAME == "cd82"
    assert exp4504.TARGET_GAME in adaptered_games()
    assert adapter is not None
    assert adapter.game == "cd82"
    assert callable(adapter.action_labels)
    assert callable(adapter.apply)
    assert callable(adapter.state_key)
    assert callable(adapter.hand_verifier)
    assert adapter.branch_mode == "fresh_env"


def test_req_arc_wmte_4504_success_artifact_requires_l2_reproduction(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4504: success requires one newly reproduced level beyond L1."""

    artifact = exp4504.build_artifact(
        preconditions_checked=_preconditions(),
        target_game="cd82",
        adapter_registered=True,
        solution_labels=[json.dumps({"action": 5})],
        solve_reached_level=2,
        reproduction_gate={"game": "cd82", "reached_level": 2, "reproduced": True},
        depth_cap=80,
        states_expanded=12,
        tests_pass=True,
        adapter_branch_mode="fresh_env",
    )

    assert artifact["experiment"] == "experiment_4504_adapter_deepen_l2"
    assert artifact["schema"] == "carnot.adapter_deepen_l2_4504.v1"
    assert artifact["spec_refs"] == exp4504.SPEC_REFS
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["schema_errors"] == []
    assert exp4504.artifact_schema_errors(artifact) == []

    out = exp4504.write_artifact(artifact, root=tmp_path)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    fabricated = dict(artifact)
    fabricated["offline_reproduced"] = False
    assert any("success artifact" in error for error in exp4504.artifact_schema_errors(fabricated))
    with pytest.raises(ValueError, match="success artifact"):
        exp4504.write_artifact(fabricated, root=tmp_path)

    mutations = [
        (lambda item: item.__setitem__("experiment", "experiment_4494_adapter_deepen_l2"), "experiment"),
        (lambda item: item.__setitem__("schema", "bad"), "schema"),
        (lambda item: item.__setitem__("spec_refs", []), "spec_refs"),
        (lambda item: item.__setitem__("target_game", "lf52"), "target_game"),
    ]
    for mutate, expected in mutations:
        changed = dict(artifact)
        mutate(changed)
        assert any(expected in error for error in exp4504.artifact_schema_errors(changed))


def test_scenario_arc_wmte_4504_runner_writes_injected_cd82_l2_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4504: injected solver/replay success writes stable JSON."""

    fake_adapter = SimpleNamespace(
        game="cd82",
        apply=lambda env, label, frame: frame,
        warmup_label=None,
        depth_caps={2: 80},
        branch_mode="fresh_env",
    )
    calls: list[tuple[str, int, int]] = []

    def fake_solver_runner(game: str, adapter: object, target_level: int, depth_cap: int):
        calls.append((game, target_level, depth_cap))
        return [json.dumps({"action": 5})], 2, 7

    def fake_reproduction_runner(
        game: str,
        labels: list[str],
        apply_fn: object,
        *,
        warmup_label: str | None,
        claimed_level: int,
    ) -> dict[str, object]:
        return {
            "game": game,
            "claimed_level": claimed_level,
            "reached_level": claimed_level,
            "reproduced": bool(labels),
        }

    artifact = exp4504.run_experiment(
        root=tmp_path,
        adapter_lookup=lambda game: fake_adapter,
        solver_runner=fake_solver_runner,
        reproduction_runner=fake_reproduction_runner,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )
    written = json.loads((tmp_path / exp4504.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert calls == [("cd82", 2, 80)]
    assert artifact == written
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["adapter_registered"] is True


def test_scenario_arc_wmte_4504_runner_reports_reverse_engineering_delta(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4504: blocked L2 emits RE delta, not a fabricated success."""

    artifact = exp4504.run_experiment(
        root=tmp_path,
        adapter_lookup=lambda game: None,
        preconditions_checked=_preconditions(),
        tests_pass=True,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["adapter_registered"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["residual_blockers"] == [
        "cd82_adapter_not_registered",
        "cd82_solver_reached_level_1",
        "cd82_l2_not_reproduced",
    ]
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4504_runner_rejects_missing_resource_preconditions() -> None:
    """REQ-ARC-WMTE-4504: missing import or torch preconditions block before replay."""

    with pytest.raises(RuntimeError, match="blocked_offline_arcade_import_smoke"):
        exp4504.ensure_preconditions_ready(
            {"offline_arcade_import_smoke": False, "torch_import": True}
        )
    with pytest.raises(RuntimeError, match="blocked_torch_import"):
        exp4504.ensure_preconditions_ready(
            {"offline_arcade_import_smoke": True, "torch_import": False}
        )
