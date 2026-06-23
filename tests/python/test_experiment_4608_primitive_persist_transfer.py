"""Tests for Exp 4608 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4608, SCENARIO-ARC-WMTE-4608.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import yaml

from carnot import experiment_4608_primitive_persist_transfer as mod
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit
from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _transition(i: int) -> Transition:
    grid = np.array([[i, 0], [0, 0]], dtype=np.int16)
    next_grid = grid.copy()
    next_grid[0, 0] = i + 1
    next_grid[0, 1] = 1
    return Transition(grid, 1, None, next_grid, 0, 0)


def _identity(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    return np.asarray(grid).copy()


def _partial_change(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    pred = np.asarray(grid).copy()
    pred[0, 0] = int(pred[0, 0]) + 1
    return pred


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_wmte_4608_spec_declares_persisted_trust_transfer_contract() -> None:
    """REQ-ARC-WMTE-4608: OpenSpec declares the persisted trust primitive."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4608",
        "SCENARIO-ARC-WMTE-4608",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4608_solver_kit_trust_operator_promotes_heldout_model() -> None:
    """REQ-ARC-WMTE-4608: persisted operator ranks by change-weighted held-out trust."""

    result = kit.world_model_trust_energy_gate_operator(
        [_transition(i) for i in range(6)],
        [
            {"name": "identity", "engine": _identity},
            {"name": "partial_change", "engine": _partial_change},
        ],
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["selected_candidate_name"] == "partial_change"
    assert result["trust_pass"] is True
    assert result["binary_gate_pass"] is False
    assert result["value_added"] is True
    assert result["verifier_is_oracle"] is False
    assert result["selected_score"]["correct_changed_cells"] > 0


def test_req_arc_wmte_4608_routing_and_registry_surface_trust_operator() -> None:
    """REQ-ARC-WMTE-4608: routing and registry expose the reusable trust gate."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}

    selected = kit.select_primitive_operators(mechanic_class="object_motion_world_model")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("ka59")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert mod.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "trust" in gotchas[0]["note"].lower()
    assert "latest_exp4608_transfer" in gotchas[0]


def test_req_arc_wmte_4608_selects_a1_trust_over_a2_null() -> None:
    """REQ-ARC-WMTE-4608: A1 wins as the strongest persisted primitive."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "success: world_model_trust_energy_pass_rate_up_6_first_win_up",
            "trust_pass_rate_delta": 1.0,
            "first_win_delta": 1.0,
            "world_model_games": ["ar25", "cn04"],
        },
        a2_artifact={
            "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
            "first_win_delta": 0.0,
            "actions_delta": 0.0,
        },
    )

    assert decision["source"] == "A1_world_model_trust_energy"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["measured_signal"] > 0.0
    assert decision["source_tuning_games"] == ["ar25", "cn04"]


def test_req_arc_wmte_4608_transfer_measurement_reports_value_and_null() -> None:
    """REQ-ARC-WMTE-4608: transfer records per-game trust-pass value-add."""

    value = mod.measure_trust_transfer_game(
        "bp35",
        source_tuning_games=("ar25",),
        transitions=[_transition(i) for i in range(6)],
        candidates=[
            {"name": "identity", "engine": _identity},
            {"name": "partial_change", "engine": _partial_change},
        ],
    )
    assert value["game"] == "bp35"
    assert value["not_tuned_on_source"] is True
    assert value["value_added"] is True
    assert value["transfer_value"]["trust_pass"] is True
    assert value["transfer_value"]["binary_gate_pass"] is False

    null = mod.measure_trust_transfer_game(
        "ar25",
        source_tuning_games=("ar25",),
        transitions=[_transition(i) for i in range(6)],
        candidates=[
            {"name": "identity", "engine": _identity},
            {"name": "partial_change", "engine": _partial_change},
        ],
    )
    assert null["value_added"] is False
    assert null["not_tuned_on_source"] is False
    assert "source tuning game" in null["dead_end"]


def test_scenario_arc_wmte_4608_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4608: artifact schema records value-add or honest null."""

    decision = {
        "source": "A1_world_model_trust_energy",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": ["ar25"],
        "selection_rationale": "A1 raised trust pass and first-win while A2 was null.",
    }
    transfer_results = [
        {
            "game": "bp35",
            "value_added": True,
            "transfer_value": {
                "trust_pass": True,
                "first_win_lift": True,
                "binary_gate_pass": False,
                "offline_reproduced_new_level": False,
                "value_added": True,
            },
            "dead_end": "",
        },
        {
            "game": "dc22",
            "value_added": False,
            "transfer_value": {
                "trust_pass": False,
                "first_win_lift": False,
                "binary_gate_pass": False,
                "offline_reproduced_new_level": False,
                "value_added": False,
            },
            "dead_end": "no trusted candidate",
        },
    ]
    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={"A1_world_model_trust_energy": {"measured_signal": 1.0}},
        preconditions_checked={"ok": True},
        transfer_results=transfer_results,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success: primitive_persisted_transfer_bp35_value_added"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"]["new_levels_banked"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    null_artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[dict(row, value_added=False, dead_end="no transfer") for row in transfer_results],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert null_artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert mod.artifact_schema_errors(null_artifact) == []

    errors = mod.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least two games" in errors


def test_scenario_arc_wmte_4608_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4608: run writes the requested result JSON."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry = {
        "schema_version": 1,
        "general_gotchas": [
            {"id": mod.PRIMITIVE_GOTCHA_ID, "operator": mod.PRIMITIVE_OPERATOR, "note": "fixture"}
        ],
        "games": [],
    }
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    _write_json(
        tmp_path / mod.A1_RELATIVE_PATH,
        {
            "honest_verdict": "success: world_model_trust_energy_pass_rate_up_6_first_win_up",
            "trust_pass_rate_delta": 1.0,
            "first_win_delta": 1.0,
            "world_model_games": ["ar25", "cn04"],
        },
    )
    _write_json(
        tmp_path / mod.A2_RELATIVE_PATH,
        {
            "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
            "first_win_delta": 0.0,
            "actions_delta": 0.0,
        },
    )

    artifact = mod.run(
        tmp_path,
        transfer_games=("bp35", "dc22"),
        offline_arcade_checker=lambda: True,
        now=iter([5.0, 5.25]).__next__,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["transfer_games"] == ["bp35", "dc22"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    blocked = mod.build_artifact(
        selected_upstream={
            "source": "A1_world_model_trust_energy",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
            "source_tuning_games": ["ar25"],
        },
        upstream_signals={},
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []


def test_req_arc_wmte_4608_defensive_branches_are_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4608: defensive branches remain explicit and schema-gated."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_json) == {}

    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": {}}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_int(False) == 0

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]

    assert mod._source_tuning_games_from_a1(
        {"measurements": [{"game": "bp35"}, {"game": "bp35"}, {}]}
    ) == ["bp35"]

    a2_wins = mod.select_primitive_from_upstreams(
        a1_artifact={"trust_pass_rate_delta": 0.0},
        a2_artifact={"first_win_delta": 0.5},
    )
    assert "A2 had the larger" in a2_wins["selection_rationale"]
    all_null = mod.select_primitive_from_upstreams(a1_artifact={}, a2_artifact={})
    assert "value-null" in all_null["selection_rationale"]

    decision = {
        "source": "A1_world_model_trust_energy",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "source_tuning_games": [],
    }
    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "bp35",
                "value_added": True,
                "transfer_value": {
                    "trust_pass": True,
                    "first_win_lift": True,
                    "binary_gate_pass": False,
                    "offline_reproduced_new_level": True,
                    "existing_reproduced_level": True,
                    "value_added": True,
                },
                "dead_end": "",
            },
            {
                "game": "dc22",
                "value_added": False,
                "transfer_value": {"value_added": False},
                "dead_end": "no transfer",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    assert artifact["offline_reproduced"]["new_levels_banked"] == 1
    assert artifact["offline_reproduced"]["existing_source_levels"]["bp35"] == 0

    wrong_gotcha = dict(artifact)
    wrong_gotcha["primitive_persisted"] = dict(artifact["primitive_persisted"])
    wrong_gotcha["primitive_persisted"]["registry_general_gotcha_id"] = "wrong"
    wrong_gotcha["reproducibility_checksum"] = mod.payload_checksum(wrong_gotcha)
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    fake_success = dict(artifact)
    fake_success["transfer_value_per_game"] = {"bp35": {"value_added": False}}
    fake_success["reproducibility_checksum"] = mod.payload_checksum(fake_success)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        fake_success
    )

    bank_mismatch = dict(artifact)
    bank_mismatch["offline_reproduced"] = dict(artifact["offline_reproduced"])
    bank_mismatch["offline_reproduced"]["new_level_records"] = []
    bank_mismatch["reproducibility_checksum"] = mod.payload_checksum(bank_mismatch)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        bank_mismatch
    )

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum must match artifact content" in mod.artifact_schema_errors(
        bad_checksum
    )

    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        mod.run(tmp_path, offline_arcade_checker=lambda: True, write=False)
