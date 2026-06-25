"""Tests for Exp 4730 .435 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4730,
SCENARIO-ARC-WMTE-4730-PERSIST-STRONGEST-435-PRIMITIVE,
SCENARIO-ARC-WMTE-4730-LEAVE-ONE-GAME-TRANSFER.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4730_primitive_persist_transfer as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _row(
    game: str,
    state_key: str,
    action_id: int,
    *,
    changed: bool,
    x: int | None = None,
    y: int | None = None,
    online_warm_score: float = 0.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game": game,
        "env": game,
        "state_key": state_key,
        "action_id": action_id,
        "changed": changed,
        "frame_delta": 1.0 if changed else 0.0,
        "level_progress": 0.0,
        "online_warm_score": float(online_warm_score),
    }
    if x is not None and y is not None:
        row["x"] = int(x)
        row["y"] = int(y)
    return row


def _candidate(candidate_id: str, action_id: int, *, target: bool, score: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "action_id": action_id,
        "online_warm_score": float(score),
        "reaches_levelup": bool(target),
    }


def _a1_artifact(*, arms_non_degenerate: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4726_online_action_learning_driver_valid_test",
        "honest_verdict": "complete: online_action_learning_no_first_win_lift_residual_online_signal_genuinely_too_sparse",
        "arms_non_degenerate": bool(arms_non_degenerate),
        "per_arm_action_distribution_distinct": bool(arms_non_degenerate),
        "online_train_steps_executed": 66 if arms_non_degenerate else 0,
        "online_warm_vs_frozen_delta": 0.0,
        "online_warm_first_win": 0.04,
        "frozen_first_win": 0.04,
        "non_degeneracy_gate": {
            "arms_non_degenerate": bool(arms_non_degenerate),
            "train_steps_with_positive_grad_norm": 66 if arms_non_degenerate else 0,
            "coordinate_head_differs_from_frozen": bool(arms_non_degenerate),
        },
        "arm_source_artifacts": {
            "frozen": "results/experiment_4710_online_action_learning_arms_frozen.json",
            "online-scratch": "results/experiment_4710_online_action_learning_arms_online_scratch.json",
            "online-warm": "results/experiment_4710_online_action_learning_arms_online_warm_propose.json",
        },
    }


def _a2_artifact(*, probe_actions: int = 0) -> dict[str, Any]:
    return {
        "experiment": "experiment_4727_active_probe_disambiguation",
        "honest_verdict": "complete: active_probe_no_new_level_residual_budget_insufficient",
        "hypothesis_posterior_built": bool(probe_actions),
        "probe_actions_taken": int(probe_actions),
        "posterior_entropy_reduction": 0.25 if probe_actions else 0.0,
        "generic_agent_reached_level": 0,
        "offline_reproduced": False,
    }


def _arm_artifact(game: str, *, first_win: bool, actions: int = 10) -> dict[str, Any]:
    return {
        "variant_attempts": [
            {
                "game": game,
                "first_win": bool(first_win),
                "solved": bool(first_win),
                "actions": int(actions),
                "actions_to_first_levelup": int(actions) if first_win else None,
            }
        ]
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "general_gotchas": [
                    {
                        "id": mod.PRIMITIVE_GOTCHA_ID,
                        "operator": mod.PRIMITIVE_OPERATOR,
                        "note": "fixture online-warm action-effect controller",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())
    _write_json(tmp_path / mod.FROZEN_ARM_RELATIVE_PATH, _arm_artifact("zz99", first_win=False))
    _write_json(
        tmp_path / mod.ONLINE_WARM_ARM_RELATIVE_PATH,
        _arm_artifact("zz99", first_win=True, actions=4),
    )


def test_req_arc_wmte_4730_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4730: OpenSpec declares the persistence/transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4730",
        "SCENARIO-ARC-WMTE-4730-PERSIST-STRONGEST-435-PRIMITIVE",
        "SCENARIO-ARC-WMTE-4730-LEAVE-ONE-GAME-TRANSFER",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4730_solver_kit_operator_ranks_online_warm_actions() -> None:
    """REQ-ARC-WMTE-4730: the persisted operator ranks without using the oracle."""

    memory = kit.PersistentAEM.from_effect_rows(
        [
            _row("train", "s1", 1, changed=False),
            _row("train", "s2", 2, changed=True),
        ]
    )
    result = kit.online_warm_action_effect_controller_operator(
        [
            _candidate("noop", 1, target=False, score=0.0),
            _candidate("warm_target", 2, target=True, score=3.0),
        ],
        memory=memory,
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["candidate_count"] == 2
    assert result["score_source"] == "persistent_aem_plus_online_warm"
    assert result["actions_to_first_levelup_before"] == 2
    assert result["actions_to_first_levelup_after"] == 1
    assert result["actions_reduced"] == 1.0
    assert result["value_added"] is True
    assert result["ranked_candidates"][0]["candidate_id"] == "warm_target"

    empty = kit.online_warm_action_effect_controller_operator([], memory=memory)
    assert empty["candidate_count"] == 0
    assert empty["actions_to_first_levelup_before"] is None
    assert empty["value_added"] is False


def test_req_arc_wmte_4730_routing_and_registry_surface_persisted_operator() -> None:
    """REQ-ARC-WMTE-4730: solver-kit routing and registry expose the operator."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "online-warm" in gotchas[0]["note"]
    assert "transfer_dead_ends" in gotchas[0]


def test_scenario_arc_wmte_4730_selects_a1_when_non_degenerate() -> None:
    """SCENARIO-ARC-WMTE-4730-PERSIST-STRONGEST-435-PRIMITIVE: A1 wins."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert decision["source"] == "A1_online_action_learning_driver"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["selected_reason"] == "a1_non_degenerate_online_warm_controller"
    assert decision["upstream_signal_rank"][0]["source"] == "A1_online_action_learning_driver"

    fallback = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(arms_non_degenerate=False),
        a2_artifact=_a2_artifact(probe_actions=2),
    )
    assert fallback["source"] == "A2_active_probe_controller"
    assert fallback["operator"] == mod.A2_FALLBACK_OPERATOR


def test_scenario_arc_wmte_4730_leave_one_game_transfer_reports_deltas() -> None:
    """SCENARIO-ARC-WMTE-4730-LEAVE-ONE-GAME-TRANSFER: per-game deltas are honest."""

    effect_rows = [
        _row("train", "a", 1, changed=False),
        _row("train", "b", 6, changed=True, x=32, y=32),
        _row("zz99", "target", 1, changed=False),
        _row("zz99", "target", 6, changed=True, x=32, y=32, online_warm_score=1.0),
        _row("zz99", "solo", 2, changed=True),
    ]

    result = mod.measure_transfer_game(
        "zz99",
        effect_rows=effect_rows,
        frozen_artifact=_arm_artifact("zz99", first_win=False),
        online_warm_artifact=_arm_artifact("zz99", first_win=True, actions=4),
    )

    assert result["game"] == "zz99"
    assert result["excluded_from_memory"] is True
    assert result["value_added"] is True
    assert result["transfer_value"]["action_efficiency_delta"] == 1.0
    assert result["transfer_value"]["first_win_rate_delta"] == 1.0
    assert result["transfer_value"]["live_solve_rate_delta"] == 1.0
    assert result["transfer_value"]["candidate_generation_coverage_delta"] == 0.0
    assert result["transfer_value"]["offline_reproduced_new_level"] is False

    null = mod.measure_transfer_game(
        "flat",
        effect_rows=[
            _row("train", "a", 1, changed=True),
            _row("flat", "s", 1, changed=True),
            _row("flat", "s", 2, changed=False),
        ],
        frozen_artifact=_arm_artifact("flat", first_win=False),
        online_warm_artifact=_arm_artifact("flat", first_win=False),
    )
    assert null["value_added"] is False
    assert "did not improve" in null["dead_end"]


def test_scenario_arc_wmte_4730_artifact_schema_success_null_and_run(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4730-LEAVE-ONE-GAME-TRANSFER: artifact writes stably."""

    _write_minimal_repo(tmp_path)
    rows = [
        {
            "game": "zz99",
            "value_added": True,
            "transfer_value": {
                "operator": mod.PRIMITIVE_OPERATOR,
                "live_solve_rate_delta": 1.0,
                "first_win_rate_delta": 1.0,
                "candidate_generation_coverage_delta": 0.0,
                "action_efficiency_delta": 1.0,
                "offline_reproduced_new_level": False,
                "value_added": True,
            },
            "offline_reproduced_new_level": False,
            "dead_end": "",
        },
        {
            "game": "yy88",
            "value_added": False,
            "transfer_value": {
                "operator": mod.PRIMITIVE_OPERATOR,
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "candidate_generation_coverage_delta": 0.0,
                "action_efficiency_delta": 0.0,
                "offline_reproduced_new_level": False,
                "value_added": False,
            },
            "offline_reproduced_new_level": False,
            "dead_end": "null",
        },
        {
            "game": "xx77",
            "value_added": False,
            "transfer_value": {
                "operator": mod.PRIMITIVE_OPERATOR,
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "candidate_generation_coverage_delta": 0.0,
                "action_efficiency_delta": 0.0,
                "offline_reproduced_new_level": False,
                "value_added": False,
            },
            "offline_reproduced_new_level": False,
            "dead_end": "null",
        },
    ]
    artifact = mod.build_artifact(
        selected_upstream=mod.select_primitive_from_upstreams(
            a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
        ),
        preconditions_checked={"ok": True, "blocked_resource": ""},
        transfer_results=rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == (
        "complete: online_warm_action_effect_controller_operator_persisted_transfer_characterized"
    )
    assert artifact["offline_reproduced_new_level"] is False
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    null = mod.build_artifact(
        selected_upstream=mod.select_primitive_from_upstreams(
            a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
        ),
        preconditions_checked={"ok": True, "blocked_resource": ""},
        transfer_results=[{**row, "value_added": False, "dead_end": "null"} for row in rows],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.25,
    )
    assert null["honest_verdict"] == (
        "complete: online_warm_action_effect_controller_operator_persisted_transfer_null"
    )
    assert mod.artifact_schema_errors(null) == []

    run_artifact = mod.run(
        tmp_path,
        transfer_games=("zz99", "yy88", "xx77"),
        offline_arcade_checker=lambda: True,
        effect_rows_provider=lambda _root: [
            _row("train", "a", 1, changed=False),
            _row("train", "b", 6, changed=True, x=32, y=32),
            _row("zz99", "target", 1, changed=False),
            _row("zz99", "target", 6, changed=True, x=32, y=32, online_warm_score=1.0),
            _row("yy88", "flat", 1, changed=True),
            _row("yy88", "flat", 2, changed=False),
            _row("xx77", "flat", 1, changed=True),
            _row("xx77", "flat", 2, changed=False),
        ],
        now=iter([10.0, 10.5]).__next__,
    )
    assert run_artifact["preconditions_checked"]["ok"] is True
    assert run_artifact["transfer_games"] == ["zz99", "yy88", "xx77"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4730_defensive_schema_and_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4730: blocked and malformed cases stay explicit."""

    assert "missing:honest_verdict" in mod.artifact_schema_errors({})
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(False) == 0
    assert mod._as_int("bad") == 0
    assert mod._first_failed_resource(  # noqa: SLF001
        {
            "offline_arcade": True,
            "a1_artifact": True,
            "a2_artifact": True,
            "spec_has_req_4730": True,
            "registry_has_primitive_gotcha": True,
            "operator_registered": True,
        }
    ) == ""
    assert mod._load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False

    checks = mod.check_preconditions(tmp_path, offline_arcade_checker=lambda: False)
    assert checks["ok"] is False
    assert checks["blocked_resource"] == "offline_arcade"
    checks_error = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks_error["offline_arcade"] is False
    assert "RuntimeError" in checks_error["offline_arcade_error"]

    neither = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(arms_non_degenerate=False),
        a2_artifact=_a2_artifact(probe_actions=0),
    )
    assert neither["selected_reason"] == "a1_null_but_a2_no_probe_actions"

    obj = SimpleNamespace(
        env="objgame",
        state_key="objstate",
        action=6,
        x="bad",
        y=1,
        changed=None,
        frame_delta=1.0,
        level_progress=0.0,
        online_warm_score=2.0,
    )
    assert mod._row_game(obj) == "objgame"  # noqa: SLF001
    assert mod._row_state_key(obj) == "objstate"  # noqa: SLF001
    assert mod._row_action_id({"action": 5}) == 5  # noqa: SLF001
    assert mod._row_action_id({"action_id": "bad"}) is None  # noqa: SLF001
    assert mod._row_xy({"x": "bad", "y": 1}) is None  # noqa: SLF001
    assert mod._row_online_score(obj) == 2.0  # noqa: SLF001
    assert mod._row_effective_target(obj) is True  # noqa: SLF001
    assert mod._attempts_for_game({}, "zz99") == []  # noqa: SLF001

    no_rows = mod.measure_transfer_game(
        "none",
        effect_rows=[],
        frozen_artifact={},
        online_warm_artifact={},
    )
    assert no_rows["dead_end"].startswith("no cached held-out")
    no_groups = mod.measure_transfer_game(
        "bad",
        effect_rows=[
            {"game": "bad", "state_key": "", "action_id": 1, "changed": True},
            {"game": "bad", "state_key": "s", "action_id": "bad", "changed": True},
        ],
        frozen_artifact={},
        online_warm_artifact={},
    )
    assert no_groups["dead_end"].startswith("cached rows were present")
    no_target_group = mod.measure_transfer_game(
        "noop",
        effect_rows=[
            _row("train", "t", 1, changed=True),
            _row("noop", "s", 1, changed=False),
            _row("noop", "s", 2, changed=False),
        ],
        frozen_artifact={},
        online_warm_artifact={},
    )
    assert no_target_group["dead_end"].startswith("cached rows contained no")

    blocked = mod.build_artifact(
        selected_upstream={"operator": mod.PRIMITIVE_OPERATOR},
        preconditions_checked={"ok": False, "blocked_resource": "a1_artifact"},
        transfer_results=[],
        registry_updated=False,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert blocked["honest_verdict"] == "blocked_a1_artifact"
    assert mod.artifact_schema_errors(blocked) == []

    with_new_level = mod.build_artifact(
        selected_upstream={
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        preconditions_checked={"ok": True, "blocked_resource": ""},
        transfer_results=[
            {
                "game": game,
                "value_added": game == "aa00",
                "offline_reproduced_new_level": game == "aa00",
                "transfer_value": {"operator": mod.PRIMITIVE_OPERATOR},
                "dead_end": "" if game == "aa00" else "null",
            }
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.0,
    )
    assert with_new_level["offline_reproduced_new_level"] is True
    assert with_new_level["new_levels_banked"] == 1

    malformed = dict(blocked)
    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "bad"
    malformed["persisted_operator"] = {}
    malformed["transfer_value_per_game"] = []
    malformed["offline_reproduced_new_level"] = "no"
    malformed["verifier_is_oracle"] = True
    malformed["random_seed"] = "bad"
    malformed["field_principles"] = {}
    malformed["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "inference_substrate_mismatch" in errors
    assert "persisted_operator_mismatch" in errors
    assert "offline_reproduced_new_level_must_be_bool" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "random_seed_must_be_int" in errors
    assert "field_principles_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors

    wrong_gotcha = dict(blocked)
    wrong_gotcha["persisted_operator"] = {
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": "wrong",
    }
    wrong_gotcha["reproducibility_checksum"] = mod.payload_checksum(wrong_gotcha)
    assert "persisted_operator_registry_entry_mismatch" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    bad_characterized = dict(blocked)
    bad_characterized["honest_verdict"] = (
        "complete: online_warm_action_effect_controller_operator_persisted_transfer_characterized"
    )
    bad_characterized["transfer_games"] = ["aa00", "bb00", "cc00"]
    bad_characterized["transfer_value_per_game"] = {"aa00": {"value_added": False}}
    bad_characterized["reproducibility_checksum"] = mod.payload_checksum(bad_characterized)
    assert "characterized_transfer_requires_value_added" in mod.artifact_schema_errors(
        bad_characterized
    )

    bad_offline = dict(blocked)
    bad_offline["offline_reproduced_new_level"] = True
    bad_offline["new_levels_banked"] = 0
    bad_offline["reproducibility_checksum"] = mod.payload_checksum(bad_offline)
    assert "offline_reproduced_new_level_requires_banked_record" in mod.artifact_schema_errors(
        bad_offline
    )

    with pytest.raises(ValueError, match="missing:honest_verdict"):
        mod.write_artifact({}, root=tmp_path)

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(
            tmp_path,
            offline_arcade_checker=lambda: True,
            effect_rows_provider=lambda _root: [],
            write=False,
        )
