"""Tests for Exp5971 ARC strip-swap battery.

Spec refs: REQ-ARC-CPTB-5971,
SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL,
SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH,
SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic import arc_strip_swap_battery as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"


def _fixture_rows(*, transformed_anchor_live: bool = False) -> list[dict]:
    rows: list[dict] = []
    games = ["g1", "g2", "g3"]
    seeds = [101, 102]
    for game in games:
        for seed in seeds:
            for condition in mod.CONDITIONS:
                for arm in mod.ARMS:
                    levels = 0
                    if condition == "original":
                        if game == "g1" and arm in {"SHIP", "HUDO"}:
                            levels = 1
                        if game == "g2" and arm == "CTRL":
                            levels = 1
                    elif transformed_anchor_live:
                        if game == "g1" and arm == "SHIP":
                            levels = 1
                        if game == "g3" and arm == "FRONT":
                            levels = 1
                    rows.append(
                        {
                            "cell_id": f"{game}|{arm}|{seed}|{condition}",
                            "game": game,
                            "seed": seed,
                            "condition": condition,
                            "arm": arm,
                            "terminal_state": "completed",
                            "completed": True,
                            "missing": False,
                            "errored": False,
                            "generator_invalid": False,
                            "ran": True,
                            "levels": levels,
                            "progress": float(levels),
                            "actions": 7 + levels,
                            "actions_to_first_levelup": 4 if levels else None,
                            "elapsed_s": 0.01,
                            "error": None,
                            "transform_selected_condition_id": (
                                None if condition == "original" else "C5_strip_swap_rows_bottom_t2"
                            ),
                            "hud_predicate_changed": condition == "strip_swap",
                            "hud_mask_resolved_before": True,
                            "hud_mask_resolved_after": condition == "original",
                            "frontier_predicate_dose": 0.0,
                            "policy_decisions": [{"kind": "RESET"}, {"kind": "ACTION1"}],
                            "observations": [{"level": 0}],
                            "health": {"valid_action_count": 1, "step_ok_count": 1},
                        }
                    )
    return rows


def _row(
    game: str,
    seed: int,
    condition: str,
    arm: str,
    *,
    levels: int,
    hud_changed: bool = True,
    terminal_state: str = "completed",
) -> dict[str, Any]:
    return {
        "cell_id": f"{game}|{arm}|{seed}|{condition}",
        "game": game,
        "seed": seed,
        "condition": condition,
        "arm": arm,
        "terminal_state": terminal_state,
        "completed": terminal_state == "completed",
        "missing": False,
        "errored": terminal_state == "errored",
        "generator_invalid": False,
        "ran": True,
        "levels": levels,
        "progress": float(levels),
        "actions": 1,
        "actions_to_first_levelup": 1 if levels else None,
        "elapsed_s": 0.001,
        "error": None,
        "transform_selected_condition_id": None
        if condition == "original"
        else "C5_strip_swap_rows_bottom_t2",
        "hud_predicate_changed": condition == "strip_swap" and hud_changed,
        "hud_mask_resolved_before": True,
        "hud_mask_resolved_after": condition == "original",
        "frontier_predicate_dose": 0.0,
        "policy_decisions": [],
        "observations": [],
        "health": {"valid_action_count": 1, "step_ok_count": 1},
    }


def _hud_contrast_rows(
    directions: list[int],
    *,
    original_anchor: bool = True,
    hud_changed: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = 1
    for index, direction in enumerate(directions, start=1):
        game = f"p{index}"
        rows.append(
            _row(game, seed, "original", "SHIP", levels=1 if original_anchor and index == 1 else 0)
        )
        rows.append(_row(game, seed, "original", "FRONT", levels=0))
        if direction > 0:
            rows.append(
                _row(game, seed, "strip_swap", "SHIP", levels=1, hud_changed=hud_changed)
            )
            rows.append(
                _row(game, seed, "strip_swap", "FRONT", levels=0, hud_changed=hud_changed)
            )
        elif direction < 0:
            rows.append(
                _row(game, seed, "strip_swap", "SHIP", levels=0, hud_changed=hud_changed)
            )
            rows.append(
                _row(game, seed, "strip_swap", "FRONT", levels=1, hud_changed=hud_changed)
            )
        else:
            rows.append(
                _row(game, seed, "strip_swap", "SHIP", levels=1, hud_changed=hud_changed)
            )
            rows.append(
                _row(game, seed, "strip_swap", "FRONT", levels=1, hud_changed=hud_changed)
            )
    return rows


def test_req_arc_cptb_5971_spec_declares_battery_contract() -> None:
    """REQ-ARC-CPTB-5971: OpenSpec freezes gate replay, matrix, fields, and safeguards."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-CPTB-5971") :]

    for marker in (
        "SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL",
        "SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH",
        "SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL",
        mod.RESULT_RELATIVE_PATH,
        "make_carnot_agent",
        "E3AgentPolicy",
        "strip_swap_sentinel_ready_score == 1.0",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_arc_cptb_5971_gate_replay_and_matrix_seal() -> None:
    """SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL: preconditions seal all planned cells."""

    gate = mod.replay_exp5970_gate(REPO)
    arms = mod.load_preregistered_arms(REPO)
    seal = mod.build_matrix_seal(REPO, arms=arms, action_budget=3, wall_time_s=5.0)

    assert gate["ready"] is True
    assert gate["strip_swap_sentinel_ready_score"] == 1.0
    assert gate["artifact_sha256"]
    assert list(arms) == list(mod.ARMS)
    assert seal["sealed_before_outcomes"] is True
    assert seal["n_games"] == 25
    assert seal["n_arms"] == 4
    assert seal["n_seeds"] == 5
    assert seal["n_conditions"] == 2
    assert seal["n_cells_expected"] == 25 * 4 * 5 * 2
    assert seal["action_budget"] == 3


def test_scenario_arc_cptb_5971_live_path_cell_health_smoke() -> None:
    """SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH: a bounded cell uses the submitted path."""

    row = mod.run_live_cell(
        REPO,
        game="tn36",
        arm="SHIP",
        seed=20260726,
        condition="strip_swap",
        arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
        action_budget=1,
        wall_time_s=10.0,
    )

    assert row["game"] == "tn36"
    assert row["arm"] == "SHIP"
    assert row["condition"] == "strip_swap"
    assert row["live_path"] == "make_carnot_agent/E3AgentPolicy.choose_action"
    assert row["terminal_state"] in {"completed", "errored"}
    assert row["health"]["source_bfs_adapter_prior_game_hidden_state_access_count"] == 0
    assert row["transform_selected_condition_id"] in {
        "C4_strip_swap_rows_top_t2",
        "C5_strip_swap_rows_bottom_t2",
        "C6_strip_swap_cols_left_t2",
        "C7_strip_swap_cols_right_t2",
    }


def test_scenario_arc_cptb_5971_game_unit_forced_verdict_refusal() -> None:
    """SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL: destroyed anchors force null."""

    analysis = mod.analyze_rows(_fixture_rows(transformed_anchor_live=False), expected_cells=48)

    hud = analysis["anchor_survival_and_discriminating_game_support"]["hud_given_frontier_on"]
    stats = analysis["game_unit_sign_jackknife_intervals_and_p_floors"]["hud_given_frontier_on"]

    assert hud["original_anchor_won_by_matched_arm"] is True
    assert hud["transformed_anchor_retains_valid_support"] is False
    assert hud["interpretable"] is False
    assert stats["strip_swap"]["exact_one_sided_sign_test"]["n_independent_games"] <= 2
    assert analysis["convention_dependence_decision"]["status"] == "complete_null"
    assert "anchor support" in analysis["convention_dependence_decision"]["reason"]
    assert analysis["overall_hud_value_not_identified_receipt"]["flag_flip_recommended"] is False


def test_req_arc_cptb_5971_artifact_schema_validation_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-CPTB-5971: artifact fields, immutability receipts, and checksum are enforced."""

    rows = _fixture_rows(transformed_anchor_live=True)
    monkeypatch.setattr(mod, "run_frozen_matrix", lambda *args, **kwargs: rows)

    artifact = mod.build_artifact(
        root=REPO,
        result_output_path=tmp_path / "experiment_5971.json",
        action_budget=3,
        wall_time_s=5.0,
        test_exit_codes={"focused_unit": 0},
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["status"] in {"complete_null", "complete_underpowered", "complete_positive"}
    assert artifact["preconditions_checked"]["checked"] is True
    assert artifact["gate_replay_receipt"]["ready"] is True
    assert artifact["expected_completed_missing_errored_and_generator_invalid_cells"]["expected"] == 1000
    assert artifact["live_agent_path_and_disabled_escape_hatches"]["normal_path"].endswith(
        "E3AgentPolicy.choose_action"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_solve_credit_receipt"]["solve_credit_claimed"] is False
    mod.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        bad = dict(artifact)
        del bad["status"]
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact({**artifact, "verifier_is_oracle": True})
    with pytest.raises(ValueError, match="solve credit"):
        bad = json.loads(json.dumps(artifact))
        bad["no_solve_credit_receipt"]["solve_credit_claimed"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="flag flip"):
        bad = json.loads(json.dumps(artifact))
        bad["overall_hud_value_not_identified_receipt"]["flag_flip_recommended"] = True
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="registry"):
        bad = json.loads(json.dumps(artifact))
        bad["shipped_flag_and_registry_immutability"]["registry_unchanged"] = False
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        bad = json.loads(json.dumps(artifact))
        bad["honest_verdict"] = "ready: bad"
        bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="inference_substrate"):
        bad = json.loads(json.dumps(artifact))
        bad["inference_substrate"] = "live_llm_inference"
        bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="policy flags"):
        bad = json.loads(json.dumps(artifact))
        bad["shipped_flag_and_registry_immutability"]["policy_flags_modified_by_task"] = True
        bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="protected files"):
        bad = json.loads(json.dumps(artifact))
        bad["protected_files_unchanged"]["all_unchanged"] = False
        bad["reproducibility_checksum"] = mod.artifact_checksum(bad)
        mod.validate_artifact(bad)


def test_req_arc_cptb_5971_writer_round_trips_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-CPTB-5971: writer emits the validated artifact returned by the builder."""

    payload = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    payload.update(
        {
            "status": "complete_null",
            "honest_verdict": "complete_null: fixture",
            "reproducibility_checksum": "sha256:fixture",
        }
    )
    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: payload)

    out = tmp_path / "experiment_5971.json"
    written = mod.write_artifact(root=REPO, result_output_path=out, test_exit_codes={"unit": 0})

    assert written is payload
    assert json.loads(out.read_text(encoding="utf-8"))["honest_verdict"] == "complete_null: fixture"


def test_req_arc_cptb_5971_precondition_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-CPTB-5971: gate replay reports every failed precondition explicitly."""

    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "arc_solve_registry.yaml").write_text("full_game_clear: true\n", encoding="utf-8")
    monkeypatch.setattr(mod, "public_game_manifest", lambda root: [])
    monkeypatch.setattr(mod, "preregistered_seeds", lambda root: [1])
    monkeypatch.setattr(mod, "load_preregistered_arms", lambda root: {"BAD": {}})
    monkeypatch.setattr(
        mod,
        "_resource_receipt",
        lambda root: {
            "disk_free_bytes": 1,
            "disk_total_bytes": 2,
            "ram_available_bytes": None,
            "arc_sdk_available": False,
            "offline_arcade_cache_available": True,
        },
    )

    gate = mod.replay_exp5970_gate(tmp_path)

    assert gate["ready"] is False
    assert {
        "missing_exp5970_artifact",
        "strip_swap_sentinel_ready_score_not_1",
        "transform_schema_hash_mismatch",
        "public_game_manifest_not_25",
        "arm_definitions_not_four_preregistered_arms",
        "seed_manifest_not_5",
        "arc_sdk_unavailable",
    } <= set(gate["blocked_reasons"])


def test_req_arc_cptb_5971_resource_and_arm_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-CPTB-5971: resource and arm helpers fail closed under drift."""

    monkeypatch.setattr(mod.os, "sysconf", lambda name: (_ for _ in ()).throw(OSError(name)))
    real_import = __import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "arcengine":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    resources = mod._resource_receipt(REPO)
    assert resources["ram_available_bytes"] is None
    assert resources["arc_sdk_available"] is False

    monkeypatch.setattr(mod, "_outer_cptb", lambda root: {})
    assert len(mod.public_game_manifest(REPO)) == 25
    assert mod.preregistered_seeds(REPO) == [20260726, 20260727, 20260728, 20260729, 20260730]

    fake_module = types.SimpleNamespace(
        CPTB_ARMS={name: {"kwargs": {}} for name in mod.ARMS}
    )
    monkeypatch.setitem(sys.modules, "scripts.experiments.cptb_arms", fake_module)
    with pytest.raises(ValueError, match="does not pin gated flags"):
        mod.load_preregistered_arms(REPO)


def test_req_arc_cptb_5971_environment_and_matrix_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL: helper branches are explicit."""

    env_name = mod.ARM_ENV_VARS["tier_exhaustion"]
    monkeypatch.setenv(env_name, "original")
    with mod._arm_environment({"tier_exhaustion": True}):
        assert os.environ[env_name] == "1"
    assert os.environ[env_name] == "original"

    grid = mod.np.zeros((64, 64), dtype=mod.np.uint8)
    selected = mod._select_strip_condition_for_grid(grid)
    assert selected.condition_id in {row.condition_id for row in mod.sentinel.STRIP_SWAP_CONDITIONS}

    calls: list[tuple[str, str, int, str]] = []

    def fake_run_live_cell(root: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["game"], kwargs["arm"], kwargs["seed"], kwargs["condition"]))
        return _row(kwargs["game"], kwargs["seed"], kwargs["condition"], kwargs["arm"], levels=0)

    seal = {
        "games": ["g"],
        "seeds": [1],
        "action_budget": 2,
        "wall_time_s": 3.0,
    }
    arms = {name: {"kwargs": {}} for name in mod.ARMS}
    monkeypatch.setattr(mod, "run_live_cell", fake_run_live_cell)
    rows = mod.run_frozen_matrix(root=REPO, seal=seal, arms=arms)
    assert len(rows) == len(mod.ARMS) * len(mod.CONDITIONS)
    assert calls[0] == ("g", "CTRL", 1, "original")


def test_scenario_arc_cptb_5971_live_cell_defensive_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH: live cell errors stay on row."""

    original = mod.run_live_cell(
        REPO,
        game="tn36",
        arm="SHIP",
        seed=20260726,
        condition="original",
        arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
        action_budget=0,
        wall_time_s=10.0,
    )
    assert original["condition"] == "original"
    assert original["terminal_state"] == "completed"

    timed = mod.run_live_cell(
        REPO,
        game="tn36",
        arm="SHIP",
        seed=20260726,
        condition="original",
        arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
        action_budget=1,
        wall_time_s=-1.0,
    )
    assert timed["terminal_state"] == "errored"
    assert timed["error"] == "wall_time_budget_exhausted"

    with monkeypatch.context() as mp:
        levels = iter([0, 0, 1, 1])
        mp.setattr(mod, "_frame_level", lambda frame: next(levels, 1))
        leveled = mod.run_live_cell(
            REPO,
            game="tn36",
            arm="SHIP",
            seed=20260726,
            condition="original",
            arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
            action_budget=1,
            wall_time_s=10.0,
        )
    assert leveled["actions_to_first_levelup"] == 1

    def raise_action_id(action: Any) -> int:
        raise RuntimeError("decision boom")

    monkeypatch.setattr(mod.sentinel, "_action_id", raise_action_id)
    errored = mod.run_live_cell(
        REPO,
        game="tn36",
        arm="SHIP",
        seed=20260726,
        condition="original",
        arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
        action_budget=1,
        wall_time_s=10.0,
    )
    assert errored["terminal_state"] == "errored"
    assert "decision boom" in errored["error"]

    bad_condition = mod.run_live_cell(
        REPO,
        game="tn36",
        arm="SHIP",
        seed=20260726,
        condition="bad",
        arm_kwargs=mod.load_preregistered_arms(REPO)["SHIP"]["kwargs"],
        action_budget=1,
        wall_time_s=10.0,
    )
    assert bad_condition["terminal_state"] == "errored"
    assert "unknown condition" in bad_condition["error"]


def test_scenario_arc_cptb_5971_game_unit_decision_regions() -> None:
    """SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL: all verdict regions are named."""

    no_transform = mod.analyze_rows(
        _hud_contrast_rows([1, 1, 1, 1, 1], hud_changed=False),
        expected_cells=20,
    )
    assert no_transform["convention_dependence_decision"]["status"] == "complete_null"
    assert "did not change" in no_transform["convention_dependence_decision"]["reason"]

    no_original = mod.analyze_rows(
        _hud_contrast_rows([1, 1, 1, 1, 1], original_anchor=False),
        expected_cells=20,
    )
    assert no_original["convention_dependence_decision"]["status"] == "complete_null"
    assert "original anchor" in no_original["convention_dependence_decision"]["reason"]

    p_floor = mod.analyze_rows(_hud_contrast_rows([1, 1, 1]), expected_cells=12)
    assert p_floor["convention_dependence_decision"]["status"] == "complete_underpowered"
    assert "p-floor" in p_floor["convention_dependence_decision"]["reason"]

    positive = mod.analyze_rows(_hud_contrast_rows([1, 1, 1, 1, 1]), expected_cells=20)
    assert positive["convention_dependence_decision"]["status"] == "complete_positive"

    null = mod.analyze_rows(_hud_contrast_rows([1, 1, 1, -1, -1]), expected_cells=20)
    assert null["convention_dependence_decision"]["status"] == "complete_null"
    assert "does not support" in null["convention_dependence_decision"]["reason"]

    skipped = _fixture_rows(transformed_anchor_live=True)
    skipped[0]["terminal_state"] = "errored"
    skipped[0]["completed"] = False
    stats = mod._contrast_stats(skipped, treatment="SHIP", control="FRONT", condition="original")
    assert stats["n_games_with_paired_rows"] >= 1
    assert mod._jackknife({})["reason"] == "n_games_lt_2"


def test_req_arc_cptb_5971_blocked_artifact_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-CPTB-5971: blocked preconditions stop execution and CLI delegates cleanly."""

    gate = mod.replay_exp5970_gate(REPO)
    gate = {**gate, "ready": False, "blocked_reasons": ["fixture_block"]}
    monkeypatch.setattr(mod, "replay_exp5970_gate", lambda root: gate)
    monkeypatch.setattr(mod, "run_frozen_matrix", lambda **kwargs: (_ for _ in ()).throw(AssertionError))
    blocked = mod.build_artifact(
        root=REPO,
        result_output_path=tmp_path / "blocked.json",
        test_exit_codes={"unit": 0},
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["honest_verdict"].startswith("blocked:")

    captured: dict[str, Any] = {}

    def fake_write_artifact(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "complete_null"}

    monkeypatch.setattr(mod, "write_artifact", fake_write_artifact)
    rc = mod.main(
        [
            "--root",
            str(REPO),
            "--out",
            str(tmp_path / "out.json"),
            "--action-budget",
            "2",
            "--wall-time-s",
            "3",
            "--test-exit-codes-json",
            '{"unit": 0}',
        ]
    )
    assert rc == 0
    assert captured["action_budget"] == 2
    assert captured["wall_time_s"] == 3.0
    assert json.loads(capsys.readouterr().out)["status"] == "complete_null"
