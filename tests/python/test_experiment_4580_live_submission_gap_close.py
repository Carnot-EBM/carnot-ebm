"""Tests for Exp 4580 live-submission gap close.

Spec refs: REQ-CAPSTONE-4580, SCENARIO-CAPSTONE-4580,
SCENARIO-CAPSTONE-4580-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4580_live_submission_gap_close as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _registry(*rows: tuple[str, int]) -> JsonDict:
    return {
        "schema_version": 1,
        "prior_submitted_baseline_levels": 33,
        "games": [
            {"game": game, "reproducibility": "reproduced", "levels_reproduced": levels}
            for game, levels in rows
        ],
    }


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "environment_files" / "dummy").mkdir(parents=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(_registry(("ft09", 1), ("g50t", 1), ("s5i5", 1), ("sc25", 5))),
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "arc3_live_submit.json",
        {
            "live_total_levels": 33,
            "per_game": [{"game": "sc25", "claimed": 1, "live_level": 0, "env_match": False}],
        },
    )
    _write_json(
        root / "results" / "experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json",
        {
            "per_game_scorecard": [
                {
                    "game": "ft09",
                    "offline_reproduced": True,
                    "reproduced_levels": 1,
                    "plan": ['{"action": 6, "data": {"x": 36, "y": 36}}'],
                }
            ]
        },
    )
    _write_json(
        root / "results" / "experiment_4443_bank_g50t_example_conditioned_win.json",
        {
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "solver": {"solution": ["4", "5"]},
        },
    )
    _write_json(
        root / "results" / "experiment_4421_config_rule_solve_unseen.json",
        {
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "solver": {"solution": ["h_extend", "v_extend"]},
        },
    )
    _write_json(
        root / "results" / "experiment_4468_bank_sc25_provisional_levels.json",
        {
            "offline_reproduced": True,
            "sc25_levels_reproduced_total": 5,
            "solution_by_level": {"5": ["cell0,1", "move3"]},
        },
    )
    return root


def test_req_capstone_4580_spec_anchor_declares_gap_close_contract() -> None:
    """REQ-CAPSTONE-4580: OpenSpec declares the live-submit gap-close contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4580" in spec
    assert "SCENARIO-CAPSTONE-4580" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_capstone_4580_banks_trajectories_and_counts_without_overclaim(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4580: trajectory-backed rows count only to fresh replay depth."""

    root = _fixture_root(tmp_path)
    registry = mod.load_registry(root)
    replayed = {"ft09": 1, "g50t": 1, "s5i5": 1}

    rows, banked = mod.build_submittable_rows(
        root,
        registry=registry,
        replay_actions_fn=lambda game, _actions: replayed.get(game, 0),
        sc25_reproduce_fn=lambda _root, _target: 5,
    )

    by_game = {row["game"]: row for row in rows}
    assert {"ft09", "g50t", "s5i5"}.issubset(set(banked))
    assert by_game["ft09"]["submittable_level"] == 1
    assert by_game["g50t"]["trajectory_path"].endswith("g50t.json")
    assert by_game["s5i5"]["trajectory_action_count"] == 2
    assert by_game["sc25"]["submittable_level"] == 5
    assert by_game["sc25"]["has_trajectory"] is True
    assert by_game["sc25"]["has_env_adaptive_resolver"] is True
    assert by_game["sc25"]["adaptive_labels"] == ["cell0,1", "move3"]
    assert sum(row["submittable_level"] for row in rows) == 8

    s5i5_bank = json.loads((root / by_game["s5i5"]["trajectory_path"]).read_text(encoding="utf-8"))
    assert s5i5_bank["solution"] == [
        {"action": 6, "data": {"x": 47, "y": 21}},
        {"action": 6, "data": {"x": 22, "y": 47}},
    ]


def test_scenario_capstone_4580_artifact_schema_and_package_gate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4580-FIELD-PRINCIPLES: artifact fields are bare and package-gated."""

    rows = [
        {
            "game": "alpha",
            "registry_reproduced_level": 20,
            "offline_reproduced_level": 20,
            "submittable_level": 20,
            "has_trajectory": True,
            "has_env_adaptive_resolver": False,
            "drift_robust": False,
            "trajectory_path": "results/arc3_live_banked_trajectories/alpha.json",
            "trajectory_action_count": 2,
            "source": "fixture",
            "claim_capped": False,
        },
        {
            "game": "sc25",
            "registry_reproduced_level": 15,
            "offline_reproduced_level": 15,
            "submittable_level": 15,
            "has_trajectory": False,
            "has_env_adaptive_resolver": True,
            "drift_robust": True,
            "trajectory_path": "",
            "trajectory_action_count": 0,
            "source": "adaptive_fixture",
            "claim_capped": False,
        },
    ]

    artifact = mod.build_artifact(
        root=tmp_path,
        registry={"prior_submitted_baseline_levels": 33},
        preconditions_checked={"ok": True},
        per_game_rows=rows,
        trajectories_banked=["alpha"],
        env_adaptive_resolve_recovered=["sc25"],
        duration_s=1.05,
    )
    package_path = mod.write_refreshed_package(tmp_path, artifact)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["live_submittable_level_count"] == 35
    assert artifact["count_delta"] == 2
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["refreshed_package_path"] == mod.PACKAGE_RELATIVE_PATH
    assert package_path.exists()
    assert json.loads(package_path.read_text(encoding="utf-8"))["claimed_total_levels"] == 35


def test_req_capstone_4580_blocks_on_failed_preconditions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4580: missing preconditions stop before fabricated counting."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(":", encoding="utf-8")

    artifact = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
        replay_actions_fn=lambda _game, _actions: 0,
    )

    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["live_submittable_level_count"] == 0
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["preconditions_checked"]["offline_arcade"]["ok"] is False
    assert Path(tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4580_sc25_adaptive_proxy_recovers_drift() -> None:
    """REQ-CAPSTONE-4580: sc25 adaptive coordinates recover a drift where frozen replay fails."""

    labels = ["cell0,1", "move3"]
    result = mod.validate_sc25_drift_proxy(labels, drift_origin=(27, 43), step=6)

    assert result["frozen_flat_replay_reached"] is False
    assert result["env_adaptive_replay_reached"] is True
    assert result["recovered"] is True
    assert result["adaptive_actions"][0] == {"action": 6, "data": {"x": 33, "y": 43}}


def test_req_capstone_4580_schema_rejects_false_ready_and_null_delta_note() -> None:
    """REQ-CAPSTONE-4580: schema catches false readiness and requires null notes."""

    artifact = {
        field: None for field in mod.REQUIRED_ARTIFACT_FIELDS
    }
    artifact.update(
        {
            "honest_verdict": "success: live_submittable_count_33_above_33",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "live_submittable_level_count": 33,
            "live_submittable_count_baseline": 33,
            "count_delta": 0,
            "trajectories_banked": [],
            "env_adaptive_resolve_recovered": [],
            "refreshed_package_path": mod.PACKAGE_RELATIVE_PATH,
            "per_game_submittable": [],
            "ready_for_operator_submit": True,
            "offline_reproduced": {},
            "random_seed": mod.RANDOM_SEED,
            "reproducibility_checksum": "0" * 64,
            "preconditions_checked": {"ok": True},
        }
    )

    errors = mod.artifact_schema_errors(artifact)

    assert "ready_for_operator_submit requires count above baseline" in errors
    assert "null_delta_methodology_note required when count_delta is zero" in errors


def test_req_capstone_4580_pure_helper_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4580: helper branches normalize package inputs deterministically."""

    root = _fixture_root(tmp_path)
    assert mod.check_preconditions(root, offline_arcade_checker=lambda: True)["ok"] is True
    assert mod.first_precondition_miss({"offline_arcade": {"ok": True}, "registry": {"ok": True}}) is None
    assert mod.first_precondition_miss({"offline_arcade": {"ok": True}, "registry": {"ok": False}}) == "registry"

    assert mod.label_to_action({"x": 3, "y": 4}) == {"action": 6, "data": {"x": 3, "y": 4}}
    assert mod.label_to_action({"action": 6, "x": 8, "y": 9}) == {"action": 6, "data": {"x": 8, "y": 9}}
    assert mod.label_to_action({"not_action": 1}) is None
    assert mod.label_to_action(4) == {"action": 4}
    assert mod.label_to_action("validate") == {"action": 5}
    assert mod.label_to_action("undo") == {"action": 7}
    assert mod.label_to_action("click:1,2") == {"action": 6, "data": {"x": 1, "y": 2}}
    assert mod.label_to_action("not-json") is None
    assert mod.label_to_action("[1]") is None
    assert mod._labels_to_actions(["not-json", "4"]) == [{"action": 4}]
    assert mod._extract_nested_list({}, ("missing",)) == []

    _write_json(root / "results" / "nested.json", {"solve_trace": {"actions": [{"action": 2}]}})
    assert mod._actions_from_artifact(root, "results/nested.json")[0] == [{"action": 2}]
    _write_json(root / "results" / "empty.json", {"unused": []})
    assert mod._actions_from_artifact(root, "results/empty.json")[0] == []
    assert mod._scorecard_plan(root, "results/empty.json", "rows", "none")[0] == []
    _write_json(root / "results" / "rows.json", {"rows": [{"game": "other", "plan": ["4"]}]})
    assert mod._scorecard_plan(root, "results/rows.json", "rows", "none")[0] == []

    assert mod.sc25_adaptive_actions(["click1,2", "move3"]) == [
        {"action": 6, "data": {"x": 1, "y": 2}},
        {"action": 3},
    ]
    assert mod._honest_verdict(10, 33, []) == "complete: live_submission_gap_partially_closed_10_gaps_sharpened"
    assert mod._package_manifest({"per_game_submittable": "bad"}) == []
    assert mod._package_manifest({"per_game_submittable": ["bad", {"submittable_level": 0}]}) == []

    bad = {
        "honest_verdict": "partial",
        "inference_substrate": "bad",
        "verifier_is_oracle": True,
        "live_submittable_level_count": "x",
        "live_submittable_count_baseline": "x",
        "count_delta": "x",
        "trajectories_banked": "x",
        "env_adaptive_resolve_recovered": "x",
        "refreshed_package_path": "",
        "per_game_submittable": [
            "bad",
            {
                "game": "over",
                "submittable_level": 2,
                "offline_reproduced_level": 1,
                "registry_reproduced_level": 1,
                "has_trajectory": False,
                "has_env_adaptive_resolver": False,
            },
        ],
        "ready_for_operator_submit": "x",
        "offline_reproduced": [],
        "random_seed": "x",
        "reproducibility_checksum": "bad",
        "preconditions_checked": [],
    }
    errors = mod.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "per_game_submittable rows must be mappings" in errors
    assert "over submittable exceeds offline reproduction" in errors
    assert "over counted without trajectory or adaptive resolver" in errors

    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, bad)

    missing_and_mismatched = dict(bad)
    missing_and_mismatched.pop("honest_verdict")
    missing_and_mismatched.update(
        {
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "live_submittable_level_count": 2,
            "live_submittable_count_baseline": 1,
            "count_delta": 3,
            "trajectories_banked": [],
            "env_adaptive_resolve_recovered": [],
            "per_game_submittable": [],
            "ready_for_operator_submit": True,
            "offline_reproduced": {},
            "random_seed": mod.RANDOM_SEED,
            "reproducibility_checksum": "0" * 64,
            "preconditions_checked": {},
        }
    )
    errors = mod.artifact_schema_errors(missing_and_mismatched)
    assert "missing honest_verdict" in errors
    assert "count_delta must equal live_submittable_level_count - baseline" in errors


def test_scenario_capstone_4580_run_success_path_writes_artifacts(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4580: run writes the final artifact and refreshed package."""

    root = _fixture_root(tmp_path)
    artifact = mod.run(
        root,
        offline_arcade_checker=lambda: True,
        replay_actions_fn=lambda _game, _actions: 1,
        sc25_reproduce_fn=lambda _root, _target: 5,
        now=iter([10.0, 11.25, 11.5]).__next__,
    )

    assert artifact["live_submittable_level_count"] == 8
    assert artifact["env_adaptive_resolve_recovered"] == ["sc25"]
    assert (root / mod.RESULT_RELATIVE_PATH).exists()
    assert (root / mod.PACKAGE_RELATIVE_PATH).exists()
