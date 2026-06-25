"""Tests for Exp 4741 .436 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4741,
SCENARIO-ARC-WMTE-4741-PERSIST-STRONGEST-436-PRIMITIVE,
SCENARIO-ARC-WMTE-4741-LEAVE-ONE-GAME-TRANSFER.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from carnot import experiment_4741_primitive_persist_transfer as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _a1_artifact(*, arms_non_degenerate: bool = True) -> dict[str, Any]:
    return {
        "arms_non_degenerate": bool(arms_non_degenerate),
        "candidate_pool_differs_from_baseline": bool(arms_non_degenerate),
        "goal_energy_score_variance": 0.0004 if arms_non_degenerate else 0.0,
        "goal_energy_vs_baseline_delta": 0.0,
        "goal_free_l2_reached": False,
        "offline_reproduced": False,
        "reproduced_levels": 1,
        "reproducibility_checksum": "sha256:a1",
    }


def _a2_artifact(*, arms_non_degenerate: bool = True) -> dict[str, Any]:
    return {
        "arms_non_degenerate": bool(arms_non_degenerate),
        "novel_candidates_generated": 8 if arms_non_degenerate else 0,
        "energy_qd_vs_naive_delta": 0.0,
        "goal_free_l2_reached": False,
        "offline_reproduced": False,
        "reproduced_levels": 1,
        "reproducibility_checksum": "sha256:a2",
    }


def _attempt(
    game: str,
    variant: int,
    *,
    policy_mode: str,
    first_win: bool = False,
    actions: int = 19,
    input_count: int = 10,
    output_count: int = 14,
) -> dict[str, Any]:
    return {
        "game": game,
        "variant_signature": f"{game}~color{variant:02d}",
        "attempted": True,
        "first_win": bool(first_win),
        "solved": bool(first_win),
        "actions": int(actions),
        "actions_to_first_levelup": int(actions) if first_win else None,
        "policy_mode": policy_mode,
        "qd_generation_diagnostics": {
            "generator": {
                "candidate_pool": {
                    "input_candidate_count": int(input_count),
                    "output_candidate_count": int(output_count),
                    "generated_candidate_count": max(0, int(output_count) - int(input_count)),
                    "novel_candidates_generated": max(0, int(output_count) - int(input_count)),
                    "candidate_pool_jaccard_vs_naive": 0.75,
                    "behavior_descriptors": [[6, variant, 9], [6, variant, 10]],
                    "verifier_is_oracle": False,
                }
            }
        },
    }


def _measurement(attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in attempts]
    wins = [row for row in rows if row.get("first_win") is True or row.get("solved") is True]
    rate = 0.0 if not rows else round(len(wins) / len(rows), 6)
    return {
        "variant_attempts": rows,
        "variant_attempts_count": len(rows),
        "variant_solved_count": len(wins),
        "first_win_rate": rate,
        "solve_rate": rate,
        "variant_signatures": [str(row.get("variant_signature") or "") for row in rows],
    }


def _upstream_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    return _a1_artifact(), _a2_artifact()


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
    registry.write_text(REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(
        tmp_path / mod.A2_RELATIVE_PATH,
        {
            **_a2_artifact(),
            "naive_measurement": _measurement(
                [_attempt("aa00", 1, policy_mode="naive-search", input_count=10, output_count=10)]
            ),
            "qd_measurement": _measurement(
                [_attempt("aa00", 1, policy_mode="energy-QD", input_count=10, output_count=14)]
            ),
        },
    )


def test_req_arc_wmte_4741_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4741: OpenSpec declares the persistence/transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4741",
        "SCENARIO-ARC-WMTE-4741-PERSIST-STRONGEST-436-PRIMITIVE",
        "SCENARIO-ARC-WMTE-4741-LEAVE-ONE-GAME-TRANSFER",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4741_solver_kit_operator_ranks_qd_candidates() -> None:
    """REQ-ARC-WMTE-4741: persisted QD operator ranks without oracle authority."""

    result = kit.energy_fitness_qd_generator_operator(
        [
            {"candidate_id": "baseline_a", "action": 1, "energy_fitness": 0.6},
            {"candidate_id": "baseline_b", "action": 2, "energy_fitness": 0.4},
            {
                "candidate_id": "qd_target",
                "action": 6,
                "data": {"x": 17, "y": 25},
                "generated_by": "energy-QD",
                "energy_fitness": 0.1,
                "reaches_goal": True,
            },
        ]
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["generated_candidate_count"] == 1
    assert result["candidate_generation_coverage_delta"] == 1
    assert result["ranked_candidates"][0]["candidate_id"] == "qd_target"
    assert result["actions_to_first_goal_before"] == 3
    assert result["actions_to_first_goal_after"] == 1
    assert result["action_efficiency_lift"] == 2.0

    empty = kit.energy_fitness_qd_generator_operator([])
    assert empty["candidate_count"] == 0
    assert empty["value_added"] is False


def test_req_arc_wmte_4741_registry_and_live_selection_expose_operator() -> None:
    """REQ-ARC-WMTE-4741: registry and solver-kit selection expose the QD operator."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert gotchas[0]["derived_from"] == [
        "results/experiment_4738_energy_fitness_qd_generation_valid_test.json"
    ]
    assert gotchas[0]["transfer_dead_ends"]


def test_scenario_arc_wmte_4741_selects_a2_when_better_characterized() -> None:
    """SCENARIO-ARC-WMTE-4741-PERSIST-STRONGEST-436-PRIMITIVE: A2 wins."""

    a1, a2 = _upstream_fixture()
    decision = mod.select_primitive_from_upstreams(a1_artifact=a1, a2_artifact=a2)

    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["source"] == "A2_energy_fitness_qd_generator"
    assert decision["selected_reason"] == "a2_non_degenerate_qd_generator_with_novel_candidates"
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["upstream_signal_rank"][0]["source"] == "A2_energy_fitness_qd_generator"

    fallback = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(arms_non_degenerate=True),
        a2_artifact=_a2_artifact(arms_non_degenerate=False),
    )
    assert fallback["source"] == "A1_goal_energy_candidate_generation_guidance"


def test_scenario_arc_wmte_4741_leave_one_game_transfer_reports_deltas() -> None:
    """SCENARIO-ARC-WMTE-4741-LEAVE-ONE-GAME-TRANSFER: per-game deltas are honest."""

    naive = {
        "naive_measurement": _measurement(
            [
                _attempt("aa00", 1, policy_mode="naive-search", input_count=10, output_count=10),
                _attempt("aa00", 2, policy_mode="naive-search", input_count=10, output_count=10),
            ]
        )
    }
    qd = {
        "qd_measurement": _measurement(
            [
                _attempt("aa00", 1, policy_mode="energy-QD", input_count=10, output_count=14),
                _attempt("aa00", 2, policy_mode="energy-QD", input_count=10, output_count=14),
            ]
        )
    }

    result = mod.measure_transfer_game("aa00", a2_artifact={**naive, **qd})

    assert result["game"] == "aa00"
    assert result["excluded_from_characterization"] is True
    assert result["value_added"] is True
    assert result["transfer_value"]["candidate_generation_coverage_delta"] == 4.0
    assert result["transfer_value"]["first_win_rate_delta"] == 0.0
    assert result["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert result["transfer_value"]["action_efficiency_delta"] == 0.0
    assert result["transfer_value"]["offline_reproduced_new_level"] is False
    assert "generated coverage but no solve" in result["dead_end"]

    missing = mod.measure_transfer_game("missing", a2_artifact={})
    assert missing["value_added"] is False
    assert missing["dead_end"].startswith("no cached held-out")


def test_scenario_arc_wmte_4741_artifact_schema_and_run(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4741-LEAVE-ONE-GAME-TRANSFER: artifact writes stably."""

    _write_minimal_repo(tmp_path)
    a1, a2 = _upstream_fixture()
    selected = mod.select_primitive_from_upstreams(a1_artifact=a1, a2_artifact=a2)
    transfer_results = [
        {
            "game": game,
            "value_added": game == "aa00",
            "offline_reproduced_new_level": False,
            "dead_end": "" if game == "aa00" else "null transfer",
            "transfer_value": {
                "operator": mod.PRIMITIVE_OPERATOR,
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "candidate_generation_coverage_delta": 1.0 if game == "aa00" else 0.0,
                "action_efficiency_delta": 0.0,
                "offline_reproduced_new_level": False,
                "value_added": game == "aa00",
            },
        }
        for game in ("aa00", "bb00", "cc00")
    ]
    artifact = mod.build_artifact(
        selected_upstream=selected,
        preconditions_checked={"ok": True, "blocked_resource": ""},
        transfer_results=transfer_results,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: energy_fitness_qd_generator_operator_persisted_transfer_characterized"
    )
    assert artifact["verifier_is_oracle"] is False
    assert artifact["offline_reproduced_new_level"] is False
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    null_artifact = mod.build_artifact(
        selected_upstream=selected,
        preconditions_checked={"ok": True, "blocked_resource": ""},
        transfer_results=[{**row, "value_added": False, "dead_end": "null"} for row in transfer_results],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )
    assert null_artifact["honest_verdict"] == (
        "complete: energy_fitness_qd_generator_operator_persisted_transfer_null"
    )
    assert mod.artifact_schema_errors(null_artifact) == []

    run_artifact = mod.run(
        tmp_path,
        transfer_games=("aa00", "bb00", "cc00"),
        offline_arcade_checker=lambda: True,
        now=iter([10.0, 11.0]).__next__,
    )
    assert run_artifact["preconditions_checked"]["ok"] is True
    assert run_artifact["transfer_games"] == ["aa00", "bb00", "cc00"]
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
