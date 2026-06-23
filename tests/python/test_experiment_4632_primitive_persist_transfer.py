"""Tests for Exp 4632 primitive persistence and transfer.

Spec refs: REQ-ARC-WMTE-4632, SCENARIO-ARC-WMTE-4632.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4632_primitive_persist_transfer as exp4632
from carnot.agentic import arc_solve_learning, arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _row(
    game: str,
    state_key: str,
    action_id: int,
    *,
    changed: bool,
    level_progress: bool = False,
    x: int | None = None,
    y: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "game": game,
        "env": game,
        "state_key": state_key,
        "action_id": action_id,
        "changed": changed,
        "level_progress": 1.0 if level_progress else 0.0,
    }
    if x is not None and y is not None:
        row["x"] = x
        row["y"] = y
    return row


class ScoreMap:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def candidate_score(self, _frame: object, candidate: object) -> float:
        if isinstance(candidate, dict):
            return float(self.scores.get(str(candidate.get("candidate_id")), 0.0))
        return float(self.scores.get(str(getattr(candidate, "candidate_id", "")), 0.0))


def test_req_arc_wmte_4632_spec_declares_transfer_contract() -> None:
    """REQ-ARC-WMTE-4632: OpenSpec declares the 4629 primitive transfer contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4632",
        "SCENARIO-ARC-WMTE-4632",
        exp4632.RESULT_RELATIVE_PATH,
        exp4632.PRIMITIVE_OPERATOR,
        exp4632.PRIMITIVE_GOTCHA_ID,
        "persistent_aem_plus_optional_cnn",
    ):
        assert marker in spec
    for field, principle in exp4632.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4632_solver_kit_operator_accepts_live_scorer() -> None:
    """REQ-ARC-WMTE-4632: the persisted AEM operator supports the 4629 live scorer."""

    memory = kit.PersistentAEM.from_effect_rows(
        [
            _row("train", "s", 1, changed=True),
            _row("train", "s", 6, changed=False, x=9, y=9),
        ]
    )
    result = kit.persistent_action_effect_memory_operator(
        [
            {"candidate_id": "memory_prefers_first", "action_id": 1, "reaches_levelup": False},
            {
                "candidate_id": "live_scorer_prefers_second",
                "action_id": 6,
                "data": {"x": 9, "y": 9},
                "reaches_levelup": True,
            },
        ],
        memory=memory,
        frame=object(),
        scorer=ScoreMap({"live_scorer_prefers_second": 0.99}),
    )

    assert result["operator"] == exp4632.PRIMITIVE_OPERATOR
    assert result["score_source"] == "live_action_effect_scorer"
    assert result["best_candidate_id"] == "live_scorer_prefers_second"
    assert result["actions_to_first_levelup_before"] == 2
    assert result["actions_to_first_levelup_after"] == 1
    assert result["value_added"] is True


def test_req_arc_wmte_4632_registry_and_routing_extend_existing_aem_primitive() -> None:
    """REQ-ARC-WMTE-4632: routing and registry extend, rather than duplicate, AEM."""

    operators = [
        row for row in kit.primitive_operator_registry() if row.operator == exp4632.PRIMITIVE_OPERATOR
    ]
    assert len(operators) == 1
    assert "exp4629_graduate_action_effect_predictor_live" in operators[0].derived_from_games
    assert "live_action_pruner" in operators[0].selector_tags

    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert exp4632.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    recommendation = arc_solve_learning.recommend_approach("sp80")
    recommended_ops = [row["operator"] for row in recommendation["selected_generic_operators"]]
    assert exp4632.PRIMITIVE_OPERATOR in recommended_ops

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == exp4632.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == exp4632.PRIMITIVE_OPERATOR
    assert gotchas[0]["latest_exp4632_transfer"]["artifact"] == exp4632.RESULT_RELATIVE_PATH


def test_req_arc_wmte_4632_selects_4629_action_effect_ranker_over_4628_null() -> None:
    """REQ-ARC-WMTE-4632: A2 wins when it has live action-efficiency lift."""

    decision = exp4632.select_primitive_from_upstreams(
        a1_artifact={
            "honest_verdict": "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened",
            "state_coverage_delta": 2,
            "first_win_rate_delta": 0.0,
        },
        a2_artifact={
            "honest_verdict": "success: action_effect_predictor_graduated_live_efficiency_up_1",
            "actions_delta": 1.0,
            "first_win_rate_delta": 0.18,
            "solve_rate_preserved": True,
            "chosen_submitted_config": "frame_change_predictor_enabled:persistent_aem_plus_optional_cnn",
        },
    )

    assert decision["source"] == "A2_action_effect_candidate_ranker"
    assert decision["operator"] == exp4632.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == exp4632.PRIMITIVE_GOTCHA_ID
    assert decision["solve_provenance"] == "live_agent_self_discovery"
    assert decision["measured_signal"] == 1.0
    assert decision["persisted_as_best_characterized_null"] is False

    null_decision = exp4632.select_primitive_from_upstreams(
        a1_artifact={"honest_verdict": "complete: no_live_lift", "state_coverage_delta": 2},
        a2_artifact={"honest_verdict": "complete: no_live_lift", "actions_delta": 0.0},
    )
    assert null_decision["persisted_as_best_characterized_null"] is True
    assert "degrade gracefully" in null_decision["selection_rationale"]


def test_req_arc_wmte_4632_transfer_measurement_excludes_target_game() -> None:
    """REQ-ARC-WMTE-4632: transfer uses target-excluded action-effect memory."""

    rows = [
        _row("train", "t", 6, changed=True, level_progress=True, x=32, y=32),
        _row("zz99", "s", 1, changed=False),
        _row("zz99", "s", 6, changed=True, level_progress=True, x=32, y=32),
        _row("yy88", "u", 1, changed=False),
        _row("yy88", "u", 2, changed=False),
    ]

    result = exp4632.measure_action_effect_ranker_transfer_game("zz99", effect_rows=rows)

    assert result["game"] == "zz99"
    assert result["target_game_excluded_from_memory"] is True
    assert result["value_added"] is True
    assert result["transfer_value"]["actions_delta"] == 1.0
    assert result["transfer_value"]["median_actions_to_first_levelup_bare"] == 2.0
    assert result["transfer_value"]["median_actions_to_first_levelup_predictor"] == 1.0
    assert result["transfer_value"]["target_game_excluded_from_memory"] is True
    assert result["offline_reproduced_new_level"] is False

    null_result = exp4632.measure_action_effect_ranker_transfer_game("yy88", effect_rows=rows)
    assert null_result["value_added"] is False
    assert null_result["dead_end"].startswith("target game rows had no effective")

    no_rows = exp4632.measure_action_effect_ranker_transfer_game("none", effect_rows=[])
    assert no_rows["dead_end"].startswith("no cached action-effect rows")


def test_scenario_arc_wmte_4632_artifact_schema_records_success_and_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4632: artifact schema records value-add or honest null."""

    decision = {
        "source": "A2_action_effect_candidate_ranker",
        "operator": exp4632.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": exp4632.PRIMITIVE_GOTCHA_ID,
        "inference_substrate": exp4632.INFERENCE_SUBSTRATE,
        "solve_provenance": "live_agent_self_discovery",
    }
    success = exp4632.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": True, "offline_arcade_import_smoke": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": True,
                "transfer_value": {
                    "actions_delta": 1.0,
                    "solve_rate_lift": 0.0,
                    "value_added": True,
                },
                "offline_reproduced_new_level": False,
                "dead_end": "",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {
                    "actions_delta": 0.0,
                    "solve_rate_lift": 0.0,
                    "value_added": False,
                },
                "offline_reproduced_new_level": False,
                "dead_end": "candidate generation remained the bottleneck",
            },
        ],
        registry_updated=True,
        random_seed=4632,
        duration_s=0.0,
        reproducible_total_levels=56,
    )

    assert success["honest_verdict"] == "success: primitive_persisted_transfer_zz99_value_added"
    assert success["verifier_is_oracle"] is False
    assert success["solve_provenance"] == "live_agent_self_discovery"
    assert success["offline_reproduced"] is False
    assert success["new_levels_banked"] == 0
    assert exp4632.artifact_schema_errors(success) == []

    out = exp4632.write_artifact(success, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == success

    null = exp4632.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "zz99",
                "value_added": False,
                "transfer_value": {"actions_delta": 0.0, "value_added": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no action-efficiency lift",
            },
            {
                "game": "yy88",
                "value_added": False,
                "transfer_value": {"actions_delta": 0.0, "value_added": False},
                "offline_reproduced_new_level": False,
                "dead_end": "no action-efficiency lift",
            },
        ],
        registry_updated=True,
        random_seed=4632,
        duration_s=0.0,
        reproducible_total_levels=56,
    )

    assert null["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert null["transfer_dead_ends"]["zz99"] == "no action-efficiency lift"
    assert exp4632.artifact_schema_errors(null) == []

    blocked = exp4632.build_artifact(
        upstream_decision=decision,
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed=4632,
        duration_s=None,
        reproducible_total_levels=56,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert exp4632.artifact_schema_errors(blocked) == []

    errors = exp4632.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "solve_provenance must be live_agent_self_discovery or development_proxy" in errors
    assert f"primitive_persisted must name {exp4632.PRIMITIVE_OPERATOR}" in errors
    assert "transfer_games must contain at least two games" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    wrong_gotcha = dict(success)
    wrong_gotcha["primitive_persisted"] = dict(success["primitive_persisted"])
    wrong_gotcha["primitive_persisted"]["registry_general_gotcha_id"] = "wrong"
    wrong_gotcha["reproducibility_checksum"] = exp4632.payload_checksum(wrong_gotcha)
    assert f"primitive_persisted must name {exp4632.PRIMITIVE_GOTCHA_ID}" in (
        exp4632.artifact_schema_errors(wrong_gotcha)
    )

    bad_success = dict(blocked)
    bad_success["honest_verdict"] = "success: primitive_persisted_transfer_x_value_added"
    bad_success["transfer_games"] = ["x", "y"]
    bad_success["transfer_value_per_game"] = {"x": {"value_added": False}}
    bad_success["reproducibility_checksum"] = exp4632.payload_checksum(bad_success)
    assert "success requires at least one transfer value_added=true" in (
        exp4632.artifact_schema_errors(bad_success)
    )

    bad_offline = dict(success)
    bad_offline["offline_reproduced"] = True
    bad_offline["new_levels_banked"] = 0
    bad_offline["reproducibility_checksum"] = exp4632.payload_checksum(bad_offline)
    assert "offline_reproduced=true requires at least one new level banked" in (
        exp4632.artifact_schema_errors(bad_offline)
    )

    tampered = dict(success)
    tampered["random_seed"] = 1
    assert "reproducibility_checksum must match artifact content" in (
        exp4632.artifact_schema_errors(tampered)
    )


def test_req_arc_wmte_4632_defensive_helpers_are_deterministic() -> None:
    """REQ-ARC-WMTE-4632: defensive null paths remain deterministic."""

    assert exp4632._as_float("bad") == 0.0  # noqa: SLF001
    assert exp4632._as_int("bad") == 0  # noqa: SLF001
    assert exp4632._row_game(SimpleNamespace(env="objgame")) == "objgame"  # noqa: SLF001

    rows = [
        _row("aa00", "s", 1, changed=False),
        _row("aa00", "s", 2, changed=True),
        _row("bb00", "t", 1, changed=False),
        _row("bb00", "t", 2, changed=False),
    ]
    selected = exp4632.select_transfer_results(rows, games=("aa00", "bb00"), limit=2)
    assert [row["game"] for row in selected] == ["aa00", "bb00"]

    blocked = exp4632.build_artifact(
        upstream_decision={
            "source": "A2_action_effect_candidate_ranker",
            "operator": exp4632.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": exp4632.PRIMITIVE_GOTCHA_ID,
            "inference_substrate": "wrong",
            "solve_provenance": "outer_loop_re",
        },
        preconditions_checked={"ok": False},
        transfer_results=[],
        registry_updated=False,
        random_seed="bad",  # type: ignore[arg-type]
        duration_s=None,
        reproducible_total_levels=None,
    )
    assert "inference_substrate must match the 4632 offline substrate" in (
        exp4632.artifact_schema_errors(blocked)
    )
