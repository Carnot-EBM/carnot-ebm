"""Tests for Exp 4692 directed-exploration primitive persistence.

Spec refs: REQ-ARC-WMTE-4692,
SCENARIO-ARC-WMTE-4692-PERSIST-STRONGEST-COMPONENT,
SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from carnot import experiment_4692_primitive_persist_transfer as mod
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _variant_attempt(
    game: str,
    *,
    policy_mode: str,
    reached_level: int = 0,
    first_win: bool = False,
    offline_reproduced: bool = False,
    diagnostics: Mapping[str, Any] | None = None,
    coverage_delta: float = 0.0,
) -> dict[str, Any]:
    return {
        "game": game,
        "policy_mode": policy_mode,
        "reached_level": reached_level,
        "first_win": first_win,
        "offline_reproduced": offline_reproduced,
        "candidate_generation_coverage_delta": coverage_delta,
        "controllable_novelty_diagnostics": dict(diagnostics or {}),
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level if offline_reproduced else 0,
            "reproduced": offline_reproduced,
        },
    }


def _a1_artifact() -> dict[str, Any]:
    controllable = {
        "enabled": True,
        "candidate_scores": 4641,
        "observed_effects": 194,
        "episodic_embeddings": 194,
        "rnd_updates": 194,
        "controllability_gate_on": True,
        "raw_frame_novelty": False,
        "controllability_gate_rejected": 0,
        "temperature": 0.5,
        "bonus_weight": 1.0,
        "verifier_is_oracle": False,
    }
    cosmetic = {
        "enabled": True,
        "candidate_scores": 5610,
        "observed_effects": 194,
        "episodic_embeddings": 194,
        "rnd_updates": 194,
        "controllability_gate_on": False,
        "raw_frame_novelty": True,
        "verifier_is_oracle": False,
    }
    return {
        "honest_verdict": (
            "complete: controllable_novelty_no_new_level_residual_winning_prefix_still_not_proposed"
        ),
        "chosen_submitted_config": "unchanged",
        "generic_agent_reached_level": 0,
        "reproduced_levels": 0,
        "generic_first_win_by_config": {
            "controllable_novelty_t0.5": {
                "variant_attempts": [
                    _variant_attempt(
                        "bp35",
                        policy_mode="controllable_novelty_t0.5",
                        diagnostics=controllable,
                    )
                ]
            },
            "no_novelty_bonus": {
                "variant_attempts": [
                    _variant_attempt("bp35", policy_mode="no_novelty_bonus")
                ]
            },
            "cosmetic_novelty_gate_off": {
                "variant_attempts": [
                    _variant_attempt(
                        "bp35",
                        policy_mode="cosmetic_novelty_gate_off",
                        diagnostics=cosmetic,
                    )
                ]
            },
        },
        "offline_reproduced": False,
        "residual_cause_hypothesis": "winning_prefix_still_not_proposed",
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: program_synthesis_filter_no_coverage_gain_residual_logged",
        "chosen_submitted_config": "unchanged",
        "candidate_generation_coverage_filter": 0.0,
        "candidate_generation_coverage_blind_baseline": 0.0,
        "coverage_delta": 0.0,
        "first_win_rate_delta": -0.04,
        "heldout_programs_kept": 0,
        "heldout_programs_rejected": 2,
        "target_arm_results": {
            "candidate_generation_probe": {
                "rows": [
                    {
                        "game": "bp35",
                        "filter_winner_in_pool": False,
                        "blind_winner_in_pool": False,
                        "heldout_programs_kept": 0,
                        "heldout_programs_rejected": 2,
                    }
                ]
            }
        },
        "offline_reproduced": False,
        "residual_bridge_gap": "heldout_transitions_too_sparse",
    }


def test_req_arc_wmte_4692_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4692: OpenSpec declares the directed-exploration contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4692",
        "SCENARIO-ARC-WMTE-4692-PERSIST-STRONGEST-COMPONENT",
        "SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT",
        mod.RESULT_RELATIVE_PATH,
        mod.PRIMITIVE_OPERATOR,
        mod.PRIMITIVE_GOTCHA_ID,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4692_solver_kit_operator_keeps_controllable_embeddings() -> None:
    """REQ-ARC-WMTE-4692: the persisted primitive rejects cosmetic raw-frame novelty."""

    result = kit.controllable_novelty_embedding_operator(
        [
            {
                "game": "bp35",
                "policy_mode": "controllable_novelty_t0.5",
                "observed_effects": 4,
                "episodic_embeddings": 3,
                "candidate_scores": 12,
                "controllability_gate_on": True,
                "raw_frame_novelty": False,
            },
            {
                "game": "bp35",
                "policy_mode": "cosmetic_novelty_gate_off",
                "observed_effects": 4,
                "episodic_embeddings": 3,
                "candidate_scores": 12,
                "controllability_gate_on": False,
                "raw_frame_novelty": True,
            },
            {"game": "bad", "observed_effects": "bad", "episodic_embeddings": True},
        ],
        min_observed_effects=1,
    )

    assert result["operator"] == mod.PRIMITIVE_OPERATOR
    assert result["verifier_is_oracle"] is False
    assert result["embedding_row_count"] == 3
    assert result["usable_embedding_count"] == 1
    assert result["rejected_embedding_count"] == 2
    assert result["coverage_ready"] is True
    assert result["controllable_novelty_embeddings"][0]["game"] == "bp35"
    assert result["controllable_novelty_embeddings"][0]["usable"] is True
    assert result["controllable_novelty_embeddings"][1]["usable"] is False

    empty = kit.controllable_novelty_embedding_operator([])
    assert empty["coverage_ready"] is False
    assert empty["residual"] == "no_controllable_novelty_rows"

    cosmetic_only = kit.controllable_novelty_embedding_operator(
        [
            {
                "observed_effects": 2,
                "episodic_embeddings": 2,
                "controllability_gate_on": False,
                "raw_frame_novelty": True,
            }
        ],
        min_observed_effects=True,
    )
    assert cosmetic_only["min_observed_effects"] == 1
    assert cosmetic_only["coverage_ready"] is False
    assert cosmetic_only["residual"] == "cosmetic_novelty_not_controllable"


def test_req_arc_wmte_4692_routing_and_registry_surface_persisted_operator() -> None:
    """REQ-ARC-WMTE-4692: routing and registry expose the persisted novelty primitive."""

    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in kit.primitive_operator_registry()}
    selected = kit.select_primitive_operators(mechanic_class="graph_explore", action_model="click")
    assert mod.PRIMITIVE_OPERATOR in {row.operator for row in selected}

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    gotchas = [
        row for row in registry["general_gotchas"] if row.get("id") == mod.PRIMITIVE_GOTCHA_ID
    ]
    assert len(gotchas) == 1
    assert gotchas[0]["operator"] == mod.PRIMITIVE_OPERATOR
    assert "controllable-novelty embedding" in gotchas[0]["note"]
    assert "latest_exp4692_transfer" in gotchas[0]


def test_scenario_arc_wmte_4692_selects_a1_embedding_when_a1_a2_null() -> None:
    """SCENARIO-ARC-WMTE-4692-PERSIST-STRONGEST-COMPONENT: null A1/A2 persists A1 embedding."""

    decision = mod.select_primitive_from_upstreams(
        a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert decision["source"] == "A1_controllable_novelty_embedding"
    assert decision["operator"] == mod.PRIMITIVE_OPERATOR
    assert decision["registry_general_gotcha_id"] == mod.PRIMITIVE_GOTCHA_ID
    assert decision["upstream_signal_rank"][0]["source"] == "A1_controllable_novelty_embedding"
    assert "both A1 and A2 were value-null" in decision["selection_rationale"]

    a1_cleared = dict(
        _a1_artifact(),
        honest_verdict="success: controllable_novelty_generic_agent_new_level_bp35_L1",
        generic_agent_reached_level=1,
        reproduced_levels=1,
    )
    assert (
        mod.select_primitive_from_upstreams(a1_artifact=a1_cleared, a2_artifact=_a2_artifact())[
            "source"
        ]
        == "A1_controllable_novelty_proposal_policy"
    )

    a2_cleared = dict(
        _a2_artifact(),
        honest_verdict="success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_bp35",
        coverage_delta=1.0,
    )
    assert (
        mod.select_primitive_from_upstreams(a1_artifact={}, a2_artifact=a2_cleared)["source"]
        == "A2_program_synthesis_action_effect_filter"
    )


def test_scenario_arc_wmte_4692_transfer_measurement_reports_cached_null_and_value() -> None:
    """SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT: transfer rows report value or null."""

    null = mod.measure_transfer_game(
        "bp35", a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )

    assert null["game"] == "bp35"
    assert null["value_added"] is False
    assert null["transfer_value"]["candidate_generation_coverage_delta"] == 0.0
    assert null["transfer_value"]["first_win_rate_delta"] == 0.0
    assert null["transfer_value"]["live_solve_rate_delta"] == 0.0
    assert null["transfer_value"]["offline_reproduced_new_level"] is False
    assert null["transfer_value"]["usable_embedding_count"] == 1
    assert "controllable embeddings produced no" in null["dead_end"]

    no_rows = mod.measure_transfer_game(
        "cd82", a1_artifact=_a1_artifact(), a2_artifact=_a2_artifact()
    )
    assert no_rows["value_added"] is False
    assert "no cached controllable-novelty transfer rows" in no_rows["dead_end"]

    lifted_a1 = _a1_artifact()
    lifted_a1["generic_first_win_by_config"]["controllable_novelty_t0.5"][
        "variant_attempts"
    ] = [
        _variant_attempt(
            "bp35",
            policy_mode="controllable_novelty_t0.5",
            reached_level=2,
            first_win=True,
            offline_reproduced=True,
            coverage_delta=1.0,
            diagnostics={
                "enabled": True,
                "candidate_scores": 10,
                "observed_effects": 4,
                "episodic_embeddings": 4,
                "controllability_gate_on": True,
                "raw_frame_novelty": False,
            },
        )
    ]
    value = mod.measure_transfer_game("bp35", a1_artifact=lifted_a1, a2_artifact=_a2_artifact())

    assert value["value_added"] is True
    assert value["transfer_value"]["candidate_generation_coverage_delta"] == 1.0
    assert value["transfer_value"]["first_win_rate_delta"] == 1.0
    assert value["transfer_value"]["live_solve_rate_delta"] == 1.0
    assert value["transfer_value"]["offline_reproduced_new_level"] is True

    cosmetic_a1 = _a1_artifact()
    cosmetic_a1["generic_first_win_by_config"]["controllable_novelty_t0.5"][
        "variant_attempts"
    ] = [
        _variant_attempt(
            "dc22",
            policy_mode="controllable_novelty_t0.5",
            diagnostics={
                "observed_effects": 4,
                "episodic_embeddings": 4,
                "controllability_gate_on": False,
                "raw_frame_novelty": True,
            },
        )
    ]
    cosmetic = mod.measure_transfer_game("dc22", a1_artifact=cosmetic_a1, a2_artifact={})
    assert "rejected raw-frame or cosmetic novelty" in cosmetic["dead_end"]

    insufficient_a1 = _a1_artifact()
    insufficient_a1["generic_first_win_by_config"]["controllable_novelty_t0.5"][
        "variant_attempts"
    ] = [
        _variant_attempt(
            "dc22",
            policy_mode="controllable_novelty_t0.5",
            diagnostics={
                "observed_effects": 0,
                "episodic_embeddings": 0,
                "controllability_gate_on": True,
                "raw_frame_novelty": False,
            },
        )
    ]
    insufficient = mod.measure_transfer_game("dc22", a1_artifact=insufficient_a1, a2_artifact={})
    assert "no usable controllable action-effect embeddings" in insufficient["dead_end"]


def test_scenario_arc_wmte_4692_artifact_schema_success_null_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT: artifact schema records transfer value."""

    decision = {
        "source": "A1_controllable_novelty_embedding",
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        "selection_rationale": "fixture",
    }
    rows = [
        {
            "game": game,
            "value_added": False,
            "transfer_value": {
                "candidate_generation_coverage_delta": 0.0,
                "live_solve_rate_delta": 0.0,
                "first_win_rate_delta": 0.0,
                "offline_reproduced_new_level": False,
            },
            "operator_result": {"operator": mod.PRIMITIVE_OPERATOR},
            "dead_end": "null",
        }
        for game in ("bp35", "cd82", "dc22")
    ]

    artifact = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["offline_reproduced_new_level"] is False
    assert "controllable-novelty embedding" in artifact["residual_dead_end"]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(mod.write_artifact(artifact, root=tmp_path).read_text(encoding="utf-8")) == artifact

    success_rows = [dict(row) for row in rows]
    success_rows[0] = {
        **success_rows[0],
        "value_added": True,
        "dead_end": "",
        "transfer_value": {
            **success_rows[0]["transfer_value"],
            "candidate_generation_coverage_delta": 1.0,
            "offline_reproduced_new_level": True,
        },
    }
    success = mod.build_artifact(
        selected_upstream=decision,
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=success_rows,
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=0.25,
    )
    assert success["honest_verdict"] == "success: primitive_persisted_transfer_value_characterized"
    assert success["residual_dead_end"] == ""
    assert mod.artifact_schema_errors(success) == []


def test_scenario_arc_wmte_4692_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT: run writes a stable artifact."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / mod.SPEC_RELATIVE_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / mod.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "general_gotchas": [
                    {
                        "id": mod.PRIMITIVE_GOTCHA_ID,
                        "operator": mod.PRIMITIVE_OPERATOR,
                        "note": "fixture",
                    }
                ],
                "games": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())

    artifact = mod.run(
        tmp_path,
        transfer_games=("bp35", "cd82", "dc22"),
        offline_arcade_checker=lambda: True,
        now=iter([4.0, 4.5]).__next__,
    )

    assert artifact["honest_verdict"] == "complete: primitive_persisted_transfer_null_characterized"
    assert artifact["transfer_games"] == ["bp35", "cd82", "dc22"]
    assert artifact["duration_s"] == 1.0
    assert artifact["preconditions_checked"]["ok"] is True
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4692_defensive_branches_are_schema_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4692: blocked and malformed inputs remain explicit."""

    assert "missing required field honest_verdict" in mod.artifact_schema_errors({})
    assert mod._load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert mod._load_json(bad) == {}
    assert mod._load_registry(tmp_path) == {}
    assert mod._registry_has_gotcha({"general_gotchas": "bad"}) is False
    assert mod._as_float(True) == 0.0
    assert mod._as_float("bad") == 0.0
    assert mod._as_int(False) == 0
    assert mod._as_int("bad") == 0
    assert mod._attempt_by_game({"variant_attempts": "bad"}, "bp35") is None
    assert mod._controllable_attempt(_a1_artifact(), "missing") is None
    assert mod._baseline_attempt(_a1_artifact(), "missing") is None
    prefix_a1 = {
        "generic_first_win_by_config": {
            "controllable_novelty_custom": {
                "variant_attempts": [
                    _variant_attempt("zz99", policy_mode="controllable_novelty_custom")
                ]
            },
            "controllable_novelty_malformed": "bad",
        }
    }
    assert mod._controllable_attempt(prefix_a1, "zz99")["policy_mode"] == (
        "controllable_novelty_custom"
    )
    assert mod._all_novelty_rows(prefix_a1) == []
    assert mod._program_probe_by_game(_a2_artifact(), "missing") is None
    assert (
        mod._program_probe_by_game(
            {"target_arm_results": {"candidate_generation_probe": {"rows": "bad"}}},
            "bp35",
        )
        is None
    )

    checks = mod.check_preconditions(
        tmp_path,
        offline_arcade_checker=lambda: (_ for _ in ()).throw(RuntimeError("offline missing")),
    )
    assert checks["offline_arcade"] is False
    assert "RuntimeError" in checks["offline_arcade_error"]

    blocked = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: False,
        now=iter([1.0, 1.1]).__next__,
    )
    assert blocked["honest_verdict"] == "blocked_primitive_persist_transfer_precondition"
    assert mod.artifact_schema_errors(blocked) == []

    malformed = mod.build_artifact(
        selected_upstream={
            "source": "A1_controllable_novelty_embedding",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {
                "game": "aa00",
                "value_added": True,
                "transfer_value": {"offline_reproduced_new_level": True},
                "dead_end": "",
            },
            {
                "game": "bb00",
                "value_added": False,
                "transfer_value": {"offline_reproduced_new_level": False},
                "dead_end": "null",
            },
            {
                "game": "cc00",
                "value_added": False,
                "transfer_value": {"offline_reproduced_new_level": False},
                "dead_end": "null",
            },
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=None,
    )
    assert malformed["offline_reproduced_new_level"] is True

    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "bad"
    malformed["verifier_is_oracle"] = True
    malformed["primitive_persisted"] = {}
    malformed["transfer_games"] = []
    malformed["transfer_value_per_game"] = []
    malformed["offline_reproduced_new_level"] = "yes"
    malformed["residual_dead_end"] = []
    malformed["random_seed"] = "bad"
    malformed["registry_updated"] = "yes"
    malformed["field_principles"] = {}
    malformed["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match REQ-ARC-WMTE-4692" in errors
    assert "verifier_is_oracle must be false" in errors
    assert f"primitive_persisted must name {mod.PRIMITIVE_OPERATOR}" in errors
    assert "offline_reproduced_new_level must be a bare bool" in errors
    assert "reproducibility_checksum must match artifact content" in errors

    wrong_gotcha = mod.build_artifact(
        selected_upstream={
            "source": "A1_controllable_novelty_embedding",
            "operator": mod.PRIMITIVE_OPERATOR,
            "registry_general_gotcha_id": "wrong",
        },
        upstream_signals={},
        preconditions_checked={"ok": True},
        transfer_results=[
            {"game": game, "value_added": False, "transfer_value": {}, "dead_end": "null"}
            for game in ("aa00", "bb00", "cc00")
        ],
        registry_updated=True,
        random_seed=mod.RANDOM_SEED,
        duration_s=1.0,
    )
    assert f"primitive_persisted must name {mod.PRIMITIVE_GOTCHA_ID}" in mod.artifact_schema_errors(
        wrong_gotcha
    )

    success_without_value = dict(wrong_gotcha)
    success_without_value["honest_verdict"] = (
        "success: primitive_persisted_transfer_value_characterized"
    )
    success_without_value["primitive_persisted"] = {
        "operator": mod.PRIMITIVE_OPERATOR,
        "registry_general_gotcha_id": mod.PRIMITIVE_GOTCHA_ID,
    }
    success_without_value["reproducibility_checksum"] = mod.payload_checksum(success_without_value)
    assert "success requires at least one transfer value_added=true" in mod.artifact_schema_errors(
        success_without_value
    )

    offline_mismatch = dict(success_without_value)
    offline_mismatch["honest_verdict"] = "complete: primitive_persisted_transfer_null_characterized"
    offline_mismatch["offline_reproduced"] = {"new_levels_banked": 2, "new_level_records": []}
    offline_mismatch["reproducibility_checksum"] = mod.payload_checksum(offline_mismatch)
    assert "offline_reproduced new_levels_banked must match records" in mod.artifact_schema_errors(
        offline_mismatch
    )

    with pytest.raises(ValueError):
        mod.write_artifact({}, root=tmp_path)

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        mod.run(tmp_path, offline_arcade_checker=lambda: False, now=iter([1.0, 1.1]).__next__)
