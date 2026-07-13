"""Tests for Exp 4505 submitted-agent scoreboard refresh.

Spec refs: REQ-ARC-FCP-4505, SCENARIO-ARC-FCP-4505.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4505_submitted_agent_scoreboard as exp4505
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _value_weight_source() -> dict[str, object]:
    return {
        "experiment": "experiment_4500_value_weight_remeasure",
        "honest_verdict": "complete: value_weight_remeasure_null_keep_0_1_of_7",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "selected_value_weight": 0.0,
        "submitted_value_weight_after": 0.0,
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "selection": {
            "reason": "no_positive_weight_beats_control_within_budget",
            "selected_value_weight": 0.0,
            "should_raise_submitted_value_weight": False,
        },
        "heldout_games": ["tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09"],
        "per_weight": [
            {
                "value_weight": 0.0,
                "heldout_solve_rate": 1.0 / 7.0,
                "solved_games": 1,
                "attempted_games": 7,
                "median_actions_to_first_levelup": 20,
                "median_per_game_wall_seconds": 1.3,
                "per_game": [
                    {
                        "game": "lp85",
                        "solved": True,
                        "env_game_access_blocked": True,
                        "frame_only": True,
                    }
                ],
            }
        ],
        "flagged_adversarial": True,
    }


def _variant_source() -> dict[str, object]:
    return {
        "experiment": "experiment_4499_capstone_v415",
        "honest_verdict": (
            "complete: v415_a1_no_heldout_win_a2_oracle_distinct_"
            "a3_beats_0503_l2_not_banked_variant_transfer_0.28"
        ),
        "variant_transfer_rate": 0.28,
        "variant_transfer_scoreboard": {
            "honest_verdict": "complete: submitted_agent_scoreboard_generic_1_of_7_variant_7_of_25",
            "state": "variant_transfer_measured",
            "variant_transfer_solved": 7,
            "variant_transfer_attempted": 25,
            "variant_transfer_rate": 0.28,
        },
    }


def _write_upstream(root: Path) -> None:
    (root / exp4505.VALUE_WEIGHT_SOURCE_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / exp4505.VALUE_WEIGHT_SOURCE_RELATIVE_PATH).write_text(
        json.dumps(_value_weight_source(), indent=2),
        encoding="utf-8",
    )
    (root / exp4505.VARIANT_SOURCE_RELATIVE_PATH).write_text(
        json.dumps(_variant_source(), indent=2),
        encoding="utf-8",
    )
    (root / exp4505.REGISTRY_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / exp4505.REGISTRY_RELATIVE_PATH).write_text(
        "reproducible_total_levels: 47\n",
        encoding="utf-8",
    )


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "fixture-torch",
        "env_game_access_blocked": True,
        "value_weight_source_artifact_present": True,
        "variant_source_artifact_present": True,
        "registry_context_present": True,
        "parity_test_target": "tests/python/test_arc_submitted_agent_parity.py",
    }


def _parity_green() -> dict[str, object]:
    return {
        "command": ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q",
        "passed": True,
        "value_weight_assertion": f"value_weight=={SUBMITTED_AGENT_CONFIG.get('value_weight')}",
    }


def test_req_arc_fcp_4505_spec_declares_scoreboard_refresh_contract() -> None:
    """REQ-ARC-FCP-4505: OpenSpec names the refreshed scoreboard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4505" in spec
    assert "SCENARIO-ARC-FCP-4505" in spec
    assert exp4505.RESULT_RELATIVE_PATH in spec
    assert "value_weight==0.0" in spec
    assert "reproducible_total_levels" in spec
    for field, principle in exp4505.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_4505_refresh_tracks_real_leaderboard_signal(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4505: headline metrics exclude banked level count."""

    _write_upstream(tmp_path)
    artifact = exp4505.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        parity_gate_verified=_parity_green(),
        write=True,
    )

    assert artifact["honest_verdict"] == (
        "complete: submitted_agent_scoreboard_refresh_generic_1_of_7_variant_7_of_25_value_weight_1e-12"
    )
    assert artifact["inference_substrate"] == exp4505.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert artifact["preconditions_checked"]["torch_import"] is True
    assert artifact["field_principles"] == exp4505.FIELD_PRINCIPLES
    assert artifact["requirements"] == ["REQ-ARC-FCP-4505"]
    assert artifact["scenarios"] == ["SCENARIO-ARC-FCP-4505"]

    assert artifact["headline_metrics"][
        "submitted_default_heldout_generic_solve_rate"
    ] == pytest.approx(1.0 / 7.0)
    assert artifact["headline_metrics"]["submitted_default_heldout_generic_solved"] == 1
    assert artifact["headline_metrics"]["submitted_default_heldout_generic_attempted"] == 7
    assert artifact["headline_metrics"]["variant_transfer_rate"] == pytest.approx(7.0 / 25.0)
    assert artifact["headline_metrics"]["variant_transfer_solved"] == 7
    assert artifact["headline_metrics"]["variant_transfer_attempted"] == 25
    assert "reproducible_total_levels" not in artifact["headline_metrics"]

    assert artifact["context_metrics"]["reproducible_total_levels_context_only"] == 47
    assert artifact["context_metrics"]["reproducible_total_levels_is_headline"] is False
    assert artifact["context_metrics"]["leaderboard_signal"] == [
        "submitted_default_heldout_generic_solve_rate",
        "variant_transfer_rate",
    ]

    row = artifact["scoreboard_row"]
    assert row["submitted_agent_config"] == SUBMITTED_AGENT_CONFIG
    assert row["submitted_agent_config"]["value_weight"] == SUBMITTED_AGENT_CONFIG["value_weight"]
    assert row["heldout_generic_measurement"]["env_game_access_blocked"] is True
    assert row["heldout_generic_measurement"]["frame_only"] is True
    assert row["variant_transfer_measurement"]["solved"] == 7
    assert row["variant_transfer_measurement"]["attempted"] == 25

    # NOTE (2026-07-12): exp4500's checked-in artifact correctly still records
    # ITS OWN historical finding (selected/after == 0.0, the .415 B2
    # recommendation); PHASE A1 (REQ-LEARN-4652) later, deliberately moved
    # SUBMITTED_VALUE_WEIGHT to a tiny bounded-positive value for unrelated
    # reasons, so this scoreboard now correctly reports "value_weight_drift"
    # (a real, expected, non-error state) rather than the old
    # "keep_zero_value_weight" -- see _a1_value_weight_verdict's docstring.
    assert artifact["a1_value_weight_verdict"]["state"] == "value_weight_drift"
    assert artifact["a1_value_weight_verdict"]["selected_value_weight"] == 0.0
    assert artifact["a1_value_weight_verdict"]["submitted_value_weight_after"] == 0.0
    assert (
        artifact["a1_value_weight_verdict"]["current_submitted_value_weight"]
        == (SUBMITTED_AGENT_CONFIG["value_weight"])
    )
    assert artifact["a1_value_weight_verdict"]["value_weight_consistent_with_current"] is False
    assert artifact["a1_value_weight_verdict"]["source_flagged_adversarial"] is True
    assert artifact["parity_gate"]["verified_green"] is True
    assert artifact["parity_gate"]["value_weight_assertion"] == (
        f"value_weight=={SUBMITTED_AGENT_CONFIG.get('value_weight')}"
    )
    assert exp4505.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4505.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_req_arc_fcp_4505_schema_rejects_stale_config_and_banked_headline(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4505: schema rejects stale value_weight and banked-level headlines."""

    _write_upstream(tmp_path)
    artifact = exp4505.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        parity_gate_verified=_parity_green(),
        write=False,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: stale",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "preconditions_checked": [],
        "field_principles": {
            **artifact["field_principles"],
            "honest_verdict": {"principle": "wrong"},
        },
        "headline_metrics": {
            **artifact["headline_metrics"],
            "submitted_default_heldout_generic_solve_rate": {"value": 1.0 / 7.0},
            "submitted_default_heldout_generic_solved": "1",
            "submitted_default_heldout_generic_attempted": "7",
            "variant_transfer_rate": "0.28",
            "variant_transfer_solved": "7",
            "variant_transfer_attempted": "25",
            "reproducible_total_levels": 47,
        },
        "context_metrics": {"reproducible_total_levels_is_headline": True},
        "scoreboard_row": {
            **artifact["scoreboard_row"],
            "submitted_agent_config": {**SUBMITTED_AGENT_CONFIG, "value_weight": 5.0},
            "heldout_generic_measurement": {
                **artifact["scoreboard_row"]["heldout_generic_measurement"],
                "env_game_access_blocked": False,
            },
        },
        "a1_value_weight_verdict": {
            **artifact["a1_value_weight_verdict"],
            "submitted_value_weight_after": 5.0,
            # A wrong current_submitted_value_weight (not matching the LIVE
            # SUBMITTED_AGENT_CONFIG) is what the schema now actually rejects
            # -- see _a1_value_weight_verdict's docstring.
            "current_submitted_value_weight": 5.0,
        },
        "parity_gate": {**artifact["parity_gate"], "verified_green": False},
    }

    errors = exp4505.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must equal verifier_ensemble_against_cached_candidates" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "field_principles must match required principles" in errors
    assert (
        "headline_metrics.submitted_default_heldout_generic_solve_rate must be bare float" in errors
    )
    assert "headline_metrics.submitted_default_heldout_generic_solved must be bare int" in errors
    assert "headline_metrics.submitted_default_heldout_generic_attempted must be bare int" in errors
    assert "headline_metrics.variant_transfer_rate must be bare float" in errors
    assert "headline_metrics.variant_transfer_solved must be bare int" in errors
    assert "headline_metrics.variant_transfer_attempted must be bare int" in errors
    assert "headline_metrics must not include reproducible_total_levels" in errors
    assert "reproducible_total_levels must remain context-only" in errors
    assert "scoreboard_row.submitted_agent_config must match SUBMITTED_AGENT_CONFIG" in errors
    assert "scoreboard_row.heldout_generic_measurement must block env._game" in errors
    assert (
        "a1_value_weight_verdict.current_submitted_value_weight must match "
        "SUBMITTED_AGENT_CONFIG['value_weight']"
    ) in errors
    assert "parity_gate must record test_arc_submitted_agent_parity.py as green" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        exp4505.write_artifact(tmp_path, bad)

    missing = dict(artifact)
    del missing["headline_metrics"]
    assert "missing required field headline_metrics" in exp4505.artifact_schema_errors(missing)

    bad_resources = {
        **artifact,
        "preconditions_checked": {**_preconditions(), "torch_import": False},
    }
    assert (
        "preconditions_checked must record offline_arcade and torch resources"
        in exp4505.artifact_schema_errors(bad_resources)
    )

    bad_shapes = {
        **artifact,
        "headline_metrics": [],
        "context_metrics": [],
        "scoreboard_row": [],
        "a1_value_weight_verdict": [],
        "parity_gate": [],
    }
    shape_errors = exp4505.artifact_schema_errors(bad_shapes)
    assert "headline_metrics must be a mapping" in shape_errors
    assert "context_metrics must be a mapping" in shape_errors
    assert "scoreboard_row must be a mapping" in shape_errors
    assert "a1_value_weight_verdict must be a mapping" in shape_errors
    assert "parity_gate must be a mapping" in shape_errors

    heldout_shape = {
        **artifact,
        "scoreboard_row": {
            **artifact["scoreboard_row"],
            "heldout_generic_measurement": "not-a-measurement",
        },
    }
    assert (
        "scoreboard_row.heldout_generic_measurement must be a mapping"
        in exp4505.artifact_schema_errors(heldout_shape)
    )

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="env._game"):
        exp4505.run(
            root=empty_root,
            preconditions_checked=_preconditions(),
            parity_gate_verified=_parity_green(),
            write=False,
        )


def test_req_arc_fcp_4505_preconditions_record_verified_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4505: preconditions list the verified resources."""

    _write_upstream(tmp_path)
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")

    class FakeKit:
        @staticmethod
        def offline_arcade() -> object:
            return object()

    monkeypatch.setattr(exp4505, "_import_arc_solver_kit", lambda: FakeKit)
    monkeypatch.setattr(exp4505, "_import_torch_version", lambda: "fixture-torch")

    checks = exp4505.check_preconditions(tmp_path)

    assert checks == _preconditions()


def test_req_arc_fcp_4505_defensive_helpers_are_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4505: malformed local artifacts produce deterministic defaults."""

    _write_upstream(tmp_path)
    (tmp_path / exp4505.VALUE_WEIGHT_SOURCE_RELATIVE_PATH).write_text("{bad json", encoding="utf-8")
    (tmp_path / exp4505.VARIANT_SOURCE_RELATIVE_PATH).write_text("[]", encoding="utf-8")
    (tmp_path / exp4505.REGISTRY_RELATIVE_PATH).write_text("no count here\n", encoding="utf-8")

    assert exp4505._load_json(tmp_path, exp4505.VALUE_WEIGHT_SOURCE_RELATIVE_PATH) == {}
    assert exp4505._load_json(tmp_path, exp4505.VARIANT_SOURCE_RELATIVE_PATH) == {}
    assert exp4505._load_reproducible_total_levels(tmp_path) == 0
    assert exp4505._rate(0, 0) == 0.0
    assert exp4505._zero_weight_row({"per_weight": []}) == {}
    assert exp4505._as_int(2.0) == 2
    assert exp4505._as_int("3") == 3
    assert exp4505._as_int({"bad": True}) == 0
    assert exp4505._as_float("0.5") == pytest.approx(0.5)
    assert exp4505._as_float("bad") == 0.0
    assert exp4505._as_float({"bad": True}) == 0.0

    import carnot.agentic as agentic_pkg

    fake_kit = SimpleNamespace(offline_arcade=lambda: object())
    monkeypatch.setattr(agentic_pkg, "arc_solver_kit", fake_kit, raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fixture-torch"))

    assert exp4505._import_arc_solver_kit().offline_arcade
    assert exp4505._import_torch_version() == "fixture-torch"
