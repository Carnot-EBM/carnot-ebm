"""Tests for Exp 4496 submitted-agent headline scoreboard.

Spec refs: REQ-ARC-FCP-4496, SCENARIO-ARC-FCP-4496.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_4496_submitted_agent_scoreboard as exp4496
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _submitted_benchmark() -> dict[str, object]:
    return {
        "experiment": "experiment_4475_wire_stronger_generic_stack",
        "honest_verdict": "complete: submitted_default_stronger_generic_stack_wired",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "after_generic_solve_rate": 1.0 / 7.0,
        "after_solved": 1,
        "attempted_games": 7,
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "tests_pass": True,
        "preconditions_checked": {"env_game_blocked": True},
        "benchmark": {
            "measurement": "heldout_loo_generic_set_exact_submitted_default_env_game_blocked",
            "games": ["tr87", "tu93", "lp85", "sc25", "ka59", "ar25", "ft09"],
        },
    }


def _variant_context() -> dict[str, object]:
    return {
        "experiment": "experiment_4481_variant_transfer_benchmark",
        "honest_verdict": "success: reflection_variant_transfer_1_of_25_rate_0.0400_games_25",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "variants_solved": 1,
        "variants_attempted": 25,
        "transfer_solve_rate": 0.04,
        "reproducible_total_levels": 47,
        "result_path": "results/experiment_4481_variant_transfer_benchmark.json",
    }


def _write_upstream(root: Path) -> None:
    (root / exp4496.SUBMITTED_BENCHMARK_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / exp4496.SUBMITTED_BENCHMARK_RELATIVE_PATH).write_text(
        json.dumps(_submitted_benchmark(), indent=2),
        encoding="utf-8",
    )
    (root / exp4496.VARIANT_CONTEXT_RELATIVE_PATH).write_text(
        json.dumps(_variant_context(), indent=2),
        encoding="utf-8",
    )


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "fixture",
        "env_game_access_blocked": True,
        "submitted_benchmark_artifact_present": True,
        "variant_context_artifact_present": True,
        "parity_test_target": "tests/python/test_arc_submitted_agent_parity.py",
    }


def test_req_arc_fcp_4496_spec_declares_scoreboard_contract() -> None:
    """REQ-ARC-FCP-4496: OpenSpec names the submitted-agent scoreboard contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4496" in spec
    assert "SCENARIO-ARC-FCP-4496" in spec
    assert exp4496.RESULT_RELATIVE_PATH in spec
    assert "SUBMITTED_AGENT_CONFIG" in spec
    assert "reproducible_total_levels" in spec
    for field, principle in exp4496.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4496_scoreboard_tracks_headline_not_levels(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4496: headline metrics are solve-rate plus variant transfer."""

    _write_upstream(tmp_path)
    artifact = exp4496.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        write=True,
        variant_transfer_signal={
            "variants_solved": 7,
            "variants_attempted": 25,
            "source": "operator_milestone_prompt_current_value",
            "milestone": "2026.06.415",
        },
    )

    assert artifact["honest_verdict"] == (
        "complete: submitted_agent_scoreboard_generic_1_of_7_variant_7_of_25"
    )
    assert artifact["inference_substrate"] == exp4496.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["offline_arcade_import_smoke"] is True
    assert artifact["preconditions_checked"]["torch_import"] is True
    assert artifact["parity_gate"]["test_path"] == "tests/python/test_arc_submitted_agent_parity.py"
    assert artifact["parity_gate"]["expected_green"] is True
    assert artifact["headline_metrics"]["submitted_default_heldout_generic_solve_rate"] == pytest.approx(
        1.0 / 7.0
    )
    assert artifact["headline_metrics"]["submitted_default_heldout_generic_solved"] == 1
    assert artifact["headline_metrics"]["submitted_default_heldout_generic_attempted"] == 7
    assert artifact["headline_metrics"]["variant_transfer_rate"] == pytest.approx(7.0 / 25.0)
    assert artifact["headline_metrics"]["variant_transfer_solved"] == 7
    assert artifact["headline_metrics"]["variant_transfer_attempted"] == 25
    assert "reproducible_total_levels" not in artifact["headline_metrics"]
    assert artifact["context_metrics"]["reproducible_total_levels_context_only"] == 47
    assert artifact["context_metrics"]["reproducible_total_levels_is_headline"] is False

    row = artifact["milestone_rows"][0]
    assert row["milestone"] == "2026.06.415"
    assert row["submitted_agent_config"] == SUBMITTED_AGENT_CONFIG
    assert row["heldout_generic_measurement"]["env_game_access_blocked"] is True
    assert row["heldout_generic_measurement"]["frame_only"] is True
    assert row["heldout_generic_measurement"]["solved"] == 1
    assert row["variant_transfer_measurement"]["solved"] == 7
    assert row["variant_transfer_measurement"]["source"] == "operator_milestone_prompt_current_value"
    assert row["variant_context"]["checked_in_artifact_solved"] == 1
    assert row["variant_context"]["checked_in_artifact_attempted"] == 25
    assert exp4496.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4496.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_req_arc_fcp_4496_schema_rejects_headline_confusion(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4496: schema rejects wrapped metrics and banked-level headlines."""

    _write_upstream(tmp_path)
    artifact = exp4496.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        write=False,
        variant_transfer_signal={"variants_solved": 7, "variants_attempted": 25},
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: stale",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "preconditions_checked": [],
        "field_principles": {**artifact["field_principles"], "honest_verdict": "wrong"},
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
        "milestone_rows": [
            {
                **artifact["milestone_rows"][0],
                "submitted_agent_config": {"policy": "CarnotAgentPolicy"},
                "heldout_generic_measurement": {
                    **artifact["milestone_rows"][0]["heldout_generic_measurement"],
                    "env_game_access_blocked": False,
                },
            }
        ],
    }

    errors = exp4496.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must equal verifier_ensemble_against_cached_candidates" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "field_principles must match required principles" in errors
    assert "headline_metrics.submitted_default_heldout_generic_solve_rate must be bare float" in errors
    assert "headline_metrics.submitted_default_heldout_generic_solved must be bare int" in errors
    assert "headline_metrics.submitted_default_heldout_generic_attempted must be bare int" in errors
    assert "headline_metrics.variant_transfer_rate must be bare float" in errors
    assert "headline_metrics.variant_transfer_solved must be bare int" in errors
    assert "headline_metrics.variant_transfer_attempted must be bare int" in errors
    assert "headline_metrics must not include reproducible_total_levels" in errors
    assert "reproducible_total_levels must remain context-only" in errors
    assert "milestone_rows[0].submitted_agent_config must match SUBMITTED_AGENT_CONFIG" in errors
    assert "milestone_rows[0].heldout_generic_measurement must block env._game" in errors
    with pytest.raises(ValueError, match="honest_verdict"):
        exp4496.write_artifact(tmp_path, bad)


def test_req_arc_fcp_4496_preconditions_record_verified_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4496: preconditions list the import smoke and Torch resource."""

    _write_upstream(tmp_path)
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")

    class FakeKit:
        @staticmethod
        def offline_arcade() -> object:
            return object()

    monkeypatch.setattr(exp4496, "_import_arc_solver_kit", lambda: FakeKit)
    monkeypatch.setattr(exp4496, "_import_torch_version", lambda: "fixture-torch")

    checks = exp4496.check_preconditions(tmp_path)

    assert checks["agents_md_read"] is True
    assert checks["codex_md_read"] is True
    assert checks["offline_arcade_import_smoke"] is True
    assert checks["torch_import"] is True
    assert checks["torch_version"] == "fixture-torch"
    assert checks["env_game_access_blocked"] is True
    assert checks["submitted_benchmark_artifact_present"] is True
    assert checks["variant_context_artifact_present"] is True


def test_req_arc_fcp_4496_defensive_branches_remain_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-FCP-4496: helper and malformed-shape branches remain deterministic."""

    _write_upstream(tmp_path)
    artifact = exp4496.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        write=False,
        variant_transfer_signal={"variants_solved": 7.0, "variants_attempted": "25"},
    )

    import carnot.agentic as agentic_pkg

    fake_kit = SimpleNamespace(offline_arcade=lambda: object())
    monkeypatch.setattr(agentic_pkg, "arc_solver_kit", fake_kit, raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fixture-torch"))

    assert exp4496._import_arc_solver_kit().offline_arcade
    assert exp4496._import_torch_version() == "fixture-torch"
    assert exp4496._as_int(2.0) == 2
    assert exp4496._as_int("3") == 3
    assert exp4496._as_int({"bad": True}) == 0
    assert exp4496._as_float("0.5") == pytest.approx(0.5)
    assert exp4496._as_float("bad") == 0.0
    assert exp4496._as_float({"bad": True}) == 0.0

    missing = dict(artifact)
    del missing["headline_metrics"]
    assert "missing required field headline_metrics" in exp4496.artifact_schema_errors(missing)

    bad_shapes = {
        **artifact,
        "headline_metrics": [],
        "context_metrics": [],
        "milestone_rows": [],
    }
    shape_errors = exp4496.artifact_schema_errors(bad_shapes)
    assert "headline_metrics must be a mapping" in shape_errors
    assert "context_metrics must be a mapping" in shape_errors
    assert "milestone_rows must be a non-empty list" in shape_errors

    row_shape = {**artifact, "milestone_rows": ["not-a-row"]}
    assert "milestone_rows[0] must be a mapping" in exp4496.artifact_schema_errors(row_shape)

    heldout_shape = {
        **artifact,
        "milestone_rows": [
            {
                **artifact["milestone_rows"][0],
                "heldout_generic_measurement": "not-a-measurement",
            }
        ],
    }
    assert (
        "milestone_rows[0].heldout_generic_measurement must be a mapping"
        in exp4496.artifact_schema_errors(heldout_shape)
    )

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="env._game"):
        exp4496.run(root=empty_root, preconditions_checked=_preconditions(), write=False)
