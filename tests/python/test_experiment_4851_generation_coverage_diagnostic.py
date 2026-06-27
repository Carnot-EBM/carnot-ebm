"""Tests for Exp 4851 L1 first-contact generation coverage diagnostic.

Spec refs: REQ-ARC-WMTE-4851,
SCENARIO-ARC-WMTE-4851-COVERAGE-BUCKETS,
SCENARIO-ARC-WMTE-4851-POSITIVE-CONTROL,
SCENARIO-ARC-WMTE-4851-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4851_generation_coverage_diagnostic as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _act(action: int, data: dict[str, int] | None = None) -> dict[str, Any]:
    return {"action": action, "data": data}


def _ground_truth() -> dict[str, list[dict[str, Any]]]:
    return {
        "aa00": [_act(1), _act(6, {"x": 3, "y": 4})],
        "bb00": [_act(2)],
        "cc00": [_act(3)],
    }


def _coverage_rows() -> dict[str, dict[str, Any]]:
    return {
        "aa00": {
            "bucket": "ENUMERATED_BUT_LOST",
            "winning_prefix_len": 2,
            "pool_size": 3,
            "reached_l1_win": False,
            "budget_actions": 12,
            "matched_winning_prefix_len": 2,
        },
        "bb00": {
            "bucket": "NEVER_ENUMERATED",
            "winning_prefix_len": 1,
            "pool_size": 2,
            "reached_l1_win": False,
            "budget_actions": 12,
            "matched_winning_prefix_len": 0,
        },
        "cc00": {
            "bucket": "NEVER_ENUMERATED",
            "winning_prefix_len": 1,
            "pool_size": 2,
            "reached_l1_win": False,
            "budget_actions": 12,
            "matched_winning_prefix_len": 0,
        },
    }


def test_req_arc_wmte_4851_spec_declares_generation_diagnostic_contract() -> None:
    """REQ-ARC-WMTE-4851: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4851",
        "SCENARIO-ARC-WMTE-4851-COVERAGE-BUCKETS",
        "SCENARIO-ARC-WMTE-4851-POSITIVE-CONTROL",
        "SCENARIO-ARC-WMTE-4851-BLOCKED-PRECONDITION",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4851_classifies_coverage_buckets() -> None:
    """SCENARIO-ARC-WMTE-4851-COVERAGE-BUCKETS: each game gets exactly one bucket."""

    winner = [_act(1), _act(6, {"x": 3, "y": 4})]
    records = [
        {"prefix": [], "candidates": [_act(1), _act(2)]},
        {"prefix": [_act(1)], "candidates": [_act(6, {"x": 3, "y": 4})]},
    ]
    lost = mod.classify_game_coverage(
        "aa00",
        winner,
        records,
        reached_l1_win=False,
        budget_actions=7,
    )
    covered = mod.classify_game_coverage(
        "aa00",
        winner,
        records,
        reached_l1_win=True,
        budget_actions=7,
    )
    missing = mod.classify_game_coverage(
        "aa00",
        winner,
        [{"prefix": [], "candidates": [_act(2)]}],
        reached_l1_win=False,
        budget_actions=7,
    )

    assert lost["bucket"] == "ENUMERATED_BUT_LOST"
    assert lost["pool_size"] == 3
    assert lost["matched_winning_prefix_len"] == 2
    assert covered["bucket"] == "COVERED"
    assert covered["reached_l1_win"] is True
    assert missing["bucket"] == "NEVER_ENUMERATED"
    assert missing["matched_winning_prefix_len"] == 0


def test_scenario_arc_wmte_4851_run_blocks_missing_preconditions(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4851-BLOCKED-PRECONDITION: missing resources never fabricate buckets."""

    blocked_arcade = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: False,
        ground_truth_loader=lambda _root: _ground_truth(),
        coverage_measurer=lambda **_kwargs: _coverage_rows()["aa00"],
        positive_control_runner=lambda **_kwargs: dict(_coverage_rows()["aa00"], bucket="COVERED"),
        live_path_checker=lambda _root: True,
        now=iter([1.0, 1.1]).__next__,
        write=False,
    )
    blocked_truth = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: {"aa00": [_act(1)]},
        coverage_measurer=lambda **_kwargs: _coverage_rows()["aa00"],
        positive_control_runner=lambda **_kwargs: dict(_coverage_rows()["aa00"], bucket="COVERED"),
        live_path_checker=lambda _root: True,
        now=iter([2.0, 2.1]).__next__,
        write=False,
    )

    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_arcade["per_game_coverage"] == {}
    assert blocked_arcade["preconditions_checked"]["offline_arcade"]["ok"] is False
    assert blocked_truth["honest_verdict"] == "blocked_no_banked_ground_truth"
    assert blocked_truth["n_games_measured"] == 0


def test_scenario_arc_wmte_4851_artifact_schema_positive_control_and_write(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4851-POSITIVE-CONTROL: covered control gates a real diagnostic."""

    rows = _coverage_rows()

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: _ground_truth(),
        coverage_measurer=lambda game, **_kwargs: dict(rows[game]),
        positive_control_runner=lambda **_kwargs: {
            "bucket": "COVERED",
            "winning_prefix_len": 3,
            "pool_size": 4,
            "reached_l1_win": True,
            "budget_actions": 12,
            "matched_winning_prefix_len": 3,
        },
        live_path_checker=lambda _root: True,
        now=iter([10.0, 10.2]).__next__,
        write=True,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == "complete_generation_wall_never_enumerated_dominant"
    assert artifact["dominant_bucket"] == "NEVER_ENUMERATED"
    assert artifact["positive_control_game"] == "tu93"
    assert artifact["positive_control_covered"] is True
    assert artifact["proposer_blind_to_banked_answer"] is True
    assert artifact["n_games_measured"] == 3
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4851_positive_control_failure_retires_measurement(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4851-POSITIVE-CONTROL: failed control is a harness artifact."""

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: _ground_truth(),
        coverage_measurer=lambda game, **_kwargs: dict(_coverage_rows()[game]),
        positive_control_runner=lambda **_kwargs: {
            "bucket": "NEVER_ENUMERATED",
            "winning_prefix_len": 3,
            "pool_size": 1,
            "reached_l1_win": False,
            "budget_actions": 12,
            "matched_winning_prefix_len": 0,
        },
        live_path_checker=lambda _root: True,
        now=iter([20.0, 20.2]).__next__,
        write=False,
    )

    assert artifact["honest_verdict"] == (
        "complete_generation_coverage_diagnostic_retired_positive_control_failed"
    )
    assert artifact["positive_control_covered"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4851_schema_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4851: malformed artifacts fail closed with named errors."""

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: _ground_truth(),
        coverage_measurer=lambda game, **_kwargs: dict(_coverage_rows()[game]),
        positive_control_runner=lambda **_kwargs: {
            "bucket": "COVERED",
            "winning_prefix_len": 3,
            "pool_size": 4,
            "reached_l1_win": True,
            "budget_actions": 12,
            "matched_winning_prefix_len": 3,
        },
        live_path_checker=lambda _root: True,
        now=iter([30.0, 30.2]).__next__,
        write=False,
    )
    malformed = dict(artifact)
    malformed.update(
        {
            "honest_verdict": "not_terminal",
            "per_game_coverage": {"aa00": {"bucket": "MAYBE"}},
            "positive_control_covered": False,
            "proposer_blind_to_banked_answer": False,
            "verifier_is_oracle": False,
            "live_path_reachable": False,
            "solve_provenance": "live_agent_self_discovery",
            "inference_substrate": "live_llm_inference",
            "reproducibility_checksum": "sha256:bad",
        }
    )

    errors = mod.artifact_schema_errors(malformed)
    assert "honest_verdict_terminal_prefix" in errors
    assert "per_game_coverage.aa00.bucket" in errors
    assert "positive_control_covered" in errors
    assert "proposer_blind_to_banked_answer" in errors
    assert "verifier_is_oracle" in errors
    assert "live_path_reachable" in errors
    assert "solve_provenance" in errors
    assert "inference_substrate" in errors
    assert "reproducibility_checksum" in errors


def test_req_arc_wmte_4851_helper_branches_and_blocked_writes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4851: defensive helper branches remain deterministic."""

    assert mod.normalize_action(None) is None
    assert mod.normalize_action("not-json") is None
    assert mod.normalize_action(7) is None
    assert mod.normalize_action({"data": None}) is None
    assert mod.normalize_action({"action": "bad"}) is None
    assert mod.normalize_action({"kind": "6", "x": 1.0, "y": 2}) == {
        "action": 6,
        "data": {"x": 1, "y": 2},
    }
    assert mod.normalize_action(
        {"action": 1, "data": {"a": None, "b": True, "c": 2.0, "d": "raw"}}
    ) == {"action": 1, "data": {"a": None, "b": True, "c": 2, "d": "raw"}}
    assert mod.normalize_action({"action": 2, "data": "raw"}) == {
        "action": 2,
        "data": "raw",
    }
    assert mod.action_key({"data": None}) == "<invalid>"
    assert mod.compute_dominant_bucket({}) is None

    no_distribution = mod.build_artifact(
        per_game_coverage={},
        positive_control_game="tu93",
        positive_control_coverage={"bucket": "COVERED"},
        preconditions_checked={},
        live_path_reachable=True,
        action_budget=12,
        max_depth=4,
        max_games=None,
        duration_s=0.0,
    )
    assert no_distribution["honest_verdict"] == (
        "complete_generation_coverage_diagnostic_retired_no_distribution"
    )
    assert "n_games_measured_minimum" in mod.artifact_schema_errors(no_distribution)

    assert mod.artifact_schema_errors({})[0].startswith("missing_field:")
    assert "field_principles" in mod.artifact_schema_errors(
        dict(no_distribution, field_principles=[])
    )
    bad_principles = {
        **no_distribution["field_principles"],
        "honest_verdict": {"principle": "different"},
    }
    assert "field_principles.honest_verdict" in mod.artifact_schema_errors(
        dict(no_distribution, field_principles=bad_principles)
    )
    assert "per_game_coverage" in mod.artifact_schema_errors(
        dict(no_distribution, per_game_coverage=[], n_games_measured=0)
    )
    assert "per_game_coverage.aa00" in mod.artifact_schema_errors(
        dict(no_distribution, per_game_coverage={"aa00": []}, n_games_measured=1)
    )

    invalid_rows = dict(no_distribution)
    invalid_rows.update(
        {
            "per_game_coverage": {
                "aa00": {
                    "bucket": "COVERED",
                    "winning_prefix_len": 0,
                    "pool_size": -1,
                    "budget_actions": 0,
                    "reached_l1_win": "yes",
                }
            },
            "dominant_bucket": "COVERED",
            "n_games_measured": "bad",
            "retire_if_same_verdict": False,
        }
    )
    row_errors = mod.artifact_schema_errors(invalid_rows)
    assert "per_game_coverage.aa00.winning_prefix_len" in row_errors
    assert "per_game_coverage.aa00.pool_size" in row_errors
    assert "per_game_coverage.aa00.budget_actions" in row_errors
    assert "per_game_coverage.aa00.reached_l1_win" in row_errors
    assert "n_games_measured" in row_errors
    assert "retire_if_same_verdict" in row_errors

    blocked = mod.build_blocked_artifact(
        "blocked_test",
        preconditions_checked={},
        duration_s=0.0,
    )
    blocked_bad = dict(blocked, per_game_coverage={"aa00": _coverage_rows()["aa00"]})
    blocked_bad["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_bad)
    assert "blocked_artifact_has_coverage" in mod.artifact_schema_errors(blocked_bad)

    try:
        mod._validate_or_raise({"bad": True})
    except mod.DiagnosticError as exc:
        assert "missing_field:" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("DiagnosticError not raised")

    written_arcade = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: False,
        ground_truth_loader=lambda _root: _ground_truth(),
        coverage_measurer=lambda **_kwargs: _coverage_rows()["aa00"],
        positive_control_runner=lambda **_kwargs: {"bucket": "COVERED"},
        live_path_checker=lambda _root: True,
        now=iter([40.0, 40.2]).__next__,
        write=True,
    )
    assert written_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    written_truth = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: {"aa00": [_act(1)]},
        coverage_measurer=lambda **_kwargs: _coverage_rows()["aa00"],
        positive_control_runner=lambda **_kwargs: {"bucket": "COVERED"},
        live_path_checker=lambda _root: True,
        now=iter([41.0, 41.2]).__next__,
        write=True,
    )
    assert written_truth["honest_verdict"] == "blocked_no_banked_ground_truth"

    live_blocked = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: _ground_truth(),
        coverage_measurer=lambda **_kwargs: _coverage_rows()["aa00"],
        positive_control_runner=lambda **_kwargs: {"bucket": "COVERED"},
        live_path_checker=lambda _root: False,
        now=iter([42.0, 42.2]).__next__,
        write=True,
    )
    assert live_blocked["honest_verdict"] == "blocked_live_path_unreachable"

    rows = _coverage_rows()
    max_games_artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        ground_truth_loader=lambda _root: {
            **_ground_truth(),
            "dd00": [_act(4)],
            "ee00": [_act(5)],
        },
        coverage_measurer=lambda game, **_kwargs: dict(rows.get(game, rows["bb00"])),
        positive_control_runner=lambda **_kwargs: {
            "bucket": "COVERED",
            "winning_prefix_len": 1,
            "pool_size": 1,
            "reached_l1_win": True,
            "budget_actions": 12,
        },
        live_path_checker=lambda _root: True,
        now=iter([43.0, 43.2]).__next__,
        write=False,
        max_games=3,
    )
    assert max_games_artifact["n_games_measured"] == 3
    assert max_games_artifact["preconditions_checked"]["ground_truth"]["n_available"] == 3


def test_req_arc_wmte_4851_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4851: final artifact is the requested diagnostic deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["positive_control_game"] == "tu93"
    assert artifact["positive_control_covered"] is True
    assert artifact["proposer_blind_to_banked_answer"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["live_path_reachable"] is True
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["n_games_measured"] >= 3
    assert artifact["dominant_bucket"] in mod.BUCKETS
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    for row in artifact["per_game_coverage"].values():
        assert row["bucket"] in mod.BUCKETS
        assert row["winning_prefix_len"] >= 1
        assert row["pool_size"] >= 0
        assert row["budget_actions"] > 0
