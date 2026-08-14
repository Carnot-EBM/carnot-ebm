"""REQ-ARC-ARM-6434: Exp6434 artifact schema and readiness gates."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from carnot import experiment_6434_arc_state_key_reachability_ab as exp


def _row(
    *,
    game: str,
    seed: int,
    arm: str,
    levels: int,
    actions: int,
    terminal: str,
    certificates: int = 0,
    states: int = 1,
    error: str | None = None,
) -> dict:
    return {
        "game": game,
        "seed": seed,
        "arm": arm,
        "levels_cleared": levels,
        "actions_spent": actions,
        "terminal_reason": terminal,
        "frontier_exhausted": terminal == "frontier_exhausted",
        "premature_frontier_collapse": terminal == "frontier_exhausted" and actions < 100,
        "unique_states": states,
        "alias_certificate_count": certificates,
        "environment_steps": actions,
        "legal_actions_checked": True,
        "exact_observations_checked": True,
        "cleared_state_observations": [{"level": 1}] if levels else [],
        "wall_s": 0.01,
        "action_cost": actions,
        "error": error,
    }


def _passing_rows() -> list[dict]:
    rows: list[dict] = []
    for seed in (20260814, 20260815, 20260816):
        rows.append(
            _row(
                game="base_clear",
                seed=seed,
                arm="baseline",
                levels=1,
                actions=10,
                terminal="cleared",
                states=5,
            )
        )
        rows.append(
            _row(
                game="base_clear",
                seed=seed,
                arm="opt_in",
                levels=1,
                actions=10,
                terminal="cleared",
                states=5,
            )
        )
        rows.append(
            _row(
                game="alias_case",
                seed=seed,
                arm="baseline",
                levels=0,
                actions=24,
                terminal="frontier_exhausted",
                states=1,
            )
        )
        rows.append(
            _row(
                game="alias_case",
                seed=seed,
                arm="opt_in",
                levels=0,
                actions=240,
                terminal="budget_exhausted",
                certificates=1,
                states=12,
            )
        )
    return rows


def test_build_artifact_sets_ready_only_when_reachability_gates_pass(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6434-MATCHED-AB: passing rows produce the required artifact."""
    artifact = exp.build_artifact(
        date="20260814",
        rows=_passing_rows(),
        collision_certificate_rows=[
            {
                "base_key": "frame:a",
                "observation_history_hashes": ["h0", "h1"],
                "alias_evidence": {"known_history_count": 2},
                "minimal_suffix_k": 1,
                "forbidden_inputs": [],
            }
        ],
        duration_s=1.0,
    )

    assert set(exp.REQUIRED_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["arc_state_key_reachability_ready_score"] == 1.0
    assert artifact["premature_frontier_collapse_delta"] < 0
    assert artifact["baseline_cleared_game_regression_count"] == 0
    assert artifact["new_error_count"] == 0
    assert artifact["level_solve_claimed"] is False
    assert artifact["solve_registry_modified"] is False
    assert artifact["route_default_promoted"] is False
    assert artifact["honest_verdict"].startswith("complete:")

    out = tmp_path / "artifact.json"
    exp.write_artifact(artifact, out)
    loaded = json.loads(out.read_text())
    assert exp.validate_artifact(loaded) == []


def test_regression_blocks_ready_score() -> None:
    """SCENARIO-ARC-ARM-6434-NO-SOLVE-OR-PROMOTION: a baseline clear cannot regress."""
    rows = _passing_rows()
    for row in rows:
        if row["game"] == "base_clear" and row["arm"] == "opt_in":
            row["levels_cleared"] = 0
            row["terminal_reason"] = "frontier_exhausted"

    artifact = exp.build_artifact(date="20260814", rows=rows, collision_certificate_rows=[])

    assert artifact["arc_state_key_reachability_ready_score"] == 0.0
    assert artifact["baseline_cleared_game_regression_count"] > 0
    assert "baseline_cleared_game_regression" in artifact["blocked_reason"]


def test_attack_matrix_failure_blocks_ready_score() -> None:
    """SCENARIO-ARC-ARM-6434-ATTACKS-FAIL-CLOSED: attacks fail before readiness."""
    artifact = exp.build_artifact(
        date="20260814",
        rows=_passing_rows(),
        collision_certificate_rows=[
            {
                "base_key": "frame:a",
                "observation_history_hashes": ["h0", "h1"],
                "alias_evidence": {"known_history_count": 2},
                "minimal_suffix_k": 1,
                "forbidden_inputs": [],
            }
        ],
        attack_overrides={"source_access": False},
    )

    assert artifact["arc_state_key_reachability_ready_score"] == 0.0
    assert "attack_failed" in artifact["blocked_reason"]


def test_validator_rejects_solve_credit_leakage() -> None:
    """SCENARIO-ARC-ARM-6434-NO-SOLVE-OR-PROMOTION: no solve claim can validate."""
    artifact = exp.build_artifact(date="20260814", rows=_passing_rows())
    artifact["level_solve_claimed"] = True

    errors = exp.validate_artifact(artifact)

    assert any("level_solve_claimed" in error for error in errors)


def test_validator_requires_terminal_verdict_prefix() -> None:
    artifact = exp.build_artifact(date="20260814", rows=_passing_rows())
    artifact["honest_verdict"] = "done without terminal prefix"

    assert any("honest_verdict" in error for error in exp.validate_artifact(artifact))


def test_validator_reports_all_no_solve_field_violations() -> None:
    artifact = exp.build_artifact(date="20260814", rows=_passing_rows())
    artifact.pop("status")
    artifact["solve_registry_modified"] = True
    artifact["route_default_promoted"] = True
    artifact["public_arc_claim_eligibility"] = True
    artifact["field_principles"] = {}
    artifact["reproducibility_checksum"] = "wrong"

    errors = exp.validate_artifact(artifact)

    assert any("missing required field status" in error for error in errors)
    assert any("solve_registry_modified" in error for error in errors)
    assert any("route_default_promoted" in error for error in errors)
    assert any("public_arc_claim_eligibility" in error for error in errors)
    assert any("field_principles" in error for error in errors)
    assert any("reproducibility_checksum" in error for error in errors)


def test_build_artifact_collects_certificates_from_row_stats_and_blocks_new_errors() -> None:
    rows = _passing_rows()
    for row in rows:
        if row["arm"] == "opt_in":
            row["stats"] = {
                "state_key_collision_certificates": [
                    {
                        "base_key": "frame:a",
                        "observation_history_hashes": ["h0", "h1"],
                        "alias_evidence": {"known_history_count": 2},
                        "minimal_suffix_k": 1,
                        "forbidden_inputs": [],
                    }
                ]
            }
            break
    rows[1]["error"] = "RuntimeError: boom"

    artifact = exp.build_artifact(date="20260814", rows=rows)

    assert artifact["collision_certificate_rows"]
    assert artifact["new_error_count"] == 1
    assert "new_errors" in artifact["blocked_reason"]


def test_build_artifact_compacts_heavy_stats_and_observation_payloads() -> None:
    rows = _passing_rows()
    rows[0]["cleared_state_observations"] = [{"pixels": [[1, 2], [3, 4]]}]
    rows[0]["stats"] = {
        "expansions": 3,
        "state_key_collision_certificate_count": 1,
        "state_key_collision_certificates": [{"huge": ["x"] * 32}],
        "frontier_seed_diagnostics": {"large": ["x"] * 32},
    }

    artifact = exp.build_artifact(
        date="20260814",
        rows=rows,
        collision_certificate_rows=[
            {
                "base_key": "frame:a",
                "observation_history_hashes": ["h0", "h1"],
                "alias_evidence": {"known_history_count": 2},
                "minimal_suffix_k": 1,
                "forbidden_inputs": [],
            }
        ],
    )
    row = artifact["per_unit_rows"][0]

    assert row["stats"] == {
        "expansions": 3,
        "state_key_collision_certificate_count": 1,
    }
    assert "state_key_collision_certificates" not in row["stats"]
    assert "frontier_seed_diagnostics" not in row["stats"]
    assert row["cleared_state_observations"][0]["observation_hash"]


def test_build_artifact_blocks_registry_mutation_and_no_collapse_delta() -> None:
    rows = [
        _row(game="g", seed=1, arm="baseline", levels=0, actions=200, terminal="budget_exhausted"),
        _row(game="g", seed=1, arm="opt_in", levels=0, actions=200, terminal="budget_exhausted"),
    ]

    artifact = exp.build_artifact(
        date="20260814",
        rows=rows,
        collision_certificate_rows=[
            {
                "base_key": "frame:a",
                "observation_history_hashes": ["h0", "h1"],
                "alias_evidence": {"known_history_count": 2},
                "minimal_suffix_k": 1,
                "forbidden_inputs": [],
            }
        ],
        solve_registry_modified=True,
    )

    assert artifact["arc_state_key_reachability_ready_score"] == 0.0
    assert "premature_frontier_collapse_not_decreased" in artifact["blocked_reason"]
    assert "solve_registry_modified" in artifact["blocked_reason"]


def test_small_helpers_cover_missing_and_dict_roster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{bad")
    registry = tmp_path / "registry.yaml"
    registry.write_text("games:\n  z9: {}\n  a1: {}\n")
    monkeypatch.setattr(exp, "REGISTRY", registry)

    assert exp._sha256_path(missing) is None
    assert exp._load_json(bad) == {}
    assert exp._roster() == ["a1", "z9"]


def _install_stub_arc(monkeypatch: pytest.MonkeyPatch, *, mode: str = "cleared") -> None:
    class _Grid:
        def tolist(self) -> list[list[int]]:
            return [[0]]

    class _Env:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self) -> object:
            return object()

        def step(self, *_args: object, **_kwargs: object) -> object:
            self.steps += 1
            return object()

    class _Arc:
        def open_scorecard(self) -> str:
            return "scorecard"

        def make(self, _game: str, scorecard_id: str) -> _Env:
            assert scorecard_id == "scorecard"
            return _Env()

    kit = ModuleType("carnot.agentic.arc_solver_kit")
    if mode == "error":
        kit.offline_arcade = lambda: (_ for _ in ()).throw(RuntimeError("arcade down"))  # type: ignore[attr-defined]
    else:
        kit.offline_arcade = lambda: _Arc()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_solver_kit", kit)
    import carnot.agentic as agentic
    import carnot.agentic.arc_graph_explore as graph

    monkeypatch.setattr(agentic, "arc_solver_kit", kit, raising=False)
    monkeypatch.setattr(graph, "grid_of", lambda _frame: _Grid())

    def fake_graph(env: _Env, *_args: object, **kwargs: object) -> tuple[list[dict] | None, int]:
        stats = kwargs["stats"]
        env.step("a")
        if mode == "frontier":
            stats.update(
                {
                    "states": 1,
                    "distinct_frames": 1,
                    "expansions": 1,
                    "state_key_collision_certificate_count": 0,
                    "state_key_collision_certificates": [],
                }
            )
            return None, 0
        if mode == "budget":
            stats.update(
                {
                    "states": 3,
                    "distinct_frames": 3,
                    "expansions": kwargs["max_expansions"],
                    "state_key_collision_certificate_count": 0,
                    "state_key_collision_certificates": [],
                }
            )
            return None, 0
        stats.update(
            {
                "states": 2,
                "distinct_frames": 1,
                "expansions": 1,
                "state_key_collision_certificate_count": 1,
                "state_key_collision_certificates": [
                    {
                        "base_key": "frame:a",
                        "observation_history_hashes": ["h0", "h1"],
                        "alias_evidence": {"known_history_count": 2},
                        "minimal_suffix_k": 1,
                        "forbidden_inputs": [],
                    }
                ],
            }
        )
        return [{"action": 1, "data": None}], 1

    monkeypatch.setattr(graph, "graph_explore_solve_v2", fake_graph)


def test_run_one_records_clears_and_certificates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_arc(monkeypatch)

    row, certs = exp._run_one("g", 7, "opt_in", max_expansions=5, max_depth=3)

    assert row["levels_cleared"] == 1
    assert row["terminal_reason"] == "cleared"
    assert row["actions_spent"] == 1
    assert row["alias_certificate_count"] == 1
    assert "state_key_collision_certificates" not in row["stats"]
    assert row["state_key_collision_certificate_receipts"]
    assert certs and certs[0]["game"] == "g"


def test_run_one_records_budget_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_arc(monkeypatch, mode="frontier")
    row, _certs = exp._run_one("g", 7, "baseline", max_expansions=20, max_depth=3)
    assert row["terminal_reason"] == "frontier_exhausted"
    assert row["premature_frontier_collapse"] is True

    _install_stub_arc(monkeypatch, mode="budget")
    row, _certs = exp._run_one("g", 7, "baseline", max_expansions=5, max_depth=3)
    assert row["terminal_reason"] == "budget_exhausted"

    _install_stub_arc(monkeypatch, mode="error")
    row, _certs = exp._run_one("g", 7, "baseline", max_expansions=5, max_depth=3)
    assert row["terminal_reason"] == "error"
    assert "arcade down" in row["error"]


def test_run_matched_ab_uses_both_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, str]] = []

    def fake_run_one(game: str, seed: int, arm: str, **_kwargs: object) -> tuple[dict, list]:
        calls.append((game, seed, arm))
        return _row(game=game, seed=seed, arm=arm, levels=0, actions=1, terminal="budget_exhausted"), []

    monkeypatch.setattr(exp, "_run_one", fake_run_one)

    rows, certs = exp.run_matched_ab(games=["g"], seeds=(1,), max_expansions=5, max_depth=3)

    assert [row["arm"] for row in rows] == ["baseline", "opt_in"]
    assert calls == [("g", 1, "baseline"), ("g", 1, "opt_in")]
    assert certs == []


def test_required_run_writes_the_task_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6434-MATCHED-AB: `main` writes the required JSON shape."""
    out = tmp_path / "experiment_6434_arc_state_key_reachability_ab.json"
    monkeypatch.setattr(exp, "ARTIFACT_PATH", out)
    monkeypatch.setattr(exp, "run_matched_ab", lambda **_kw: (_passing_rows(), []))

    rc = exp.main(["--date", "20260814"])

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["status"].startswith("complete")
    assert exp.validate_artifact(artifact) == []


def test_main_marks_invalid_artifact_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "invalid.json"
    monkeypatch.setattr(exp, "run_matched_ab", lambda **_kw: (_passing_rows(), []))
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced invalid"])

    rc = exp.main(["--date", "20260814", "--out", str(out), "--games", "g"])

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "complete_invalid"
    assert artifact["arc_state_key_reachability_ready_score"] == 0.0
