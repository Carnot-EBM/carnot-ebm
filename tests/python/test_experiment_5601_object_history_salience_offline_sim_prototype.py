"""Tests for Exp5601 ObjectHistorySaliencePrior offline-sim prototype.

Spec refs: REQ-ARC-FCP-5591-2,
SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from carnot import experiment_5601_object_history_salience_offline_sim_prototype as exp


REPO = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


class _Env:
    def reset(self) -> None:
        return None


class _Arcade:
    def __init__(self, *, fail_make: bool = False) -> None:
        self.fail_make = fail_make

    def open_scorecard(self) -> str:
        return "scorecard"

    def make(self, _game: str, *, scorecard_id: str) -> _Env:
        if self.fail_make:
            raise RuntimeError(f"blocked {scorecard_id}")
        return _Env()


class _FakeProposer:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakePolicy:
    def __init__(self, game: str, *, proposer: _FakeProposer, explore_budget: int) -> None:
        self.game = game
        self.proposer = proposer
        self.explore_budget = explore_budget
        self.transitions = [
            SimpleNamespace(action=6, data={"x": 1, "y": 1}, grid=object(), next_grid=object()),
            SimpleNamespace(action=6, data={"x": 2, "y": 2}, grid=object(), next_grid=object()),
            SimpleNamespace(action=6, data={"x": 99, "y": 99}, grid=object(), next_grid=object()),
        ]


class _FakeBasePrior:
    def score(self, _grid: object, candidate: dict[str, object]) -> float:
        data = candidate["data"]
        assert isinstance(data, dict)
        return 99.0 if data["x"] == 99 else 1.0


class _FakeObjectHistoryPrior:
    def __init__(self, *, min_observations: int) -> None:
        self.min_observations = min_observations
        self.base_prior = _FakeBasePrior()
        self._tally: dict[str, dict[str, int]] = {}

    @property
    def tracked_hash_count(self) -> int:
        return len(self._tally)

    def observe_transition(
        self,
        _grid: object,
        _action: int,
        _data: object,
        _next_grid: object,
    ) -> None:
        self._tally["hash-a"] = {"obs": self.min_observations, "changed": 1}

    def score(self, _grid: object, candidate: dict[str, object]) -> float:
        data = candidate["data"]
        assert isinstance(data, dict)
        return float(data["x"])


def _patch_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(exp.Path, "home", classmethod(lambda cls: tmp_path))


def _module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _patch_precondition_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    arcade: _Arcade | None = None,
    fail_arcade_import: bool = False,
    fail_e3_import: bool = False,
) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "carnot.agentic" and "arc_solver_kit" in fromlist:
            if fail_arcade_import:
                raise ImportError("blocked arcade import")
            return SimpleNamespace(
                arc_solver_kit=SimpleNamespace(offline_arcade=lambda: arcade or _Arcade())
            )
        if name == "carnot.agentic.arc_competition_agent":
            if fail_e3_import:
                raise ImportError("blocked e3 import")
            return _module(name, E3AgentPolicy=object)
        if name == "carnot.agentic.arc_executable_world_model":
            return _module(name, LocalGGUFProposer=object)
        if name == "carnot.agentic.arc_object_history_salience":
            return _module(name, ObjectHistorySaliencePrior=object)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_scenario_5601_preconditions_success_and_cache_detection(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL preflight passes."""

    _patch_home(monkeypatch, tmp_path)
    (tmp_path / ".cache" / "huggingface" / "hub" / "models--Qwen3.5-9B-MTP-GGUF").mkdir(
        parents=True
    )
    llama = tmp_path / ".cache" / "llama.cpp-local" / "build" / "bin" / "llama-server"
    llama.parent.mkdir(parents=True)
    llama.write_text("#!/bin/sh\n", encoding="utf-8")

    _patch_precondition_imports(monkeypatch, arcade=_Arcade())

    checks = exp.preconditions(tmp_path)

    assert checks == {
        "offline_arcade_importable": True,
        "offline_arcade_makes_env": True,
        "e3_and_prior_import": True,
        "gguf_cached": True,
        "llama_server_binary_present": True,
        "ok": True,
    }


def test_scenario_5601_preconditions_report_failed_edges(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL preflight fails closed."""

    _patch_home(monkeypatch, tmp_path)
    _patch_precondition_imports(
        monkeypatch,
        arcade=_Arcade(fail_make=True),
        fail_e3_import=True,
    )

    checks = exp.preconditions(tmp_path)

    assert checks["offline_arcade_importable"] is True
    assert checks["offline_arcade_makes_env"] is False
    assert checks["e3_and_prior_import"] is False
    assert checks["gguf_cached"] is False
    assert checks["llama_server_binary_present"] is False
    assert checks["ok"] is False


def test_scenario_5601_preconditions_handles_offline_arcade_import_failure(
    monkeypatch,
    tmp_path,
) -> None:
    """REQ-ARC-FCP-5591-2: missing arcade support is a blocked precondition."""

    _patch_home(monkeypatch, tmp_path)
    _patch_precondition_imports(monkeypatch, fail_arcade_import=True)

    checks = exp.preconditions(tmp_path)

    assert checks["offline_arcade_importable"] is False
    assert checks["e3_and_prior_import"] is True
    assert checks["ok"] is False


def test_scenario_5601_helpers_find_first_miss_and_stable_checksum() -> None:
    """REQ-ARC-FCP-5591-2: helper outputs are deterministic."""

    assert exp._first_precondition_miss({"ok": True}) is None
    assert exp._first_precondition_miss({"ok": False, "arcade": False}) == "arcade"
    assert exp._checksum({"b": 2, "a": 1}) == exp._checksum({"a": 1, "b": 2})


def test_scenario_5601_measure_one_game_with_mocked_real_clicks(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL measures real-click shape."""

    monkeypatch.setitem(
        exp.sys.modules,
        "arc_leaderboard_eval",
        SimpleNamespace(run_game=lambda _game, _policy, *, budget: None),
    )
    monkeypatch.setitem(
        exp.sys.modules,
        "carnot.agentic.arc_competition_agent",
        _module("carnot.agentic.arc_competition_agent", E3AgentPolicy=_FakePolicy),
    )
    monkeypatch.setitem(
        exp.sys.modules,
        "carnot.agentic.arc_executable_world_model",
        _module("carnot.agentic.arc_executable_world_model", LocalGGUFProposer=_FakeProposer),
    )
    monkeypatch.setitem(
        exp.sys.modules,
        "carnot.agentic.arc_object_history_salience",
        _module(
            "carnot.agentic.arc_object_history_salience",
            ObjectHistorySaliencePrior=_FakeObjectHistoryPrior,
        ),
    )

    row = exp._measure_one_game("m0r0", explore_budget=2, total_budget=3)

    assert row["game"] == "m0r0"
    assert row["transitions_collected"] == 3
    assert row["click_transitions"] == 3
    assert row["hashes_tracked_after"] == 1
    assert row["hashes_with_evidence_after"] == 1
    assert row["hashes_with_evidence_and_nonzero_change_rate_after"] == 1
    assert row["degeneracy_pairs_checked"] == 1
    assert row["degeneracy_pairs_differentiated_by_history"] == 1


def test_scenario_5601_build_artifact_blocked_path(monkeypatch) -> None:
    """REQ-ARC-FCP-5591-2: blocked preconditions still write a terminal artifact."""

    monkeypatch.setattr(
        exp,
        "preconditions",
        lambda _root: {"offline_arcade_importable": False, "ok": False},
    )

    artifact = exp.build_artifact(roster=("m0r0",), explore_budget=2, total_budget=3)

    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade_importable"
    assert artifact["per_game_rows"] == []
    assert all(field in artifact for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"]


def test_scenario_5601_build_artifact_success_verdicts(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL verdict taxonomy."""

    monkeypatch.setattr(exp, "preconditions", lambda _root: {"ok": True})

    def raises(_game: str, *, explore_budget: int, total_budget: int) -> dict[str, object]:
        raise RuntimeError(f"failed {explore_budget} {total_budget}")

    monkeypatch.setattr(exp, "_measure_one_game", raises)
    artifact = exp.build_artifact(roster=("m0r0",), explore_budget=2, total_budget=3)
    assert (
        artifact["honest_verdict"]
        == "complete: object_history_salience_prototype_no_games_measured"
    )
    assert "error" in artifact["per_game_rows"][0]

    monkeypatch.setattr(
        exp,
        "_measure_one_game",
        lambda _game, *, explore_budget, total_budget: {
            "click_transitions": 0,
            "hashes_tracked_after": 0,
            "hashes_with_evidence_and_nonzero_change_rate_after": 0,
            "degeneracy_pairs_checked": 0,
            "degeneracy_pairs_differentiated_by_history": 0,
        },
    )
    artifact = exp.build_artifact(roster=("m0r0",), explore_budget=2, total_budget=3)
    assert (
        artifact["honest_verdict"]
        == "complete: object_history_salience_prototype_no_click_transitions_observed"
    )

    monkeypatch.setattr(
        exp,
        "_measure_one_game",
        lambda _game, *, explore_budget, total_budget: {
            "click_transitions": 3,
            "hashes_tracked_after": 1,
            "hashes_with_evidence_and_nonzero_change_rate_after": 0,
            "degeneracy_pairs_checked": 2,
            "degeneracy_pairs_differentiated_by_history": 0,
        },
    )
    artifact = exp.build_artifact(roster=("m0r0",), explore_budget=2, total_budget=3)
    assert (
        artifact["honest_verdict"]
        == "complete: object_history_salience_prototype_ran_but_no_hash_cleared_evidence_floor"
    )

    monkeypatch.setattr(
        exp,
        "_measure_one_game",
        lambda _game, *, explore_budget, total_budget: {
            "click_transitions": 3,
            "hashes_tracked_after": 2,
            "hashes_with_evidence_and_nonzero_change_rate_after": 1,
            "degeneracy_pairs_checked": 2,
            "degeneracy_pairs_differentiated_by_history": 1,
        },
    )
    artifact = exp.build_artifact(roster=("m0r0",), explore_budget=2, total_budget=3)
    assert artifact["honest_verdict"].startswith(
        "complete: object_history_salience_prototype_confirmed_1_hashes"
    )
    assert artifact["total_click_transitions_observed"] == 3
    assert artifact["total_hashes_tracked"] == 2
    assert artifact["total_hashes_with_evidence_and_nonzero_change_rate"] == 1
    assert artifact["adversarial_degeneracy_check"] == {
        "pairs_checked": 2,
        "pairs_differentiated": 1,
    }


def test_req_arc_fcp_5591_2_repository_artifact_is_a_real_measured_result() -> None:
    """REQ-ARC-FCP-5591-2: the checked-in real run measured ObjectHistorySaliencePrior
    against real click transitions from a real offline-arcade game (m0r0) -- not a
    fabricated or blocked stub. The gate the mechanism was built for is CONFIRMED: real
    objects with a genuine track record of changing the frame when clicked exist in real
    play (2 hashes cleared the evidence floor with a nonzero change rate)."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["honest_verdict"].startswith(
        "complete: object_history_salience_prototype_confirmed_"
    )
    assert result["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert result["solve_provenance"] == "development_proxy"
    assert result["total_click_transitions_observed"] > 0
    assert result["total_hashes_with_evidence_and_nonzero_change_rate"] > 0
    assert len(result["per_game_rows"]) >= 1
    assert result["duration_s"] > 5.0
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in result
