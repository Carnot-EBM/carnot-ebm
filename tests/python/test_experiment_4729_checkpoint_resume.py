"""Tests for Exp 4729 checkpoint/resume + soft-budget graceful stop.

Spec refs: REQ-CAPSTONE-4729, SCENARIO-CAPSTONE-4729.

These cover ONLY the persistence machinery added so the held-out first-win readiness SCORE survives
codex's 4800s hard wall-clock cap: incremental per-game checkpointing, resume-skips-done-games, and the
soft-budget graceful partial stop. They use cheap fakes for the exp4605 per-variant runner (no real LLM
/ GPU / ARC arcade) but exercise the REAL exp4605 aggregation (variant_specs / measurement_from_attempts
/ build_artifact) so the SCORE path under test is the production one.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4605_live_integration_scored_agent as exp4605
from carnot import experiment_4729_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]


def _fake_runner_factory(solved_games: set[str]):
    """Build a deterministic, cheap stand-in for exp4605.default_variant_runner_factory.

    Each returned runner emits an attempt row with the SAME shape exp4605.run_variant_attempt produces
    (variant_signature, attempted, solved/first_win, actions...) so the real measurement_from_attempts
    can aggregate it -- but it does ZERO ARC arcade work. A game is "solved" only if it is in
    solved_games, letting a test fix the first-win count deterministically.
    """

    def factory(mode: str):
        def runner(game: str, spec: dict[str, Any], budget: int) -> JsonDict:
            solved = mode == "integrated" and game in solved_games
            return {
                "game": game,
                "variant_signature": spec["variant_signature"],
                "variant": int(spec["variant"]),
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": solved,
                "first_win": solved,
                "reached_level": 1 if solved else 0,
                "actions": 3 if solved else budget,
                "actions_to_first_levelup": 3 if solved else None,
                "solution_labels": [],
                "reproduction_gate": {"reproduced": solved, "reached_level": 1 if solved else 0},
                "blocked_reason": "",
                "policy_mode": mode,
                "depth_reached": 1 if solved else 0,
                "actions_to_second_levelup": None,
            }

        return runner

    return factory


def _patch_cheap_runner(monkeypatch: pytest.MonkeyPatch, solved_games: set[str], games: list[str]):
    """Patch the expensive bits of exp4605 with cheap deterministic fakes, keeping the SCORE path real.

    Also stubs exp4605._submitted_* so build_artifact does NOT import the heavy arc_competition_agent
    JAX/torch stack (which trips the 500MB per-test memory watchdog). These snapshot helpers only fill
    METADATA fields (submitted_agent_config etc.), never the SCORE math under test.
    """

    monkeypatch.setattr(exp4605, "default_variant_runner_factory", _fake_runner_factory(solved_games))
    monkeypatch.setattr(exp4605, "_public_games", lambda _root: list(games))
    # Speed: 1 bootstrap leaves the CI math deterministic without 1000 resamples in a unit test.
    monkeypatch.setattr(exp4605, "DEFAULT_BOOTSTRAPS", 1)
    monkeypatch.setattr(exp4605, "_submitted_config_snapshot", lambda: {"policy": "stub"})
    monkeypatch.setattr(exp4605, "_submitted_value_weight", lambda: 0.0)
    monkeypatch.setattr(exp4605, "_submitted_target_levels", lambda: 3)


def test_checkpoint_resume_skips_done_games_and_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4729: a resumed run loads the partial, SKIPS done games, and merges the rest."""

    games = ["gameA", "gameB", "gameC"]
    _patch_cheap_runner(monkeypatch, solved_games={"gameA"}, games=games)

    # Seed a partial ledger as if gameA already ran in a prior (capped) run, with a SENTINEL row whose
    # variant_signature is unmistakable -- so we can prove the resumed run does NOT recompute gameA.
    sentinel = {
        "game": "gameA",
        "variant_signature": "gameA~SENTINEL",
        "variant": 1,
        "kind": "color",
        "reflect": None,
        "attempted": True,
        "solved": True,
        "first_win": True,
        "reached_level": 1,
        "actions": 3,
        "actions_to_first_levelup": 3,
        "solution_labels": [],
        "reproduction_gate": {"reproduced": True, "reached_level": 1},
        "blocked_reason": "",
        "policy_mode": "integrated",
        "depth_reached": 1,
        "actions_to_second_levelup": None,
    }
    mod._write_partial(
        tmp_path,
        {"games": {"gameA": {"integrated_attempts": [sentinel], "bare_attempts": [sentinel]}}},
    )

    # Track which games the runner is actually invoked for (proves the skip).
    ran_games: list[str] = []
    base_factory = _fake_runner_factory({"gameA"})

    def tracking_factory(mode: str):
        inner = base_factory(mode)

        def runner(game: str, spec: dict[str, Any], budget: int) -> JsonDict:
            ran_games.append(game)
            return inner(game, spec, budget)

        return runner

    monkeypatch.setattr(exp4605, "default_variant_runner_factory", tracking_factory)

    # now() far below the soft budget -> never trips the soft-budget stop -> runs to COMPLETION.
    proxy = mod.run_held_out_proxy_checkpointed(
        tmp_path,
        {"passed": True},
        now=lambda: 0.0,
        soft_budget_s=10_000.0,
    )

    # gameA must NOT have been re-run; gameB + gameC must have been.
    assert "gameA" not in ran_games
    assert set(ran_games) == {"gameB", "gameC"}

    # The merged proxy must carry the SENTINEL (loaded) row -> proof the loaded results were merged,
    # not discarded. Each of 3 games x 4 variants = 12 fresh rows + gameA contributed the 1 sentinel
    # (gameA was loaded with a single sentinel row, gameB/gameC computed 4 variants each = 8 rows).
    sigs = [row["variant_signature"] for row in proxy["integrated_measurement"]["variant_attempts"]]
    assert "gameA~SENTINEL" in sigs
    # gameA contributed the loaded sentinel only; gameB + gameC contributed 4 variants each.
    assert sum(1 for s in sigs if s.startswith("gameB")) == 4
    assert sum(1 for s in sigs if s.startswith("gameC")) == 4

    # COMPLETION clears the resume ledger.
    assert not (tmp_path / mod.PARTIAL_RESULT_RELATIVE_PATH).exists()


def test_checkpoint_written_after_each_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4729: the partial ledger is flushed after EACH per-game unit, not only at the end."""

    games = ["g1", "g2", "g3"]
    _patch_cheap_runner(monkeypatch, solved_games=set(), games=games)

    # Snapshot how many games are in the ledger at the moment each game's runner is invoked. The first
    # game's runner sees 0 flushed games; by the time g3 runs, g1+g2 must already be on disk.
    flushed_counts: list[int] = []
    base_factory = _fake_runner_factory(set())

    def snapshotting_factory(mode: str):
        inner = base_factory(mode)

        def runner(game: str, spec: dict[str, Any], budget: int) -> JsonDict:
            if mode == "integrated":
                ledger = mod.load_partial(tmp_path)
                flushed_counts.append(len(ledger.get("games", {})))
            return inner(game, spec, budget)

        return runner

    monkeypatch.setattr(exp4605, "default_variant_runner_factory", snapshotting_factory)

    mod.run_held_out_proxy_checkpointed(
        tmp_path, {"passed": True}, now=lambda: 0.0, soft_budget_s=10_000.0
    )

    # 3 games x 4 variants = 12 integrated invocations. The ledger count must be MONOTONE non-decreasing
    # and reach 2 (g1, g2 flushed) before the last game's variants run -> proves per-game incremental
    # flush, not a single end-of-run write.
    assert flushed_counts == sorted(flushed_counts)
    assert max(flushed_counts) >= 2


def test_soft_budget_emits_partial_artifact_and_keeps_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-4729: a soft-budget stop emits partial: true + keeps the resume ledger."""

    games = ["gX", "gY", "gZ"]
    _patch_cheap_runner(monkeypatch, solved_games=set(), games=games)

    # The runner anchors started=now() on its FIRST call, then checks elapsed = now()-started per game.
    # Clock = 0, 1000, 2000, ...: started=0; gX check elapsed=1000<1500 -> run gX; gY check elapsed=
    # 2000>=1500 -> graceful stop. So done={gX}, remaining={gY,gZ}.
    clock = itertools.count(start=0, step=1000)

    def proxy_runner(root: Path, parity_test: dict[str, Any]) -> JsonDict:
        return mod.run_held_out_proxy_checkpointed(
            root,
            parity_test,
            now=lambda: next(clock),
            soft_budget_s=1500.0,
        )

    artifact = mod.run(
        root=tmp_path,
        preconditions_checker=lambda _root: {"ok": True, "offline_arcade": True},
        parity_check=lambda _root: {"passed": True},
        proxy_runner=proxy_runner,
        replay_floor_loader=lambda _root: {
            "package_path": "results/x.json",
            "package_exists": True,
            "replay_package_floor_reproduced": True,
            "live_submittable_level_count": 60,
            "note": mod.REPLAY_FLOOR_NOTE,
        },
        lever_input_loader=lambda _root: {"a1": {}, "a2": {}},
        now=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )

    # GRACEFUL partial: terminal-prefix verdict, partial: true, NOT ready, schema-clean, exit-0-shaped.
    assert artifact["partial"] is True
    assert artifact["honest_verdict"].startswith(mod.TERMINAL_PREFIXES)
    assert "soft_budget_stop_partial" in artifact["honest_verdict"]
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["completed_variants"]  # gX's 4 variants completed
    assert artifact["remaining_variants"]  # gY + gZ still to do
    assert set(artifact["completed_games"]) == {"gX"}
    assert set(artifact["remaining_games"]) == {"gY", "gZ"}
    assert mod.artifact_schema_errors(artifact) == []

    # The resume ledger is INTENTIONALLY kept so the next run finishes the rest.
    ledger = mod.load_partial(tmp_path)
    assert set(ledger.get("games", {})) == {"gX"}

    # The final artifact is on disk too (codex sees a successful run, well under the hard cap).
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact


def test_resolve_soft_budget_env_override_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4729: the soft budget honors the env override but never disables itself."""

    monkeypatch.delenv(mod.SOFT_BUDGET_ENV, raising=False)
    assert mod.resolve_soft_budget_s() == mod.DEFAULT_SOFT_BUDGET_S

    monkeypatch.setenv(mod.SOFT_BUDGET_ENV, "1234.5")
    assert mod.resolve_soft_budget_s() == 1234.5

    # A non-positive or garbage override must FALL BACK to the default (never disable the graceful stop).
    monkeypatch.setenv(mod.SOFT_BUDGET_ENV, "0")
    assert mod.resolve_soft_budget_s() == mod.DEFAULT_SOFT_BUDGET_S
    monkeypatch.setenv(mod.SOFT_BUDGET_ENV, "not-a-number")
    assert mod.resolve_soft_budget_s() == mod.DEFAULT_SOFT_BUDGET_S


def test_completed_full_run_equals_resumed_run_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4729: resuming across a budget stop yields the SAME SCORE as one uncapped run."""

    games = ["a", "b", "c", "d"]
    solved = {"a", "c"}

    # 1) One uncapped run (huge budget) -> the reference proxy artifact.
    _patch_cheap_runner(monkeypatch, solved_games=solved, games=games)
    full = mod.run_held_out_proxy_checkpointed(
        tmp_path / "full", {"passed": True}, now=lambda: 0.0, soft_budget_s=10_000.0
    )

    # 2) A capped run that stops partway (elapsed exceeds the 1500s budget after the first game), then a
    # resume with budget headroom that finishes the rest -> the resumed proxy.
    resumed_root = tmp_path / "resumed"
    clock = itertools.count(start=0, step=800)
    with pytest.raises(mod._BudgetExceeded):
        mod.run_held_out_proxy_checkpointed(
            resumed_root, {"passed": True}, now=lambda: next(clock), soft_budget_s=1500.0
        )
    # The ledger holds the partial progress; a fresh run with budget headroom finishes the rest.
    resumed = mod.run_held_out_proxy_checkpointed(
        resumed_root, {"passed": True}, now=lambda: 0.0, soft_budget_s=10_000.0
    )

    # The SCORE-bearing fields must be identical: resuming changed only WHEN rows were persisted.
    for field in (
        "first_win_rate_integrated",
        "first_win_rate_bare",
        "first_win_delta",
        "first_win_ci",
        "solve_rate_integrated",
        "honest_verdict",
    ):
        assert resumed[field] == full[field], field
    assert (
        resumed["integrated_measurement"]["variant_attempts_count"]
        == full["integrated_measurement"]["variant_attempts_count"]
        == len(games) * len(mod.HELD_OUT_VARIANT_IDS)
    )


def test_soft_budget_is_elapsed_not_absolute_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4729: the budget is ELAPSED time, so a large absolute clock does NOT trip it.

    Regression guard: a naive now()>=budget would compare a ~1.7e9 Unix timestamp against 4200s and
    stop before game 1 every time (producing a 0-game partial), defeating the whole fix. With a clock
    anchored far in the future but ADVANCING only a little, all games must complete.
    """

    games = ["p", "q", "r"]
    _patch_cheap_runner(monkeypatch, solved_games={"p"}, games=games)

    # Absolute clock near a real Unix timestamp, advancing only +1s per read -> elapsed stays ~seconds,
    # far under the default 4200s budget, so NOTHING should trip even though now() >> budget.
    clock = itertools.count(start=1_782_000_000, step=1)
    proxy = mod.run_held_out_proxy_checkpointed(
        tmp_path, {"passed": True}, now=lambda: float(next(clock))
    )

    assert proxy["integrated_measurement"]["variant_attempts_count"] == len(games) * len(
        mod.HELD_OUT_VARIANT_IDS
    )
    # Completed -> ledger cleared.
    assert not (tmp_path / mod.PARTIAL_RESULT_RELATIVE_PATH).exists()


def test_run_resumes_partial_ledger_to_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4729: run() with a pre-existing partial ledger finishes and writes partial: false.

    End-to-end through run() (not just the proxy): enough games to clear B>=100, a partial ledger left
    by a prior capped run, and the resumed run completes -> a final partial: false artifact identical in
    SCORE to a one-shot run, with the ledger cleared.
    """

    games = [f"g{i:02d}" for i in range(30)]  # 30 x 4 = 120 attempts > 100 (clears B>=100)
    solved = {"g00", "g05"}
    _patch_cheap_runner(monkeypatch, solved_games=solved, games=games)

    # Reference one-shot run (separate root, never capped).
    ref_root = tmp_path / "ref"
    ref = mod.run(
        root=ref_root,
        preconditions_checker=lambda _r: {"ok": True, "offline_arcade": True},
        parity_check=lambda _r: {"passed": True},
        replay_floor_loader=lambda _r: {
            "package_path": "results/x.json",
            "package_exists": True,
            "replay_package_floor_reproduced": True,
            "live_submittable_level_count": 60,
            "note": mod.REPLAY_FLOOR_NOTE,
        },
        lever_input_loader=lambda _r: {"a1": {}, "a2": {}},
        now=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )
    assert ref["partial"] is False
    assert ref["held_out_variant_attempts"] == 120

    # Resumed run (separate root): pre-seed a partial ledger with the FIRST 10 games already computed,
    # then run() to completion. The default checkpointed proxy_runner resumes and finishes the rest.
    resumed_root = tmp_path / "resumed"
    seed_done: dict[str, Any] = {}
    integrated_factory = exp4605.default_variant_runner_factory("integrated")
    bare_factory = exp4605.default_variant_runner_factory("bare")
    for game in sorted(games)[:10]:
        specs = exp4605.variant_specs([game], mod.HELD_OUT_VARIANT_IDS)
        seed_done[game] = {
            "integrated_attempts": [integrated_factory(game, s, 1) for s in specs],
            "bare_attempts": [bare_factory(game, s, 1) for s in specs],
        }
    mod._write_partial(resumed_root, {"games": seed_done})

    resumed = mod.run(
        root=resumed_root,
        preconditions_checker=lambda _r: {"ok": True, "offline_arcade": True},
        parity_check=lambda _r: {"passed": True},
        replay_floor_loader=lambda _r: {
            "package_path": "results/x.json",
            "package_exists": True,
            "replay_package_floor_reproduced": True,
            "live_submittable_level_count": 60,
            "note": mod.REPLAY_FLOOR_NOTE,
        },
        lever_input_loader=lambda _r: {"a1": {}, "a2": {}},
        now=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )

    assert resumed["partial"] is False
    assert resumed["held_out_variant_attempts"] == 120
    assert not (resumed_root / mod.PARTIAL_RESULT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(resumed) == []
    # Resuming changed only persistence timing -> the SCORE fields match the one-shot reference.
    for field in (
        "first_win_rate_integrated",
        "first_win_ci_lower",
        "multi_level_deepen_rate_integrated",
        "held_out_variant_attempts",
        "honest_verdict",
    ):
        assert resumed[field] == ref[field], field


def test_write_partial_is_atomic_and_leaves_no_temp(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4729: the ledger flush is atomic (temp + os.replace) and leaves a readable file."""

    ledger = {"games": {"g1": {"integrated_attempts": [{"x": 1}], "bare_attempts": []}}}
    mod._write_partial(tmp_path, ledger)

    path = tmp_path / mod.PARTIAL_RESULT_RELATIVE_PATH
    assert path.exists()
    # The atomic write must NOT leave its .tmp sidecar behind.
    assert not path.with_name(path.name + ".tmp").exists()
    # The persisted ledger round-trips through load_partial unchanged.
    assert mod.load_partial(tmp_path) == ledger


def test_load_partial_is_defensive_against_corruption(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4729: a corrupt/garbage ledger NEVER crashes resume -> treated as no-progress rows."""

    path = tmp_path / mod.PARTIAL_RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Truncated JSON (the codex-mid-flush failure mode) -> no progress, no crash.
    path.write_text('{"games": {"g1": {"integrated_atte', encoding="utf-8")
    assert mod.load_partial(tmp_path) == {"games": {}}

    # 'games' not a mapping -> no progress, no crash.
    path.write_text('{"games": [1, 2, 3]}', encoding="utf-8")
    assert mod.load_partial(tmp_path) == {"games": {}}

    # Scalar (non-Mapping) attempt rows -> dropped, NOT a TypeError from dict(scalar).
    path.write_text(
        '{"games": {"g1": {"integrated_attempts": [42, {"ok": 1}], "bare_attempts": ["bad"]}}}',
        encoding="utf-8",
    )
    loaded = mod.load_partial(tmp_path)
    assert loaded == {"games": {"g1": {"integrated_attempts": [{"ok": 1}], "bare_attempts": []}}}

    # Missing file -> empty skeleton.
    path.unlink()
    assert mod.load_partial(tmp_path) == {"games": {}}
