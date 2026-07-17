"""Tests for the ARC-AGI-3 level-up guarantee lint and its 2026-07-17 retirement.

Spec refs: CLAUDE.md "ARC Level-Up Attempt Guarantee" (RETIRED 2026-07-17),
"ARC-AGI-3 November-Submission Standing Floor" (RETIRED 2026-07-17),
"ARC-AGI-3 Generalization-Testing Floor" (2026-07-17, the redirect).

Covers: the original level-up-attempt detection (still exercised for the
pre-retirement code path), the registry-driven retirement check
(`_all_public_games_cleared`), and the new soft generalization-testing-floor
detector (`_is_generalization_attempt` / `count_generalization_attempts`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts import arc_levelup_guarantee_lint as lint


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _registry_payload(*, all_cleared: bool) -> dict[str, Any]:
    games = [{"game": name, "full_game_clear": all_cleared} for name in lint._GAME_NAMES]
    if not all_cleared:
        # Leave exactly one game short of clear, matching a realistic in-progress registry.
        games[0]["full_game_clear"] = False
    return {"games": games, "reproducible_total_levels": 183, "reproducible_total_games": 25}


def _roadmap(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"milestone": "2026.07.999", "tasks": tasks}


class TestAllPublicGamesCleared:
    def test_true_when_every_tracked_game_cleared(self, tmp_path: Path) -> None:
        registry = _write_yaml(tmp_path / "registry.yaml", _registry_payload(all_cleared=True))
        assert lint._all_public_games_cleared(registry) is True

    def test_false_when_one_game_not_cleared(self, tmp_path: Path) -> None:
        registry = _write_yaml(tmp_path / "registry.yaml", _registry_payload(all_cleared=False))
        assert lint._all_public_games_cleared(registry) is False

    def test_fails_open_on_missing_file(self, tmp_path: Path) -> None:
        assert lint._all_public_games_cleared(tmp_path / "does_not_exist.yaml") is False

    def test_fails_open_on_malformed_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "registry.yaml"
        bad.write_text("games: [this is not: valid: yaml: at all", encoding="utf-8")
        assert lint._all_public_games_cleared(bad) is False


class TestLevelUpAttemptDetection:
    """The original (pre-retirement) detector; still exercised via lint_roadmap's non-retired branch."""

    def test_bank_signal_required(self) -> None:
        assert lint._is_levelup_attempt(
            "Solve sk48: gate is offline_reproduced=true AND reproduced_levels>=1 on a first-contact game."
        )

    def test_generic_resolve_without_bank_signal_does_not_count(self) -> None:
        assert not lint._is_levelup_attempt(
            "Re-solve bp35 L3 via the generic operator for generalization validation; "
            "offline_reproduced=true but no new-level condition."
        )

    def test_no_offline_reproduced_does_not_count(self) -> None:
        assert not lint._is_levelup_attempt("Run a benchmark sweep across all games, reproduced_levels>=1.")


class TestGeneralizationAttemptDetection:
    def test_held_out_live_path_task_matches(self) -> None:
        assert lint._is_generalization_attempt(
            "Run a held-out generalization test: disable the sk48 GameAdapter and measure how far the "
            "live E3AgentPolicy path gets using only reusable arc_solver_kit primitives."
        )

    def test_arc_solver_kit_hardening_matches(self) -> None:
        assert lint._is_generalization_attempt(
            "Harden the ARC arc_solver_kit.py verifier-routed search primitive based on a gap found "
            "during held-out testing."
        )

    def test_unrelated_infra_task_does_not_match(self) -> None:
        assert not lint._is_generalization_attempt("Reconcile documentation and run infra hygiene checks.")

    def test_arc_mention_without_generalization_signal_does_not_match(self) -> None:
        # Bare game-name mention with no held-out/transfer/primitive-hardening language should not count.
        assert not lint._is_generalization_attempt("Update the bp35 registry entry's win_condition prose.")

    def test_generalization_signal_without_arc_scope_does_not_match(self) -> None:
        # Generic ML "generalization" language with no ARC/game-code mention is a different domain
        # entirely (e.g. Phase D off-ARC verifier work) and must not false-positive this floor.
        assert not lint._is_generalization_attempt(
            "Improve model generalization via better regularization on the held-out validation split."
        )

    def test_count_generalization_attempts_sums_matches(self, tmp_path: Path) -> None:
        roadmap = _write_yaml(
            tmp_path / "roadmap.yaml",
            _roadmap(
                [
                    {
                        "id": "exp1",
                        "prompt": "ARC-AGI-3 leave-one-game-out generalization test on the live path.",
                    },
                    {"id": "exp2", "prompt": "Unrelated infra hygiene, no ARC content."},
                    {"id": "exp3", "prompt": "Mine cross-game general_gotchas into arc_solver_kit.py."},
                ]
            ),
        )
        assert lint.count_generalization_attempts(roadmap) == 2

    def test_count_generalization_attempts_returns_zero_on_malformed_roadmap(self, tmp_path: Path) -> None:
        bad = tmp_path / "roadmap.yaml"
        bad.write_text("tasks: [not: valid: yaml", encoding="utf-8")
        assert lint.count_generalization_attempts(bad) == 0


class TestLintRoadmapRetirement:
    def test_retired_path_passes_regardless_of_task_content(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        registry = _write_yaml(tmp_path / "registry.yaml", _registry_payload(all_cleared=True))
        monkeypatch.setattr(lint, "_REGISTRY_PATH", registry)
        roadmap = _write_yaml(
            tmp_path / "roadmap.yaml",
            _roadmap([{"id": "exp1", "prompt": "No ARC content whatsoever, zero level-up attempts."}]),
        )
        # This is the exact scenario that used to hard-block: zero level-up attempts, min=1.
        assert lint.lint_roadmap(roadmap, 1) == 0

    def test_not_retired_path_still_enforces_minimum(self, tmp_path: Path, monkeypatch: Any) -> None:
        registry = _write_yaml(tmp_path / "registry.yaml", _registry_payload(all_cleared=False))
        monkeypatch.setattr(lint, "_REGISTRY_PATH", registry)
        roadmap = _write_yaml(
            tmp_path / "roadmap.yaml",
            _roadmap([{"id": "exp1", "prompt": "No ARC content whatsoever."}]),
        )
        assert lint.lint_roadmap(roadmap, 1) == 1

    def test_not_retired_path_passes_with_a_real_levelup_attempt(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        registry = _write_yaml(tmp_path / "registry.yaml", _registry_payload(all_cleared=False))
        monkeypatch.setattr(lint, "_REGISTRY_PATH", registry)
        roadmap = _write_yaml(
            tmp_path / "roadmap.yaml",
            _roadmap(
                [
                    {
                        "id": "exp1",
                        "prompt": "First-contact solve: offline_reproduced=true, reproduced_levels>=1.",
                    }
                ]
            ),
        )
        assert lint.lint_roadmap(roadmap, 1) == 0
