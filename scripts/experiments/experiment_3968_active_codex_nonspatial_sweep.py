"""Exp 3968: owed active-data codex sweep over the 6 non-spatial ARC-AGI-3 games.

Spec traces: REQ-PHASE4-008, SCENARIO-PHASE4-008-3968.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from arc3_m2_active_codex import codex_best_energy  # noqa: E402
from arc3_m2_active_data import _common_test, _keys, active_collect  # noqa: E402
from arc3_m2_world_model import _collect  # noqa: E402

DEFAULT_GAMES = ("r11l", "sc25", "lp85", "tn36", "dc22", "su15")
RESULT_NAME = "experiment_3968_active_codex_nonspatial_sweep.json"
TRUST_THRESHOLD = 0.15
VC33_BASELINE_ENERGY = 0.005
INFERENCE_SUBSTRATE = "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified"


def _codex_cli_available() -> bool:  # pragma: no cover - exercised by the real experiment preflight
    return subprocess.run(
        ["bash", "-lc", "command -v codex"],
        capture_output=True,
        check=False,
        text=True,
    ).returncode == 0


def _load_offline_arcade():  # pragma: no cover - exercised by the real experiment preflight
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arc = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    if not arc.get_environments():
        raise RuntimeError("offline arcade returned no environments")
    return arc


def _load_hidden_state_games() -> set[str]:
    path = REPO / "results" / "arc3_determinism_probe.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text("utf-8")).get("hidden_state_games", []))


def _codex_call_count(history) -> int:
    return sum(1 for row in history if "codex_s" in row)


def _markov_vs_hidden_split(per_game: list[dict], hidden_state_games: set[str]) -> dict:
    games = [row["game"] for row in per_game]
    return {
        "determinism_probe_markov": [game for game in games if game not in hidden_state_games],
        "determinism_probe_hidden_state": [game for game in games if game in hidden_state_games],
        "energy_trustworthy_low": [row["game"] for row in per_game if row["trustworthy"]],
        "energy_high_or_missing": [row["game"] for row in per_game if not row["trustworthy"]],
        "note": (
            "The determinism split comes from results/arc3_determinism_probe.json. "
            "The energy split certifies held-out prediction trustworthiness, not whether the "
            "mechanic has been correctly named."
        ),
    }


def _empty_artifact(games: list[str], seed: int, iters: int, started: float, verdict: str) -> dict:
    return {
        "experiment": "experiment_3968_active_codex_nonspatial_sweep",
        "title": "active_data_codex_nonspatial_sweep_reattempt",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "n_games": len(games),
        "iters": min(int(iters), 3),
        "trust_threshold": TRUST_THRESHOLD,
        "vc33_baseline_energy": VC33_BASELINE_ENERGY,
        "n_trustworthy_at_0.15": 0,
        "per_game_best_energy": {},
        "total_codex_calls": 0,
        "total_codex_seconds": 0.0,
        "markov_vs_hidden_split": {
            "determinism_probe_markov": [],
            "determinism_probe_hidden_state": [],
            "energy_trustworthy_low": [],
            "energy_high_or_missing": [],
            "note": "Blocked before per-game evaluation.",
        },
        "random_seed": seed,
        "duration_s": round(time.time() - started, 1),
        "per_game": [],
        "caveat": (
            "Consistency energy certifies that the synthesized program predicts transitions on "
            "held-out data; it does not certify that the mechanic interpretation is correctly named."
        ),
    }


def _write_artifact(artifact: dict) -> None:
    out_path = REPO / "results" / RESULT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")


def run(
    games: list[str] | None = None,
    train_budget: int = 900,
    test_budget: int = 1400,
    episodes: int = 32,
    iters: int = 3,
    seed: int = 0,
    write: bool = True,
    _arc_client=None,
    _codex_available: bool | None = None,
) -> dict:
    games = list(games or DEFAULT_GAMES)
    iters = min(int(iters), 3)
    started = time.time()

    has_codex = _codex_cli_available() if _codex_available is None else bool(_codex_available)
    if not has_codex:
        artifact = _empty_artifact(games, seed, iters, started, "blocked_codex_unavailable")
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        arc = _arc_client if _arc_client is not None else _load_offline_arcade()
        envs = arc.get_environments()
        if not envs:
            raise RuntimeError("offline arcade returned no environments")
    except Exception:
        artifact = _empty_artifact(games, seed, iters, started, "blocked_arc_offline_env_unavailable")
        if write:
            _write_artifact(artifact)
        return artifact

    from arcengine.enums import GameAction, GameState

    rng = random.Random(seed)
    all_ids = sorted(getattr(env, "game_id", None) for env in envs if getattr(env, "game_id", None))
    selected = [game_id for game_id in all_ids if game_id.split("-")[0] in set(games)]
    hidden_state_games = _load_hidden_state_games()

    per_game = []
    total_codex_seconds = 0.0
    total_codex_calls = 0
    for game_id in selected:
        short = game_id.split("-")[0]
        test_all = _collect(arc, game_id, test_budget, episodes, rng, GameAction, GameState)
        active = active_collect(arc, game_id, train_budget, episodes, rng, GameAction, GameState)
        held_out = _common_test(test_all, _keys(active))
        best_energy, history, codex_seconds = codex_best_energy(active, held_out, iters, rng)
        trustworthy = best_energy is not None and best_energy <= TRUST_THRESHOLD
        total_codex_seconds += codex_seconds
        total_codex_calls += _codex_call_count(history)

        row = {
            "game": short,
            "best_energy": best_energy,
            "trustworthy": trustworthy,
            "diff_from_vc33_0.005": (
                round(best_energy - VC33_BASELINE_ENERGY, 4) if best_energy is not None else None
            ),
            "determinism_probe_hidden_state": short in hidden_state_games,
            "n_active": len(active),
            "n_test": len(held_out),
            "codex_calls": _codex_call_count(history),
            "codex_seconds": codex_seconds,
            "history": history,
        }
        per_game.append(row)
        print(
            f"  {short:6s} best_energy={best_energy} trustworthy={trustworthy} "
            f"diff_vs_vc33={row['diff_from_vc33_0.005']} hidden={short in hidden_state_games}",
            flush=True,
        )

    n_trustworthy = sum(1 for row in per_game if row["trustworthy"])
    verdict = f"complete: exp3968_active_codex_nonspatial_sweep_trustworthy_{n_trustworthy}of{len(per_game)}"
    artifact = {
        "experiment": "experiment_3968_active_codex_nonspatial_sweep",
        "title": "active_data_codex_nonspatial_sweep_reattempt",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "games_evaluated": [row["game"] for row in per_game],
        "n_games": len(per_game),
        "train_budget": train_budget,
        "test_budget": test_budget,
        "episodes": episodes,
        "iters": iters,
        "trust_threshold": TRUST_THRESHOLD,
        "vc33_baseline_energy": VC33_BASELINE_ENERGY,
        "n_trustworthy_at_0.15": n_trustworthy,
        "per_game_best_energy": {row["game"]: row["best_energy"] for row in per_game},
        "total_codex_calls": total_codex_calls,
        "total_codex_seconds": round(total_codex_seconds, 1),
        "markov_vs_hidden_split": _markov_vs_hidden_split(per_game, hidden_state_games),
        "random_seed": seed,
        "duration_s": round(time.time() - started, 1),
        "per_game": per_game,
        "caveat": (
            "Consistency energy certifies that the synthesized program predicts transitions on "
            "held-out data; it does not certify that the mechanic interpretation is correctly named. "
            "Report energy as the trustworthiness signal."
        ),
        "source_artifacts": [
            "results/arc3_m2_active_codex.json",
            "results/arc3_win_condition_survey.json",
            "results/arc3_determinism_probe.json",
        ],
        "submitted_to_leaderboard": False,
        "no_gpu_used": True,
    }
    if write:
        _write_artifact(artifact)
    print(f"\n-> {verdict}")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES))
    parser.add_argument("--train_budget", type=int, default=900)
    parser.add_argument("--test_budget", type=int, default=1400)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        games=[game.strip() for game in args.games.split(",") if game.strip()],
        train_budget=args.train_budget,
        test_budget=args.test_budget,
        episodes=args.episodes,
        iters=args.iters,
        seed=args.seed,
    )
