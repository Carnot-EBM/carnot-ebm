"""Exp 3979: execution-guided world-model synthesis for non-spatial ARC-AGI-3 games.

Spec refs: REQ-PHASE4-017, SCENARIO-PHASE4-017.
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

from carnot.agentic.arc_execution_guided_world_model import induce_execution_guided  # noqa: E402
from arc3_m2_active_data import _common_test, _keys, active_collect  # noqa: E402
from arc3_m2_codex_synth import safe_predict_from_code  # noqa: E402
from arc3_m2_world_model import _collect  # noqa: E402

DEFAULT_GAMES = ("r11l", "sc25", "lp85", "tn36", "dc22", "su15")
RESULT_NAME = "experiment_3979_world_model_gen_execution_guided.json"
TRUST_THRESHOLD = 0.15
POSITIVE_CONTROL_THRESHOLD = 0.05
VC33_BASELINE_ENERGY = 0.005
INFERENCE_SUBSTRATE = (
    "offline_arc_agi3_execution_guided_program_synthesis_exact_replay_consistency_verified"
)


def _codex_cli_available() -> bool:  # pragma: no cover - only used when fresh Codex is explicitly requested
    return subprocess.run(["bash", "-lc", "command -v codex"], capture_output=True, text=True).returncode == 0


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


def _write_artifact(artifact: dict) -> None:
    out_path = REPO / "results" / RESULT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")


def _select_game_ids(arc, games: list[str]) -> dict[str, str]:
    requested = set(games) | {"vc33"}
    envs = arc.get_environments()
    out = {}
    for env in envs:
        game_id = getattr(env, "game_id", None)
        if game_id and game_id.split("-")[0] in requested:
            out[game_id.split("-")[0]] = game_id
    return out


def _load_hidden_state_games() -> set[str]:
    path = REPO / "results" / "arc3_determinism_probe.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text("utf-8")).get("hidden_state_games", []))


def _load_exp3968_baseline() -> tuple[int, dict[str, float]]:
    path = REPO / "results" / "experiment_3968_active_codex_nonspatial_sweep.json"
    if not path.exists():
        return 0, {}
    payload = json.loads(path.read_text("utf-8"))
    return int(payload.get("n_trustworthy_at_0.15", 0) or 0), dict(payload.get("per_game_best_energy", {}))


def _load_gap4_gate() -> dict:
    path = REPO / "results" / "experiment_3975_gap4_execution_verifier_build.json"
    if not path.exists():
        return {"source_artifact": str(path), "exists": False}
    payload = json.loads(path.read_text("utf-8"))
    return {
        "source_artifact": "results/experiment_3975_gap4_execution_verifier_build.json",
        "exists": True,
        "prior_positive_control_passed": bool(payload.get("positive_control_passed")),
        "prior_honest_verdict": payload.get("honest_verdict"),
        "gate_applied_here": "candidate programs must exactly replay accepted observed transitions",
    }


def _load_cached_vc33_predictor():
    path = REPO / "results" / "arc3_vc33_world_model_program.py"
    if not path.exists():
        return None
    fn = safe_predict_from_code(path.read_text("utf-8"))
    if fn is None:
        return None
    return ("cached_verified_vc33_program", fn)


def _collect_train_and_heldout(arc, game_id, train_budget, test_budget, episodes, rng, GameAction, GameState):
    train = active_collect(arc, game_id, train_budget, episodes, rng, GameAction, GameState)
    test_all = _collect(arc, game_id, test_budget, episodes, rng, GameAction, GameState)
    held_out = _common_test(test_all, _keys(train))
    return train, held_out


def _markov_vs_hidden_split(per_game: list[dict], hidden_state_games: set[str]) -> dict:
    games = [row["game"] for row in per_game]
    return {
        "determinism_probe_markov": [game for game in games if game not in hidden_state_games],
        "determinism_probe_hidden_state": [game for game in games if game in hidden_state_games],
        "energy_trustworthy_low": [row["game"] for row in per_game if row["trustworthy"]],
        "energy_high_or_missing": [row["game"] for row in per_game if not row["trustworthy"]],
        "note": (
            "The determinism split comes from results/arc3_determinism_probe.json. "
            "Consistency energy is reported as prediction trustworthiness, not understanding."
        ),
    }


def _empty_artifact(games: list[str], seed: int, iters: int, started: float, verdict: str) -> dict:
    return {
        "experiment": "experiment_3979_world_model_gen_execution_guided",
        "title": "execution_guided_world_model_synthesis_nonspatial_sweep",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "games_evaluated": [],
        "n_games": 0,
        "iters": int(iters),
        "trust_threshold": TRUST_THRESHOLD,
        "positive_control_threshold": POSITIVE_CONTROL_THRESHOLD,
        "positive_control_passed": False,
        "positive_control": {},
        "n_trustworthy_at_0.15": 0,
        "per_game_best_energy": {},
        "beats_exp3968": False,
        "exp3968_n_trustworthy_at_0.15": 0,
        "vc33_baseline_energy": VC33_BASELINE_ENERGY,
        "total_synthesis_calls": 0,
        "total_synthesis_seconds": 0.0,
        "random_seed": seed,
        "duration_s": round(time.time() - started, 1),
        "per_game": [],
        "markov_vs_hidden_split": {
            "determinism_probe_markov": [],
            "determinism_probe_hidden_state": [],
            "energy_trustworthy_low": [],
            "energy_high_or_missing": [],
            "note": "Blocked before per-game evaluation.",
        },
        "gap4_executed_consistency_gate": _load_gap4_gate(),
        "caveat": (
            "Consistency energy certifies prediction trustworthiness on held-out transitions; "
            "it does not certify mechanistic understanding or interpretation."
        ),
        "submitted_to_leaderboard": False,
        "no_gpu_used": True,
    }


def run(
    games: list[str] | None = None,
    train_budget: int = 900,
    test_budget: int = 1400,
    episodes: int = 32,
    iters: int = 3,
    seed: int = 0,
    write: bool = True,
    use_fresh_codex: bool = False,
    _arc_client=None,
) -> dict:
    games = list(games or DEFAULT_GAMES)
    started = time.time()

    if use_fresh_codex and not _codex_cli_available():
        artifact = _empty_artifact(games, seed, iters, started, "blocked_codex_unavailable")
        if write:
            _write_artifact(artifact)
        return artifact

    try:
        arc = _arc_client if _arc_client is not None else _load_offline_arcade()
        selected_ids = _select_game_ids(arc, games)
        if "vc33" not in selected_ids:
            raise RuntimeError("vc33 positive-control environment unavailable")
        if not any(game in selected_ids for game in games):
            raise RuntimeError("no requested offline game environments available")
    except Exception:
        artifact = _empty_artifact(games, seed, iters, started, "blocked_arc_offline_env_unavailable")
        if write:
            _write_artifact(artifact)
        return artifact

    from arcengine.enums import GameAction, GameState

    rng = random.Random(seed)
    total_synthesis_calls = 0
    total_synthesis_seconds = 0.0

    vc33_seed = _load_cached_vc33_predictor()
    vc33_extra = [vc33_seed] if vc33_seed is not None else []
    vc33_train, vc33_held = _collect_train_and_heldout(
        arc, selected_ids["vc33"], train_budget, test_budget, episodes, rng, GameAction, GameState
    )
    positive = induce_execution_guided(
        "vc33",
        vc33_train,
        vc33_held,
        max_synthesis_iters=iters,
        extra_predictors=vc33_extra,
    )
    total_synthesis_calls += positive["total_synthesis_calls"]
    total_synthesis_seconds += positive["total_synthesis_seconds"]
    positive_energy = positive["best_energy"]
    positive_passed = positive_energy is not None and positive_energy <= POSITIVE_CONTROL_THRESHOLD
    if not positive_passed:
        artifact = _empty_artifact(games, seed, iters, started, "blocked_positive_control_failed")
        artifact["positive_control"] = {
            "game": "vc33",
            "best_energy": positive_energy,
            "threshold": POSITIVE_CONTROL_THRESHOLD,
            "best_program": positive["best_program"],
            "history": positive["history"],
            "accepted_train_count": positive["accepted_train_count"],
            "rejected_conflict_count": positive["rejected_conflict_count"],
            "cached_vc33_seed_used": vc33_seed is not None,
        }
        artifact["total_synthesis_calls"] = total_synthesis_calls
        artifact["total_synthesis_seconds"] = round(total_synthesis_seconds, 4)
        artifact["duration_s"] = round(time.time() - started, 1)
        if write:
            _write_artifact(artifact)
        return artifact

    hidden_state_games = _load_hidden_state_games()
    exp3968_n, exp3968_energy = _load_exp3968_baseline()

    per_game = []
    for short in games:
        game_id = selected_ids.get(short)
        if game_id is None:
            continue
        train, held_out = _collect_train_and_heldout(
            arc, game_id, train_budget, test_budget, episodes, rng, GameAction, GameState
        )
        result = induce_execution_guided(
            short,
            train,
            held_out,
            max_synthesis_iters=iters,
        )
        total_synthesis_calls += result["total_synthesis_calls"]
        total_synthesis_seconds += result["total_synthesis_seconds"]
        best_energy = result["best_energy"]
        trustworthy = best_energy is not None and best_energy <= TRUST_THRESHOLD
        baseline_energy = exp3968_energy.get(short)
        row = {
            "game": short,
            "best_energy": best_energy,
            "trustworthy": trustworthy,
            "best_program": result["best_program"],
            "exp3968_best_energy": baseline_energy,
            "improvement_over_exp3968": (
                round(baseline_energy - best_energy, 4)
                if baseline_energy is not None and best_energy is not None
                else None
            ),
            "diff_from_vc33_0.005": (
                round(best_energy - VC33_BASELINE_ENERGY, 4) if best_energy is not None else None
            ),
            "determinism_probe_hidden_state": short in hidden_state_games,
            "n_active": len(train),
            "n_test": len(held_out),
            "accepted_train_count": result["accepted_train_count"],
            "rejected_conflict_count": result["rejected_conflict_count"],
            "synthesis_calls": result["total_synthesis_calls"],
            "synthesis_seconds": result["total_synthesis_seconds"],
            "history": result["history"],
        }
        per_game.append(row)
        print(
            f"  {short:6s} best_energy={best_energy} trustworthy={trustworthy} "
            f"exp3968={baseline_energy} hidden={short in hidden_state_games}",
            flush=True,
        )

    n_trustworthy = sum(1 for row in per_game if row["trustworthy"])
    beats_exp3968 = n_trustworthy > exp3968_n
    prefix = "success" if beats_exp3968 else "complete"
    verdict = f"{prefix}: exec_guided_trustworthy_{n_trustworthy}of{len(per_game)}"
    artifact = {
        "experiment": "experiment_3979_world_model_gen_execution_guided",
        "title": "execution_guided_world_model_synthesis_nonspatial_sweep",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "games_evaluated": [row["game"] for row in per_game],
        "n_games": len(per_game),
        "train_budget": train_budget,
        "test_budget": test_budget,
        "episodes": episodes,
        "iters": int(iters),
        "trust_threshold": TRUST_THRESHOLD,
        "positive_control_threshold": POSITIVE_CONTROL_THRESHOLD,
        "positive_control_passed": True,
        "positive_control": {
            "game": "vc33",
            "best_energy": positive_energy,
            "threshold": POSITIVE_CONTROL_THRESHOLD,
            "best_program": positive["best_program"],
            "history": positive["history"],
            "accepted_train_count": positive["accepted_train_count"],
            "rejected_conflict_count": positive["rejected_conflict_count"],
            "cached_vc33_seed_used": vc33_seed is not None,
        },
        "n_trustworthy_at_0.15": n_trustworthy,
        "per_game_best_energy": {row["game"]: row["best_energy"] for row in per_game},
        "beats_exp3968": beats_exp3968,
        "exp3968_n_trustworthy_at_0.15": exp3968_n,
        "vc33_baseline_energy": VC33_BASELINE_ENERGY,
        "total_synthesis_calls": total_synthesis_calls,
        "total_synthesis_seconds": round(total_synthesis_seconds, 4),
        "random_seed": seed,
        "duration_s": round(time.time() - started, 1),
        "per_game": per_game,
        "markov_vs_hidden_split": _markov_vs_hidden_split(per_game, hidden_state_games),
        "gap4_executed_consistency_gate": _load_gap4_gate(),
        "caveat": (
            "Consistency energy certifies prediction trustworthiness on held-out transitions; "
            "it does not certify mechanistic understanding or interpretation. Report energy as "
            "the trustworthiness signal, not as understanding."
        ),
        "source_artifacts": [
            "results/experiment_3968_active_codex_nonspatial_sweep.json",
            "results/experiment_3975_gap4_execution_verifier_build.json",
            "results/arc3_m2_active_codex.json",
            "results/arc3_determinism_probe.json",
        ],
        "fresh_codex_used": bool(use_fresh_codex),
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
    parser.add_argument("--use_fresh_codex", action="store_true")
    args = parser.parse_args()
    run(
        games=[game.strip() for game in args.games.split(",") if game.strip()],
        train_budget=args.train_budget,
        test_budget=args.test_budget,
        episodes=args.episodes,
        iters=args.iters,
        seed=args.seed,
        use_fresh_codex=args.use_fresh_codex,
    )
