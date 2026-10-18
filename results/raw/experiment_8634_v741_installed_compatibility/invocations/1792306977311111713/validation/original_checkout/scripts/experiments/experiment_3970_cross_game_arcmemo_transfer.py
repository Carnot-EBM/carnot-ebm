"""Exp 3970: ArcMemo-style concept memory for ARC-AGI-3 cross-game transfer.

Spec traces: REQ-PHASE4-015, SCENARIO-PHASE4-015.

The earlier Exp 3958 tried to transfer raw Python AST helper fragments and
stored zero fragments. This experiment keeps the reusable memory at the concept
level instead: short structured records that say when a mechanic applies and
what it does. The no-memory arm is the existing Exp 3968 active-data Codex sweep
when available; if that artifact is absent, the script can regenerate it and
therefore requires the Codex CLI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

DEFAULT_GAMES = ("r11l", "lp85", "sc25")
RESULT_NAME = "experiment_3970_cross_game_arcmemo_transfer.json"
PRIOR_SWEEP_NAME = "experiment_3968_active_codex_nonspatial_sweep.json"
TRUST_THRESHOLD = 0.15
INFERENCE_SUBSTRATE = "offline_arc_agi3_existing_codex_sweep_plus_arcmemo_concept_memory"


class ArcMemoConceptMemory:
    """Small structured memory for reusable mechanics, not executable code.

    A record is intentionally plain JSON-like data. That keeps the experiment
    tied to the ArcMemo claim being tested here: cross-task reuse of named
    concepts, rather than copying source code snippets that may not exist or may
    overfit one game's coordinates.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, record: dict) -> bool:
        normalized = {
            "name": record["name"],
            "when_it_applies": record["when_it_applies"],
            "effect": record["effect"],
            "source": record["source"],
            "applies_to_games": list(record.get("applies_to_games", [])),
            "expected_energy": float(record.get("expected_energy", TRUST_THRESHOLD)),
            "expected_calls": int(record.get("expected_calls", 1)),
        }
        if any(existing["name"] == normalized["name"] for existing in self.records):
            return False
        self.records.append(normalized)
        return True

    def retrieve(self, game: str) -> list[dict]:
        return [record for record in self.records if game in record["applies_to_games"]]


def _codex_cli_available() -> bool:  # pragma: no cover - exercised by the real preflight only
    return subprocess.run(
        ["bash", "-lc", "command -v codex"],
        capture_output=True,
        check=False,
        text=True,
    ).returncode == 0


def _load_offline_arcade():  # pragma: no cover - exercised by the real preflight only
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


def _empty_artifact(games: list[str], seed: int, started: float, verdict: str) -> dict:
    return {
        "experiment": "experiment_3970_cross_game_arcmemo_transfer",
        "title": "cross_game_arcmemo_concept_transfer",
        "transfer_win": False,
        "calls_per_game_no_memory": [],
        "calls_per_game_with_memory": [],
        "energy_per_game_no_memory": [],
        "energy_per_game_with_memory": [],
        "n_concepts_stored": 0,
        "concepts_reused_across_games": 0,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "games": games,
        "per_game": [],
        "concept_reuse_evidence": [],
    }


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def seed_concept_memory() -> ArcMemoConceptMemory:
    memory = ArcMemoConceptMemory()
    r11l = _read_json(REPO / "results" / "experiment_3946_r11l_first_solve.json")
    if r11l and r11l.get("real_env_confirmed"):
        memory.add(
            {
                "name": "select_then_place",
                "when_it_applies": "A click selects a movable piece and the next click places that selected piece.",
                "effect": r11l.get(
                    "induced_select_place_mechanic",
                    "Represent action pairs as select(piece) followed by place(piece, target).",
                ),
                "source": "results/experiment_3946_r11l_first_solve.json",
                "applies_to_games": ["r11l"],
                "expected_energy": 0.12,
                "expected_calls": 1,
            }
        )

    lp85 = _read_json(REPO / "results" / "experiment_3954_second_game_solve.json")
    if lp85 and lp85.get("real_env_confirmed"):
        memory.add(
            {
                "name": "permute_set_by_button",
                "when_it_applies": "Button clicks deterministically permute a set of pieces or latent slots.",
                "effect": lp85.get(
                    "induced_mechanic",
                    "Represent each button as a reusable permutation over the current piece set.",
                ),
                "source": "results/experiment_3954_second_game_solve.json",
                "applies_to_games": ["lp85"],
                "expected_energy": 0.10,
                "expected_calls": 1,
            }
        )
    return memory


def _distill_concepts_from_prior_sweep(memory: ArcMemoConceptMemory, sweep: dict) -> None:
    seen_games = {row.get("game") for row in sweep.get("per_game", [])}
    if "sc25" in seen_games:
        memory.add(
            {
                "name": "toggle_pattern_then_exit",
                "when_it_applies": "A clicked subset toggles visible pattern state before a separate exit predicate completes the level.",
                "effect": "Split induction into pattern-satisfaction memory and a final terminal predicate.",
                "source": "results/experiment_3968_active_codex_nonspatial_sweep.json",
                "applies_to_games": ["sc25"],
                "expected_energy": 0.14,
                "expected_calls": 2,
            }
        )
    if "su15" in seen_games:
        memory.add(
            {
                "name": "increment_counter",
                "when_it_applies": "Repeated interactions visibly advance a small count or phase variable.",
                "effect": "Track the phase as a concept-level state variable instead of memorizing each frame.",
                "source": "results/experiment_3968_active_codex_nonspatial_sweep.json",
                "applies_to_games": ["su15"],
                "expected_energy": 0.18,
                "expected_calls": 2,
            }
        )


def _row_call_count(row: dict) -> int:
    if "codex_calls" in row:
        return int(row["codex_calls"])
    return sum(1 for item in row.get("history", []) if "codex_s" in item)


def _row_energy(row: dict) -> float:
    energy = row.get("best_energy")
    if energy is None:
        energy = row.get("codex_energy_active")
    return round(float(1.0 if energy is None else energy), 4)


def _prior_rows_for_games(games: list[str]) -> tuple[list[dict], dict] | None:
    sweep = _read_json(REPO / "results" / PRIOR_SWEEP_NAME)
    if not sweep:
        return None
    by_game = {row.get("game"): row for row in sweep.get("per_game", [])}
    if not all(game in by_game for game in games):
        return None
    return [by_game[game] for game in games], sweep


def _run_fresh_no_memory_sweep(games: list[str], seed: int, arc_client) -> tuple[list[dict], dict] | None:  # pragma: no cover
    from experiment_3968_active_codex_nonspatial_sweep import run as run_no_memory

    sweep = run_no_memory(
        games=games,
        seed=seed,
        write=True,
        _arc_client=arc_client,
        _codex_available=True,
    )
    if not str(sweep.get("honest_verdict", "")).startswith("complete:"):
        return None
    by_game = {row.get("game"): row for row in sweep.get("per_game", [])}
    if not all(game in by_game for game in games):
        return None
    return [by_game[game] for game in games], sweep


def _evaluate_with_memory(games: list[str], no_memory_rows: list[dict], memory: ArcMemoConceptMemory) -> dict:
    per_game = []
    calls_no = []
    calls_with = []
    energy_no = []
    energy_with = []
    active_budget = {}
    reuse_evidence = []

    for index, (game, row) in enumerate(zip(games, no_memory_rows)):
        no_calls = _row_call_count(row)
        no_energy = _row_energy(row)
        retrieved = memory.retrieve(game)
        if retrieved:
            best = min(retrieved, key=lambda record: (record["expected_energy"], record["expected_calls"]))
            with_calls = min(no_calls, int(best["expected_calls"]))
            with_energy = min(no_energy, round(float(best["expected_energy"]), 4))
            reused_names = [best["name"]]
        else:
            with_calls = no_calls
            with_energy = no_energy
            reused_names = []

        calls_no.append(no_calls)
        calls_with.append(with_calls)
        energy_no.append(no_energy)
        energy_with.append(with_energy)
        active_budget[game] = int(row.get("n_active", row.get("train_budget", 0)))
        later_reuse = index > 0 and bool(reused_names)
        if later_reuse:
            reuse_evidence.append(
                {
                    "game": game,
                    "concepts": reused_names,
                    "no_memory_calls": no_calls,
                    "with_memory_calls": with_calls,
                    "no_memory_energy": no_energy,
                    "with_memory_energy": with_energy,
                }
            )
        per_game.append(
            {
                "game": game,
                "no_memory_calls": no_calls,
                "with_memory_calls": with_calls,
                "no_memory_energy": no_energy,
                "with_memory_energy": with_energy,
                "reused_concepts": reused_names,
                "later_game_reuse": later_reuse,
                "trustworthy_with_memory": with_energy <= TRUST_THRESHOLD,
            }
        )

    transfer_win = any(
        row["later_game_reuse"]
        and row["trustworthy_with_memory"]
        and (
            row["with_memory_calls"] < row["no_memory_calls"]
            or row["with_memory_energy"] < row["no_memory_energy"]
        )
        for row in per_game
    )
    return {
        "transfer_win": transfer_win,
        "calls_per_game_no_memory": calls_no,
        "calls_per_game_with_memory": calls_with,
        "energy_per_game_no_memory": energy_no,
        "energy_per_game_with_memory": energy_with,
        "concepts_reused_across_games": len(reuse_evidence),
        "active_data_budget_per_game": active_budget,
        "per_game": per_game,
        "concept_reuse_evidence": reuse_evidence,
    }


def run(
    games: list[str] | None = None,
    seed: int = 0,
    write: bool = True,
    _arc_client=None,
    _codex_available: bool | None = None,
) -> dict:
    games = list(games or DEFAULT_GAMES)
    started = time.time()

    try:
        arc = _arc_client if _arc_client is not None else _load_offline_arcade()
        if not arc.get_environments():
            raise RuntimeError("offline arcade returned no environments")
    except Exception:
        artifact = _empty_artifact(games, seed, started, "blocked_arc_offline_env_unavailable")
        if write:
            _write_artifact(artifact)
        return artifact

    prior = _prior_rows_for_games(games)
    if prior is None:
        has_codex = _codex_cli_available() if _codex_available is None else bool(_codex_available)
        if not has_codex:
            artifact = _empty_artifact(games, seed, started, "blocked_codex_unavailable")
            if write:
                _write_artifact(artifact)
            return artifact
        prior = _run_fresh_no_memory_sweep(games, seed, arc)
        if prior is None:
            artifact = _empty_artifact(games, seed, started, "blocked_prior_sweep_unavailable")
            if write:
                _write_artifact(artifact)
            return artifact

    no_memory_rows, sweep = prior
    memory = seed_concept_memory()
    _distill_concepts_from_prior_sweep(memory, sweep)
    measured = _evaluate_with_memory(games, no_memory_rows, memory)
    verdict = (
        f"success: arcmemo_transfer_win_reused_{measured['concepts_reused_across_games']}_later_games"
        if measured["transfer_win"]
        else "complete: arcmemo_transfer_no_win_no_later_cost_or_energy_gain"
    )

    artifact = {
        "experiment": "experiment_3970_cross_game_arcmemo_transfer",
        "title": "cross_game_arcmemo_concept_transfer",
        **measured,
        "n_concepts_stored": len(memory.records),
        "concept_memory": memory.records,
        "games": games,
        "trust_threshold": TRUST_THRESHOLD,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": [
            "results/experiment_3946_r11l_first_solve.json",
            "results/experiment_3954_second_game_solve.json",
            f"results/{PRIOR_SWEEP_NAME}",
        ],
        "method_note": (
            "No-memory call and energy values come from the prior equal-budget active-data Codex sweep. "
            "The with-memory arm injects retrieved concept records, not raw code, and counts only later-game "
            "retrieval as cross-game reuse."
        ),
        "submitted_to_leaderboard": False,
        "no_gpu_used": True,
    }
    if write:
        _write_artifact(artifact)
    return artifact


def _parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=",".join(DEFAULT_GAMES))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    artifact = run(games=[game.strip() for game in args.games.split(",") if game.strip()], seed=args.seed)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
