#!/usr/bin/env python3
"""CHEAP MEASUREMENT (operator-directed, 2026-06-27): does a structurally-correct
STATE-energy E(s) actually discriminate win-states from non-win states?

THE QUESTION
------------
The whole thread converged on: the binding ARC wall is GENERATION ("make-a-winner-appear"),
and energy of any type (transition or state) is a SELECTION/scoring device. The operator's
proposal -- "generate a NEW energy model that is structurally the right type" (a STATE-energy,
not the transition-energy) -- is genuinely untried as a goal AND structurally sound in the
narrow sense: a state-energy is a DISCRIMINATOR, which the perception-note reconciliation
MEASURED as adequate (value/discrimination LOO-AUROC 0.725), unlike the exact goal-predicate
that the .446 object-identity wall blocks.

So before any planning/generation integration (which is generation-gated and would re-null),
this isolates the MODEL: does a contrastive state-energy, trained on the agent's OWN observed
win/non-win states, discriminate them on a HELD-OUT game (leave-one-game-out, the same
methodology as the 0.725 value-head number)? This is the clean, cheap, decisive closure
measurement -- CPU, offline, no LLM, no 3090.

INTERPRETATION
--------------
- LOGO AUROC >> 0.5 (shuffled control ~0.5): the state-energy IS a working discriminator --
  the operator's "right type" works as a model. It still cannot MOVE the wall (it is a
  selector, the wall is generation) -- but the energy model itself is sound. Clean framing:
  "the structurally-correct energy works; the wall is generation, not the energy's type."
- LOGO AUROC ~0.5: even the discriminator fails (too few / too-heterogeneous cross-game
  win-states) -- the state-energy does not generalize across games. Clean negative.

DESIGN (avoids the S2 degenerate-pool / FALSE_NEGATIVE traps)
------------------------------------------------------------
- Corpus: replay each deepening game's banked solution through the OFFLINE sim, capture
  per-step (_to_grid(frame), levels_completed). WIN-states = grids at a level INCREMENT;
  NON-WIN = the rest of the trajectory.
- Model: reuse the SpatialValueNet position-preserving CNN body (arc_value_net) trained as an
  ENERGY: win-states -> target 0.0 (low energy), non-win -> 1.0 (high). E(s)=predict_grid.
- Eval: LEAVE-ONE-GAME-OUT (LOGO) -- train on the other games' states, test win-vs-non-win
  discrimination AUROC on the held-out game (cross-game generalization, the regime that matters).
- CONTROL (mandatory): a SHUFFLED-LABEL arm (train on shuffled win/non-win labels) must score
  ~0.5 -- proves a real LOGO AUROC is signal, not a harness artifact.
- Win-states are scarce, so D4 (dihedral) + color-permutation augmentation (both ARC-invariant)
  balances the classes for training. AUROC is reported on UN-augmented held-out states.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402

SEED = 4664
MAX_NONWIN_PER_GAME = 120  # subsample to keep the corpus balanced + fast


def _trace_labels(game: str) -> list[str]:
    p = REPO / "results" / f"arc_loop_solve_{game}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    sl = d.get("solution_labels") or d.get("solution")
    # only a LIST of action labels is replayable; `moves` is an int count, not labels
    return list(sl) if isinstance(sl, list) else []


def _discover_games() -> list[str]:
    """Auto-discover deepening games with REPLAYABLE multi-step solves (a non-empty
    solution_labels/solution LIST + reached_level >= 1). Many banked traces store only a
    move COUNT or are empty; those are skipped."""
    games = []
    for p in sorted((REPO / "results").glob("arc_loop_solve_*.json")):
        game = p.stem.replace("arc_loop_solve_", "")
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        sl = d.get("solution_labels") or d.get("solution")
        if isinstance(sl, list) and len(sl) >= 2 and int(d.get("reached_level") or 0) >= 1:
            games.append(game)
    return games


GAMES = _discover_games()


def _capture_states(game: str):
    """Replay the banked solution in the offline sim; return (win_grids, nonwin_grids)."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_value_net import _to_grid
    from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
    from arcengine import GameAction

    labels = _trace_labels(game)
    if not labels:
        return [], [], "no_trace"
    arc = kit.offline_arcade()
    # resolve game id
    gid = None
    for env_meta in arc.get_environments():
        if str(getattr(env_meta, "game_id", "")).split("-")[0] == game:
            gid = str(env_meta.game_id)
            break
    if gid is None:
        return [], [], "game_unavailable"
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    prev_level = _levels_completed(frame)
    win, nonwin = [], []
    for label in labels:
        if label == "RESET":
            frame = env.reset()
            continue
        try:
            step = json.loads(label)
            frame = env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))
        except Exception:
            break
        if frame is None:
            break
        lvl = _levels_completed(frame)
        g = _to_grid(frame)
        if lvl > prev_level:
            win.append(g)  # the level-COMPLETION grid
        else:
            nonwin.append(g)
        prev_level = lvl
    return win, nonwin, "ok"


def _augment(grids: list, rng: np.random.RandomState, factor: int):
    """D4 dihedral + color-permutation augmentation (both ARC-label-invariant)."""
    out = []
    for g in grids:
        out.append(g)
        for _ in range(factor):
            h = g.copy()
            k = rng.randint(4)
            h = np.rot90(h, k)
            if rng.rand() < 0.5:
                h = np.fliplr(h)
            # color permutation of the non-zero palette (0 stays background)
            colors = sorted(int(c) for c in np.unique(h) if c != 0)
            if colors:
                perm = rng.permutation(colors)
                mapping = {c: int(p) for c, p in zip(colors, perm)}
                h = np.vectorize(lambda v: mapping.get(int(v), int(v)))(h).astype(h.dtype)
            out.append(np.ascontiguousarray(h))
    return out


def _auroc(scores_pos, scores_neg) -> float:
    """AUROC that POSITIVES (win-states) get LOWER energy than NEGATIVES (non-win).
    Rank by -energy so 'win ranked above non-win' = correct."""
    pos = [-float(s) for s in scores_pos]
    neg = [-float(s) for s in scores_neg]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _train_energy(train_grids, train_labels, seed):
    """Reuse SpatialValueNet body; train scalar output as energy (win->0, nonwin->1)."""
    from carnot.agentic.arc_value_net import SpatialValueNet

    net = SpatialValueNet()
    net.fit(list(train_grids), [float(v) for v in train_labels], epochs=80, lr=1e-3, seed=seed)
    return net


def run() -> dict:
    started = time.time()
    rng = np.random.RandomState(SEED)
    per_game_states = {}
    for game in GAMES:
        win, nonwin, status = _capture_states(game)
        if len(nonwin) > MAX_NONWIN_PER_GAME:
            idx = rng.choice(len(nonwin), MAX_NONWIN_PER_GAME, replace=False)
            nonwin = [nonwin[i] for i in idx]
        per_game_states[game] = {"win": win, "nonwin": nonwin, "status": status}
        print(f"[{game}] status={status} n_win={len(win)} n_nonwin={len(nonwin)}", flush=True)

    usable = [g for g, s in per_game_states.items() if s["win"] and s["nonwin"]]
    if len(usable) < 2:
        artifact = {
            "experiment": "arc3_state_goal_energy_discrimination",
            "honest_verdict": "blocked_insufficient_win_states_for_logo",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "usable_games": usable,
            "per_game_counts": {g: {"n_win": len(s["win"]), "n_nonwin": len(s["nonwin"])} for g, s in per_game_states.items()},
            "duration_s": round(time.time() - started, 2),
        }
        _write(artifact)
        return artifact

    # LEAVE-ONE-GAME-OUT discrimination AUROC (real labels + shuffled control)
    per_game_auroc = {}
    per_game_auroc_shuffled = {}
    for held in usable:
        train_win, train_nonwin = [], []
        for g in usable:
            if g == held:
                continue
            train_win += per_game_states[g]["win"]
            train_nonwin += per_game_states[g]["nonwin"]
        # augment scarce win-states to ~balance non-wins
        aug_factor = max(1, min(8, len(train_nonwin) // max(1, len(train_win))))
        train_win_aug = _augment(train_win, rng, aug_factor)
        grids = train_win_aug + train_nonwin
        labels = [0.0] * len(train_win_aug) + [1.0] * len(train_nonwin)

        # REAL labels
        net = _train_energy(grids, labels, seed=SEED)
        held_win_e = [net.predict_grid(g) for g in per_game_states[held]["win"]]
        held_nonwin_e = [net.predict_grid(g) for g in per_game_states[held]["nonwin"]]
        per_game_auroc[held] = round(_auroc(held_win_e, held_nonwin_e), 4)

        # SHUFFLED-label control (same grids, permuted labels)
        shuffled = list(labels)
        rng.shuffle(shuffled)
        net_s = _train_energy(grids, shuffled, seed=SEED + 1)
        held_win_es = [net_s.predict_grid(g) for g in per_game_states[held]["win"]]
        held_nonwin_es = [net_s.predict_grid(g) for g in per_game_states[held]["nonwin"]]
        per_game_auroc_shuffled[held] = round(_auroc(held_win_es, held_nonwin_es), 4)
        print(
            f"[LOGO held={held}] auroc={per_game_auroc[held]} shuffled={per_game_auroc_shuffled[held]}",
            flush=True,
        )

    real_vals = [v for v in per_game_auroc.values() if v == v]  # drop nan
    shuf_vals = [v for v in per_game_auroc_shuffled.values() if v == v]
    mean_auroc = round(float(np.mean(real_vals)), 4) if real_vals else float("nan")
    mean_shuffled = round(float(np.mean(shuf_vals)), 4) if shuf_vals else float("nan")

    # Verdict: discriminates iff mean LOGO AUROC materially beats the shuffled control AND > 0.6
    discriminates = bool(real_vals and mean_auroc >= 0.6 and mean_auroc > mean_shuffled + 0.1)
    if discriminates:
        verdict = (
            "complete: state_energy_discriminates_win_states_logo_"
            f"{mean_auroc}_but_role_is_selection_not_generation"
        )
    else:
        verdict = (
            "complete: state_energy_does_not_discriminate_win_states_cross_game_logo_"
            f"{mean_auroc}_vs_shuffled_{mean_shuffled}"
        )

    artifact = {
        "experiment": "arc3_state_goal_energy_discrimination",
        "schema": "carnot.arc3_state_goal_energy_discrimination.v1",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "question": (
            "does a structurally-correct contrastive STATE-energy discriminate win-states from "
            "non-win states cross-game (LOGO), isolating the model from the generation-gated "
            "planning integration?"
        ),
        "mean_logo_auroc": mean_auroc,
        "mean_shuffled_control_auroc": mean_shuffled,
        "state_energy_discriminates": discriminates,
        "per_game_logo_auroc": per_game_auroc,
        "per_game_shuffled_auroc": per_game_auroc_shuffled,
        "per_game_counts": {
            g: {"n_win": len(s["win"]), "n_nonwin": len(s["nonwin"]), "status": s["status"]}
            for g, s in per_game_states.items()
        },
        "usable_games": usable,
        "value_head_reference_loo_auroc": 0.725,
        "methodology_note": (
            "Cross-game LEAVE-ONE-GAME-OUT, the same methodology as the 0.725 value-head number. "
            "Win-states = level-increment grids from banked offline solves; non-win = the rest. "
            "Win-states augmented (D4 + color-permute, ARC-invariant) to balance; AUROC reported on "
            "UN-augmented held-out states. SHUFFLED-label control must be ~0.5 (degenerate-pool / "
            "FALSE_NEGATIVE guard). This measures the MODEL (discrimination) only; it does NOT test "
            "whether the energy moves the GENERATION wall -- that is gated on the generation problem "
            "the whole thread diagnosed as binding."
        ),
        "interpretation": (
            "A high LOGO AUROC means the structurally-correct state-energy WORKS as a discriminator "
            "(the operator's 'right type' is sound) -- but it remains a SELECTOR; the wall is "
            "generation (make-a-winner-appear), so a working discriminator still cannot move the "
            "headline wall, only rank/shape a frontier a generator must first populate with a winner."
        ),
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    _write(artifact)
    print("\n=== VERDICT:", verdict)
    print(f"mean LOGO AUROC={mean_auroc}  shuffled control={mean_shuffled}  (value-head ref 0.725)")
    return artifact


def _write(artifact: dict) -> None:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    (REPO / "results" / "arc3_state_goal_energy_discrimination.json").write_text(
        json.dumps(artifact, indent=2, default=str) + "\n"
    )
    print("-> results/arc3_state_goal_energy_discrimination.json", flush=True)


if __name__ == "__main__":
    run()
