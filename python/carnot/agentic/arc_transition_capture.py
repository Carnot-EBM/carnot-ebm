"""Transition-capture sink: a growing, on-disk corpus of played ARC-AGI-3 transitions.

Operator directive (2026-06-21): capture gameplay for learning/training. We were discarding every
transition (recordings dir empty; collect_transitions gathered on-demand and threw the data away). This
sink persists every observed (game, grid, action, data, next_grid, level_before, level_after) so it can
fuel (a) the PRETRAINED MECHANIC PRIOR -- a dynamics backbone trained on the public-game corpus that the
live per-game learner warm-starts from (faster adaptation = fewer real probe actions = more of the 5n
per-level budget left to solve), and (b) the offline dev harness, and (c) debugging (the sc25-style
offline-reproduce-but-live-fail case).

IMPORTANT competition caveat: this captures only the 25 PUBLIC games (the only accessible signal) -- the
55 semi-private + 55 fully-private eval games are held out and NEVER seen before scoring. So the corpus
is for training a GENERAL, transferable prior + the machinery, NOT for memorizing the eval games. The
public games are our window into the mechanic DISTRIBUTION the hidden games are drawn from.

Storage: one compressed .npz per game under data/arc_transition_corpus/<game>.npz, deduped by
(grid.tobytes(), action, x, y) so repeated probes don't bloat it. Append-only across runs; survives.
Pure numpy, no torch -- the capture path stays light.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = _REPO / "data" / "arc_transition_corpus"
# The human-recorded play corpus (frame_action_delta.v1, ~14,797 rows over the 25 public games, CC BY 4.0
# from ARC Prize). Goal-directed human play -- reaches wins, demonstrates mechanics competently -- so it is
# far higher-signal than random/salience probes for pretraining the mechanic prior.
HUMAN_REPLAY_DIR = _REPO / "data" / "arc_public_demo_human_replay_corpus"


def _parse_action_id(raw: Any) -> Optional[int]:
    """Action ids appear as ints (2) OR strings ('ACTION4'); RESET/None -> not a transition action."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).upper().replace("ACTION", "").strip()
    return int(s) if s.isdigit() else None


def _akey(action: int, data: Any) -> tuple:
    if int(action) == 6 and isinstance(data, dict) and "x" in data and "y" in data:
        return (6, int(data["x"]), int(data["y"]))
    return (int(action), -1, -1)


class TransitionCorpus:
    """Append-only per-game store of played transitions. Dedups on (grid, action, x, y)."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._buf: dict[str, dict[str, list]] = {}     # game -> {grids, next_grids, actions, xs, ys, lb, la}
        self._seen: dict[str, set] = {}                # game -> set of dedup keys (loaded lazily)

    # --- write path ---------------------------------------------------------------------------------
    def _ensure_seen(self, game: str) -> set:
        if game not in self._seen:
            keys: set = set()
            p = self.root / f"{game}.npz"
            if p.exists():
                d = np.load(p, allow_pickle=False)
                for g, a, x, y in zip(d["grids"], d["actions"], d["xs"], d["ys"]):
                    keys.add((g.tobytes(), int(a), int(x), int(y)))
            self._seen[game] = keys
        return self._seen[game]

    def add(self, game: str, grid: Any, action: int, data: Any, next_grid: Any,
            level_before: int = 0, level_after: int = 0) -> bool:
        """Buffer one transition; returns True if NEW (not a dedup hit). Call flush() to persist."""
        g = np.asarray(grid).astype(np.int16)
        ng = np.asarray(next_grid).astype(np.int16)
        ak = _akey(action, data)
        key = (g.tobytes(), ak[0], ak[1], ak[2])
        if key in self._ensure_seen(game):
            return False
        self._seen[game].add(key)
        b = self._buf.setdefault(game, {k: [] for k in ("grids", "next_grids", "actions", "xs", "ys", "lb", "la")})
        b["grids"].append(g); b["next_grids"].append(ng)
        b["actions"].append(ak[0]); b["xs"].append(ak[1]); b["ys"].append(ak[2])
        b["lb"].append(int(level_before)); b["la"].append(int(level_after))
        return True

    def add_transition(self, game: str, t: Any) -> bool:
        return self.add(game, t.grid, t.action, t.data, t.next_grid,
                        getattr(t, "level_before", 0), getattr(t, "level_after", 0))

    def flush(self) -> dict:
        """Persist all buffered games (merging with any existing on-disk corpus). Returns per-game counts."""
        counts = {}
        for game, b in self._buf.items():
            if not b["grids"]:
                continue
            p = self.root / f"{game}.npz"
            arrs = {k: np.asarray(v) for k, v in b.items()}
            arrs["grids"] = np.stack(b["grids"]).astype(np.int16)
            arrs["next_grids"] = np.stack(b["next_grids"]).astype(np.int16)
            if p.exists():
                old = np.load(p, allow_pickle=False)
                for k in arrs:
                    arrs[k] = np.concatenate([old[k], arrs[k]], axis=0)
            np.savez_compressed(p, **arrs)
            counts[game] = int(arrs["grids"].shape[0])
        self._buf.clear()
        return counts

    # --- read path ----------------------------------------------------------------------------------
    def games(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.npz"))

    def load(self, game: str) -> list:
        """Return stored transitions as Transition objects (for training / the harness)."""
        from carnot.agentic.arc_executable_world_model import Transition

        p = self.root / f"{game}.npz"
        if not p.exists():
            return []
        d = np.load(p, allow_pickle=False)
        # CRITICAL: a COMPRESSED NpzFile re-decompresses the WHOLE array on EVERY d[key] access. Materialise
        # each array exactly ONCE here -- indexing d["grids"][i] inside the loop decompressed the full 5.6MB
        # grids array 679x = ~7.5GB of transient host RAM per game, which OOM-kills a 16GB-RAM eval notebook
        # (operator 2026-06-21: the Kaggle notebook's ~16GB system RAM is as scarce as its 16GB VRAM, and the
        # live agent loads its captured corpus to warm-start). After this fix, loading a game is ~11MB.
        grids = d["grids"]; next_grids = d["next_grids"]
        actions = d["actions"]; xs = d["xs"]; ys = d["ys"]; lb = d["lb"]; la = d["la"]
        out = []
        for i in range(grids.shape[0]):
            a = int(actions[i]); x = int(xs[i]); y = int(ys[i])
            data = {"x": x, "y": y} if a == 6 and x >= 0 else None
            out.append(Transition(grid=grids[i], action=a, data=data, next_grid=next_grids[i],
                                  level_before=int(lb[i]), level_after=int(la[i])))
        return out

    def stats(self) -> dict:
        per_game = {}
        total = 0
        for g in self.games():
            d = np.load(self.root / f"{g}.npz", allow_pickle=False)
            gr = d["grids"]; ngr = d["next_grids"]  # materialise once (compressed NpzFile re-decompresses per access)
            n = int(gr.shape[0])
            chg = int(np.sum([not np.array_equal(gr[i], ngr[i]) for i in range(n)]))
            per_game[g] = {"transitions": n, "changing": chg, "grid_shape": list(gr.shape[1:])}
            total += n
        return {"root": str(self.root), "games": len(per_game), "total_transitions": total, "per_game": per_game}


def ingest_human_replays(corpus_dir: Optional[Path] = None, games: Optional[list[str]] = None,
                         root: Optional[Path] = None) -> dict:
    """Fold the HUMAN play corpus into the transition store. Each row is (env, frame, action,
    level_progress) ordered within a session `guid`; transitions are derived by pairing CONSECUTIVE
    frames: (frame_i, action_i) -> frame_{i+1}. A level-up is flagged when level_progress RESETS (a high
    progress followed by ~0 = the action completed the level). Deduped against the existing corpus, so it
    AUGMENTS the probe transitions without duplication."""
    d = Path(corpus_dir) if corpus_dir is not None else HUMAN_REPLAY_DIR
    shards = sorted((d / "shards").glob("*.jsonl"))
    if not shards:
        return {"error": f"no shards under {d}/shards"}
    corpus = TransitionCorpus(root)
    sessions: dict[str, list] = {}
    for sh in shards:
        with sh.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if games and r.get("env") not in games:
                    continue
                sessions.setdefault(r.get("guid"), []).append(r)
    added = 0
    for rows in sessions.values():
        rows.sort(key=lambda r: r.get("step_index", 0))
        for a, b in zip(rows, rows[1:]):
            act = a.get("action") or {}
            aid = _parse_action_id(act.get("id"))
            if aid is None:  # RESET / unparseable -> not a frame transition
                continue
            adata = act.get("data") or {}
            x, y = adata.get("x"), adata.get("y")
            data = {"x": int(x), "y": int(y)} if (aid == 6 and x is not None and y is not None) else None
            la = 1 if float(b.get("level_progress", 0.0)) < float(a.get("level_progress", 0.0)) - 1e-6 else 0
            if corpus.add(a["env"], a["frame"], aid, data, b["frame"], 0, la):
                added += 1
    corpus.flush()
    return {"human_transitions_added": added, "sessions": len(sessions), "stats": corpus.stats()}


def capture_public_games(games: Optional[list[str]] = None, n_per_game: int = 240,
                         root: Optional[Path] = None) -> dict:
    """Run the offline arcade probe over the public games and persist every transition to the corpus.
    The 25 public games are the only accessible signal; this is the training fuel for the mechanic prior."""
    from carnot.agentic.arc_executable_world_model import collect_transitions

    corpus = TransitionCorpus(root)
    if games is None:
        env_dir = _REPO / "environment_files"
        games = sorted(p.name for p in env_dir.iterdir() if p.is_dir()) if env_dir.exists() else []
    added = {}
    for game in games:
        try:
            transitions, _ = collect_transitions(game, n=n_per_game)
        except Exception as e:  # never fabricate; record the skip
            added[game] = f"ERROR: {type(e).__name__}"
            continue
        new = sum(corpus.add_transition(game, t) for t in transitions)
        added[game] = new
    corpus.flush()
    return {"added": added, "stats": corpus.stats()}
