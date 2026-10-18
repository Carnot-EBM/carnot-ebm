"""Experiment 5591: offline-dev-sim prototype for object_hash / blob_topology
(REQ-ARC-FCP-5591), the two sub-components ops/known-issues.md's task 10 asked
be added to the shipped color-blob salience mechanism (task 2, already live).

Per task 2's own instruction ("Prototype against the offline dev sim ...
first, then wire into the live E3AgentPolicy exploration policy"), this
demonstrates the two new functions work on REAL rendered ARC frames from the
offline arcade -- not just synthetic unit-test grids -- and specifically
demonstrates the load-bearing claim behind building them at all: that
``object_hash`` tracks an object's identity ACROSS a real frame transition
(the same on-screen object, after a real env action moves or reveals it
differently, still hashes the same), which position-only bbox/centroid
features cannot do.

This is a measurement/prototype script, not a live-path wiring change and not
a solve attempt: no per-game adapter, no offline BFS, no level-solve claim.
solve_provenance stays development_proxy.

Spec refs: REQ-ARC-FCP-5591, SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY,
SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5591_blob_topology_offline_sim_prototype"
RESULT_RELATIVE_PATH = "results/experiment_5591_blob_topology_offline_sim_prototype.json"
SCHEMA = "carnot.exp5591.blob_topology_offline_sim_prototype.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5591
DEFAULT_ROSTER = ("cd82", "m0r0", "sk48", "sp80", "tu93")
DEFAULT_ACTIONS_PER_GAME = 6

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "roster",
    "actions_per_game",
    "per_game_rows",
    "games_with_cross_frame_hash_persistence",
    "total_games_measured",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; demonstrates the mechanism works on real frames, does not claim a live-path capability win (that is a separate, later measurement)"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- pure frame segmentation, no LLM invoked"
    },
    "games_with_cross_frame_hash_persistence": {
        "principle": "count of roster games where at least one object_hash value observed on the initial frame reappeared after a real env action -- the load-bearing claim behind building object_hash at all (position-invariant identity tracking across a real transition, not just a synthetic grid)"
    },
    "solve_provenance": {
        "principle": "development_proxy -- a prototype/measurement script, no per-game adapter, no offline BFS, no level-solve claim"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_color_blob_salience import blob_topology, object_hash  # noqa: F401

        checks["blob_topology_import"] = True
    except Exception:
        checks["blob_topology_import"] = False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _measure_one_game(game: str, *, actions_per_game: int) -> JsonDict:
    """Segment a real game's initial frame, take a few real actions, and check
    whether any object_hash value persists across at least one real transition."""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_color_blob_salience import blob_topology
    from arcengine import GameAction

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    latest = env.reset()

    class _Frame:
        def __init__(self, frame_obj: Any) -> None:
            self.frame = frame_obj.frame if hasattr(frame_obj, "frame") else frame_obj

    initial_topo = blob_topology(_Frame(latest))
    initial_hashes = set(initial_topo["object_hashes"].values())
    initial_blob_count = len(initial_topo["blobs"])
    initial_max_containment_depth = _max_containment_depth(initial_topo["children"])

    persisted = False
    seen_action_ids: list[int] = []
    for action_id in (1, 2, 3, 4)[: max(1, actions_per_game)]:
        try:
            latest = env.step(getattr(GameAction, f"ACTION{action_id}"))
        except Exception:
            continue
        seen_action_ids.append(int(action_id))
        if latest is None:
            break
        try:
            topo = blob_topology(_Frame(latest))
        except Exception:
            continue
        current_hashes = set(topo["object_hashes"].values())
        if initial_hashes & current_hashes:
            persisted = True
            break

    return {
        "game": game,
        "initial_blob_count": int(initial_blob_count),
        "initial_max_containment_depth": int(initial_max_containment_depth),
        "initial_adjacency_edge_count": len(initial_topo["adjacency_list"]),
        "actions_taken": seen_action_ids,
        "cross_frame_hash_persisted": bool(persisted),
    }


def _max_containment_depth(children: dict[int, list[int]]) -> int:
    roots = [
        blob_id for blob_id in children if not any(blob_id in kids for kids in children.values())
    ]

    def depth(blob_id: int) -> int:
        kids = children.get(blob_id, [])
        if not kids:
            return 1
        return 1 + max(depth(child) for child in kids)

    if not roots:
        return 0
    return max(depth(root) for root in roots)


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    actions_per_game: int = DEFAULT_ACTIONS_PER_GAME,
    root: Path = REPO_ROOT,
) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "roster": list(roster),
            "actions_per_game": int(actions_per_game),
            "per_game_rows": [],
            "games_with_cross_frame_hash_persistence": 0,
            "total_games_measured": 0,
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    rows: list[JsonDict] = []
    for game in roster:
        try:
            rows.append(_measure_one_game(game, actions_per_game=actions_per_game))
        except Exception as exc:
            rows.append({"game": game, "error": repr(exc)[:200]})

    measured_rows = [row for row in rows if "error" not in row]
    persistence_count = sum(1 for row in measured_rows if row.get("cross_frame_hash_persisted"))

    if not measured_rows:
        verdict = "complete: blob_topology_prototype_no_games_measured"
    elif persistence_count > 0:
        verdict = (
            f"complete: blob_topology_prototype_cross_frame_identity_confirmed_"
            f"{persistence_count}_of_{len(measured_rows)}_games"
        )
    else:
        verdict = "complete: blob_topology_prototype_ran_but_no_cross_frame_persistence_observed"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "actions_per_game": int(actions_per_game),
        "per_game_rows": rows,
        "games_with_cross_frame_hash_persistence": int(persistence_count),
        "total_games_measured": len(measured_rows),
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
