"""Experiment 5601: offline-dev-sim prototype for ObjectHistorySaliencePrior
(REQ-ARC-FCP-5591-2), the live-consuming mechanism ops/known-issues.md's
2026-07-11 task 10 DONE note deferred as "a distinct, separately-scoped
design + empirical-validation step" ("preferring an object whose hash was
seen to change in a prior frame").

Demonstrates the change-history-bonus mechanism (built and unit-tested in
`arc_object_history_salience.py`, wired into `E3AgentPolicy` gated OFF by
default) actually accumulates non-degenerate tally evidence -- some
`object_hash` clearing BOTH the observation floor and showing a nonzero
change rate -- from REAL clicks collected during a REAL `E3AgentPolicy`
exploration run, not just synthetic unit-test grids. This is the empirical
half of the Phase Prototype + Empirical Validation discipline the live-wiring
step needed (the software-prototype half is `arc_object_history_salience.py`
itself plus its unit tests).

This is a measurement/prototype script, not a live-path parameter flip:
`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED` stays `False` regardless of this
script's result -- flipping it needs its own matched-budget offline A/B
(states/actions-expanded reduction, zero regression in reproduced levels),
per the `solve_rate_dropped` guardrail, exactly like `InertClickSigPruner`'s
own still-pending flip decision. `solve_provenance` stays `development_proxy`:
no per-game adapter, no offline BFS, no level-solve claim.

GAME SELECTION: reuses `m0r0`, confirmed click-heavy by exp5595's direct
probe (21 of 22 transitions were action=6) -- the same real-click-collection
need this prototype has. Reuses exp5595/exp5594's proven-reliable explicit
`LocalGGUFProposer` construction pattern (a bare `E3AgentPolicy(game,
explore_budget=N)` with no explicit `proposer=` has been observed to stall).

Spec refs: REQ-ARC-FCP-5591-2, SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL.
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
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5601_object_history_salience_offline_sim_prototype"
RESULT_RELATIVE_PATH = "results/experiment_5601_object_history_salience_offline_sim_prototype.json"
SCHEMA = "carnot.exp5601.object_history_salience_offline_sim_prototype.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5601
DEFAULT_ROSTER = ("m0r0",)
DEFAULT_EXPLORE_BUDGET = 6
DEFAULT_TOTAL_BUDGET = 40
GGUF_REPO_SUBSTR = "Qwen3.5-9B-MTP"
GGUF_PORT = 8921
MIN_OBSERVATIONS = 3

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy default tier-3 exploration proposer (real, live, port 8921)",
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "verifier_is_oracle",
    "roster",
    "explore_budget",
    "total_budget",
    "min_observations",
    "per_game_rows",
    "total_click_transitions_observed",
    "total_hashes_tracked",
    "total_hashes_with_evidence_and_nonzero_change_rate",
    "adversarial_degeneracy_check",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; demonstrates the change-history-bonus mechanism "
        "accumulates real, non-degenerate tally evidence on real clicks, does not claim a "
        "live-path efficiency win (a matched-budget A/B, like HazardMovePruner's tu93 "
        "measurement, is a separate, later step before SUBMITTED_OBJECT_HISTORY_SALIENCE_"
        "ENABLED is ever flipped True)"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- a real "
        "LocalGGUFProposer is constructed and wired into E3AgentPolicy (required to avoid the "
        "bare-construction stall exp5595 documented), but pure classical-salience exploration "
        "never invokes it; the GGUF entry in model_specs is vestigial, not a live-inference claim"
    },
    "total_hashes_with_evidence_and_nonzero_change_rate": {
        "principle": "count of distinct object_hash values that cleared min_observations AND "
        "have changed>0 -- the load-bearing claim behind wiring this mechanism at all: real "
        "objects with a genuine, non-zero track record of changing the frame when clicked, "
        "measured against real click data rather than synthetic grids"
    },
    "adversarial_degeneracy_check": {
        "principle": "per REQ-ARC-FCP-5591-2's adversarial-check unit test (same base_prior "
        "score for two same-shape-and-color candidates, differentiated only by history), "
        "confirms on THIS real game's actual data whether any two real click candidates "
        "shared an identical base_prior score before history and were differentiated only by "
        "the change-rate bonus -- catches whether the bonus is genuinely new information on "
        "real games, not just synthetic constructions"
    },
    "solve_provenance": {
        "principle": "development_proxy -- a prototype/measurement script, no per-game "
        "adapter, no offline BFS, no level-solve claim"
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
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: F401
        from carnot.agentic.arc_object_history_salience import (  # noqa: F401
            ObjectHistorySaliencePrior,
        )

        checks["e3_and_prior_import"] = True
    except Exception:
        checks["e3_and_prior_import"] = False
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    checks["gguf_cached"] = (
        any(GGUF_REPO_SUBSTR.lower() in p.name.lower() for p in hub.glob("models--*"))
        if hub.exists()
        else False
    )
    checks["llama_server_binary_present"] = bool(
        list((Path.home() / ".cache").glob("llama.cpp*/build/bin/llama-server"))
    )
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


def _measure_one_game(game: str, *, explore_budget: int, total_budget: int) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.agentic.arc_object_history_salience import ObjectHistorySaliencePrior

    proposer = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=GGUF_PORT,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=2560,
    )
    policy = E3AgentPolicy(game, proposer=proposer, explore_budget=explore_budget)
    lb.run_game(game, policy, budget=total_budget)
    transitions = list(policy.transitions)

    prior = ObjectHistorySaliencePrior(min_observations=MIN_OBSERVATIONS)
    click_count = 0
    for t in transitions:
        if int(t.action) == 6:
            click_count += 1
        prior.observe_transition(t.grid, int(t.action), t.data, t.next_grid)

    evidenced = {h: v for h, v in prior._tally.items() if v["obs"] >= MIN_OBSERVATIONS}
    nonzero_change = {h: v for h, v in evidenced.items() if v["changed"] > 0}

    # Adversarial degeneracy check (REQ-ARC-FCP-5591-2): among the click transitions actually
    # collected, do any two share an identical base_prior score (same tier) despite ending up
    # with different final (history-adjusted) scores once observed? That would confirm the
    # bonus differentiates candidates the base tier alone could not, on THIS real game's own
    # data -- not just the synthetic construction in the unit test.
    degeneracy_pairs_checked = 0
    degeneracy_confirmed_differentiating = 0
    click_rows = [t for t in transitions if int(t.action) == 6 and t.data]
    for i in range(min(len(click_rows), 12)):
        for j in range(i + 1, min(len(click_rows), 12)):
            a, b = click_rows[i], click_rows[j]
            cand_a = {"action": 6, "data": a.data}
            cand_b = {"action": 6, "data": b.data}
            base_a = prior.base_prior.score(a.grid, cand_a)
            base_b = prior.base_prior.score(b.grid, cand_b)
            if abs(base_a - base_b) > 1e-9:
                continue
            degeneracy_pairs_checked += 1
            full_a = prior.score(a.grid, cand_a)
            full_b = prior.score(b.grid, cand_b)
            if abs(full_a - full_b) > 1e-9:
                degeneracy_confirmed_differentiating += 1

    return {
        "game": game,
        "transitions_collected": len(transitions),
        "click_transitions": click_count,
        "hashes_tracked_after": prior.tracked_hash_count,
        "hashes_with_evidence_after": len(evidenced),
        "hashes_with_evidence_and_nonzero_change_rate_after": len(nonzero_change),
        "degeneracy_pairs_checked": degeneracy_pairs_checked,
        "degeneracy_pairs_differentiated_by_history": degeneracy_confirmed_differentiating,
    }


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    explore_budget: int = DEFAULT_EXPLORE_BUDGET,
    total_budget: int = DEFAULT_TOTAL_BUDGET,
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
            "model_specs": MODEL_SPECS,
            "field_principles": FIELD_PRINCIPLES,
            "verifier_is_oracle": False,
            "roster": list(roster),
            "explore_budget": int(explore_budget),
            "total_budget": int(total_budget),
            "min_observations": MIN_OBSERVATIONS,
            "per_game_rows": [],
            "total_click_transitions_observed": 0,
            "total_hashes_tracked": 0,
            "total_hashes_with_evidence_and_nonzero_change_rate": 0,
            "adversarial_degeneracy_check": {"pairs_checked": 0, "pairs_differentiated": 0},
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
            rows.append(
                _measure_one_game(game, explore_budget=explore_budget, total_budget=total_budget)
            )
        except Exception as exc:
            rows.append({"game": game, "error": repr(exc)[:200]})

    measured_rows = [row for row in rows if "error" not in row]
    total_clicks = sum(int(row.get("click_transitions", 0)) for row in measured_rows)
    total_tracked = sum(int(row.get("hashes_tracked_after", 0)) for row in measured_rows)
    total_nonzero = sum(
        int(row.get("hashes_with_evidence_and_nonzero_change_rate_after", 0))
        for row in measured_rows
    )
    total_pairs_checked = sum(int(row.get("degeneracy_pairs_checked", 0)) for row in measured_rows)
    total_pairs_differentiated = sum(
        int(row.get("degeneracy_pairs_differentiated_by_history", 0)) for row in measured_rows
    )

    if not measured_rows:
        verdict = "complete: object_history_salience_prototype_no_games_measured"
    elif total_clicks == 0:
        verdict = "complete: object_history_salience_prototype_no_click_transitions_observed"
    elif total_nonzero > 0:
        verdict = (
            f"complete: object_history_salience_prototype_confirmed_{total_nonzero}_"
            f"hashes_with_real_change_signal_across_{len(measured_rows)}_games"
        )
    else:
        verdict = (
            "complete: object_history_salience_prototype_ran_but_no_hash_cleared_evidence_floor"
        )

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "roster": list(roster),
        "explore_budget": int(explore_budget),
        "total_budget": int(total_budget),
        "min_observations": MIN_OBSERVATIONS,
        "per_game_rows": rows,
        "total_click_transitions_observed": int(total_clicks),
        "total_hashes_tracked": int(total_tracked),
        "total_hashes_with_evidence_and_nonzero_change_rate": int(total_nonzero),
        "adversarial_degeneracy_check": {
            "pairs_checked": int(total_pairs_checked),
            "pairs_differentiated": int(total_pairs_differentiated),
        },
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
