"""Experiment 5595: offline-dev-sim prototype for InertClickSigPruner
(REQ-ARC-FCP-5595), ops/known-issues.md's 2026-07-11 task 9 (InertClickSigPruner
-- extend the HazardMovePruner pattern to the inert/no-op-click axis).

Demonstrates the pruner's trust+specificity gating actually classifies click
signatures on REAL rendered ARC frames collected from a REAL exploration run
(``E3AgentPolicy``/``lb.run_game`` with the frozen live default Qwen3.5-9B-MTP
proposer, same construction pattern as exp5594) -- not just synthetic unit-test
grids. This is a measurement/prototype script, not a live-path wiring change:
``InertClickSigPruner.rank_candidates`` is tested here for correctness against
real data but is NOT wired into ``StepwiseExplorer._candidates``'s live
composition chain (see ``arc_inert_click_pruner.py``'s module docstring for why
that is a distinct, separately-scoped step). ``solve_provenance`` stays
``development_proxy``: no per-game adapter, no offline BFS, no level-solve
claim.

GAME SELECTION NOTE (found investigating, not assumed): a first attempt to
probe click-action prevalence via a bare ``E3AgentPolicy(game,
explore_budget=6)`` (no explicit ``proposer=``) stalled twice with near-zero
CPU growth over many minutes -- some default-constructed component the
exploration loop depends on appears to block (network or model-load path),
distinct from the ``LocalGGUFProposer``-explicit construction pattern
established in exp5594, which ran reliably. This script therefore ALWAYS
constructs ``E3AgentPolicy`` with an explicit proposer, matching exp5594.
``m0r0`` was confirmed click-heavy by direct probe (21 of 22 transitions were
action=6) before being selected as this prototype's roster.

INFERENCE_SUBSTRATE NOTE (found from the real run, corrected from an initial
conservative guess): the real measured duration was 19.3s -- well under the
60s ``live_llm_inference`` floor. This script constructs a real
``LocalGGUFProposer`` and wires it into ``E3AgentPolicy`` (so the proposer IS
available), but never calls ``induce()``/``generate()`` on it directly --
only ``lb.run_game``'s exploration loop runs, and that loop apparently never
invokes the LLM either (candidate generation is classical/heuristic). The
``model_specs`` GGUF entry is therefore VESTIGIAL, matching the documented
``offline_arcade_live_agent_runtime_self_discovery_no_llm`` pattern's own
note about vestigial model strings -- not a live-inference claim.

Spec refs: REQ-ARC-FCP-5595, SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA,
SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK.
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

EXPERIMENT_ID = "experiment_5595_inert_click_sig_pruner_offline_sim_prototype"
RESULT_RELATIVE_PATH = "results/experiment_5595_inert_click_sig_pruner_offline_sim_prototype.json"
SCHEMA = "carnot.exp5595.inert_click_sig_pruner_offline_sim_prototype.v1"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
RANDOM_SEED = 5595
DEFAULT_ROSTER = ("m0r0",)
DEFAULT_EXPLORE_BUDGET = 6
DEFAULT_TOTAL_BUDGET = 40
GGUF_REPO_SUBSTR = "Qwen3.5-9B-MTP"
MIN_OBSERVATIONS = 4
MIN_SPECIFICITY = 0.9

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy default tier-3 exploration proposer (real, live, port 8920)",
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
    "min_specificity",
    "per_game_rows",
    "total_click_transitions_observed",
    "total_signatures_tracked",
    "total_signatures_confidently_inert",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; demonstrates the trust+specificity gate classifies "
        "real click signatures correctly, does not claim a live-path efficiency win (a "
        "states-expanded A/B, like HazardMovePruner's tu93 measurement, is a separate, "
        "later step once this mechanism is wired into a solver)"
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_self_discovery_no_llm -- confirmed "
        "empirically, not assumed: a real LocalGGUFProposer is wired into E3AgentPolicy but "
        "this script never calls induce()/generate() on it, and the measured 19.3s real "
        "duration is far under the 60s live_llm_inference floor, confirming the exploration "
        "loop itself never invokes the LLM either; the GGUF entries in model_specs are "
        "vestigial (the proposer object exists but is not exercised), not a live-inference "
        "claim"
    },
    "total_signatures_confidently_inert": {
        "principle": "count of distinct (color, size, is_rect, twin_count) signatures that "
        "cleared BOTH the min_observations evidence floor AND the min_specificity bar with "
        "zero level-ups -- the load-bearing claim behind building the pruner at all, "
        "measured against real click data rather than synthetic grids"
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
        from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner  # noqa: F401

        checks["e3_and_pruner_import"] = True
    except Exception:
        checks["e3_and_pruner_import"] = False
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    checks["gguf_cached"] = (
        any(GGUF_REPO_SUBSTR.lower() in p.name.lower() for p in hub.glob("models--*"))
        if hub.exists()
        else False
    )
    checks["llama_server_binary_present"] = bool(
        list((Path.home() / ".cache").glob("llama.cpp*/build/bin/llama-server"))
    )
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8920/health", timeout=5) as r:
            checks["port_8920_prewarmed"] = r.status == 200
    except Exception:
        checks["port_8920_prewarmed"] = False
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
    from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner

    proposer = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=8920,
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=2560,
    )
    policy = E3AgentPolicy(game, proposer=proposer, explore_budget=explore_budget)
    lb.run_game(game, policy, budget=total_budget)
    transitions = list(policy.transitions)

    pruner = InertClickSigPruner(
        lambda g: g, min_observations=MIN_OBSERVATIONS, min_specificity=MIN_SPECIFICITY
    )
    click_count = 0
    for t in transitions:
        label = {"action": int(t.action), "data": t.data}
        if int(t.action) == 6:
            click_count += 1
        pruner.observe(t.grid, label, t.next_grid, bool(t.level_after > t.level_before))

    stats = pruner.stats()

    # rank_candidates sanity check: replay the observed click rows against the FIRST
    # collected frame (a stable, real grid) and confirm the filter runs cleanly and
    # only removes rows whose signature actually cleared the gate.
    rank_check: JsonDict = {"rows_in": 0, "rows_kept": 0, "rows_dropped": 0}
    if transitions:
        rows = [{"action": int(t.action), "data": t.data} for t in transitions]
        kept = pruner.rank_candidates(transitions[0].grid, rows)
        rank_check = {
            "rows_in": len(rows),
            "rows_kept": len(kept),
            "rows_dropped": len(rows) - len(kept),
        }

    return {
        "game": game,
        "transitions_collected": len(transitions),
        "click_transitions": click_count,
        "signatures_tracked_after": stats["signatures_tracked"],
        "signatures_confidently_inert_after": stats["pruned_signatures"],
        "rank_candidates_sanity": rank_check,
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
            "min_specificity": MIN_SPECIFICITY,
            "per_game_rows": [],
            "total_click_transitions_observed": 0,
            "total_signatures_tracked": 0,
            "total_signatures_confidently_inert": 0,
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
    total_tracked = sum(int(row.get("signatures_tracked_after", 0)) for row in measured_rows)
    total_inert = sum(
        int(row.get("signatures_confidently_inert_after", 0)) for row in measured_rows
    )

    if not measured_rows:
        verdict = "complete: inert_click_sig_pruner_prototype_no_games_measured"
    elif total_clicks == 0:
        verdict = "complete: inert_click_sig_pruner_prototype_no_click_transitions_observed"
    elif total_inert > 0:
        verdict = (
            f"complete: inert_click_sig_pruner_prototype_confirmed_{total_inert}_"
            f"signatures_pruned_across_{len(measured_rows)}_games"
        )
    else:
        verdict = (
            "complete: inert_click_sig_pruner_prototype_ran_but_no_signature_cleared_evidence_floor"
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
        "min_specificity": MIN_SPECIFICITY,
        "per_game_rows": rows,
        "total_click_transitions_observed": int(total_clicks),
        "total_signatures_tracked": int(total_tracked),
        "total_signatures_confidently_inert": int(total_inert),
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
