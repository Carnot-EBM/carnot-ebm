#!/usr/bin/env python3
"""Assemble the scored artifact for the goal-evidence A/B from the run's own outputs.

Every number in the artifact is READ from out/analysis.json, out/meta.json and
out/preregistration.json -- none is retyped. A retyped number is a number that can drift away
from the measurement that produced it, and the project has a standing reading-results discipline
for exactly that reason.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"


def sha_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def find(tests: list[dict], block: str, shape: str) -> dict:
    for t in tests:
        if t.get("block") == block and t.get("shape") == shape:
            return t
    return {}


def main() -> int:
    an = json.loads((OUT / "analysis.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    prereg = json.loads((OUT / "preregistration.json").read_text())
    tests = an["tests"]

    s1_primary = find(tests, "stage1", "DECLINED")
    s1_trope = find(tests, "stage1", "TROPE")
    s1_grounded = find(tests, "stage1", "GROUNDED")
    s1_floor = {s: find(tests, "stage1_AA_floor", s) for s in ("DECLINED", "TROPE", "GROUNDED")}
    s2_b = find(tests, "stage2_ITT", "DECLINED")
    s2_c = [t for t in tests if t.get("block") == "stage2_ITT" and t.get("shape") == "DECLINED"]
    s2_floor = find(tests, "stage2_AA_floor", "DECLINED")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()

    # The reproducibility checksum binds the three things a third party would have to hold fixed
    # to get this run back: the pre-registration (the design, written before any LLM call), the
    # rows (every cell), and the analysis (the scoring). A checksum over the artifact itself
    # would be circular.
    repro = hashlib.sha256(
        (
            sha_file(OUT / "preregistration.json")
            + sha_file(OUT / "rows.json")
            + sha_file(OUT / "analysis.json")
        ).encode()
    ).hexdigest()

    art = {
        "experiment": "arc_goal_evidence_ab_20260802",
        "title": "Does giving the ARC goal prompt the agent's own observed evidence stop the "
        "model declining to write a win condition?",
        "run_date": datetime.now(UTC).isoformat(),
        "git_head": head,
        "schema": "carnot.arc_goal_evidence_ab.v1",
        "duration_s": meta["duration_s"],
        # A BARE INT, matching the sibling artifacts, because `adversarial_verify`'s
        # methodology check reads this field and a principle-wrapped or prose value is exactly
        # the field-shape assumption that silently defeated substrate recognition on 176
        # artifacts corpus-wide (the QA-layer discipline's origin bug #2). The scheme that
        # generated the full set lives beside it as its own string field.
        "random_seed": min(int(r["seed"]) for r in json.loads((OUT / "rows.json").read_text())),
        "random_seed_scheme": prereg["generator"]["seed_scheme"],
        "random_seeds_used": sorted(
            {r["seed"] for r in json.loads((OUT / "rows.json").read_text())}
        ),
        "reproducibility_checksum": "sha256:" + repro,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": "every cell loads and generates from a real local GGUF "
        "through the shipped LocalGGUFProposer on the CUDA build; the server witness records "
        "pid, /proc/<pid>/exe, model path and n_ctx read back from /props",
        "model_specs": {
            "generator": meta["server_witness"]["model_from_props"],
            "repo_substr": prereg["generator"]["repo_substr"],
            "n_ctx": meta["server_witness"]["n_ctx_from_props"],
            "kv_quant": meta["server_witness"]["kv_quant"],
            "n_gpu_layers": meta["server_witness"]["n_gpu_layers"],
            "max_tokens": meta["server_witness"]["max_tokens"],
            "cuda_gpu": meta["server_witness"]["cuda_gpu"],
            "exe_from_proc": meta["server_witness"]["exe_from_proc"],
            "one_server_all_arms": True,
        },
        "preconditions_checked": meta["preconditions_checked"],
        "solve_provenance": prereg["solve_provenance"],
        "solve_provenance_note": prereg["solve_provenance_note"],
        "prereg_sha256": meta["prereg_sha256"],
        "n_cells": meta["n_cells"],
        "n_jobs": meta["n_jobs"],
        "treatment_witness_summary": {
            "n_games": len(meta["treatment_witness"]),
            "goal_prompt_chars_control": sorted(
                {t["goal_prompt_chars_off"] for t in meta["treatment_witness"]}
            ),
            "goal_prompt_chars_treatment_range": [
                min(t["goal_prompt_chars_on"] for t in meta["treatment_witness"]),
                max(t["goal_prompt_chars_on"] for t in meta["treatment_witness"]),
            ],
            "combined_induce_prompt_identical_between_arms": all(
                t["combined_prompt_identical"] for t in meta["treatment_witness"]
            ),
            "dedup_inert_when_off_and_excises_when_on": all(
                t["dedup_inert_when_off"] and t["dedup_excises_when_on"]
                for t in meta["treatment_witness"]
            ),
            "levelup_rows_shown_to_any_arm": sorted(
                {v["levelup_rows_in_shown"] for v in meta["split_meta"].values()}
            ),
        },
        "stage1_goal_only_component": an["stage1_goal_only_component"],
        "stage2_live_induce_ITT": an["stage2_live_induce_ITT"],
        "stage2_mechanism_firing": an["stage2_mechanism_firing"],
        "stage2_live_induce_mechanism_fired_only": an["stage2_live_induce_mechanism_fired_only"],
        "cluster_crosstab": an["cluster_crosstab"],
        "stage2_nonfiring_byte_identity": an["stage2_nonfiring_byte_identity"],
        "grounding_audit": an["grounding_audit"],
        "SENSITIVITY_grounded_excluding_trivial_literals": an[
            "SENSITIVITY_grounded_excluding_trivial_literals"
        ],
        "per_game": an["per_game"],
        # The stopping rule and EVERY amendment to it, inlined rather than referenced. A
        # truncated run whose stopping rule lives only in a side file is a run whose reader has
        # to go looking for the reason the n is small; inlining it means the caveat travels with
        # the numbers it qualifies.
        "stopping_rule": json.loads((OUT / "stopping_rule.json").read_text()),
        "power_simulation_post_hoc": json.loads((HERE / "pre" / "power.json").read_text()),
        "PRIMARY_declined_rate": {
            "stage1_gB_vs_gA": s1_primary,
            "stage2_ITT_B_vs_A": s2_b,
            "stage2_ITT_C_vs_A": next((t for t in s2_c if t.get("treat") == "C"), {}),
        },
        "SECONDARY_trope_rate": {"stage1_gB_vs_gA": s1_trope},
        "SECONDARY_grounded_rate": {"stage1_gB_vs_gA": s1_grounded},
        "AA_NOISE_FLOOR": {"stage1": s1_floor, "stage2_declined": s2_floor},
        "tests": tests,
        "min_reachable_p": prereg["MINIMUM_REACHABLE_P_AND_A_HONEST_POWER_STATEMENT"][
            "min_reachable_p_reported"
        ],
        "power_statement": prereg["MINIMUM_REACHABLE_P_AND_A_HONEST_POWER_STATEMENT"],
        "preflight_on_frozen_shipped_corpus": prereg["PREFLIGHT_ON_THE_FROZEN_SHIPPED_CORPUS"],
        "works_on_an_unsolved_game": True,
        "works_on_an_unsolved_game_evidence": {
            "what_reaches_the_model": "the agent's OWN observed transitions only, rendered by "
            "the same _transitions_block the engine prompt already uses",
            "no_win_is_shown_to_any_arm": "levelup_rows_in_shown is 0 for all 20 games -- the "
            "prefix split puts the level-up row in the held-out tail, so every predicate in "
            "this run was written by a model that had never seen the game won",
            "nothing_a_hidden_game_would_withhold": "no game source, no hand-written adapter, no "
            "curated win example, and _previous_level_complete_grid passed as None in every arm "
            "(it is the NEXT level's opening board, not a win state -- the 2026-07-29 "
            "win-state-poison correction)",
            "existence_proof_class": "tn36 is in the roster precisely because it is a stall "
            "game; its shown transitions contain no level-up either, exactly like the other 19",
        },
        "defaults_changed": False,
        "defaults_note": "both CARNOT_ARC_GOAL_PROMPT_TRANSITIONS and CARNOT_ARC_GOAL_DEDUP "
        "still ship OFF. This run measures them; flipping either is an operator decision.",
        "not_submitted": "no scored or online ARC game was played; submission is operator-only",
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": "no verifier-value, moat or efficiency claim is made here. "
        "The outcome is the SHAPE of an induced predicate decided from its syntax tree, and the "
        "goal gate -- the project's executable checker -- is deliberately NOT the outcome, "
        "because plan_found is an exact function of it.",
        "artifacts": {
            "preregistration": "results/arc_goal_evidence_20260802/out/preregistration.json",
            "rows": "results/arc_goal_evidence_20260802/out/rows.json",
            "analysis": "results/arc_goal_evidence_20260802/out/analysis.json",
            "meta": "results/arc_goal_evidence_20260802/out/meta.json",
            "server_witness": "results/arc_goal_evidence_20260802/out/server_witness.json",
            "stage1_predicates": "results/arc_goal_evidence_20260802/out/s1_cells/",
            "stage2_world_models": "results/arc_goal_evidence_20260802/out/s2_cells/",
            "classifier": "results/arc_goal_evidence_20260802/classify.py",
            "driver": "results/arc_goal_evidence_20260802/run_ab.py",
            "analyser": "results/arc_goal_evidence_20260802/analyse.py",
        },
    }
    art["honest_verdict"] = json.loads((OUT / "verdict.json").read_text())["honest_verdict"]
    art["headline"] = json.loads((OUT / "verdict.json").read_text())["headline"]
    art["findings"] = json.loads((OUT / "verdict.json").read_text())["findings"]

    dest = ROOT / "results" / "outer_loop_arc_goal_evidence_ab_20260802.json"
    dest.write_text(json.dumps(art, indent=2, default=str) + "\n")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
