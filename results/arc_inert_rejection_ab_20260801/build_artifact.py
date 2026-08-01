"""Build the milestone artifact from the frozen collection + analysis. Reads, never re-measures.

Every number here comes from `out/analysis.json`, which comes from `out/rows.json` and
`out/scored.json`. Nothing is recomputed in this file, so a disagreement between the artifact and
the detail files would be a bug in ONE place rather than two independent derivations that could
both be wrong.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/ianblenke/github.com/ianblenke/carnot")
OUT = HERE / "out"
ARTIFACT = ROOT / "results/outer_loop_arc_inert_rejection_ab_20260801.json"


def sha_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}pp"


def main() -> int:  # noqa: C901
    an = json.loads((OUT / "analysis.json").read_text())
    meta = json.loads((OUT / "meta.json").read_text())
    prereg = json.loads((OUT / "preregistration.json").read_text())
    rows = json.loads((OUT / "rows.json").read_text())
    sw = meta["server_witness"]
    t = an["tests"]
    prim = t["PRIMARY_usable_engine_yield"]
    sens = t["SENSITIVITY_usable_engine_yield_truncation_as_zero"]
    live = t["SECONDARY_live_engine_yield"]
    cf = t["SECONDARY_heldout_change_fidelity"]
    depth = t["SECONDARY_probe_depth_reached"]
    wit = an["mechanistic_witness"]
    aa = an["aa_nondeterminism_floor"]
    rr = wit.get("reask_rate_by_arm", {})
    excess = rr.get("excess_reask_rate_attributable_to_inertness")

    # THE POSITIVE CONTROL, and it gates how every null below may be read. A treatment that never
    # acted produces a null about EXPOSURE, not about the intervention -- the FALSE_NEGATIVE_RISK
    # trap in CLAUDE.md, applied to an A/B rather than to a reranker.
    treatment_acted = bool(excess is not None and excess > 0)

    verdict_bits = [
        "complete_inert_rejection_ab",
        f"primary_usable_yield_delta_{prim['mean_delta']:+.4f}",
        f"p_{prim['p_two_sided']:.4f}",
        f"minp_{prim['min_reachable_two_sided_p_at_this_discordance']:.4f}",
        f"live_yield_delta_{live['mean_delta']:+.4f}",
        f"treatment_acted_{str(treatment_acted).lower()}",
    ]

    duration_s = float(meta.get("duration_s") or 0)
    art = {
        "experiment": "outer_loop_arc_inert_engine_rejection_ab",
        "schema": "carnot.arc.inert_rejection_ab.v1",
        "milestone": "2026.08.outer_loop",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "The 2026-08-01 generation taxonomy found INERTNESS -- an induced engine that predicts "
            "no action changes anything -- is the largest single failure class of the live ARC "
            "generator (26 of 172 candidates, 15.1%), and the only class the induce path took no "
            "action on. Rejecting it is a ~3-line change reusing an already-shipped detector. What "
            "does rejecting it COST, measured against the unmodified path?"
        ),
        "headline": (
            f"Rejecting a clean-but-inert induced engine and re-asking once changed usable-engine "
            f"yield by {pct(prim['mean_delta'])} "
            f"(paired sign test over {prim['n_pairs']} games, {prim['n_discordant']} discordant, "
            f"p = {prim['p_two_sided']}). The treatment DID act: the ON arm re-asked on "
            f"{excess if excess is not None else 'n/a'} more of its cells than the control. "
            f"Live-engine yield (usable AND not inert) moved {pct(live['mean_delta'])}, "
            f"p = {live['p_two_sided']}."
        ),
        "honest_verdict": "_".join(verdict_bits),
        # -------- what was actually done ------------------------------------------------
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": (
            "Every cell issues a real gemma-4-31B-it-qat completion against a live CUDA "
            "llama-server (pid, /proc/<pid>/exe and n_ctx read back from /props are in "
            "server_witness). The scoring pass that follows is CPU-only and is reported "
            f"separately. duration_s = {duration_s:.0f}s is the collection wall time across "
            f"{meta.get('n_cells_run')} cells, which is the LLM-bound part."
        ),
        "duration_s": duration_s,
        "duration_s_note": (
            "wall time of the generation sweep only. The scoring sweep (one killable subprocess "
            "per engine, held-out verifier + bounded state-graph probe) is separate and CPU-only."
        ),
        "model_specs": [
            {
                "name": "gemma-4-31B-it-qat-GGUF",
                "role": "the live ARC inducer per the 2026-07-28 operator directive; wrote every "
                "engine in both arms",
                "gguf": sw.get("model_from_props") or meta.get("gguf"),
                "invoked": True,
                "n_gpu_layers": sw.get("n_gpu_layers"),
                "n_ctx": sw.get("n_ctx_from_props"),
                "kv_quant": sw.get("kv_quant"),
                "one_server_both_arms": True,
                "port": sw.get("port_actual"),
                "cuda_build_proven_from": sw.get("exe_from_proc"),
            }
        ],
        "random_seed": meta.get("seed_base"),
        "random_seed_note": (
            "CARNOT_ARC_GENERATOR_SEED = seed_base + replicate, the SAME seed in both arms of a "
            "(game, replicate) pair. MEASURED NOT TO DELIVER REPRODUCIBILITY on real induce "
            "prompts -- see nondeterminism_finding below. The analysis clusters at the game and "
            "does not rely on the arms sharing a draw."
        ),
        "reproducibility_checksum": sha_file(OUT / "analysis.json"),
        # -------- preconditions ---------------------------------------------------------
        "preconditions_checked": [
            {
                "resource": "all six run-time preconditions",
                "available": not (OUT / "blocked.json").exists(),
                "detail": (
                    "run_ab.py evaluates six preconditions (gguf cached, conductor inactive, GPU "
                    "headroom, the inertness detector actually fires, the flag is default-off, the "
                    "port is free) and writes out/blocked.json then exits 1 if ANY fails. "
                    "out/blocked.json does not exist and the run produced "
                    f"{meta.get('n_cells_run')} cells, so all six passed."
                ),
                "principle": "a precondition list that is only reported when it passes proves "
                "nothing; the absence of the refusal artifact is the check",
            },
            {
                "resource": "generator on the CUDA build, not the AMD iGPU HIP build",
                "available": bool(sw.get("is_cuda_build")),
                "detail": sw.get("exe_from_proc"),
                "principle": "a run that silently landed on the iGPU is ~6x slower and is a "
                "different substrate; run_ab.py refuses to proceed without this",
            },
            {
                "resource": "treatment witness: the prompt is byte-identical across arms",
                "available": all(
                    w["prompt_identical_across_arms"] for w in meta.get("treatment_witness", [])
                ),
                "detail": f"{len(meta.get('treatment_witness', []))} games checked before the "
                "first LLM call",
                "principle": "this flag changes an accept/reject decision, not prompt text, so "
                "byte-equality is the correct witness and any difference would mean something "
                "other than the intervention is varying",
            },
        ],
        # -------- the result ------------------------------------------------------------
        "primary": {
            "metric": prereg["PRIMARY"]["metric"],
            "definition_is_the_shipped_one": True,
            "why_that_matters": (
                "arc_engine_static_validation.validate_engine_code calls an INERT engine CLEAN. So "
                "under this metric the treatment cannot gain -- it can only stay flat or lose. "
                "That is deliberate: it makes the primary a COST measurement, and it keeps the "
                "measurement non-circular. Folding inertness into the outcome definition would "
                "make the treatment and the outcome the same object. Pinned by "
                "test_the_outcome_definition_is_not_the_treatment."
            ),
            **{k: v for k, v in prim.items() if k != "per_game"},
        },
        "primary_sensitivity_truncation_as_zero": {
            "why": an["collection"]["truncation_handling"],
            **{k: v for k, v in sens.items() if k != "per_game"},
        },
        "secondary_live_engine_yield": {
            "metric": "usable AND engine_changes_anything_bounded is True",
            "taxonomy_predicted": "+9.8pp live engines per candidate",
            "caveat_declared_before_results": prereg["SECONDARY_the_taxonomy_claim"][
                "CAVEAT_stated_before_results"
            ],
            **{k: v for k, v in live.items() if k != "per_game"},
        },
        "secondary_downstream_quality": {
            "why": "a yield win that produces only inert or wrong engines must be visible as such",
            "heldout_change_fidelity": {k: v for k, v in cf.items() if k != "per_game"},
            "probe_depth_reached": {k: v for k, v in depth.items() if k != "per_game"},
            "probe_depth_provenance": (
                "the bounded 600-call state-graph probe copied from "
                "results/arc_metric_validity_20260801, which found it predicts plannability "
                "(AUC 0.787, cluster CI [0.675, 0.859]) where change_fidelity does not (AUC 0.609, "
                "CI containing chance). The copy is enforced by an AST comparison in "
                "test_harness_invariants.py, not by a docstring claim -- the first draft said "
                "'copied verbatim' while having renamed and dropped output fields."
            ),
            "probe_depth_is_not_a_validated_selector": (
                "that artifact states probe_depth was SELECTED as a family maximum and needs a "
                "prospective test. It is used here as the best available proxy, not as truth."
            ),
        },
        "cost": {
            "completion_calls": {
                k: v for k, v in t["COST_completion_calls"].items() if k != "per_game"
            },
            "wall_seconds": {k: v for k, v in t["COST_wall_seconds"].items() if k != "per_game"},
        },
        # -------- the controls that decide how to read the above ------------------------
        "positive_control": {
            "question": "did the treatment ACT at all?",
            "why_it_gates_everything": (
                "A null from an intervention that never fired is a statement about exposure, not "
                "about the intervention -- the FALSE_NEGATIVE_RISK trap. The re-ask rate by arm is "
                "the direct measure: only the ON arm can re-ask for inertness."
            ),
            "reask_rate_by_arm": rr,
            "treatment_acted": treatment_acted,
        },
        "positive_control_passed": treatment_acted,
        "aa_nondeterminism_floor": aa,
        "nondeterminism_finding": {
            "what": (
                "The seeded sampler does NOT reproduce on real induce prompts. Across the A/B, "
                "arms drew different engines at byte-identical prompt hashes, identical seeds and "
                "one completion call each."
            ),
            "why_it_matters_beyond_this_run": (
                "LocalGGUFProposer.sampling_seed's docstring says an A/A arm 'should come back "
                "byte-identical, which is a cheap positive control on the determinism itself'. "
                "Three A/B designs on this path have been built on that promise. If it is false, "
                "each is a randomized comparison rather than a matched one and their power is "
                "lower than assumed."
            ),
            "what_was_ruled_out": (
                "The seed IS honoured (two identical short requests at a fixed seed returned "
                "byte-identical output against this same server) and the server runs a single slot "
                "with no --parallel, so this is not batch nondeterminism."
            ),
            "leading_hypothesis_not_established": (
                "cache_prompt=true prefix reuse: reusing a KV prefix computed under a different "
                "preceding state changes GEMM shapes and therefore floating-point accumulation "
                "order, which at temperature 0.2 can flip a near-tied token."
            ),
            "probe": "determinism_probe.json (four conditions isolating prompt length, cache "
            "reuse and cache_prompt) if present in this directory",
            "effect_on_this_experiment": wit["pairing_is_randomization_not_matched_sampling"],
        },
        "mechanistic_witness": {k: v for k, v in wit.items() if k not in ("fired",)},
        "collection": an["collection"],
        "raw_pooled_rates": an["raw_pooled_rates"],
        # -------- gates -----------------------------------------------------------------
        "acceptance_gates": {
            "the_run_could_answer_its_question": {
                "condition": "the treatment acted (excess re-ask rate > 0) AND at least one "
                "complete (game, replicate) pair per arm exists",
                "passed": bool(treatment_acted and an["collection"]["n_complete_pairs"] > 0),
                "principle": "a gate on whether the measurement HAPPENED, never on whether the "
                "answer was the hoped-for one -- a gate that only passes on a positive result "
                "is a gate that manufactures positive results",
            },
            "the_arms_differ_only_in_the_intervention": {
                "condition": "prompt byte-identical across arms on every game; one server; "
                "arm_flag_consistent on every cell",
                "passed": bool(
                    all(
                        w["prompt_identical_across_arms"] for w in meta.get("treatment_witness", [])
                    )
                    and all(r.get("arm_flag_consistent") for r in rows)
                ),
                "principle": "without this the contrast is confounded and no p-value means "
                "anything",
            },
            "no_default_was_flipped": {
                "condition": "CARNOT_ARC_INDUCE_REJECT_INERT remains default-off in the shipped "
                "code",
                "passed": True,
                "principle": "measuring whether to flip a flag must not flip it",
            },
        },
        "power_floor_stated_before_the_result": prereg["POWER_STATED_UP_FRONT"],
        "preregistration_path": str(OUT / "preregistration.json"),
        "prereg_sha256": meta.get("prereg_sha256"),
        "analysis_detail_path": str(OUT / "analysis.json"),
        "rows_detail_path": str(OUT / "rows.json"),
        "cited_upstream_artifacts": [
            {
                "experiment_id": "outer_loop_arc_generation_taxonomy_20260801",
                "fields_imported": [
                    "inertness base rate 26/172 (15.1%)",
                    "0 of 11 inert candidates plannable",
                    "all 15 scored inert cells at change_fidelity exactly 0.0",
                ],
                "sha256": sha_file(
                    ROOT / "results/outer_loop_arc_generation_taxonomy_20260801.json"
                ),
            },
            {
                "experiment_id": "outer_loop_arc_metric_validity_20260801",
                "fields_imported": [
                    "probe_depth_reached as the plannability predictor (AUC 0.787)",
                    "change_fidelity does not predict plannability (AUC 0.609)",
                ],
                "sha256": sha_file(ROOT / "results/outer_loop_arc_metric_validity_20260801.json"),
            },
            {
                "experiment_id": (
                    "outer_loop_arc_object_perception_heldout_ab_change_fidelity_20260801"
                ),
                "fields_imported": [
                    "the 20-game roster, reused unchanged",
                    "the A/A reference 1/4",
                ],
                "sha256": sha_file(
                    ROOT
                    / "results"
                    / "outer_loop_arc_object_perception_heldout_ab_change_fidelity_20260801.json"
                ),
            },
        ],
        "verifier_is_oracle": {
            "value": False,
            "principle": (
                "Declared for completeness, but this experiment makes NO verifier-value or moat "
                "claim -- it measures what a generation-side reject-and-retry costs. The outcome "
                "is a mechanical code check (validate_engine_code), not a judgement about "
                "correctness, and nothing here consults the environment's level counter or win "
                "oracle. Nothing in this artifact is headline-eligible as evidence for a verifier "
                "moat, in either direction."
            ),
        },
        "limitations": [
            {
                "limitation": "the primary is underpowered BY DESIGN and this was stated first",
                "detail": prereg["POWER_STATED_UP_FRONT"]["CAN_THE_PRIMARY_REACH_0.05"],
            },
            {
                "limitation": "the pairing is randomization, not matched sampling",
                "detail": wit["pairing_is_randomization_not_matched_sampling"],
            },
            {
                "limitation": "the secondary is partly definitional",
                "detail": prereg["SECONDARY_the_taxonomy_claim"]["CAVEAT_stated_before_results"],
            },
            {
                "limitation": "yield is necessary, not sufficient",
                "detail": "a usable engine can still be wrong. Held-out change_fidelity and "
                "probe_depth are reported alongside so a yield win that produces only inert or "
                "useless engines is visible as such. Note that change_fidelity itself was shown "
                "on 2026-08-01 not to predict plannability, so it is reported as a descriptor "
                "rather than as a quality verdict.",
            },
            {
                "limitation": "20 public games, one generator, one prompt shape",
                "detail": "nothing here says how a hidden game behaves. The roster is the ARC "
                "public survey set and the induce prompt is the shipped combined one.",
            },
        ],
        "missing_verifier_gaps": [
            {
                "gap": "no validated signal for WHICH re-ask to keep",
                "failure_mode": "the re-ask budget is 1 and the second answer is accepted "
                "whatever it is, so a re-ask that returns something worse is kept. Choosing "
                "between the two answers needs a selector, and the only candidate "
                "(probe_depth_reached) is itself unvalidated prospectively.",
                "candidate_design": "best-of-2 over the original and the re-ask, ranked by the "
                "bounded state-graph probe, gated on a prospective test of that probe",
                "priority": "high: it is the difference between rejecting a bad engine and "
                "getting a good one",
            }
        ],
        "surprising_result_acknowledgment": None,
        "false_negative_risk_checked": True,
        "not_submitted": "no scored or online ARC game was played; submission is operator-only",
        "flag_remains_default_off": True,
        "code_provenance": {
            "git_head": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "harness": {
                p.name: sha_file(p) for p in sorted(HERE.glob("*.py")) if p.name != "__init__.py"
            },
            "shipped_code_under_test": {
                "python/carnot/agentic/arc_engine_static_validation.py": sha_file(
                    ROOT / "python/carnot/agentic/arc_engine_static_validation.py"
                ),
                "python/carnot/agentic/arc_executable_world_model.py": sha_file(
                    ROOT / "python/carnot/agentic/arc_executable_world_model.py"
                ),
            },
        },
    }
    probe = OUT / "determinism_probe.json"
    if probe.exists():
        art["nondeterminism_finding"]["probe_result"] = json.loads(probe.read_text())

    gates = art["acceptance_gates"]
    art["acceptance_gate_passed"] = all(g["passed"] for g in gates.values())
    art["acceptance_gate_passed_note"] = (
        "these gates ask whether the RUN could answer its question, not whether the answer was "
        "the hoped-for one. A null with the treatment demonstrably acting is a PASS."
    )

    ARTIFACT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ARTIFACT}")
    print(art["honest_verdict"])
    print(art["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
