#!/usr/bin/env python3
"""Experiment 548: Milestone 2026.04.41 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.41 ran Exps 537-547 covering: teardown fix (RETRO-054),
    live precision benchmarks (RETRO-033/038/055), GRPO EORM retrain, constraint
    addition wire-in, FOVER expansion, JEPA v8 retrain (FR-11), LowRankKAEM cascade,
    internal state probe, AutoRefine templates, and legacy modernization sprint.

    This retro computes aggregate metrics, assesses RETRO closure status, identifies
    new blockers, and provides meta-reflection for milestone 2026.04.42 planning.

    Key outcome: teardown debt finally resolved (RETRO-054 closed after five milestones
    of deferral), but RETRO-033 and RETRO-038 remain open after attempts #10 and #8
    respectively, and six of eleven experiments fell back to synthetic proxies due to
    insufficient live data in the FOVER corpus.

Spec: REQ-INFRA-058, REQ-INFRA-076, SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

# apply_env_autofix MUST be called first, before any other carnot import.
# This ensures CARNOT_FORCE_LIVE and related env vars are set consistently
# before any pipeline code reads them.  The RETRO-053 bug (checking presence
# not value of CARNOT_FORCE_LIVE) was fixed in Exp 526; this call is safe.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate, _utc_now

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 548
TITLE = "Milestone 2026.04.41 Retrospective"
DELIVERABLE = "results/experiment_548_retro_2026_04_41.json"
MILESTONE = "2026.04.41"
SCHEMA = "carnot.operational_retro.v16"

# Cumulative totals from prior milestone (2026.04.40 retro, Exp 536).
# These are the running totals across all milestones, used to compute
# the updated cumulative average so trend charts remain meaningful.
_PRIOR_CUMULATIVE_WALL_TIME_MIN = 4484.0
_PRIOR_CUMULATIVE_EXPERIMENTS = 387

# Experiment result files for this milestone (Exps 537-547).
_MILESTONE_RESULTS = [
    ("537", "results/experiment_537_teardown_fix.json"),
    ("538", "results/experiment_538_live_25q_precision_v9.json"),
    ("539", "results/experiment_539_live_100q_vericot_v8.json"),
    ("540", "results/experiment_540_grpo_eorm_retrain.json"),
    ("541", "results/experiment_541_constraint_addition_live.json"),
    ("542", "results/experiment_542_fover_expansion.json"),
    ("543", "results/experiment_543_jepa_v8_live_retrain.json"),
    ("544", "results/experiment_544_lowrank_kaem_cascade.json"),
    ("545", "results/experiment_545_internal_state_probe.json"),
    ("546", "results/experiment_546_autorefine_constraint_templates.json"),
    ("547", "results/experiment_547_legacy_modernization.json"),
]

# Open RETRO items at the START of milestone .41 (from .40 retro).
# Count is used to compute retro_closure_rate accurately.
_RETROS_OPEN_AT_MILESTONE_START = [
    "RETRO-031",  # partial carry — verify closure status in result files
    "RETRO-033",  # Live 25q precision — attempt #10 in this milestone
    "RETRO-038",  # Live 100q VeriCoT — attempt #8 in this milestone
    "RETRO-049",  # NUP Probe v4 contrastive margin loss redesign
    "RETRO-054",  # ExperimentTemplate teardown debt (proposed in .40 retro)
]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _load_result(path_str: str) -> dict:
    """Load a JSON result file, returning an empty dict if missing or malformed.

    Missing results are recorded as n_missing increments rather than crashing
    the whole retro, so we always produce a complete artifact even if a single
    experiment file is absent.
    """
    path = _REPO_ROOT / path_str
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_retro() -> dict:
    """Load all milestone experiment results and compute aggregate metrics.

    Returns a dict suitable for passing directly to tmpl.build_result().
    All keys are named to match the v16 schema fields exactly.
    """
    results = {exp_id: _load_result(path) for exp_id, path in _MILESTONE_RESULTS}

    # --- Per-experiment status counts ----------------------------------------

    n_experiments = len(_MILESTONE_RESULTS)
    n_completed = sum(1 for r in results.values() if r.get("status") == "success")
    n_timed_out = sum(1 for r in results.values() if r.get("status") == "timed_out")
    n_deferred_to_gpu = sum(
        1 for r in results.values() if r.get("status") == "deferred_to_gpu"
    )
    n_missing = sum(1 for r in results.values() if not r)

    # --- Wall time -----------------------------------------------------------

    # Sum all duration_s values, convert to minutes.
    # Exp 544 (LowRankKAEM) dominated at 1745s = 29 min (70% of total wall time).
    total_wall_time_minutes = round(
        sum(r.get("duration_s", 0.0) for r in results.values()) / 60.0, 3
    )
    average_minutes_per_experiment = round(
        total_wall_time_minutes / n_experiments, 3
    )

    # --- RETRO closure status ------------------------------------------------

    # RETRO-054: ExperimentTemplate teardown debt.  Five milestones of deferral (.36-.40).
    # Exp 537 implemented teardown() + atexit registration. retro_054_resolved=true.
    retro_054_resolved: bool = bool(results["537"].get("retro_054_resolved", False))

    # RETRO-055: env_autofix check for CARNOT_FORCE_LIVE='0' false-negative (new in .41).
    # Exp 538 confirmed fixed by the v9 precision run executing live_gpu mode.
    retro_055_resolved: bool = bool(results["538"].get("retro_055_resolved", False))

    # RETRO-033: Live 25q precision benchmark — attempt #10 via Exp 538.
    # retro_033_closed=false means the accuracy delta was 0 (pipeline == baseline).
    # The benchmark ran successfully in live_gpu mode, but the pipeline provided
    # no measurable improvement over baseline — the constraint is not resolved.
    retro_033_closed: bool = bool(results["538"].get("retro_033_closed", False))

    # RETRO-038: Live 100q VeriCoT+VPRM — attempt #8 via Exp 539.
    # Same outcome: ran in live_gpu mode, signed_improvement=0.0, not resolved.
    retro_038_closed: bool = bool(results["539"].get("retro_038_closed", False))

    # Closure rate: retros closed this milestone / retros open at milestone start.
    # RETRO-054 closed.  RETRO-055 opened AND closed within this milestone (net zero).
    # So 1 of 5 pre-existing open items was closed.
    n_closed_this_milestone = sum([retro_054_resolved])  # only pre-existing closures
    retro_closure_rate = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )

    # --- FR-11 and JEPA v8 ---------------------------------------------------

    # FR-11 requires JEPA to relay on live CoT pairs from the active pipeline run.
    # Exp 543 retrained on data_source='live_fover_expanded' but fr11_live_relay=false
    # because the FOVER corpus only had 24 pairs (20 train, 4 test) — statistically
    # insufficient for reliable relay; the model achieved AUC 0.444 (below random 0.5).
    fr11_live_relay: bool = bool(results["543"].get("fr11_live_relay", False))
    jepa_v8_auc: float = float(results["543"].get("final_auc", 0.0))

    # --- Component wiring status ---------------------------------------------

    # GRPO EORM: Exp 540 trained on 3 synthetic pairs from fover_exp442 (no live pairs
    # available).  AUC jumped from 0.0 to 1.0 — but this is trivially expected with
    # only 3 samples; honest_verdict='synthetic_fallback'.  Not credible for production.
    grpo_eorm_improved: bool = results["540"].get("auc_improvement", 0.0) > 0
    grpo_eorm_honest_verdict: str = results["540"].get("honest_verdict", "unknown")

    # LowRankKAEM: Exp 544 wired lowrank as the default tier (tier_kaem_default='lowrank')
    # and demonstrated speedups of 4-155x depending on variable count.  However,
    # energy_tolerance_within_5pct=False on all tested configurations — the SVD
    # approximation degrades energy accuracy beyond acceptable bounds.  Wired but
    # not accuracy-verified.
    lowrank_kaem_wired: bool = results["544"].get("tier_kaem_default") == "lowrank"
    lowrank_kaem_energy_within_5pct: bool = bool(
        results["544"].get("energy_tolerance_within_5pct", False)
    )

    # Legacy modernization: Exp 547 audited Exps 308, 260, 309, 425, 410.
    # All five were classified 'fully_modern' — they already had env_autofix,
    # watchdog, and ExperimentTemplate.  The slowest-5 recurrence was NOT due to
    # missing modernization; it was due to conductor re-selection of infrastructure
    # experiments during infrastructure sweeps.  The exclusion manifest still needs
    # to be written to prevent re-selection in .42.
    legacy_scripts_modernized: int = len(
        [
            d
            for d in results["547"].get("script_details", [])
            if d.get("classification") == "fully_modern"
        ]
    )

    # --- Headline results per experiment -------------------------------------

    headline_results = {
        "exp537_teardown_fix": (
            "RETRO-054 CLOSED after 5 milestones of deferral. "
            "ExperimentTemplate.teardown() + atexit registration implemented. "
            "Zombie VRAM carryover prevention now in framework."
        ),
        "exp538_live_25q_v9": (
            f"RETRO-055 CLOSED (env_autofix value-check fix confirmed working in live_gpu mode). "
            f"RETRO-033 attempt #10: retro_033_closed=False. "
            f"Pipeline accuracy {results['538'].get('pipeline_accuracy', 'N/A')} == "
            f"baseline {results['538'].get('baseline_accuracy', 'N/A')} "
            f"(signed_improvement=0.0). Live GPU mode confirmed operational."
        ),
        "exp539_live_100q_v8": (
            f"RETRO-038 attempt #8: retro_038_closed=False. "
            f"Pipeline accuracy {results['539'].get('pipeline_accuracy', 'N/A')} == "
            f"baseline {results['539'].get('baseline_accuracy', 'N/A')} "
            f"(signed_improvement=0.0). Wilson CI spans zero — no statistical signal."
        ),
        "exp540_grpo_eorm": (
            f"GRPO EORM retrained on {results['540'].get('n_pairs', 0)} synthetic pairs. "
            f"AUC: {results['540'].get('before_auc', 0):.2f} -> "
            f"{results['540'].get('after_auc', 0):.2f}. "
            f"honest_verdict='{grpo_eorm_honest_verdict}' — not credible for production."
        ),
        "exp541_constraint_addition": (
            f"Constraint addition wire-in: 0 new constraints added. "
            f"fp_rate_delta={results['541'].get('fp_rate_delta', 'N/A')}. "
            f"Pattern carry=59 already exceeded threshold=3 but no new patterns emerged."
        ),
        "exp542_fover_expansion": (
            f"FOVER corpus expanded: {results['542'].get('n_prior_pairs', 0)} prior -> "
            f"{results['542'].get('n_total_pairs', 0)} total pairs. "
            f"honest_verdict='synthetic_fallback' — corpus still too small for live training."
        ),
        "exp543_jepa_v8": (
            f"JEPA v8 retrained on live_fover_expanded ({results['543'].get('n_train_pairs', 0)} train pairs). "
            f"final_auc={jepa_v8_auc:.4f} (BELOW random 0.5). "
            f"fr11_live_relay=False. Insufficient corpus for reliable relay."
        ),
        "exp544_lowrank_kaem": (
            f"LowRankKAEM wired as default tier. Speedup: 4.6x at n_vars=10, 154.7x at n_vars=200. "
            f"energy_tolerance_within_5pct=False on all configs "
            f"(energy_mad_normalized ≈ 0.96-0.99). Wired but not accuracy-verified."
        ),
        "exp545_internal_probe": (
            f"Internal state probe: probe_auc={results['545'].get('probe_auc', 'N/A')}, "
            f"eorm_auc={results['545'].get('eorm_auc', 'N/A')}. "
            f"is_tier2_viable=False. Both classifiers at random baseline on 24-pair corpus."
        ),
        "exp546_autorefine_templates": (
            f"AutoRefine distilled {results['546'].get('n_templates_distilled', 0)} constraint templates "
            f"from {results['546'].get('n_violations_ingested', 0)} violations "
            f"(carry=59, semantic=8). Templates stored; retrieval_verified=True."
        ),
        "exp547_legacy_sprint": (
            f"Legacy audit of 5 scripts (308, 260, 309, 425, 410): all {legacy_scripts_modernized} "
            f"already classified 'fully_modern'. Slowest-5 recurrence was conductor re-selection, "
            f"not missing modernization. Exclusion manifest still needed."
        ),
    }

    # --- New RETRO items discovered this milestone ---------------------------

    new_retro_items = [
        {
            "id": "RETRO-056",
            "title": "JEPA AUC below random on live FOVER corpus",
            "opened_milestone": "2026.04.41",
            "carry_count": 0,
            "description": (
                "Exp 543 produced final_auc=0.444 (below the 0.5 random baseline) after "
                "retraining on 24 FOVER pairs.  The predictor is anti-correlated with "
                "correctness labels.  Root cause: FOVER corpus is too small (24 pairs) and "
                "skewed (carry violations dominate at 59/67 = 88%).  Fix: grow FOVER corpus "
                "to >=100 diverse pairs before the next JEPA retrain."
            ),
            "priority": "high",
        },
        {
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "opened_milestone": "2026.04.41",
            "carry_count": 0,
            "description": (
                "Exp 544 wired LowRankKAEM as the default tier and demonstrated 4-155x "
                "computational speedup.  However, energy_mad_normalized ≈ 0.96-0.99 on all "
                "tested variable counts — far outside the 5% tolerance required for production "
                "use.  Fix: tune SVD rank k (currently implicit), add calibration layer, or "
                "lower the default tier back to full-rank until accuracy is verified."
            ),
            "priority": "high",
        },
        {
            "id": "RETRO-058",
            "title": "Synthetic proxy fallback epidemic — 6 of 11 experiments",
            "opened_milestone": "2026.04.41",
            "carry_count": 0,
            "description": (
                "Six of eleven .41 experiments fell back to synthetic or proxy data: "
                "Exp 540 (GRPO, n_pairs=3 synthetic), Exp 542 (FOVER, synthetic), "
                "Exp 543 (JEPA, live_fover_expanded but only 24 pairs), "
                "Exp 545 (probe, synthetic_proxy), Exp 546 (templates, no label diversity). "
                "The teardown fix (RETRO-054) prevents zombie VRAM but does not inject live "
                "data into the pipeline.  The bottleneck is FOVER corpus size, not GPU access. "
                "Fix: run a dedicated live data collection sprint (>=100 real CoT pairs from "
                "production GSM8K runs) before scheduling any model retraining experiments."
            ),
            "priority": "critical",
        },
        {
            "id": "RETRO-059",
            "title": "Conductor exclusion manifest not written for fully-modern legacy scripts",
            "opened_milestone": "2026.04.41",
            "carry_count": 0,
            "description": (
                "Exp 547 confirmed that Exps 308, 260, 309, 425, 410 are already fully modern. "
                "Their appearance in the slowest-5 is due to conductor re-selection, not "
                "missing infrastructure.  Without an exclusion manifest entry, the conductor "
                "will continue to select these experiments for infrastructure sweeps in .42+. "
                "Fix: add all five to the conductor exclusion manifest before .42 planning."
            ),
            "priority": "medium",
        },
    ]

    # --- Open RETRO items carry-forward + new --------------------------------

    open_retro_items = [
        {
            "id": "RETRO-031",
            "title": "Partial carry — closure status unverified",
            "carry_count": ">=2",
            "action_required": "Verify closure in result files before .42 planning",
        },
        {
            "id": "RETRO-033",
            "title": "Live 25q precision benchmark — 10 attempts, still not closed",
            "carry_count": 10,
            "action_required": (
                "Root cause is not the live_gpu gate (confirmed working by RETRO-055 fix). "
                "Pipeline constraint logic produces zero improvement on GSM8K arithmetic. "
                "Requires fundamental constraint redesign before attempt #11."
            ),
        },
        {
            "id": "RETRO-038",
            "title": "Live 100q VeriCoT+VPRM — 8 attempts, still not closed",
            "carry_count": 8,
            "action_required": (
                "Same root cause as RETRO-033.  Wilson CI spans zero — no statistical signal. "
                "Requires new verifier training data (real live pairs) before attempt #9."
            ),
        },
        {
            "id": "RETRO-049",
            "title": "NUP Probe v4 contrastive margin loss redesign",
            "carry_count": ">=1",
            "action_required": "Confirm closure status via Exp 530 result file inspection",
        },
        {
            "id": "RETRO-056",
            "title": "JEPA AUC below random — FOVER corpus too small",
            "carry_count": 0,
            "action_required": "Grow FOVER corpus to >=100 diverse pairs before next retrain",
        },
        {
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": 0,
            "action_required": "Tune SVD rank k or add calibration layer; revert to full-rank default until fixed",
        },
        {
            "id": "RETRO-058",
            "title": "Synthetic proxy fallback epidemic — live data pipeline starvation",
            "carry_count": 0,
            "action_required": "Dedicated live data collection sprint; >=100 real CoT pairs required",
        },
        {
            "id": "RETRO-059",
            "title": "Conductor exclusion manifest missing for fully-modern legacy scripts",
            "carry_count": 0,
            "action_required": "Add Exps 308, 260, 309, 425, 410 to conductor exclusion manifest",
        },
    ]

    # --- Meta-reflection -----------------------------------------------------

    meta_reflection = {
        "top_3_bottlenecks": [
            (
                "Live data pipeline starvation: the FOVER corpus had only 24 pairs at "
                "milestone close (down from 57 prior pairs in the .40 corpus — Exp 542 "
                "reported n_prior_pairs=57 but n_total_pairs=24, suggesting a data-path "
                "regression). Six of eleven experiments fell back to synthetic proxies. "
                "Teardown fix prevents zombie VRAM but does not solve the upstream problem: "
                "there are not enough real CoT correctness labels to train any model reliably."
            ),
            (
                "RETRO-033 and RETRO-038 have now consumed 18 combined experiment attempts "
                "across multiple milestones without a single positive improvement delta. "
                "The live_gpu gate is confirmed working (RETRO-055 closed). "
                "The root cause is the pipeline itself: constraint-based verification "
                "produces signed_improvement=0.0 on GSM8K arithmetic for current model outputs. "
                "Further live benchmark attempts without first redesigning the constraint logic "
                "are wasted wall time — each attempt costs 2-10 minutes and produces identical results."
            ),
            (
                "LowRankKAEM wired as default with energy accuracy far outside tolerance "
                "(MAD normalized ≈ 0.96-0.99 vs required <=0.05). The speedup is real "
                "(4-155x) but the approximation degrades correctness to an unacceptable degree. "
                "Setting an inaccurate approximation as the DEFAULT tier before accuracy is "
                "verified is a credibility risk: any downstream benchmark that reports "
                "energy scores is now reporting degraded values silently."
            ),
        ],
        "top_3_improvements_for_42": [
            (
                "Run a dedicated live data collection sprint FIRST in .42 before any model "
                "retraining: design an experiment that runs >=100 real GSM8K questions through "
                "the live pipeline, records per-step correctness labels, and writes them to "
                "the FOVER corpus.  Do not schedule Exp 549-550 until this corpus sprint "
                "(budget: 1 experiment, 30-60 min GPU time) is complete.  This unblocks: "
                "JEPA retrain (RETRO-056), GRPO EORM (from 3 to >=30 real pairs), "
                "internal probe (from AUC=0.5 to measurable signal), and all six blocked "
                "experiments that fell back to synthetic proxies this milestone."
            ),
            (
                "Revert LowRankKAEM tier default to full-rank before .42 begins: "
                "write a one-line fix to set tier_kaem_default='full' until energy "
                "accuracy is verified within 5%.  Open a separate experiment in .42 to "
                "tune SVD rank k (try k=32, 64, 128) and find the knee of the "
                "speedup-vs-accuracy curve.  Do not deploy an inaccurate approximation "
                "as the default — it corrupts all downstream energy-based decisions."
            ),
            (
                "Write the conductor exclusion manifest for Exps 308, 260, 309, 425, 410 "
                "as the very first commit of .42, before any research experiment starts. "
                "All five are confirmed fully modern (Exp 547).  Without the manifest, "
                "the conductor will select them again for infrastructure sweeps in .42, "
                "burning ~385 min of recurring overhead for no new information."
            ),
        ],
        "credibility_verdict": (
            "Teardown fix (RETRO-054) is credible — atexit registration is a genuine "
            "infrastructure improvement.  AutoRefine template distillation (Exp 546) is "
            "credible — 2 templates from 67 violations is a real signal.  "
            "All other results are either synthetic fallbacks, zero-improvement live runs, "
            "or accuracy-failed approximations.  No publishable headline result this milestone. "
            "The GRPO AUC jump (0.0->1.0 on 3 pairs) is numerically valid but statistically "
            "meaningless.  The JEPA AUC of 0.444 is the most concerning result: it means the "
            "current verifier is WORSE than random on live data — a negative result that must "
            "be investigated, not ignored."
        ),
        "wall_time_42_estimate_minutes": (
            "If .42 follows the same scope as .41 (11 experiments), the milestone wall time "
            "will be approximately 40-60 minutes of pure execution time.  However, if the "
            "live data collection sprint is scheduled first and requires GPU inference on "
            ">=100 questions (at ~24 s/question based on Exp 538 latency = ~40 min GPU time), "
            "a single data-collection experiment could account for 40-50% of total .42 wall time. "
            "Estimate: 80-120 minutes total if live data collection is included; 40-60 minutes "
            "without it (but without it, all model retraining experiments will again fall back "
            "to synthetic proxies)."
        ),
    }

    return {
        # Schema identification
        "schema": SCHEMA,
        "milestone": MILESTONE,
        # Aggregate counts
        "n_experiments": n_experiments,
        "n_completed": n_completed,
        "n_timed_out": n_timed_out,
        "n_deferred_to_gpu": n_deferred_to_gpu,
        "n_missing": n_missing,
        # Wall time
        "total_wall_time_minutes": total_wall_time_minutes,
        "average_minutes_per_experiment": average_minutes_per_experiment,
        # RETRO status (pre-existing items)
        "retro_054_resolved": retro_054_resolved,
        "retro_055_resolved": retro_055_resolved,
        "retro_033_closed": retro_033_closed,
        "retro_038_closed": retro_038_closed,
        "retro_closure_rate": retro_closure_rate,
        # FR-11 and JEPA
        "fr11_live_relay": fr11_live_relay,
        "jepa_v8_auc": jepa_v8_auc,
        # Component wiring
        "grpo_eorm_improved": grpo_eorm_improved,
        "grpo_eorm_honest_verdict": grpo_eorm_honest_verdict,
        "lowrank_kaem_wired": lowrank_kaem_wired,
        "lowrank_kaem_energy_within_5pct": lowrank_kaem_energy_within_5pct,
        "legacy_scripts_modernized": legacy_scripts_modernized,
        # Narrative
        "headline_results": headline_results,
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "meta_reflection": meta_reflection,
        # Verdict
        "honest_verdict": "milestone_partial",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and write the Exp 548 milestone retro artifact."""
    # Watchdog: abort if the retro script itself hangs (unlikely, but defensive).
    # The 20-minute limit is generous — pure JSON computation should complete in <5 s.
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    retro_data = compute_retro()

    artifact = tmpl.build_result(retro_data, status="success")
    # build_result() overwrites "schema" with a sorted key list.  Replace it
    # with the named schema identifier AFTER the call so downstream tests can
    # assert the correct schema version string.
    artifact["schema"] = SCHEMA
    artifact["env_autofix"] = True  # applied at module top via apply_env_autofix()

    deliverable_path = _REPO_ROOT / DELIVERABLE
    deliverable_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()
    print(f"[Exp {EXP_ID}] Deliverable written: {DELIVERABLE}")
    print(f"[Exp {EXP_ID}] honest_verdict={artifact['honest_verdict']}")
    print(f"[Exp {EXP_ID}] retro_054_resolved={artifact['retro_054_resolved']}")
    print(f"[Exp {EXP_ID}] retro_033_closed={artifact['retro_033_closed']}")
    print(f"[Exp {EXP_ID}] retro_038_closed={artifact['retro_038_closed']}")
    print(f"[Exp {EXP_ID}] fr11_live_relay={artifact['fr11_live_relay']}")


if __name__ == "__main__":
    main()
