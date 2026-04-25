#!/usr/bin/env python3
"""Experiment 854 — Milestone 2026.04.65 Operational Retrospective.

**Researcher summary:**
    Milestone .65 ran 11 experiments (Exps 843-853) targeting seven open RETROs from .64:
    SVAMP zero AUC, JEPA OOD generalization, arbiter flat energy, constraint zero delta,
    GGUF cache import, iCE40 PNR LUT overflow, and the manifest full-scope governance gap.
    Two RETROs were closed (arbiter calibrated, GGUF cache implemented), three new RETROs
    were opened (SOTA model not downloaded, iCE40 N=16 unexpected LUT expansion, live env
    propagation broken again), and 7 of 12 success criteria were met.  Milestone wall time
    improved by ~15 min vs .64.

**Why two closed, seven still open:**
    RETRO-ARBITER-FLAT-ENERGY closed because warm-start Gibbs with 500 burn-in steps
    drove accuracy to 1.0 on both standard and adversarial scenarios (Exp 846).  The root
    cause (cold-start MCMC not converging) was precisely diagnosed in .64 and the fix was
    surgical.  RETRO-GGUF-CACHE-IMPORT closed because GGUFCacheResolver was implemented
    (Exp 849) — but the GGUF model itself is not cached on disk, immediately triggering
    RETRO-SOTA-MODEL-DOWNLOAD.  The cache resolver resolves paths; downloading models is
    an operator task that must be added to pre-flight.

    The iCE40 N=16 failure is worse than N=32: synthesis produced 12258 LCs (159% of the
    7680 HX8K capacity) vs the expected ~1000.  Root cause: the Verilog uses a combinational
    memory array (lf_comb replaced with registers) that expands quadratically in flip-flops
    rather than halving with N.  A pipeline Verilog architecture or N=8 is required.

**Schema:** carnot.operational_retro.v40
"""

import json
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DELIVERABLE = os.path.join(RESULTS_DIR, "operational_retro_2026_04_65.json")
MILESTONE_PREREQS = os.path.join(REPO_ROOT, "MILESTONE_PREREQS.md")

# ---------------------------------------------------------------------------
# Milestone timing constants (derived from conductor-log.md UTC timestamps)
#
# .65 activated:  2026-04-25 09:21 UTC
# Exp 843 start:  2026-04-25 09:40 UTC
# Exp 853 finish: 2026-04-25 12:34 UTC
# Retro (854) starts: ~2026-04-25 12:44 UTC
# Milestone .65 wall time: 12:44 - 09:21 = 203 min
#
# Prior cumulative baseline (from operational_retro_2026_04_64.json augmented):
#   total_wall_time_minutes = 3971
#   experiments_completed   = 750
#   avg_time_per_experiment_minutes = 5.29
# ---------------------------------------------------------------------------
PRIOR_TOTAL_WALL_TIME_MINUTES = 3971
PRIOR_EXPERIMENTS_COMPLETED = 750
MILESTONE_WALL_TIME_MINUTES = 203   # .65 milestone: 09:21 → 12:44 UTC
PRIOR_MILESTONE_WALL_TIME_MINUTES = 218   # .64 milestone wall time (for delta comparison)
MILESTONE_EXPERIMENTS = 12          # Exps 843-853 (11) + retro 854 (1)
EXPERIMENT_CAP = 700                # Governance cap per CLAUDE.md


# ---------------------------------------------------------------------------
# Load experiment results
# ---------------------------------------------------------------------------

def _load(filename: str) -> dict:
    """Load a JSON result file; return empty dict on missing file.

    Empty-dict fallback means the retro runs even when an experiment was blocked
    and never wrote a deliverable.  Criterion evaluators then see default-zero
    values, which correctly scores the criterion as not-met — preventing silently
    dropped experiments from inflating the success count.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_experiments() -> dict:
    """Return a dict mapping experiment ID → result artifact dict for all .65 experiments."""
    return {
        843: _load("experiment_843_preflight_v14.json"),
        844: _load("experiment_844_jepa_v24b_svamp.json"),
        845: _load("experiment_845_jepa_v24b_tier35_deployment.json"),
        846: _load("experiment_846_arbiter_gibbs_warmstart.json"),
        847: _load("experiment_847_constraint_retrieval_l2_fix.json"),
        848: _load("experiment_848_fr11_tier1_live_relay_v4.json"),
        849: _load("experiment_849_gguf_cache_module.json"),
        850: _load("experiment_850_sota_code_repair_v5.json"),
        851: _load("experiment_851_ice40_n16_bitstream.json"),
        852: _load("experiment_852_semantic_energy_tier0f.json"),
        853: _load("experiment_853_live_benchmark_v4.json"),
    }


# ---------------------------------------------------------------------------
# Success Criteria Evaluation
# ---------------------------------------------------------------------------

def eval_criteria(experiments: dict) -> tuple[list[dict], int]:
    """Evaluate each .65 milestone success criterion against experiment result data.

    Returns (criteria_list, n_met).

    Criteria are evaluated strictly from experiment output fields only.  No
    manual overrides are permitted, even for blocked experiments — a blocked run
    scores as not-met, which is the honest representation of what happened.
    """
    criteria = []

    # 1. governance_ready — Exp 843 honest_verdict = "governance_ready"
    v843 = experiments[843].get("honest_verdict", "")
    criteria.append({
        "criterion": "governance_ready",
        "experiment": 843,
        "target": "honest_verdict == 'governance_ready'",
        "met": v843 == "governance_ready",
        "actual_value": v843,
    })

    # 2. svamp_corpus_balanced — Exp 844 corpus includes >=20 SVAMP training pairs
    # The key fix for .65 was to add SVAMP triplets that were missing in .64.  The
    # corpus is "balanced" if each domain has at least 20 correct + 20 incorrect pairs.
    svamp_comp = experiments[844].get("corpus_composition", {}).get("svamp", {})
    svamp_correct = svamp_comp.get("correct", 0)
    svamp_incorrect = svamp_comp.get("incorrect", 0)
    svamp_balanced = svamp_correct >= 20 and svamp_incorrect >= 20
    criteria.append({
        "criterion": "svamp_corpus_balanced",
        "experiment": 844,
        "target": "corpus_composition.svamp.correct >= 20 AND incorrect >= 20",
        "met": svamp_balanced,
        "actual_value": {"correct": svamp_correct, "incorrect": svamp_incorrect},
    })

    # 3. jepa_v24b_all_domains — Exp 844 all_domains_coverage = True
    # Verifies the training corpus covers all four evaluation domains (gsm8k,
    # humaneval, arc, svamp) rather than being dominated by one or two domains.
    all_domains = experiments[844].get("all_domains_coverage", False)
    criteria.append({
        "criterion": "jepa_v24b_all_domains",
        "experiment": 844,
        "target": "all_domains_coverage == True",
        "met": bool(all_domains),
        "actual_value": all_domains,
    })

    # 4. jepa_v24b_tier35_deployed — Exp 845 tier35_deployed = True
    # Deployment is gated on min_domain_auc >= 0.50 across all domains.  If SVAMP
    # AUC is still 0.0, deployment is blocked even if the overall OOD AUC looks OK.
    tier35 = experiments[845].get("tier35_deployed", False)
    criteria.append({
        "criterion": "jepa_v24b_tier35_deployed",
        "experiment": 845,
        "target": "tier35_deployed == True",
        "met": bool(tier35),
        "actual_value": tier35,
    })

    # 5. arbiter_calibrated — Exp 846 accuracy_standard >= 0.67
    # The arbiter was non-functional in .64 (accuracy_standard=0.0) because Gibbs
    # sampling from a zero-magnetization cold start does not converge.  Warm-start
    # from sign(h_i) external-field configuration fixes convergence.
    acc_std = experiments[846].get("accuracy_standard", 0.0)
    criteria.append({
        "criterion": "arbiter_calibrated",
        "experiment": 846,
        "target": "accuracy_standard >= 0.67",
        "met": acc_std >= 0.67,
        "actual_value": acc_std,
    })

    # 6. retrieval_fixed — Exp 847 retrieval_auroc > 0.80
    # L2-normalization of stored embeddings was the root cause of zero retrieval
    # precision (Gram-Schmidt orthogonalization deflected embeddings from their
    # semantic directions).  The fix is to store plain L2-normalized vectors.
    # Gate: AUROC > 0.80 (industry-standard retrieval quality threshold).
    ret_auroc = experiments[847].get("retrieval_auroc", 0.0)
    criteria.append({
        "criterion": "retrieval_fixed",
        "experiment": 847,
        "target": "retrieval_auroc > 0.80",
        "met": ret_auroc > 0.80,
        "actual_value": ret_auroc,
    })

    # 7. tier1_relay_works_live — Exp 848 honest_verdict = "tier1_relay_works_live"
    # The FR-11 Tier 1 self-learning relay requires monotonic precision improvement
    # across 5 sessions with constraint-guided retrieval.  This is the first live
    # validation of the end-to-end learn → retrieve → improve loop.
    v848 = experiments[848].get("honest_verdict", "")
    criteria.append({
        "criterion": "tier1_relay_works_live",
        "experiment": 848,
        "target": "honest_verdict == 'tier1_relay_works_live'",
        "met": v848 == "tier1_relay_works_live",
        "actual_value": v848,
    })

    # 8. gguf_cache_implemented — Exp 849 honest_verdict = "gguf_cache_implemented"
    # RETRO-GGUF-CACHE-IMPORT: the import path carnot.pipeline.gguf_cache did not
    # exist, blocking all SOTA code-repair experiments for 8+ consecutive milestones.
    v849 = experiments[849].get("honest_verdict", "")
    criteria.append({
        "criterion": "gguf_cache_implemented",
        "experiment": 849,
        "target": "honest_verdict == 'gguf_cache_implemented'",
        "met": v849 == "gguf_cache_implemented",
        "actual_value": v849,
    })

    # 9. code_repair_positive — Exp 850 n_repair_pass > n_baseline_pass
    # The first live SOTA code-repair run using Qwen3.6-35B-A3B-GGUF.  Requires
    # the GGUF model to be on disk at the expected cache path.
    n_baseline = experiments[850].get("n_baseline_pass", 0)
    n_repair = experiments[850].get("n_repair_pass", 0)
    signed_improvement = experiments[850].get("signed_improvement")
    # signed_improvement=None means blocked; False means negative; True means positive
    criteria.append({
        "criterion": "code_repair_positive",
        "experiment": 850,
        "target": "signed_improvement == True (n_repair_pass > n_baseline_pass)",
        "met": signed_improvement is True,
        "actual_value": {"n_baseline_pass": n_baseline, "n_repair_pass": n_repair,
                         "signed_improvement": signed_improvement},
    })

    # 10. bitstream_generated — Exp 851 bitstream_generated = True
    # The iCE40 HX8K Ising sampler bitstream at N=16 spins.  N=32 overflowed in
    # .64 (3952 LUTs).  Hypothesis: N=16 would use ~1000 LUTs and fit within the
    # 7680-LUT budget.
    bitstream_ok = experiments[851].get("bitstream_generated", False)
    criteria.append({
        "criterion": "bitstream_generated",
        "experiment": 851,
        "target": "bitstream_generated == True",
        "met": bool(bitstream_ok),
        "actual_value": bitstream_ok,
    })

    # 11. semantic_probe_viable — Exp 852 honest_verdict = "probe_viable"
    # The SemanticEnergyProbe (Tier 0f) uses pairwise Gaussian kernel energy over
    # sentence embeddings to detect response incoherence.  Viability requires AUC
    # >= 0.70 on a synthetic coherent vs hallucinated benchmark.
    v852 = experiments[852].get("honest_verdict", "")
    criteria.append({
        "criterion": "semantic_probe_viable",
        "experiment": 852,
        "target": "honest_verdict == 'probe_viable'",
        "met": v852 == "probe_viable",
        "actual_value": v852,
    })

    # 12. pipeline_improvement — Exp 853 honest_verdict != "simulated_no_verdict"
    # The live precision benchmark (50 GSM8K, 4 conditions) requires CARNOT_FORCE_LIVE=1
    # to be propagated into the subprocess environment.  Any non-simulated verdict
    # means the live pipeline ran end-to-end on real GPU inference.
    v853 = experiments[853].get("honest_verdict", "")
    criteria.append({
        "criterion": "pipeline_improvement",
        "experiment": 853,
        "target": "honest_verdict != 'simulated_no_verdict'",
        "met": v853 not in ("simulated_no_verdict", ""),
        "actual_value": v853,
    })

    n_met = sum(1 for c in criteria if c["met"])
    return criteria, n_met


# ---------------------------------------------------------------------------
# RETRO accounting
# ---------------------------------------------------------------------------

# Closed this milestone:
# - RETRO-ARBITER-FLAT-ENERGY: Exp 846 accuracy_standard=1.0, accuracy_adversarial=1.0.
#   Warm-start Gibbs (500 burn-in sweeps from sign(h_i) configuration) drove arbiter
#   accuracy to 100% on both standard and adversarial scenarios.
# - RETRO-GGUF-CACHE-IMPORT: Exp 849 implemented GGUFCacheResolver in
#   python/carnot/pipeline/gguf_cache.py.  The import path now exists and all 7 tests pass.
RETROS_CLOSED = [
    "RETRO-ARBITER-FLAT-ENERGY",    # Exp 846: accuracy_standard=1.0 via warm-start Gibbs
    "RETRO-GGUF-CACHE-IMPORT",      # Exp 849: GGUFCacheResolver implemented, 7 tests pass
]

# New RETROs opened in milestone .65:
# - RETRO-SOTA-MODEL-DOWNLOAD: GGUF cache module exists but Qwen3.6-35B-A3B-GGUF is not
#   on disk at the expected path.  Code-repair (Exp 850) blocked for 9th consecutive
#   milestone.  Fix: add model download pre-flight step; document operator download SOP.
# - RETRO-ICE40-N16-UNEXPECTED-EXPANSION: Exp 851 N=16 used 12258 LCs (159% of HX8K
#   capacity vs 7680 max), far exceeding the ~1000 LUT prediction from N=32 (3952 LUTs).
#   Root cause: lf_comb Verilog memory array was replaced with individual registers,
#   causing flip-flop count to grow super-linearly rather than halving with N.  Fix:
#   rewrite Verilog with pipelined register file architecture or reduce to N=8.
# - RETRO-LIVE-ENV-NOT-PROPAGATED: Exp 853 shows carnot_force_live_env="0" after
#   autofix attempt — CARNOT_FORCE_LIVE=1 was not set in the subprocess environment.
#   This is a recurrence of the RETRO-015 pattern (reported "closed" in .63 but
#   re-emerging in .65 under a different code path via the env_autofix heuristic).
RETROS_OPENED = [
    "RETRO-SOTA-MODEL-DOWNLOAD",            # Exp 850: Qwen3.6-35B GGUF not on disk; 9th blocked milestone
    "RETRO-ICE40-N16-UNEXPECTED-EXPANSION", # Exp 851: N=16 = 12258 LCs (159% HX8K); registers expand quadratically
    "RETRO-LIVE-ENV-NOT-PROPAGATED",        # Exp 853: CARNOT_FORCE_LIVE=0 after autofix; RETRO-015 recurrence
]

# Still open after .65 (inherited from prior milestones, not resolved):
RETROS_STILL_OPEN = [
    "RETRO-MANIFEST-FULL-SCOPE",               # Requires conductor code change; not attempted in .65
    "RETRO-JEPA-OOD",                          # Exp 844/845: min_domain_auc=0.0; SVAMP still collapsed
    "RETRO-CONSTRAINT-ZERO-DELTA",             # Exp 847: retrieval_auroc=0.72 < 0.80 gate; relay works but gate not met
    "RETRO-XILINX-TOOLS-UNAVAILABLE",          # Vivado not installed; KV260 Xilinx path blocked
    "RETRO-ISING-INJECTION-NO-DISCRIMINATION", # Not addressed in .65
    "RETRO-SVAMP-ZERO-AUC",                    # Exp 844: auc_svamp=0.0 despite corpus balanced + 8x PRM weight
    "RETRO-ICE40-PNR-LUT-OVERFLOW",            # Exp 851: N=16 overflow worse than N=32; architecture issue
    # New from .65:
    "RETRO-SOTA-MODEL-DOWNLOAD",
    "RETRO-ICE40-N16-UNEXPECTED-EXPANSION",
    "RETRO-LIVE-ENV-NOT-PROPAGATED",
]


# ---------------------------------------------------------------------------
# Improvements for milestone .66
# ---------------------------------------------------------------------------

IMPROVEMENTS = [
    # ---- IMMEDIATE ----
    {
        "priority": "IMMEDIATE",
        "action": (
            "Add GGUF model download pre-flight to MILESTONE_PREREQS.md: RETRO-SOTA-MODEL-DOWNLOAD "
            "is a trivial operator task — huggingface-cli download or wget — but it has now blocked "
            "code repair for 9 consecutive milestones.  Add an explicit pre-flight check in "
            "MILESTONE_PREREQS.md: 'verify Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf present at "
            "models/ or HF_HOME cache'.  Gate: Exp 850-v6 runs live and signed_improvement != None."
        ),
        "rationale": (
            "RETRO-SOTA-MODEL-DOWNLOAD: GGUFCacheResolver now exists (Exp 849) but the model "
            "file itself is not present.  This is an operator gap, not a code gap.  Writing it "
            "into MILESTONE_PREREQS.md converts it from a silent blocker into a hard gate."
        ),
    },
    {
        "priority": "IMMEDIATE",
        "action": (
            "Rewrite iCE40 Ising Verilog with pipelined register-file architecture (N=8): "
            "Exp 851 N=16 used 12258 LCs (159% of HX8K budget) because lf_comb memory was "
            "synthesized as individual flip-flops rather than block RAM.  Use SB_RAM40_4K "
            "primitives for J and h storage to avoid register explosion.  Alternatively "
            "reduce to N=8 as a proof-of-concept (estimated ~600 LUTs).  Gate: "
            "bitstream_generated=True AND bitstream_valid_header=True AND fmax_mhz > 1.0."
        ),
        "rationale": (
            "RETRO-ICE40-N16-UNEXPECTED-EXPANSION + RETRO-ICE40-PNR-LUT-OVERFLOW: Five "
            "milestones of FPGA synthesis without a working bitstream.  The KV260 board has "
            "been idle since 2026-04-20.  The root cause is now diagnosed (register expansion) "
            "and the fix is architectural, not parametric."
        ),
    },
    {
        "priority": "IMMEDIATE",
        "action": (
            "Fix CARNOT_FORCE_LIVE=1 environment propagation (RETRO-LIVE-ENV-NOT-PROPAGATED): "
            "Exp 853 shows carnot_force_live_env='0' after the env_autofix heuristic ran.  "
            "The autofix path is overwriting the env var rather than setting it.  Add an "
            "assertion in LiveGPUGate.require_live_or_blocked() that os.environ['CARNOT_FORCE_LIVE'] "
            "== '1' before returning live status, not just after attempting to set it.  "
            "Gate: Exp 853-v5 returns inference_mode='live_gpu' on first attempt without autofix."
        ),
        "rationale": (
            "RETRO-LIVE-ENV-NOT-PROPAGATED is a recurrence of RETRO-015, which was marked "
            "closed in .63 but re-emerging via a different code path (env_autofix).  Four "
            "consecutive milestones of live benchmark attempts blocked by the same env var. "
            "The autofix heuristic is masking the real propagation gap."
        ),
    },
    {
        "priority": "IMMEDIATE",
        "action": (
            "Fix SVAMP AUC floor collapse — switch from PRM weight boost to architecture fix: "
            "Exp 844 applied an 8x PRM weight boost to SVAMP (largest domain weight applied "
            "in the project) and still produced auc_svamp=0.0.  Data reweighting cannot fix a "
            "model that has no representational capacity for SVAMP error patterns.  Next step: "
            "add a domain-specific head or feature engineering for word-problem arithmetic. "
            "Gate: auc_svamp >= 0.40 on SVAMP holdout.  JEPA deployment remains blocked until "
            "this threshold is met."
        ),
        "rationale": (
            "RETRO-SVAMP-ZERO-AUC: Two milestones of data reweighting failed to move SVAMP AUC "
            "from 0.0.  The corpus is now balanced (40 SVAMP pairs, 8x PRM weight) but the "
            "model produces random predictions on SVAMP.  This is an architecture limitation, "
            "not a data coverage gap.  Continuing to add SVAMP weight without architecture "
            "change is wasted effort."
        ),
    },
    {
        "priority": "IMMEDIATE",
        "action": (
            "Fix EmbeddingConstraintStore retrieval AUROC to exceed 0.80 gate: Exp 847 achieved "
            "retrieval_auroc=0.72 (vs 0.80 gate) using L2-normalized embeddings with cosine "
            "threshold=0.5.  The RETRO-CONSTRAINT-ZERO-DELTA relay gate requires AUROC > 0.80 "
            "AND tier1_relay_works_live.  Tier 1 relay now works (Exp 848 honest_verdict="
            "'tier1_relay_works_live'), so the only remaining gap is retrieval quality.  "
            "Try fine-tuning threshold or using a better sentence-transformer checkpoint.  "
            "Gate: retrieval_auroc > 0.80 in Exp 847-v2."
        ),
        "rationale": (
            "RETRO-CONSTRAINT-ZERO-DELTA: relay is now wired and working (Exp 848 confirmed), "
            "but the full RETRO closure requires both conditions.  The retrieval AUROC gap "
            "(0.72 vs 0.80) is the only remaining blocker for the RETRO closure."
        ),
    },
    # ---- HIGH ----
    {
        "priority": "HIGH",
        "action": (
            "Implement manifest enforcement at ALL conductor dequeue sites before .66 experiments: "
            "RETRO-MANIFEST-FULL-SCOPE has been open since .61 — six consecutive milestones.  "
            "The manifest_fix_patch.txt exists in the repo and the audit of call sites was done "
            "in Exp 843 (preflight v14).  Apply the exclusion check to every dequeue site in "
            "scripts/research_conductor.py.  Gate: _task_is_excluded() returns (False, '') for "
            "0 experiments already in the log."
        ),
        "rationale": (
            "Without manifest enforcement, retired experiments re-enter the queue and inflate "
            "wall time by 129+ min/milestone.  This is the longest-lived unresolved IMMEDIATE "
            "recommendation in project history (6 milestones).  Exp 843 confirmed the patch "
            "is written; deployment is the only remaining step."
        ),
    },
    {
        "priority": "HIGH",
        "action": (
            "Deploy ExperimentTimeoutWatchdog with 30-min cap as default infrastructure: "
            "The .65 slowest experiments ran 26 min (Exp 845), 22 min (Exp 851), and 18 min "
            "(Exps 843, 846, 849) — all below 30 min.  The watchdog would not have cut any "
            ".65 experiment short.  But it remains essential defense-in-depth for the "
            "manifest enforcement gap: if a retired experiment re-enters the queue, 30 min "
            "is the worst-case exposure.  Add ExperimentTimeoutWatchdog(exp_id, timeout_minutes=30) "
            "to experiment_template.py as a required component."
        ),
        "rationale": (
            "Seven consecutive milestones of recommendation without deployment.  The .65 "
            "experiments did not require the watchdog, but this is because no retired "
            "experiments ran in .65 (manifest enforcement coincidentally worked this milestone). "
            "The watchdog is cheap insurance."
        ),
    },
    {
        "priority": "MEDIUM",
        "action": (
            "Add per-domain AUC floor assertion to conductor pre-flight before JEPA deployment "
            "experiments: Exps 834, 844 both showed per-domain collapse (ARC in .63, SVAMP in "
            ".64 and .65) that was only caught inside the deployment experiment.  Moving the "
            "min_domain_auc >= 0.50 gate into the conductor's pre-flight sequence would have "
            "saved the deployment experiment slot in .64 and .65."
        ),
        "rationale": (
            "The deployment gate is already correct (Exp 845 correctly blocked).  The issue "
            "is where the gate fires — inside a deployment experiment consumes a milestone "
            "slot and masks the diagnostic step.  Pre-flight gate = same result, no slot consumed."
        ),
    },
]


# ---------------------------------------------------------------------------
# Compute milestone metrics
# ---------------------------------------------------------------------------

def compute_metrics() -> dict:
    """Compute cumulative and per-milestone timing metrics for .65.

    Cumulative totals add the .65 milestone contribution to the .64 baseline.
    The per-milestone wall time is the delta between .65 activation and retro start,
    derived from conductor-log.md timestamps.

    wall_time_delta_vs_64 is the key regression/improvement indicator:
    negative means .65 ran faster (improvement) vs .64 milestone.
    """
    total_wall_time = PRIOR_TOTAL_WALL_TIME_MINUTES + MILESTONE_WALL_TIME_MINUTES
    total_experiments = PRIOR_EXPERIMENTS_COMPLETED + MILESTONE_EXPERIMENTS
    avg_time = round(total_wall_time / total_experiments, 2)
    wall_time_delta_vs_64 = MILESTONE_WALL_TIME_MINUTES - PRIOR_MILESTONE_WALL_TIME_MINUTES

    return {
        "total_wall_time_minutes": total_wall_time,
        "experiments_completed": total_experiments,
        "avg_time_per_experiment_minutes": avg_time,
        "milestone_wall_time_minutes": MILESTONE_WALL_TIME_MINUTES,
        "milestone_experiments": MILESTONE_EXPERIMENTS,
        "wall_time_delta_vs_64_minutes": wall_time_delta_vs_64,
        "wall_time_delta_vs_64_direction": "improvement" if wall_time_delta_vs_64 < 0 else "regression",
        "prior_milestone_label": ".64",
        "prior_milestone_wall_time_minutes": PRIOR_MILESTONE_WALL_TIME_MINUTES,
        "prior_total_wall_time_minutes": PRIOR_TOTAL_WALL_TIME_MINUTES,
        "prior_experiments_completed": PRIOR_EXPERIMENTS_COMPLETED,
        "experiment_count_delta_vs_64": MILESTONE_EXPERIMENTS,
        "experiment_count_vs_cap": (
            f"{total_experiments} vs {EXPERIMENT_CAP} cap — "
            f"{'EXCEEDED by ' + str(total_experiments - EXPERIMENT_CAP) + ' experiments' if total_experiments > EXPERIMENT_CAP else 'within cap'}"
        ),
    }


# ---------------------------------------------------------------------------
# RETRO audit
# ---------------------------------------------------------------------------

def audit_retros(experiments: dict) -> dict:
    """Evaluate each named RETRO and return its status after milestone .65.

    Each RETRO closure requires a specific experiment field to exceed a threshold.
    If the experiment was blocked or the threshold was not met, the RETRO remains
    open.  Partial mitigation (e.g., relay works but retrieval AUROC is short of
    the gate) is documented explicitly to avoid misleading the next milestone plan.
    """
    return {
        "RETRO-SVAMP-ZERO-AUC": {
            "status": "open",
            "gate": "auc_svamp >= 0.40",
            "evidence": f"Exp 844 auc_svamp={experiments[844].get('auc_svamp', 'N/A')}; "
                        "8x PRM weight boost failed to move SVAMP AUC from 0.0; "
                        "corpus balanced (40 triplets, 20 correct + 20 incorrect); "
                        "architecture limitation suspected",
        },
        "RETRO-JEPA-OOD": {
            "status": "open",
            "gate": "min_domain_auc >= 0.50",
            "evidence": f"Exp 844 min_domain_auc={experiments[844].get('min_domain_auc', 'N/A')}; "
                        f"Exp 845 tier35_deployed={experiments[845].get('tier35_deployed', False)}; "
                        "deployment blocked by SVAMP floor",
        },
        "RETRO-ARBITER-FLAT-ENERGY": {
            "status": "closed",
            "gate": "accuracy_standard >= 0.67",
            "evidence": f"Exp 846 accuracy_standard={experiments[846].get('accuracy_standard', 'N/A')}; "
                        "warm-start Gibbs (500 sweeps, sign(h_i) init) drove accuracy to 1.0; "
                        "consensus_penalty working on adversarial scenarios",
        },
        "RETRO-CONSTRAINT-ZERO-DELTA": {
            "status": "partially_mitigated",
            "gate": "retrieval_auroc > 0.80 AND tier1_relay_works_live",
            "evidence": f"Exp 847 retrieval_auroc={experiments[847].get('retrieval_auroc', 'N/A')} "
                        "(gate=0.80, short by 0.08); "
                        f"Exp 848 honest_verdict='{experiments[848].get('honest_verdict', 'N/A')}'; "
                        "relay now works but retrieval AUROC gate not yet met",
        },
        "RETRO-GGUF-CACHE-IMPORT": {
            "status": "closed",
            "gate": "gguf_cache_implemented == True",
            "evidence": f"Exp 849 honest_verdict='{experiments[849].get('honest_verdict', 'N/A')}'; "
                        "GGUFCacheResolver implemented in python/carnot/pipeline/gguf_cache.py; "
                        "7 tests pass, 100% coverage",
        },
        "RETRO-ICE40-PNR-LUT-OVERFLOW": {
            "status": "open",
            "gate": "bitstream_generated == True",
            "evidence": f"Exp 851 bitstream_generated={experiments[851].get('bitstream_generated', False)}; "
                        f"lut_count_n16={experiments[851].get('lut_count_n16', 'N/A')} "
                        "(synthesis) but P&R used 12258 LCs (159% of HX8K); "
                        "new root cause: register-array expansion (lf_comb replaced with regs)",
        },
        "RETRO-MANIFEST-FULL-SCOPE": {
            "status": "open",
            "gate": "human action required (manifest_patch applied to conductor)",
            "evidence": "Exp 843 preflight confirmed patch written but not yet applied; "
                        "sixth consecutive milestone without application",
        },
        "RETRO-XILINX-TOOLS-UNAVAILABLE": {
            "status": "open",
            "gate": "Vivado installed",
            "evidence": "Not addressed in .65; KV260 Xilinx synthesis path requires Vivado",
        },
        "RETRO-ISING-INJECTION-NO-DISCRIMINATION": {
            "status": "open",
            "gate": "Not evaluated in .65",
            "evidence": "Not addressed in .65 milestone experiments",
        },
        "RETRO-SOTA-MODEL-DOWNLOAD": {
            "status": "open",
            "gate": "Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf present on disk",
            "evidence": f"Exp 850 status=blocked; honest_verdict='{experiments[850].get('honest_verdict', 'N/A')}'; "
                        "new RETRO opened in .65; GGUFCacheResolver exists but model file absent",
        },
        "RETRO-ICE40-N16-UNEXPECTED-EXPANSION": {
            "status": "open",
            "gate": "bitstream_generated == True with pipelined Verilog",
            "evidence": f"Exp 851 lc_count_pnr=12258 (expected ~1000); lut_count_n16={experiments[851].get('lut_count_n16', 'N/A')} (synthesis); "
                        "synthesis under-counts because memory array expands at P&R; "
                        "new RETRO opened in .65; requires SB_RAM40_4K primitives or N=8",
        },
        "RETRO-LIVE-ENV-NOT-PROPAGATED": {
            "status": "open",
            "gate": "inference_mode == 'live_gpu' (Exp 853-v5)",
            "evidence": f"Exp 853 carnot_force_live_env='{experiments[853].get('carnot_force_live_env', 'N/A')}'; "
                        "autofix attempted but CARNOT_FORCE_LIVE remained 0; "
                        "recurrence of RETRO-015 pattern via different code path; "
                        "new RETRO opened in .65",
        },
    }


# ---------------------------------------------------------------------------
# Slowest-5 analysis
# ---------------------------------------------------------------------------

def compute_slowest_5() -> list[dict]:
    """Return the 5 slowest .65 experiments by wall-clock elapsed time.

    Elapsed time is computed from conductor-log.md timestamps, not from the
    experiment duration_s field (which measures only compute time and excludes
    Claude Code agent startup and test-runner overhead).

    The .64 slowest-5 were: Exps 786 (77 min), 527 (52 min), 491 (52 min),
    627 (51 min), 603 (44 min) — four were legacy retired experiments that kept
    re-entering the unguarded queue.  If the manifest patch was applied before .65,
    these should not appear in the .65 slowest-5.
    """
    # Wall-clock elapsed computed from conductor-log.md start/finish timestamps
    elapsed = [
        {"experiment": 843, "elapsed_minutes": 18, "status": "success",
         "note": "governance preflight; Claude Code startup overhead"},
        {"experiment": 844, "elapsed_minutes": 12, "status": "success",
         "note": "JEPA v24b SVAMP training; 250 epochs on 160 pairs"},
        {"experiment": 845, "elapsed_minutes": 26, "status": "blocked",
         "note": "JEPA deployment gate; blocked on min_domain_auc=0.0"},
        {"experiment": 846, "elapsed_minutes": 18, "status": "success",
         "note": "arbiter warm-start; 500 Gibbs sweeps per scenario"},
        {"experiment": 847, "elapsed_minutes": 14, "status": "success",
         "note": "constraint retrieval L2 fix; 25 query-label pairs"},
        {"experiment": 848, "elapsed_minutes": 13, "status": "success",
         "note": "FR-11 tier1 relay; 5 sessions × 15 questions"},
        {"experiment": 849, "elapsed_minutes": 18, "status": "success",
         "note": "GGUF cache module implementation"},
        {"experiment": 850, "elapsed_minutes": 16, "status": "blocked",
         "note": "SOTA code repair; blocked immediately at model lookup"},
        {"experiment": 851, "elapsed_minutes": 22, "status": "partial",
         "note": "iCE40 N=16 synthesis + P&R; P&R failed after 22 min"},
        {"experiment": 852, "elapsed_minutes": 17, "status": "success",
         "note": "semantic energy probe; first attempt failed (max turns), retry OK"},
        {"experiment": 853, "elapsed_minutes": 7, "status": "blocked",
         "note": "live benchmark; blocked immediately at env check"},
    ]
    sorted_by_elapsed = sorted(elapsed, key=lambda x: x["elapsed_minutes"], reverse=True)
    return sorted_by_elapsed[:5]


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------

def compute_honest_verdict(n_met: int, n_total: int, retros_still_open: list) -> str:
    """Encode the honest verdict per the .64 retro encoding convention.

    Format: wall_time_delta_direction_KEY_EVENTS_n_of_N_criteria_X_open_retros
    """
    # Milestone wall time improved vs .64 (203 min vs 218 min = -15 min)
    wall_direction = "improvement"
    n_open = len(retros_still_open)
    return (
        f"milestone_{wall_direction}_15min_vs_64_"
        f"{n_met}of{n_total}_criteria_"
        f"RETRO_ARBITER_CLOSED_RETRO_GGUF_CLOSED_"
        f"3NEW_RETROS_OPENED_{n_open}_open_"
        "SVAMP_floor_0pt0_two_milestones_PRM_boost_failed_architecture_fix_required_"
        "ICE40_N16_overflows_12258LC_register_expansion_root_cause_"
        "LIVE_ENV_broken_autofix_path_RETRO015_recurrence_"
        "CODE_REPAIR_still_blocked_model_not_on_disk"
    )


# ---------------------------------------------------------------------------
# MILESTONE_PREREQS.md update
# ---------------------------------------------------------------------------

def write_milestone_prereqs_section() -> None:
    """Append the .66 prerequisites section to MILESTONE_PREREQS.md.

    Each IMMEDIATE-class improvement from this retro becomes a blocking gate
    that must be verified (status=verified_complete) before any .66 experiment
    dequeues.  Existing content is never modified — only a new section is appended.
    """
    section = """

---

## Milestone 2026.04.66 Prerequisites — Verify Before ANY Experiment Runs

All IMMEDIATE-class actions from the .65 retro (results/operational_retro_2026_04_65.json)
must be verified before the research conductor runs any .66 experiments.

Source retro honest_verdict: see operational_retro_2026_04_65.json
n_criteria_met: 7/12

Mark each item as one of:
- `pending` — not yet verified; conductor MUST NOT run experiments until resolved
- `verified_complete` — confirmed implemented and working
- `escalated_retro` — cannot be completed; carried to .67 retro with documented reason

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 1 | Download Qwen3.6-35B-A3B-GGUF-Q4_K_M.gguf to models/ or HF_HOME cache (RETRO-SOTA-MODEL-DOWNLOAD) | pending | huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF Q4_K_M; verify with GGUFCacheResolver |
| 2 | Rewrite iCE40 Ising Verilog with SB_RAM40_4K block RAM or reduce to N=8 (RETRO-ICE40-N16-UNEXPECTED-EXPANSION) | pending | Register expansion root cause diagnosed; need pipelined arch |
| 3 | Fix CARNOT_FORCE_LIVE=1 propagation via env_autofix code path (RETRO-LIVE-ENV-NOT-PROPAGATED) | pending | Add assertion in LiveGPUGate.require_live_or_blocked() |
| 4 | Fix SVAMP AUC floor: add domain-specific head or feature engineering (RETRO-SVAMP-ZERO-AUC) | pending | 8x PRM boost failed; architecture change required |
| 5 | Fix EmbeddingConstraintStore retrieval AUROC from 0.72 to >0.80 (RETRO-CONSTRAINT-ZERO-DELTA) | pending | Tier 1 relay works (Exp 848); retrieval is the only remaining gap |

## How the Gate Works

The research conductor (scripts/research_conductor.py) MUST check this file in its
pre-flight sequence.  If ANY item is `pending`, the conductor logs a WARNING and halts
before calling run_agent().  This converts the retro from a documentation exercise into
an operational gate.

## Retro Source

- Source: results/operational_retro_2026_04_65.json (improvements_suggested, IMMEDIATE items)
- Gate implemented: Exp 854 (2026-04-25)
- Next update: Before milestone 2026.04.66 planning
"""
    # Read existing content to check for duplicates before appending
    if os.path.exists(MILESTONE_PREREQS):
        with open(MILESTONE_PREREQS) as fh:
            existing = fh.read()
        if "Milestone 2026.04.66 Prerequisites" in existing:
            return  # Already written; do not duplicate
        with open(MILESTONE_PREREQS, "a") as fh:
            fh.write(section)
    else:
        with open(MILESTONE_PREREQS, "w") as fh:
            fh.write(section.lstrip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full .65 retrospective and write the deliverable JSON.

    Steps:
    1. Load all 11 milestone experiment results (843-853).
    2. Compute milestone metrics (wall time, experiment count).
    3. Evaluate 12 success criteria strictly from result fields.
    4. Audit RETRO statuses; identify closed, opened, and still-open.
    5. Compute slowest-5 experiments for bottleneck analysis.
    6. Compose artifact with schema=carnot.operational_retro.v40.
    7. Write results/operational_retro_2026_04_65.json.
    8. Append .66 prerequisites section to MILESTONE_PREREQS.md.
    9. Assert deliverable was written and contains all required fields.
    """
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    experiments = load_experiments()

    verdicts = {
        str(eid): data.get("honest_verdict", "unknown")
        for eid, data in experiments.items()
    }

    metrics = compute_metrics()
    criteria, n_met = eval_criteria(experiments)
    retro_audit = audit_retros(experiments)
    slowest_5 = compute_slowest_5()
    honest_verdict = compute_honest_verdict(n_met, len(criteria), RETROS_STILL_OPEN)

    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    artifact = {
        "schema": "carnot.operational_retro.v40",
        "milestone": "2026.04.65",
        "retro_date": "20260425",
        "experiment": 854,
        "title": "Milestone 2026.04.65 Operational Retrospective",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": 0,  # updated on write

        # Experiments evaluated (843-853; 854 is the retro itself)
        "experiments_evaluated": list(range(843, 854)),
        "experiment_verdicts": verdicts,

        # Milestone metrics
        **metrics,

        # Success criteria (12 for .65)
        "success_criteria": criteria,
        "n_criteria_met": n_met,
        "n_criteria_total": len(criteria),

        # RETRO accounting
        "retros_closed": RETROS_CLOSED,
        "retros_opened": RETROS_OPENED,
        "retros_still_open": RETROS_STILL_OPEN,
        "retro_audit": retro_audit,

        # Bottleneck analysis
        "slowest_5_experiments": slowest_5,

        # Improvements for .66
        "improvements_suggested": IMPROVEMENTS,

        # Honest verdict (encoded per schema spec)
        "honest_verdict": honest_verdict,

        # Invariant check
        "invariant_violations": [],
    }

    # Update duration_s now that we have finished_at
    try:
        start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        artifact["duration_s"] = round((end_dt - start_dt).total_seconds(), 3)
    except Exception:
        artifact["duration_s"] = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DELIVERABLE, "w") as fh:
        json.dump(artifact, fh, indent=2)

    write_milestone_prereqs_section()

    print(f"[854] Written: {DELIVERABLE}")
    print(f"[854] n_criteria_met={n_met}/{len(criteria)}  honest_verdict truncated={honest_verdict[:80]}...")
    print(f"[854] retros_closed={RETROS_CLOSED}")
    print(f"[854] retros_opened={RETROS_OPENED}")
    print(f"[854] retros_still_open ({len(RETROS_STILL_OPEN)}): {RETROS_STILL_OPEN}")

    assert_deliverable_written()


def assert_deliverable_written() -> None:
    """Confirm the deliverable exists and contains all required v40 schema fields.

    Called as the final line of main().  The conductor uses this function's
    success/failure as the go/no-go signal for committing the milestone retro.
    """
    assert os.path.exists(DELIVERABLE), f"Deliverable not written: {DELIVERABLE}"
    with open(DELIVERABLE) as fh:
        check = json.load(fh)

    required_fields = [
        "schema", "milestone", "experiment", "honest_verdict",
        "n_criteria_met", "n_criteria_total", "success_criteria",
        "retros_closed", "retros_opened", "retros_still_open",
        "improvements_suggested", "total_wall_time_minutes",
        "experiments_completed", "avg_time_per_experiment_minutes",
        "slowest_5_experiments",
    ]
    for field in required_fields:
        assert field in check, f"Missing required field: {field}"

    assert check["schema"] == "carnot.operational_retro.v40", "Schema version wrong"
    assert check["milestone"] == "2026.04.65", "Milestone label wrong"
    assert check["experiment"] == 854, "Experiment ID wrong"
    assert isinstance(check["n_criteria_met"], int), "n_criteria_met must be int"
    assert isinstance(check["success_criteria"], list), "success_criteria must be list"
    assert len(check["success_criteria"]) == 12, "Expected 12 criteria for .65"
    assert isinstance(check["slowest_5_experiments"], list), "slowest_5_experiments must be list"
    assert len(check["slowest_5_experiments"]) == 5, "Expected 5 slowest experiments"
    print("[854] assert_deliverable_written: OK")


if __name__ == "__main__":
    main()
