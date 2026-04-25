#!/usr/bin/env python3
"""Experiment 830 — Milestone 2026.04.63 Operational Retrospective.

**Researcher summary:**
    At each milestone boundary the conductor runs a retrospective experiment that
    evaluates every prior experiment in the milestone, scores success criteria,
    identifies which open RETROs were closed vs opened, and produces a ranked list
    of improvements for the next milestone.  This file is the retrospective for
    milestone 2026.04.63 (Exps 819-829).

**Why this is a separate Python script and not prose:**
    The conductor's ``_run_operational_retrospective()`` method calls a script file
    so that the retro output lives in the standard JSON artifact schema alongside all
    other experiment results.  This allows the conductor to gate .64 planning on the
    retro's honest_verdict and n_criteria_met fields just like any other deliverable.

**Schema:** carnot.operational_retro.v38
"""

import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DELIVERABLE = os.path.join(RESULTS_DIR, "experiment_830_milestone_retro_2026_04_63.json")
MILESTONE_PREREQS = os.path.join(REPO_ROOT, "MILESTONE_PREREQS.md")

# ---------------------------------------------------------------------------
# Load experiment results
# ---------------------------------------------------------------------------

def _load(exp_id: int, filename: str) -> dict:
    """Load a JSON result file; return empty dict on missing file."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


EXPERIMENTS = {
    819: _load(819, "experiment_819_injection_field_fix.json"),
    820: _load(820, "experiment_820_gguf_import_fix_code_repair_v5.json"),
    821: _load(821, "experiment_821_constraint_addition_live_v2.json"),
    822: _load(822, "experiment_822_arbiter_fix_v2_agent_auditor.json"),
    823: _load(823, "experiment_823_fr11_tier1_live_relay_v2.json"),
    824: _load(824, "experiment_824_jepa_v23_limo_corpus.json"),
    825: _load(825, "experiment_825_jepa_v23_eval_fr11_tier3.json"),
    826: _load(826, "experiment_826_prm_cross_domain_benchmark.json"),
    827: _load(827, "experiment_827_kv260_nextpnr_xilinx_v3.json"),
    828: _load(828, "experiment_828_activation_jailbreak_probe.json"),
    829: _load(829, "experiment_829_huggingface_v3_publish.json"),
}

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

VERDICTS = {
    exp_id: data.get("honest_verdict", "unknown")
    for exp_id, data in EXPERIMENTS.items()
}

# ---------------------------------------------------------------------------
# Success Criteria Evaluation
#
# Each criterion is a tuple:
#   (criterion_name, experiment_id_or_ids, target_description, met_bool, actual_value)
# ---------------------------------------------------------------------------

def _eval_criteria():
    """Evaluate each success criterion and return a list of dicts plus total met count.

    The criteria are taken from the vNEXT.md milestone design table.  Each criterion
    tests one observable value against the threshold that was agreed before the milestone
    started.  This prevents retroactive goalpost movement — a fundamental discipline rule
    in spec-anchored research.
    """
    criteria = []

    # 1. injection_field_fixed
    dr = EXPERIMENTS[819].get("discrimination_rate", 0.0)
    criteria.append({
        "criterion": "injection_field_fixed",
        "experiment": 819,
        "target": "discrimination_rate >= 0.8",
        "met": dr >= 0.8,
        "actual_value": dr,
    })

    # 2. gguf_import_fixed
    v820 = VERDICTS[820]
    criteria.append({
        "criterion": "gguf_import_fixed",
        "experiment": 820,
        "target": "honest_verdict != still_blocked",
        "met": v820 != "still_blocked",
        "actual_value": v820,
    })

    # 3. constraint_addition_works_live
    delta_overall = EXPERIMENTS[821].get("delta_overall", 0.0)
    criteria.append({
        "criterion": "constraint_addition_works_live",
        "experiment": 821,
        "target": "delta_overall > 0",
        "met": delta_overall is not None and delta_overall > 0,
        "actual_value": delta_overall,
    })

    # 4. arbiter_correct
    acc_overall = EXPERIMENTS[822].get("accuracy_overall", 0.0)
    criteria.append({
        "criterion": "arbiter_correct",
        "experiment": 822,
        "target": "accuracy_overall >= 0.80",
        "met": acc_overall >= 0.80,
        "actual_value": acc_overall,
    })

    # 5. tier1_relay_works_live
    delta_s1_s5 = EXPERIMENTS[823].get("delta_s1_to_s5", None)
    criteria.append({
        "criterion": "tier1_relay_works_live",
        "experiment": 823,
        "target": "delta_s1_to_s5 > 0",
        "met": delta_s1_s5 is not None and delta_s1_s5 > 0,
        "actual_value": delta_s1_s5,
    })

    # 6. jepa_v23_viable (joint Exps 824 + 825)
    # Exp 824 trains with ood_auc=0.811; Exp 825 cross-domain eval returns
    # overall_ood_auc=0.40.  The milestone gate required the *deployed* verifier
    # to show ood_auc >= 0.65, which is only confirmed by the cross-domain eval
    # in Exp 825.  Exp 824 alone is not sufficient: training-set AUC does not
    # prove generalization.
    ood_auc_825 = EXPERIMENTS[825].get("overall_ood_auc", 0.0)
    criteria.append({
        "criterion": "jepa_v23_viable",
        "experiment": "824+825",
        "target": "ood_auc >= 0.65 (cross-domain, Exp 825)",
        "met": ood_auc_825 >= 0.65,
        "actual_value": ood_auc_825,
    })

    # 7. cross_domain_at_baseline
    deg_max = EXPERIMENTS[826].get("cross_domain_degradation_max", 1.0)
    criteria.append({
        "criterion": "cross_domain_at_baseline",
        "experiment": 826,
        "target": "degradation <= 0.08 (8%)",
        "met": deg_max <= 0.08,
        "actual_value": deg_max,
    })

    # 8. bitstream_or_synthesis_clean
    bitstream_ok = EXPERIMENTS[827].get("ice40_bitstream_generated", False)
    xilinx_ok = EXPERIMENTS[827].get("xilinx_synthesis_clean", False)
    criteria.append({
        "criterion": "bitstream_or_synthesis_clean",
        "experiment": 827,
        "target": "any clean synthesis (iCE40 bitstream or Xilinx clean)",
        "met": bitstream_ok or xilinx_ok,
        "actual_value": {
            "ice40_bitstream_generated": bitstream_ok,
            "xilinx_synthesis_clean": xilinx_ok,
        },
    })

    # 9. probe_viable
    probe_auc = EXPERIMENTS[828].get("probe_auc", 0.0)
    latency_ms = EXPERIMENTS[828].get("latency_ms", 999.0)
    criteria.append({
        "criterion": "probe_viable",
        "experiment": 828,
        "target": "AUC >= 0.85 AND latency < 1ms",
        "met": probe_auc >= 0.85 and latency_ms < 1.0,
        "actual_value": {"probe_auc": probe_auc, "latency_ms": latency_ms},
    })

    # 10. hf_publish_success
    n_existing = EXPERIMENTS[829].get("n_existing", 0)
    n_after = EXPERIMENTS[829].get("n_after", 0)
    criteria.append({
        "criterion": "hf_publish_success",
        "experiment": 829,
        "target": "n_after > n_existing",
        "met": n_after > n_existing,
        "actual_value": {"n_existing": n_existing, "n_after": n_after},
    })

    n_met = sum(1 for c in criteria if c["met"])
    return criteria, n_met


# ---------------------------------------------------------------------------
# RETRO accounting
# ---------------------------------------------------------------------------

# Open from milestone .62 (documented in vNEXT.md):
#   RETRO-ISING-INJECTION-NO-DISCRIMINATION  → closed by Exp 819
#   RETRO-GGUF-CACHE-IMPORT                 → closed by Exp 820
#   RETRO-ARBITER-FLAT-ENERGY               → remains open; accuracy_overall=0.5
#   RETRO-JEPA-V22-OOD-BELOW-GATE          → superseded by jepa_v23 work; OOD still below gate
#   RETRO-CONSTRAINT-ZERO-DELTA             → confirmed open; Exp 821 delta=0.0 in all sessions
#   RETRO-TIER1-PLATEAU                     → remains open; Exp 823 blocked_gate

RETROS_CLOSED = [
    "RETRO-ISING-INJECTION-NO-DISCRIMINATION",  # Exp 819: discrimination_rate=1.0, retro_injection_closed=True
    "RETRO-GGUF-CACHE-IMPORT",                  # Exp 820: import_fixed_repair_positive, live repair delta=+14
]

RETROS_OPENED = [
    # New failure mode discovered in Exp 825: JEPA v23 planning domain collapses
    # to AUC=0.04 (far below random 0.5), making the ARC domain actively harmful.
    # Root cause unknown: contrastive triplet loss may overfit to arithmetic/code
    # structure and produce anti-correlated features for planning problems.
    "RETRO-JEPA-PLANNING-DOMAIN-COLLAPSE",
    # Exp 827: yosys executed SYNTH_ICE40 on ising_sampler_v3 but the run ended
    # without a valid bitstream (ice40_bitstream_generated=False, valid_header=False).
    # nextpnr-ice40 either timed out or hit a routing constraint for N=32 spins on
    # the iCE40 HX8K device family.  Blocking the hardware-acceleration roadmap item.
    "RETRO-ICE40-BITSTREAM-FAILURE",
]

# ---------------------------------------------------------------------------
# Improvements for milestone .64
# ---------------------------------------------------------------------------

IMPROVEMENTS = [
    # ---- IMMEDIATE (must appear in MILESTONE_PREREQS.md before .64 runs) ----
    {
        "priority": "IMMEDIATE",
        "action": (
            "Fix EmbeddingConstraintStore precision: precision=0.0 across all 3 sessions "
            "in Exp 821.  Root cause: constraint retrieval is returning random-walk scores "
            "rather than semantic matches.  Inspect ConstraintRetriever.retrieve() — likely "
            "cosine threshold too permissive or embedding normalization missing.  Gate: "
            "precision > 0.1 in at least 1 of 3 sessions before Exp 821-v3 runs."
        ),
        "rationale": (
            "RETRO-CONSTRAINT-ZERO-DELTA blocks Exp 821, 822 (arbiter), and 823 (Tier 1 relay). "
            "It is the deepest blocker in the pipeline — arbiter accuracy is only 0.5 because "
            "it selects by energy which is 0 everywhere.  Nothing downstream can improve until "
            "precision > 0."
        ),
    },
    {
        "priority": "IMMEDIATE",
        "action": (
            "Add IsingEBM MCMC sampling warm-start to MultiAgentArbiter: the arbiter standard "
            "scenario accuracy is only 0.17.  Diagnosis: the Gibbs sampler is initialising "
            "from a random spin configuration each call, which means energy estimates have "
            "O(N) variance for N=16 spins.  Fix: start from the external-field h-aligned "
            "configuration (s_i = sign(h_i)) and run 500 burn-in steps before reading energy. "
            "Gate: accuracy_overall >= 0.70 on 12 standard+adversarial scenarios."
        ),
        "rationale": (
            "RETRO-ARBITER-FLAT-ENERGY: accuracy_overall=0.5 in Exp 822 despite external field "
            "fix from Exp 819.  Standard scenarios are being selected wrong (1/6 correct).  "
            "The external field is wired correctly but the energy estimate is noisy without "
            "proper MCMC convergence."
        ),
    },
    {
        "priority": "IMMEDIATE",
        "action": (
            "Run nextpnr-ice40 routing with --hx8k --package ct256 --pcf constraints.pcf "
            "on the ising_sampler_v3_n32 netlist from Exp 816.  Capture the full nextpnr "
            "log, including any routing failure messages.  If routing fails due to resource "
            "exhaustion (3952 LUTs exceeds HX8K capacity after P&R reserve), reduce N to 16 "
            "and retry.  Gate: ice40_bitstream_generated=True and valid_header=True."
        ),
        "rationale": (
            "RETRO-ICE40-BITSTREAM-FAILURE: KV260 FPGA board arrived 2026-04-20 and has been "
            "idle since.  We have yosys synthesis working (Exp 816) but no bitstream.  This "
            "blocks the hardware acceleration roadmap.  The fix is a simple invocation of "
            "nextpnr-ice40 that Exp 827 omitted from the script."
        ),
    },
    # ---- HIGH ----
    {
        "priority": "HIGH",
        "action": (
            "Retrain JEPA v23 with planning domain corpus: add 20 ARC-Challenge pairs to the "
            "triplet training set.  ARC OOD AUC=0.04 (Exp 825) means the model actively "
            "misclassifies planning errors.  Add ARC pairs with Z3-planning constraints "
            "(not arithmetic).  Gate: auc_arc >= 0.45 on ARC holdout."
        ),
        "rationale": (
            "RETRO-JEPA-PLANNING-DOMAIN-COLLAPSE: JEPA v23 trained on GSM8K+HumanEval only. "
            "ARC domain collapse is expected — the model has never seen a planning-domain "
            "training example.  This is a data coverage gap, not an architecture failure."
        ),
    },
    {
        "priority": "HIGH",
        "action": (
            "Publish JEPA v23 checkpoint to HuggingFace after domain fix: Exp 829 published "
            "the injection fix (n_after=27) but jepa_published=False.  Once JEPA v23 meets "
            "the ARC fix gate, publish the checkpoint with the standard card format."
        ),
        "rationale": (
            "HuggingFace v3 publish sequence is confirmed working (Exp 829).  JEPA v23 "
            "was intentionally not published because it failed the cross-domain eval.  "
            "Publish after the planning fix."
        ),
    },
    # ---- MEDIUM ----
    {
        "priority": "MEDIUM",
        "action": (
            "Switch code repair experiments to SOTA GGUF models: Exp 820 used "
            "unsloth/Qwen3.5-0.8B-GGUF and showed repair_delta=+14 on 20 problems.  "
            "Re-run with unsloth/Qwen3.6-35B-A3B-GGUF to establish a headline-quality "
            "live code repair result.  The small model result is encouraging but not "
            "publishable as a headline claim."
        ),
        "rationale": (
            "CLAUDE.md mandates SOTA models for headline results.  Exp 820 used a tiny "
            "model for an initial unblock.  The repair pipeline is now confirmed working; "
            "upgrade to SOTA for the authoritative benchmark."
        ),
    },
    {
        "priority": "MEDIUM",
        "action": (
            "Add cross-domain degradation monitoring to the conductor's milestone gate: "
            "if any domain's OOD AUC drops below 0.3 (far below random), block deployment "
            "and flag RETRO automatically.  Exp 826 discovered the ARC collapse post-hoc; "
            "the conductor should have caught this before Tier 3.5 was attempted."
        ),
        "rationale": (
            "The ARC domain collapse (AUC=0.04) in Exp 825 was not caught by any gate.  "
            "The Tier 3.5 deployment in Exp 825 was attempted (tier35_deployed=False only "
            "because of the overall_ood_auc gate).  A per-domain floor would have triggered "
            "earlier."
        ),
    },
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    criteria, n_met = _eval_criteria()

    # honest_verdict thresholds from task specification
    if n_met >= 7:
        honest_verdict = "retro_63_strong"
    elif n_met >= 4:
        honest_verdict = "retro_63_mixed"
    else:
        honest_verdict = "retro_63_blocked"

    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    artifact = {
        "schema": "carnot.operational_retro.v38",
        "milestone": "2026.04.63",
        "retro_date": "20260425",
        "experiment": 830,
        "title": "Milestone 2026.04.63 Operational Retrospective",
        "started_at": started_at,
        "finished_at": finished_at,
        "experiments_evaluated": list(range(819, 830)),
        "experiment_verdicts": VERDICTS,
        "success_criteria": criteria,
        "n_criteria_met": n_met,
        "n_criteria_total": 10,
        "retros_closed": RETROS_CLOSED,
        "retros_opened": RETROS_OPENED,
        "improvements_suggested": IMPROVEMENTS,
        "honest_verdict": honest_verdict,
        "invariant_violations": [],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DELIVERABLE, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"[830] Written: {DELIVERABLE}")
    print(f"[830] n_criteria_met={n_met}/10  honest_verdict={honest_verdict}")
    print(f"[830] retros_closed={RETROS_CLOSED}")
    print(f"[830] retros_opened={RETROS_OPENED}")

    _update_milestone_prereqs(artifact)

    # Final assertion: confirm the deliverable file exists and is parseable.
    assert os.path.exists(DELIVERABLE), f"Deliverable not written: {DELIVERABLE}"
    with open(DELIVERABLE) as fh:
        check = json.load(fh)
    assert check["schema"] == "carnot.operational_retro.v38", "Schema field wrong"
    assert check["n_criteria_met"] == n_met, "n_criteria_met mismatch"
    print("[830] assert_deliverable_written: OK")


def _update_milestone_prereqs(artifact: dict) -> None:
    """Append a Milestone 2026.04.64 section to MILESTONE_PREREQS.md.

    Reads the existing file first (per CLAUDE.md: never remove content) and
    appends a new section for .64 with all IMMEDIATE-class improvements listed
    as pending gates.
    """
    if not os.path.exists(MILESTONE_PREREQS):
        existing = ""
    else:
        with open(MILESTONE_PREREQS) as fh:
            existing = fh.read()

    # Guard: do not double-append if already present
    if "## Milestone 2026.04.64 Prerequisites" in existing:
        print("[830] MILESTONE_PREREQS.md already has .64 section — skipping append.")
        return

    immediate_items = [
        item for item in artifact["improvements_suggested"]
        if item["priority"] == "IMMEDIATE"
    ]

    rows = []
    for i, item in enumerate(immediate_items, start=1):
        # Truncate long action strings for table readability
        action_short = item["action"][:120].rstrip()
        rows.append(f"| {i} | {action_short}... | pending |")

    table_rows = "\n".join(rows)

    section = f"""

---

## Milestone 2026.04.64 Prerequisites — Verify Before ANY Experiment Runs

All IMMEDIATE-class actions from the .63 retro (results/experiment_830_milestone_retro_2026_04_63.json)
must be verified before the research conductor runs any .64 experiments.

Source retro honest_verdict: **{artifact['honest_verdict']}**
n_criteria_met: {artifact['n_criteria_met']}/{artifact['n_criteria_total']}

Mark each item as one of:
- `pending` — not yet verified; conductor MUST NOT run experiments until resolved
- `verified_complete` — confirmed implemented and working
- `escalated_retro` — cannot be completed; carried to .65 retro with documented reason

| # | Action | Status |
|---|--------|--------|
{table_rows}

## How the Gate Works

The research conductor (scripts/research_conductor.py) MUST check this file in its
pre-flight sequence.  If ANY item is `pending`, the conductor logs a WARNING and halts
before calling run_agent().  This converts the retro from a documentation exercise into
an operational gate.

## Retro Source

- Source: results/experiment_830_milestone_retro_2026_04_63.json
- Gate implemented: Exp 830 (2026-04-25)
- Next update: Before milestone 2026.04.64 planning
"""

    with open(MILESTONE_PREREQS, "w") as fh:
        fh.write(existing + section)
    print(f"[830] Updated: {MILESTONE_PREREQS}")


if __name__ == "__main__":
    main()
