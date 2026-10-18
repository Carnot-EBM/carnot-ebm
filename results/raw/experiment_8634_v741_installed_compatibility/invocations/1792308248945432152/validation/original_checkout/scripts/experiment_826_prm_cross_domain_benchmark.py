#!/usr/bin/env python3
"""Experiment 826 — PRM Cross-Domain Degradation Benchmark.

**Researcher summary:**
    arXiv 2506.00027 reports that Process Reward Models trained on math reasoning
    degrade ~8% AUC when applied to code verification (OOD domain).  This experiment
    measures whether Carnot's JEPA v23 verifier stays within that published baseline
    when generalising from its GSM8K training distribution to HumanEval (code) and
    ARC-Challenge (planning).

    This is a CPU-only analysis experiment — it reads stored results from:
      - Exp 825: cross-domain AUC values (auc_gsm8k, auc_humaneval, auc_arc)
      - Exp 824: in-distribution AUC baseline (in_dist_auc on GSM8K)
      - Exp 825: VerificationCertificates for 20 high-energy OOD steps

    The key metric is cross_domain_degradation = in_dist_auc - auc_ood.
    If degradation_max <= 0.08 we beat the published baseline ("above_baseline").
    If abs(degradation_max - 0.08) <= 0.01 we match it ("at_baseline").
    If degradation_max > 0.09 we fall short ("below_baseline").

    VerificationCertificate corroboration (arXiv 2601.17223):
      For each of the 20 certificates from Exp 825, we check whether the z3_verdict
      ("unsat" = step is wrong) aligns with the jepa_energy_delta direction (positive
      delta = high energy = JEPA thinks the step is wrong).  The corroboration_rate
      is the fraction of certificates where z3 and JEPA agree on the step's validity.

Spec: REQ-VERIFY-145, SCENARIO-VERIFY-174
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make project root importable when running as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# apply_env_autofix MUST be the first Carnot import — it sets JAX_PLATFORMS=cpu
# and other environment guards before JAX initialises.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 826
TITLE = "PRM Cross-Domain Degradation Benchmark (arXiv 2506.00027 vs JEPA v23)"
DELIVERABLE = "results/experiment_826_prm_cross_domain_benchmark.json"
EXP825_FILE = "results/experiment_825_jepa_v23_eval_fr11_tier3.json"
EXP824_FILE = "results/experiment_824_jepa_v23_limo_corpus.json"

# arXiv 2506.00027 baseline: PRM cross-domain degradation when moving from math → code.
PUBLISHED_BASELINE = 0.08
# "at_baseline" tolerance band around the 8% mark.
AT_BASELINE_TOLERANCE = 0.01


# ---------------------------------------------------------------------------
# Core analysis functions (importable for unit tests)
# ---------------------------------------------------------------------------


def compute_degradation(in_dist_auc: float, ood_auc: float) -> float:
    """Return cross-domain degradation = in_dist_auc - ood_auc.

    A positive value means the model performs worse on the OOD domain than it
    does on its training distribution.  This matches the definition used in
    arXiv 2506.00027 so our result is directly comparable to their 8% figure.
    """
    return in_dist_auc - ood_auc


def determine_honest_verdict(
    degradation_max: float,
    *,
    data_unavailable: bool = False,
) -> str:
    """Map numeric degradation to a human-readable verdict string.

    The verdict categories are:
      - "data_unavailable": upstream experiment (Exp 824 or 825) is blocked.
      - "above_baseline": degradation_max <= 0.08 (we beat the arXiv baseline).
      - "at_baseline": |degradation_max - 0.08| <= 0.01 (within noise of baseline).
      - "below_baseline": degradation_max > 0.09 (we fall short of baseline).

    WHY these thresholds: the 8% boundary is taken directly from arXiv 2506.00027.
    The 1% tolerance band avoids falsely claiming "above baseline" on noise.
    A model at 8.5% degradation is not meaningfully below baseline; only at >9%
    do we consider it a genuine generalisation failure.

    NOTE: "above_baseline" takes precedence over "at_baseline" because the two
    conditions can overlap (e.g. degradation=0.075 satisfies both).  The intent
    of "above_baseline" is to flag a genuine win; "at_baseline" is for the grey zone.
    """
    if data_unavailable:
        return "data_unavailable"
    if degradation_max <= PUBLISHED_BASELINE:
        return "above_baseline"
    if abs(degradation_max - PUBLISHED_BASELINE) <= AT_BASELINE_TOLERANCE:
        return "at_baseline"
    return "below_baseline"


def compute_corroboration_rate(certificates: list[dict[str, Any]]) -> float:
    """Return fraction of certificates where z3_verdict and jepa_energy_delta agree.

    Agreement means:
      - z3_verdict == "unsat" (step is wrong) AND jepa_energy_delta > 0 (high energy = error)
      - z3_verdict == "sat"   (step is correct) AND jepa_energy_delta <= 0 (low energy = ok)

    This measures how well JEPA's continuous energy signal corroborates the discrete
    Z3 formal verdict, as described in arXiv 2601.17223 (step-level VerificationCertificates).
    A high corroboration_rate means JEPA and Z3 are pointing in the same direction —
    which is a prerequisite for using JEPA as a proxy for formal verification.
    """
    if not certificates:
        return 0.0
    n_agree = sum(
        1
        for cert in certificates
        if (cert["z3_verdict"] == "unsat") == (cert["jepa_energy_delta"] > 0)
    )
    return n_agree / len(certificates)


def analyze_failing_steps(
    certificates: list[dict[str, Any]],
    worst_domain: str,
    top_n: int = 5,
) -> dict[str, Any]:
    """Summarise the top-N highest-energy OOD steps for the worst domain.

    Returns a dict with:
      - top_steps: list of step_id (sorted by energy_delta descending)
      - constraint_type_counts: Counter mapping constraint_type -> count
      - most_common_constraint: the single most frequent constraint_type

    WHY top-N analysis: if JEPA degrades on a specific domain, understanding
    WHICH constraint types dominate the failures tells us where to focus future
    training data collection.  A concentration in "code_logic" steps, for example,
    would suggest we need more HumanEval-style contrastive pairs.
    """
    # Filter to the worst domain prefix (e.g. "humaneval_" or "arc_").
    domain_certs = [c for c in certificates if c["step_id"].startswith(worst_domain)]
    # Sort by energy delta descending (most-wrong first).
    domain_certs_sorted = sorted(domain_certs, key=lambda c: c["jepa_energy_delta"], reverse=True)
    top_steps = [c["step_id"] for c in domain_certs_sorted[:top_n]]
    constraint_type_counts: Counter = Counter(
        c["constraint_type"] for c in domain_certs_sorted[:top_n]
    )
    most_common = constraint_type_counts.most_common(1)[0][0] if constraint_type_counts else None
    return {
        "top_steps": top_steps,
        "constraint_type_counts": dict(constraint_type_counts),
        "most_common_constraint": most_common,
    }


# ---------------------------------------------------------------------------
# Deliverable writer
# ---------------------------------------------------------------------------


def _write_deliverable(artifact: dict) -> None:
    """Write the artifact dict as JSON to the deliverable path.

    Called explicitly before assert_deliverable_written() — the ExperimentTemplate
    checks for the file on disk but does not write it itself.
    """
    deliverable_path = _REPO_ROOT / DELIVERABLE
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable_path, "w") as fh:
        json.dump(artifact, fh, indent=2)


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the PRM cross-domain degradation benchmark and write the artifact."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)

    # ------------------------------------------------------------------
    # Load Exp 825 (cross-domain AUC + VerificationCertificates)
    # ------------------------------------------------------------------
    exp825_path = _REPO_ROOT / EXP825_FILE
    if not exp825_path.exists():
        artifact = tmpl.build_result(
            {"reason": f"{EXP825_FILE} not found"},
            status="data_unavailable",
            honest_verdict="data_unavailable",
        )
        _write_deliverable(artifact)
        tmpl.assert_deliverable_written()
        return

    with open(exp825_path) as fh:
        exp825 = json.load(fh)

    # If Exp 825 itself was blocked (gate failure), propagate that.
    if exp825.get("blocked_gate") or exp825.get("status") not in ("success",):
        artifact = tmpl.build_result(
            {"blocked_reason": exp825.get("honest_verdict", "unknown")},
            status="data_unavailable",
            honest_verdict="data_unavailable",
        )
        _write_deliverable(artifact)
        tmpl.assert_deliverable_written()
        return

    auc_gsm8k: float = exp825["auc_gsm8k"]
    auc_humaneval: float = exp825["auc_humaneval"]
    auc_arc: float = exp825["auc_arc"]
    overall_ood_auc: float = exp825["overall_ood_auc"]
    certificates: list[dict] = exp825.get("verification_certificates", [])

    # ------------------------------------------------------------------
    # Load Exp 824 (in-distribution AUC baseline)
    # ------------------------------------------------------------------
    exp824_path = _REPO_ROOT / EXP824_FILE
    if not exp824_path.exists():
        artifact = tmpl.build_result(
            {"reason": f"{EXP824_FILE} not found"},
            status="data_unavailable",
            honest_verdict="data_unavailable",
        )
        _write_deliverable(artifact)
        tmpl.assert_deliverable_written()
        return

    with open(exp824_path) as fh:
        exp824 = json.load(fh)

    in_dist_auc: float = exp824["in_dist_auc"]

    # ------------------------------------------------------------------
    # Compute cross-domain degradation (REQ-VERIFY-145)
    # ------------------------------------------------------------------
    deg_humaneval = compute_degradation(in_dist_auc, auc_humaneval)
    deg_arc = compute_degradation(in_dist_auc, auc_arc)
    deg_max = max(deg_humaneval, deg_arc)
    beats_baseline = deg_max <= PUBLISHED_BASELINE

    # ------------------------------------------------------------------
    # VerificationCertificate corroboration (arXiv 2601.17223)
    # ------------------------------------------------------------------
    corroboration_rate = compute_corroboration_rate(certificates)
    n_certificates = len(certificates)

    # ------------------------------------------------------------------
    # Worst-domain analysis (only when we fall short of baseline)
    # ------------------------------------------------------------------
    worst_domain_analysis: dict[str, Any] = {}
    worst_domain: str | None = None
    if deg_max > PUBLISHED_BASELINE:
        worst_domain = "humaneval" if deg_humaneval >= deg_arc else "arc"
        worst_domain_analysis = analyze_failing_steps(certificates, worst_domain)

    # ------------------------------------------------------------------
    # Determine honest verdict (SCENARIO-VERIFY-174)
    # ------------------------------------------------------------------
    honest_verdict = determine_honest_verdict(deg_max)

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    payload: dict[str, Any] = {
        "in_dist_auc": in_dist_auc,
        "auc_gsm8k": auc_gsm8k,
        "auc_humaneval": auc_humaneval,
        "auc_arc": auc_arc,
        "overall_ood_auc": overall_ood_auc,
        "cross_domain_degradation_humaneval": deg_humaneval,
        "cross_domain_degradation_arc": deg_arc,
        "cross_domain_degradation_max": deg_max,
        "beats_baseline": beats_baseline,
        "published_baseline": PUBLISHED_BASELINE,
        "corroboration_rate": corroboration_rate,
        "n_certificates": n_certificates,
        "honest_verdict": honest_verdict,
    }
    if worst_domain is not None:
        payload["worst_domain"] = worst_domain
        payload["worst_domain_analysis"] = worst_domain_analysis

    artifact = tmpl.build_result(payload, status="success", honest_verdict=honest_verdict)
    _write_deliverable(artifact)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
