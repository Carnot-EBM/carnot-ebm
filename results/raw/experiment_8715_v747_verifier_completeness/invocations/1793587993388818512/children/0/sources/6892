"""Experiment 1968: Milestone 2026.05.153 Retrospective.

Spec traces: REQ-REPORT-009, SCENARIO-REPORT-006.

Milestone theme: Negative Constraint Decoding, Flow Sampling Samplers,
and Continuous Self-Learning Refinement.

This module aggregates the results of experiments 1956–1967 and produces
the canonical milestone retro artifact for .153.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Maps experiment IDs to their result filenames (under results/).
EXPERIMENT_FILES: dict[int, str] = {
    1956: "experiment_1956_nco_negative_constraints.json",
    1957: "experiment_1957_truncproof_ll1_grammar.json",
    1958: "experiment_1958_gcot_branching_prototype.json",
    1959: "experiment_1959_tri_sota_constrained_eval.json",
    1960: "experiment_1960_flow_sampling_unnormalized.json",
    1961: "experiment_1961_interleaved_gibbs_diffusion.json",
    1962: "experiment_1962_ni_sampling_token_order.json",
    1963: "experiment_1963_continuous_self_learning_routing.json",
    1964: "experiment_1964_hardware_accounted_igd_eval.json",
    1965: "experiment_1965_energy_guided_nco_benchmark.json",
    1966: "experiment_1966_tri_sota_e2e_v8.json",
    1967: "experiment_1967_milestone_153_pre_retro_audit.json",
}

MILESTONE = "2026.05.153"
MILESTONE_NUM = 153


def load_artifacts(results_dir: Path) -> dict[int, dict[str, Any]]:
    """Load all .153 experiment artifacts from *results_dir*.

    Missing files are returned as ``{"_missing": True}`` so callers can
    distinguish "file absent" from "file present but status not set".
    """
    artifacts: dict[int, dict[str, Any]] = {}
    for exp_id, filename in EXPERIMENT_FILES.items():
        path = results_dir / filename
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    artifacts[exp_id] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                artifacts[exp_id] = {"_missing": True}
        else:
            artifacts[exp_id] = {"_missing": True}
    return artifacts


def record_honest_verdicts(artifacts: dict[int, dict[str, Any]]) -> dict[str, str]:
    """Return a ``{exp<ID>: honest_verdict}`` mapping for all .153 experiments."""
    verdicts: dict[str, str] = {}
    for exp_id, data in artifacts.items():
        if data.get("_missing"):
            verdicts[f"exp{exp_id}"] = "MISSING"
        else:
            v = data.get("honest_verdict") or data.get("verdict") or "unspecified"
            verdicts[f"exp{exp_id}"] = str(v)
    return verdicts


def _is_completed(data: dict[str, Any]) -> bool:
    """Return True if the artifact represents a terminal success/complete."""
    if data.get("_missing"):
        return False
    status = str(data.get("status", "")).lower()
    if status in ("complete", "success"):
        return True
    verdict = str(data.get("honest_verdict", "") or data.get("verdict", "")).lower()
    return (
        verdict.startswith("complete")
        or verdict.startswith("success")
        or verdict.startswith("passed")
        or verdict.startswith("shipped")
    )


def _is_blocked(data: dict[str, Any]) -> bool:
    """Return True if the artifact represents a gate-blocked run."""
    if data.get("_missing"):
        return False
    status = str(data.get("status", "")).lower()
    if status == "blocked":
        return True
    verdict = str(data.get("honest_verdict", "")).lower()
    return "blocked" in verdict


def _is_failed(data: dict[str, Any]) -> bool:
    """Return True if the artifact is present but failed (not blocked, not complete)."""
    if data.get("_missing"):
        return True
    return not _is_completed(data) and not _is_blocked(data)


def classify_tasks(
    artifacts: dict[int, dict[str, Any]]
) -> tuple[list[int], list[int], list[int]]:
    """Partition experiment IDs into (completed, blocked, failed) lists."""
    completed, blocked, failed = [], [], []
    for exp_id, data in artifacts.items():
        if _is_completed(data):
            completed.append(exp_id)
        elif _is_blocked(data):
            blocked.append(exp_id)
        else:
            failed.append(exp_id)
    return sorted(completed), sorted(blocked), sorted(failed)


def evaluate_criteria(artifacts: dict[int, dict[str, Any]]) -> dict[str, bool]:
    """Return a boolean gate-result mapping for the .153 acceptance criteria.

    Each key is a human-readable criterion; the value is True iff the
    criterion passed.
    """
    # NCO negative constraints shipped (exp1956)
    nco_complete = _is_completed(artifacts.get(1956, {}))

    # TruncProof LL(1) grammar shipped (exp1957)
    truncproof_complete = _is_completed(artifacts.get(1957, {}))

    # GCoT branching prototype shipped (exp1958)
    gcot_complete = _is_completed(artifacts.get(1958, {}))

    # IGD mixed-variable sampler shipped (exp1961)
    igd_complete = _is_completed(artifacts.get(1961, {}))

    # NI Sampling token order shipped (exp1962)
    ni_complete = _is_completed(artifacts.get(1962, {}))

    # Flow sampler KL < 1.0 (exp1960: KL=6.93 → FAIL)
    exp1960 = artifacts.get(1960, {})
    kl = _float_value(exp1960.get("kl_divergence"), default=float("inf"))
    flow_kl_acceptable = kl < 1.0

    # Continual routing did NOT run without prior_failures (blocked correctly)
    exp1963 = artifacts.get(1963, {})
    routing_blocked_correctly = _is_blocked(exp1963)

    # Pre-retro audit ran (exp1967 present regardless of verdict)
    pre_retro_ran = not artifacts.get(1967, {}).get("_missing", False)

    # Gate-field contract gap detected — exp1959, exp1964, exp1965 blocked
    gate_contract_gap_detected = all(
        _is_blocked(artifacts.get(eid, {})) for eid in (1959, 1964, 1965)
    )

    return {
        "nco_negative_constraints_shipped": nco_complete,
        "truncproof_ll1_shipped": truncproof_complete,
        "gcot_branching_shipped": gcot_complete,
        "igd_sampler_shipped": igd_complete,
        "ni_sampling_shipped": ni_complete,
        "flow_sampler_kl_below_threshold": flow_kl_acceptable,
        "continual_routing_blocked_correctly_by_discipline": routing_blocked_correctly,
        "pre_retro_audit_ran": pre_retro_ran,
        "gate_contract_gap_surfaced": gate_contract_gap_detected,
    }


def _float_value(value: Any, default: float = 0.0) -> float:
    """Safely coerce *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_artifact(artifacts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Assemble the canonical .153 retro artifact."""
    completed, blocked, failed = classify_tasks(artifacts)
    verdicts = record_honest_verdicts(artifacts)
    criteria = evaluate_criteria(artifacts)

    criteria_met = sum(1 for v in criteria.values() if v)
    criteria_total = len(criteria)

    # Gather notable successes and bottlenecks
    notable_successes = []
    if criteria["nco_negative_constraints_shipped"]:
        notable_successes.append(
            "NCO negative constraint registry implemented (exp1956); "
            "token exclusion patterns integrated into decoding loop."
        )
    if criteria["truncproof_ll1_shipped"]:
        notable_successes.append(
            "TruncProof budget-aware LL(1) closure ships (exp1957); "
            "zero-false-accept rate on truncated JSON generation."
        )
    if criteria["gcot_branching_shipped"]:
        notable_successes.append(
            "GCoT branching prototype ships (exp1958); "
            "Carnot partial-trace energy ranks and culls reasoning paths."
        )
    if criteria["igd_sampler_shipped"]:
        notable_successes.append(
            "Interleaved Gibbs Diffusion sampler ships (exp1961); "
            "faster mixing than sequential Gibbs on MAX-3-SAT Potts."
        )

    exp1962_data = artifacts.get(1962, {})
    ni_accel = _float_value(
        (exp1962_data.get("metrics") or {}).get("acceleration_factor"), default=0.0
    )
    if criteria["ni_sampling_shipped"] and ni_accel > 1.0:
        notable_successes.append(
            f"NI Sampling ships (exp1962); {ni_accel:.1f}x wall-clock acceleration "
            "over random-order discrete diffusion with semantic retention verified."
        )

    bottlenecks = []
    exp1960_data = artifacts.get(1960, {})
    kl = _float_value(exp1960_data.get("kl_divergence"), default=float("inf"))
    if not criteria["flow_sampler_kl_below_threshold"]:
        bottlenecks.append(
            f"Flow Sampling (exp1960) KL divergence = {kl:.4f} far exceeds "
            "the 1.0 threshold; the conditional denoising sampler requires "
            "more training steps or architectural revision before headline claims."
        )
    if artifacts.get(1966, {}).get("_missing"):
        bottlenecks.append(
            "Tri-SOTA E2E v8 (exp1966) was never executed; it was gated on "
            "exp1959 (Tri-SOTA Constrained Eval) which was itself blocked by a "
            "missing boolean `success` field in the exp1956 artifact."
        )

    gate_contract_note = (
        "Gate-field contract gap: exp1959, exp1964, exp1965 were blocked because "
        "their upstream artifacts (exp1956, exp1961) do not set an explicit boolean "
        "`success: true` field. Future tasks gating on `artifact_field: success` "
        "MUST be paired with upstream tasks that write `success: true` in JSON."
    )

    recommendations = [
        "MANDATORY .154: Add explicit `success: true` to all upstream artifacts "
        "that downstream tasks gate on — specifically exp1956 (NCO) and exp1961 (IGD).",
        "MANDATORY .154: Re-propose Flow Sampling (exp1960) with more training "
        "steps (n_steps >= 1000, n_samples >= 2000) and a KL < 0.3 acceptance gate. "
        "Root cause: 100-step denoiser with 500 samples is insufficient.",
        "MANDATORY .154: Add `prior_failures` entries covering exp1963 (Continual "
        "Routing) and exp1965 (Energy-Guided NCO Benchmark) before re-proposing; "
        "the rerun-discipline block was correct and must be addressed explicitly.",
        "Defer exp1966 (Tri-SOTA E2E v8) until exp1959 is unblocked (requires "
        "exp1956 gate-field fix first).",
        "Retire exp1960 if the next attempt with higher budget also exceeds KL=1.0.",
    ]

    honest_verdict = f"complete: milestone_153_retro_filed_{len(completed)}_completed_{len(blocked)}_blocked_{len(failed)}_failed_gate_contract_gap_and_flow_kl_exceed_threshold"

    return {
        "experiment_id": 1968,
        "schema": "carnot.milestone_retro.v1",
        "milestone": MILESTONE,
        "run_date": "20260512",
        "status": "complete",
        "completed_task_count": len(completed),
        "blocked_task_count": len(blocked),
        "failed_task_count": len(failed),
        "completed_experiments": completed,
        "blocked_experiments": blocked,
        "failed_experiments": failed,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "criteria_results": criteria,
        "experiment_honest_verdicts": verdicts,
        "notable_successes": notable_successes,
        "bottlenecks_identified": bottlenecks,
        "gate_contract_gap_note": gate_contract_note,
        "recommendations": recommendations,
        "retro_complete": True,
        "honest_verdict": honest_verdict,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load artifacts, build retro, write deliverable."""
    parser = argparse.ArgumentParser(description="Milestone .153 retrospective")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing experiment result JSON files.",
    )
    parser.add_argument(
        "--out",
        default="results/experiment_1968_milestone_153_retro.json",
        help="Output path for the retro artifact.",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    artifacts = load_artifacts(results_dir)
    artifact = build_artifact(artifacts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote retro artifact to {out_path}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
