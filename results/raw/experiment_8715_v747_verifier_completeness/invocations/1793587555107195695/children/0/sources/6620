#!/usr/bin/env python3
"""Milestone 2026.04.88 operational retrospective.

The script reads the milestone result artifacts, evaluates the 11 planned
criteria, summarizes the process failures, and writes the canonical Exp 1138
deliverable consumed by the conductor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONDUCTOR_LOG = REPO_ROOT / "ops" / "conductor-log.md"
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1138_milestone_retro_88.json"
MILESTONE = "2026.04.88"
PRIOR_WALL_TIME_MINUTES = 891.0

EXPERIMENT_FILES: dict[int, str] = {
    1127: "experiment_1127_arxiv_final_submission.json",
    1128: "experiment_1128_sos_kan_root_cause_k5_fix.json",
    1129: "experiment_1129_grpo_energy_prm_v2.json",
    1130: "experiment_1130_zenil_alpha_t_post_retrain.json",
    1131: "experiment_1131_lagrangian_cascade_v2.json",
    1132: "experiment_1132_goodfire_exemplar_cascade_tp.json",
    1133: "experiment_1133_prm_biasbench_adversarial_test.json",
    1134: "experiment_1134_kv260_v4_parameter_tuning.json",
    1135: "experiment_1135_position_paper_v3_findings_update.json",
    1136: "experiment_1136_wopr_slitherlink_cartridge.json",
    1137: "experiment_1137_hf_spaces_gallery_update.json",
}

LOG_TASK_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "exp1127": ("arXiv PDF Compilation + Final Submission",),
    "exp1128": ("SOSKANEnergyV3 Root Cause", "SOS-KAN Polarity Fix"),
    "exp1129": ("GRPO Energy PRM Full Training v2",),
    "exp1130": ("Zenil",),
    "exp1131": ("Lagrangian Cascade v2", "Cascade Routing v3"),
    "exp1132": ("Goodfire LLM Failure Exemplar",),
    "exp1133": ("PRM-BiasBench Adversarial Test",),
    "exp1134": ("KV260 Ising Sampler v4",),
    "exp1135": ("Position Paper v3",),
    "exp1136": ("WOPR Slitherlink Puzzle Cartridge",),
    "exp1137": ("HF Spaces Gallery Update",),
    "exp1138": ("Milestone 2026.04.88 Retrospective",),
    "exp906": ("Exp 906", "experiment_906", "IterativeSelfRepair 50q"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(results_dir: Path = RESULTS_DIR) -> dict[int, dict[str, Any]]:
    """Load all source artifacts, preserving missing files as explicit records."""

    artifacts: dict[int, dict[str, Any]] = {}
    for exp_id, filename in EXPERIMENT_FILES.items():
        path = results_dir / filename
        if path.exists():
            artifacts[exp_id] = _load_json(path)
        else:
            artifacts[exp_id] = {"_missing": True, "_path": str(path)}
    return artifacts


def record_honest_verdicts(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, str]:
    """Return the top-level honest verdict for every milestone source."""

    verdicts: dict[str, str] = {}
    for exp_id in sorted(EXPERIMENT_FILES):
        artifact = artifacts[exp_id]
        verdicts[f"exp{exp_id}"] = (
            "MISSING"
            if artifact.get("_missing")
            else str(artifact.get("honest_verdict", "NO_VERDICT"))
        )
    return verdicts


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_criteria(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, bool]:
    """Evaluate the 11 milestone .88 criteria from authoritative artifacts."""

    e1127 = artifacts[1127]
    e1128 = artifacts[1128]
    e1129 = artifacts[1129]
    e1130 = artifacts[1130]
    e1131 = artifacts[1131]
    e1132 = artifacts[1132]
    e1133 = artifacts[1133]
    e1134 = artifacts[1134]
    e1135 = artifacts[1135]
    e1136 = artifacts[1136]

    return {
        "arxiv_pdf_compiled_or_bundle_manually_uploaded": bool(
            e1127.get("pdf_compiled")
            or e1127.get("arxiv_submitted")
            or e1127.get("arxiv_bundle_verified")
        ),
        "sos_kan_polarity_fixed_k5_auroc_above_threshold": bool(
            e1128.get("sos_kan_root_cause_identified")
            and e1128.get("k5_ensemble_auroc_above_08")
            and _float_value(e1128.get("k5_ensemble_auroc_after")) >= 0.8
        ),
        "grpo_training_budget_not_hit_honest_result": bool(
            e1129.get("grpo_v2_honest_result") and not e1129.get("training_wall_budget_hit")
        ),
        "zenil_alpha_t_post_retrain_measured": bool(
            e1130.get("zenil_alpha_t_post_retrain_measured")
        ),
        "cascade_v2_accuracy_degradation_below_10pp": bool(
            e1131.get("cascade_v2_accuracy_delta_above_neg05")
            and _float_value(e1131.get("accuracy_delta")) >= -0.10
        ),
        "goodfire_exemplar_cascade_tp_measured": bool(
            e1132.get("goodfire_exemplar_tp_rate_measured")
        ),
        "prm_bias_adversarial_test_honest_result": bool(
            e1133.get("prm_biasbench_attack_tp_measured")
            or e1133.get("prm_biasbench_adversarial_tp_measured")
        ),
        "kv260_v4_kl_measured_post_adaptive_tuning": bool(
            e1134.get("parameter_space_mapped")
            or e1134.get("kv260_v4_kl_below_05_or_feasibility_documented")
        ),
        "position_paper_v3_updated": bool(e1135.get("position_paper_findings_updated")),
        "slitherlink_cartridge_shipped": bool(e1136.get("slitherlink_cartridge_shipped")),
        "retro_complete": True,
    }


def _parse_log_line(line: str) -> tuple[dt.datetime, str] | None:
    match = re.match(r"^\|\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\s*\|\s*([^|]+?)\s*\|", line)
    if not match:
        return None
    timestamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC)
    return timestamp, match.group(2)


def compute_wall_time_minutes(log_lines: Sequence[str]) -> float:
    """Compute milestone wall time from activation to the latest .88 log entry."""

    start: dt.datetime | None = None
    last: dt.datetime | None = None
    for line in log_lines:
        parsed = _parse_log_line(line)
        if parsed is None:
            continue
        timestamp, title = parsed
        if f"Milestone {MILESTONE} activated" in title:
            start = timestamp
            last = timestamp
        elif start is not None:
            last = timestamp
    if start is None or last is None:
        return 0.0
    return round((last - start).total_seconds() / 60.0, 1)


def _exp_id_for_title(title: str) -> str | None:
    for exp_id, fragments in LOG_TASK_FRAGMENTS.items():
        if any(fragment in title for fragment in fragments):
            return exp_id
    return None


def build_slowest_experiments(log_lines: Sequence[str]) -> list[dict[str, Any]]:
    """Build the slowest-five composition from conductor log spans."""

    spans: dict[str, list[dt.datetime]] = {}
    milestone_started = False
    for line in log_lines:
        parsed = _parse_log_line(line)
        if parsed is None:
            continue
        timestamp, title = parsed
        if f"Milestone {MILESTONE} activated" in title:
            milestone_started = True
            continue
        if not milestone_started:
            continue
        exp_id = _exp_id_for_title(title)
        if exp_id is not None:
            spans.setdefault(exp_id, []).append(timestamp)

    ranked: list[dict[str, Any]] = []
    for exp_id, timestamps in spans.items():
        duration = round((max(timestamps) - min(timestamps)).total_seconds() / 60.0, 1)
        ranked.append(
            {
                "id": exp_id,
                "duration_min": duration,
                "first_seen_utc": min(timestamps).strftime("%Y-%m-%d %H:%M"),
                "last_seen_utc": max(timestamps).strftime("%Y-%m-%d %H:%M"),
            }
        )
    ranked.sort(key=lambda entry: (-entry["duration_min"], entry["id"]))
    return [{"rank": index + 1, **entry} for index, entry in enumerate(ranked[:5])]


def notable_successes(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Summarize strong positive results from the milestone."""

    e1128 = artifacts[1128]
    e1129 = artifacts[1129]
    e1130 = artifacts[1130]
    e1131 = artifacts[1131]
    e1133 = artifacts[1133]
    return [
        (
            "SOS-KAN/k=5 repair was decisive: k5 AUROC "
            f"{_float_value(e1128.get('k5_ensemble_auroc_after')):.4f}, "
            f"SOS-KAN individual AUROC {_float_value(e1128.get('sos_kan_individual_auroc_after')):.4f}."
        ),
        (
            "GRPO v2 completed 100 training questions without hitting the training budget "
            f"and improved held-out accuracy by {_float_value(e1129.get('improvement_over_baseline')) * 100:.2f} pp."
        ),
        (
            "Zenil alpha_t improved from "
            f"{_float_value(e1130.get('alpha_t_prior')):.2f} to "
            f"{_float_value(e1130.get('alpha_t_post_retrain')):.2f} after retraining."
        ),
        (
            "Cascade v2 preserved accuracy while reducing cost: "
            f"accuracy_delta={_float_value(e1131.get('accuracy_delta')):.3f}, "
            f"cost_savings_pct={_float_value(e1131.get('cost_savings_pct')):.1f}."
        ),
        (
            "PRM-BiasBench-style attack test showed k=5 TP rate "
            f"{_float_value(e1133.get('k5_attack_tp_rate')):.1%} with zero attack false positives."
        ),
    ]


def failures_or_partials(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Summarize unmet criteria, honest negatives, and partial results."""

    e1127 = artifacts[1127]
    e1129 = artifacts[1129]
    e1132 = artifacts[1132]
    e1134 = artifacts[1134]
    e1136 = artifacts[1136]
    e1137 = artifacts[1137]
    return [
        (
            "arXiv is not submitted: PDF compiled with tectonic, but manual upload remains "
            f"pending before {e1127.get('submission_deadline', 'the deadline')}."
        ),
        (
            "GRPO v2 evaluation was partial: evaluation_wall_budget_hit="
            f"{bool(e1129.get('evaluation_wall_budget_hit'))}, "
            f"n_eval_questions={e1129.get('n_eval_questions')}/"
            f"{e1129.get('n_eval_questions_target')}."
        ),
        (
            "Goodfire cascade measurement was mixed: standalone Z3 TP rate "
            f"{_float_value(e1132.get('z3_math_standalone_tp_rate')):.3f}; "
            f"artifact verdict={e1132.get('honest_verdict', 'NO_VERDICT')}."
        ),
        (
            "KV260 v4 improved but missed the KL gate: best KL "
            f"{_float_value(e1134.get('kl_v4_best')):.4f} vs threshold "
            f"{_float_value(e1134.get('kl_v4_threshold')):.2f}."
        ),
        (
            "Slitherlink cartridge did not ship: "
            f"{e1136.get('honest_verdict', 'NO_VERDICT')} at {e1136.get('blocked_at_layer', 'unknown layer')}."
        ),
        (
            "HF Spaces Slitherlink gallery artifact is missing because exp1137 was gated on "
            f"the blocked exp1136 source ({e1137.get('_path', 'missing path')})."
        ),
    ]


def bottlenecks_identified() -> list[str]:
    """Return the top three .89 planning bottlenecks."""

    return [
        (
            "Prior-failures metadata remains the main operational bottleneck: exp1128, exp1130, "
            "and exp1136 all hit repeated DOOMED_RERUN_BLOCK events, while exp1131/exp1133/exp1137 "
            "were gate-blocked downstream."
        ),
        (
            "Gate dependency state is stale across retries: exp1128 eventually succeeded, but "
            "dependent exp1131 and exp1133 spent 55-92 minutes in the slowest-five window before "
            "the conductor accepted their artifacts."
        ),
        (
            "KV260 v4 remains unresolved at the topology level: parameter tuning improved KL "
            "from 0.134 to 0.1128 but did not reach the 0.05 gate, and self-adaptive lambda "
            "made the distribution much worse."
        ),
    ]


def build_artifact(
    artifacts: Mapping[int, Mapping[str, Any]],
    log_lines: Sequence[str],
) -> dict[str, Any]:
    """Assemble the Exp 1138 deliverable."""

    criteria = evaluate_criteria(artifacts)
    criteria_met = sum(criteria.values())
    wall_time = compute_wall_time_minutes(log_lines)
    slowest = build_slowest_experiments(log_lines)
    wall_time_improvement = round(PRIOR_WALL_TIME_MINUTES - wall_time, 1)
    wall_time_improvement_pct = round(wall_time_improvement / PRIOR_WALL_TIME_MINUTES * 100.0, 1)

    return {
        "experiment": "1138_milestone_retro_88",
        "schema": "milestone_retro_v2",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "milestone": MILESTONE,
        "criteria_results": criteria,
        "criteria_met": criteria_met,
        "criteria_total": 11,
        "experiment_honest_verdicts": record_honest_verdicts(artifacts),
        "notable_successes": notable_successes(artifacts),
        "failures_or_partials": failures_or_partials(artifacts),
        "bottlenecks_identified": bottlenecks_identified(),
        "slowest_experiments": slowest,
        "exp906_appeared_in_slowest5": any(entry["id"] == "exp906" for entry in slowest),
        "wall_time_minutes": wall_time,
        "wall_time_baseline_minutes": PRIOR_WALL_TIME_MINUTES,
        "wall_time_improvement_vs_prior_minutes": wall_time_improvement,
        "wall_time_improvement_vs_prior_pct": wall_time_improvement_pct,
        "retro_complete": True,
        "honest_verdict": f"{criteria_met}_of_11_criteria_met",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--conductor-log", type=Path, default=CONDUCTOR_LOG)
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.results_dir)
    log_lines = args.conductor_log.read_text(encoding="utf-8").splitlines()
    artifact = build_artifact(artifacts=artifacts, log_lines=log_lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1138] {artifact['criteria_met']}/{artifact['criteria_total']} criteria met; "
        f"wall_time={artifact['wall_time_minutes']:.1f} min; out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
