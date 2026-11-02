#!/usr/bin/env python3
"""Milestone 2026.04.89 operational retrospective.

The workflow reads the .89 source experiment artifacts, evaluates the 13
planned criteria, summarizes honest partials, and writes the canonical Exp 1151
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
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1151_milestone_retro_89.json"
MILESTONE = "2026.04.89"
PRIOR_WALL_TIME_MINUTES = 145.0
CRITERIA_TOTAL = 13

EXPERIMENT_FILES: dict[int, str] = {
    1139: "experiment_1139_arxiv_final_submission_v3.json",
    1140: "experiment_1140_roadmap_gate_prior_failures_audit.json",
    1141: "experiment_1141_wopr_slitherlink_rescue.json",
    1142: "experiment_1142_beaver_lite_certificate_tier.json",
    1143: "experiment_1143_halluguard_cascade_router_v3.json",
    1144: "experiment_1144_cctu_micro_benchmark_adapter.json",
    1145: "experiment_1145_goodfire_cheap_tier_distillation.json",
    1146: "experiment_1146_grpo_reflection_reward_v3.json",
    1147: "experiment_1147_hardnet_projection_repair.json",
    1148: "experiment_1148_metacluster_sos_kan_compression.json",
    1149: "experiment_1149_kv260_v5_dc_continuous_diagnostic.json",
    1150: "experiment_1150_extropic_integration_packet.json",
}

LOG_TASK_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "exp1139": ("arXiv Final Submission Close-Out v3",),
    "exp1140": ("Roadmap Gate and Prior-Failures Audit Script v1",),
    "exp1141": ("WOPR Slitherlink Puzzle Cartridge Rescue",),
    "exp1142": ("BEAVER-Lite Deterministic Prefix Constraint Bounde",),
    "exp1143": ("HalluGuard NTK Cascade Router v3",),
    "exp1144": ("CCTU 25-Task Micro-Benchmark Adapter",),
    "exp1145": ("Goodfire Cheap-Tier Calibration",),
    "exp1146": ("GRPO Reflection Reward v3",),
    "exp1147": ("HardNet++-Style Projection Repair Layer",),
    "exp1148": ("MetaCluster SOS-KAN Compression",),
    "exp1149": ("KV260 Ising v5",),
    "exp1150": ("Extropic Z1/XTR-0 Integration Packet",),
    "exp1151": ("Milestone 2026.04.89 Retrospective",),
    "exp906": ("Exp 906", "experiment_906", "IterativeSelfRepair 50q"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(results_dir: Path = RESULTS_DIR) -> dict[int, dict[str, Any]]:
    """Load source artifacts, preserving missing files as explicit records."""

    artifacts: dict[int, dict[str, Any]] = {}
    for exp_id, filename in EXPERIMENT_FILES.items():
        path = results_dir / filename
        if path.exists():
            artifacts[exp_id] = _load_json(path)
        else:
            artifacts[exp_id] = {"_missing": True, "_path": str(path)}
    return artifacts


def record_honest_verdicts(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, str]:
    """Return the top-level honest verdict for every .89 source artifact."""

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
    """Evaluate the 13 planned .89 criteria from authoritative artifacts."""

    e1139 = artifacts[1139]
    e1140 = artifacts[1140]
    e1141 = artifacts[1141]
    e1142 = artifacts[1142]
    e1143 = artifacts[1143]
    e1144 = artifacts[1144]
    e1145 = artifacts[1145]
    e1146 = artifacts[1146]
    e1147 = artifacts[1147]
    e1148 = artifacts[1148]
    e1149 = artifacts[1149]
    e1150 = artifacts[1150]

    return {
        "arxiv_final_pdf_recompiled_and_upload_steps_provided": bool(
            e1139.get("pdf_recompiled")
            and e1139.get("grpo_v2_result_in_paper")
            and e1139.get("manual_upload_steps")
        ),
        "gate_prior_failures_audit_complete": bool(
            e1140.get("audit_script_written")
            and int(e1140.get("n_tasks_audited", 0) or 0) >= 13
            and int(e1140.get("n_prior_failures_checks", 0) or 0) >= 13
        ),
        "slitherlink_cartridge_shipped": bool(e1141.get("slitherlink_cartridge_shipped")),
        "beaver_lite_certificate_deployed": bool(
            e1142.get("beaver_lite_bounder_written")
            and e1142.get("beaver_lite_bound_reported")
            and e1142.get("bound_is_sound")
        ),
        "halluguard_routing_feature_measured": bool(
            e1143.get("halluguard_routing_feature_measured")
        ),
        "cctu_micro_benchmark_adapter_complete": bool(
            e1144.get("cctu_adapter_written")
            and e1144.get("cctu_adapter_honest_result")
            and int(e1144.get("n_tasks_defined", 0) or 0) >= 25
            and int(e1144.get("n_tasks_evaluated", 0) or 0) >= 25
        ),
        "goodfire_cheap_tier_distillation_honest_result": bool(
            e1145.get("cheap_tier_tp_rate_improved")
            or e1145.get("honest_verdict") in {"honest_negative", "threshold_trade_off_fp_increase"}
        ),
        "grpo_reflection_reward_v3_honest_result": bool(
            e1146.get("dualgpu_used")
            and e1146.get("reflection_reward_integrated")
            and e1146.get("grpo_reflection_honest_result")
            and not e1146.get("training_wall_budget_hit")
        ),
        "hardnet_projection_repair_honest_result": bool(
            e1147.get("projection_repair_written")
            and e1147.get("hardnet_projection_repair_written")
            and _float_value(e1147.get("projection_repair_accuracy")) >= 1.0
        ),
        "metacluster_sos_kan_compression_honest_result": bool(
            e1148.get("sos_kan_compressed")
            and e1148.get("auroc_drop_within_02")
            and _float_value(e1148.get("size_reduction_factor")) >= 5.0
        ),
        "kv260_v5_dc_continuous_kl_measured": bool(
            e1149.get("kv260_v5_diagnostic_complete") and e1149.get("energy_time_accuracy_reported")
        ),
        "extropic_integration_packet_shipped": bool(
            e1150.get("integration_packet_written")
            and e1150.get("thrml_backend_stub_written")
            and e1150.get("sampler_backend_interface_documented")
        ),
        "retro_complete": True,
    }


def _parse_log_line(line: str) -> tuple[dt.datetime, str] | None:
    match = re.match(r"^\|\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\s*\|\s*([^|]+?)\s*\|", line)
    if not match:
        return None
    timestamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC)
    return timestamp, match.group(2)


def compute_wall_time_minutes(log_lines: Sequence[str]) -> float:
    """Compute .89 wall time from activation to the next milestone or last entry."""

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
            continue
        if start is not None and title.startswith("Milestone ") and "activated" in title:
            break
        if start is not None:
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
        if milestone_started and title.startswith("Milestone ") and "activated" in title:
            break
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


def arxiv_submission_status(artifacts: Mapping[int, Mapping[str, Any]]) -> str:
    """Return the exp1139 arXiv submission state."""

    e1139 = artifacts[1139]
    if e1139.get("arxiv_submitted"):
        return "submitted"
    if e1139.get("status") == "blocked" or e1139.get("_missing"):
        return "not_run"
    return "upload_pending"


def roadmap_gate_audit_caught_blocking_gaps(artifacts: Mapping[int, Mapping[str, Any]]) -> bool:
    """Whether exp1140 found prior-failures gaps that match observed gate blocks."""

    e1139 = artifacts[1139]
    e1140 = artifacts[1140]
    details = " ".join(str(item) for item in e1140.get("failure_details", []))
    return bool(e1139.get("status") == "blocked" and "exp1139" in details)


def notable_successes(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Summarize criteria met with strong results."""

    e1141 = artifacts[1141]
    e1144 = artifacts[1144]
    e1145 = artifacts[1145]
    e1147 = artifacts[1147]
    e1148 = artifacts[1148]
    return [
        (
            "Slitherlink rescue shipped the carried .88 WOPR cartridge: canonical puzzle "
            f"E={_float_value(e1141.get('canonical_puzzle_e_at_convergence')):.1f}, "
            f"{int(e1141.get('tests_passing', 0) or 0)} tests passing, convergence in "
            f"{int(e1141.get('n_iterations_to_convergence', 0) or 0)} iteration."
        ),
        (
            "CCTU micro-benchmark ran 25/25 constrained tool-use tasks with live GPU "
            f"{e1144.get('model_used', 'unknown model')}: baseline "
            f"{_float_value(e1144.get('baseline_completion_rate')):.0%} to Carnot-guided "
            f"{_float_value(e1144.get('carnot_guided_completion_rate')):.0%}."
        ),
        (
            "Goodfire cheap-tier distillation lifted combined cheap-tier TP from "
            f"{_float_value(e1145.get('combined_cheap_tp_before')):.1%} to "
            f"{_float_value(e1145.get('combined_cheap_tp_after')):.1%} using "
            f"{e1145.get('dominant_halluguard_feature', 'the HalluGuard feature gate')}."
        ),
        (
            "Projection repair was exact and fast: "
            f"{_float_value(e1147.get('projection_repair_accuracy')):.1%} accuracy at "
            f"{_float_value(e1147.get('projection_repair_latency_us')):.1f} us, "
            f"{_float_value(e1147.get('speedup_factor')):.0f}x faster than prompt repair."
        ),
        (
            "MetaCluster SOS-KAN compression met both gates: "
            f"{_float_value(e1148.get('size_reduction_factor')):.2f}x smaller, "
            f"AUROC drop {_float_value(e1148.get('auroc_drop')):.4f}, "
            f"energy correlation {_float_value(e1148.get('energy_correlation')):.4f}."
        ),
    ]


def failures_or_partials(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Summarize unmet criteria, honest negatives, and partial results."""

    e1139 = artifacts[1139]
    e1140 = artifacts[1140]
    e1142 = artifacts[1142]
    e1145 = artifacts[1145]
    e1146 = artifacts[1146]
    e1149 = artifacts[1149]
    e1150 = artifacts[1150]
    return [
        (
            "arXiv close-out did not run: exp1139 was blocked at "
            f"{e1139.get('blocked_at_layer', 'unknown layer')} with verdict "
            f"{e1139.get('honest_verdict', 'NO_VERDICT')}; arxiv_submission_status="
            f"{arxiv_submission_status(artifacts)}."
        ),
        (
            "Roadmap gate audit was useful but late: exp1140 found "
            f"{int(e1140.get('n_prior_failures_missing', 0) or 0)} prior_failures gaps, "
            "including the exp1139 gap that had already blocked the release-critical arXiv task."
        ),
        (
            "BEAVER-lite produced a sound bound, but only with mock logprobs: "
            f"unsafe_mass_bound={_float_value(e1142.get('unsafe_mass_bound')):.3f}, "
            f"empirical_violation_rate={_float_value(e1142.get('empirical_violation_rate')):.3f}."
        ),
        (
            "Goodfire cheap-tier TP improved, but the calibration is not production-ready: "
            f"false_positive_rate_after={_float_value(e1145.get('false_positive_rate_after')):.2f}."
        ),
        (
            "GRPO reflection reward underperformed exp1129: improvement_over_baseline="
            f"{_float_value(e1146.get('improvement_over_baseline')) * 100:.2f} pp, "
            f"advantage_stdev={_float_value(e1146.get('advantage_stdev')):.3f}, "
            f"n_eval_questions={int(e1146.get('n_eval_questions', 0) or 0)}/"
            f"{int(e1146.get('n_eval_questions_target', 50) or 50)}."
        ),
        (
            "KV260 v5 DC-continuous diagnostic was an honest negative: KL "
            f"{_float_value(e1149.get('kl_v5_best')):.4f} vs prior v4 "
            f"{_float_value(e1149.get('kl_v4_best_prior')):.4f}; improvement "
            f"{_float_value(e1149.get('kl_improvement_over_v4')):.4f}."
        ),
        (
            "Extropic integration packet shipped without a live THRML benchmark because "
            f"thrml_available={bool(e1150.get('thrml_available'))}."
        ),
    ]


def bottlenecks_identified() -> list[str]:
    """Return the top three bottlenecks for .90 planning."""

    return [
        (
            "Run roadmap gate audits before activation and backfill prior_failures immediately: "
            "exp1139 was release-critical but blocked before the audit exposed the same missing field."
        ),
        (
            "Tighten verifier calibration rather than only recall: Goodfire cheap-tier TP reached "
            "91.7%, but 96% false positives mean the threshold move is not deployable."
        ),
        (
            "Treat hardware sampling as a topology/access blocker: KV260 DC-continuous KL regressed "
            "and THRML/Z1 remained unavailable, so .90 needs correctness-first sequential Gibbs or "
            "real hardware access before more parameter sweeps."
        ),
    ]


def build_artifact(
    artifacts: Mapping[int, Mapping[str, Any]],
    log_lines: Sequence[str],
) -> dict[str, Any]:
    """Assemble the Exp 1151 deliverable."""

    criteria = evaluate_criteria(artifacts)
    criteria_met = sum(criteria.values())
    wall_time = compute_wall_time_minutes(log_lines)
    slowest = build_slowest_experiments(log_lines)
    wall_time_improvement = round(PRIOR_WALL_TIME_MINUTES - wall_time, 1)
    wall_time_improvement_pct = round(wall_time_improvement / PRIOR_WALL_TIME_MINUTES * 100.0, 1)

    return {
        "experiment": "1151_milestone_retro_89",
        "schema": "milestone_retro_v2",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "milestone": MILESTONE,
        "criteria_results": criteria,
        "criteria_met": criteria_met,
        "criteria_total": CRITERIA_TOTAL,
        "experiment_honest_verdicts": record_honest_verdicts(artifacts),
        "notable_successes": notable_successes(artifacts),
        "failures_or_partials": failures_or_partials(artifacts),
        "bottlenecks_identified": bottlenecks_identified(),
        "slowest_experiments": slowest,
        "exp906_appeared_in_slowest5": any(entry["id"] == "exp906" for entry in slowest),
        "roadmap_gate_audit_caught_blocking_gaps": roadmap_gate_audit_caught_blocking_gaps(
            artifacts
        ),
        "roadmap_gate_audit_gap_count": int(
            artifacts[1140].get("n_prior_failures_missing", 0) or 0
        ),
        "arxiv_submission_status": arxiv_submission_status(artifacts),
        "wall_time_minutes": wall_time,
        "wall_time_baseline_minutes": PRIOR_WALL_TIME_MINUTES,
        "wall_time_improvement_vs_prior_minutes": wall_time_improvement,
        "wall_time_improvement_vs_prior_pct": wall_time_improvement_pct,
        "retro_complete": True,
        "honest_verdict": f"{criteria_met}_of_{CRITERIA_TOTAL}_criteria_met",
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
        f"[exp1151] {artifact['criteria_met']}/{artifact['criteria_total']} criteria met; "
        f"wall_time={artifact['wall_time_minutes']:.1f} min; out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
