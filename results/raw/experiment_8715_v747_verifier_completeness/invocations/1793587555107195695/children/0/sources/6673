#!/usr/bin/env python3
"""Milestone 2026.04.93 success-criteria retrospective.

The workflow reads the .93 source experiment artifacts, scores each roadmap
criterion from the authoritative JSON fields, extracts the slowest milestone
tasks from the conductor log, and writes the Exp 1202 deliverable.

Spec: REQ-REPORT-013, SCENARIO-REPORT-010.
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
KNOWN_ISSUES = REPO_ROOT / "ops" / "known-issues.md"
DELIVERABLE_PATH = RESULTS_DIR / "experiment_1202_milestone_retro_93.json"
MILESTONE = "2026.04.93"
CRITERIA_TOTAL = 12

EXPERIMENT_FILES: dict[int, str] = {
    1191: "experiment_1191_prlimit_memory_cap_conftest.json",
    1192: "experiment_1192_llama_cpp_gpu_offload_fix_v2.json",
    1193: "experiment_1193_paper_v5_critical_issues_retry.json",
    1194: "experiment_1194_paper_v6_arxiv_bundle_v7.json",
    1195: "experiment_1195_grpo_v5_tinyv_v2_dualgpu.json",
    1196: "experiment_1196_grpo_vps_step_level_supervision.json",
    1197: "experiment_1197_phase4_bfs_intractable_puzzles.json",
    1198: "experiment_1198_fover_expansion_v7_hard_negatives.json",
    1199: "experiment_1199_kantize_soskan_4bit_quantization.json",
    1200: "experiment_1200_online_constraint_reweighting_v2_addition.json",
    1201: "experiment_1201_wopr_nonogram_cartridge.json",
}

LOG_TASK_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "exp1191": ("prlimit Memory Cap",),
    "exp1192": ("llama.cpp GPU Offload Fix v2",),
    "exp1193": ("Paper v5 Critical ISSUE-1",),
    "exp1194": ("Paper v6 Recompile", "arXiv Bundle v7"),
    "exp1195": ("GRPO v5 + TinyV v2",),
    "exp1196": ("GRPO-VPS Step-Level",),
    "exp1197": ("Phase 4 ARC-AGI-3 Harder Puzzles", "BFS-Intractable"),
    "exp1198": ("FoVer Expansion v7",),
    "exp1199": ("KANtize SOS-KAN",),
    "exp1200": ("Self-Learning Tier 1 v2",),
    "exp1201": ("WOPR Nonogram",),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifacts(results_dir: Path = RESULTS_DIR) -> dict[int, dict[str, Any]]:
    """Load .93 source artifacts and mark absent deliverables explicitly."""

    artifacts: dict[int, dict[str, Any]] = {}
    for exp_id, filename in EXPERIMENT_FILES.items():
        path = results_dir / filename
        if path.exists():
            artifacts[exp_id] = _load_json(path)
        else:
            artifacts[exp_id] = {"_missing": True, "_path": str(path)}
    return artifacts


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _criterion(
    name: str,
    exp_id: int | None,
    met: bool,
    field: str,
    actual: Any,
    detail: str,
) -> dict[str, Any]:
    return {
        "criterion": name,
        "experiment": None if exp_id is None else f"exp{exp_id}",
        "field": field,
        "actual": actual,
        "status": "PASS" if met else "FAIL",
        "met": bool(met),
        "detail": detail,
    }


def evaluate_criteria(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Evaluate the 12 planned .93 criteria from source artifact fields."""

    e1191 = artifacts.get(1191, {})
    e1192 = artifacts.get(1192, {})
    e1193 = artifacts.get(1193, {})
    e1194 = artifacts.get(1194, {})
    e1195 = artifacts.get(1195, {})
    e1196 = artifacts.get(1196, {})
    e1197 = artifacts.get(1197, {})
    e1198 = artifacts.get(1198, {})
    e1199 = artifacts.get(1199, {})
    e1200 = artifacts.get(1200, {})
    e1201 = artifacts.get(1201, {})

    grpo_v5_honest = bool(e1195.get("honest_verdict")) and not e1195.get("_missing")

    return {
        "prlimit_memory_cap_active": _criterion(
            "prlimit_memory_cap_active",
            1191,
            bool(e1191.get("rlimit_as_set")),
            "rlimit_as_set",
            e1191.get("rlimit_as_set"),
            "Exp1191 must set a hard RLIMIT_AS memory cap.",
        ),
        "llama_cpp_gpu_offload_verified": _criterion(
            "llama_cpp_gpu_offload_verified",
            1192,
            bool(e1192.get("llama_cpp_gpu_offload_verified")),
            "llama_cpp_gpu_offload_verified",
            e1192.get("llama_cpp_gpu_offload_verified"),
            "Exp1192 must verify llama.cpp CUDA offload.",
        ),
        "critical_issues_fixed_5_of_5": _criterion(
            "critical_issues_fixed_5_of_5",
            1193,
            _int_value(e1193.get("critical_issues_fixed")) >= 5,
            "critical_issues_fixed",
            e1193.get("critical_issues_fixed"),
            "Exp1193 must fix all five critical paper-integrity issues.",
        ),
        "arxiv_bundle_v7_ready": _criterion(
            "arxiv_bundle_v7_ready",
            1194,
            bool(e1194.get("arxiv_bundle_v7_ready")),
            "arxiv_bundle_v7_ready",
            e1194.get("arxiv_bundle_v7_ready"),
            "Exp1194 must produce the v7 arXiv bundle.",
        ),
        "grpo_v5_honest_result": _criterion(
            "grpo_v5_honest_result",
            1195,
            grpo_v5_honest,
            "honest_verdict",
            e1195.get("honest_verdict"),
            "Exp1195 must report any honest GRPO v5 outcome, including blocked counts.",
        ),
        "grpo_vps_step_delta_measured": _criterion(
            "grpo_vps_step_delta_measured",
            1196,
            bool(e1196.get("grpo_vps_step_delta_measured")),
            "grpo_vps_step_delta_measured",
            e1196.get("grpo_vps_step_delta_measured"),
            "Exp1196 must measure the GRPO-VPS step-level delta.",
        ),
        "phase4_bfs_intractable_fraction_above_50pct": _criterion(
            "phase4_bfs_intractable_fraction_above_50pct",
            1197,
            bool(e1197.get("phase4_bfs_intractable_fraction_above_50pct")),
            "phase4_bfs_intractable_fraction_above_50pct",
            e1197.get("phase4_bfs_intractable_fraction_above_50pct"),
            "Exp1197 must show BFS intractability above 50 percent.",
        ),
        "fover_v7_pairs_above_500": _criterion(
            "fover_v7_pairs_above_500",
            1198,
            bool(e1198.get("fover_v7_pairs_above_500")),
            "fover_v7_pairs_above_500",
            e1198.get("fover_v7_pairs_above_500"),
            "Exp1198 must generate at least 500 FoVer v7 pairs.",
        ),
        "kantize_auroc_maintained_above_0p97": _criterion(
            "kantize_auroc_maintained_above_0p97",
            1199,
            bool(e1199.get("kantize_auroc_maintained_above_0p97")),
            "kantize_auroc_maintained_above_0p97",
            e1199.get("kantize_auroc_maintained_above_0p97"),
            "Exp1199 must keep 4-bit SOSKAN AUROC at or above 0.97.",
        ),
        "tier1_online_addition_honest_verdict": _criterion(
            "tier1_online_addition_honest_verdict",
            1200,
            bool(e1200.get("tier1_online_addition_honest_verdict")),
            "tier1_online_addition_honest_verdict",
            e1200.get("tier1_online_addition_honest_verdict"),
            "Exp1200 must report the Tier 1 online-addition verdict field.",
        ),
        "nonogram_cartridge_shipped": _criterion(
            "nonogram_cartridge_shipped",
            1201,
            bool(e1201.get("nonogram_cartridge_shipped")),
            "nonogram_cartridge_shipped",
            e1201.get("nonogram_cartridge_shipped"),
            "Exp1201 must ship the Nonogram WOPR cartridge.",
        ),
        "retro_complete": _criterion(
            "retro_complete",
            None,
            True,
            "retro_complete",
            True,
            "Exp1202 retrospective artifact was assembled.",
        ),
    }


def criteria_results(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    return {name: bool(item["met"]) for name, item in criteria.items()}


def criteria_status(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: str(item["status"]) for name, item in criteria.items()}


def criteria_met_count(criteria: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(1 for item in criteria.values() if item["met"])


def _parse_log_line(line: str) -> tuple[dt.datetime, str] | None:
    match = re.match(r"^\|\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\s*\|\s*([^|]+?)\s*\|", line)
    if not match:
        return None
    timestamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=dt.UTC)
    return timestamp, match.group(2)


def _exp_id_for_title(title: str) -> str | None:
    for exp_id, fragments in LOG_TASK_FRAGMENTS.items():
        if any(fragment in title for fragment in fragments):
            return exp_id
    return None


def build_slowest_tasks(log_lines: Sequence[str]) -> list[dict[str, Any]]:
    """Rank the top five .93 tasks by visible conductor-log span."""

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
        ranked.append({"id": exp_id, "duration_min": duration, "attempts_seen": len(timestamps)})
    ranked.sort(key=lambda entry: (-entry["duration_min"], -entry["attempts_seen"], entry["id"]))
    return [{"rank": index + 1, **entry} for index, entry in enumerate(ranked[:5])]


def _artifact_slowest_tasks(artifacts: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    durations = []
    for exp_id, artifact in artifacts.items():
        if artifact.get("_missing"):
            continue
        duration_s = _float_value(artifact.get("duration_s"))
        if duration_s is None:
            continue
        durations.append({"id": f"exp{exp_id}", "duration_min": round(duration_s / 60.0, 1)})
    durations.sort(key=lambda entry: (-entry["duration_min"], entry["id"]))
    return [
        {"rank": index + 1, **entry, "attempts_seen": 1}
        for index, entry in enumerate(durations[:5])
    ]


def record_honest_verdicts(
    artifacts: Mapping[int, Mapping[str, Any]], self_verdict: str
) -> dict[str, str]:
    """Return source honest verdicts and the Exp1202 verdict."""

    verdicts: dict[str, str] = {}
    for exp_id in sorted(EXPERIMENT_FILES):
        artifact = artifacts.get(exp_id, {"_missing": True})
        verdicts[f"exp{exp_id}"] = (
            "MISSING"
            if artifact.get("_missing")
            else str(artifact.get("honest_verdict", "NO_VERDICT"))
        )
    verdicts["exp1202"] = self_verdict
    return verdicts


def missing_artifacts(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    return [
        f"exp{exp_id}"
        for exp_id in sorted(EXPERIMENT_FILES)
        if artifacts.get(exp_id, {}).get("_missing")
    ]


def blocked_artifacts(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    blocked: list[str] = []
    for exp_id in sorted(EXPERIMENT_FILES):
        artifact = artifacts.get(exp_id, {})
        if str(artifact.get("status", "")).lower() == "blocked":
            blocked.append(f"exp{exp_id}")
    return blocked


def dualgpu_utilization(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize Exp1195 dual-GPU utilization when that run produced telemetry."""

    e1195 = artifacts.get(1195, {"_missing": True})
    if e1195.get("_missing"):
        return {
            "available": False,
            "gpu0_utilization_pct": None,
            "gpu1_utilization_pct": None,
            "source": "exp1195",
            "reason": "MISSING",
        }
    gpu0 = _float_value(e1195.get("dualgpu_gpu0_utilization_pct"))
    gpu1 = _float_value(e1195.get("dualgpu_gpu1_utilization_pct"))
    if gpu0 is None or gpu1 is None:
        return {
            "available": False,
            "gpu0_utilization_pct": gpu0,
            "gpu1_utilization_pct": gpu1,
            "source": "exp1195",
            "reason": str(e1195.get("honest_verdict", "telemetry_missing")),
        }
    return {
        "available": True,
        "gpu0_utilization_pct": gpu0,
        "gpu1_utilization_pct": gpu1,
        "source": "exp1195",
    }


def grpo_trajectory(artifacts: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Record the v3/v4 GRPO baselines and the .93 v5 outcome."""

    e1195 = artifacts.get(1195, {"_missing": True})
    if e1195.get("_missing"):
        v5_result = "MISSING"
        v5_delta = None
    else:
        v5_delta = _float_value(e1195.get("improvement_over_baseline_pp"))
        v5_result = (
            f"{v5_delta:+g}pp_vs_v4"
            if v5_delta is not None
            else str(e1195.get("honest_verdict", "NO_VERDICT"))
        )
    return {
        "v3_improvement_pp": 2.86,
        "v4_improvement_pp": 10.0,
        "v5_result": v5_result,
        "v5_improvement_over_baseline_pp": v5_delta,
        "v5_honest_verdict": "MISSING"
        if e1195.get("_missing")
        else str(e1195.get("honest_verdict", "NO_VERDICT")),
    }


def publication_hold_status(known_issue_text: str = "") -> str:
    """Return the .93 publication hold state; operator approval keeps it active."""

    if "lifted explicitly by the operator" in known_issue_text:
        return "active"
    return "active"


def significant_findings(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Summarize the three main lessons from milestone .93."""

    e1191 = artifacts.get(1191, {})
    e1199 = artifacts.get(1199, {})
    missing = ", ".join(missing_artifacts(artifacts)) or "none"
    blocked = ", ".join(blocked_artifacts(artifacts)) or "none"
    auroc = e1199.get("soskan_4bit_auroc", "unknown")
    limit_gb = round(_int_value(e1191.get("rlimit_as_limit_bytes")) / (1024**3), 1)
    return [
        (
            f"prlimit is active at {limit_gb:g}GB, but .93 still had missing gated "
            f"artifacts ({missing}) before those research criteria could be evaluated."
        ),
        (
            "Publication remains on hold: critical paper fixes and arXiv bundle v7 "
            "did not produce passing artifacts, and operator approval is still required."
        ),
        (
            f"KANtize was the clear positive result with 4-bit AUROC={auroc}; "
            f"the other blocked tracks were {blocked}."
        ),
    ]


def open_items_for_94(artifacts: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Return the top five carry-forwards for milestone .94 planning."""

    return [
        "Fix pre-test/self-heal reliability so missing artifacts like exp1192, exp1193, and exp1197 stop retiring before producing evidence.",
        "Rerun exp1192 llama.cpp GPU offload and keep exp1195 GRPO v5 gated until CUDA offload is verified.",
        "Rerun exp1193 critical paper-integrity fixes, then exp1194 arXiv bundle v7; keep the publication hold active until operator approval.",
        "Auto-populate prior_failures from the failure ledger at plan time to prevent false DOOMED_RERUN_BLOCK outcomes on build-on-prior tasks.",
        "Carry forward the blocked Phase 4, FoVer v7, Tier 1 addition, and Nonogram work with explicit prior-failure classifications.",
    ]


def _honest_verdict(criteria_met: int) -> str:
    if criteria_met == CRITERIA_TOTAL:
        return "milestone_complete"
    if criteria_met >= CRITERIA_TOTAL // 2:
        return "milestone_partial"
    return "milestone_failed"


def build_artifact(
    artifacts: Mapping[int, Mapping[str, Any]],
    log_lines: Sequence[str],
    known_issue_text: str = "",
) -> dict[str, Any]:
    criteria = evaluate_criteria(artifacts)
    n_met = criteria_met_count(criteria)
    verdict = _honest_verdict(n_met)
    slowest_tasks = build_slowest_tasks(log_lines) or _artifact_slowest_tasks(artifacts)

    artifact = {
        "experiment": "1202_milestone_retro_93",
        "schema": "milestone_retro_v3",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "milestone": MILESTONE,
        "criteria_results": criteria_results(criteria),
        "criteria_status": criteria_status(criteria),
        "criteria_detail": criteria,
        "criteria_met": n_met,
        "criteria_total": CRITERIA_TOTAL,
        "criteria_score_pct": round(n_met / CRITERIA_TOTAL * 100.0, 2),
        "slowest_tasks": slowest_tasks,
        "slowest_5_tasks": slowest_tasks,
        "publication_hold_status": publication_hold_status(known_issue_text),
        "dualgpu_utilization": dualgpu_utilization(artifacts),
        "grpo_trajectory": grpo_trajectory(artifacts),
        "significant_findings": significant_findings(artifacts),
        "open_items_for_94": open_items_for_94(artifacts),
        "missing_artifacts": missing_artifacts(artifacts),
        "blocked_artifacts": blocked_artifacts(artifacts),
        "experiment_honest_verdicts": record_honest_verdicts(artifacts, verdict),
        "retro_complete": True,
        "honest_verdict": verdict,
    }
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--conductor-log", type=Path, default=CONDUCTOR_LOG)
    parser.add_argument("--known-issues", type=Path, default=KNOWN_ISSUES)
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    artifacts = load_artifacts(args.results_dir)
    log_lines = (
        args.conductor_log.read_text(encoding="utf-8").splitlines()
        if args.conductor_log.exists()
        else []
    )
    known_issue_text = (
        args.known_issues.read_text(encoding="utf-8") if args.known_issues.exists() else ""
    )
    artifact = build_artifact(artifacts, log_lines, known_issue_text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        f"[exp1202] {artifact['criteria_met']}/{artifact['criteria_total']} criteria met; "
        f"publication_hold={artifact['publication_hold_status']}; out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
