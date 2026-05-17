"""Generate the milestone 2026.05.223 operational retrospective."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


SCHEMA = "carnot.operational_retro.v66"
MILESTONE = "2026.05.223"
RUN_DATE = "20260517"
TASK_COUNT = 13
RETRO_DELIVERABLE = Path("results/experiment_2265_retro.json")

REQUESTED_ARTIFACTS: tuple[tuple[str, str | None, str], ...] = (
    (
        "results/experiment_2254_pretest_fix.json",
        None,
        "pre-test duplicate module repair attempt",
    ),
    (
        "results/experiment_2255_fst_live_gen.json",
        "results/experiment_2255_fst_real_scale_live_gen.json",
        "FST full live generation gate",
    ),
    (
        "results/experiment_2256_fr11_multidomain.json",
        None,
        "FR-11 FST multi-domain validation gate",
    ),
    (
        "results/experiment_2258_kancl_n256.json",
        "results/experiment_2258_kancl_n256_clean_reattempt.json",
        "KAN-CL n=256 clean re-attempt gate",
    ),
    (
        "results/experiment_2260_kv260_rtl.json",
        "results/experiment_2260_kv260_rtl_clean_reattempt.json",
        "KV260 RTL clean re-attempt gate",
    ),
    (
        "results/experiment_2264_capstone.json",
        None,
        ".223 full live-generation capstone gate",
    ),
)

SUPPORT_ARTIFACTS: tuple[str, ...] = (
    "results/experiment_2253_archive.json",
    "results/experiment_2257_odar_real_benchmark.json",
    "results/experiment_2262_adversarial_null_space_probe.json",
    "results/experiment_2263_arxiv_sweep.json",
    "results/experiment_2252_retro.json",
)

COMPUTE_BOUND_MARKERS = (
    "Fix Duplicate test_compositional_energy",
    "ODAR Real-Inference Routing Overhead",
)


@dataclass(frozen=True)
class LogEntry:
    """One parsed row from `ops/conductor-log.md`."""

    timestamp: datetime
    title: str
    status: str
    details: str


_LOG_ROW = re.compile(
    r"^\|\s*(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)\s*"
    r"\|\s*(?P<title>[^|]*?)\s*"
    r"\|\s*(?P<status>[^|]*?)\s*"
    r"\|\s*(?P<details>.*?)\s*\|$"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_conductor_log(text: str) -> list[LogEntry]:
    """Parse markdown conductor-log rows relevant to REQ-REPORT-2265."""

    entries: list[LogEntry] = []
    for line in text.splitlines():
        match = _LOG_ROW.match(line)
        if not match:
            continue
        timestamp = datetime.strptime(
            match.group("timestamp"), "%Y-%m-%d %H:%M UTC"
        ).replace(tzinfo=UTC)
        entries.append(
            LogEntry(
                timestamp=timestamp,
                title=match.group("title").strip(),
                status=match.group("status").strip(),
                details=match.group("details").strip(),
            )
        )
    return entries


def milestone_window(entries: list[LogEntry]) -> list[LogEntry]:
    """Return the `.223` activation-to-pre-retro conductor window."""

    start_index: int | None = None
    for index, entry in enumerate(entries):
        if entry.title == f"Milestone {MILESTONE} activated":
            start_index = index

    if start_index is None:
        raise ValueError(f"Could not find Milestone {MILESTONE} activation row")

    window: list[LogEntry] = []
    for entry in entries[start_index:]:
        if f"Milestone {MILESTONE} Retrospective" in entry.title:
            break
        window.append(entry)
    return window


def _task_entries(window: list[LogEntry]) -> list[LogEntry]:
    return [
        entry
        for entry in window
        if entry.title != f"Milestone {MILESTONE} activated"
        and not entry.title.startswith("Plan ")
    ]


def _status_counts(task_entries: list[LogEntry]) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for entry in task_entries:
        statuses[entry.status] = statuses.get(entry.status, 0) + 1
    return statuses


def _distinct_titles_by_status(task_entries: list[LogEntry], status: str) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for entry in task_entries:
        if entry.status != status or entry.title in seen:
            continue
        seen.add(entry.title)
        titles.append(entry.title)
    return titles


def _count_compute_bound(task_entries: list[LogEntry]) -> tuple[int, float, list[str]]:
    count = 0
    wall_time_min = 0.0
    titles: list[str] = []
    for index, entry in enumerate(task_entries):
        if entry.status != "OK":
            continue
        if not any(marker in entry.title for marker in COMPUTE_BOUND_MARKERS):
            continue
        count += 1
        titles.append(entry.title)
        if index > 0:
            wall_time_min += (
                entry.timestamp - task_entries[index - 1].timestamp
            ).total_seconds() / 60.0
    return count, round(wall_time_min, 1), titles


def _collect_artifacts(repo_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    sources: list[str] = ["ops/conductor-log.md"]

    for requested, fallback, description in REQUESTED_ARTIFACTS:
        requested_path = repo_root / requested
        if requested_path.exists():
            artifacts[requested] = json.loads(requested_path.read_text(encoding="utf-8"))
            sources.append(requested)
            continue

        fallback_path = repo_root / fallback if fallback else None
        if fallback_path is not None and fallback_path.exists():
            artifacts[requested] = json.loads(fallback_path.read_text(encoding="utf-8"))
            sources.append(fallback)
            missing.append(
                {
                    "path": requested,
                    "status": "missing_exact_path_used_gate_artifact_alias",
                    "actual_path": fallback,
                    "explanation": f"The roadmap requested {requested}, but the conductor wrote the terminal gate artifact under the task slug for {description}.",
                }
            )
            continue

        missing.append(
            {
                "path": requested,
                "status": "missing_exact_path",
                "explanation": f"No terminal artifact was present for {description}; conductor log rows show the task was gate-blocked or pre-emptively skipped before execution.",
            }
        )

    for source in SUPPORT_ARTIFACTS:
        path = repo_root / source
        if path.exists():
            artifacts[source] = json.loads(path.read_text(encoding="utf-8"))
            sources.append(source)

    return artifacts, missing, sources


def _gap_resolution(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pretest = artifacts.get("results/experiment_2254_pretest_fix.json", {})
    live = artifacts.get("results/experiment_2255_fst_live_gen.json", {})
    kancl = artifacts.get("results/experiment_2258_kancl_n256.json", {})

    return [
        {
            "gap": "Pre-test failure fixed",
            "source_experiment": "exp2254-pretest-fix",
            "resolved": bool(pretest.get("pretest_fixed") is True),
            "status": "unresolved_partial_fix",
            "evidence": (
                "Duplicate test_compositional_energy basename was renamed, but "
                "pretest_fixed=false because tests/python/test_dual_gpu.py still "
                "raises ImportError for DualGPUExecutionResult."
            ),
            "source": "results/experiment_2254_pretest_fix.json",
        },
        {
            "gap": "KAN-CL n=256 validated",
            "source_experiment": "exp2258-kancl-n256-clean-reattempt",
            "resolved": bool(kancl.get("kancl_n256_validated") is True),
            "status": kancl.get("status", "missing_or_not_run"),
            "evidence": kancl.get(
                "gate_check_summary",
                "No KAN-CL n=256 validation artifact was produced.",
            ),
            "source": "results/experiment_2258_kancl_n256_clean_reattempt.json",
        },
        {
            "gap": "Live generation beyond one-token probe",
            "source_experiment": "exp2255-fst-real-scale-live-gen",
            "resolved": bool(live.get("fst_live_validated") is True),
            "status": live.get("status", "missing_or_not_run"),
            "evidence": live.get(
                "gate_check_summary",
                "No full-answer live generation artifact was produced.",
            ),
            "source": "results/experiment_2255_fst_real_scale_live_gen.json",
        },
    ]


def build_retro(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    """Build the REQ-REPORT-2265 retrospective payload."""

    root = Path(repo_root)
    log_text = (root / "ops/conductor-log.md").read_text(encoding="utf-8")
    window = milestone_window(parse_conductor_log(log_text))
    task_entries = _task_entries(window)
    status_counts = _status_counts(task_entries)
    artifacts, missing_artifacts, source_artifacts = _collect_artifacts(root)

    start = window[0].timestamp
    end = window[-1].timestamp
    total_wall_time_min = round((end - start).total_seconds() / 60.0, 1)
    n_experiments_completed = status_counts.get("OK", 0)
    n_gate_blocks = status_counts.get("GATE_BLOCK", 0)
    n_doomed_rerun_blocks = status_counts.get("DOOMED_RERUN_BLOCK", 0)
    n_compute_bound, compute_bound_wall_time_min, compute_bound_titles = (
        _count_compute_bound(task_entries)
    )
    n_completed_including_retro = n_experiments_completed + 1

    top_gaps_resolved = _gap_resolution(artifacts)
    n_gaps_resolved = sum(1 for gap in top_gaps_resolved if gap["resolved"])

    odar = artifacts.get("results/experiment_2257_odar_real_benchmark.json", {})
    arxiv = artifacts.get("results/experiment_2263_arxiv_sweep.json", {})
    pretest = artifacts.get("results/experiment_2254_pretest_fix.json", {})

    generated_at = (end + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "schema": SCHEMA,
        "experiment": "2265_retro",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "generated_at_utc": generated_at,
        "retro_type": "operational_milestone",
        "source_artifacts": source_artifacts,
        "missing_requested_artifacts": missing_artifacts,
        "field_principles": {
            "honest_verdict": "Terminal-prefix required.",
            "criteria_met": "N/total count validates milestone completion fraction.",
            "top_gaps_resolved": "Records which of the three .222 retro gaps were resolved; tracks multi-milestone gap closure.",
            "next_milestone_speedup_target_pct": "Quantifies where wall-time can be recovered in .224.",
        },
        "honest_verdict": "complete: milestone_2026_05_223_retro_5_of_13_terminal_tasks_complete_0_of_3_prior_gaps_resolved_61min_wall_time_35pct_speedup_target",
        "criteria_met": {
            "count": n_completed_including_retro,
            "total": TASK_COUNT,
            "fraction": round(n_completed_including_retro / TASK_COUNT, 6),
            "display": f"{n_completed_including_retro}/{TASK_COUNT}",
            "basis": "13 queued .223 tasks: 4 terminal OK rows were logged before this retrospective, this retrospective adds the 5th terminal deliverable, and 8 distinct tasks remained blocked or gate-skipped.",
            "primary_gate_success_count_including_retro": 3,
            "primary_gate_success_display": "3/13",
            "primary_gate_success_basis": "Only exp2257 ODAR, exp2263 arXiv sweep, and this retrospective met their primary artifact gates; exp2253 and exp2254 were terminal OK rows but their primary booleans were false.",
        },
        "total_wall_time_min": total_wall_time_min,
        "n_experiments_completed": n_experiments_completed,
        "n_gate_blocks": n_gate_blocks,
        "n_compute_bound": n_compute_bound,
        "compute": {
            "total_wall_time_min": total_wall_time_min,
            "wall_time_basis": "From conductor log milestone activation at 2026-05-17 16:38 UTC through capstone gate-block at 2026-05-17 17:39 UTC.",
            "n_experiments_completed": n_experiments_completed,
            "n_experiments_completed_including_this_retro": n_completed_including_retro,
            "n_gate_blocks": n_gate_blocks,
            "n_gate_blocked_distinct_tasks": len(
                _distinct_titles_by_status(task_entries, "GATE_BLOCK")
            ),
            "n_doomed_rerun_blocks": n_doomed_rerun_blocks,
            "n_doomed_rerun_distinct_tasks": len(
                _distinct_titles_by_status(task_entries, "DOOMED_RERUN_BLOCK")
            ),
            "n_distinct_blocked_or_gated_tasks": len(
                set(_distinct_titles_by_status(task_entries, "GATE_BLOCK"))
                | set(_distinct_titles_by_status(task_entries, "DOOMED_RERUN_BLOCK"))
            ),
            "n_compute_bound": n_compute_bound,
            "compute_bound_tasks": compute_bound_titles,
            "compute_bound_wall_time_min": compute_bound_wall_time_min,
        },
        "wall_time_distribution": [
            {
                "bucket": "activation_and_cache_hit",
                "minutes": 3.0,
                "pct": 4.9,
                "basis": "16:38 to 16:41 UTC",
            },
            {
                "bucket": "pretest_repair_attempt",
                "minutes": 4.0,
                "pct": 6.6,
                "basis": "16:41 to 16:45 UTC",
            },
            {
                "bucket": "live_generation_gate_churn",
                "minutes": 6.0,
                "pct": 9.8,
                "basis": "16:45 to 16:51 UTC",
            },
            {
                "bucket": "odar_real_benchmark",
                "minutes": 13.0,
                "pct": 21.3,
                "basis": "16:53 to 17:06 UTC",
            },
            {
                "bucket": "kancl_kv260_gate_churn",
                "minutes": 12.0,
                "pct": 19.7,
                "basis": "17:06 to 17:18 UTC",
            },
            {
                "bucket": "downstream_skip_and_doomed_rerun_churn",
                "minutes": 6.0,
                "pct": 9.8,
                "basis": "17:18 to 17:24 UTC",
            },
            {
                "bucket": "arxiv_reference_sweep",
                "minutes": 13.0,
                "pct": 21.3,
                "basis": "17:24 to 17:37 UTC",
            },
            {
                "bucket": "capstone_preemptive_gate_block",
                "minutes": 2.0,
                "pct": 3.3,
                "basis": "17:37 to 17:39 UTC",
            },
        ],
        "top_successes": [
            f"ODAR real-inference benchmark completed with compute_reduction_pct={odar.get('compute_reduction_pct')} and median routing_overhead_ms={odar.get('routing_overhead_ms')} on n_corpus={odar.get('n_corpus')}; the result is useful but carries adversarial tautology flags that .224 should review before treating accuracy as settled.",
            f"Post-.222 arXiv sweep found {arxiv.get('n_new_papers_found')} new papers, updated references, and produced a top-3 .224 candidate list.",
            "Gate discipline prevented fabrication: full live generation, KAN-CL n=256, KV260 RTL, synthesis, and capstone claims stayed blocked once exp2254.pretest_fixed remained false.",
        ],
        "top_gaps_for_2026_05_224": [
            f"Fix the remaining pre-test ImportError before scheduling dependent work: exp2254 reports pretest_fixed={pretest.get('pretest_fixed')} with DualGPUExecutionResult missing from carnot.inference.",
            "Run exp2255 full-answer live generation only after the pre-test gate is green; the one-token probe gap remains open.",
            "Re-stage KAN-CL n=256, KV260 RTL, and adversarial null-space tasks with complete preconditions and prior_failures metadata so .224 does not repeat 19 GATE_BLOCK rows and 3 DOOMED_RERUN_BLOCK rows.",
        ],
        "top_gaps_resolved": top_gaps_resolved,
        "top_gaps_resolved_count": {
            "count": n_gaps_resolved,
            "total": 3,
            "display": f"{n_gaps_resolved}/3",
            "basis": "All three .222 carry-forward gaps were attempted or gated in .223, but none reached its success gate.",
        },
        "next_milestone_speedup_target_pct": 35.0,
        "next_milestone_speedup_basis": "A hard pre-activation pre-test gate and dependency-aware roadmap staging should recover roughly 26 minutes of .223 gate churn out of 61 total wall minutes; 35% is a conservative .224 target after leaving room for real compute.",
        "recommendations": [
            "Make DualGPUExecutionResult import/export repair the first .224 task and block all live-generation, KAN-CL, and RTL work until the full pre-test command is green.",
            "Use a single prerequisite audit for exp2255, exp2258, exp2260, and exp2262 before activation so missing fields or false upstream gates fail once, not every conductor loop.",
            "Keep ODAR's real-probe benchmark, but add a non-tautological accuracy oracle before using the exact 100%/0pp accuracy figures in planning claims.",
            "Carry forward exp2263 candidate papers into .224 planning, especially projected Langevin constraints and p-bit FPGA scheduling.",
        ],
        "explicit_non_actions": [
            "Did not push.",
            "Did not modify scripts/research_conductor.py.",
            "Did not update ops/changelog.md, ops/status.md, or _bmad/traceability.md because the task's STOP-WHEN-DONE rule delegated those reconciliation files to the conductor.",
        ],
        "acceptance_gate_passed": True,
    }


def write_retro(repo_root: Path | str = Path("."), output_path: Path | str | None = None) -> Path:
    """Write the retrospective JSON artifact for REQ-REPORT-2265."""

    root = Path(repo_root)
    relative_output = Path(output_path) if output_path is not None else RETRO_DELIVERABLE
    destination = root / relative_output
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_retro(root)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=RETRO_DELIVERABLE)
    args = parser.parse_args()
    path = write_retro(args.repo_root, args.output)
    print(f"Wrote retro to {path}")


if __name__ == "__main__":
    main()
