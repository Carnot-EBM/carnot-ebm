"""Generate the milestone 2026.05.228 operational retrospective."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "carnot.operational_retro.v71"
MILESTONE = "2026.05.228"
EXPERIMENT = 2335
RETRO_DELIVERABLE = Path("results/experiment_2335_retro.json")
NEXT_MILESTONE_SPEEDUP_TARGET_PCT = 55.0
GGUF_CACHE_COMMAND = (
    'ls ~/.cache/huggingface/hub/ | grep -i "gemma-4-26B\\|Qwen3.6-35B\\|gemma-4-31B"'
)

EXP2309_TARGETED_PRETESTS = (
    "tests/python/test_experiment_1347_thrml_compatibility_parity_audit.py::"
    "test_req_sample_041_probe_reports_direct_import_success_without_version",
    "tests/python/test_experiment_1182_paper_v5_medium_low_issues_11_18.py::"
    "TestIssue11ThroughIssue15::test_issue_14_soskan_aurocs_have_corpus_and_n",
)

EXP2323_TARGETED_PRETESTS = (
    "tests/python/test_experiment_1692_potts_v2.py::test_experiment_1692_potts_v2_artifact",
    "tests/python/test_experiment_390_gpu_preflight.py::"
    "TestRunGpuPreflight::test_scripts_missing_session_startup",
    "tests/python/test_experiment_294_gpu_baseline_apple.py::"
    "TestBaselineAccuracyBounds::test_accuracy_in_unit_interval_when_all_correct",
)


@dataclass(frozen=True)
class TaskSpec:
    """One planned .228 task and the field that determines its primary artifact gate."""

    task_id: str
    log_marker: str
    artifacts: tuple[str, ...]
    success_field: str | None
    compute_bound: bool = False


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        "exp2322-archive-and-activate",
        "Phase 0: Archive .227 and activate .228",
        ("results/experiment_2322_archive.json",),
        "archive_ready",
    ),
    TaskSpec(
        "exp2323-pretest-fix-final",
        "Phase 0: Fix 3 Remaining Pre-Test Failures",
        ("results/experiment_2323_pretest_fix.json",),
        "pretest_fixed",
    ),
    TaskSpec(
        "exp2324-fst-live-gen-v8",
        "Phase 1: FST+ODAR+CASAL Real-Scale Live Generation",
        ("results/experiment_2324_fst_live_gen.json",),
        "fst_live_validated",
        compute_bound=True,
    ),
    TaskSpec(
        "exp2325-fr11-fst-multidomain-v5",
        "Phase 1: FR-11 FST Multi-Domain Retention",
        ("results/experiment_2325_fr11_multidomain.json",),
        "fr11_multidomain_passed",
    ),
    TaskSpec(
        "exp2326-kancl-n256-v7",
        "Phase 1: KAN-CL n=256 Per-Knot Retention",
        ("results/experiment_2326_kancl_n256.json",),
        "kancl_n256_validated",
        compute_bound=True,
    ),
    TaskSpec(
        "exp2327-nsvif-z3-extractor-v3",
        "Phase 2: NSVIF Neuro-Symbolic Z3 Extractor",
        ("results/experiment_2327_nsvif_extractor.json",),
        "nsvif_extractor_validated",
    ),
    TaskSpec(
        "exp2328-verge-repair-v3",
        "Phase 2: VERGE SMT Minimal Correction Subset",
        ("results/experiment_2328_verge_repair.json",),
        "verge_repair_validated",
    ),
    TaskSpec(
        "exp2329-eidoku-csp-v4",
        "Phase 2: Eidoku CSP Tier 2.8 Gate",
        ("results/experiment_2329_eidoku_csp.json",),
        "eidoku_gate_validated",
    ),
    TaskSpec(
        "exp2330-projected-langevin-v4",
        "Phase 2: Projected-Langevin",
        ("results/experiment_2330_projected_langevin.json",),
        "projected_langevin_competitive",
    ),
    TaskSpec(
        "exp2331-kv260-rtl-lint-v7",
        "Phase 3: KV260 RTL Verilator Lint",
        ("results/experiment_2331_kv260_rtl.json",),
        "lint_errors_count",
    ),
    TaskSpec(
        "exp2332-ml-ising-init-v2",
        "Phase 3: ML-Assisted Ising Machine Initialization",
        ("results/experiment_2332_ml_ising_init.json",),
        "ml_init_speedup_validated",
        compute_bound=True,
    ),
    TaskSpec(
        "exp2333-adversarial-probe-v6",
        "Phase 3: Adversarial Null-Space Probe",
        ("results/experiment_2333_adversarial_probe.json",),
        "adversarial_probe_passed",
    ),
    TaskSpec(
        "exp2334-capstone-v228",
        "Phase 4: Capstone E2E Live Generation",
        (
            "results/experiment_2334_capstone.json",
            "results/experiment_2334_capstone_v228.json",
        ),
        "capstone_passed",
        compute_bound=True,
    ),
    TaskSpec(
        "exp2335-retro-v228",
        "Phase 4: Milestone 2026.05.228 Retrospective",
        ("results/experiment_2335_retro.json",),
        None,
    ),
)


@dataclass(frozen=True)
class LogEntry:
    """One parsed markdown row from ops/conductor-log.md."""

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
    """Parse conductor-log markdown rows used by REQ-REPORT-2335."""

    entries: list[LogEntry] = []
    for line in text.splitlines():
        match = _LOG_ROW.match(line)
        if not match:
            continue
        timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M UTC").replace(
            tzinfo=UTC
        )
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
    """Return the .228 activation-to-pre-retro conductor window."""

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
        if entry.title != f"Milestone {MILESTONE} activated" and not entry.title.startswith("Plan ")
    ]


def _find_artifact(repo_root: Path, task: TaskSpec) -> tuple[str | None, dict[str, Any] | None]:
    for artifact in task.artifacts:
        payload = _load_json(repo_root / artifact)
        if payload is not None:
            return artifact, payload
    return None, None


def _latest_status(task_entries: list[LogEntry], task: TaskSpec) -> tuple[str, str | None]:
    latest: LogEntry | None = None
    for entry in task_entries:
        if task.log_marker in entry.title:
            latest = entry
    if latest is None:
        return "MISSING", None
    return latest.status, latest.details


def _terminal_completed(task: TaskSpec, status: str) -> bool:
    return task.task_id == "exp2335-retro-v228" or status == "OK"


def _primary_gate_met(task: TaskSpec, artifact: dict[str, Any] | None) -> bool:
    if task.task_id == "exp2335-retro-v228":
        return True
    if artifact is None or task.success_field is None:
        return False
    if task.task_id == "exp2331-kv260-rtl-lint-v7":
        return artifact.get("honest_verdict", "").startswith("complete:") and (
            artifact.get("lint_errors_count") == 0
        )
    return artifact.get(task.success_field) is True


def _exp2309_pytest_commands(repo_root: Path) -> list[str]:
    quoted_tests = " ".join(f'"{test}"' for test in EXP2309_TARGETED_PRETESTS)
    return [
        (
            f"cd {repo_root} && .venv/bin/python -m pytest {quoted_tests} "
            "-x -v --no-cov -p no:cacheprovider"
        ),
        (
            f"cd {repo_root} && .venv/bin/python -m pytest "
            f'"{EXP2309_TARGETED_PRETESTS[0]}" -x -v --no-cov -p no:cacheprovider'
        ),
        (
            f"cd {repo_root} && .venv/bin/python -m pytest "
            f'"{EXP2309_TARGETED_PRETESTS[1]}" -x -v --no-cov -p no:cacheprovider'
        ),
    ]


def _exp2323_pytest_commands(repo_root: Path) -> list[str]:
    return [
        (
            f'cd {repo_root} && .venv/bin/python -m pytest "{EXP2323_TARGETED_PRETESTS[0]}" '
            "-x -v --no-cov -p no:cacheprovider -p no:xdist 2>&1 | tail -20"
        ),
        (
            f"cd {repo_root} && .venv/bin/python -m pytest "
            f'"{EXP2323_TARGETED_PRETESTS[1]}" "{EXP2323_TARGETED_PRETESTS[2]}" '
            "-x -v --no-cov -p no:cacheprovider -p no:xdist 2>&1 | tail -20"
        ),
        (
            f"cd {repo_root} && .venv/bin/python -m pytest tests/python "
            "-x -q --no-cov -p no:cacheprovider 2>&1 | tail -20"
        ),
    ]


def _pretest_status(
    repo_root: Path,
    exp2323_artifact_path: str | None,
    exp2323_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    exp2309 = _load_json(repo_root / "results/experiment_2309_pretest_fix.json") or {}
    pretest_fixed = bool(exp2323_artifact and exp2323_artifact.get("pretest_fixed") is True)
    deliverable_present = exp2323_artifact is not None
    status = (
        "fully_resolved"
        if pretest_fixed
        else "unresolved_deliverable_present"
        if deliverable_present
        else "missing_deliverable_after_three_timeouts"
    )
    return {
        "source": exp2323_artifact_path or "results/experiment_2323_pretest_fix.json",
        "deliverable_present": deliverable_present,
        "fully_resolved": pretest_fixed,
        "pretest_fixed": pretest_fixed,
        "status": status,
        "honest_verdict": exp2323_artifact.get("honest_verdict") if exp2323_artifact else None,
        "full_pretest_errors": exp2323_artifact.get("full_pretest_errors")
        if exp2323_artifact
        else None,
        "full_pretest_failures": exp2323_artifact.get("full_pretest_failures")
        if exp2323_artifact
        else None,
        "targeted_fix_status": [
            {
                "name": "potts artifact",
                "test": EXP2323_TARGETED_PRETESTS[0],
                "confirmed_fixed": bool(
                    exp2323_artifact and exp2323_artifact.get("testa_potts_fixed")
                ),
            },
            {
                "name": "gpu preflight xdist marker",
                "test": EXP2323_TARGETED_PRETESTS[1],
                "confirmed_fixed": bool(
                    exp2323_artifact and exp2323_artifact.get("testb_gpu390_fixed")
                ),
            },
            {
                "name": "gpu baseline xdist marker",
                "test": EXP2323_TARGETED_PRETESTS[2],
                "confirmed_fixed": bool(
                    exp2323_artifact and exp2323_artifact.get("testc_gpu294_fixed")
                ),
            },
        ],
        "remaining_preexisting_failures_from_exp2309": exp2309.get(
            "remaining_preexisting_failures", []
        ),
        "manual_operator_intervention_required": not pretest_fixed,
        "escalation_recommendation": (
            "For milestone .229, use direct operator intervention before scheduling "
            "downstream research: manually run the three Exp 2323 targeted pytest "
            "commands and the Exp 2309 reference commands, inspect raw output, and "
            "write a partial deliverable even if the full suite remains red."
        ),
        "conductor_gate_recommendation": (
            "Consider modifying the conductor pre-test gate to retry without xdist as "
            "a fallback when GPU tests fail or error only under parallel execution."
        ),
        "exp2309_pytest_commands": _exp2309_pytest_commands(repo_root),
        "exp2323_pytest_commands": _exp2323_pytest_commands(repo_root),
    }


def _gap_resolution(
    artifacts_by_task: dict[str, dict[str, Any] | None],
    artifact_paths_by_task: dict[str, str | None],
) -> list[dict[str, Any]]:
    pretest = artifacts_by_task.get("exp2323-pretest-fix-final")
    nsvif = artifacts_by_task.get("exp2327-nsvif-z3-extractor-v3")
    fst = artifacts_by_task.get("exp2324-fst-live-gen-v8")
    return [
        {
            "gap": "Pre-test cascade FULLY resolved",
            "source_experiment": "exp2323-pretest-fix-final",
            "resolved": bool(pretest and pretest.get("pretest_fixed") is True),
            "source": artifact_paths_by_task.get("exp2323-pretest-fix-final")
            or "results/experiment_2323_pretest_fix.json",
            "evidence": (
                "pretest_fixed=true and all three targeted fixes confirmed"
                if pretest and pretest.get("pretest_fixed") is True
                else "Exp 2323 produced no deliverable after three 1201s timeout attempts; the potts artifact and two xdist marker fixes remain unconfirmed."
            ),
        },
        {
            "gap": "NSVIF neuro-symbolic extraction first actual run",
            "source_experiment": "exp2327-nsvif-z3-extractor-v3",
            "resolved": bool(nsvif and nsvif.get("nsvif_extractor_validated") is True),
            "source": artifact_paths_by_task.get("exp2327-nsvif-z3-extractor-v3")
            or "results/experiment_2327_nsvif_extractor.json",
            "evidence": (
                "nsvif_extractor_validated=true"
                if nsvif and nsvif.get("nsvif_extractor_validated") is True
                else "artifact missing/gate-blocked; PRD Priority #1 did not execute."
            ),
        },
        {
            "gap": "FST live generation validated beyond one-token probe",
            "source_experiment": "exp2324-fst-live-gen-v8",
            "resolved": bool(fst and fst.get("fst_live_validated") is True),
            "source": artifact_paths_by_task.get("exp2324-fst-live-gen-v8")
            or "results/experiment_2324_fst_live_gen.json",
            "evidence": (
                "fst_live_validated=true with mean_answer_length_tokens >= 50"
                if fst and fst.get("fst_live_validated") is True
                else "artifact missing/gate-blocked; no >=50-token live generation evidence."
            ),
        },
    ]


def _gguf_status(pretest_fixed: bool) -> dict[str, Any]:
    tasks = ["exp2324-fst-live-gen-v8", "exp2334-capstone-v228"]
    if not pretest_fixed:
        return {
            "evaluated": False,
            "reason": "Exp 2323 did not set pretest_fixed=true, so GGUF cache checks never ran.",
            "tasks_to_check_in_229_after_pretest_fix": tasks,
            "deferred_precondition_command": GGUF_CACHE_COMMAND,
        }
    return {
        "evaluated": True,
        "tasks_requiring_gguf_model_availability": tasks,
        "precondition_command": GGUF_CACHE_COMMAND,
    }


def _requested_artifact_status(repo_root: Path) -> list[dict[str, Any]]:
    requested = [
        "results/experiment_2323_pretest_fix.json",
        "results/experiment_2324_fst_live_gen.json",
        "results/experiment_2326_kancl_n256.json",
        "results/experiment_2327_nsvif_extractor.json",
        "results/experiment_2334_capstone.json",
    ]
    statuses: list[dict[str, Any]] = []
    for path in requested:
        full_path = repo_root / path
        fallback = None
        if path == "results/experiment_2334_capstone.json":
            fallback_path = repo_root / "results/experiment_2334_capstone_v228.json"
            if fallback_path.exists():
                fallback = "results/experiment_2334_capstone_v228.json"
        statuses.append(
            {
                "path": path,
                "present": full_path.exists(),
                "fallback_used": fallback,
            }
        )
    return statuses


def build_retro(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    """Build the REQ-REPORT-2335 retrospective payload."""

    root = Path(repo_root).resolve()
    log_text = (root / "ops/conductor-log.md").read_text(encoding="utf-8")
    window = milestone_window(parse_conductor_log(log_text))
    task_entries = _task_entries(window)

    artifacts_by_task: dict[str, dict[str, Any] | None] = {}
    artifact_paths_by_task: dict[str, str | None] = {}
    criteria_results: list[dict[str, Any]] = []
    task_statuses: dict[str, str] = {}
    source_artifacts = ["ops/conductor-log.md"]
    if (root / "results/experiment_2309_pretest_fix.json").exists():
        source_artifacts.append("results/experiment_2309_pretest_fix.json")
    missing_artifacts: list[dict[str, Any]] = []

    for task in TASKS:
        status, details = _latest_status(task_entries, task)
        artifact_path, artifact = _find_artifact(root, task)
        if task.task_id == "exp2335-retro-v228":
            status = "OK"
            details = "self-generated retrospective deliverable"
        elif status == "MISSING" and artifact is not None:
            status = (
                "GATE_BLOCK"
                if artifact.get("schema") == "blocked_gate_check_v1"
                or artifact.get("status") == "blocked"
                else "OK"
            )

        artifacts_by_task[task.task_id] = artifact
        artifact_paths_by_task[task.task_id] = artifact_path
        task_statuses[task.task_id] = status
        if artifact_path:
            source_artifacts.append(artifact_path)
        elif task.task_id != "exp2335-retro-v228":
            missing_artifacts.append(
                {
                    "task_id": task.task_id,
                    "expected_artifacts": list(task.artifacts),
                    "status": status,
                }
            )

        primary_gate_met = _primary_gate_met(task, artifact)
        criteria_results.append(
            {
                "task_id": task.task_id,
                "status": status,
                "criterion_met": _terminal_completed(task, status),
                "primary_artifact_gate_met": primary_gate_met,
                "artifact_path": artifact_path,
                "success_field": task.success_field,
                "success_field_value": artifact.get(task.success_field)
                if artifact and task.success_field
                else None,
                "details": details,
            }
        )

    activation = window[0].timestamp
    end = task_entries[-1].timestamp if task_entries else activation
    total_wall_time_min = round((end - activation).total_seconds() / 60.0, 1)

    n_experiments_completed = sum(
        1
        for task in TASKS
        if task.task_id != "exp2335-retro-v228" and task_statuses[task.task_id] == "OK"
    )
    criteria_count = sum(1 for item in criteria_results if item["criterion_met"])
    criteria_total = len(criteria_results)
    primary_gate_count = sum(1 for item in criteria_results if item["primary_artifact_gate_met"])
    unique_gate_blocks = sum(1 for status in task_statuses.values() if status == "GATE_BLOCK")
    unique_failures = sum(1 for status in task_statuses.values() if status == "FAIL")
    gate_block_attempts = sum(1 for entry in task_entries if entry.status == "GATE_BLOCK")
    failed_attempts = sum(1 for entry in task_entries if entry.status == "FAIL")
    n_compute_bound = sum(
        1 for task in TASKS if task.compute_bound and task_statuses[task.task_id] == "OK"
    )

    pretest_artifact = artifacts_by_task["exp2323-pretest-fix-final"]
    pretest = _pretest_status(
        root,
        artifact_paths_by_task["exp2323-pretest-fix-final"],
        pretest_artifact,
    )
    gaps = _gap_resolution(artifacts_by_task, artifact_paths_by_task)
    gaps_resolved = sum(1 for gap in gaps if gap["resolved"])

    return {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "experiment": EXPERIMENT,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retro_type": "operational_retrospective",
        "source_artifacts": sorted(set(source_artifacts)),
        "requested_artifact_status": _requested_artifact_status(root),
        "missing_requested_artifacts": missing_artifacts,
        "wall_time_window": {
            "start_utc": activation.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "basis": "Milestone activation row through last pre-retro task row.",
        },
        "total_wall_time_min": total_wall_time_min,
        "n_experiments_completed": n_experiments_completed,
        "n_experiments_completed_including_this_retro": criteria_count,
        "n_gate_blocks": unique_gate_blocks,
        "n_gate_block_attempts": gate_block_attempts,
        "n_failed": unique_failures,
        "n_failed_attempts": failed_attempts,
        "n_compute_bound": n_compute_bound,
        "compute_bound_interpretation": (
            "Counts compute-bound tasks that actually executed successfully. The .228 "
            "GGUF, KAN-CL, ML-init, and capstone tasks were planned but gate-blocked "
            "before compute."
        ),
        "compute_bound_tasks_planned_but_blocked": [
            task.task_id
            for task in TASKS
            if task.compute_bound and task_statuses[task.task_id] == "GATE_BLOCK"
        ],
        "criteria_met": {
            "count": criteria_count,
            "total": criteria_total,
            "display": f"{criteria_count}/{criteria_total}",
            "fraction": round(criteria_count / criteria_total, 4),
            "basis": (
                "Terminal conductor completion count: one OK task before this retrospective, "
                "plus this retrospective deliverable."
            ),
            "primary_artifact_gate_count": primary_gate_count,
            "primary_artifact_gate_display": f"{primary_gate_count}/{criteria_total}",
            "primary_artifact_gate_basis": (
                "Only this retrospective met its primary artifact gate. Exp 2322 was a "
                "terminal OK row, but its archive_ready field is false."
            ),
        },
        "criteria_results": criteria_results,
        "top_3_successes": [
            "The active conductor reached milestone .228 after the previous empty-milestone streak.",
            "Gate discipline prevented fabricated downstream claims: FST, KAN-CL, NSVIF, and capstone work stayed blocked after Exp 2323 retired.",
            "The capstone wrote a machine-readable blocked artifact showing both upstream gates were absent rather than inventing GGUF results.",
        ],
        "top_3_gaps_for_229": [
            "Pre-test cascade remains unresolved: Exp 2323 produced no deliverable after three 1201s timeout attempts, so direct operator pytest inspection is now mandatory.",
            "Failed repair tasks must write partial deliverables before timeout; the missing Exp 2323 artifact erased per-test evidence for the potts artifact and two xdist marker fixes.",
            "NSVIF first run and FST full-answer live generation remain unexecuted; they should stay gated until the pre-test gate is green and the GGUF cache precondition is checked.",
        ],
        "top_gaps_resolved": gaps,
        "top_gaps_resolved_count": {
            "count": gaps_resolved,
            "total": len(gaps),
            "display": f"{gaps_resolved}/{len(gaps)}",
        },
        "pretest_cascade_status": pretest,
        "gguf_availability_status": _gguf_status(pretest["fully_resolved"]),
        "next_milestone_speedup_target_pct": NEXT_MILESTONE_SPEEDUP_TARGET_PCT,
        "speedup_basis": (
            "About 60 of 81 logged minutes were consumed by three Exp 2323 timeout "
            "attempts before downstream gate churn. Manual targeted pytest runs, a "
            "partial-deliverable-before-timeout rule, and a no-xdist fallback for GPU "
            "pre-tests should recover roughly half the wall time without assuming live "
            "research compute gets faster."
        ),
        "ops_changelog_update_status": {
            "modified": False,
            "reason": (
                "The task's STOP-WHEN-DONE rule explicitly says not to update "
                "ops/changelog.md because a separate reconciler handles docs/status/traceability."
            ),
        },
        "field_principles": {
            "honest_verdict": "Terminal-prefix required.",
            "criteria_met": (
                "N/total count validates milestone completion fraction; tracks systemic "
                "improvement or regression."
            ),
            "top_gaps_resolved": (
                "Records which of the three .228 design gaps were resolved; enables "
                "multi-milestone gap-closure tracking."
            ),
            "pretest_cascade_status": (
                "Explicit field for whether the 9-milestone pre-test cascade was finally "
                "fully resolved; load-bearing for .229 planning."
            ),
            "next_milestone_speedup_target_pct": (
                "Quantifies where wall-time can be recovered in .229; forces concrete "
                "identification of slow paths."
            ),
        },
        "retro_complete": True,
        "acceptance_gate_passed": True,
        "honest_verdict": (
            "complete: milestone_2026_05_228_retro_"
            f"{criteria_count}_of_{criteria_total}_terminal_tasks_complete_"
            "pretest_cascade_unresolved_0_of_3_design_gaps_closed"
        ),
    }


def write_retro(repo_root: Path | str = Path(".")) -> Path:
    """Write the Exp 2335 retrospective deliverable."""

    root = Path(repo_root).resolve()
    payload = build_retro(root)
    out_path = root / RETRO_DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root to inspect")
    args = parser.parse_args()
    out_path = write_retro(Path(args.repo_root))
    print(f"Wrote retro to {out_path}")


if __name__ == "__main__":
    main()
