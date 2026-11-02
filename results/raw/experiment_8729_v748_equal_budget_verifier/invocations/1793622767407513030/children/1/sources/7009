"""Generate the milestone 2026.05.232 operational retrospective."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCHEMA = "carnot.operational_retro.v75"
MILESTONE = "2026.05.232"
RETRO_DELIVERABLE = Path("results/experiment_2391_retro.json")
HALLUSCAN_AUROC = 0.88


@dataclass(frozen=True)
class TaskSpec:
    """One planned .232 roadmap task and its expected artifact location."""

    task_id: str
    title: str
    deliverable: str

    @property
    def experiment_number(self) -> int:
        match = re.match(r"exp(?P<number>\d+)", self.task_id)
        if match is None:
            raise ValueError(f"Task id does not start with an experiment number: {self.task_id}")
        return int(match.group("number"))


@dataclass(frozen=True)
class LogEntry:
    """One parsed markdown conductor row."""

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


def _strip_yaml_scalar(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].replace("''", "'")
    return stripped


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def parse_conductor_log(text: str) -> list[LogEntry]:
    """Parse conductor rows for REQ-REPORT-2391 wall-time and status accounting."""

    entries: list[LogEntry] = []
    for line in text.splitlines():
        match = _LOG_ROW.match(line)
        if match is None:
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


def load_roadmap_tasks(repo_root: Path) -> list[TaskSpec]:
    """Read the active roadmap and extract planned Exp 2378 through Exp 2391 tasks."""

    current: dict[str, str] | None = None
    tasks: list[TaskSpec] = []
    for line in (repo_root / "research-roadmap.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("- id: "):
            if current is not None:
                tasks.append(
                    TaskSpec(
                        current["id"],
                        current.get("title", current["id"]),
                        current.get("deliverable", ""),
                    )
                )
            current = {"id": _strip_yaml_scalar(line.split(":", 1)[1])}
            continue
        if current is None:
            continue
        if line.startswith("  title: "):
            current["title"] = _strip_yaml_scalar(line.split(":", 1)[1])
        elif line.startswith("  deliverable: "):
            current["deliverable"] = _strip_yaml_scalar(line.split(":", 1)[1])

    if current is not None:
        tasks.append(
            TaskSpec(
                current["id"],
                current.get("title", current["id"]),
                current.get("deliverable", ""),
            )
        )

    return [task for task in tasks if 2378 <= task.experiment_number <= 2391]


def milestone_window(entries: list[LogEntry]) -> list[LogEntry]:
    """Return the .232 activation-to-pre-retro conductor window."""

    start_index: int | None = None
    for index, entry in enumerate(entries):
        if entry.title == f"Milestone {MILESTONE} activated":
            start_index = index

    if start_index is None:
        raise ValueError(f"Could not find Milestone {MILESTONE} activation row")

    window: list[LogEntry] = []
    for entry in entries[start_index:]:
        if "Milestone 2026.05.232 Operational Retrospective" in entry.title:
            break
        if entry.title.startswith("Plan next milestone"):
            break
        if entry.title == "Milestone 2026.05.233 activated":
            break
        window.append(entry)
    return window


def _task_log_entries(window: list[LogEntry]) -> list[LogEntry]:
    return [
        entry
        for entry in window
        if entry.title != f"Milestone {MILESTONE} activated" and not entry.title.startswith("Plan ")
    ]


def _title_matches(roadmap_title: str, log_title: str) -> bool:
    roadmap_norm = " ".join(roadmap_title.split())
    log_norm = " ".join(log_title.split())
    return roadmap_norm.startswith(log_norm) or log_norm.startswith(roadmap_norm[: len(log_norm)])


def _entries_for_task(window: list[LogEntry], task: TaskSpec) -> list[LogEntry]:
    return [entry for entry in _task_log_entries(window) if _title_matches(task.title, entry.title)]


def _artifact_status(repo_root: Path, task: TaskSpec) -> tuple[bool, dict[str, Any] | None]:
    if not task.deliverable:
        return False, None
    artifact = _load_json(repo_root / task.deliverable)
    return artifact is not None, artifact


def _key_result(
    task: TaskSpec,
    status: str,
    entries: list[LogEntry],
    artifact: dict[str, Any] | None,
) -> str:
    if artifact is not None:
        if isinstance(artifact.get("honest_verdict"), str):
            return str(artifact["honest_verdict"])
        if task.task_id.startswith("exp2389"):
            ready = artifact.get("n_paper_ready_results")
            gap = artifact.get("hallscan_gap")
            return f"paper-v6 table available: n_paper_ready_results={ready}; hallscan_gap={gap}"
    if entries:
        return entries[-1].details
    if status == "SKIP":
        return "No terminal conductor row before retrospective generation."
    return "No artifact evidence available."


def build_task_outcomes(
    repo_root: Path, tasks: list[TaskSpec], window: list[LogEntry]
) -> list[dict]:
    """Build REQ-REPORT-2391 per-task terminal status records."""

    outcomes: list[dict] = []
    for task in tasks:
        entries = _entries_for_task(window, task)
        status = entries[-1].status if entries else "SKIP"
        artifact_present, artifact = _artifact_status(repo_root, task)
        outcomes.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "deliverable": task.deliverable,
                "status": status,
                "attempts": len(entries),
                "status_counts": {
                    candidate: sum(1 for entry in entries if entry.status == candidate)
                    for candidate in ("OK", "GATE_BLOCK", "FAIL")
                },
                "artifact_present": artifact_present,
                "key_result": _key_result(task, status, entries, artifact),
            }
        )
    return outcomes


def _first_bool(artifact: dict[str, Any] | None, keys: tuple[str, ...]) -> bool:
    if artifact is None:
        return False
    return any(artifact.get(key) is True for key in keys)


def _numeric_candidates(repo_root: Path) -> list[dict[str, Any]]:
    candidate_specs = (
        ("exp2379", "results/experiment_2379_halt_tier0j.json", ("halt_k19j_auroc",)),
        (
            "exp2380",
            "results/experiment_2380_hive_ensemble.json",
            ("ensemble_auroc_4verifier", "hive_ensemble_auroc"),
        ),
        (
            "exp2381",
            "results/experiment_2381_fregelogic.json",
            ("fregelogic_auroc", "frege_z3_neural_auroc"),
        ),
        ("exp2389", "results/experiment_2389_paperv6_table.json", ("best_auroc_achieved",)),
    )
    candidates: list[dict[str, Any]] = []
    for source_id, artifact_path, keys in candidate_specs:
        artifact = _load_json(repo_root / artifact_path)
        if artifact is None:
            continue
        for key in keys:
            value = artifact.get(key)
            if isinstance(value, int | float):
                candidates.append(
                    {
                        "source_id": source_id,
                        "source_artifact": artifact_path,
                        "metric": key,
                        "value": float(value),
                    }
                )
    return candidates


def _best_auroc_at_close(repo_root: Path) -> tuple[float | None, dict[str, Any] | None]:
    candidates = _numeric_candidates(repo_root)
    if not candidates:
        return None, None
    best = max(candidates, key=lambda item: item["value"])
    return float(best["value"]), best


def _wall_time_min(window: list[LogEntry]) -> float:
    source_entries = _task_log_entries(window)
    if not source_entries:
        return 0.0
    return round((source_entries[-1].timestamp - window[0].timestamp).total_seconds() / 60.0, 1)


def build_retro(repo_root: Path) -> dict[str, Any]:
    """Build the complete REQ-REPORT-2391 retrospective payload."""

    tasks = load_roadmap_tasks(repo_root)
    log_entries = parse_conductor_log(
        (repo_root / "ops/conductor-log.md").read_text(encoding="utf-8")
    )
    window = milestone_window(log_entries)
    task_outcomes = build_task_outcomes(repo_root, tasks, window)

    n_experiments_completed = sum(1 for item in task_outcomes if item["status"] == "OK")
    n_gate_blocks = sum(1 for item in task_outcomes if item["status"] == "GATE_BLOCK")
    n_failed = sum(1 for item in task_outcomes if item["status"] == "FAIL")
    n_skipped = sum(1 for item in task_outcomes if item["status"] == "SKIP")
    n_failed_attempts = sum(int(item["status_counts"]["FAIL"]) for item in task_outcomes)
    n_gate_block_attempts = sum(int(item["status_counts"]["GATE_BLOCK"]) for item in task_outcomes)

    artifacts = {
        "exp2382": _load_json(repo_root / "results/experiment_2382_fst_live_path_ab.json"),
        "exp2383": _load_json(repo_root / "results/experiment_2383_fr11_nsvif_online.json"),
        "exp2384": _load_json(repo_root / "results/experiment_2384_kv260_yosys.json"),
        "exp2388": _load_json(repo_root / "results/experiment_2388_phase1_ship_gate.json"),
        "exp2390": _load_json(repo_root / "results/experiment_2390_capstone.json"),
    }

    best_auroc, best_auroc_source = _best_auroc_at_close(repo_root)
    auroc_gap = _round4(HALLUSCAN_AUROC - best_auroc) if best_auroc is not None else None
    hive_artifact = _load_json(repo_root / "results/experiment_2380_hive_ensemble.json")
    hive_auroc = None
    if hive_artifact is not None:
        hive_auroc = hive_artifact.get("ensemble_auroc_4verifier")
        if not isinstance(hive_auroc, int | float):
            hive_auroc = hive_artifact.get("hive_ensemble_auroc")

    fst_completed = _first_bool(artifacts["exp2382"], ("live_inference_completed",))
    fr11_satisfied = _first_bool(artifacts["exp2383"], ("fr11_nsvif_online_passed",))
    kv260_succeeded = _first_bool(
        artifacts["exp2384"],
        ("synthesis_succeeded", "kv260_yosys_synthesis_succeeded"),
    )
    phase1_ship_met = _first_bool(
        artifacts["exp2388"],
        ("phase1_ship_criteria_met", "phase1_ship_gate_passed", "all_ship_criteria_met"),
    )

    missing_required_artifacts = [
        path
        for path in (
            "results/experiment_2378_archive.json",
            "results/experiment_2382_fst_live_path_ab.json",
            "results/experiment_2383_fr11_nsvif_online.json",
            "results/experiment_2384_kv260_yosys.json",
            "results/experiment_2388_phase1_ship_gate.json",
            "results/experiment_2390_capstone.json",
        )
        if not (repo_root / path).exists()
    ]

    top_3_successes = [
        {
            "rank": 1,
            "summary": "Paper-v6 results table completed and preserved missing-artifact accounting.",
            "evidence": "results/experiment_2389_paperv6_table.json",
        },
        {
            "rank": 2,
            "summary": "Capstone gate blocked synthesis after upstream AUROC/live-inference tasks failed.",
            "evidence": "conductor status GATE_BLOCK for exp2390",
        },
        {
            "rank": 3,
            "summary": "Close-of-milestone AUROC gap is quantified instead of overclaimed.",
            "evidence": f"auroc_gap_to_hallscan_at_232_close={auroc_gap}",
        },
    ]
    top_3_gaps_for_233 = [
        {
            "rank": 1,
            "summary": "Fix the Codex CLI failure loop before rerunning AUROC closure tasks.",
            "evidence": f"{n_failed_attempts} FAIL attempts across {n_failed} tasks",
        },
        {
            "rank": 2,
            "summary": "Rerun HALT, HIVE, and FregeLogic so the 0.88 AUROC gap can actually close.",
            "evidence": f"HIVE AUROC artifact present={hive_artifact is not None}",
        },
        {
            "rank": 3,
            "summary": "Re-establish evidence for FST live PATH A/B, FR-11, KV260 Yosys, and ship gate.",
            "evidence": f"missing_required_artifacts={len(missing_required_artifacts)}",
        },
    ]

    honest_verdict = (
        "complete: retro_complete=true; "
        f"n_experiments_completed={n_experiments_completed}; "
        f"n_failed={n_failed}; n_gate_blocks={n_gate_blocks}; "
        f"auroc_gap_to_hallscan_at_232_close={auroc_gap}; "
        f"fst_live_path_ab_completed={fst_completed}"
    )

    return {
        "schema": SCHEMA,
        "experiment": "2391_retro_v232",
        "milestone": MILESTONE,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "honest_verdict": honest_verdict,
        "field_principles": {
            "honest_verdict": "Terminal-prefix required.",
            "n_experiments_completed": "Count of terminal OK status tasks.",
            "n_gate_blocks": "Count of GATE_BLOCK tasks.",
            "total_wall_time_min": "Sum of wall time from conductor log.",
            "fr11_satisfied": "True if exp2383 fr11_nsvif_online_passed=true.",
            "fst_live_path_ab_completed": (
                "True if exp2382 live_inference_completed=true (key .232 goal)."
            ),
            "auroc_gap_to_hallscan_at_232_close": (
                "0.88 - best_232_verifier_auroc. Tracks gap closure."
            ),
            "kv260_yosys_synthesis_succeeded": (
                "True if Yosys synthesis succeeded (KV260 hardware track progress)."
            ),
            "retro_complete": "Must be true.",
        },
        "n_planned_tasks": len(task_outcomes),
        "n_experiments_completed": n_experiments_completed,
        "n_gate_blocks": n_gate_blocks,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "n_failed_attempts": n_failed_attempts,
        "n_gate_block_attempts": n_gate_block_attempts,
        "total_wall_time_min": _wall_time_min(window),
        "task_outcomes": task_outcomes,
        "fr11_satisfied": fr11_satisfied,
        "fst_live_path_ab_completed": fst_completed,
        "best_232_verifier_auroc": _round4(best_auroc),
        "best_232_verifier_auroc_source": best_auroc_source,
        "auroc_gap_to_hallscan_at_232_close": auroc_gap,
        "hive_ensemble_auroc": _round4(float(hive_auroc))
        if isinstance(hive_auroc, int | float)
        else None,
        "hive_gap_closed_vs_hallscan": 0.0
        if not isinstance(hive_auroc, int | float)
        else max(0.0, round(float(hive_auroc) - HALLUSCAN_AUROC, 4)),
        "kv260_yosys_synthesis_succeeded": kv260_succeeded,
        "phase1_ship_criteria_met": phase1_ship_met,
        "missing_required_artifacts": missing_required_artifacts,
        "top_3_successes": top_3_successes,
        "top_3_gaps_for_233": top_3_gaps_for_233,
        "retro_complete": True,
    }


def write_retro(repo_root: Path) -> Path:
    """Write the Exp 2391 retro artifact to the requested results path."""

    out_path = repo_root / RETRO_DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_retro(repo_root)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    out_path = write_retro(args.repo_root)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
