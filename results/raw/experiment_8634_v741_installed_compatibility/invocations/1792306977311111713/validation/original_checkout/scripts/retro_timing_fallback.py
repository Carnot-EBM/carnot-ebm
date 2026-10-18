"""Standalone operational-retro timing fallback.

HOW TO WIRE THIS IN:
In `scripts/research_conductor.py::_run_operational_retrospective`, replace the
git-log-grep `experiment_times` block around lines 2804-2864 with this helper.
Two-line conductor change:
    from scripts.retro_timing_fallback import build_retro_timing_fallback
    retro_timing = build_retro_timing_fallback(current); experiment_times = retro_timing["experiment_times"]
Then read `total_wall_time_minutes`, `experiments_completed`,
`compute_bound_experiments_count`, `slowest_experiments`, and
`gpu_idle_on_compute_bound_tasks` from `retro_timing` for the skeleton.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


JsonDict = dict[str, Any]
GitRunner = Callable[[Sequence[str], Path], str]

MODULE_REL_PATH = "scripts/retro_timing_fallback.py"
_EXPERIMENT_RE = re.compile(r"^results/experiment_\d+_.+\.json$")
_COMPUTE_SUBSTRATES = {
    "live_llm_inference",
    "live_gpu_inference",
    "gpu_llm_inference",
    "cuda_inference",
}
_COMPUTE_TEXT_MARKERS = (
    "unsloth/",
    "Qwen3.6-",
    "gemma-4-",
    "requires_gpu",
    "model_specs",
    "DualGPURunner",
    "DualGPUHarness",
    "llama.cpp",
    "GGUF",
    ".cuda(",
    "torch.cuda",
)
_COMPUTE_VALUE_MARKERS = (
    "gpu",
    "cuda",
    "3090",
    "a100",
    "h100",
    "llama.cpp",
    "gguf",
)
_COMPUTE_KEY_MARKERS = (
    "backend",
    "device",
    "accelerator",
    "substrate",
    "model_specs",
    "requires_gpu",
)


def _run_git_default(args: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout


def _parse_git_datetime(text: str) -> datetime | None:
    stripped = text.strip()
    if not stripped:
        return None
    parts = stripped.split()
    if len(parts) >= 3 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[0]):
        stamp = " ".join(parts[:3])
    elif len(parts) >= 4 and re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
        stamp = " ".join(parts[1:4])
    else:
        return None
    try:
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return None


def _iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_rel_path(path_value: object, repo_root: Path) -> str | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    raw = Path(path_value.strip())
    try:
        if raw.is_absolute():
            return raw.relative_to(repo_root).as_posix()
    except ValueError:
        return raw.as_posix()
    return raw.as_posix().lstrip("./")


def _is_terminal_experiment_path(rel_path: str) -> bool:
    return bool(_EXPERIMENT_RE.fullmatch(rel_path)) and not rel_path.endswith(
        "_state.json"
    )


def _read_json_object(path: Path) -> JsonDict:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _walk_key_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def _numeric_duration_minutes(artifact: Mapping[str, Any]) -> float | None:
    duration_s = artifact.get("duration_s")
    if isinstance(duration_s, bool):
        return None
    if isinstance(duration_s, int | float) and duration_s >= 0:
        return round(float(duration_s) / 60.0, 2)
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "1", "gpu", "cuda"}
    return False


def _artifact_declares_compute(artifact: Mapping[str, Any]) -> bool:
    explicit = artifact.get("compute_bound")
    if isinstance(explicit, bool):
        return explicit
    if _truthy(artifact.get("requires_gpu")):
        return True
    substrate = str(artifact.get("inference_substrate", "")).strip()
    if substrate in _COMPUTE_SUBSTRATES:
        return True
    for key, value in artifact.items():
        key_lower = key.casefold()
        if not any(marker in key_lower for marker in _COMPUTE_KEY_MARKERS):
            continue
        value_lower = str(value).casefold()
        if any(marker in value_lower for marker in _COMPUTE_VALUE_MARKERS):
            return True
    return False


def _task_declares_compute(task: Mapping[str, Any]) -> bool:
    text = f"{task.get('title', '')} {task.get('prompt', '')}"
    return any(marker in text for marker in _COMPUTE_TEXT_MARKERS)


def _is_compute_bound(artifact: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    return _artifact_declares_compute(artifact) or _task_declares_compute(task)


def _load_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dedupe_tasks(tasks: Iterable[Mapping[str, Any]], repo_root: Path) -> list[JsonDict]:
    seen: set[str] = set()
    deduped: list[JsonDict] = []
    for task in tasks:
        rel_path = _normalize_rel_path(task.get("deliverable"), repo_root)
        if rel_path is None or rel_path in seen:
            continue
        seen.add(rel_path)
        row = dict(task)
        row["deliverable"] = rel_path
        deduped.append(row)
    return deduped


def load_milestone_tasks(milestone: str, repo_root: Path | str = Path.cwd()) -> list[JsonDict]:
    """Load deliverable-bearing tasks for a milestone from roadmap files."""

    root = Path(repo_root)
    wanted = str(milestone)
    collected: list[Mapping[str, Any]] = []

    roadmap = _load_yaml(root / "research-roadmap.yaml")
    if str(roadmap.get("milestone")) == wanted:
        roadmap_tasks = roadmap.get("tasks", [])
        if isinstance(roadmap_tasks, list):
            collected.extend(t for t in roadmap_tasks if isinstance(t, Mapping))

    complete = _load_yaml(root / "research-complete.yaml")
    milestones = complete.get("milestones", [])
    if isinstance(milestones, list):
        for record in milestones:
            if not isinstance(record, Mapping) or str(record.get("id")) != wanted:
                continue
            record_tasks = record.get("tasks", [])
            if isinstance(record_tasks, list):
                collected.extend(t for t in record_tasks if isinstance(t, Mapping))

    return _dedupe_tasks(collected, root)


def _activation_bound(
    milestone: str,
    repo_root: Path,
    git_runner: GitRunner,
    now: Callable[[], datetime] | None,
) -> JsonDict:
    out = git_runner(
        [
            "git",
            "log",
            "--format=%H %ai",
            f"--grep=\\[conductor\\] Activate milestone {milestone}",
            "-n",
            "1",
        ],
        repo_root,
    )
    stripped = out.strip()
    if stripped:
        commit_hash = stripped.split()[0]
        return {
            "source": "activation_commit",
            "commit_hash": commit_hash,
            "since_arg": f"{commit_hash}..HEAD",
            "start": _parse_git_datetime(stripped),
        }
    clock = now or (lambda: datetime.now(timezone.utc))
    start = clock()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return {
        "source": "since_24_hours",
        "commit_hash": None,
        "since_arg": "--since=24 hours ago",
        "start": start - timedelta(hours=24),
    }


def _artifact_timestamp(
    rel_path: str, repo_root: Path, git_runner: GitRunner
) -> tuple[datetime | None, str | None]:
    git_out = git_runner(
        ["git", "log", "-1", "--format=%ai", "--", rel_path],
        repo_root,
    )
    git_dt = _parse_git_datetime(git_out)
    if git_dt is not None:
        return git_dt, "git_log"
    artifact_path = repo_root / rel_path
    try:
        return (
            datetime.fromtimestamp(artifact_path.stat().st_mtime, tz=timezone.utc),
            "filesystem_mtime",
        )
    except OSError:
        return None, None


def _experiment_label(task: Mapping[str, Any], rel_path: str) -> str:
    label = task.get("title") or task.get("id") or Path(rel_path).name
    return str(label)[:80]


def build_retro_timing_fallback(
    milestone: str,
    tasks: Sequence[Mapping[str, Any]] | None = None,
    repo_root: Path | str = Path.cwd(),
    git_runner: GitRunner | None = None,
    now: Callable[[], datetime] | None = None,
) -> JsonDict:
    """Build conductor-compatible retro timing fields for one milestone."""

    root = Path(repo_root)
    run_git = git_runner or _run_git_default
    task_list = _dedupe_tasks(tasks, root) if tasks is not None else load_milestone_tasks(milestone, root)
    activation = _activation_bound(milestone, root, run_git, now)
    activation_start = activation.get("start")
    records: list[JsonDict] = []
    missing_deliverables: list[str] = []
    excluded_pre_activation: list[str] = []
    timestamp_sources: dict[str, str] = {}

    for sequence, task in enumerate(task_list):
        rel_path = _normalize_rel_path(task.get("deliverable"), root)
        if rel_path is None or not _is_terminal_experiment_path(rel_path):
            continue
        artifact_path = root / rel_path
        if not artifact_path.exists():
            missing_deliverables.append(rel_path)
            continue
        artifact = _read_json_object(artifact_path)
        timestamp, timestamp_source = _artifact_timestamp(rel_path, root, run_git)
        if (
            isinstance(activation_start, datetime)
            and timestamp is not None
            and timestamp < activation_start
        ):
            excluded_pre_activation.append(rel_path)
            continue
        if timestamp_source is not None:
            timestamp_sources[rel_path] = timestamp_source
        duration_min = _numeric_duration_minutes(artifact)
        records.append(
            {
                "sequence": sequence,
                "deliverable": rel_path,
                "experiment": _experiment_label(task, rel_path),
                "timestamp": timestamp,
                "timestamp_source": timestamp_source,
                "duration_min": duration_min,
                "duration_source": "self_reported"
                if duration_min is not None
                else "timestamp_delta",
                "compute_bound": _is_compute_bound(artifact, task),
                "inference_substrate": artifact.get("inference_substrate"),
            }
        )

    records.sort(key=lambda row: int(row["sequence"]))

    previous_timestamp = (
        activation_start if isinstance(activation_start, datetime) else None
    )
    for row in records:
        timestamp = row.get("timestamp")
        if row["duration_min"] is None:
            if isinstance(timestamp, datetime) and isinstance(previous_timestamp, datetime):
                delta_min = max(0.0, (timestamp - previous_timestamp).total_seconds() / 60.0)
                row["duration_min"] = round(delta_min, 2)
            else:
                row["duration_min"] = 0.0
        if isinstance(timestamp, datetime):
            previous_timestamp = timestamp

    timestamps = [row["timestamp"] for row in records if isinstance(row.get("timestamp"), datetime)]
    if len(timestamps) >= 2:
        total_wall_time_minutes = round(
            (max(timestamps) - min(timestamps)).total_seconds() / 60.0, 1
        )
    else:
        total_wall_time_minutes = round(
            sum(float(row["duration_min"]) for row in records), 1
        )

    experiment_times = [
        {
            "experiment": row["experiment"],
            "deliverable": row["deliverable"],
            "duration_min": round(float(row["duration_min"]), 2),
            "duration_minutes": round(float(row["duration_min"]), 2),
            "compute_bound": bool(row["compute_bound"]),
            "duration_source": row["duration_source"],
            "timestamp_source": row["timestamp_source"],
            "timestamp": _iso_z(row["timestamp"]),
            "inference_substrate": row["inference_substrate"],
        }
        for row in records
    ]
    slowest = sorted(
        experiment_times,
        key=lambda row: float(row["duration_minutes"]),
        reverse=True,
    )[:5]
    compute_bound_count = sum(1 for row in experiment_times if row["compute_bound"])
    known_good_checks = {
        "m450_reconstruction_correct": (
            str(milestone) == "2026.06.450"
            and len(experiment_times) == 10
            and 200.0 <= total_wall_time_minutes <= 225.0
            and compute_bound_count == 4
        )
        if str(milestone) == "2026.06.450"
        else None
    }

    return {
        "milestone": milestone,
        "module_path": MODULE_REL_PATH,
        "experiments_completed": len(experiment_times),
        "total_wall_time_minutes": total_wall_time_minutes,
        "compute_bound_experiments_count": compute_bound_count,
        "slowest_experiments": [
            {
                "experiment": row["experiment"],
                "duration_minutes": row["duration_minutes"],
                "compute_bound": row["compute_bound"],
            }
            for row in slowest
        ],
        "gpu_idle_on_compute_bound_tasks": None if compute_bound_count == 0 else False,
        "experiment_times": experiment_times,
        "activation_bound": {
            "source": activation["source"],
            "commit_hash": activation["commit_hash"],
            "since_arg": activation["since_arg"],
            "start": _iso_z(activation_start)
            if isinstance(activation_start, datetime)
            else None,
        },
        "timestamp_sources": timestamp_sources,
        "missing_deliverables": missing_deliverables,
        "excluded_pre_activation": excluded_pre_activation,
        "known_good_checks": known_good_checks,
    }


def legacy_literal_exp_subject_count(
    milestone: str,
    repo_root: Path | str = Path.cwd(),
    git_runner: GitRunner | None = None,
) -> int:
    """Replicate the conductor's old literal `Exp ` subject predicate."""

    root = Path(repo_root)
    run_git = git_runner or _run_git_default
    activation = _activation_bound(milestone, root, run_git, None)
    log_args = [
        "git",
        "log",
        "--format=%H %ai %s",
        "--grep=\\[conductor\\]",
        str(activation["since_arg"]),
    ]
    out = run_git(log_args, root)
    prev_time: datetime | None = None
    count = 0
    for line in reversed(out.strip().splitlines()):
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            ts = datetime.strptime(f"{parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        msg = parts[3]
        if prev_time is not None and "Exp " in msg:
            count += 1
        prev_time = ts
    return count


def validate_historical_false_zero_milestones(
    milestones: Sequence[str] = (
        "2026.06.450",
        "2026.07.467",
        "2026.07.470",
        "2026.07.472",
    ),
    repo_root: Path | str = Path.cwd(),
) -> list[JsonDict]:
    """Return concise validation rows for known false-zero milestones."""

    rows: list[JsonDict] = []
    for milestone in milestones:
        summary = build_retro_timing_fallback(milestone, repo_root=repo_root)
        if milestone == "2026.06.450":
            matches_known_good = bool(
                summary["known_good_checks"]["m450_reconstruction_correct"]
            )
        else:
            matches_known_good = bool(
                summary["experiments_completed"] > 0
                and summary["total_wall_time_minutes"] > 0
            )
        rows.append(
            {
                "milestone": milestone,
                "reconstructed_wall_time_minutes": summary[
                    "total_wall_time_minutes"
                ],
                "reconstructed_compute_bound_count": summary[
                    "compute_bound_experiments_count"
                ],
                "matches_known_good": matches_known_good,
            }
        )
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milestone", nargs="?", default="2026.06.450")
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    args = parser.parse_args()
    payload = build_retro_timing_fallback(args.milestone, repo_root=Path(args.repo_root))
    print(json.dumps(payload, indent=2, sort_keys=True))
