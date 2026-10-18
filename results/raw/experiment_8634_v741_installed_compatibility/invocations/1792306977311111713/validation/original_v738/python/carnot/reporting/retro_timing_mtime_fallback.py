"""Results-mtime fallback for milestone retro timing.

Spec refs: REQ-REPORT-4920, SCENARIO-REPORT-4920.

The pure core accepts already-collected artifact records so tests and conductor
wiring can verify the window without touching git history or wall-clock state.
Small scanning helpers turn result files into those records for operator wiring.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]
_EXPERIMENT_RE = re.compile(r"experiment_(\d+)")
_COMPUTE_BOUND_MARKERS = ("cuda", "gpu0", "gpu1", "3090", "a100", "h100")
_BACKEND_KEY_MARKERS = ("backend", "device", "accelerator")


@dataclass(frozen=True)
class ArtifactMtimeRecord:
    """One result artifact's path, mtime, and compute-bound classification."""

    path: str
    mtime_ns: int
    compute_bound: bool = False


def _short_milestone(milestone: str) -> str:
    parts = str(milestone).split(".")
    return parts[-1] if parts and parts[-1].isdigit() else ""


def _experiment_id(path: Path | str) -> int | None:
    match = _EXPERIMENT_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _result_relative_path(path: Path, results_dir: Path | None = None) -> str:
    if results_dir is not None:
        return str(Path(Path(results_dir).name) / path.name)
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return str(Path(*parts[index:]))
    return str(path)


def find_milestone_arm_paths(results_dir: Path | str, milestone: str) -> list[Path]:
    """Return milestone arm artifacts using archive/capstone filename bounds."""

    results_path = Path(results_dir)
    short = _short_milestone(milestone)
    if not short or not results_path.exists():
        return []
    files = sorted(
        (
            path
            for path in results_path.glob("experiment_*.json")
            if _experiment_id(path) is not None
        ),
        key=lambda path: (_experiment_id(path) or -1, path.name),
    )
    start_ids = [
        exp_id
        for path in files
        if (exp_id := _experiment_id(path)) is not None and f"activate_{short}" in path.name
    ]
    end_ids = [
        exp_id
        for path in files
        if (exp_id := _experiment_id(path)) is not None and f"capstone_v{short}" in path.name
    ]
    if start_ids and end_ids and min(end_ids) >= min(start_ids):
        start = min(start_ids)
        end = min(end_ids)
        return [
            path
            for path in files
            if (exp_id := _experiment_id(path)) is not None and start <= exp_id <= end
        ]
    return [
        path
        for path in files
        if f"_v{short}" in path.name or f"_{short}_" in path.name
    ]


def _read_json_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _walk_key_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child)


def compute_bound_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Infer legacy compute-bound status without treating all LLM use as GPU."""

    explicit = artifact.get("compute_bound")
    if isinstance(explicit, bool):
        return explicit
    for key, value in _walk_key_values(artifact):
        key_lower = key.casefold()
        if not any(marker in key_lower for marker in _BACKEND_KEY_MARKERS):
            continue
        value_lower = str(value).casefold()
        if any(marker in value_lower for marker in _COMPUTE_BOUND_MARKERS):
            return True
    return False


def scan_milestone_records(results_dir: Path | str, milestone: str) -> list[ArtifactMtimeRecord]:
    """Scan result files for a milestone and return deterministic mtime records."""

    results_path = Path(results_dir)
    records: list[ArtifactMtimeRecord] = []
    for path in find_milestone_arm_paths(results_path, milestone):
        artifact = _read_json_object(path)
        records.append(
            ArtifactMtimeRecord(
                path=_result_relative_path(path, results_path),
                mtime_ns=path.stat().st_mtime_ns,
                compute_bound=compute_bound_from_artifact(artifact),
            )
        )
    return sorted(records, key=lambda record: (_experiment_id(record.path) or -1, record.path))


def _iso_z(mtime_ns: int) -> str:
    dt = datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def reconstruct_mtime_window(
    milestone: str,
    records: Iterable[ArtifactMtimeRecord],
) -> JsonDict:
    """Purely reconstruct a milestone window from supplied artifact mtimes."""

    ordered = sorted(records, key=lambda record: (record.mtime_ns, record.path))
    if not ordered:
        return {
            "milestone": milestone,
            "n_arms": 0,
            "window_start": None,
            "window_end": None,
            "wall_minutes": 0.0,
            "compute_bound_count": 0,
            "artifact_paths": [],
        }
    start_ns = ordered[0].mtime_ns
    end_ns = ordered[-1].mtime_ns
    return {
        "milestone": milestone,
        "n_arms": len(ordered),
        "window_start": _iso_z(start_ns),
        "window_end": _iso_z(end_ns),
        "wall_minutes": round((end_ns - start_ns) / 60_000_000_000, 2),
        "compute_bound_count": sum(1 for record in ordered if record.compute_bound),
        "artifact_paths": [record.path for record in ordered],
    }


def mtime_fallback_window(results_dir: Path | str, milestone: str) -> JsonDict:
    """Scan results and reconstruct the milestone mtime fallback window."""

    return reconstruct_mtime_window(milestone, scan_milestone_records(results_dir, milestone))
