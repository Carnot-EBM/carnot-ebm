"""Write-time runtime stamping and result-artifact audit helpers.

Spec refs: REQ-REPORT-4920, SCENARIO-REPORT-4920-STAMPING-AUDIT.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re
from typing import Any


JsonDict = dict[str, Any]
MIN_DURATION_S = 0.0001
_EXPERIMENT_RE = re.compile(r"experiment_(\d+)")


def _experiment_id(path: Path | str) -> int | None:
    match = _EXPERIMENT_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _relative_result_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return str(Path(*parts[index:]))
    return str(path)


def stamp_runtime_metadata(
    artifact: Mapping[str, Any],
    *,
    started_s: float,
    finished_s: float,
    inference_substrate: str,
    compute_bound: bool,
) -> JsonDict:
    """Return an artifact copy stamped with duration, substrate, and compute flag."""

    stamped = dict(artifact)
    duration_s = round(max(MIN_DURATION_S, float(finished_s) - float(started_s)), 6)
    stamped["duration_s"] = duration_s
    stamped["inference_substrate"] = str(inference_substrate)
    stamped["compute_bound"] = bool(compute_bound)
    return stamped


def _read_json_object(path: Path) -> tuple[JsonDict, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, "json_error"
    if not isinstance(value, dict):
        return {}, "json_error"
    return value, ""


def _duration_missing(value: Any) -> bool:
    return value is None or isinstance(value, bool) or not isinstance(value, (int, float))


def _substrate_missing(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _compute_bound_missing(value: Any) -> bool:
    return not isinstance(value, bool)


def audit_runtime_stamps(paths: Iterable[Path | str]) -> JsonDict:
    """Scan artifacts and list missing duration/substrate/compute-bound stamps."""

    missing_by_field: dict[str, list[JsonDict]] = {
        "duration_s": [],
        "inference_substrate": [],
        "compute_bound": [],
    }
    missing_any: list[JsonDict] = []
    ordered_paths = sorted(
        (Path(path) for path in paths),
        key=lambda path: (_experiment_id(path) or 10**9, _relative_result_path(path)),
    )
    for path in ordered_paths:
        base_record: JsonDict = {
            "path": _relative_result_path(path),
            "experiment_id": _experiment_id(path),
        }
        artifact, error = _read_json_object(path)
        missing_fields: list[str] = []
        if error:
            missing_fields.append(error)
        else:
            checks = {
                "duration_s": _duration_missing(artifact.get("duration_s")),
                "inference_substrate": _substrate_missing(artifact.get("inference_substrate")),
                "compute_bound": _compute_bound_missing(artifact.get("compute_bound")),
            }
            for field, missing in checks.items():
                if missing:
                    missing_fields.append(field)
                    missing_by_field[field].append(dict(base_record))
        if missing_fields:
            missing_any.append({**base_record, "missing_fields": missing_fields})
    return {
        "scanned_count": len(ordered_paths),
        "missing_by_field": missing_by_field,
        "missing_any": missing_any,
    }
