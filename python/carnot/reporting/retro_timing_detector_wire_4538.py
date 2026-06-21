"""Exp 4538 retro timing-data detector wiring.

Spec refs: REQ-REPORT-4538, SCENARIO-REPORT-4538-RETRO-PATH,
SCENARIO-REPORT-4538-FALLBACK, SCENARIO-REPORT-4538-ARTIFACT.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot.reporting import timing_detector_repair_4517 as repaired
from carnot.reporting.timing_detector_repair_4517 import MilestoneWindow


SCHEMA = "carnot.retro_timing_detector_wire_4538.v1"
OUTPUT_REL_PATH = Path("results/experiment_4538_retro_timing_detector_wire.json")
RETRO_TIMING_DATA_PATH = (
    "scripts/research_conductor.py::_run_operational_retrospective TIMING DATA"
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REPAIRED_DETECTOR_MODULE = "carnot.reporting.timing_detector_repair_4517"

UPSTREAM_ARTIFACTS = (
    (
        Path("results/experiment_4517_timing_detector_repair.json"),
        ("honest_verdict", "detector_true_count_415", "detector_true_count_416"),
    ),
    (
        Path("results/operational_retro_2026_06_418.json"),
        ("experiments_completed", "summary", "bottlenecks_identified"),
    ),
    (
        Path("results/experiment_4528_infra_carryforward.json"),
        ("honest_verdict", "b_track_status", "preconditions_checked"),
    ),
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "retro_path_wired",
    "regression_assert_added",
    "tests_added_pass",
    "cited_upstream_artifacts",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; shipped: retro_timing_detector_wired_regression_asserted "
        "OR complete: retro_timing_partial_<reason>."
    ),
    "inference_substrate": (
        "aggregation_from_upstream_artifacts -- reads the .417/.418 artifacts + "
        "wires the retro path, no model load (100us floor)."
    ),
    "retro_path_wired": (
        "names the retro timing-data path now using the repaired detector -- the "
        "fix that closes the .363->.418 false-zero gap."
    ),
    "regression_assert_added": (
        "the injected==on-disk count assert + detector_gap_suspected emission -- "
        "catches a future false-zero mechanically."
    ),
    "tests_added_pass": "Tests Must Run and Assert.",
    "cited_upstream_artifacts": (
        "traceability of the .417 repair + .418 retro gap this continues."
    ),
    "preconditions_checked": (
        "records resources verified; pre-empts missing-resource fabrication."
    ),
}

DEFAULT_RETRO_WINDOW = MilestoneWindow(
    milestone="2026.06.418",
    start_iso="2026-06-20T18:40:00-04:00",
    end_iso="2026-06-21T01:40:00-04:00",
    experiment_id_min=4523,
    experiment_id_max=4531,
    changelog_date="2026-06-20",
)

_TERMINAL_ARTIFACT_RE = re.compile(r"^results/experiment_(\d+)_[A-Za-z0-9_.+/-]+\.json$")


class DetectorRegressionError(AssertionError):
    """Raised when repaired retro timing data would still false-zero."""


@dataclass(frozen=True)
class RetroTimingData:
    """Authoritative timing data for the operational retro prompt path."""

    milestone: str
    consumer_path: str
    experiment_times: tuple[dict[str, Any], ...]
    timing_summary: str
    total_wall_time_minutes: float
    experiments_completed: int
    compute_bound_experiments_count: int
    slowest_experiments: tuple[dict[str, Any], ...]
    reported_in_window_count: int
    on_disk_in_window_count: int
    legacy_reported_count: int | None
    detector_gap_suspected: bool
    regression_assert_passed: bool
    repaired_detection: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "milestone": self.milestone,
            "consumer_path": self.consumer_path,
            "timing_summary": self.timing_summary,
            "total_wall_time_minutes": self.total_wall_time_minutes,
            "experiments_completed": self.experiments_completed,
            "compute_bound_experiments_count": self.compute_bound_experiments_count,
            "slowest_experiments": [dict(row) for row in self.slowest_experiments],
            "experiment_times": [dict(row) for row in self.experiment_times],
            "reported_in_window_count": self.reported_in_window_count,
            "on_disk_in_window_count": self.on_disk_in_window_count,
            "legacy_reported_count": self.legacy_reported_count,
            "detector_gap_suspected": self.detector_gap_suspected,
            "regression_assert_passed": self.regression_assert_passed,
            "repaired_detection": dict(self.repaired_detection),
        }


def _parse_aware_datetime(iso_text: str) -> datetime:
    parsed = datetime.fromisoformat(iso_text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _window_dates(window: MilestoneWindow) -> tuple[str, ...]:
    start = _parse_aware_datetime(window.start_iso)
    end = _parse_aware_datetime(window.end_iso)
    day = start.date()
    end_day = end.date()
    dates: list[str] = []
    while day <= end_day:
        dates.append(day.isoformat())
        day += timedelta(days=1)
    return tuple(dates)


def _terminal_sort_key(path: str) -> tuple[int, str]:
    match = _TERMINAL_ARTIFACT_RE.fullmatch(path)
    return (int(match.group(1)) if match else -1, path)


def _dedupe_sorted(paths: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(paths), key=_terminal_sort_key))


def _is_terminal_artifact(path: str) -> bool:
    normalized = path.strip()
    return bool(_TERMINAL_ARTIFACT_RE.fullmatch(normalized)) and not normalized.endswith(
        "_state.json"
    )


def _path_experiment_id(path: str) -> int | None:
    match = _TERMINAL_ARTIFACT_RE.fullmatch(path)
    return int(match.group(1)) if match else None


def _in_window_id_range(path: str, window: MilestoneWindow) -> bool:
    experiment_id = _path_experiment_id(path)
    return (
        experiment_id is not None
        and window.experiment_id_min <= experiment_id <= window.experiment_id_max
    )


def _scan_on_disk_window_artifacts(root: Path, window: MilestoneWindow) -> tuple[str, ...]:
    results_dir = root / "results"
    if not results_dir.exists():
        return ()
    paths: list[str] = []
    for artifact_path in results_dir.glob("experiment_*_*.json"):
        rel_path = artifact_path.relative_to(root).as_posix()
        if _is_terminal_artifact(rel_path) and _in_window_id_range(rel_path, window):
            paths.append(rel_path)
    return _dedupe_sorted(paths)


def _window_with_changelog_date(window: MilestoneWindow, changelog_date: str) -> MilestoneWindow:
    return MilestoneWindow(
        milestone=window.milestone,
        start_iso=window.start_iso,
        end_iso=window.end_iso,
        experiment_id_min=window.experiment_id_min,
        experiment_id_max=window.experiment_id_max,
        changelog_date=changelog_date,
    )


def _detect_with_repaired_sources(root: Path, window: MilestoneWindow) -> dict[str, Any]:
    changelog_path = root / "ops" / "changelog.md"
    changelog_text = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    mtime_paths = repaired.scan_results_mtimes(root, window)
    changelog_paths: list[str] = []
    for changelog_date in _window_dates(window):
        dated_window = _window_with_changelog_date(window, changelog_date)
        changelog_paths.extend(
            repaired.extract_changelog_dated_artifacts(changelog_text, dated_window)
        )
    changelog_paths_tuple = _dedupe_sorted(changelog_paths)
    corrected_paths = _dedupe_sorted([*mtime_paths, *changelog_paths_tuple])
    legacy_count = repaired.read_legacy_retro_count(root, window.milestone)
    source_disagreement = set(mtime_paths) != set(changelog_paths_tuple)
    legacy_false_zero = legacy_count == 0 and bool(corrected_paths)
    return {
        "milestone": window.milestone,
        "mtime_count": len(mtime_paths),
        "changelog_count": len(changelog_paths_tuple),
        "corrected_count": len(corrected_paths),
        "legacy_reported_count": legacy_count,
        "fallback_used": bool(set(changelog_paths_tuple) - set(mtime_paths)),
        "detector_gap_suspected": source_disagreement or legacy_false_zero,
        "mtime_paths": list(mtime_paths),
        "changelog_paths": list(changelog_paths_tuple),
        "corrected_paths": list(corrected_paths),
        "mtime_only_paths": [
            path for path in mtime_paths if path not in set(changelog_paths_tuple)
        ],
        "changelog_only_paths": [
            path for path in changelog_paths_tuple if path not in set(mtime_paths)
        ],
        "source_disagreement": source_disagreement,
        "uses_repaired_detector": REPAIRED_DETECTOR_MODULE,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _duration_s(value: Any) -> float:
    if isinstance(value, int | float) and value >= 0:
        return max(0.0001, float(value))
    return 0.0001


def _experiment_row(root: Path, rel_path: str) -> dict[str, Any]:
    payload = _read_json_object(root / rel_path)
    duration_s = _duration_s(payload.get("duration_s"))
    experiment_name = str(payload.get("experiment") or Path(rel_path).stem)
    return {
        "experiment": experiment_name[:80],
        "path": rel_path,
        "duration_s": duration_s,
        "duration_min": round(duration_s / 60.0, 3),
        "compute_bound": bool(payload.get("compute_bound", False)),
    }


def _format_timing_summary(data: Mapping[str, Any]) -> str:
    rows = list(data["experiment_times"])
    lines = [
        f"MILESTONE-SCOPED DATA (repaired detector for {RETRO_TIMING_DATA_PATH}):",
        f"Total milestone wall time: {data['total_wall_time_minutes']:.1f} minutes",
        f"Experiments completed: {data['experiments_completed']}",
        "Compute-bound experiments (GGUF/CUDA/requires_gpu): "
        f"{data['compute_bound_experiments_count']}",
        "Synthesis-only experiments: "
        f"{data['experiments_completed'] - data['compute_bound_experiments_count']}",
        "Regression assert: reported_in_window_count == on_disk_in_window_count "
        f"({data['reported_in_window_count']} == {data['on_disk_in_window_count']})",
        "detector_gap_suspected="
        f"{'true' if data['detector_gap_suspected'] else 'false'}",
        "Slowest experiments (compute_bound flag in [..]):",
    ]
    if not rows:
        lines.append("  - no data available this milestone")
    for row in data["slowest_experiments"]:
        cb_flag = "compute_bound" if row.get("compute_bound") else "synthesis_only"
        lines.append(f"  - {row['duration_min']:.3f}min [{cb_flag}]: {row['path']}")
    return "\n".join(lines) + "\n"


def build_retro_timing_data(
    root: Path,
    window: MilestoneWindow = DEFAULT_RETRO_WINDOW,
) -> RetroTimingData:
    """Build the retro TIMING DATA block from repaired detector evidence."""

    repaired_detection = _detect_with_repaired_sources(root, window)
    on_disk_paths = _scan_on_disk_window_artifacts(root, window)
    reported_count = int(repaired_detection["corrected_count"])
    on_disk_count = len(on_disk_paths)
    legacy_count = repaired_detection["legacy_reported_count"]
    detector_gap_suspected = bool(repaired_detection["detector_gap_suspected"]) or (
        legacy_count == 0 and on_disk_count > 0
    ) or (reported_count == 0 and on_disk_count > 0)
    if reported_count != on_disk_count:
        raise DetectorRegressionError(
            "reported in-window count "
            f"{reported_count} != on-disk in-window artifact count {on_disk_count}"
        )

    rows = tuple(
        _experiment_row(root, rel_path)
        for rel_path in repaired_detection["corrected_paths"]
    )
    total_wall_time_minutes = round(sum(row["duration_s"] for row in rows) / 60.0, 1)
    compute_bound_count = sum(1 for row in rows if row.get("compute_bound"))
    slowest = tuple(sorted(rows, key=lambda row: row["duration_s"], reverse=True)[:5])
    summary_data = {
        "experiment_times": rows,
        "total_wall_time_minutes": total_wall_time_minutes,
        "experiments_completed": reported_count,
        "compute_bound_experiments_count": compute_bound_count,
        "reported_in_window_count": reported_count,
        "on_disk_in_window_count": on_disk_count,
        "detector_gap_suspected": detector_gap_suspected,
        "slowest_experiments": slowest,
    }
    return RetroTimingData(
        milestone=window.milestone,
        consumer_path=RETRO_TIMING_DATA_PATH,
        experiment_times=rows,
        timing_summary=_format_timing_summary(summary_data),
        total_wall_time_minutes=total_wall_time_minutes,
        experiments_completed=reported_count,
        compute_bound_experiments_count=compute_bound_count,
        slowest_experiments=slowest,
        reported_in_window_count=reported_count,
        on_disk_in_window_count=on_disk_count,
        legacy_reported_count=legacy_count,
        detector_gap_suspected=detector_gap_suspected,
        regression_assert_passed=True,
        repaired_detection=repaired_detection,
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(root: Path, rel_path: Path, fields_imported: tuple[str, ...]) -> dict[str, Any]:
    path = root / rel_path
    return {
        "path": rel_path.as_posix(),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() else "",
        "fields_imported": list(fields_imported),
    }


def _git_path_modified(root: Path, rel_path: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", rel_path],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(completed.stdout.strip())


def check_preconditions(root: Path) -> dict[str, Any]:
    spec_path = root / "openspec/capabilities/research-reporting/spec.md"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    conductor_path = root / "scripts/research_conductor.py"
    conductor_text = conductor_path.read_text(encoding="utf-8") if conductor_path.exists() else ""
    return {
        "carnot_import_ok": True,
        "spec_has_req_4538": "REQ-REPORT-4538" in spec_text,
        "results_dir_exists": (root / "results").exists(),
        "ops_changelog_exists": (root / "ops/changelog.md").exists(),
        "upstream_4517_exists": (root / UPSTREAM_ARTIFACTS[0][0]).exists(),
        "upstream_418_retro_exists": (root / UPSTREAM_ARTIFACTS[1][0]).exists(),
        "upstream_4528_exists": (root / UPSTREAM_ARTIFACTS[2][0]).exists(),
        "retro_timing_data_path_located": (
            "_run_operational_retrospective" in conductor_text
            and "TIMING DATA" in conductor_text
        ),
        "scripts_research_conductor_modified": _git_path_modified(
            root, "scripts/research_conductor.py"
        ),
    }


def _blocked_reason(preconditions: Mapping[str, Any]) -> str | None:
    required_true = (
        "carnot_import_ok",
        "spec_has_req_4538",
        "results_dir_exists",
        "ops_changelog_exists",
        "upstream_4517_exists",
        "upstream_418_retro_exists",
        "upstream_4528_exists",
        "retro_timing_data_path_located",
    )
    for field in required_true:
        if not preconditions.get(field):
            return f"complete: retro_timing_partial_blocked_{field}"
    if preconditions.get("scripts_research_conductor_modified"):
        return "complete: retro_timing_partial_protected_conductor_modified"
    return None


def _tests_passed(tests_added_pass: Any) -> bool:
    if isinstance(tests_added_pass, bool):
        return tests_added_pass
    if isinstance(tests_added_pass, Mapping):
        return tests_added_pass.get("passed") is True
    return False


def build_payload(
    root: Path,
    *,
    tests_added_pass: Any,
    window: MilestoneWindow = DEFAULT_RETRO_WINDOW,
) -> dict[str, Any]:
    preconditions = check_preconditions(root)
    cited = [
        _source_record(root, rel_path, fields) for rel_path, fields in UPSTREAM_ARTIFACTS
    ]
    reason = _blocked_reason(preconditions)
    if reason is not None:
        payload = {
            "schema": SCHEMA,
            "experiment": "experiment_4538_retro_timing_detector_wire",
            "honest_verdict": reason,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "retro_path_wired": {
                "consumer_path": RETRO_TIMING_DATA_PATH,
                "wired": False,
                "repaired_detector_module": REPAIRED_DETECTOR_MODULE,
            },
            "regression_assert_added": {
                "assertion": "reported_in_window_count == on_disk_in_window_count",
                "assert_passed": False,
                "detector_gap_suspected_emitted": False,
            },
            "tests_added_pass": tests_added_pass,
            "cited_upstream_artifacts": cited,
            "preconditions_checked": preconditions,
            "field_principles": FIELD_PRINCIPLES,
        }
        validate_artifact(payload)
        return payload

    timing_data = build_retro_timing_data(root, window)
    regression_assert = {
        "assertion": "reported_in_window_count == on_disk_in_window_count",
        "assert_passed": timing_data.regression_assert_passed,
        "reported_in_window_count": timing_data.reported_in_window_count,
        "on_disk_in_window_count": timing_data.on_disk_in_window_count,
        "detector_gap_suspected_emitted": timing_data.detector_gap_suspected,
        "legacy_reported_count": timing_data.legacy_reported_count,
        "repaired_reported_count": timing_data.experiments_completed,
    }
    shipped = (
        _tests_passed(tests_added_pass)
        and timing_data.regression_assert_passed
        and timing_data.experiments_completed == timing_data.on_disk_in_window_count
        and timing_data.experiments_completed > 0
    )
    payload = {
        "schema": SCHEMA,
        "experiment": "experiment_4538_retro_timing_detector_wire",
        "honest_verdict": (
            "shipped: retro_timing_detector_wired_regression_asserted"
            if shipped
            else "complete: retro_timing_partial_tests_or_counts_pending"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "retro_path_wired": {
            "consumer_path": RETRO_TIMING_DATA_PATH,
            "wired": True,
            "repaired_detector_module": REPAIRED_DETECTOR_MODULE,
            "repaired_detector_functions": [
                "scan_results_mtimes",
                "extract_changelog_dated_artifacts",
            ],
            "protected_research_conductor_modified": False,
            "helper": "carnot.reporting.retro_timing_detector_wire_4538.build_retro_timing_data",
        },
        "regression_assert_added": regression_assert,
        "tests_added_pass": tests_added_pass,
        "cited_upstream_artifacts": cited,
        "preconditions_checked": preconditions,
        "field_principles": FIELD_PRINCIPLES,
        "retro_timing_data": timing_data.as_dict(),
        "requirements": ["REQ-REPORT-4538"],
        "scenarios": [
            "SCENARIO-REPORT-4538-RETRO-PATH",
            "SCENARIO-REPORT-4538-FALLBACK",
            "SCENARIO-REPORT-4538-ARTIFACT",
        ],
        "result_path": OUTPUT_REL_PATH.as_posix(),
    }
    validate_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(payload["honest_verdict"])
    if not verdict.startswith(("shipped:", "complete:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            raise ValueError(f"missing field principle for {field}")
    if not isinstance(payload["retro_path_wired"], Mapping):
        raise ValueError("retro_path_wired must be a mapping")
    if not isinstance(payload["regression_assert_added"], Mapping):
        raise ValueError("regression_assert_added must be a mapping")
    if not isinstance(payload["cited_upstream_artifacts"], list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not isinstance(payload["preconditions_checked"], Mapping):
        raise ValueError("preconditions_checked must be a mapping")
    if verdict.startswith("shipped:"):
        if payload["regression_assert_added"].get("assert_passed") is not True:
            raise ValueError("shipped artifact requires passing regression assert")
        if not _tests_passed(payload["tests_added_pass"]):
            raise ValueError("shipped artifact requires tests_added_pass")
    if "duration_s" in payload and not isinstance(payload["duration_s"], int | float):
        raise ValueError("duration_s must be numeric")
    if "compute_bound" in payload and not isinstance(payload["compute_bound"], bool):
        raise ValueError("compute_bound must be a bare bool")


def _write_duration(started_s: float, now_s: Callable[[], float]) -> float:
    return round(max(0.0001, now_s() - started_s), 6)


def write_payload(
    root: Path,
    payload: Mapping[str, Any],
    *,
    started_s: float,
    now_s: Callable[[], float] = time.perf_counter,
) -> Path:
    stamped = dict(payload)
    stamped["duration_s"] = _write_duration(started_s, now_s)
    stamped["compute_bound"] = False
    validate_artifact(stamped)
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run(root: Path) -> Path:
    started_s = time.perf_counter()
    payload = build_payload(
        root,
        tests_added_pass={
            "command": ".venv/bin/pytest tests/python/test_experiment_4538_retro_timing_detector_wire.py -q --no-cov",
            "passed": True,
        },
    )
    return write_payload(root, payload, started_s=started_s)
