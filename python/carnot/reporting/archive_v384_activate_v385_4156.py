"""Archive .384, activate .385, and record the stale-Timer correction.

Spec refs: REQ-REPORT-4156, SCENARIO-REPORT-4156,
SCENARIO-REPORT-4156-BLOCKED-PRECONDITION.

This is a record-only transition. It preserves the corrected diagnosis that
the `.384` epoch-fix was a misdiagnosis: Exp 4146 never launched training
because a stale Lightning Timer value in the checkpoint tripped a pre-launch
guard. The bounded-pass chain is therefore retired, while the single contiguous
run is the working mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.384"
ACTIVATED_MILESTONE = "2026.06.385"
RANDOM_SEED = 4156
OUTPUT_REL_PATH = Path("results/experiment_4156_archive_v384_activate_v385.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
NANO_TRM_TRAIN_REL_PATH = Path("nano-trm/src/nn/train.py")
CAPSTONE_REL_PATH = Path("results/experiment_4155_capstone_v384.json")
CONTIGUOUS_CSV_GLOB = "results/trm_runs/contiguous_run_hydra/csv/version_*/metrics.csv"
CONTIGUOUS_PID_REL_PATH = Path("results/trm_runs/contiguous_run.pid")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v384_to_v385_4156.v1"
EXPERIMENT_ID = "exp4156"
TASK_ID = "exp4156-archive-v384-activate-v385"

BASELINE_VAL_DEFAULT = 0.278172343969
FAITHFUL_THRESHOLD = 0.85
CONTIGUOUS_WORKING_FLOOR = 0.42
TIMER_ELAPSED_DEFAULT = 3641.9931271109963
MAX_TIME_DEFAULT = 3600.0
CONFIG_MAX_EPOCHS_DEFAULT = 50000
EPOCH_DEFAULT = 6399
EXP4146_DURATION_DEFAULT = 0.121

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V384_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "4146",
        "deliverable": "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json",
    },
    {
        "experiment_id": "4149",
        "deliverable": "results/experiment_4149_sudoku_accumulate_pass4_convergence.json",
    },
    {
        "experiment_id": "4150",
        "deliverable": "results/experiment_4150_decisive_verifier_graft_sudoku.json",
    },
    {
        "experiment_id": "4155",
        "deliverable": str(CAPSTONE_REL_PATH),
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "active_milestone_confirmed",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "nano_trm_train_present",
    "pretest_suite_green",
    "v384_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.384.",
    "activated_milestone": "Confirms .385 is live for the contiguous-run continuation path.",
    "active_milestone_confirmed": "Records the active roadmap milestone actually observed.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "nano_trm_train_present": "Bare bool: nano-trm/src/nn/train.py exists before activation.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v384_close_state": (
        "Honest record (epoch-fix was a misdiagnosis / stale-Timer guard misfire; "
        "bounded-pass retired; contiguous run is the working fix; baseline 0.42+ climbing) "
        "so the next planner builds on the truth."
    ),
    "preconditions_checked": (
        "Records resources verified; pre-empts the silent-missing-resource fabrication mode."
    ),
    "honest_verdict": (
        "Self-declared terminal state. MUST start with complete:/success:/passed:/shipped:."
    ),
    "duration_s": "Positive bare wall-clock for this record-only aggregation.",
    "inference_substrate": "Declares aggregation only; no live training happens in this task.",
}

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.384['\"]?\s*$")


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required precondition command."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def yaml_parses(text: str) -> bool:
    """Return true when PyYAML can safe-load the supplied text."""

    try:
        yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return True


def duration_from(started_s: float | None, now_s: float | None) -> float:
    """Compute a positive duration for a record-only task."""

    if started_s is None:
        return 0.0001
    end_s = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0001, end_s - float(started_s)), 6)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 checksum over artifact content."""

    filtered = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: Any) -> bool:
    """Return true when value is a lowercase SHA-256 hex digest."""

    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON with a trailing newline."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _milestone_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("milestone:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "unknown"


def read_active_milestone(root: Path) -> tuple[str, str]:
    """Return the active milestone and the roadmap path used."""

    for rel_path in (Path("research-roadmap.yaml"), Path("research-roadmap-next.yaml")):
        path = root / rel_path
        if path.exists():
            milestone = _milestone_from_text(path.read_text(encoding="utf-8"))
            if milestone != "unknown":
                return milestone, str(rel_path)
    return "unknown", "research-roadmap.yaml"


def train_file_present(root: Path) -> bool:
    """Return true when the nano-TRM trainer entrypoint exists."""

    return (root / NANO_TRM_TRAIN_REL_PATH).exists()


def archive_record_count(text: str) -> int:
    """Count top-level `.384` archive records without counting nested task ids."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def canonical_finding(close_state: Mapping[str, Any]) -> str:
    """Build the `.384` archive finding from the corrected close-state."""

    max_val = _number(close_state.get("max_contiguous_val_exact_accuracy"), CONTIGUOUS_WORKING_FLOOR)
    return (
        ".384 close-state correction: the epoch-fix was a MISDIAGNOSIS. Exp4146 did not fail "
        "after training; it never started because diagnose_epoch_cap produced a stale-Timer "
        "guard misfire by reading Lightning Timer state from the checkpoint "
        "(timer_train_elapsed_s=3641.99 >= max_time 3600), with "
        "duration_s=0.121, post_epoch==seed_epoch==6399, and val=null. config_max_epochs was "
        "50000, so the max_epochs cap was never the cause. The bounded-pass chain is RETIRED. "
        "The single contiguous run is the working fix and advanced validation from 0.278 to "
        f"{max_val:.3f} (0.42+ climbing). The decisive graft remains deferred until val>=0.85."
    )


def build_canonical_record(close_state: Mapping[str, Any]) -> str:
    """Build a minimal `.384` record for the absent-history case."""

    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .384 and activate .385; preserve stale-Timer close-state')}",
        "  completed: '2026-06-13'",
        f"  finding: {_yaml_quote(canonical_finding(close_state))}",
        "  activation_recorded: exp4156-archive-v384-activate-v385",
        "  tasks:",
        "  - id: exp4146-sudoku-accumulate-pass1-epochfix",
        "    result: 'blocked before launch by stale Lightning Timer state'",
        "  - id: exp4149-sudoku-accumulate-pass4-convergence",
        "    result: 'bounded-pass chain retired'",
        "  - id: exp4155-capstone-v384",
        "    result: 'corrected by exp4156 archive; contiguous run is working fix'",
    ]
    return "\n".join(lines) + "\n"


def _insert_before_tasks(lines: list[str], new_line: str) -> list[str]:
    for index, line in enumerate(lines):
        if line == "  tasks:":
            return lines[:index] + [new_line] + lines[index:]
    return lines + [new_line]


def _canonicalize_target_span(lines: list[str], close_state: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    finding_written = False
    activation_written = False
    for line in lines:
        if line.startswith("  finding:"):
            if not finding_written:
                out.append(f"  finding: {_yaml_quote(canonical_finding(close_state))}")
                finding_written = True
            continue
        if line.startswith("  activation_recorded:"):
            if not activation_written:
                out.append("  activation_recorded: exp4156-archive-v384-activate-v385")
                activation_written = True
            continue
        out.append(line)
    if not finding_written:
        out = _insert_before_tasks(out, f"  finding: {_yaml_quote(canonical_finding(close_state))}")
    if not activation_written:
        out = _insert_before_tasks(out, "  activation_recorded: exp4156-archive-v384-activate-v385")
    return out


def dedupe_or_update_record(text: str, close_state: Mapping[str, Any]) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.384` record exists and carries the corrected truth."""

    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _record_id(line) is not None]
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        spans.append((start, end))
    target_spans = [(start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE]
    if not target_spans:
        return f"{text.rstrip()}\n{build_canonical_record(close_state)}", 0, "appended"

    first_start, first_end = target_spans[0]
    remove: set[int] = set()
    for start, end in target_spans[1:]:
        remove.update(range(start, end))
    replacement = _canonicalize_target_span(lines[first_start:first_end], close_state)
    rebuilt: list[str] = []
    for index, line in enumerate(lines):
        if first_start <= index < first_end:
            if index == first_start:
                rebuilt.extend(replacement)
            continue
        if index in remove:
            continue
        rebuilt.append(line)
    new_text = "\n".join(rebuilt)
    if len(target_spans) > 1:
        return new_text, len(target_spans) - 1, "deduped"
    if new_text != text:
        return new_text, 0, "updated"
    return text, 0, "unchanged"


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object from disk, returning empty dict on absence or bad shape."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def file_sha256(path: Path) -> str | None:
    """Return file SHA-256, or None when the file is absent."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_v384_sources(root: Path) -> dict[str, JsonDict]:
    """Read source artifacts that carry the `.384` close-state."""

    out: dict[str, JsonDict] = {}
    for source in V384_SOURCE_ARTIFACTS:
        out[str(source["experiment_id"])] = read_json_object(root / str(source["deliverable"]))
    return out


def _version_from_path(path: Path) -> int:
    text = path.parent.name.removeprefix("version_")
    return int(text) if text.isdigit() else -1


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int = 0) -> int:
    number = _float_or_none(value)
    return default if number is None else int(number)


def read_contiguous_metrics(root: Path) -> list[JsonDict]:
    """Read contiguous-run validation rows from all CSV logger versions."""

    rows: list[JsonDict] = []
    for path in sorted(root.glob(CONTIGUOUS_CSV_GLOB), key=lambda item: (_version_from_path(item), str(item))):
        rel_path = str(path.relative_to(root))
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row_index, row in enumerate(reader):
                    val = _float_or_none(row.get("val/exact_accuracy"))
                    if val is None:
                        continue
                    rows.append(
                        {
                            "csv_path": rel_path,
                            "version": _version_from_path(path),
                            "row_index": row_index,
                            "epoch": _int_or_default(row.get("epoch")),
                            "step": _int_or_default(row.get("step")),
                            "val_exact_accuracy": val,
                            "val_exact_accuracy_rounded": round(val, 4),
                        }
                    )
        except OSError:
            continue
    return sorted(rows, key=lambda row: (row["epoch"], row["step"], row["version"], row["row_index"]))


def read_contiguous_pid(root: Path) -> int | None:
    """Return the recorded contiguous-run PID when present."""

    try:
        digits = "".join(ch for ch in (root / CONTIGUOUS_PID_REL_PATH).read_text(encoding="utf-8") if ch.isdigit())
    except OSError:
        return None
    return int(digits) if digits else None


def pid_is_alive(pid: int | None) -> bool:
    """Return true when a recorded local PID still exists."""

    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for upstream `.384` artifacts and contiguous CSVs."""

    cited: list[JsonDict] = []
    for source in V384_SOURCE_ARTIFACTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "kind": "artifact",
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "sha256": file_sha256(root / rel),
            }
        )
    for path in sorted(root.glob(CONTIGUOUS_CSV_GLOB), key=lambda item: (_version_from_path(item), str(item))):
        rel = str(path.relative_to(root))
        cited.append({"kind": "contiguous_metric_csv", "deliverable": rel, "sha256": file_sha256(path)})
    return cited


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_v384_close_state(
    sources: Mapping[str, Mapping[str, Any]],
    contiguous_metrics: Sequence[Mapping[str, Any]],
    *,
    run_alive: bool,
) -> JsonDict:
    """Build the honest `.384` close-state from artifacts and contiguous metrics."""

    pass1 = _mapping(sources.get("4146", {}))
    pass4 = _mapping(sources.get("4149", {}))
    graft = _mapping(sources.get("4150", {}))
    capstone = _mapping(sources.get("4155", {}))
    diagnosis = _mapping(pass1.get("diagnosis"))
    capstone_trajectory = _mapping(capstone.get("baseline_val_trajectory"))

    timer_elapsed = round(_number(diagnosis.get("timer_train_elapsed_s"), 0.0), 2)
    max_time_s = _number(diagnosis.get("max_time_s"), MAX_TIME_DEFAULT)
    config_max_epochs = int(_number(diagnosis.get("config_max_epochs"), 0.0))
    seed_epoch = int(_number(pass1.get("seed_epoch"), _number(diagnosis.get("checkpoint_epoch"), 0.0)))
    post_epoch = int(_number(pass1.get("post_epoch"), seed_epoch))
    max_epochs_cap_confirmed = bool(
        pass1.get("max_epochs_cap_confirmed", diagnosis.get("max_epochs_cap_confirmed", False))
    )
    stale_timer_guard = timer_elapsed >= max_time_s and max_time_s > 0
    epoch_fix_misdiagnosis = (
        stale_timer_guard and not max_epochs_cap_confirmed and config_max_epochs > post_epoch
    )

    metric_rows = [dict(row) for row in contiguous_metrics]
    latest_metric = metric_rows[-1] if metric_rows else {}
    max_metric = max(metric_rows, key=lambda row: row["val_exact_accuracy"]) if metric_rows else {}
    latest_val = _number(latest_metric.get("val_exact_accuracy"), 0.0)
    max_val = _number(max_metric.get("val_exact_accuracy"), 0.0)
    baseline_seed = _number(
        pass4.get("val_exact_accuracy"),
        _number(capstone_trajectory.get("final_val_exact_accuracy"), BASELINE_VAL_DEFAULT),
    )
    baseline_faithful = max_val >= FAITHFUL_THRESHOLD

    return {
        "summary": "epochfix_misdiagnosis_stale_timer_guard_bounded_pass_retired_contiguous_run_working",
        "epoch_fix_was_misdiagnosis": epoch_fix_misdiagnosis,
        "stale_timer_guard_misfire": stale_timer_guard,
        "exp4146_honest_verdict": str(pass1.get("honest_verdict", "")),
        "exp4146_duration_s": round(_number(pass1.get("duration_s"), EXP4146_DURATION_DEFAULT), 3),
        "exp4146_val_exact_accuracy": pass1.get("val_exact_accuracy"),
        "timer_train_elapsed_s": timer_elapsed,
        "max_time_s": max_time_s,
        "config_max_epochs": config_max_epochs,
        "max_epochs_cap_confirmed": max_epochs_cap_confirmed,
        "seed_epoch": seed_epoch,
        "post_epoch": post_epoch,
        "checkpoint_epoch_unchanged": post_epoch == seed_epoch,
        "bounded_pass_chain_retired": True,
        "bounded_pass_chain_failure_mode": (
            "resume_60min_pass_writeback repeats stale Timer guard and never launches cleanly"
        ),
        "v384_bounded_pass_val_exact_accuracy": baseline_seed,
        "contiguous_run_is_working_fix": max_val >= CONTIGUOUS_WORKING_FLOOR,
        "contiguous_run_alive": run_alive,
        "contiguous_validation_points": len(metric_rows),
        "contiguous_latest_epoch": latest_metric.get("epoch"),
        "contiguous_latest_step": latest_metric.get("step"),
        "latest_contiguous_val_exact_accuracy": latest_val,
        "latest_contiguous_val_exact_accuracy_rounded": round(latest_val, 3),
        "max_contiguous_val_exact_accuracy": max_val,
        "max_contiguous_val_exact_accuracy_rounded": round(max_val, 3),
        "max_contiguous_val_epoch": max_metric.get("epoch"),
        "max_contiguous_val_step": max_metric.get("step"),
        "contiguous_baseline_above_042": max_val >= CONTIGUOUS_WORKING_FLOOR,
        "baseline_seed_val_exact_accuracy": baseline_seed,
        "contiguous_val_gain_vs_seed": round(max_val - baseline_seed, 6) if metric_rows else 0.0,
        "contiguous_val_trajectory": [
            {
                "source": "experiment_4149_sudoku_accumulate_pass4_convergence",
                "label": "v384_seed",
                "val_exact_accuracy": baseline_seed,
                "val_exact_accuracy_rounded": round(baseline_seed, 4),
            },
            *metric_rows,
        ],
        "baseline_faithful": baseline_faithful,
        "faithful_threshold": FAITHFUL_THRESHOLD,
        "decisive_graft_deferred": bool(graft.get("graft_deferred", True)) or not baseline_faithful,
        "decisive_graft_deferred_reason": "baseline_val_below_0.85"
        if not baseline_faithful
        else "graft_artifact_deferred",
        "verifier_value_added": bool(graft.get("verifier_value_added", False)),
        "v384_capstone": {
            "artifact_present": bool(capstone),
            "headline_outcome": str(capstone.get("headline_outcome", "")),
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "corrected_reason": "epochfix_misdiagnosis_stale_timer_guard_not_max_epochs_cap",
        },
    }


def _run_command(command: list[str], root: Path) -> CommandResult:
    try:
        completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))
    return CommandResult(
        command=command,
        exit_code=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def smart_subset_targets(root: Path) -> list[str]:
    """Return existing smart-subset targets, or the first core target as fallback."""

    targets = [target for target in CORE_SMART_SUBSET if (root / target).exists()]
    return targets or [CORE_SMART_SUBSET[0]]


def smart_subset_command(targets: Sequence[str]) -> list[str]:
    """Return the smart-subset pytest command."""

    return [str(PYTEST_BIN), *targets, "-q", "--no-header", "-n", "0", "--no-cov", "-o", "addopts="]


def run_smart_subset(root: Path) -> CommandResult:
    """Run the smart-subset pre-test gate once."""

    return _run_command(smart_subset_command(smart_subset_targets(root)), root)


def terminal_verdict(v384_close_state: Mapping[str, Any]) -> str:
    """Return the complete-path verdict for the archive transition."""

    val = _number(v384_close_state.get("max_contiguous_val_exact_accuracy_rounded"), CONTIGUOUS_WORKING_FLOOR)
    return (
        "success: archived_v384_v385_active_epochfix_misdiagnosis_stale_timer_"
        f"bounded_pass_retired_contiguous_val_{val:.3f}_graft_deferred_pretest_green"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    nano_trm_train_present: bool,
    pretest_suite_green: bool,
    v384_close_state: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    active_milestone_confirmed: str,
    active_roadmap_path: str,
    research_complete_record_action: str,
    research_complete_duplicates_removed: int,
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "nano_trm_train_present": nano_trm_train_present,
        "pretest_suite_green": pretest_suite_green,
        "v384_close_state": dict(v384_close_state),
        "preconditions_checked": dict(preconditions_checked),
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "research_complete_record_action": research_complete_record_action,
        "research_complete_duplicates_removed": research_complete_duplicates_removed,
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(reason: str, **kwargs: Any) -> JsonDict:
    """Build a blocked artifact without fabricating green resources."""

    defaults: JsonDict = {
        "research_complete_yaml_parses": False,
        "exclusion_manifest_parses": False,
        "nano_trm_train_present": False,
        "pretest_suite_green": False,
        "v384_close_state": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": "",
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
        "cited_upstream_artifacts": [],
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4156 complete artifact."""

    close_state = kwargs["v384_close_state"]
    payload = _base_payload(
        honest_verdict=terminal_verdict(close_state),
        research_complete_yaml_parses=True,
        exclusion_manifest_parses=True,
        nano_trm_train_present=True,
        pretest_suite_green=True,
        active_milestone_confirmed=ACTIVATED_MILESTONE,
        **kwargs,
    )
    validate_artifact(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate fields that stop this archive from laundering the `.384` truth."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    for field in ("honest_verdict", "v384_close_state", "preconditions_checked"):
        if principles.get(field) != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match REQ-REPORT-4156")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.384")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.385")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML must parse")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("exclusion manifest must parse")
    if artifact.get("nano_trm_train_present") is not True:
        raise ValueError("nano-trm train file must be present")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") != ACTIVATED_MILESTONE:
        raise ValueError("active milestone must be confirmed as 2026.06.385")
    close_state = artifact.get("v384_close_state")
    if not isinstance(close_state, Mapping):
        raise ValueError("v384_close_state must be a mapping")
    if close_state.get("epoch_fix_was_misdiagnosis") is not True:
        raise ValueError("epoch_fix_was_misdiagnosis must be True")
    if close_state.get("stale_timer_guard_misfire") is not True:
        raise ValueError("stale_timer_guard_misfire must be True")
    if close_state.get("bounded_pass_chain_retired") is not True:
        raise ValueError("bounded_pass_chain_retired must be True")
    if close_state.get("contiguous_run_is_working_fix") is not True:
        raise ValueError("contiguous_run_is_working_fix must be True")
    if _number(close_state.get("max_contiguous_val_exact_accuracy"), 0.0) < CONTIGUOUS_WORKING_FLOOR:
        raise ValueError("contiguous baseline 0.42 floor must be recorded")
    if close_state.get("decisive_graft_deferred") is not True:
        raise ValueError("decisive_graft_deferred must be True")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        raise ValueError("duration_s must be positive")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not is_sha256(artifact.get("reproducibility_checksum")):
        raise ValueError("reproducibility_checksum must be sha256")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    root: Path | str = REPO_ROOT,
    *,
    pretest_result: CommandResult | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Run the `.384` archive and `.385` activation guard."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    manifest_path = root_path / EXCLUSION_MANIFEST_REL_PATH
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    complete_exists = complete_path.exists()
    complete_text = complete_path.read_text(encoding="utf-8") if complete_exists else ""
    complete_parses = complete_exists and yaml_parses(complete_text)
    manifest_exists = manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8") if manifest_exists else ""
    manifest_parses = manifest_exists and yaml_parses(manifest_text)
    has_train = train_file_present(root_path)

    preconditions: JsonDict = {
        "research_complete_yaml_exists": complete_exists,
        "research_complete_yaml_parses": complete_parses,
        "exclusion_manifest_exists": manifest_exists,
        "exclusion_manifest_parses": manifest_parses,
        "nano_trm_train_present": has_train,
        "nano_trm_train_path": str(NANO_TRM_TRAIN_REL_PATH),
        "active_milestone": active_milestone,
        "active_roadmap_path": active_roadmap_path,
        "pretest_suite_green": False,
        "v384_capstone_present": False,
        "contiguous_metrics_present": False,
    }

    def blocked(reason: str, **extra: Any) -> Path:
        write_payload(
            output_path,
            build_blocked_artifact(
                reason,
                preconditions_checked=preconditions,
                duration_s=duration_from(start, now_s),
                active_milestone_confirmed=active_milestone,
                active_roadmap_path=active_roadmap_path,
                **extra,
            ),
        )
        return output_path

    if not complete_exists:
        return blocked("blocked_research_complete_yaml_missing")
    if not complete_parses:
        return blocked("blocked_research_complete_yaml_poison")
    if not manifest_exists:
        return blocked("blocked_exclusion_manifest_missing", research_complete_yaml_parses=True)
    if not manifest_parses:
        return blocked("blocked_exclusion_manifest_yaml_poison", research_complete_yaml_parses=True)
    if not has_train:
        return blocked(
            "blocked_nano_trm_train_missing",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
        )
    if active_milestone != ACTIVATED_MILESTONE:
        return blocked(
            "blocked_v385_not_active",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
        )

    pretest = pretest_result if pretest_result is not None else run_smart_subset(root_path)
    pretest_green = pretest.exit_code == 0
    preconditions["pretest_suite_green"] = pretest_green
    preconditions["pretest_command"] = pretest.command
    preconditions["pretest_exit_code"] = pretest.exit_code
    if not pretest_green:
        return blocked(
            "blocked_smart_subset_pretest_not_green",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
        )

    capstone_present = (root_path / CAPSTONE_REL_PATH).exists()
    preconditions["v384_capstone_present"] = capstone_present
    preconditions["v384_capstone_path"] = str(CAPSTONE_REL_PATH)
    if not capstone_present:
        return blocked(
            "blocked_v384_capstone_missing",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
        )

    metrics = read_contiguous_metrics(root_path)
    preconditions["contiguous_metrics_present"] = bool(metrics)
    preconditions["contiguous_metrics_glob"] = CONTIGUOUS_CSV_GLOB
    if not metrics:
        return blocked(
            "blocked_contiguous_metrics_missing",
            research_complete_yaml_parses=True,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
        )

    pid = read_contiguous_pid(root_path)
    run_alive = pid_is_alive(pid)
    preconditions["contiguous_pid"] = pid
    preconditions["contiguous_run_alive"] = run_alive
    sources = read_v384_sources(root_path)
    close_state = build_v384_close_state(sources, metrics, run_alive=run_alive)

    new_text, removed, action = dedupe_or_update_record(complete_text, close_state)
    if not yaml_parses(new_text):
        return blocked(
            "blocked_research_complete_edit_invalid",
            research_complete_yaml_parses=False,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
        )
    if new_text != complete_text:
        complete_path.write_text(new_text, encoding="utf-8")
    after_parses = yaml_parses(complete_path.read_text(encoding="utf-8"))
    preconditions["research_complete_record_action"] = action
    preconditions["research_complete_duplicates_removed"] = removed
    preconditions["research_complete_yaml_parses_after_edit"] = after_parses
    if not after_parses:
        return blocked(
            "blocked_research_complete_yaml_poison_after_edit",
            research_complete_yaml_parses=False,
            exclusion_manifest_parses=True,
            nano_trm_train_present=has_train,
            pretest_suite_green=True,
            research_complete_record_action=action,
            research_complete_duplicates_removed=removed,
        )

    payload = build_complete_artifact(
        v384_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=removed,
        cited_upstream_artifacts=build_cited_upstream(root_path),
    )
    write_payload(output_path, payload)
    return output_path


def main() -> int:
    """CLI entrypoint for the conductor-requested script."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0
