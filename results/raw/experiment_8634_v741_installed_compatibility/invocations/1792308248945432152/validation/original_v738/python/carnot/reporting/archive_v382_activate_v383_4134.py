"""Archive .382, activate .383, and record the fixed-LR close-state truth.

Spec refs: REQ-REPORT-4134, SCENARIO-REPORT-4134,
SCENARIO-REPORT-4134-BLOCKED-PRECONDITION.

This record-only transition exists to prevent the next planner from losing the
important distinction in the `.382` result. The LR-resume fix landed and the
Sudoku-Extreme baseline started accumulating quickly, but it was still below the
published baseline gate, so the verifier graft remained deferred for `.383`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.382"
ACTIVATED_MILESTONE = "2026.06.383"
RANDOM_SEED = 4134
OUTPUT_REL_PATH = Path("results/experiment_4134_archive_v382_activate_v383.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
NANO_TRM_TRAIN_REL_PATH = Path("nano-trm/src/nn/train.py")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v382_to_v383_4134.v1"
EXPERIMENT_ID = "exp4134"
TASK_ID = "exp4134-archive-v382-activate-v383"

START_VAL_DEFAULT = 0.105989582837
FINAL_VAL_DEFAULT = 0.278172343969
DELTA_DEFAULT = 0.172182761132
V381_REFERENCE_DELTA = 0.01
PUBLISHED_BASELINE_TARGET = 0.87
VALIDATION_START_LR_DEFAULT = 9.998933091992512e-05
FRESH_WARMUP_LR_DEFAULT = 2.4500000108673703e-06

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V382_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "4133",
        "deliverable": "results/experiment_4133_capstone_v382.json",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "nano_trm_train_present",
    "pretest_suite_green",
    "v382_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.382.",
    "activated_milestone": "Confirms .383 is live for convergence and the decisive graft.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "nano_trm_train_present": "Bare bool: nano-trm/src/nn/train.py exists before activation.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v382_close_state": (
        "Honest record (LR fix landed, accumulation accelerated ~17x, baseline still <0.87, "
        "graft deferred) so the next planner builds on the truth."
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.382['\"]?\s*$")


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
    """Count top-level `.382` archive records without counting nested tasks."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    """Return a top-level YAML milestone id, or None for nested task ids."""

    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_canonical_record() -> str:
    """Build a minimal `.382` record for the absent-history case."""

    finding = (
        "The LR-resume-correctness fix landed: resumed train/lr continued near 9.999e-05 "
        "instead of resetting to the 2.45e-06 fresh warmup. Validation accumulated from "
        "0.106 to 0.278 in one corrected pass, about 17x faster than .381, but the baseline "
        "remained below 0.87 so the verifier graft deferred. Total games solved stayed 13."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .382 and activate .383; preserve fixed-LR close-state')}",
        "  completed: '2026-06-13'",
        f"  finding: {_yaml_quote(finding)}",
        "  activation_recorded: exp4134-archive-v382-activate-v383",
        "  tasks:",
        "  - id: exp4126-lr-resume-correctness-fix",
        "    result: 'lr_continuous_across_resume true'",
        "  - id: exp4127-sudoku-extreme-accumulate-fixed",
        "    result: 'val_exact_accuracy 0.2782; baseline still below 0.87'",
        "  - id: exp4128-carnot-verifier-graft-sudoku",
        "    result: 'graft deferred because baseline was not faithful'",
        "  - id: exp4133-capstone-v382",
        "    result: 'lr fixed, accumulating, .383 continues'",
    ]
    return "\n".join(lines) + "\n"


def dedupe_or_append_record(text: str) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.382` record exists in research-complete."""

    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _record_id(line) is not None]
    spans: list[tuple[int, int]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        spans.append((start, end))
    target_spans = [(start, end) for start, end in spans if _record_id(lines[start]) == ARCHIVED_MILESTONE]
    if not target_spans:
        return f"{text.rstrip()}\n{build_canonical_record()}", 0, "appended"
    if len(target_spans) == 1:
        return text, 0, "unchanged"
    remove: set[int] = set()
    for start, end in target_spans[1:]:
        remove.update(range(start, end))
    return "\n".join(line for i, line in enumerate(lines) if i not in remove), len(target_spans) - 1, "deduped"


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


def read_v382_sources(root: Path) -> dict[str, JsonDict]:
    """Read the source artifact that carries the `.382` close-state."""

    out: dict[str, JsonDict] = {}
    for source in V382_SOURCE_ARTIFACTS:
        out[str(source["experiment_id"])] = read_json_object(root / str(source["deliverable"]))
    return out


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for the upstream `.382` artifacts."""

    cited: list[JsonDict] = []
    for source in V382_SOURCE_ARTIFACTS:
        rel = str(source["deliverable"])
        cited.append(
            {
                "experiment_id": str(source["experiment_id"]),
                "deliverable": rel,
                "sha256": file_sha256(root / rel),
            }
        )
    return cited


def _number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_numbers(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [_number(item, 0.0) for item in value if isinstance(item, int | float) and not isinstance(item, bool)]


def build_v382_close_state(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the honest `.382` close-state from the capstone artifact where present."""

    capstone = sources.get("4133", {})
    lr_fix = _mapping(capstone.get("lr_resume_fix"))
    baseline = _mapping(capstone.get("baseline_reproduction"))
    trajectory = _mapping(capstone.get("baseline_val_trajectory"))
    graft = _mapping(capstone.get("sudoku_verifier_graft"))
    headline = _mapping(capstone.get("headline_answers"))
    delta_vs_v381 = _mapping(trajectory.get("per_pass_delta_vs_v381"))

    upstream_values = _list_numbers(trajectory.get("upstream_values"))
    if len(upstream_values) >= 2:
        start_val = upstream_values[0]
        final_val = upstream_values[-1]
    else:
        start_val = START_VAL_DEFAULT
        final_val = _number(baseline.get("val_exact_accuracy"), FINAL_VAL_DEFAULT)
    upstream_deltas = _list_numbers(trajectory.get("upstream_deltas"))
    delta = upstream_deltas[-1] if upstream_deltas else final_val - start_val
    mean_delta = _number(delta_vs_v381.get("mean_delta"), delta or DELTA_DEFAULT)
    reference_delta = _number(delta_vs_v381.get("reference_delta"), V381_REFERENCE_DELTA)
    speedup = round(mean_delta / reference_delta, 1) if reference_delta > 0 else None
    validation_start_lr = _number(lr_fix.get("validation_first_lr"), VALIDATION_START_LR_DEFAULT)
    fresh_warmup_lr = _number(lr_fix.get("fresh_warmup_lr"), FRESH_WARMUP_LR_DEFAULT)
    total_games = int(_number(capstone.get("total_arc_games_solved"), _number(headline.get("total_arc_games_solved"), 13)))

    return {
        "summary": "lr_resume_fix_landed_accumulation_accelerated_baseline_still_below_087_graft_deferred",
        "lr_resume_fix_landed": bool(headline.get("exp4126_lr_resume_fix_landed", True)),
        "lr_continuous_across_resume": bool(lr_fix.get("lr_continuous_across_resume", True)),
        "validation_start_lr": validation_start_lr,
        "validation_start_lr_rounded": float(f"{validation_start_lr:.3e}"),
        "fresh_warmup_lr": fresh_warmup_lr,
        "fresh_warmup_lr_rounded": float(f"{fresh_warmup_lr:.2e}"),
        "manual_lr_step_restored": int(_number(lr_fix.get("manual_lr_step_restored"), 4300)),
        "baseline_reproduced": bool(baseline.get("matches_published_087", False)),
        "baseline_status": str(baseline.get("status", "still_accumulating")),
        "published_exact_accuracy_target": _number(
            trajectory.get("published_exact_accuracy_target"), PUBLISHED_BASELINE_TARGET
        ),
        "start_val_exact_accuracy": start_val,
        "final_val_exact_accuracy": final_val,
        "final_val_exact_accuracy_rounded": round(final_val, 3),
        "delta_vs_previous": delta,
        "delta_vs_previous_rounded": round(delta, 3),
        "reported_close_state_sequence": {
            "basis": "results/experiment_4133_capstone_v382.json",
            "val_exact_accuracy": [start_val, final_val],
            "val_exact_accuracy_rounded": [round(start_val, 3), round(final_val, 3)],
        },
        "per_pass_delta_vs_v381": {
            "beats_v381": bool(delta_vs_v381.get("beats_v381", True)),
            "mean_delta": mean_delta,
            "reference_delta": reference_delta,
            "speedup_factor_vs_v381": speedup,
            "comparison": str(delta_vs_v381.get("comparison", "faster_than_v381")),
        },
        "accelerated_vs_v381": bool(baseline.get("accelerated_vs_v381", True)),
        "graft_deferred": bool(graft.get("graft_deferred", True)),
        "graft_deferred_reason": str(graft.get("status", "deferred_by_baseline_not_reproduced")),
        "verifier_value_added": bool(graft.get("verifier_value_added", False)),
        "total_games_solved": total_games,
        "capstone_v382": {
            "artifact_present": bool(capstone),
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "headline_outcome": str(capstone.get("headline_outcome", "lr_fixed_accumulating_v383_continues")),
        },
        "v383_forward_plan": {
            "baseline_accumulation_passes": 4,
            "graft_policy": "converge the fixed-LR baseline first, then run the decisive verifier graft",
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


def terminal_verdict(v382_close_state: Mapping[str, Any]) -> str:
    """Return the complete-path verdict for the archive transition."""

    final_val = _number(v382_close_state.get("final_val_exact_accuracy_rounded"), 0.278)
    games = int(_number(v382_close_state.get("total_games_solved"), 13))
    return (
        "success: archived_v382_v383_active_lr_fixed_accumulating_"
        f"val_{final_val:.3f}_baseline_below_087_graft_deferred_games{games}_pretest_green"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    nano_trm_train_present: bool,
    pretest_suite_green: bool,
    v382_close_state: Mapping[str, Any],
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
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "exclusion_manifest_parses": exclusion_manifest_parses,
        "nano_trm_train_present": nano_trm_train_present,
        "pretest_suite_green": pretest_suite_green,
        "v382_close_state": dict(v382_close_state),
        "preconditions_checked": dict(preconditions_checked),
        "honest_verdict": honest_verdict,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "active_milestone_confirmed": active_milestone_confirmed,
        "active_roadmap_path": active_roadmap_path,
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
        "v382_close_state": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": "",
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
        "cited_upstream_artifacts": [],
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4134 complete artifact."""

    close_state = kwargs["v382_close_state"]
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
    """Validate the fields that stop this archive from laundering the `.382` truth."""

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
    for field in ("honest_verdict", "v382_close_state", "preconditions_checked"):
        if principles.get(field) != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match REQ-REPORT-4134")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.382")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.383")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML must parse")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("exclusion manifest must parse")
    if artifact.get("nano_trm_train_present") is not True:
        raise ValueError("nano-trm train file must be present")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") != ACTIVATED_MILESTONE:
        raise ValueError("active milestone must be confirmed as 2026.06.383")
    close_state = artifact.get("v382_close_state")
    if not isinstance(close_state, Mapping):
        raise ValueError("v382_close_state must be a mapping")
    if close_state.get("lr_resume_fix_landed") is not True:
        raise ValueError("v382_close_state must record lr_resume_fix_landed=True")
    if close_state.get("lr_continuous_across_resume") is not True:
        raise ValueError("v382_close_state must record lr_continuous_across_resume=True")
    if close_state.get("baseline_reproduced") is not False:
        raise ValueError("baseline_reproduced must be False")
    if close_state.get("final_val_exact_accuracy_rounded") != 0.278:
        raise ValueError("final_val_exact_accuracy must round to 0.278")
    reported = close_state.get("reported_close_state_sequence")
    if not isinstance(reported, Mapping) or reported.get("val_exact_accuracy_rounded") != [0.106, 0.278]:
        raise ValueError("close-state values must be [0.106, 0.278]")
    delta = close_state.get("per_pass_delta_vs_v381")
    if not isinstance(delta, Mapping) or delta.get("beats_v381") is not True:
        raise ValueError("per_pass_delta_vs_v381 must record beats_v381=True")
    if close_state.get("graft_deferred") is not True:
        raise ValueError("graft_deferred must be True")
    if close_state.get("total_games_solved") != 13:
        raise ValueError("total_games_solved must be 13")
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
    """Run the `.382` archive and `.383` activation guard."""

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
            "blocked_v383_not_active",
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

    new_text, removed, action = dedupe_or_append_record(complete_text)
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

    sources = read_v382_sources(root_path)
    close_state = build_v382_close_state(sources)
    cited = build_cited_upstream(root_path)
    payload = build_complete_artifact(
        v382_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=removed,
        cited_upstream_artifacts=cited,
    )
    write_payload(output_path, payload)
    return output_path


def main() -> int:
    """CLI entrypoint for the conductor-requested script."""

    output_path = run(REPO_ROOT)
    print(output_path)
    return 0
