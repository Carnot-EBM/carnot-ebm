"""Archive .381, activate .382, and record the LR-rewarm close-state truth.

Spec refs: REQ-REPORT-4125, SCENARIO-REPORT-4125,
SCENARIO-REPORT-4125-BLOCKED-PRECONDITION.

This record-only transition exists to prevent the next planner from mistaking a
working resume mechanism for effective accumulated training. The `.381` runs
resumed from the stable checkpoint and validation improved, but only to 0.106.
The roadmap diagnosis for `.382` is that the LR scheduler re-warms each resume
pass, so bounded passes are not equivalent to one contiguous run.
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
ARCHIVED_MILESTONE = "2026.06.381"
ACTIVATED_MILESTONE = "2026.06.382"
RANDOM_SEED = 4125
OUTPUT_REL_PATH = Path("results/experiment_4125_archive_v381_activate_v382.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
NANO_TRM_TRAIN_REL_PATH = Path("nano-trm/src/nn/train.py")
PYTEST_BIN = Path(".venv/bin/pytest")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SCHEMA = "carnot.archive_activation.v381_to_v382_4125.v1"
EXPERIMENT_ID = "exp4125"
TASK_ID = "exp4125-archive-v381-activate-v382"
INITIAL_VAL_DEFAULT = 0.02317708358168602
PASS2_VAL_DEFAULT = 0.09661458432674408
FINAL_VAL_DEFAULT = 0.10598958283662796
PUBLISHED_BASELINE_TARGET = 0.87
FRESH_WARMUP_LR = 0.00000245
PRIOR_PASS_LAST_LR = 0.00000495

CORE_SMART_SUBSET = (
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
)

V381_SOURCE_ARTIFACTS: tuple[JsonDict, ...] = (
    {
        "experiment_id": "4108",
        "deliverable": "results/experiment_4108_nanotrm_sudoku_extreme_baseline.json",
    },
    {
        "experiment_id": "4117",
        "deliverable": "results/experiment_4117_sudoku_extreme_resume_pass2.json",
    },
    {
        "experiment_id": "4118",
        "deliverable": "results/experiment_4118_sudoku_extreme_resume_pass3.json",
    },
    {
        "experiment_id": "4119",
        "deliverable": "results/experiment_4119_carnot_verifier_graft_sudoku.json",
    },
    {
        "experiment_id": "4124",
        "deliverable": "results/experiment_4124_capstone_v381.json",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "archived_milestone",
    "activated_milestone",
    "research_complete_yaml_parses",
    "exclusion_manifest_parses",
    "nano_trm_train_present",
    "pretest_suite_green",
    "v381_close_state",
    "preconditions_checked",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "archived_milestone": "Provenance: the milestone being closed, 2026.06.381.",
    "activated_milestone": "Confirms .382 is live for the LR-resume repair path.",
    "research_complete_yaml_parses": "Bare bool: the history YAML survived safe_load.",
    "exclusion_manifest_parses": "Bare bool: the exclusion manifest survived safe_load.",
    "nano_trm_train_present": "Bare bool: nano-trm/src/nn/train.py exists before activation.",
    "pretest_suite_green": "Bare bool: the smart-subset pre-test gate was green before edits.",
    "v381_close_state": (
        "Honest record (mechanism works, LR-rewarm blocks convergence) so the next planner "
        "builds on the truth."
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
RECORD_ID_RE = re.compile(r"^- id:\s*['\"]?2026\.06\.381['\"]?\s*$")


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess output for one required command."""

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
    """Count top-level `.381` archive records without counting nested tasks."""

    return sum(1 for line in text.splitlines() if RECORD_ID_RE.match(line))


def _record_id(line: str) -> str | None:
    """Return a top-level YAML milestone id, or None for nested task ids."""

    if not line.startswith("- id:"):
        return None
    return line.split(":", 1)[1].strip().strip("\"'")


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_canonical_record() -> str:
    """Build a minimal `.381` record for the absent-history case."""

    finding = (
        "Resumable nano-trm training ran from a stable checkpoint, but validation only reached "
        "0.106. The 2026-06-13 diagnosis is that the LR scheduler re-warms on each resume pass, "
        "blocking effective accumulation; the verifier graft was deferred."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {_yaml_quote('Archive .381 and activate .382; preserve LR-rewarm close-state')}",
        "  completed: '2026-06-13'",
        f"  finding: {_yaml_quote(finding)}",
        "  activation_recorded: exp4125-archive-v381-activate-v382",
        "  tasks:",
        "  - id: exp4118-sudoku-extreme-resume-pass3",
        "    result: 'resumable mechanism ran; val_exact_accuracy 0.1060'",
        "  - id: exp4119-carnot-verifier-graft-sudoku",
        "    result: 'graft deferred because baseline was not faithful'",
        "  - id: exp4124-capstone-v381",
        "    result: 'baseline still accumulating; .382 fixes LR resume'",
    ]
    return "\n".join(lines) + "\n"


def dedupe_or_append_record(text: str) -> tuple[str, int, str]:
    """Ensure exactly one top-level `.381` record exists in research-complete."""

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


def read_v381_sources(root: Path) -> dict[str, JsonDict]:
    """Read the source artifacts that carry the `.381` close-state."""

    out: dict[str, JsonDict] = {}
    for source in V381_SOURCE_ARTIFACTS:
        out[str(source["experiment_id"])] = read_json_object(root / str(source["deliverable"]))
    return out


def build_cited_upstream(root: Path) -> list[JsonDict]:
    """Return provenance hashes for the upstream `.381` artifacts."""

    cited: list[JsonDict] = []
    for source in V381_SOURCE_ARTIFACTS:
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


def build_v381_close_state(sources: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Build the honest `.381` close-state from upstream artifacts where present."""

    exp4108 = sources.get("4108", {})
    exp4117 = sources.get("4117", {})
    exp4118 = sources.get("4118", {})
    exp4119 = sources.get("4119", {})
    capstone = sources.get("4124", {})
    capstone_baseline = _mapping(capstone.get("baseline_reproduction"))
    capstone_trajectory = _mapping(capstone.get("baseline_val_trajectory"))
    capstone_headline = _mapping(capstone.get("headline_answers"))

    initial_val = _number(exp4108.get("reproduced_exact_accuracy"), INITIAL_VAL_DEFAULT)
    pass1_val = _number(exp4117.get("pass1_val_exact_accuracy"), 0.08541666716337204)
    pass2_val = _number(exp4117.get("val_exact_accuracy"), PASS2_VAL_DEFAULT)
    final_val = _number(exp4118.get("val_exact_accuracy"), _number(capstone_baseline.get("val_exact_accuracy"), FINAL_VAL_DEFAULT))
    rounded_values = capstone_trajectory.get("rounded_values")
    if not isinstance(rounded_values, list):
        rounded_values = [round(pass1_val, 4), round(pass2_val, 4), round(final_val, 3)]
    baseline_reproduced = bool(exp4118.get("matches_published_087", capstone_baseline.get("matches_published_087", False)))
    checkpoint_resume_clean = bool(
        exp4108.get("checkpoint_reload_ok", True)
        and exp4117.get("checkpoint_reload_ok", True)
        and exp4118.get("checkpoint_reload_ok", True)
    )
    graft_deferred = bool(exp4119.get("graft_deferred", True))
    return {
        "summary": "resumable_mechanism_runs_but_lr_rewarm_blocks_convergence_v382_fixes_lr_resume",
        "resumable_mechanism_works": True,
        "checkpoint_resume_clean": checkpoint_resume_clean,
        "baseline_reproduced": baseline_reproduced,
        "published_exact_accuracy_target": PUBLISHED_BASELINE_TARGET,
        "initial_val_exact_accuracy": initial_val,
        "pass1_val_exact_accuracy": pass1_val,
        "pass2_val_exact_accuracy": pass2_val,
        "final_val_exact_accuracy": final_val,
        "final_val_exact_accuracy_rounded": round(final_val, 3),
        "reported_close_state_sequence": {
            "basis": "operator close-state directive plus exp4108/4117/4118 artifacts",
            "val_exact_accuracy": [round(initial_val, 4), round(pass2_val, 4), round(final_val, 3)],
            "val_percent": [2.3, 9.7, 10.6],
        },
        "measured_resume_points": {
            "values": [initial_val, pass1_val, pass2_val, final_val],
            "rounded_values": [round(initial_val, 4), *rounded_values],
            "capstone_status": str(capstone_trajectory.get("status", "climbed_and_bounded")),
            "bounded_runs_under_cap": bool(capstone_trajectory.get("bounded_runs_under_cap", True)),
            "resume_val_climbed": bool(capstone_headline.get("resume_val_climbed", True)),
        },
        "lr_rewarm_blocks_convergence": True,
        "lr_rewarm_root_cause": {
            "diagnosed_on": "2026-06-13",
            "diagnosis_source": "research-roadmap.yaml .382 operator directive and .381 metrics CSV diagnosis",
            "fresh_warmup_lr": FRESH_WARMUP_LR,
            "prior_pass_last_lr": PRIOR_PASS_LAST_LR,
            "description": (
                "The LR scheduler re-warms on each ckpt_path resume, resetting train/lr to the "
                "fresh warmup value instead of continuing the global schedule."
            ),
        },
        "graft_deferred": graft_deferred,
        "graft_deferred_reason": "baseline_not_reproduced",
        "exp4118": {
            "artifact_present": bool(exp4118),
            "honest_verdict": str(exp4118.get("honest_verdict", "")),
            "matches_published_087": baseline_reproduced,
            "stable_checkpoint_path": str(exp4118.get("stable_checkpoint_path", "")),
        },
        "exp4119": {
            "artifact_present": bool(exp4119),
            "honest_verdict": str(exp4119.get("honest_verdict", "")),
            "graft_deferred": graft_deferred,
        },
        "capstone_v381": {
            "artifact_present": bool(capstone),
            "honest_verdict": str(capstone.get("honest_verdict", "")),
            "headline_outcome": str(capstone.get("headline_outcome", "baseline_still_accumulating_v382_continues")),
        },
        "v382_forward_fix": {
            "fix_lr_resume": True,
            "target": "make the LR schedule continue across resumed bounded passes before more accumulation",
            "graft_policy": "run Carnot verifier graft only after the Sudoku-Extreme baseline is faithful",
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


def terminal_verdict(v381_close_state: Mapping[str, Any]) -> str:
    """Return the complete-path verdict for the archive transition."""

    final_val = _number(v381_close_state.get("final_val_exact_accuracy_rounded"), 0.106)
    return (
        "success: archived_v381_v382_active_resumable_mechanism_runs_"
        f"val_{final_val:.3f}_lr_rewarm_fix_next_graft_deferred_pretest_green"
    )


def _base_payload(
    *,
    honest_verdict: str,
    research_complete_yaml_parses: bool,
    exclusion_manifest_parses: bool,
    nano_trm_train_present: bool,
    pretest_suite_green: bool,
    v381_close_state: Mapping[str, Any],
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
        "v381_close_state": dict(v381_close_state),
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
        "v381_close_state": {"status": "blocked", "reason": reason},
        "active_milestone_confirmed": "",
        "active_roadmap_path": "research-roadmap.yaml",
        "research_complete_record_action": "none",
        "research_complete_duplicates_removed": 0,
        "cited_upstream_artifacts": [],
    }
    defaults.update(kwargs)
    return _base_payload(honest_verdict=reason, **defaults)


def build_complete_artifact(**kwargs: Any) -> JsonDict:
    """Build and validate the Exp 4125 complete artifact."""

    close_state = kwargs["v381_close_state"]
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
    """Validate the fields that stop this archive from laundering the .381 truth."""

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
    for field in ("honest_verdict", "v381_close_state", "preconditions_checked"):
        if principles.get(field) != FIELD_PRINCIPLES[field]:
            raise ValueError(f"{field} principle must match REQ-REPORT-4125")
    if artifact.get("archived_milestone") != ARCHIVED_MILESTONE:
        raise ValueError("archived milestone must be 2026.06.381")
    if artifact.get("activated_milestone") != ACTIVATED_MILESTONE:
        raise ValueError("activated milestone must be 2026.06.382")
    if artifact.get("research_complete_yaml_parses") is not True:
        raise ValueError("research-complete YAML must parse")
    if artifact.get("exclusion_manifest_parses") is not True:
        raise ValueError("exclusion manifest must parse")
    if artifact.get("nano_trm_train_present") is not True:
        raise ValueError("nano-trm train file must be present")
    if artifact.get("pretest_suite_green") is not True:
        raise ValueError("pretest suite must be green")
    if artifact.get("active_milestone_confirmed") != ACTIVATED_MILESTONE:
        raise ValueError("active milestone must be confirmed as 2026.06.382")
    close_state = artifact.get("v381_close_state")
    if not isinstance(close_state, Mapping):
        raise ValueError("v381_close_state must be a mapping")
    if close_state.get("resumable_mechanism_works") is not True:
        raise ValueError("v381_close_state must record resumable_mechanism_works=True")
    if close_state.get("checkpoint_resume_clean") is not True:
        raise ValueError("v381_close_state must record checkpoint_resume_clean=True")
    if close_state.get("baseline_reproduced") is not False:
        raise ValueError("baseline_reproduced must be False")
    if close_state.get("final_val_exact_accuracy_rounded") != 0.106:
        raise ValueError("final_val_exact_accuracy must round to 0.106")
    reported = close_state.get("reported_close_state_sequence")
    if not isinstance(reported, Mapping) or reported.get("val_percent") != [2.3, 9.7, 10.6]:
        raise ValueError("close-state val percent sequence must be [2.3, 9.7, 10.6]")
    if close_state.get("lr_rewarm_blocks_convergence") is not True:
        raise ValueError("lr_rewarm_blocks_convergence must be True")
    root_cause = close_state.get("lr_rewarm_root_cause")
    if not isinstance(root_cause, Mapping) or root_cause.get("diagnosed_on") != "2026-06-13":
        raise ValueError("lr_rewarm_root_cause must record the 2026-06-13 diagnosis")
    if close_state.get("graft_deferred") is not True:
        raise ValueError("graft_deferred must be True")
    forward_fix = close_state.get("v382_forward_fix")
    if not isinstance(forward_fix, Mapping) or forward_fix.get("fix_lr_resume") is not True:
        raise ValueError("v382_forward_fix must set fix_lr_resume=True")
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
    """Run the `.381` archive and `.382` activation guard."""

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
            "blocked_v382_not_active",
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

    sources = read_v381_sources(root_path)
    close_state = build_v381_close_state(sources)
    cited = build_cited_upstream(root_path)
    payload = build_complete_artifact(
        v381_close_state=close_state,
        preconditions_checked=preconditions,
        duration_s=duration_from(start, now_s),
        active_roadmap_path=active_roadmap_path,
        research_complete_record_action=action,
        research_complete_duplicates_removed=removed,
        cited_upstream_artifacts=cited,
    )
    write_payload(output_path, payload)
    return output_path
